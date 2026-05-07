# V4 Imatrix Fix + IQ1 Quant Build — Design

**Date:** 2026-05-07
**Status:** Design approved, ready for implementation plan
**Branches involved:** `feat/v4-port-I-imatrix`, `feat/v4-port-J-iq1-quants`, both off `feat/v4-port`

## Goal

Unblock V4 imatrix calibration and produce the smallest practical V4-Flash GGUF artifacts (IQ1_S + IQ1_M, ~50–58 GiB).

## Scope

### In scope

- Diagnose and fix the V4 imatrix segfault in `tools/imatrix/imatrix.cpp`
- Produce `imatrix-v4-flash.dat` from wikitext-103 calibration data
- Build IQ1_S and IQ1_M GGUFs from the existing Q8_0 source + the new imatrix data
- Add `tests/v4-port/gate-imatrix.sh` regression gate so future llama.cpp upstream changes don't silently break V4 imatrix
- Validation: gate-loader, gate-coherence, gate-tools must pass on both quants
- Upload to `teamblobfish/DeepSeek-V4-Flash-GGUF` HF repo under `IQ1_S/` and `IQ1_M/` subdirs

### Out of scope (explicit non-goals)

- IQ2 variants (IQ2_XXS, IQ2_M) — separate future work
- Domain-matched (SWE/agent-flavored) calibration corpus — start with wikitext-103, swap later only if IQ1 quality on agent tasks is anomalously bad
- Terminal-Bench validation of IQ1 quants — IQ1 quality on real agent tasks is unknown and not gated on this work
- Upstreaming the imatrix fix to ggml-org/llama.cpp — happens with the rest of the V4 port post-#21149

## Success criteria

1. `llama-imatrix` runs end-to-end on V4 Q8 source without crashing, collecting activations from all attention projections (`attn_q_a/b`, `attn_kv`, `attn_output_a/b`) and MoE expert tensors (`ffn_*_exps`)
2. `tests/v4-port/gate-imatrix.sh` passes (regression-protects #1)
3. `IQ1_S` and `IQ1_M` GGUFs exist, load cleanly, decode coherently, pass `gate-tools.sh` (5/5 tool calls)
4. Both quants uploaded to `teamblobfish/DeepSeek-V4-Flash-GGUF` under `IQ1_S/` and `IQ1_M/` subdirs, model card updated

## Architecture

Two split tasks mirroring the existing G/H pattern:

```
feat/v4-port (target)
├── feat/v4-port-I-imatrix  → fix imatrix, produce calibration data, add gate
└── feat/v4-port-J-iq1-quants  → build IQ1 quants from I's output, validate, upload
```

**Hard ordering:** Task I must merge before Task J starts (Task J consumes `imatrix-v4-flash.dat`). No parallelism between I and J. Within Task J, IQ1_S and IQ1_M can build sequentially or in parallel (different output files, same Q8 input).

**Why split:**
- Task I is genuinely reusable. Once imatrix works on V4, any future quant build benefits — IQ2 variants, custom recipes, anything. Worth merging to `feat/v4-port` independently.
- Cleaner handoff: Task J's input is just "an `imatrix.dat` file at this path." Lets us regenerate quants without rerunning the fix.
- Easier review: two smaller PRs review more cleanly than one combined "fix+build" PR with mixed concerns.

## Task I — imatrix fix

**Branch:** `feat/v4-port-I-imatrix` off `feat/v4-port`. Merge back once gate passes.

### Phase 2.1 — Diagnose first, then fix

**Hard rule: do NOT pre-commit to a fix strategy.** Build a sanitized debug binary, reproduce, get an exact backtrace before writing any fix code:

```bash
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DLLAMA_SANITIZE_ADDRESS=ON
cmake --build build-asan --target llama-imatrix -j

./build-asan/bin/llama-imatrix \
  -m ~/models/DeepSeek-V4-Flash-Q8_0.gguf \
  -f wikitext-103-test.txt \
  --chunks 5 \
  -ngl 999
```

Output: `docs/plans/v4-port-imatrix-diagnosis.md` documenting:
- Exact crash site (file, line, op, tensor type/shape)
- Which V4 tensor or op triggered it
- Which fix strategy was chosen and why the others were rejected

### Phase 2.2 — Apply the fix

Three candidate strategies, ranked by likelihood from the brainstorm. The diagnosis picks one:

| # | Strategy | Likely if backtrace shows… |
|---|---|---|
| 1 | I32-passthrough in imatrix collector — skip integer tensors before any dereference | Crash on `ffn_gate_tid2eid.weight` (I32) or similar lookup tensor |
| 2 | Op-class skip — `if (t->op not in {MUL_MAT, MUL_MAT_ID}) return false` early | Crash inside the V4-specific op (`LIGHTNING_INDEXER`, `DSV4_HC_*`) when collector tries to read its output |
| 3 | MUL_MAT_ID expert-routing layout fix — handle V4's `n_as` / shape correctly at lines 263–306 | Crash inside the ID unpacking loop with V4's `ffn_gate_inp` shape |

**Constraint:** the fix must be the smallest patch that satisfies success criterion #1. Don't refactor imatrix to be V4-aware globally if a 5-line skip works.

### Phase 2.3 — Run end-to-end

Calibration data: `wikitext-103-raw-v1` test split (the standard imatrix calibration set, ~1M tokens). Download via the standard llama.cpp tooling:

```bash
# Either: HF datasets API (preferred — handles the file split layout cleanly)
hf download wikitext --repo-type dataset --include "wikitext-103-raw-v1/test-*.parquet" \
  --local-dir tests/v4-port/calibration/

# Or: direct curl from a known mirror; the implementer may use whichever
# canonical wikitext-103 source the rest of the llama.cpp imatrix workflow uses.
```

The implementer should follow the existing llama.cpp imatrix workflow for sourcing wikitext-103; the exact URL is not load-bearing on this spec, only the corpus identity.

Imatrix run (1000 chunks, ~1M tokens):

```bash
./build/bin/llama-imatrix \
  -m ~/models/DeepSeek-V4-Flash-Q8_0.gguf \
  -f wikitext-103-test.txt \
  -o ~/work/llama.cpp/imatrix-v4-flash.dat \
  --chunks 1000 \
  -ngl 999
```

**Verify output coverage:** tensor count in `imatrix-v4-flash.dat` should match expected (43 layers × 5 attention projections = 215 attention tensors + 43 layers × 256 experts × 3 expert tensors). If coverage is materially below expected, the fix is too aggressive (skipping too much) — re-evaluate.

### Phase 2.4 — Regression gate

Create `tests/v4-port/gate-imatrix.sh`:
- Runs `llama-imatrix` against V4 Q8 with `--chunks 2` (tiny, ~30s)
- Asserts no segfault (`set -e`)
- Asserts the output `.dat` file exists and has tensor count ≥ expected minimum

Add to `tests/v4-port/run-all-gates.sh`. Future PRs that touch imatrix will catch V4 regressions before merge.

### Task I deliverables

- Patched `tools/imatrix/imatrix.cpp` on `feat/v4-port-I-imatrix`
- `tests/v4-port/calibration/imatrix-v4-flash.dat` checked into the fork (small file, ~few MB) so Task J doesn't need to regenerate it
- `tests/v4-port/gate-imatrix.sh` added
- `docs/plans/v4-port-imatrix-diagnosis.md` written
- Branch merged to `feat/v4-port`

## Task J — build IQ1_S + IQ1_M

**Branch:** `feat/v4-port-J-iq1-quants` off `feat/v4-port` *after* Task I is merged.

**Dependency:** `tests/v4-port/calibration/imatrix-v4-flash.dat` from Task I.

### Recipes

```bash
# IQ1_S — smallest viable
./build/bin/llama-quantize --allow-requantize \
  --imatrix tests/v4-port/calibration/imatrix-v4-flash.dat \
  --output-tensor-type q5_K \
  --token-embedding-type q5_K \
  ~/models/DeepSeek-V4-Flash-Q8_0.gguf \
  ~/models/DeepSeek-V4-Flash-IQ1_S.gguf \
  IQ1_S
# Expected: ~52 GiB, 1.5 BPW body + Q5_K output/embed

# IQ1_M — slight-quality-bump variant
./build/bin/llama-quantize --allow-requantize \
  --imatrix tests/v4-port/calibration/imatrix-v4-flash.dat \
  --output-tensor-type q5_K \
  --token-embedding-type q5_K \
  ~/models/DeepSeek-V4-Flash-Q8_0.gguf \
  ~/models/DeepSeek-V4-Flash-IQ1_M.gguf \
  IQ1_M
# Expected: ~60 GiB, 1.75 BPW body + Q5_K output/embed
```

### Recipe rationale

**Why Q5_K on output_tensor + token_embd at IQ1:** IQ1 body is borderline for special-token discrimination. Pinning output at Q5_K is the bartowski-style standard for IQ1 builds and protects the tool-call grammar trigger (`<｜DSML｜tool_calls>`). If `gate-tools` fails: bump to Q6_K, rebuild, retry. If still fails: declare that quant unviable for agent use, document, ship anyway as a research artifact.

**Why no XL/XXL-style pinning on attn/indexer/hc:** IQ1 quality is dominated by body bit-rate, not the same context-scaling kernels that drove the Q2_K-XXL recipe. Adding XXL pins would inflate IQ1_S to ~58 GiB — at that size IQ1_M plain is the better choice. Keep the recipe minimal for "smallest possible."

### Validation gates

For each quant:
1. `tests/v4-port/gate-loader.sh` — architecture recognized
2. `NGL=999 tests/v4-port/gate-coherence.sh` — coherent decode at temp 0
3. `tests/v4-port/gate-tools.sh` — 5/5 weather-fixture tool calls

If both quants pass → success. If only one passes → ship the one that works, document the other.

### Distribution

After gate validation, split each into 50 GiB shards and upload to existing HF repo:

```bash
# Split (each fits in 1-2 shards at 50 GiB max-size)
mkdir -p ~/models/DeepSeek-V4-Flash-GGUF/IQ1_S ~/models/DeepSeek-V4-Flash-GGUF/IQ1_M

./build/bin/llama-gguf-split --split --split-max-size 50G \
  ~/models/DeepSeek-V4-Flash-IQ1_S.gguf \
  ~/models/DeepSeek-V4-Flash-GGUF/IQ1_S/DeepSeek-V4-Flash-IQ1_S

./build/bin/llama-gguf-split --split --split-max-size 50G \
  ~/models/DeepSeek-V4-Flash-IQ1_M.gguf \
  ~/models/DeepSeek-V4-Flash-GGUF/IQ1_M/DeepSeek-V4-Flash-IQ1_M

# Upload (use the recipe in blobfish docs/hf-quant-uploads.md)
HF_TOKEN=hf_xxx hf upload teamblobfish/DeepSeek-V4-Flash-GGUF \
  ~/models/DeepSeek-V4-Flash-GGUF/IQ1_S IQ1_S \
  --commit-message "Add IQ1_S shards"

HF_TOKEN=hf_xxx hf upload teamblobfish/DeepSeek-V4-Flash-GGUF \
  ~/models/DeepSeek-V4-Flash-GGUF/IQ1_M IQ1_M \
  --commit-message "Add IQ1_M shards"
```

**README update:** Add `IQ1_S/` and `IQ1_M/` rows to the quant table on the HF model card. Include the imatrix calibration corpus in the README ("calibrated against wikitext-103, 1000 chunks").

### Task J deliverables

- `~/models/DeepSeek-V4-Flash-IQ1_S.gguf` and `~/models/DeepSeek-V4-Flash-IQ1_M.gguf` produced
- Both pass loader + coherence + gate-tools (or documented as unviable)
- Both uploaded to `teamblobfish/DeepSeek-V4-Flash-GGUF` under `IQ1_S/` and `IQ1_M/` subdirs
- HF README updated with new quant rows + calibration provenance line
- Branch merged to `feat/v4-port`

## Artifacts inventory

| Artifact | Path | Size | Owner |
|---|---|---|---|
| Patched imatrix | `tools/imatrix/imatrix.cpp` | small diff | Task I |
| Calibration data | `tests/v4-port/calibration/imatrix-v4-flash.dat` | ~few MB | Task I |
| Imatrix regression gate | `tests/v4-port/gate-imatrix.sh` | small | Task I |
| Diagnosis writeup | `docs/plans/v4-port-imatrix-diagnosis.md` | small | Task I |
| `IQ1_S` GGUF | `~/models/DeepSeek-V4-Flash-IQ1_S.gguf` (single) + split shards | ~52 GiB | Task J |
| `IQ1_M` GGUF | `~/models/DeepSeek-V4-Flash-IQ1_M.gguf` (single) + split shards | ~60 GiB | Task J |
| HF: `IQ1_S/` subdir | `teamblobfish/DeepSeek-V4-Flash-GGUF/IQ1_S/*.gguf` | ~52 GiB | Task J |
| HF: `IQ1_M/` subdir | `teamblobfish/DeepSeek-V4-Flash-GGUF/IQ1_M/*.gguf` | ~60 GiB | Task J |
| HF: README updated | `teamblobfish/DeepSeek-V4-Flash-GGUF/README.md` | small edit | Task J |

## Disk budget

- Free at design time: ~589 GiB (after `docs/hf-quant-uploads.md` cleanup)
- Task I: imatrix reads Q8 source (already on disk), writes a few-MB `.dat`. Negligible disk impact.
- Task J: writes IQ1_S (~52 GiB) + IQ1_M (~60 GiB) = ~112 GiB during build. Comfortable headroom.

## Open risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Diagnosis reveals a fundamental V4-imatrix architecture mismatch (e.g. needs full V4-aware activation collection) | Low | High — Task I expands well beyond scope | Stop and re-brainstorm; don't expand scope silently |
| imatrix produces low-coverage data (collects from too few tensors) — IQ1 quality unexpectedly bad | Medium | Medium — IQ1 may be unusable | Phase 2.3's tensor-count check (43 layers × 5 attention + 43 × 256 × 3 expert tensors) flags this before declaring Task I done |
| Both IQ1 variants fail `gate-tools` even with Q5_K output pin | Medium | High — research goal of "smallest viable" not met | Documented escape hatches: bump output to Q6_K, then Q8_0; if all fail at IQ1 body, declare 1.5–1.75 BPW the floor below which V4 tool-calling collapses, ship as documented limitation |
| Disk runs out mid-build on Task J | Low | Medium | 589 GiB free; sequential build pattern + delete-source-after-split if needed |
| HF upload of IQ1 shards fails partway | Low | Low | `hf upload` is resumable; just rerun |
| Calibration with wikitext-103 produces poor agent-task quality | Unknown | Low (this work doesn't gate on agent quality per scope) | Documented as out-of-scope; future work can swap calibration corpus |

## Explicit non-mitigations

- We are *not* attempting to validate IQ1 quality against agent benchmarks — explicitly excluded from scope. If IQ1 quants are useless on real tasks, that's information for future work, not a Task J failure.
- We are *not* attempting to upstream the imatrix fix in this work. Same upstreaming gate as the rest of the V4 port (waits on V3.2 PR #21149).

## Implementation note

After this design is approved, hand off to the `writing-plans` skill to produce the actual `.claude/agents/v4-port-I-imatrix.md` and `v4-port-J-iq1-quants.md` task files for the dev-team subagent dispatch.
