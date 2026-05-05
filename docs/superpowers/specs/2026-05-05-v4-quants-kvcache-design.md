# V4 Port — Q4_K_M Quants and Q8 KV Cache Fix

Two follow-on dev-team tasks for the DeepSeek V4 Flash port. Builds on
`docs/plans/v4-port-debug-completion.md` (the chat-completion bug bisection
that landed the q8-KV-removed launcher).

## Background

The V4 port works end-to-end: `tests/v4-port/run-all-gates.sh` (loader,
coherence, speed, tools, server-chat) all PASS at HEAD `58693d523` on
`feat/v4-port`. Two open items remain:

1. **Quants.** Only the IQ2XXS GGUF (~80 GiB, 2.44 BPW) exists locally. That
   quant is too aggressive for agent workloads — recent terminal-bench trials
   show the model confusing `numpy.linalg.eig` and `scipy.linalg.eig` API
   surfaces, exactly the fine-grained factual recall that low-bit quants
   clobber first. We need higher-quality quants built from base weights.
2. **q8 KV cache silently corrupts V4.** `--cache-type-k q8_0 --cache-type-v
   q8_0` against V4 produces `=`-loops / `"Mirror …"` garbage. We worked
   around this by dropping the flags from
   `claude-cache-proxy/start-server-v4.sh`, but the underlying bug is open
   and the runtime should not silently corrupt output.

Both items have stable scope and clear acceptance criteria, so they fit the
dev-team pipeline well.

## Decomposition

Two tasks, executed sequentially on per-task feature branches. **H first**
because the Q4_K_M quant unblocks meaningful improvements to agentic
behavior (the IQ2XXS we have was confusing `numpy.linalg.eig` with
`scipy.linalg.eig` in terminal-bench — exactly the API recall a 4-bit quant
should fix). G follows because the q8 KV bug is already worked around in
the launcher (`claude-cache-proxy/start-server-v4.sh` doesn't pass those
flags), so finishing the underlying fix is hygiene, not unblocking.

```
feat/v4-port  ──┬─→ feat/v4-port-H-quants  ──merge──→ feat/v4-port  ──┬─→ feat/v4-port-G-kv-q8  ──merge──→ feat/v4-port
                │                                                     │
                H dispatched now (base weights cloned)                G dispatched after H merges
```

**Why sequential, not parallel.**
- Both tasks need the M3 Ultra for validation. Concurrent server boots
  would thrash GPU/disk.
- Q4_K_M (from H) becomes available as additional regression coverage for
  G's q8-KV fix — G can validate against both IQ2XXS and Q4_K_M.
- Wall-clock is fine: H is ~2 hours of convert+quantize+gate runs; G is
  ~1-2 days of focused investigation work.

**Why per-task branches.**
- Both touch significant code surface (G touches kernels possibly; H touches
  `convert_hf_to_gguf.py` plus configs). Codex code-review benefits from a
  clean per-task diff.
- Matches the dev-team's standard pattern. The previous "no branches" choice
  was for the small chat-template bisection; these are larger.

## Task G — `v4-port-G-kv-q8`

**Goal.** When a user passes `--cache-type-k q8_0 --cache-type-v q8_0`
against a V4 model, either produce coherent output (proper q8 KV support)
or fail at startup with a clear diagnostic. Eliminate silent corruption.

**Phased plan** (architect should reproduce this structure in the
implementation plan; the chosen Phase-3 path depends on Phase-1+2 findings):

### Phase 1 — diagnose

Boot a V4 server with `--cache-type-k q8_0 --cache-type-v q8_0` and an
instrumented build (or via targeted prints). Identify which V4-specific KV
tensor corrupts.

Suspects, in priority order:
1. `cache.attn_k` in `src/llama-memory-hybrid-iswa.cpp:226` — V4 compressed-
   attention K cache, stores latent (post-compression) representation
2. `cache.index_k` in `src/llama-memory-hybrid-iswa.cpp:231` — V4 indexer K
   cache for sparse attention; only allocated when compression ratio is 4
3. The standard K cache (if V4 still allocates one alongside the compressed
   path) in `src/llama-kv-cache.cpp`

**Diagnostic procedure.** Replace each `type_k` use with `GGML_TYPE_F16` one
at a time (in a throwaway WIP commit), rebuild, and run
`tests/v4-port/gate-server-chat.sh` patched to use q8_0 K. The first
substitution that restores coherent output points at the corrupting tensor.
Time budget: ~2-4 hours.

### Phase 2 — root cause

For whichever tensor corrupts, identify *why* q8_0 fails. Likely candidates:
- Latent K head_dim (`n_embd_head_k=512`, `indexer_head_size`) layout
  mismatches the q8 kernel's per-head striding assumption
- The q8 quant kernel assumes a tensor shape V4 violates (e.g. last-dim
  block alignment for the 32-element Q8_0 block)
- The Metal q8 kernel has a V4-specific bug not seen for standard attention

**Procedure.** Read the q8 quantize/dequantize paths in
`ggml/src/ggml-cpu/quants.c` and `ggml/src/ggml-metal/ggml-metal.metal`
against V4's actual K layout (from `src/models/deepseek4.cpp:create_deepseek4_compressor`).
Confirm the failure mode: is the stored tensor wrong, the read-back wrong,
or both? Time budget: ~half-day.

### Phase 3 — fix

Land at the right layer based on Phase 2 findings. Architect must explicitly
state in the plan which path is taken and why.

- **Best case** — layout works after a small adjustment: fix the kernel or
  the cache assembly so q8 KV just works.
- **Realistic case** — latent K is genuinely incompatible with per-block q8
  quantization: force these specific V4 caches to fp16 unconditionally
  regardless of `--cache-type-k`. Log a single-line `LLAMA_LOG_WARN` about
  the override on first allocation.
- **Bail case** — the fix exceeds task scope: error out at startup when the
  model is `deepseek4` and `--cache-type-k|v` ≠ f16, with a clear message
  pointing at this issue. The CLI guard goes in either `common/arg.cpp` or
  `tools/server/server-task.cpp` (whichever is the right layer for the
  guard).

The "realistic case" is the most likely outcome and is acceptable as a fix
— V4's compressed/indexer K stores latent representations that have never
been intended for quantization.

### Phase 4 — regression test

Add `tests/v4-port/gate-server-chat-q8.sh`. Behavior depends on Phase 3
outcome:
- If q8 KV produces coherent output: boots `llama-server` with
  `--cache-type-k q8_0 --cache-type-v q8_0`, runs the same three curl tests
  as `gate-server-chat.sh`, asserts coherent output.
- If q8 KV is silently force-fp16'd: same as above plus an assertion that
  the server log contains the expected WARN line on startup.
- If q8 KV errors at startup: asserts the server exits non-zero with the
  expected diagnostic substring, and skips the curl tests.

Wire into `tests/v4-port/run-all-gates.sh`.

### Files likely touched

- `src/llama-memory-hybrid-iswa.cpp` (V4 cache type assembly)
- `src/llama-kv-cache.cpp` (if the standard K path is implicated)
- Possibly `ggml/src/ggml-cpu/quants.c` and `ggml/src/ggml-metal/ggml-metal.metal`
  (kernel-level fix)
- Possibly `common/arg.cpp` or `tools/server/server-task.cpp` (CLI guard,
  bail case)
- `tests/v4-port/gate-server-chat-q8.sh` (new)
- `tests/v4-port/run-all-gates.sh` (wire-in)
- `docs/plans/v4-port-kv-q8-completion.md` (followup writeup)

### Definition of done

- Phase 1 diagnosis identifies the corrupting tensor(s) with evidence
  recorded in the completion doc
- Phase 3 fix lands; either q8 KV produces coherent output across all three
  curl tests in `gate-server-chat-q8.sh`, or the server fails fast with a
  clear message
- `gate-server-chat-q8.sh` exists, asserts the chosen behavior, wired into
  `run-all-gates.sh`
- All existing gates still pass (no regression)
- Codex plan-review and code-review both APPROVE
- Followup writeup committed to `docs/plans/v4-port-kv-q8-completion.md`
- Branch `feat/v4-port-G-kv-q8` pushed to `mine`

### Branch / dispatch

- Spec: `.claude/agents/v4-port-G-kv-q8.md`
- JSON: `tasks/active/v4-port-G-kv-q8.json`, `state: "roadmap"`
- Branch: `feat/v4-port-G-kv-q8` off `feat/v4-port`
- Codex reasoning effort: `high` (revert to default if stalls observed)

## Task H — `v4-port-H-quants`

**Goal.** Produce a Q4_K_M GGUF for `deepseek-ai/DeepSeek-V4-Flash` from base
safetensors, validated against the existing V4 gate suite. Land V4 support in
`convert_hf_to_gguf.py` so the recipe is reproducible.

### Phased plan

#### Phase 1 — converter

Add `DeepseekV4ForCausalLM` (and `DeepseekV4FlashForCausalLM` if HF uses that
class name — confirm from the cloned `config.json`) to
`convert_hf_to_gguf.py`.

Subclass `DeepseekV32Model` (line 9221 in `convert_hf_to_gguf.py`) since V4
inherits V3.2's DSA / sparse attention pipeline. Add V4-specific tensor name
mappings:
- Hyper-connection weights
- Attention compressors (`attn_compressor_ape/kv/gate/norm`)
- Indexer compressors (`indexer_compressor_ape/kv/gate/norm`)
- Output LoRA groups (`attn_wo_a`, `attn_wo_b`, etc. with `n_out_groups`)
- KV compressors (`attn_kv`, `attn_kv_a_norm`)

Hparams to surface:
- `attn_compress_ratio[]` (per-layer, variable)
- `indexer_n_head`, `indexer_head_size`, `indexer_top_k`

Source of truth: `src/models/deepseek4.cpp:load_arch_hparams` and
`load_arch_tensors`. Cross-check against the cloned model's `config.json`.

Verify (don't duplicate): `gguf-py/gguf/constants.py` and
`gguf-py/gguf/tensor_mapping.py` already have V4 entries from antirez's port.
If anything is missing, add it; if all present, leave alone.

Expected delta: ~200-400 lines in `convert_hf_to_gguf.py`, possibly small
additions to `gguf-py/gguf/`.

#### Phase 2 — convert + quantize

Preconditions (architect's plan must include these as fail-fast checks):
- `~/models/DeepSeek-V4-Flash/` directory exists
- `git -C ~/models/DeepSeek-V4-Flash status` shows clean working tree (no
  `.lock` files, all `.safetensors` files present, total size matches what
  HF advertises)
- Free disk space ≥ 800 GiB on the volume holding `~/models/`

Convert:
```bash
python3 convert_hf_to_gguf.py ~/models/DeepSeek-V4-Flash \
  --outfile ~/models/DeepSeek-V4-Flash-F16.gguf --outtype f16
```
Produces ~570 GiB f16 GGUF. Wall-clock ~30-60 min on the M3 Ultra.

Quantize:
```bash
./build/bin/llama-quantize \
  ~/models/DeepSeek-V4-Flash-F16.gguf \
  ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  Q4_K_M
```
Produces ~150 GiB Q4_K_M. Wall-clock ~30-90 min.

Capture sha256 of both output files for the followup doc.

#### Phase 3 — validate

Run `V4_GGUF=~/models/DeepSeek-V4-Flash-Q4_K_M.gguf ./tests/v4-port/run-all-gates.sh`,
including Task G's `gate-server-chat-q8.sh` (G has merged by then). All
gates must PASS. Capture decode tok/s in `gate-speed.sh` for the followup doc
— Q4_K_M should outperform IQ2XXS on quality and likely match or beat it on
speed.

If a gate fails on Q4_K_M but passes on IQ2XXS: diagnose. The most likely
cause is a converter bug (wrong tensor name, wrong type, missing hparam),
not a quantization issue. Fix and re-run.

### Files likely touched

- `convert_hf_to_gguf.py` (new V4 model class)
- `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py` (verify; add
  only if missing)
- `tests/v4-port/run-all-gates.sh` (no-op expected; gate scripts already
  honor `V4_GGUF` env var)
- `docs/plans/v4-port-quants-completion.md` (followup with how-to-rebuild,
  perf numbers, sha256s)

### What we explicitly do NOT commit

- The Q4_K_M GGUF itself (~150 GiB)
- The intermediate F16 GGUF (~570 GiB)
- `~/models/` is not in git. Document paths and sha256 in the followup doc.

### Definition of done

- `convert_hf_to_gguf.py` recognizes `DeepseekV4ForCausalLM` (and
  `DeepseekV4FlashForCausalLM` if applicable)
- `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` exists, is valid (passes
  `gate-loader.sh`)
- `run-all-gates.sh` (including `gate-server-chat-q8.sh` from G) all PASS
  against the new Q4_K_M
- Decode tok/s recorded for the new quant
- Followup writeup committed to `docs/plans/v4-port-quants-completion.md`
  with recipe + sha256 + perf numbers
- Codex plan-review and code-review both APPROVE
- Branch `feat/v4-port-H-quants` pushed to `mine`

### Branch / dispatch

- Spec: `.claude/agents/v4-port-H-quants.md`
- JSON: `tasks/active/v4-port-H-quants.json`, `state: "roadmap"`
- Branch: `feat/v4-port-H-quants` off `feat/v4-port` (after G merges)
- Codex reasoning effort: `high` (revert to default if stalls observed)

## Dispatch protocol

Per the existing dev-team skill, each task moves through:

```
roadmap → planning → plan-review → implementing → code-review → testing → done
```

with codex review (`model_reasoning_effort=high`) at both `plan-review` and
`code-review` gates. **Watch for stalls.** If a codex call hangs more than
10 minutes, kill it, retry once at default reasoning, and note it in the
task history.

Budget per task: 8 codex review rounds + 8 builder fix rounds before
flagging `needs-human`.

Builder runs in worktree isolation (`isolation: "worktree"`), feature branch
off `feat/v4-port`. Push to `mine` freely. **No `gh pr create`.** Manual merge
gate between G and H — human reviews G's branch, merges into `feat/v4-port`,
pushes, then dispatches H.

### Dispatch order

1. **Now:** write H's spec (`.claude/agents/v4-port-H-quants.md`) and JSON
   (`tasks/active/v4-port-H-quants.json`). Commit on `feat/v4-port`.
2. **Now:** dispatch the dev-team orchestrator with H only in the task list
   (background agent).
3. **After H merges to `feat/v4-port`:** write G's spec and JSON, commit,
   dispatch the orchestrator with G only.

### Cross-cutting ground rules

- Never push to `origin` (= ggml-org/llama.cpp upstream)
- Never run `gh pr create`
- Don't amend existing commits
- Don't skip git hooks
- Codex review at high reasoning effort, fall back to default on stall
- 8+8 exhaustion → mark `needs-human`, surviving streams continue

## Out of scope

- Other quant levels (Q6_K, Q8_0, IQ4_XS, etc.) — H targets Q4_K_M only.
  Future tasks can re-quantize from the F16 produced by H.
- Imatrix calibration. Not used for the IQ2XXS we have; we'll use
  whatever calibration baseline `llama-quantize` defaults to for Q4_K_M.
- An end-to-end terminal-bench validation under the new quant. That's a
  workflow change, not a code change; queue separately.
- Improvements to the V4 chat template, reasoning extraction, or any other
  part of the V4 port. Both tasks must stay narrowly scoped.
