# v4-port-J-iq1-quants: build IQ1_S + IQ1_M quants from imatrix calibration

## Goal
Produce the smallest practical V4-Flash GGUF artifacts (`IQ1_S` ~52 GiB and `IQ1_M` ~60 GiB) using the imatrix calibration data landed by Task I, validate via the standard V4 gates, split into 50 GiB shards, and publish to the existing `teamblobfish/DeepSeek-V4-Flash-GGUF` HF repo.

Success criterion: both quants load cleanly, decode coherently, pass `gate-tools.sh`, and are accessible on Hugging Face under `IQ1_S/` and `IQ1_M/` subdirs. If only one passes, ship the one that works and document the other.

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-07-v4-port-J-iq1-quants.md`. Follow it as your starting point.**

The plan was pre-written, but **the architect is allowed to revise it** if codex plan-review finds issues. Standard dev-team pipeline applies.

## Hard prerequisite
Task I (`feat/v4-port-I-imatrix`) must be merged to `feat/v4-port` first. This task consumes:
- `tests/v4-port/calibration/imatrix-v4-flash.dat` (Task I output)
- `tests/v4-port/gate-loader.sh`, `gate-coherence.sh`, `gate-tools.sh` (existing V4 gates)

If the calibration file is missing, abort — Task I has not landed yet.

## Phased plan summary
1. **Worktree setup + verify prerequisites** — branch off `feat/v4-port` (which contains Task I's merged work), confirm `imatrix-v4-flash.dat` exists.
2. **Build IQ1_S** — `llama-quantize --imatrix ... --output-tensor-type q5_K --token-embedding-type q5_K ... IQ1_S`. ~52 GiB output.
3. **Validate IQ1_S** — gate-loader + gate-coherence + gate-tools. If gate-tools fails: bump output pin to Q6_K and rebuild, OR declare unviable and continue.
4. **Build IQ1_M** — same recipe, target `IQ1_M`. ~60 GiB output.
5. **Validate IQ1_M** — same three gates. Same fallback rules.
6. **Split + ship** — `llama-gguf-split --split-max-size 50G` on each, delete single-files, upload subdirs to HF, update README quant table.

## Recipe rationale (locked, do not deviate without re-brainstorming)
- **Q5_K on `output_tensor` and `token_embd`:** IQ1 body alone (1.5 BPW) collapses special-token discrimination. Q5_K is the bartowski-style standard for IQ1 builds and protects the `<｜DSML｜tool_calls>` grammar trigger.
- **No XL/XXL-style attn/indexer/hc pinning:** that recipe was Q2_K-specific for context-scaling kernels. At IQ1 quality is dominated by body bit-rate, and adding XXL pins inflates IQ1_S to ~58 GiB which makes IQ1_M plain (also ~58 GiB) the better choice. Keep recipe minimal.

## Gate (must pass before code-review)
For each quant that ships:
```bash
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_S.gguf ./tests/v4-port/gate-loader.sh
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_S.gguf NGL=999 ./tests/v4-port/gate-coherence.sh
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_S.gguf ./tests/v4-port/gate-tools.sh
```
(and parallel for IQ1_M).

A quant ships only if all three gates PASS. A quant that fails gate-tools at both Q5_K and Q6_K output pins is documented as unviable in the README and not uploaded.

## Test scope
Loader + coherence + tool-call validation per quant. **Out of scope:** Terminal-Bench validation, comparison against larger quants on agent tasks. IQ1 quality on real workloads is documented as unknown and is information for future work.

## What this task explicitly does NOT do
- Validate IQ1 quality on Terminal-Bench or any agent benchmark
- Build IQ2_XXS / IQ2_M variants (separate future work)
- Modify any source code in llama.cpp itself
- Re-run the imatrix calibration (Task I's deliverable)
- Upstream anything to ggml-org/llama.cpp

## Iteration budget
- Up to 4 builder rebuild rounds (e.g. Q5_K → Q6_K bump if gate-tools fails)
- Up to 4 codex review rounds for plan-review and code-review (high reasoning effort, fall back to default on stalls)

## Branch
- `feat/v4-port-J-iq1-quants` off `feat/v4-port` *after Task I is merged*
- Worktree isolation required per plan setup section

## Disk budget
- ~589 GiB free at task start (per cleanup recorded in `~/work/blobfish/docs/hf-quant-uploads.md`)
- Build pipeline: IQ1_S (~52 GiB) + IQ1_M (~60 GiB) = ~112 GiB during build, freed after split + delete single-files
- Comfortable headroom

## Definition of done
- `IQ1_S` and `IQ1_M` GGUFs built (or one declared unviable with documented reason)
- Both pass `gate-loader.sh`, `gate-coherence.sh`, `gate-tools.sh` (or unviable + documented)
- Split shards under `~/models/DeepSeek-V4-Flash-GGUF/IQ1_S/` and `IQ1_M/`
- Both uploaded to `teamblobfish/DeepSeek-V4-Flash-GGUF` HF repo, total size matches local
- HF README updated with IQ1 rows in the quant table + calibration provenance note
- Branch `feat/v4-port-J-iq1-quants` pushed to `mine`
- Codex plan-review and code-review both APPROVE
