# v4-port-G-kv-q8: fix V4 silent corruption under `--cache-type-k|v q8_0`

## Goal
When a user runs llama-server / llama-completion against a V4 GGUF with `--cache-type-k q8_0 --cache-type-v q8_0`, currently the model produces silent garbage (`=`-loops, `"Mirror …"` artifacts, single-character output). Make the V4 runtime either **(a) produce coherent output under q8 KV** or **(b) fail-fast at startup with a clear diagnostic**. Eliminate the silent corruption.

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-05-v4-port-G-kv-q8.md`. Follow it as your starting point.**

The plan was pre-written, but **the architect is allowed to revise it** if codex plan-review finds issues. Standard dev-team pipeline applies (REVISE → architect updates plan → codex re-reviews → APPROVE or escalate after 2 rounds).

## Phased plan summary
1. **Diagnose** — boot V4 server with `--cache-type-k|v q8_0` and bisect which V4 K tensor corrupts: `cache.attn_k` (compressed-attention K) at `src/llama-memory-hybrid-iswa.cpp:226`, `cache.index_k` (indexer K, only when ratio==4) at `src/llama-memory-hybrid-iswa.cpp:231`, or the standard K cache in `src/llama-kv-cache.cpp`. Replace `type_k` with `GGML_TYPE_F16` one tensor at a time and run `gate-server-chat-q8.sh` to observe.
2. **Root cause** — read the q8_0 quantize/dequantize paths in `ggml/src/ggml-cpu/quants.c` and `ggml/src/ggml-metal/ggml-metal.metal` against V4's K layout. Document why q8_0 fails for V4 latent KV (most likely: per-block stationarity violated by compressed/indexer representations).
3. **Fix** — three possible paths, architect picks based on Phase 2 findings:
   - **Best case**: fix the kernel or cache assembly so q8 KV just works.
   - **Realistic case** (most likely outcome): pin V4's compressed/indexer K to `GGML_TYPE_F16` unconditionally regardless of `--cache-type-k`. Log a single-line `LLAMA_LOG_WARN` on first override, with text matching `EXPECTED_WARN` in the regression gate.
   - **Bail case**: error fast at startup when V4 + non-fp16 KV is requested. CLI guard in `common/arg.cpp` or `tools/server/server-task.cpp`.
4. **Regression test** — add `tests/v4-port/gate-server-chat-q8.sh` (mirror of `gate-server-chat.sh` but with `--cache-type-k|v q8_0` flags). Behavior depends on Phase 3 fix path. Wire into `tests/v4-port/run-all-gates.sh`.

## Gate (must pass before code-review)
After fix, run:
```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  ./tests/v4-port/run-all-gates.sh
```
Expected: `ALL GATES PASS` including the new `gate-server-chat-q8.sh`.

## Test scope
The new gate adds q8_0 KV behavior validation. The existing 5 gates (loader, coherence×2, speed, tools, server-chat) must still pass — fp16 KV path must not regress.

## What this task explicitly does NOT do
- Improve `--cache-type-k|v` for non-V4 architectures
- Tune V4's KV cache memory footprint (separate concern)
- Fix the parallel `llama-imatrix` segfault on V4 (separate task)
- Touch the V4 chat template, converter, or quantization tooling

## Iteration budget
- Up to 8 builder fix rounds
- Up to 8 codex review rounds for plan-review and code-review (high reasoning effort, fall back to default on stalls per spec)

## Branch
- `feat/v4-port-G-kv-q8` off `feat/v4-port` (which is at `a173e2e36`, includes H's deliverables)
- Worktree isolation required

## Definition of done
- Phase 1 diagnosis identifies the corrupting tensor(s) with evidence in the completion doc
- Phase 3 fix lands; either q8 KV produces coherent output or the server fails fast with a clear message
- `gate-server-chat-q8.sh` exists, asserts the chosen behavior, wired into `run-all-gates.sh`
- All existing gates still pass against `Q4_K_M` (fp16 KV path)
- Codex plan-review and code-review both APPROVE
- `docs/plans/v4-port-kv-q8-completion.md` committed with bisection + root cause + fix layer + ruled-out hypotheses
- Branch `feat/v4-port-G-kv-q8` pushed to `mine`

## Cross-cutting ground rules
- Never push to `origin` (= ggml-org/llama.cpp upstream)
- Never run `gh pr create`
- Don't amend existing commits
- Don't skip git hooks
- 8+8 exhaustion → mark `needs-human`
