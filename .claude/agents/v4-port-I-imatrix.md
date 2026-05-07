# v4-port-I-imatrix: fix V4 imatrix segfault and produce calibration data

## Goal
`llama-imatrix` currently segfaults during the first chunk's forward pass when run against a V4 GGUF, blocking IQ-series quant builds. Diagnose the crash, apply the smallest viable fix, produce a calibration `imatrix.dat` artifact from wikitext-103, and add a regression gate.

Success criterion: `llama-imatrix` runs end-to-end on V4 Q8 source without crashing, collecting activations from all attention projections (`attn_q_a/b`, `attn_kv`, `attn_output_a/b`) and MoE expert tensors (`ffn_*_exps`).

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-07-v4-port-I-imatrix.md`. Follow it as your starting point.**

The plan was pre-written, but **the architect is allowed to revise it** if codex plan-review finds issues. Standard dev-team pipeline applies (REVISE → architect updates plan → codex re-reviews → APPROVE or escalate after 2 rounds).

## Phased plan summary
1. **Build sanitized debug binary** — `cmake -DCMAKE_BUILD_TYPE=Debug -DLLAMA_SANITIZE_ADDRESS=ON`.
2. **Reproduce + diagnose** — run `llama-imatrix` against V4 Q8 with a tiny calibration sample, capture full backtrace + ASan report, write `docs/plans/v4-port-imatrix-diagnosis.md` naming the exact crash site, root cause, and chosen fix strategy. **Hard rule: do not pre-commit to a strategy — diagnose first.**
3. **Apply fix** — three candidates ranked by likelihood:
   - **Strategy 1 (most likely):** I32-passthrough in the imatrix collector, mirroring the analogous fix in `src/llama-quant.cpp::tensor_allows_quantization`.
   - **Strategy 2:** op-class skip — bail early on V4-specific ops (`LIGHTNING_INDEXER`, `DSV4_HC_*`) that the collector doesn't understand.
   - **Strategy 3 (last resort, scope check):** `MUL_MAT_ID` expert-routing layout fix at `tools/imatrix/imatrix.cpp:263-306`. If this strategy is chosen and the patch grows beyond ~30 lines, stop and re-brainstorm.
4. **Add regression gate** — `tests/v4-port/gate-imatrix.sh` runs `llama-imatrix` with `--chunks 2` and asserts no crash + minimum tensor count (≥50). Wire into `run-all-gates.sh`.
5. **Produce calibration data** — full imatrix run on wikitext-103-raw-v1 test split (1000 chunks, ~1M tokens). Output: `tests/v4-port/calibration/imatrix-v4-flash.dat` (~few MB, committed to fork).
6. **Verify tensor coverage** — script in plan checks ≥200 attention tensors + ≥100 expert MoE tensors collected. Catches "fix accidentally skips everything" regressions.

## Gate (must pass before code-review)
After the fix and calibration are in place, run:
```bash
./tests/v4-port/run-all-gates.sh
```
Expected: `ALL GATES PASS` including the new `gate-imatrix.sh`.

## Test scope
The new gate adds imatrix-runs-end-to-end validation. The existing 8 gates (loader, coherence×2, speed, tools, server-chat, server-chat-q8 ×2) must still pass — the fix must not regress non-imatrix paths.

## What this task explicitly does NOT do
- Build IQ1 / IQ2 quants (that's Task J)
- Refactor imatrix to be V4-aware globally (smallest viable patch only)
- Validate the produced imatrix data on agent benchmarks (out of scope per spec)
- Upstream the fix to ggml-org/llama.cpp (gated on V3.2 PR #21149)

## Iteration budget
- Up to 6 builder fix rounds
- Up to 4 codex review rounds for plan-review and code-review (high reasoning effort, fall back to default on stalls per dev-team standard)

## Branch
- `feat/v4-port-I-imatrix` off `feat/v4-port` (currently at `fe470de1c`, includes G + H)
- Worktree isolation required per plan setup section

## Definition of done
- `tools/imatrix/imatrix.cpp` patched with smallest viable fix per diagnosis
- `docs/plans/v4-port-imatrix-diagnosis.md` written, names exact crash site + chosen strategy + rejected alternatives
- `tests/v4-port/gate-imatrix.sh` created, executable, passes locally
- `tests/v4-port/run-all-gates.sh` includes the new gate, full suite passes
- `tests/v4-port/calibration/imatrix-v4-flash.dat` exists and committed (≥200 attention + ≥100 expert tensors covered)
- Branch `feat/v4-port-I-imatrix` pushed to `mine`
- Codex plan-review and code-review both APPROVE
