# v4-port-B-forward: V4 long-context fix + validate CPU forward pass

## Goal
Cherry-pick antirez commit `188df615c` (Fix DeepSeek V4 long-context graph metadata). Validate that the CPU forward pass produces coherent tokens against the V4 GGUF.

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-04-v4-port-overnight.md` Task 3 (Steps 3.1 through 3.5).** Follow it exactly.

## Depends on
v4-port-A-loader must be in state `done` before this task starts.

## Gate (must pass before code-review)
```bash
V4_GGUF=/Users/cchuter/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
NGL=0 ./tests/v4-port/gate-coherence.sh
```
Expected: `PASS: coherence (NGL=0, gen='...')`. The gate checks: 30 tokens generated, >80% printable ASCII, no NaN, not a degenerate single-token decode.

## Files (touched by 188df615c)
- Modify: `src/llama-context.cpp` (+8)
- Modify: `src/llama-memory-hybrid-iswa.cpp` (+13)

## Failure mode + recovery
If gate-coherence FAILS with gibberish output (not NaN, not crash, just garbage tokens), the forward pass in `src/models/deepseek4.cpp` likely has a port bug. Most common cause: tensor-name mapping or hparam mis-load. Have codex compare our `src/models/deepseek4.cpp` against the antirez reference and find the mismatch.

## Iteration budget
- Up to 8 builder fix rounds
- Up to 8 codex review rounds

## Definition of done
- Cherry-pick succeeded
- `cmake --build build -j` exit 0
- `gate-coherence.sh` with NGL=0 PASSES
- Codex code-review APPROVED
- Pushed to `mine`
