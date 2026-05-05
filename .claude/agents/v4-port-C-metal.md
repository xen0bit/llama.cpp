# v4-port-C-metal: Metal kernels + speed optimization

## Goal
Cherry-pick antirez Metal commits: `b67f5db5c` (Optimize V4 Metal HC decode), `57c4283b5` (Remove stale quantize tool entry), `2f2d44052` (Speed up V4 prompt replay). Validate Metal forward pass + decode speed.

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-04-v4-port-overnight.md` Task 4 (Steps 4.1 through 4.6).** Follow it exactly.

## Depends on
v4-port-B-forward must be in state `done` before this task starts.

## Gates (BOTH must pass before code-review)
```bash
V4_GGUF=/Users/cchuter/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
NGL=999 ./tests/v4-port/gate-coherence.sh   # Coherence on Metal
./tests/v4-port/gate-speed.sh                # Decode > 10 tok/s on Metal
```
Expected: both PASS. Antirez reported 21 tok/s on M3 Max; this M3 Ultra should be ≥30 tok/s.

## Files (touched across 3 commits)
- Modify: `ggml/src/ggml-metal/ggml-metal.metal` (+89 across both Metal commits)
- Modify: `src/models/deepseek4.cpp`
- Modify: `src/llama-context.{cpp,h}`
- Modify: `tools/server/server-context.cpp`
- Modify: `tools/CMakeLists.txt` (-1)
- Modify: `ggml/src/ggml.c`

## Failure modes + recovery
- **Metal compile error in .metal file**: missing function declaration or Metal version mismatch. Have codex inspect the .metal diff against antirez's reference.
- **Coherent on CPU but gibberish on Metal**: Metal kernel port bug — buffer-binding or threadgroup-size issue. Compare bind list and dispatch params.
- **Speed < 10 tok/s**: Metal HC kernel optimization (b67f5db5c) didn't apply correctly. Check that the `n_hc=4` specialization was preserved.

## Iteration budget
- Up to 8 builder fix rounds
- Up to 8 codex review rounds

## Definition of done
- All 3 commits cherry-picked
- Build clean with Metal
- Both gates (coherence + speed) PASS
- Codex code-review APPROVED
- Pushed to `mine`
