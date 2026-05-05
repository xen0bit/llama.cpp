# v4-port-A-loader: cherry-pick V4 inference support (loader + forward pass)

## Goal
Cherry-pick antirez commit `06c504247` (Add DeepSeek V4 Flash inference support) onto `feat/v4-port`. This brings the V4 architecture enum, model loader, KV cache changes, and the entirety of `src/models/deepseek4.cpp` (the V4 forward pass).

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-04-v4-port-overnight.md` Task 2 (Steps 2.1 through 2.6).** Follow it exactly.

## Gate (must pass before code-review)
After cherry-pick + build, run:
```bash
V4_GGUF=/Users/cchuter/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  ./tests/v4-port/gate-loader.sh
```
Expected: `PASS: loader recognizes V4 GGUF`.

## Conflict resolution rule
When antirez and fairydreaming both touched a file:
- DSA / sparse attention / lightning indexer code → **keep fairydreaming's** (V3.2)
- V4-specific (deepseek4 arch, V4 routing, V4 expert handling) → **keep antirez's**
- Shared infrastructure (arch registry, conversion script tensor maps) → **MERGE BOTH**: V3.2 entries side-by-side with V4 entries

## Files (touched by 06c504247)
See `git show --stat 06c504247` — main impacts:
- New: `src/models/deepseek4.cpp` (1347 lines)
- Modify: `src/llama-arch.{cpp,h}`, `src/llama-model.{cpp,h}`
- Modify: `src/llama-memory-hybrid-iswa.{cpp,h}`
- Modify: `convert_hf_to_gguf.py`, `gguf-py/gguf/constants.py`

## Iteration budget
- Up to 8 builder fix rounds for build errors
- Up to 8 codex review rounds for code-review

## Definition of done
- Cherry-pick succeeded (or conflicts resolved per rule above)
- `cmake --build build -j` exit 0 with `GGML_METAL=ON`
- `gate-loader.sh` PASSES
- Codex code-review APPROVED (or auto-approved after 8 rounds)
- Pushed to `mine`
