# v4-p1-test-v32-end-to-end: V3.2 Metal validation harness

**Phase:** 1 — Metal kernels for V3.2 (gets us a Metal-capable V3.2 first)
**Complexity:** M
**Branch:** `feat/v4-p1-test-v32-end-to-end`
**Dependencies:** `v4-p1-metal-lightning-indexer-quant`, `v4-p1-metal-fattn-top-k`, `v4-p1-metal-fill-f16-verify`

## Scope

Two-part validation. **Part A (synthetic, always-on):** extend `tests/test-llama-archs.cpp`'s `LLM_ARCH_DEEPSEEK32` test to run on Metal (today it only tests model loading; needs a token of forward pass to exercise the new kernels). The arch test machinery already builds an in-process tiny GGUF for the arch via `get_gguf_ctx` (`tests/test-llama-archs.cpp:99-200`), so no external checkpoint is needed. Compares Metal logits against CPU baseline to within `1e-3` RMS. **Part B (model-conditioned, skipped if absent):** add a `tests/test-deepseek32-cli.sh` harness that, given a real DeepSeek V3.2 GGUF in a known location (path passed via env var, e.g. `LLAMA_TEST_DEEPSEEK32_GGUF`), runs `./build/bin/llama-cli` on a fixed prompt with both `--device CPU` and `--device Metal` and `diff`s the first 32 token IDs. Skipped on CI / when env var unset. **Out:** any kernel work.

## Acceptance criteria

- The change is contained to the file paths cited in the scope above. Do not edit unrelated subsystems.
- Any new code path is covered by at least one test in `tests/` (extend `test-backend-ops.cpp`, `test-llama-archs.cpp`, or add a dedicated test file as appropriate for the task).
- The build passes on the developer's M3 Ultra: `cmake --build build -j` exits 0.
- The relevant test command from `dev-team.json` `test_commands` passes locally.
- For Metal kernel tasks: confirm via `GGML_METAL=1` environment that the new path executes and matches CPU output to within `1e-3` RMS on the corresponding `test-backend-ops` case.
- For arch / loader / converter tasks: ensure the change does not regress any existing arch in `tests/test-llama-archs.cpp`.
- Commit messages reference the parent roadmap (`docs/plans/v4-roadmap.md`) and the task id.

## Out of scope

Anything explicitly listed under "**Out:**" in the scope paragraph above is **not** part of this task. If you find yourself doing it, stop and split out a follow-up task rather than expanding scope.

## How this task was generated

This spec was scaffolded by the v4-roadmap builder phase from row `v4-p1-test-v32-end-to-end` of `docs/plans/v4-roadmap.md`. Re-read the roadmap before you start — the surrounding context (Metal gap analysis in §2, V4 delta in §3, risks in §5) is load-bearing.
