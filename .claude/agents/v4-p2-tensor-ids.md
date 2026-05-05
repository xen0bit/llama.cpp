# v4-p2-tensor-ids: New tensor IDs and GGUF keys for V4-specific weights

**Phase:** 2 — V4 architecture skeleton (loadable, even if forward pass is stubbed)
**Complexity:** M
**Branch:** `feat/v4-p2-tensor-ids`
**Dependencies:** `v4-p2-arch-enum`

## Scope

`gguf-py/gguf/constants.py`: add `OUTPUT_HC_BASE / FN / SCALE`, `*COMPRESS*` tensor names. Add `LLM_KV_HYPER_CONNECTION_*`, `LLM_KV_ATTENTION_COMPRESS_RATIOS`, `LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE`, `LLM_KV_ATTENTION_OUTPUT_LORA_RANK`, `LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT`, `LLM_KV_HASH_LAYER_COUNT`, `LLM_KV_SWIGLU_CLAMP_EXP`. Wire each to `add_*` helpers in `gguf-py/gguf/gguf_writer.py`. **Out:** runtime use of these — Phase 3.

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

This spec was scaffolded by the v4-roadmap builder phase from row `v4-p2-tensor-ids` of `docs/plans/v4-roadmap.md`. Re-read the roadmap before you start — the surrounding context (Metal gap analysis in §2, V4 delta in §3, risks in §5) is load-bearing.
