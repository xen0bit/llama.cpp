# v4-p3-quant-pipeline: End-to-end FP4/FP8 → GGUF quantization recipe for V4 on M3 Ultra

**Phase:** 3 — V4 forward pass (CPU + Metal)
**Complexity:** M
**Branch:** `feat/v4-p3-quant-pipeline`
**Dependencies:** `v4-p2-conversion-script`

## Scope

Define the quantization pipeline that takes the HF FP4+FP8 release and lands a GGUF that fits in 192 GB unified memory with KV cache headroom. Two stages: (a) `convert_hf_to_gguf.py` writes routed-expert tensors as F32 (post-FP4-dequant) and other tensors as F16 (post-FP8-dequant) — produces a ~568 GB intermediate that is **not** the shippable artifact; (b) the user runs `llama-quantize` to re-quantize routed experts to IQ2_XXS (or Q2_K / IQ2_XS — antirez's Recipe), other tensors to Q8_0, producing an ~80 GB final GGUF. Document the exact `llama-quantize` invocation and target sizes in `docs/plans/v4-quant-recipe.md`. **Out:** native FP4 GGUF type.

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

This spec was scaffolded by the v4-roadmap builder phase from row `v4-p3-quant-pipeline` of `docs/plans/v4-roadmap.md`. Re-read the roadmap before you start — the surrounding context (Metal gap analysis in §2, V4 delta in §3, risks in §5) is load-bearing.
