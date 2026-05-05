# v4-p1-metal-fattn-top-k: Top-K / sparse K-V indexing in Metal flash-attn (vec **and** non-vec)

**Phase:** 1 — Metal kernels for V3.2 (gets us a Metal-capable V3.2 first)
**Complexity:** L
**Branch:** `feat/v4-p1-metal-fattn-top-k`
**Dependencies:** none

## Scope

Extend **both** the vec and non-vec paths of Metal flash-attn. V3.2's graph (`src/models/deepseek32.cpp:224, 358, 437`; `src/llama-graph.cpp:2428`) attaches `top_k` to every `ggml_flash_attn_ext` call unconditionally, so prompt processing (which routes to non-vec when `ne01 ≥ 20` per `ggml-metal-ops.cpp:2507-2515`) is *broken* without non-vec coverage. Work: (1) add an optional `top_k` buffer to `ggml_metal_kargs_flash_attn_ext` and `ggml_metal_kargs_flash_attn_ext_vec` kernel-args structs; (2) plumb it from the dispatcher in `ggml_metal_op_flash_attn_ext` (`ggml-metal-ops.cpp:2631-3045`) when `op->src[5] != NULL`; (3) replace linear `K/V + i*stride` with `K/V + top_k[i]*stride` in `kernel_flash_attn_ext_impl` and `kernel_flash_attn_ext_vec` (`ggml-metal.metal:5801-…`); (4) add a `src[5] != NULL` clause to the flash-attn entry of `ggml_metal_device_supports_op` (`ggml-metal-device.m:1147-1186`). Mirrors `ggml/src/ggml-cuda/fattn-mma-f16.cuh:382-405` semantically. Test against `tests/test-llama-archs.cpp` for the `LLM_ARCH_DEEPSEEK32` arch on Metal. **Out:** changing op-args layout for non-DSA flash-attn callers.

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

This spec was scaffolded by the v4-roadmap builder phase from row `v4-p1-metal-fattn-top-k` of `docs/plans/v4-roadmap.md`. Re-read the roadmap before you start — the surrounding context (Metal gap analysis in §2, V4 delta in §3, risks in §5) is load-bearing.
