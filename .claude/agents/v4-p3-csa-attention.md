# v4-p3-csa-attention: Compressed Sparse Attention (V3.2-DSA-equivalent)

**Phase:** 3 — V4 forward pass (CPU + Metal)
**Complexity:** M
**Branch:** `feat/v4-p3-csa-attention`
**Dependencies:** `v4-p3-hc-residual`, `v4-p1-metal-lightning-indexer-quant`, `v4-p1-metal-fattn-top-k`, `v4-p1-test-v32-end-to-end`

## Scope

Reuse `llama_kv_cache_dsa` from V3.2 + Phase-1 Metal kernels. Wire indexer Q/K to V4's tensor names. **Out:** HCA path.

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

This spec was scaffolded by the v4-roadmap builder phase from row `v4-p3-csa-attention` of `docs/plans/v4-roadmap.md`. Re-read the roadmap before you start — the surrounding context (Metal gap analysis in §2, V4 delta in §3, risks in §5) is load-bearing.
