# v4-p3-output-routing: V4 output projection (LoRA / grouped-output) and hash-routing

**Phase:** 3 — V4 forward pass (CPU + Metal)
**Complexity:** M
**Branch:** `feat/v4-p3-output-routing`
**Dependencies:** `v4-p3-hc-residual`

## Scope

Wire the hypothesised `attention.output_lora_rank`, `attention.output_group_count`, and `hash.layer_count` keys (added in `v4-p2-tensor-ids`) into the runtime forward path: split the output projection into LoRA-A/LoRA-B mat-muls when `output_lora_rank > 0`, and gate the per-layer hash routing on `hash.layer_count`. Mirror the external antirez fork's handling. **Out:** non-router parts of FFN. If the official V4 release proves these keys are stale-from-antirez, the task is closed as no-op (its existence ensures we don't ship Phase-2 IDs that nothing reads).

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

This spec was scaffolded by the v4-roadmap builder phase from row `v4-p3-output-routing` of `docs/plans/v4-roadmap.md`. Re-read the roadmap before you start — the surrounding context (Metal gap analysis in §2, V4 delta in §3, risks in §5) is load-bearing.
