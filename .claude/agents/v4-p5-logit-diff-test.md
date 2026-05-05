# v4-p5-logit-diff-test: `tests/test-deepseek4-logits.cpp` against the reference blob

**Phase:** 5 — Validation against reference outputs
**Complexity:** M
**Branch:** `feat/v4-p5-logit-diff-test`
**Dependencies:** `v4-p5-reference-logits`, `v4-p3-hca-attention`, `v4-p3-test-backend-ops`

## Scope

New test that loads a converted V4 GGUF, runs the same plain-text prompt as `v4-p5-reference-logits`, compares logits to within `1e-2` RMS. Skipped if the GGUF is absent. Requires the full V4 forward path to be working end-to-end (HCA + CSA + mHC), which means depending on the entire Phase-3 forward stack — not just the test-ops landing. Does **not** depend on tool-calling because the reference prompt does not exercise tool-call framing.

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

This spec was scaffolded by the v4-roadmap builder phase from row `v4-p5-logit-diff-test` of `docs/plans/v4-roadmap.md`. Re-read the roadmap before you start — the surrounding context (Metal gap analysis in §2, V4 delta in §3, risks in §5) is load-bearing.
