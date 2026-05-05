# v4-p5-reference-logits: Capture reference logits from a known-good runtime

**Phase:** 5 — Validation against reference outputs
**Complexity:** S
**Branch:** `feat/v4-p5-reference-logits`
**Dependencies:** `v4-p4-chat-template`

## Scope

Two-tier strategy because running 284B V4-Flash via HF `transformers` on FP16 CPU is not feasible on the M3 Ultra (would need ≈568 GB host RAM). **Tier A (preferred):** query the official DeepSeek V4 inference API for top-K logits / log-probs on a fixed plain-text prompt (no tool-calls; the API exposes log-probs in its OpenAI-compatible endpoint). Save the per-token top-K to a JSON fixture. **Tier B (fallback):** run a single-layer V4 slice via HF `transformers` on the same M3 Ultra (single-layer ≈ 600 MB FP16) and compare against our converted single-layer GGUF — covers correctness of the per-layer math even without a full-model reference. The reference prompt deliberately exercises only the chat-template path, **not** tool-calling, so this task depends on the chat template (so the encoded tokens are stable) but **not** on the V4 forward path (this is data collection from a *different* runtime). **Out:** any code in our tree.

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

This spec was scaffolded by the v4-roadmap builder phase from row `v4-p5-reference-logits` of `docs/plans/v4-roadmap.md`. Re-read the roadmap before you start — the surrounding context (Metal gap analysis in §2, V4 delta in §3, risks in §5) is load-bearing.
