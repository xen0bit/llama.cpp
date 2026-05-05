# v4-p4-chat-template: Add `LLM_CHAT_TEMPLATE_DEEPSEEK_4`

**Phase:** 4 — Chat template / tool-calling
**Complexity:** M
**Branch:** `feat/v4-p4-chat-template`
**Dependencies:** `v4-p2-conversion-script`

## Scope

New entry in `src/llama-chat.h` (enum at line 28-32) and `src/llama-chat.cpp` (string mapping at line 49-52, plus matching block in `llm_chat_template_from_str` and `llm_chat_apply_template_impl`). Reproduces `encoding_dsv4.encode_messages()` semantics (system / user / assistant / `reasoning_content` interleaving). Add a fixture in `tests/test-chat.cpp` that compares a fixed message list against tokens produced by HF's `transformers.AutoTokenizer.encode(encoding_dsv4.encode_messages(...))`. **Independent of forward-pass and indexer test work** — chat-template plumbing only needs the V3.2-style tokenizer registration that ships with V3.2 (`_set_vocab_gpt2()`).

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

This spec was scaffolded by the v4-roadmap builder phase from row `v4-p4-chat-template` of `docs/plans/v4-roadmap.md`. Re-read the roadmap before you start — the surrounding context (Metal gap analysis in §2, V4 delta in §3, risks in §5) is load-bearing.
