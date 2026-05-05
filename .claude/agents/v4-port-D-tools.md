# v4-port-D-tools: V4 chat template + tool calling

## Goal
Cherry-pick antirez commit `3ba61fbb4` (Add DeepSeek V4 tool-call chat template). Validate that `llama-server --jinja` accepts tool-call requests and returns HTTP 200 with `tool_calls` arrays.

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-04-v4-port-overnight.md` Task 5 (Steps 5.1 through 5.4).** Follow it exactly.

## Depends on
v4-port-C-metal must be in state `done` before this task starts (gate-tools.sh uses Metal `-ngl 999`).

## Gate (must pass before code-review)
```bash
V4_GGUF=/Users/cchuter/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  ./tests/v4-port/gate-tools.sh
```
Expected: `PASS: tool calling (5/5 with tool_calls)`. The gate runs 5 successive POSTs to /v1/chat/completions with the get_weather tool fixture; all must return HTTP 200 with a `tool_calls` array in the body.

## Files (touched by 3ba61fbb4)
- Modify: `common/chat.cpp` (+12 -5)
- New: `models/templates/deepseek-ai-DeepSeek-V4.jinja` (+96)

## Failure mode + recovery (Section 4 of spec)
If 500s instead of 200s: the chat template is rejecting our message format (same class of problem we hit with Mistral). Most likely: tool_use/tool_result mapping doesn't match V4's expected format.

**Contingency:** if the in-server tool-call parser is too hard to fix, fall back to:
- Use llama-server raw-text mode (no tool-call extraction in server)
- Have cache-proxy.py do parsing server-side (extract tool-call markup from raw response, format as Anthropic tool_use)

This keeps the gate achievable. Document the contingency in the commit message if used.

## Iteration budget
- Up to 8 builder fix rounds
- Up to 8 codex review rounds

## Definition of done
- Cherry-pick succeeded
- Build clean
- gate-tools.sh PASSES (5/5 at HTTP 200 with tool_calls)
- Codex code-review APPROVED
- Pushed to `mine`
