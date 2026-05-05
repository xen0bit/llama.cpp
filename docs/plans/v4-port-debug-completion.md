# V4 Port — Debug Completion

Follow-up to `docs/plans/v4-port-debug-handoff.md`.

## TL;DR

The chat-completion bug in the V4 port had nothing to do with the chat
template, the reasoning-budget hack, YaRN scaling, the tokenizer, or MoE/sparse
attention routing (handoff hypotheses H1–H5). It was caused by a single
launcher flag pair: **`--cache-type-k q8_0 --cache-type-v q8_0`**, which
corrupts V4's compressed/indexer KV cache and produces degenerate output
("=" loops, "Mirror …" garbage, single-token completions).

Removing those two flags fixes inference under the same flag set that was
previously broken, with all other flags from the handoff baseline (including
`--reasoning-budget 0`, `--ctx-size 32768`, `--flash-attn on`,
`--threads-batch 32`, the V4 sampling defaults) preserved.

## How the bisection went

Server boots tested in sequence, each with the tiny prompt
`"What is 2+2?"`, max_tokens=200, temperature=0:

| Server flags | Result |
|---|---|
| Full handoff baseline (incl. q8 KV + `--reasoning-budget 0`) | ❌ 4 tokens then stop, content `" a\r"` |
| Same, no `--reasoning-budget 0` (H1) | ❌ gibberish in `reasoning_content` |
| Minimal `--jinja -ngl 999` only (gate-tools.sh-style) | ✅ proper reasoning, content `"4"` |
| Minimal + `--cache-type-k|v q8_0` only (everything else default) | ❌ `"Mirror …  …"` then stop |
| Full handoff baseline **without** `--cache-type-k|v q8_0` | ✅ coherent reasoning + answer |

The same A/B with the medium prompt (200x `"The quick brown fox jumps. "`)
reproduced the `=` loop with q8 KV and produced the coherent summary
`"The text repeatedly states that a quick brown fox jumps."` without it.

## Why the original handoff missed this

The handoff's "asymmetry that matters" table treated `gate-tools.sh` as the
"tools work" baseline and concluded that bare-chat requests were the regression.
But `gate-tools.sh` boots `llama-server` with only `--jinja -ngl 999` — it
does not pass any `--cache-type-*` flag. `gate-coherence.sh` runs
`llama-completion` (a different binary), also without cache-type flags. The
"tools work, chat doesn't" comparison was therefore not apples-to-apples on
flags; it was actually "minimal flags work, full flag set doesn't." Once the
two flag sets were diffed and the q8 KV pair removed, the comparison reverses
and chat works in both shapes.

## Hypotheses ruled out

- **H1 (reasoning-budget=0 hack)** — boot without it; bug persists. Reasoning
  content is gibberish, content is empty. Forced `</think>` is not the cause.
- **H2 (chat template malformed for non-tool requests)** — the rendered prompt
  is identical in shape between the broken and the fixed configurations
  (verified via `__verbose.prompt` in the OpenAI-compat response). Removing
  q8 KV with the template untouched fixes inference, so the template is fine.
- **H3 (YaRN scaling math)** — same `--ctx-size 32768` works once q8 KV is
  removed. The scaling threshold is not implicated.
- **H4 (tokenizer issue with special tokens)** — the user/assistant/think
  tokens tokenize the same way under both flag sets; only the KV-cache layout
  changes between them. Special tokens are not implicated.
- **H5 (MoE routing or sparse-attention port bug)** — both code paths run
  identically in fp16 and q8_0 KV cache configurations; the routing graph
  doesn't change with the cache type. Not implicated.

## Working theory for why q8 KV breaks V4

V4 stores compressed latent KV (MLA-style) and a separate lightning indexer
K/V for sparse attention. The Metal q8_0 K/V quantization paths assume the
standard per-head K/V tensor layout. Either an indexer or compressor tensor is
being routed through the q8 quant path when it shouldn't be (its head_dim or
group structure differs from the regular attention K/V), or the q8 Metal
kernels produce wrong values for V4's odd shapes.

A proper fix likely lives in the V4 KV-cache assembly (`src/llama-context.cpp`
position-dependent decode + `src/models/deepseek4.cpp` KV compressor / indexer
allocation) and either keeps the indexer/compressor tensors fp16 unconditionally
or surfaces a runtime guard. This is **deferred** as a follow-up — the immediate
fix is to drop the flag from the V4 launcher recipe.

## What landed

- `claude-cache-proxy/start-server-v4.sh` — V4-specific launcher that omits
  `--cache-type-k|v q8_0`. (Lives in the cache-proxy repo, not this fork; see
  that repo for the file.)
- `tests/v4-port/gate-server-chat.sh` — regression gate. Boots the server with
  the V4-recommended flag set and asserts coherent output for tiny + medium
  chat plus tool-call fixture. Wired into `tests/v4-port/run-all-gates.sh`.

## Validation

```text
$ V4_GGUF=...chat-v2.gguf ./tests/v4-port/run-all-gates.sh
PASS: loader recognizes V4 GGUF
Printable ratio: 100%
Decode tokens: 30
PASS: coherence (NGL=0, gen='...')
Printable ratio: 100%
Decode tokens: 30
PASS: coherence (NGL=999, gen='...')
Decode tok/s: 25.38 (min: 10)
PASS: speed (NGL=999, 25.38 tok/s)
PASS: tool calling (5/5 with tool_calls)
OK[tiny-no-tools]: 62 tokens, 217 chars, 41 unique
OK[medium-no-tools]: 13 tokens, 56 chars, 26 unique
OK[tools]: 1 tool_call(s)
PASS: server-chat (3/3 tests coherent)
ALL GATES PASS
```

## Deferred / open

- **Underlying q8 KV bug**: surface a guard or fix the V4 KV path so q8/q4
  cache types either work or fail loudly with a clear message instead of
  silently corrupting output. Tracker: not opened yet — this is a "real
  investigation" follow-up, not a quick patch.
- **Reasoning-content extraction with `--reasoning-budget 0`**: when the model
  emits thinking content followed by `</think>` and a final answer, the
  parser puts the whole thing in `content` and leaves `reasoning_content`
  null. Cosmetic for terminal-bench since the answer is still there, but
  worth tightening later.
- **terminal-bench end-to-end**: the handoff DoD asks for a single
  `largest-eigenval*` run with `reward: 1.0`. That validation step is
  **not yet executed** in this debug pass; it requires booting cache-proxy +
  the new V4 server and running the blobfish harness. Recommended as the
  next step once the user has approved the new launcher script.
