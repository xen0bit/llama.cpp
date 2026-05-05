# DeepSeek V4 Overnight Port — Design

**Date:** 2026-05-04
**Author:** Brainstorm session between user (cchuter) and Claude (Opus 4.7, 1M ctx)
**Working branch:** `feat/v4-port` (off `deepseek-dsa`)
**Goal:** By tomorrow morning, our llama.cpp build can load antirez's DeepSeek V4 GGUF, produce coherent tokens at >10 tok/s on Metal, and serve tool calls via `llama-server` returning HTTP 200 (not 500).

## 1. Branch topology & port strategy

```
ggml-org/llama.cpp:master              ─────●───────────●──────────●─── (upstream)
                                              │           │
fairydreaming:deepseek-dsa            ────────┴───●──●──● (V3.2 + DSA, our base)
                                                       │
                                                       ▼
                                                 feat/v4-port  ◄── overnight target
                                                       ▲
                                                       │ port commits from
antirez:main (V4 fork)               ────────────●──●──●  (V4 architecture, kernels, chat template)
```

`feat/v4-port` is a single integration branch off `deepseek-dsa`. We port antirez's V4 changes onto it via `git format-patch` + `git am --3way`, resolving conflicts component-by-component. Final layering: `master` + V3.2/DSA (fairydreaming) + V4 (antirez).

**Conflict resolution rule** (when antirez and fairydreaming both touch a file):

- Sparse attention / lightning indexer / DSA → **keep fairydreaming's** (upstream-preferred)
- V4 architecture / V4 routing / V4 expert handling / V4 chat template → **keep antirez's** (only V4 implementation)
- Shared infrastructure (arch registry, conversion script tensor maps, op dispatch tables) → **merge** — keep fairydreaming's V3.2 entries, add antirez's V4 entries alongside

**Working units:** Each parallel work stream gets its own git worktree off `feat/v4-port`. Builder sub-agents work in worktrees; finished work merges back into `feat/v4-port` for codex review and integration testing.

**Output by morning:**

- `feat/v4-port` branch on `mine` remote
- Builds clean: `cmake --build build -j` exit 0 with both `GGML_METAL=ON` and CPU
- Loads antirez's GGUF: `llama-cli -m DeepSeek-V4-Flash-IQ2XXS-...gguf -p "hi" -n 5` produces 5 tokens, no NaN
- Tool-calling smoke: 5 successive curl POSTs to `/v1/chat/completions` all return HTTP 200 with `tool_calls` array

## 2. Phase breakdown & parallel dispatch

Five work streams. A → D is critical path; E supports the gates.

```
                  ┌──────────────────────────┐
                  │ Stream A — V4 loader     │
                  │ - LLM_ARCH_DEEPSEEK4     │
                  │ - tensor name maps       │
                  │ - KV cache config        │
                  │ - convert_hf_to_gguf.py  │
                  │ blocks: B, C, D          │
                  └──────────┬───────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
   ┌──────────▼──────┐ ┌────▼─────────┐ ┌──▼──────────────┐
   │ Stream B        │ │ Stream C     │ │ Stream E        │
   │ V4 forward pass │ │ V4 Metal     │ │ Validation      │
   │ (CPU)           │ │ kernels      │ │ harness         │
   │ blocks: D       │ │              │ │                 │
   └──────────┬──────┘ └──────┬───────┘ └─────────────────┘
              │               │
              └────────┬──────┘
                       │
              ┌────────▼─────────────┐
              │ Stream D — Tool calling│
              │ - chat template       │
              │ - --jinja wiring      │
              │ - cache-proxy tweaks  │
              └──────────────────────┘
```

| ID | Title | Worktree | Depends on | Est |
|---|---|---|---|---|
| `v4-port-A-loader` | V4 architecture + loader + conversion | `wt-A-loader` | — | 1-2h |
| `v4-port-B-forward` | V4 forward pass on CPU | `wt-B-forward` | A | 2-3h |
| `v4-port-C-metal` | V4 Metal kernels | `wt-C-metal` | A | 2-3h |
| `v4-port-D-tools` | Chat template + tool calling | `wt-D-tools` | A, B | 1-2h |
| `v4-port-E-validate` | Validation harness | `wt-E-validate` | (parallel from start) | 1h |

**Critical path:** A → B → D → integration gate (≈ 5-7h).
**Parallelism:** Once A merges, B + C + E fire in parallel. D fires after B + C are integrated.

### Three explicit scope decisions

1. **DSA Metal kernels are punted.** Fairydreaming hasn't done Metal for V3.2's sparse-attention ops. We do not do them tonight. Our build won't run V3.2 models on Metal — only V4. Acceptable since V4 is the goal.
2. **CUDA is ignored.** No NVIDIA GPU on this machine. CPU + Metal only.
3. **Quantization scheme is "trust antirez."** We don't re-derive his IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8 mixed quant. We port enough dequant code that his GGUF loads. Re-quantizing from official safetensors is a separate task.

### Iteration budget per stream

- **Up to 8 codex-review rounds** before marking `needs-human`
- **Up to 8 builder fix rounds** between revisions
- 16 iterations max per stream — encourages persistence

## 3. Validation & gates

Loose-parity bar — narrow mechanical gates per stream, no token-level diff against antirez's binary.

| Stream | Gate (must pass to merge into `feat/v4-port`) |
|---|---|
| A — loader | Build clean. `llama-cli --no-warmup -p "" -n 0` exits 0, prints metadata. |
| B — CPU forward | Generate 30 tokens for `"The capital of France is"`. Output must (i) contain printable ASCII text (>80% printable chars), (ii) NOT be a single repeated token, (iii) no NaN. |
| C — Metal kernels | Same prompt + coherence check on Metal (`-ngl 999`). PLUS speed gate: time a 50-token decode after warmup, must be **> 10 tok/s**. |
| D — tool calling | Boot `llama-server --jinja`. Run 5 successive POSTs to `/v1/chat/completions` using fixture `tests/v4-port/tool-call-fixture.json` (single `get_weather` tool, prompt: "What is the weather in Paris?"). **All 5 must return HTTP 200.** Response body must contain a `tool_calls` array (any args). No 500s. |
| E — harness | Implements the above checks as a callable script. |

### Integration gate (`feat/v4-port` is "done")

1. All 4 critical-path stream gates pass on the integrated branch
2. Build clean with both CPU and Metal
3. End-to-end smoke: load GGUF, run prompt, get >10 tok/s coherent output, then tool-call 5x at HTTP 200
4. Pushed to `mine`
5. Completion report at `docs/plans/v4-port-completion.md`

### Three validation requirements baked in (user's overnight asks)

1. **Token coherence** — Stream B and C gates check this (printable ASCII, not single repeated token)
2. **Tool-calling 200s not 500s** — Stream D gate, 5×200 required
3. **Metal decode > 10 tok/s** — Stream C gate

### Still in "manual tomorrow" punt list

- Token quality vs antirez's fork (we check coherence, not parity)
- Tool-argument quality (we check status code, not whether args are sensible)
- Long-context behavior beyond ~50 tokens
- V3.2 still works in our build (V3.2 may break; not validated tonight)

### Failure handling

If a stream's gate fails after 8+8 rounds → marked `needs-human`. Downstream-dependent streams skip if they depend on it; otherwise continue.

**Stream A failure is the only true blocker.** If the loader fails, orchestrator stops, writes diagnostic to `docs/plans/v4-port-blocked.md`, posts a clear `BLOCKED: Stream A failed` summary, exits.

For all other failures, orchestrator continues with surviving streams. Partial work is more valuable than no work.

## 4. Tool calling specifics

Highest-uncertainty section. We don't yet know V4's chat-template format.

### End-to-end flow

```
terminal-bench harness
       │ Anthropic-format /v1/messages
       │ (assistant tool_use, user tool_result)
       ▼
cache-proxy.py (port 8081)
       │ rewrite reminders, billing headers, etc.
       ▼
llama-server --jinja (port 8080, our build)
       │ apply chat template embedded in GGUF
       │ render: V4 chat format with tool blocks
       ▼
V4 model (forward pass)
       │ generate tokens including tool-call markup
       ▼
llama-server tool-call parser
       │ extract tool_calls array from raw output
       ▼
response: { "tool_calls": [{...}] }
```

### Discovery step (Stream A's first action — before any code is written)

```bash
~/work/llama.cpp/build/bin/llama-gguf-info \
  ~/models/DeepSeek-V4-Flash-IQ2XXS-...gguf
```

Dump `tokenizer.chat_template` field. Document exact format in working-notes file. We mirror that template in our build.

If antirez didn't embed a Jinja template, fall back to extracting it from antirez fork's source.

### Stream D sub-steps

| ID | Action |
|---|---|
| D.1 | Inspect antirez's GGUF chat template. Document format. |
| D.2 | Confirm `--jinja` correctly applies it (smoke test in `--verbose` mode). |
| D.3 | Add tool-call parsing rules — likely regex/grammar in `common/json-tool-call.cpp`. May need to port antirez's custom parser. |
| D.4 | Wire `cache-proxy.py` for V4-specific quirks. Existing `--rewrite-reminders` may suffice; V4 may handle reminders natively. |
| D.5 | Run 5x curl gate. |

### Tool-call parser contingency

If antirez's parser doesn't port cleanly:

- Use `llama-server` raw-text mode (no tool-call extraction)
- Have `cache-proxy.py` do parsing server-side (extract tool-call markup from raw response, format as Anthropic tool_use)

Keeps Stream D's pass criterion achievable even if in-server parser is broken.

### Specific risks tonight

1. **No embedded chat template in antirez's GGUF.** Mitigation: pull from antirez fork source. Adds ~30 min to Stream A.
2. **V4's tool-call format requires special tokens our tokenizer doesn't recognize.** Fixable via `add_special_tokens` config, but fiddly. Mitigation: if hit, accept HTTP 200 + raw text in body for tonight's gate, defer parser to tomorrow.

## 5. Risks, abort conditions, and completion behavior

### Top risks (with mitigations)

| Risk | Mitigation |
|---|---|
| GGUF not done downloading when Stream A needs it | Stream A's first action checks for the file. **For chat template discovery (Section 4 D.1), use antirez fork source as an alternative** — does not require the GGUF. For runtime gates (B/C/D), poll the file every 5 min; if absent after 90 min, mark `needs-human` for runtime-dependent streams but keep loader code work moving. |
| Antirez's fork point is far from ours — patches don't apply cleanly | Builder agents use `git format-patch` + `git am --3way`. On 3-way fail, fall back to manual port. 8-round retry budget absorbs this. |
| Disk space — 80GB GGUF + multiple build artifacts + worktrees | Pre-flight: ensure ≥150GB free. Periodically `du -sh build*`; clean stale worktrees if needed. |
| Codex rate-limit / outage | Per orchestrator skill: retry once, then auto-approve with warning. |
| Antirez used CUDA-only ops with no Metal analog | Stream C marks those `needs-human`; B (CPU forward) still ships. CPU-only build available, Metal patched in via follow-on. |
| Tool-call parser too hard | Fall back to parsing in cache-proxy (Section 4 contingency). |

### Stop conditions

Orchestrator exits when **any** of these is true:

1. All 4 critical-path streams pass gates AND tool-calling smoke passes (success)
2. Stream A failed (true blocker)
3. Wall clock has exceeded 10 hours since dispatch (safety cap)
4. User Ctrl-C interrupts via the harness

### Completion report (`docs/plans/v4-port-completion.md`)

When orchestrator reaches a terminal state:

```markdown
# V4 Port — Overnight Run Completion Report

## What shipped
- Stream A (loader): ✓ / × (commit + gate evidence)
- Stream B (CPU forward): ✓ / × / NEEDS-HUMAN
- Stream C (Metal): ✓ / × / NEEDS-HUMAN
- Stream D (tool calling): ✓ / × / NEEDS-HUMAN
- Stream E (validation harness): ✓ / × / NEEDS-HUMAN

## Integration gate results
- Build clean: ✓/×
- Coherence test: ✓/× (sample output)
- 5×200 tool calls: ✓/× (timings)
- Metal decode tok/s: <number>

## What needs human attention tomorrow
- <list each NEEDS-HUMAN stream with last error and pointer to last commit>

## Repro
~/work/llama.cpp/build/bin/llama-server -m ~/models/.../DeepSeek-V4-Flash-...gguf --jinja
curl -s localhost:8080/v1/chat/completions -d @<test-payload>.json
```

### What the orchestrator must NOT do

- Push to `origin` (ggml-org/llama.cpp upstream) — this is the line that alerts maintainers
- Run `gh pr create` against any repo
- Touch `deepseek-dsa` or `feat/v4-roadmap` branches (read-only references)
- Run destructive git operations on uncommitted work
- `rm -rf` anything outside `build/` or worktree dirs

The orchestrator MAY push to `mine` (cchuter/llama.cpp) freely, including pushing/merging to `mine/master`. The boundary is `origin` (upstream), not `master` (which is just a branch on our fork).

### User authorization (granted during brainstorm)

- May push any branch (including `feat/v4-port`, worktree branches, and `master`) to `mine`
- May merge `feat/v4-port` into `mine/master` if streams complete cleanly
- May NOT push to `origin` (ggml-org/llama.cpp) — that's the line that surfaces work to upstream maintainers
- May NOT run `gh pr create` against any repo
- May iterate up to 16 rounds (8 codex + 8 builder) per stream
- "Don't block unless we have to"

## Goal restated

By tomorrow morning, ssh into ~/work/llama.cpp:

```bash
cd ~/work/llama.cpp
./build/bin/llama-server \
  -m ~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --jinja --port 8080 -ngl 999

# In another terminal:
curl -s localhost:8080/v1/chat/completions -d '{
  "model": "v4-flash",
  "messages": [{"role":"user","content":"What is the weather in Paris?"}],
  "tools": [{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
}'
# → HTTP 200, response contains tool_calls array
```

And see >10 tok/s decode on Metal. That's the bar.
