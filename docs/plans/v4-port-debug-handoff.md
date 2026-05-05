# V4 Port — Debug Handoff

**For the next Claude Code session.** Built on `feat/v4-port` overnight; works on raw-completion test, broken via `/v1/chat/completions`. This document hands you everything you need to diagnose and fix it.

---

## TL;DR

V4 architecture port is structurally complete but **chat-template-wrapped inference is broken**. Raw completion (via `llama-completion -no-cnv`) produces coherent text. Chat-completion via the OpenAI-compat `/v1/chat/completions` endpoint produces garbage tokens or single-character loops, even for trivial prompts.

Your job: find why the chat-template path corrupts inference, fix it, and validate end-to-end.

## Where to start (5 minutes)

1. `cd /Users/cchuter/work/llama.cpp && git log --oneline -5` — should show HEAD at `443e03d95`
2. Read `docs/superpowers/specs/2026-05-04-v4-port-overnight-design.md` (the spec) — 5 min skim
3. Read `tasks/done/v4-port-*.json` history arrays — chronological view of what each task did
4. Read this file's "Bug evidence" section below
5. Read the failure-mode hypotheses, pick the most likely one, start there

---

## Concrete evidence of the bug

### What works

```bash
# Raw completion — produces coherent text
V4_GGUF=/Users/cchuter/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
NGL=0   ./tests/v4-port/gate-coherence.sh   # PASS — "Paris. It is a beautiful city..."
NGL=999 ./tests/v4-port/gate-coherence.sh   # PASS — "Paris. The capital of Italy is Rome..."
./tests/v4-port/gate-speed.sh               # PASS — 25.91 tok/s on Metal

# Chat-format with tool defs — passes too (200 status, tool_calls returned)
./tests/v4-port/gate-tools.sh               # PASS — 5/5 tool calls
```

### What's broken

```bash
# Boot the server first:
./build/bin/llama-server \
  --host 0.0.0.0 --port 8080 \
  -m /Users/cchuter/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --jinja --reasoning-budget 0 \
  --ctx-size 32768 -ngl 999 --parallel 1 --flash-attn on \
  --threads-batch 32 --cache-type-k q8_0 --cache-type-v q8_0 \
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05 \
  --metrics --verbose

# Tiny prompt, no tools — produces incoherent multilingual garbage
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"v4","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50,"temperature":0}' \
  | jq -r '.choices[0].message.content'
# Output observed:
#  this---... 4 0 ,在旁边(;"); \td домаћинствима-1

# Medium prompt (5400 tokens of repeated "The quick brown fox jumps. ") — degenerate `=` loop
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d "$(python3 -c 'import json; print(json.dumps({"model":"v4","messages":[{"role":"user","content":"Summarize this in one sentence: " + ("The quick brown fox jumps. " * 200)}],"max_tokens":50,"temperature":0}))')" \
  | jq -r '.choices[0].message.content'
# Output observed:
# =================================================
```

### The asymmetry that matters

| Path | Result |
|---|---|
| `llama-completion -no-cnv -p "The capital of France is" -n 30` | ✅ coherent |
| `llama-server` + `/v1/chat/completions` + tool definition + small prompt | ✅ coherent (5/5 in gate-tools.sh) |
| `llama-server` + `/v1/chat/completions` + NO tools + small prompt | ❌ multilingual garbage |
| `llama-server` + `/v1/chat/completions` + NO tools + medium prompt | ❌ `=`-loop |
| `llama-server` + `/v1/chat/completions` + Claude Code system prompt (~30k tokens) | ❌ `=`-loop (overnight observation) |

This pattern says: **the chat template is doing something to the prompt that breaks inference, and it's not just prompt size.** Tool definitions seem to "unlock" working behavior. That's a clue.

---

## Project context (skim if needed)

### What was built overnight

We ported antirez's DeepSeek V4 Flash inference (his fork at `antirez/llama.cpp-deepseek-v4-flash`) onto fairydreaming's V3.2/DSA branch (`fairydreaming/llama.cpp:deepseek-dsa`, upstream PR #21149). Antirez's commits predate an upstream refactor (`994118a18`, May 4) that moved per-arch logic from monolithic switches in `src/llama-model.cpp` to per-arch classes in `src/models/<arch>.cpp`. So we did a structural translation, not a clean cherry-pick.

### Branch state

- Branch: `feat/v4-port` on remote `mine` (cchuter/llama.cpp), HEAD at `443e03d95`
- 28 commits since base
- Pushed to `mine` (NOT to `origin` — origin is ggml-org/llama.cpp upstream, do not push there)
- **Master merge deferred** — upstream advanced with a walsh-hadamard PR (`a817a22bc`) that conflicts with our `GGML_OP_COUNT=102` bump in `ggml/src/ggml-cpu/ops.cpp`. Resolve before pushing to `mine/master`.

### Files you'll be reading

- `src/models/deepseek4.cpp` (1500+ lines) — V4 model class, includes `load_arch_hparams`, `load_arch_tensors`, `build_arch_graph`, plus the `llama_model_deepseek4::graph::graph` constructor (the actual forward pass — 1300+ lines)
- `src/models/models.h` — `llama_model_deepseek4` class declaration (after `llama_model_deepseek32`)
- `src/llama-arch.cpp/h` — `LLM_ARCH_DEEPSEEK4` enum + tensor name maps
- `src/llama-memory-hybrid-iswa.cpp/h` — V4 hybrid KV cache (heavily ported)
- `src/llama-context.cpp` — V4-specific graph_max_nodes, prompt-replay speedup
- `models/templates/deepseek-ai-DeepSeek-V4.jinja` — V4 chat template (96 lines, has the Anthropic mixed-content fix)
- `common/chat.cpp` — chat-format detection, registers V4
- `ggml/src/ggml-metal/ggml-metal.metal` — V4 Metal kernels (HC sinkhorn etc.)

### Reference materials

- `docs/superpowers/specs/2026-05-04-v4-port-overnight-design.md` — the spec
- `docs/superpowers/plans/2026-05-04-v4-port-overnight.md` — the implementation plan
- `docs/plans/v4-port-completion.md` — the completion report
- Antirez's reference: `git log antirez/main` (remote `antirez` is configured) — particularly `06c504247` (the big V4 commit) and `3ba61fbb4` (chat template)

---

## Hypotheses, ordered by likelihood

### H1 (most likely): Reasoning-budget=0 hack is corrupting state

**Reasoning:** The chat template emits `<｜Assistant｜><think>` as the generation prompt. With `--reasoning-budget 0`, the server forcibly injects `</think>` as the first generated token. We saw in logs:

```
common_sampler_init: prefill token: 128821 = <think>
common_sampler_init: reasoning-budget accepted prefill token (128804)
reasoning-budget: budget=0, forcing immediately
reasoning-budget: forced sequence complete, done
slot process_toke: n_decoded = 1, next token: 128822 '</think>'   <-- forced
slot process_toke: n_decoded = 2, next token: 31 '='              <-- start of garbage
```

The forced `</think>` may be desyncing the KV cache or the sampler's internal state. The model wasn't trained on `<think></think>` (empty thinking) — it expects content between them.

**Test:**

```bash
# Restart server WITHOUT --reasoning-budget 0
./build/bin/llama-server -m $V4_GGUF --jinja \
  --ctx-size 32768 -ngl 999 ... # all other flags same as before, but no --reasoning-budget

# Same curl test
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"v4","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50,"temperature":0}'
```

If the model now produces coherent text (with thinking content followed by an answer), **H1 is confirmed.** Then the fix is one of:

- Live with thinking enabled, accept slower terminal-bench iterations
- Use `--reasoning-budget N` with N > 0 (let it think briefly)
- Patch the V4 chat template to skip the `<think>` start token entirely when reasoning is off (instead of relying on the forced-end hack)

The clean fix is the third — modify `models/templates/deepseek-ai-DeepSeek-V4.jinja` to conditionally emit `<think>` based on a `thinking` variable, similar to how it already handles `reasoning_content`.

### H2: Chat template malformed for non-tool requests

**Reasoning:** The asymmetry — tool requests work (5/5 in gate-tools.sh), bare requests don't — points at the template branching for tool definitions. Maybe the template emits something necessary for "tool mode" that's missing in "chat mode" with no tools.

**Test:**

```bash
# Print rendered prompt for a no-tools request via the server's verbose log
# (Server is started with --verbose; check /tmp/llama-server-tools.log or your tee'd log)

# Compare to a request with tools — both via same server
curl -s localhost:8080/v1/chat/completions -d @tests/v4-port/tool-call-fixture.json
curl -s localhost:8080/v1/chat/completions -d '{"model":"v4","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'

# Look in the verbose server log for the fully-rendered prompt strings.
# Compare them character-by-character.
```

If they differ in unexpected ways (missing tokens, weird whitespace, doubled markers), the template is the bug.

**Look at the template directly:**

```bash
cat models/templates/deepseek-ai-DeepSeek-V4.jinja
```

Pay attention to:

- `<｜begin▁of▁sentence｜>` placement (these are Unicode separators, not ASCII)
- The `tools` parameter handling — what does the template do when `tools is undefined`?
- System message rendering — does it always emit `<｜System｜>` or only if there's a system message?

### H3: YaRN scaling math is wrong

**Reasoning:** V4 uses YaRN positional encoding (`rope.scaling.type=yarn`, factor 16, original 65536, freq_base 10000). Our `--ctx-size 32768` is well below `original_yarn=65536`, so YaRN scaling shouldn't activate. But:

- Maybe our YaRN check is `>=` when it should be `>`, activating scaling at exactly the boundary
- Maybe scaling is being applied even when it shouldn't be
- Maybe the position offsets emitted during chat-template rendering bypass the threshold check

**Test:**

```bash
# Try with --ctx-size 65536 to match original_yarn (no scaling needed there)
./build/bin/llama-server -m $V4_GGUF --ctx-size 65536 ... # other flags same

# Try with --ctx-size 4096 (way below) — does that change behavior?
```

If small ctx works and 32768 doesn't, YaRN is suspect.

**Look at:** `src/llama-context.cpp` and `src/models/deepseek4.cpp` for YaRN-related code paths. Compare against `src/models/deepseek32.cpp` which uses similar YaRN config.

### H4: Tokenizer issue with special tokens

**Reasoning:** V4's special tokens use Unicode separators (`<｜User｜>` not `<|User|>`). If these tokenize to wrong IDs, the model sees garbage at prompt level.

**Test:**

```bash
# Use llama-completion to inspect tokenization
./build/bin/llama-completion -m $V4_GGUF --prompt "<｜User｜>Hi<｜Assistant｜>" --verbose -n 0 2>&1 | grep -A3 "tokens"
```

Compare token IDs to the expected values from antirez's GGUF metadata. Look for:

- `<｜begin▁of▁sentence｜>` → should be a single token
- `<｜User｜>`, `<｜Assistant｜>`, `<｜System｜>` → should each be single tokens
- `<think>`, `</think>` → should be single tokens (we saw 128821, 128822 in logs)

If any of these are tokenizing as multiple sub-tokens, the prompt is corrupt before the model even sees it.

### H5: MoE routing or sparse-attention port bug

**Reasoning:** Less likely given that gate-coherence (raw completion) works. But the difference between "raw input" and "chat-formatted input" might exercise different routing patterns. V4 has 256 experts × 6 active; lightning indexer for sparse attention. Both areas are heavily ported.

**Test:** Compare logits between our build and antirez's fork on identical input. This requires building antirez's fork — non-trivial. Skip unless H1-H4 all fail.

---

## Debug playbook

### Phase 1: Reproduce + bisect (30-60 min)

1. Start server with the exact flags from "Concrete evidence" section above
2. Run all four curl tests in sequence (tiny+notools, tiny+tools, medium, large)
3. For each, capture the verbose server log output
4. Compare rendered prompts (in the verbose log) — find what differs between working and broken cases
5. **Test H1 first** — restart without `--reasoning-budget 0` and re-run

If H1 fixes it, jump to "Phase 3: clean fix."

If H1 doesn't fix it, continue.

### Phase 2: Drill into the template (1-2 hours)

If H2 looks promising:

1. Read `models/templates/deepseek-ai-DeepSeek-V4.jinja` line by line
2. Read antirez's original at `git show antirez/main:models/templates/deepseek-ai-DeepSeek-V4.jinja` (the reference)
3. Diff them. Did our Anthropic-mixed-content fix introduce a regression for the no-tools path?
4. Look at `common/chat.cpp` `common_chat_format_detect` and the V4 registration — does our build correctly route V4 to the right format?
5. Print the rendered prompt: in the server's `--verbose` log, look for lines starting with `slot launch_slot_:` — they include the formatted prompt string

If you find the template bug, fix it, rebuild, re-test all four curl cases.

### Phase 3: clean fix + validation

Once you have a fix:

1. **Add a regression test.** Write a new gate `tests/v4-port/gate-server-chat.sh` that:
   - Boots llama-server (similar to gate-tools.sh)
   - Sends the four curl tests above
   - Asserts each returns coherent text (not garbage, not single-char loops, not multilingual fragments)
2. Add it to `tests/v4-port/run-all-gates.sh`
3. Commit with a descriptive message. Don't amend existing commits.
4. Push to `mine`: `git push mine feat/v4-port`
5. **Do NOT push to origin.** The boundary is `origin` (ggml-org), not `master`.

### Phase 4: terminal-bench validation

Once chat-format inference works on small prompts, validate end-to-end:

```bash
cd /Users/cchuter/work/blobfish

# Start cache-proxy (separate terminal):
cd /Users/cchuter/work/claude-cache-proxy && ./start-proxy.sh

# Run a single terminal-bench task:
ANTHROPIC_BASE_URL=http://localhost:8081 ANTHROPIC_API_KEY=no-key \
./scripts/run-terminal-bench.sh \
  --backend claude \
  --model deepseek/deepseek-v4-flash \
  -k 1 -n 1 \
  -t "largest-eigenval*"
```

`largest-eigenval` is in the project's "stable passes" set per blobfish memory — known to work on functioning models.

If task succeeds (`reward: 1.0`), V4 is ready for wider testing. If it fails, examine the trial dir at `~/work/blobfish/jobs/<timestamp>/largest-eigenval__*/` for trajectory and trial.log.

---

## Tools you have available

- **Bash** — for running commands, capturing logs, running tests
- **Read/Edit/Write** — for code investigation and fixes
- **Agent** — to dispatch builder/reviewer subagents (use `subagent_type: general-purpose`, `isolation: worktree`)
- **codex CLI** — independent code review via `codex exec --skip-git-repo-check --sandbox read-only --full-auto "..."`. **AVOID `model_reasoning_effort=high`** — it stalled multiple times overnight. Default reasoning works fine.
- **Existing skills:** `~/work/llama.cpp/.claude/skills/{orchestrator,reviewer,codex,dev-team,qa-engineer}/SKILL.md`

## Useful debug commands

```bash
# Verbose server log (look here first when something's wrong):
tail -f ~/work/claude-cache-proxy/logs/v4-server-*.log

# Inspect what tokens the server emitted:
grep "process_toke" /path/to/server.log | tail -20

# Check whether GPU offload happened:
grep "offloaded.*GPU" /path/to/server.log

# Watch token decode in real time:
tail -f /path/to/server.log | grep --line-buffered "process_toke"

# Compare our V4 graph against antirez's reference:
diff <(git show HEAD:src/models/deepseek4.cpp) <(git show antirez/main:src/models/deepseek4.cpp) | head -60

# View the V4 chat template:
cat models/templates/deepseek-ai-DeepSeek-V4.jinja

# Re-run all overnight gates (sanity check that you didn't regress anything):
V4_GGUF=/Users/cchuter/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
./tests/v4-port/run-all-gates.sh
```

## Constraints / ground rules

- **Never push to `origin`.** Origin = ggml-org/llama.cpp upstream. The user has explicitly asked not to surface work to upstream maintainers until ready.
- **Never run `gh pr create` against any repo.**
- Push to `mine` (cchuter/llama.cpp) freely, including merging to `mine/master` if all gates green.
- Don't amend existing commits. Add new commits.
- Don't skip git hooks.
- Codex review for code-review state is non-negotiable per dev-team rules. Use default reasoning effort (high stalls).
- The user authorized up to 8 codex review rounds + 8 builder fix rounds per task before marking `needs-human`.
- "Don't block unless we have to" — if a task hits 8+8 exhaustion, mark `needs-human` and continue with surviving streams.
- This is a debug task, not a feature task. Avoid scope creep. **Don't refactor V4 code that isn't related to the bug.**

## Definition of done

You've completed this task when:

1. The four curl tests in "Concrete evidence" produce coherent text (not garbage, not loops)
2. `tests/v4-port/run-all-gates.sh` still passes (no regression)
3. A new gate `tests/v4-port/gate-server-chat.sh` exists and is wired into `run-all-gates.sh`, asserting the chat-format path works
4. A single terminal-bench task (`-t "largest-eigenval*"`) completes with `reward: 1.0` (or, if it fails for non-V4 reasons, the trajectory shows V4 generating coherent agent output)
5. All commits pushed to `mine`
6. A short follow-up document at `docs/plans/v4-port-debug-completion.md` describing what the bug was, where the fix landed, and what hypotheses you ruled out

## Hypotheses you can deprioritize

These were considered and ruled out by the overnight pipeline:

- ❌ Build/compile issues — build is clean, all gates pass
- ❌ Metal kernel correctness on simple inputs — gate-coherence Metal PASS, gate-speed PASS
- ❌ Loader/architecture detection — gate-loader PASS, model metadata correct
- ❌ Tool-call extraction format — gate-tools 5/5 pass

---

## Background reading (in priority order)

1. `docs/plans/v4-port-completion.md` — what shipped, gate results
2. `docs/superpowers/specs/2026-05-04-v4-port-overnight-design.md` — the spec, particularly §3 (Validation) and §4 (Tool calling)
3. `tasks/done/v4-port-A-loader.json` history — the structural-port story, including the bugs codex caught
4. `tasks/done/v4-port-D-tools.json` history — the Anthropic-mixed-content Jinja fix
5. `models/templates/deepseek-ai-DeepSeek-V4.jinja` — the chat template you'll likely modify
6. Antirez's reference: `git log antirez/main`; particularly check his deepseek4.cpp diff and his Jinja template

Good luck. The 80% that works is real progress. This is a focused debug on one path, not a full rebuild.
