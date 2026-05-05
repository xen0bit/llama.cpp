# V4 Overnight Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port antirez's DeepSeek V4 Flash support onto fairydreaming's V3.2/DSA base, validate it loads + emits coherent tokens at >10 tok/s on Metal + serves tool calls with HTTP 200, all by morning.

**Architecture:** Cherry-pick antirez's 6 V4 commits onto `feat/v4-port` (branched off `deepseek-dsa`). Sequential phases: validation harness FIRST, then commit-by-commit port with gate tests after each. On gate failure, retry up to 8 codex rounds × 8 builder rounds before marking `needs-human` and continuing.

**Tech Stack:** llama.cpp (C++ + Metal Shading Language + Python conversion script), git format-patch / cherry-pick, codex CLI for review, ggml backend.

**Reference commits to port (in dependency order):**

| SHA (short) | Description | Lines |
|---|---|---|
| `06c504247` | Add DeepSeek V4 Flash inference support | +3827 |
| `188df615c` | Fix V4 long-context graph metadata | +19 |
| `b67f5db5c` | Optimize V4 Metal HC decode | +70 |
| `57c4283b5` | Remove stale V4 quantize tool entry | -1 |
| `2f2d44052` | Speed up V4 prompt replay | +317 |
| `3ba61fbb4` | Add V4 tool-call chat template | +103 |

**Branches available locally:**

- `deepseek-dsa` (read-only, our V3.2 base)
- `feat/v4-port` (current, where the port lands)
- `master` (read-only, upstream)
- `antirez/main` (read-only, source of V4 commits — fetched as remote)

---

## Task 0: Pre-flight setup

**Files:**
- Create: `tests/v4-port/README.md` (one-liner describing the dir)
- Modify: none

- [ ] **Step 0.1: Confirm working state**

```bash
cd ~/work/llama.cpp
git status --short                              # Expect: clean (or only ?? .claude/skills)
git branch --show-current                        # Expect: feat/v4-port
git remote -v | grep -E "(origin|mine|antirez)" # Expect: 3 remotes
git log --oneline -1                             # Expect: 5aae7b0f3 v4-port: clarify push policy
```

- [ ] **Step 0.2: Verify antirez commits are fetched**

```bash
cd ~/work/llama.cpp && git log --oneline antirez/main -7 | head
```

Expected output includes: `06c504247 Add DeepSeek V4 Flash inference support`. If commits absent, run `git fetch antirez`.

- [ ] **Step 0.3: Confirm GGUF download path exists (file may still be downloading)**

```bash
ls -la ~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf 2>&1
```

If absent, OK — Phase 1's chat-template extraction works without it. Phases 4+ require it; will poll.

- [ ] **Step 0.4: Verify baseline build still works**

```bash
cd ~/work/llama.cpp && cmake --build build -j 2>&1 | tail -3
```

Expected: `[100%] Built target llama-server` (or equivalent). If broken, abort and surface to user.

- [ ] **Step 0.5: Commit harness dir placeholder**

```bash
mkdir -p ~/work/llama.cpp/tests/v4-port
cat > ~/work/llama.cpp/tests/v4-port/README.md <<'EOF'
# V4 Port Validation Harness

Scripts that gate each phase of the V4 port. See docs/superpowers/plans/2026-05-04-v4-port-overnight.md.
EOF
cd ~/work/llama.cpp && git add tests/v4-port/README.md && git commit -m "v4-port: scaffold validation harness directory"
```

---

## Task 1: Validation harness (Phase E — done first, used by all later phases)

**Files:**
- Create: `tests/v4-port/gate-loader.sh`
- Create: `tests/v4-port/gate-coherence.sh`
- Create: `tests/v4-port/gate-speed.sh`
- Create: `tests/v4-port/gate-tools.sh`
- Create: `tests/v4-port/tool-call-fixture.json`
- Create: `tests/v4-port/run-all-gates.sh`

These scripts return 0 on pass, non-zero on fail. They consume two env vars:
- `V4_GGUF` — path to the V4 GGUF file
- `LLAMA_BIN` — path to llama.cpp build directory (default: `build`)

- [ ] **Step 1.1: Write `gate-loader.sh`**

```bash
cat > ~/work/llama.cpp/tests/v4-port/gate-loader.sh <<'EOF'
#!/usr/bin/env bash
# Stream A gate: model loader recognizes V4 GGUF, prints metadata, exits 0
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"

OUT=$("$LLAMA_BIN/bin/llama-cli" -m "$V4_GGUF" --no-warmup -p "" -n 0 2>&1)
echo "$OUT" | tail -20

# Required metadata markers
echo "$OUT" | grep -qE "arch\s*=\s*deepseek4|arch[: ]+deepseek4" || { echo "FAIL: arch not deepseek4"; exit 1; }
echo "$OUT" | grep -qE "n_params|llm_load_print_meta" || { echo "FAIL: no model metadata block"; exit 1; }

echo "PASS: loader recognizes V4 GGUF"
EOF
chmod +x ~/work/llama.cpp/tests/v4-port/gate-loader.sh
```

- [ ] **Step 1.2: Write `gate-coherence.sh`**

```bash
cat > ~/work/llama.cpp/tests/v4-port/gate-coherence.sh <<'EOF'
#!/usr/bin/env bash
# Stream B/C coherence gate: 30 tokens, >80% printable ASCII, not single repeated token
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"
NGL="${NGL:-0}"   # 0 = CPU, 999 = full Metal offload

OUT=$("$LLAMA_BIN/bin/llama-cli" -m "$V4_GGUF" -ngl "$NGL" \
  -p "The capital of France is" -n 30 --no-warmup --temp 0 2>&1)
echo "$OUT" | tail -5

# Extract just the generated text (last non-empty lines after prompt echo)
GEN=$(echo "$OUT" | grep -v "^llama_\|^ggml_\|^load_\|^build_\|^common_\|^llm_\|^main:" | tail -3 | tr -d '\n')

# Check 1: no NaN warnings
if echo "$OUT" | grep -qiE "nan|inf"; then echo "FAIL: NaN/Inf detected"; exit 1; fi

# Check 2: >80% printable ASCII among generated chars (filter prompt echo first)
GENONLY=$(echo "$GEN" | sed 's/The capital of France is//')
TOTAL=$(echo -n "$GENONLY" | wc -c)
PRINTABLE=$(echo -n "$GENONLY" | tr -cd '[:print:][:space:]' | wc -c)
if [ "$TOTAL" -lt 5 ]; then echo "FAIL: too few generated chars ($TOTAL)"; exit 1; fi
RATIO=$(awk "BEGIN {print ($PRINTABLE / $TOTAL) * 100}")
echo "Printable ratio: $RATIO%"
awk "BEGIN { if (($PRINTABLE / $TOTAL) < 0.80) exit 1 }" || { echo "FAIL: <80% printable"; exit 1; }

# Check 3: not all the same token (degenerate decode)
UNIQ=$(echo -n "$GENONLY" | fold -w1 | sort -u | wc -l)
if [ "$UNIQ" -lt 4 ]; then echo "FAIL: degenerate decode (<4 unique chars)"; exit 1; fi

echo "PASS: coherence (NGL=$NGL, gen='$GENONLY')"
EOF
chmod +x ~/work/llama.cpp/tests/v4-port/gate-coherence.sh
```

- [ ] **Step 1.3: Write `gate-speed.sh`**

```bash
cat > ~/work/llama.cpp/tests/v4-port/gate-speed.sh <<'EOF'
#!/usr/bin/env bash
# Stream C speed gate: Metal decode > 10 tok/s on a 50-token generation
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"
MIN_TPS="${MIN_TPS:-10}"

OUT=$("$LLAMA_BIN/bin/llama-cli" -m "$V4_GGUF" -ngl 999 \
  -p "Tell me a short story about a dog." -n 50 --temp 0 2>&1)
echo "$OUT" | tail -10

# llama.cpp prints a timing summary like:
#   eval time = 1234.56 ms / 50 tokens (24.69 ms per token, 40.50 tokens per second)
TPS=$(echo "$OUT" | grep -oE "[0-9.]+ tokens per second" | tail -1 | awk '{print $1}')
if [ -z "$TPS" ]; then echo "FAIL: no tokens-per-second metric in output"; exit 1; fi

echo "Decode tok/s: $TPS (min: $MIN_TPS)"
awk "BEGIN { if ($TPS < $MIN_TPS) exit 1 }" || { echo "FAIL: $TPS < $MIN_TPS tok/s"; exit 1; }

echo "PASS: speed (NGL=999, $TPS tok/s)"
EOF
chmod +x ~/work/llama.cpp/tests/v4-port/gate-speed.sh
```

- [ ] **Step 1.4: Write `tool-call-fixture.json`**

```bash
cat > ~/work/llama.cpp/tests/v4-port/tool-call-fixture.json <<'EOF'
{
  "model": "deepseek-v4-flash",
  "messages": [
    {"role": "user", "content": "What is the weather in Paris?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "The city name"}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "max_tokens": 128,
  "temperature": 0.0
}
EOF
```

- [ ] **Step 1.5: Write `gate-tools.sh`**

```bash
cat > ~/work/llama.cpp/tests/v4-port/gate-tools.sh <<'EOF'
#!/usr/bin/env bash
# Stream D gate: 5 successive POSTs to llama-server, all return HTTP 200 with tool_calls array
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"
PORT="${PORT:-8089}"
FIXTURE="${FIXTURE:-$(dirname "$0")/tool-call-fixture.json}"

# Boot llama-server in background
"$LLAMA_BIN/bin/llama-server" -m "$V4_GGUF" --jinja --port "$PORT" -ngl 999 \
  > /tmp/llama-server-tools.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT INT TERM

# Wait for ready (up to 60s)
for i in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then break; fi
  sleep 2
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "FAIL: server didn't become healthy"; exit 1; }

# Run 5 POSTs
PASS=0
for i in $(seq 1 5); do
  HTTP=$(curl -s -o /tmp/tools-resp-$i.json -w "%{http_code}" \
    -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "content-type: application/json" \
    -d @"$FIXTURE")
  echo "Request $i: HTTP $HTTP"
  if [ "$HTTP" = "200" ]; then
    if grep -q '"tool_calls"' /tmp/tools-resp-$i.json; then
      PASS=$((PASS+1))
    else
      echo "  FAIL: 200 but no tool_calls in body"
      head -c 500 /tmp/tools-resp-$i.json
    fi
  else
    echo "  body:"; head -c 500 /tmp/tools-resp-$i.json
  fi
done

if [ "$PASS" -lt 5 ]; then echo "FAIL: only $PASS/5 returned 200 with tool_calls"; exit 1; fi
echo "PASS: tool calling ($PASS/5 with tool_calls)"
EOF
chmod +x ~/work/llama.cpp/tests/v4-port/gate-tools.sh
```

- [ ] **Step 1.6: Write `run-all-gates.sh`**

```bash
cat > ~/work/llama.cpp/tests/v4-port/run-all-gates.sh <<'EOF'
#!/usr/bin/env bash
# Run every gate in order, stop on first failure
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
"$DIR/gate-loader.sh"
NGL=0   "$DIR/gate-coherence.sh"
NGL=999 "$DIR/gate-coherence.sh"
"$DIR/gate-speed.sh"
"$DIR/gate-tools.sh"
echo "ALL GATES PASS"
EOF
chmod +x ~/work/llama.cpp/tests/v4-port/run-all-gates.sh
```

- [ ] **Step 1.7: Smoke-test the harness against current binary (will fail at gate-loader because V4 isn't ported yet — that's expected)**

```bash
cd ~/work/llama.cpp
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  ./tests/v4-port/gate-loader.sh 2>&1 | tail -5
```

Expected: `FAIL: arch not deepseek4` (the current binary doesn't recognize V4). This validates the gate is correctly assertive — exit code should be non-zero.

- [ ] **Step 1.8: Commit**

```bash
cd ~/work/llama.cpp
git add tests/v4-port/
git commit -m "v4-port: validation harness (gates for loader, coherence, speed, tools)"
git push -u mine feat/v4-port
```

---

## Task 2: Phase A — Cherry-pick V4 inference support (loader + forward pass)

**Files (touched by commit `06c504247`):**
- New: `src/models/deepseek4.cpp` (+1347)
- Modify: `src/llama-arch.cpp`, `src/llama-arch.h` (V4 enum)
- Modify: `src/llama-model.cpp` (+200), `src/llama-model.h` (+25)
- Modify: `src/llama-memory-hybrid-iswa.cpp` (+625), `src/llama-memory-hybrid-iswa.h` (+35)
- Modify: `src/llama-model-loader.cpp` (+1)
- Modify: `convert_hf_to_gguf.py`, `gguf-py/gguf/constants.py`
- Modify: `src/models/models.h` (+4)

**Conflict expectations:** `llama-arch.cpp/h`, `llama-model.cpp/h`, `llama-memory-hybrid-iswa.cpp/h` all touched by both fairydreaming (V3.2/DSA) and antirez (V4). Use the rule from §1 of the spec: keep fairydreaming's V3.2 entries, ADD antirez's V4 entries alongside.

- [ ] **Step 2.1: Show what we're about to apply**

```bash
cd ~/work/llama.cpp
git show --stat 06c504247 | head -40
git log --oneline -1 HEAD                # Confirm we're on feat/v4-port
git status --short                        # Confirm clean
```

- [ ] **Step 2.2: Cherry-pick with 3-way merge**

```bash
cd ~/work/llama.cpp && git cherry-pick -x --strategy=recursive -X theirs 06c504247 2>&1 | tail -20
```

Note: `-X theirs` means "prefer antirez's version on conflict." We will manually walk back any cases where fairydreaming's V3.2 plumbing got overwritten — but for V4-specific code, antirez is authoritative.

If conflict arises:

- [ ] **Step 2.3: Walk every conflict**

```bash
cd ~/work/llama.cpp && git status | grep "both modified" | head -20
```

For each conflicted file, open it and apply the rule:
- DSA / sparse attention / lightning indexer code → keep fairydreaming's
- V4-specific (deepseek4 arch, V4 routing, V4 expert handling) → keep antirez's
- Shared infrastructure → merge BOTH (V3.2 entries + V4 entries side-by-side)

After resolving each file: `git add <file>`. After all files clean: `git cherry-pick --continue`.

- [ ] **Step 2.4: Build (this is where most port bugs surface)**

```bash
cd ~/work/llama.cpp && cmake --build build -j 2>&1 | tee /tmp/v4-build-A.log | tail -30
```

Expected: clean build, exit 0. If build errors, **iterate** — fix them, build again. Up to 8 builder rounds before stream is `needs-human`.

- [ ] **Step 2.5: Run the loader gate**

```bash
cd ~/work/llama.cpp
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  ./tests/v4-port/gate-loader.sh 2>&1 | tail -10
```

Expected: `PASS: loader recognizes V4 GGUF`. If GGUF not yet downloaded, skip this and move to step 2.6 with a note.

- [ ] **Step 2.6: Commit & push**

```bash
cd ~/work/llama.cpp
git status --short                            # Should be clean (cherry-pick already committed)
git push mine feat/v4-port
```

If gate-loader.sh failed: open codex review on the diff against `deepseek-dsa`, request a fix, iterate (up to 8 codex rounds).

---

## Task 3: Phase B — Cherry-pick V4 long-context fix and validate CPU forward pass

**Files (touched by commit `188df615c`):**
- Modify: `src/llama-context.cpp` (+8)
- Modify: `src/llama-memory-hybrid-iswa.cpp` (+13)

- [ ] **Step 3.1: Cherry-pick**

```bash
cd ~/work/llama.cpp && git cherry-pick -x --strategy=recursive -X theirs 188df615c 2>&1 | tail -5
```

- [ ] **Step 3.2: Resolve conflicts if any (apply same rule as Task 2.3)**

- [ ] **Step 3.3: Build**

```bash
cd ~/work/llama.cpp && cmake --build build -j 2>&1 | tail -5
```

- [ ] **Step 3.4: Run CPU coherence gate**

```bash
cd ~/work/llama.cpp
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
NGL=0 ./tests/v4-port/gate-coherence.sh 2>&1 | tail -10
```

Expected: `PASS: coherence (NGL=0, gen='...')`. If FAIL, iterate. The CPU forward pass should produce *some* coherent text — if it produces gibberish, the V4 graph in `deepseek4.cpp` has a port issue (most likely a tensor-name mapping or hparam problem in the loader). Have codex compare our `src/models/deepseek4.cpp` against antirez's reference.

- [ ] **Step 3.5: Push**

```bash
cd ~/work/llama.cpp && git push mine feat/v4-port
```

---

## Task 4: Phase C — Cherry-pick V4 Metal kernels + speed optimizations

**Files (touched by commits `b67f5db5c`, `2f2d44052`, `57c4283b5`):**
- Modify: `ggml/src/ggml-metal/ggml-metal.metal` (+89 across both)
- Modify: `src/models/deepseek4.cpp`
- Modify: `src/llama-context.cpp`, `src/llama-context.h`
- Modify: `tools/server/server-context.cpp`
- Modify: `tools/CMakeLists.txt` (–1)
- Modify: `ggml/src/ggml.c`

- [ ] **Step 4.1: Cherry-pick all three commits in order**

```bash
cd ~/work/llama.cpp
git cherry-pick -x --strategy=recursive -X theirs b67f5db5c 57c4283b5 2f2d44052 2>&1 | tail -10
```

- [ ] **Step 4.2: Resolve any conflicts (same rule)**

- [ ] **Step 4.3: Build with Metal**

```bash
cd ~/work/llama.cpp && cmake --build build -j 2>&1 | tee /tmp/v4-build-C.log | tail -10
```

Expected: Metal kernels compile (`.metal` files compile to `default.metallib`). If Metal compile fails, the .metal file has a port issue — most likely a missing function declaration or a Metal version mismatch.

- [ ] **Step 4.4: Run Metal coherence gate**

```bash
cd ~/work/llama.cpp
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
NGL=999 ./tests/v4-port/gate-coherence.sh 2>&1 | tail -10
```

Expected: `PASS: coherence (NGL=999, gen='...')`. If output is *coherent on CPU but gibberish on Metal*, the Metal kernel has a port bug — likely a buffer-binding or threadgroup-size issue. Have codex inspect the .metal diff.

- [ ] **Step 4.5: Run Metal speed gate**

```bash
cd ~/work/llama.cpp
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
./tests/v4-port/gate-speed.sh 2>&1 | tail -10
```

Expected: `PASS: speed (NGL=999, ~21 tok/s)`. (Antirez reported 21 tok/s on M3 Max; M3 Ultra should be ≥30 tok/s.) If <10 tok/s, the Metal HC kernel optimization (commit b67f5db5c) didn't apply — re-inspect.

- [ ] **Step 4.6: Push**

```bash
cd ~/work/llama.cpp && git push mine feat/v4-port
```

---

## Task 5: Phase D — Cherry-pick V4 tool-call chat template

**Files (touched by commit `3ba61fbb4`):**
- Modify: `common/chat.cpp` (+12 -5)
- New: `models/templates/deepseek-ai-DeepSeek-V4.jinja` (+96)

- [ ] **Step 5.1: Cherry-pick**

```bash
cd ~/work/llama.cpp && git cherry-pick -x --strategy=recursive -X theirs 3ba61fbb4 2>&1 | tail -5
```

- [ ] **Step 5.2: Build**

```bash
cd ~/work/llama.cpp && cmake --build build -j 2>&1 | tail -5
```

- [ ] **Step 5.3: Run tool-call gate**

```bash
cd ~/work/llama.cpp
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
./tests/v4-port/gate-tools.sh 2>&1 | tail -15
```

Expected: `PASS: tool calling (5/5 with tool_calls)`. If FAIL with HTTP 500s, the chat template is rejecting our message format — same class of problem we hit with Mistral. Most likely cause: tool_use/tool_result mapping doesn't match V4's expected format. Mitigation: add server-side parsing fallback (see spec §4 contingency).

- [ ] **Step 5.4: Push**

```bash
cd ~/work/llama.cpp && git push mine feat/v4-port
```

---

## Task 6: Phase F — Final integration gate + completion report

- [ ] **Step 6.1: Run all gates end-to-end**

```bash
cd ~/work/llama.cpp
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  ./tests/v4-port/run-all-gates.sh 2>&1 | tee /tmp/v4-port-final.log
```

Expected: `ALL GATES PASS`. If any gate fails, that gate's stream gets marked `needs-human` in the completion report; remaining gates still run.

- [ ] **Step 6.2: Write completion report**

```bash
cat > ~/work/llama.cpp/docs/plans/v4-port-completion.md <<EOF
# V4 Port — Overnight Run Completion Report

Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Branch: feat/v4-port @ $(git rev-parse --short HEAD)
Pushed to: https://github.com/cchuter/llama.cpp/tree/feat/v4-port

## Gate results

\`\`\`
$(cat /tmp/v4-port-final.log | tail -40)
\`\`\`

## Streams shipped

- A (loader): <fill in PASS/FAIL/NEEDS-HUMAN>
- B (CPU forward): <fill in>
- C (Metal kernels + speed): <fill in>
- D (tool calling): <fill in>
- E (validation harness): PASS

## What needs human attention tomorrow

<list each NEEDS-HUMAN gate with last error and pointer to /tmp/v4-build-*.log>

## Repro

\`\`\`bash
cd ~/work/llama.cpp
./build/bin/llama-server -m ~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf --jinja --port 8080 -ngl 999

# Another terminal:
curl -s localhost:8080/v1/chat/completions \\
  -H 'content-type: application/json' \\
  -d @tests/v4-port/tool-call-fixture.json
\`\`\`
EOF
```

- [ ] **Step 6.3: Commit completion report and push**

```bash
cd ~/work/llama.cpp
git add docs/plans/v4-port-completion.md
git commit -m "v4-port: completion report"
git push mine feat/v4-port
```

- [ ] **Step 6.4: If all gates passed — merge to mine/master (per user authorization)**

```bash
cd ~/work/llama.cpp
# Only run if Step 6.1 produced "ALL GATES PASS"
git fetch mine master 2>&1 | tail -2
git checkout master                              # Local master
git merge --ff-only mine/master 2>/dev/null || true   # Sync local to remote first
git checkout feat/v4-port
git push mine feat/v4-port:master                # Fast-forward mine/master to feat/v4-port
```

If gates partially failed: skip the merge. Leave `feat/v4-port` as-is for tomorrow's review.

- [ ] **Step 6.5: Final status print**

Print to stdout:
```
=== V4 Port Overnight Run — Done ===
Branch: feat/v4-port
Pushed: https://github.com/cchuter/llama.cpp/tree/feat/v4-port
Master merged: <yes/no>
Gates: <list pass/fail/needs-human>
Report: docs/plans/v4-port-completion.md
```

---

## Failure-handling decision tree

At any step where a gate fails:

1. **Codex review the failure** via `reviewer` skill — pass it the failed step's logs and the diff
2. **Builder applies fix** — edit code, rebuild, re-run gate
3. Repeat until: gate passes (move on) OR 8 codex rounds + 8 builder rounds exhausted
4. On exhaustion: mark this stream `needs-human`, leave its commit in place (don't revert), continue to next stream
5. **Stream A failure is the ONLY true blocker** — stop entirely if gate-loader fails after exhaustion. Subsequent streams have no working forward pass to test against.

## Self-review notes

This plan is for a **port**, not from-scratch development. Therefore:
- "Tests fail first → tests pass" pattern is realized as "gate fails on baseline → gate passes after cherry-pick"
- Bite-sized steps are git operations (cherry-pick, build, gate, push) rather than line-of-code edits
- The complete code being introduced is in antirez's commits, identified by SHA
- Builders don't need to write V4 forward-pass logic from scratch — they only need to resolve port conflicts and verify gates

This is appropriate for the port-not-invent strategy approved in the brainstorm.
