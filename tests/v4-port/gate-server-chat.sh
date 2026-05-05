#!/usr/bin/env bash
# V4 chat-completion regression gate.
#
# Boots llama-server with the V4-recommended flag set (matching
# claude-cache-proxy/start-server-v4.sh -- specifically WITHOUT
# --cache-type-k|v q8_0, which corrupts V4's compressed/indexer KV cache).
# Sends three curl tests through /v1/chat/completions and asserts each one
# returns coherent output rather than the degenerate "=" loop / "Mirror …"
# garbage we hit before the q8 KV cache was removed from the recipe.
#
# See docs/plans/v4-port-debug-completion.md for the bisection.

set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"
PORT="${PORT:-8089}"
TMP="$(mktemp -d -t v4-port-gate-server-chat.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# Boot the server with the V4-recommended flag set. Crucially, do NOT pass
# --cache-type-k|v q8_0 (the bug) and DO pass --reasoning-budget 0 (matches
# the failing-config from the original handoff -- proves the fix works under
# the same conditions that previously broke).
"$LLAMA_BIN/bin/llama-server" \
  -m "$V4_GGUF" --jinja --reasoning-budget 0 \
  --port "$PORT" --ctx-size 32768 -ngl 999 --parallel 1 --flash-attn on \
  --threads-batch 32 \
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05 \
  > "$TMP/server.log" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; rm -rf '$TMP'" EXIT INT TERM

# Wait up to 5 minutes for /health.
for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then break; fi
  sleep 5
done
if ! curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
  echo "FAIL: server did not become healthy within 5 minutes"
  tail -40 "$TMP/server.log"
  exit 1
fi

# Helper: assert a chat-completions response is coherent text.
# Args: response file, label, min completion tokens, min unique chars.
assert_coherent() {
  local resp="$1" label="$2" min_tokens="${3:-5}" min_uniq="${4:-4}"
  python3 - "$resp" "$label" "$min_tokens" "$min_uniq" <<'PY'
import json, sys

resp_path, label, min_tokens, min_uniq = sys.argv[1:]
min_tokens = int(min_tokens); min_uniq = int(min_uniq)

with open(resp_path) as f:
    d = json.load(f)

choices = d.get('choices') or []
if not choices:
    print(f"FAIL[{label}]: no choices in response"); sys.exit(1)
msg = choices[0].get('message', {}) or {}
content   = (msg.get('content') or '').strip()
reasoning = (msg.get('reasoning_content') or '').strip()
text = content if content else reasoning

usage = d.get('usage', {}) or {}
n_tokens = usage.get('completion_tokens', 0)

if n_tokens < min_tokens:
    print(f"FAIL[{label}]: only {n_tokens} completion_tokens (expected >= {min_tokens})")
    print(f"  message: {msg!r}")
    sys.exit(1)
if not text:
    print(f"FAIL[{label}]: empty content AND empty reasoning_content")
    print(f"  message: {msg!r}")
    sys.exit(1)
if len(text) < 5:
    print(f"FAIL[{label}]: text too short ({len(text)} chars): {text!r}"); sys.exit(1)
if len(set(text)) < min_uniq:
    print(f"FAIL[{label}]: degenerate decode ({len(set(text))} unique chars): {text!r}")
    sys.exit(1)

# Repetition: a single character occupying >70% of the text is the "=" loop.
from collections import Counter
ch, n = Counter(text).most_common(1)[0]
if n / len(text) > 0.70:
    print(f"FAIL[{label}]: repetitive decode ({ch!r} = {n}/{len(text)} chars): {text[:80]!r}")
    sys.exit(1)

print(f"OK[{label}]: {n_tokens} tokens, {len(text)} chars, {len(set(text))} unique")
PY
}

# Test 1: tiny prompt, no tools.
curl -sS -o "$TMP/r1.json" -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"v4","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200,"temperature":0}'
assert_coherent "$TMP/r1.json" "tiny-no-tools" 5 4

# Test 2: medium prompt (200 repeats of "fox jumps") -- previously hit the "=" loop.
PAYLOAD=$(python3 -c 'import json; print(json.dumps({"model":"v4","messages":[{"role":"user","content":"Summarize this in one sentence: " + ("The quick brown fox jumps. " * 200)}],"max_tokens":80,"temperature":0}))')
curl -sS -o "$TMP/r2.json" -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' -d "$PAYLOAD"
assert_coherent "$TMP/r2.json" "medium-no-tools" 5 4

# Test 3: tool-call fixture -- proves tools still work alongside the chat fix.
FIXTURE="$(dirname "$0")/tool-call-fixture.json"
curl -sS -o "$TMP/r3.json" -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' -d @"$FIXTURE"
python3 - "$TMP/r3.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: d = json.load(f)
choices = d.get('choices') or []
if not choices: print('FAIL[tools]: no choices'); sys.exit(1)
msg = choices[0].get('message', {}) or {}
tc = msg.get('tool_calls')
if not isinstance(tc, list) or len(tc) == 0:
    print(f"FAIL[tools]: tool_calls not a non-empty array: {msg!r}"); sys.exit(1)
print(f"OK[tools]: {len(tc)} tool_call(s)")
PY

echo "PASS: server-chat (3/3 tests coherent)"
