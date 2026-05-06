#!/usr/bin/env bash
# V4 q8 KV regression gate.
#
# Boots llama-server with --cache-type-k|v q8_0 against the V4 GGUF and asserts
# coherent chat output. Without the fix in this branch, the server produces
# "=" loops or "Mirror …" garbage; with the fix it should either produce
# coherent output, or fail-fast at startup with a clear diagnostic.

set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"
PORT="${PORT:-8090}"

# Mode controls expected behavior:
#   ok       (default) -- server boots and produces coherent output
#   bail     -- server exits non-zero at startup with diagnostic substring
#   warn     -- same as ok, plus log must contain the override WARN line
MODE="${MODE:-ok}"
EXPECTED_DIAG="${EXPECTED_DIAG:-DeepSeek4 KV cache requires fp16}"
EXPECTED_WARN="${EXPECTED_WARN:-DeepSeek4: forcing fp16 KV cache}"

TMP="$(mktemp -d -t v4-port-gate-server-chat-q8.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

cmd=(
  "$LLAMA_BIN/bin/llama-server"
  -m "$V4_GGUF" --jinja --reasoning-budget 0
  --port "$PORT" --ctx-size 32768 -ngl 999 --parallel 1 --flash-attn on
  --threads-batch 32
  --cache-type-k q8_0 --cache-type-v q8_0
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05
)

if [ "$MODE" = "bail" ]; then
  # Per codex round-2 guidance: do NOT run server in foreground waiting for
  # exit -- if the server somehow boots successfully despite expected bail,
  # the gate would hang forever. Start in background, give it a deterministic
  # window to exit, then fail if /health responds OR the process is still
  # alive past the window. Pass only on non-zero early exit with diagnostic.
  "${cmd[@]}" > "$TMP/server.log" 2>&1 &
  SERVER_PID=$!
  trap "kill $SERVER_PID 2>/dev/null; rm -rf '$TMP'" EXIT INT TERM

  bail_window=30
  for i in $(seq 1 "$bail_window"); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
      echo "FAIL[bail]: server became healthy (expected non-zero early exit)"
      tail -40 "$TMP/server.log"
      kill "$SERVER_PID" 2>/dev/null || true
      exit 1
    fi
    sleep 1
  done

  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL[bail]: server still alive after ${bail_window}s (expected non-zero early exit)"
    tail -40 "$TMP/server.log"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
  fi

  wait "$SERVER_PID" 2>/dev/null
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "FAIL[bail]: server exited with rc=0 (expected non-zero)"
    tail -40 "$TMP/server.log"
    exit 1
  fi
  if ! grep -q "$EXPECTED_DIAG" "$TMP/server.log"; then
    echo "FAIL[bail]: diagnostic '$EXPECTED_DIAG' not found in server log"
    tail -40 "$TMP/server.log"
    exit 1
  fi
  echo "PASS: server-chat-q8 (bail; rc=$rc; diagnostic present)"
  exit 0
fi

# ok / warn modes: boot the server in the background and run curl tests.
"${cmd[@]}" > "$TMP/server.log" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; rm -rf '$TMP'" EXIT INT TERM

for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then break; fi
  sleep 5
done
if ! curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
  echo "FAIL: server did not become healthy within 5 minutes"
  tail -40 "$TMP/server.log"
  exit 1
fi

assert_coherent() {
  local resp="$1" label="$2" min_tokens="${3:-5}" min_uniq="${4:-4}"
  python3 - "$resp" "$label" "$min_tokens" "$min_uniq" <<'PY'
import json, sys
from collections import Counter
resp_path, label, min_tokens, min_uniq = sys.argv[1:]
min_tokens = int(min_tokens); min_uniq = int(min_uniq)
with open(resp_path) as f: d = json.load(f)
choices = d.get('choices') or []
if not choices: print(f"FAIL[{label}]: no choices"); sys.exit(1)
msg = choices[0].get('message', {}) or {}
text = (msg.get('content') or '').strip() or (msg.get('reasoning_content') or '').strip()
n = (d.get('usage') or {}).get('completion_tokens', 0)
if n < min_tokens:
    print(f"FAIL[{label}]: only {n} tokens (expected >= {min_tokens})"); sys.exit(1)
if not text or len(text) < 5:
    print(f"FAIL[{label}]: empty or too-short text: {text!r}"); sys.exit(1)
if len(set(text)) < min_uniq:
    print(f"FAIL[{label}]: degenerate decode ({len(set(text))} unique): {text[:80]!r}")
    sys.exit(1)
ch, cn = Counter(text).most_common(1)[0]
if cn / len(text) > 0.70:
    print(f"FAIL[{label}]: repetitive ({ch!r} = {cn}/{len(text)}): {text[:80]!r}")
    sys.exit(1)
print(f"OK[{label}]: {n} tokens, {len(text)} chars, {len(set(text))} unique")
PY
}

curl -sS -o "$TMP/r1.json" -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"v4","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200,"temperature":0}'
assert_coherent "$TMP/r1.json" "tiny-no-tools-q8"

PAYLOAD=$(python3 -c 'import json; print(json.dumps({"model":"v4","messages":[{"role":"user","content":"Summarize this in one sentence: " + ("The quick brown fox jumps. " * 200)}],"max_tokens":80,"temperature":0}))')
curl -sS -o "$TMP/r2.json" -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' -d "$PAYLOAD"
assert_coherent "$TMP/r2.json" "medium-no-tools-q8"

FIXTURE="$(dirname "$0")/tool-call-fixture.json"
curl -sS -o "$TMP/r3.json" -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' -d @"$FIXTURE"
python3 - "$TMP/r3.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: d = json.load(f)
choices = d.get('choices') or []
if not choices: print('FAIL[tools-q8]: no choices'); sys.exit(1)
msg = choices[0].get('message', {}) or {}
tc = msg.get('tool_calls')
if not isinstance(tc, list) or len(tc) == 0:
    print(f"FAIL[tools-q8]: tool_calls not a non-empty array"); sys.exit(1)
print(f"OK[tools-q8]: {len(tc)} tool_call(s)")
PY

if [ "$MODE" = "warn" ]; then
  if ! grep -q "$EXPECTED_WARN" "$TMP/server.log"; then
    echo "FAIL[warn]: WARN '$EXPECTED_WARN' not found in server log"
    tail -40 "$TMP/server.log"
    exit 1
  fi
  echo "PASS: server-chat-q8 (warn; coherent + override WARN observed)"
  exit 0
fi

echo "PASS: server-chat-q8 (ok; 3/3 coherent under q8 KV)"
