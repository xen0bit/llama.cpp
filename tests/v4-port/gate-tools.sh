#!/usr/bin/env bash
# Stream D gate: 5 successive POSTs to llama-server, all return HTTP 200 with tool_calls array
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"
PORT="${PORT:-8089}"
FIXTURE="${FIXTURE:-$(dirname "$0")/tool-call-fixture.json}"

"$LLAMA_BIN/bin/llama-server" -m "$V4_GGUF" --jinja --port "$PORT" -ngl 999 \
  > /tmp/llama-server-tools.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT INT TERM

for i in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then break; fi
  sleep 2
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "FAIL: server didn't become healthy"; exit 1; }

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
