#!/usr/bin/env bash
# Stream B/C coherence gate: 30 tokens, >80% printable ASCII, not single repeated token
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"
NGL="${NGL:-0}"   # 0 = CPU, 999 = full Metal offload

OUT=$("$LLAMA_BIN/bin/llama-cli" -m "$V4_GGUF" -ngl "$NGL" \
  -p "The capital of France is" -n 30 --no-warmup --temp 0 2>&1) || true
echo "$OUT" | tail -5

GEN=$(echo "$OUT" | grep -v "^llama_\|^ggml_\|^load_\|^build_\|^common_\|^llm_\|^main:" | tail -3 | tr -d '\n')

if echo "$OUT" | grep -qiE "nan|inf"; then echo "FAIL: NaN/Inf detected"; exit 1; fi

GENONLY=$(echo "$GEN" | sed 's/The capital of France is//')
TOTAL=$(echo -n "$GENONLY" | wc -c)
PRINTABLE=$(echo -n "$GENONLY" | tr -cd '[:print:][:space:]' | wc -c)
if [ "$TOTAL" -lt 5 ]; then echo "FAIL: too few generated chars ($TOTAL)"; exit 1; fi
RATIO=$(awk "BEGIN {print ($PRINTABLE / $TOTAL) * 100}")
echo "Printable ratio: $RATIO%"
awk "BEGIN { if (($PRINTABLE / $TOTAL) < 0.80) exit 1 }" || { echo "FAIL: <80% printable"; exit 1; }

UNIQ=$(echo -n "$GENONLY" | fold -w1 | sort -u | wc -l)
if [ "$UNIQ" -lt 4 ]; then echo "FAIL: degenerate decode (<4 unique chars)"; exit 1; fi

echo "PASS: coherence (NGL=$NGL, gen='$GENONLY')"
