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

# Token-count assertion: llama-cli prints a timing summary like
#   "eval time = ... / N tokens" or "N tokens" in its perf block.
# We requested -n 30; allow some EOS slack and require N >= 25.
NTOKENS=$(echo "$OUT" | grep -oE "[0-9]+ runs" -m 1 >/dev/null 2>&1 || true; \
  echo "$OUT" | grep -oE "/[[:space:]]*[0-9]+[[:space:]]+tokens" | tail -1 | grep -oE "[0-9]+" | head -1)
if [ -z "${NTOKENS:-}" ]; then
  # Fallback: look for "N tokens" anywhere in the timing/perf block.
  NTOKENS=$(echo "$OUT" | grep -oE "[0-9]+ tokens" | tail -1 | awk '{print $1}')
fi
if [ -z "${NTOKENS:-}" ] || [ "$NTOKENS" -lt 25 ]; then
  echo "FAIL: only ${NTOKENS:-0} tokens generated (expected >=25)"; exit 1;
fi
echo "Tokens generated: $NTOKENS"

# Word-level repetition check: degenerate decode often loops a single token/word.
WORDCOUNT=$(echo "$GENONLY" | tr -s ' \n' '\n' | grep -v "^$" | wc -l | tr -d ' ')
TOPWORD=$(echo "$GENONLY" | tr -s ' \n' '\n' | grep -v "^$" | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
if [ "${WORDCOUNT:-0}" -ge 5 ]; then
  awk "BEGIN { if ($TOPWORD / $WORDCOUNT > 0.5) exit 1 }" || { echo "FAIL: repetitive decode (top word $TOPWORD/$WORDCOUNT)"; exit 1; }
fi

# Secondary sanity floor: unique-char check (kept as belt-and-suspenders).
UNIQ=$(echo -n "$GENONLY" | fold -w1 | sort -u | wc -l)
if [ "$UNIQ" -lt 4 ]; then echo "FAIL: degenerate decode (<4 unique chars)"; exit 1; fi

echo "PASS: coherence (NGL=$NGL, gen='$GENONLY')"
