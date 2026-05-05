#!/usr/bin/env bash
# Stream B/C coherence gate: 30 tokens, >80% printable ASCII, not single repeated token
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"
NGL="${NGL:-0}"   # 0 = CPU, 999 = full Metal offload

# Use temp files rather than $(...) capture: V4 GGUFs emit tens of MBs of
# tokenizer / control-token chatter on stderr, which can blow bash 3.2's
# command-substitution buffer ("xrealloc: cannot allocate ...").
#
# The new llama-cli (b9111+, "new CLI experience" #17824) auto-enables
# conversation mode whenever the model has a chat template and refuses to
# accept --no-conversation, telling us to "use llama-completion instead".
# llama-completion is the supported headless one-shot driver and exits
# cleanly after -n tokens (or EOS). Close stdin so it never blocks waiting
# on user input.
LOG_OUT=$(mktemp -t v4-port-gate-coherence-out.XXXXXX)
LOG_ERR=$(mktemp -t v4-port-gate-coherence-err.XXXXXX)
trap 'rm -f "$LOG_OUT" "$LOG_ERR"' EXIT

"$LLAMA_BIN/bin/llama-completion" -m "$V4_GGUF" -ngl "$NGL" \
  -p "The capital of France is" -n 30 --no-warmup --temp 0 \
  < /dev/null > "$LOG_OUT" 2> "$LOG_ERR" || true

# Combined view for global checks (NaN/Inf scan, displayed tail).
COMBINED=$(cat "$LOG_OUT" "$LOG_ERR")
echo "$COMBINED" | tail -5

if echo "$COMBINED" | grep -qiE "nan|inf"; then
  # `inf` shows up benignly in things like "info" or build flags; require
  # an actual NaN/Inf marker pattern (numeric context).
  if echo "$COMBINED" | grep -qiE '\bnan\b|[-+]?inf[^o]'; then
    echo "FAIL: NaN/Inf detected"
    exit 1
  fi
fi

# Generated text lives in stdout. llama-completion echoes the prompt prefix
# plus the model continuation; strip the prompt and any "> EOF by user"
# trailer the new CLI prints when it exits.
GEN=$(tr -d '\r' < "$LOG_OUT")
GENONLY=$(printf '%s' "$GEN" \
  | sed 's/The capital of France is//' \
  | sed 's/> EOF by user//' \
  | tr '\n' ' ')

TOTAL=$(printf '%s' "$GENONLY" | wc -c | tr -d ' ')
PRINTABLE=$(printf '%s' "$GENONLY" | tr -cd '[:print:][:space:]' | wc -c | tr -d ' ')
if [ "${TOTAL:-0}" -lt 5 ]; then
  echo "FAIL: too few generated chars ($TOTAL)"
  exit 1
fi
RATIO=$(awk "BEGIN {print ($PRINTABLE / $TOTAL) * 100}")
echo "Printable ratio: $RATIO%"
awk "BEGIN { if (($PRINTABLE / $TOTAL) < 0.80) exit 1 }" \
  || { echo "FAIL: <80% printable"; exit 1; }

# Extract decode token count from the eval-time perf line.
# llama-completion emits: "common_perf_print: eval time = ... / N runs ..."
# (older builds said "/ N tokens"). Accept either suffix; anchor on the
# common_perf_print prefix to skip the "prompt eval time" line.
TOKEN_COUNT=$(grep -aE 'common_perf_print:[[:space:]]+eval time' "$LOG_ERR" \
  | grep -oE '/[[:space:]]*[0-9]+[[:space:]]+(runs|tokens)' \
  | head -1 | grep -oE '[0-9]+')
TOKEN_COUNT=${TOKEN_COUNT:-0}
echo "Decode tokens: $TOKEN_COUNT"
if [ "$TOKEN_COUNT" -lt 25 ]; then
  echo "FAIL: only $TOKEN_COUNT tokens generated (expected >=25)"
  exit 1
fi

# Word-level repetition check: degenerate decode often loops a single token/word.
WORDCOUNT=$(printf '%s' "$GENONLY" | tr -s ' \n' '\n' | grep -v "^$" | wc -l | tr -d ' ')
TOPWORD=$(printf '%s' "$GENONLY" | tr -s ' \n' '\n' | grep -v "^$" | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
if [ "${WORDCOUNT:-0}" -ge 5 ]; then
  awk "BEGIN { if ($TOPWORD / $WORDCOUNT > 0.5) exit 1 }" \
    || { echo "FAIL: repetitive decode (top word $TOPWORD/$WORDCOUNT)"; exit 1; }
fi

# Secondary sanity floor: unique-char check (kept as belt-and-suspenders).
UNIQ=$(printf '%s' "$GENONLY" | fold -w1 | sort -u | wc -l | tr -d ' ')
if [ "${UNIQ:-0}" -lt 4 ]; then
  echo "FAIL: degenerate decode (<4 unique chars)"
  exit 1
fi

echo "PASS: coherence (NGL=$NGL, gen='$GENONLY')"
