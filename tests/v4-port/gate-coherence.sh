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

# --no-repack: V4 Q8_0 is ~282 GiB on disk. With repack ON (the default),
# the loader allocates a CPU_REPACK buffer the same size as the mmap'd model
# AND keeps the source mmap resident — the repack codepath in
# ggml/src/ggml-cpu/repack.cpp does not release the source pages after
# populating the SIMD-friendly layout (see src/llama-model-loader.cpp:1555).
# That doubles the memory requirement to ~575 GiB, which exceeds the 512 GiB
# physical RAM on M3 Ultra hosts and triggers an OOM SIGKILL well before any
# generation completes. This is a llama.cpp upstream issue independent of
# Task I's imatrix work; until that's fixed we disable repack here so the
# coherence gate exercises the (slower but correct) non-repacked CPU path.
# Prompt: open-ended narrative continuation. At --temp 0 the model picks
# argmax tokens, which means a short factual prompt ("The capital of France
# is") collapses to "Paris." and then repeats that sentence to fill the
# requested -n 30 — tripping the repetitive-decode heuristic below even
# though the output is correct. A narrative prompt forces the model to
# generate varied story text, exercising the same coherence properties
# without the repetition trap.
"$LLAMA_BIN/bin/llama-completion" -m "$V4_GGUF" -ngl "$NGL" --no-repack \
  -p "Once upon a time, in a small village by the sea, there lived" \
  -n 30 --no-warmup --temp 0 -no-cnv \
  < /dev/null > "$LOG_OUT" 2> "$LOG_ERR" || true

# Show a few last lines (use whichever stream — they're separate now).
# Avoid materializing $(cat "$LOG_OUT" "$LOG_ERR"): bash 3.2's command-
# substitution buffer chokes on tens-of-MB DeepSeek stderr.
tail -5 "$LOG_OUT" "$LOG_ERR" 2>/dev/null

# NaN/Inf scan: grep over the temp files directly to avoid command-
# substitution buffer overflow. `inf` shows up benignly in things like
# "info" or build flags; require an actual NaN/Inf marker pattern
# (numeric context).
if grep -qiE '\bnan\b|[-+]?inf[^o]' "$LOG_OUT" "$LOG_ERR" 2>/dev/null; then
  echo "FAIL: NaN/Inf detected"
  exit 1
fi

# Generated text lives in stdout. llama-completion echoes the prompt prefix
# plus the model continuation; strip the prompt and any "> EOF by user"
# trailer the new CLI prints when it exits.
GEN=$(tr -d '\r' < "$LOG_OUT")
GENONLY=$(printf '%s' "$GEN" \
  | sed 's/Once upon a time, in a small village by the sea, there lived//' \
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
