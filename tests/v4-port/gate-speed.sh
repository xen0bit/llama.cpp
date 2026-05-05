#!/usr/bin/env bash
# Stream C speed gate: Metal decode > 10 tok/s on a 50-token generation
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"
MIN_TPS="${MIN_TPS:-10}"

# Use temp files rather than $(...) capture: V4 GGUFs emit tens of MBs of
# tokenizer / control-token chatter on stderr, which can blow bash 3.2's
# command-substitution buffer ("xrealloc: cannot allocate ...").
#
# The new llama-cli (b9111+, "new CLI experience" #17824) auto-enables
# conversation mode whenever the model has a chat template and refuses to
# accept --no-conversation, telling us to "use llama-completion instead".
# llama-completion is the supported headless one-shot driver and exits
# cleanly after -n tokens (or EOS). Close stdin so it never blocks waiting
# on user input. Same pattern as gate-coherence.sh.
LOG_OUT=$(mktemp -t v4-port-gate-speed-out.XXXXXX)
LOG_ERR=$(mktemp -t v4-port-gate-speed-err.XXXXXX)
trap 'rm -f "$LOG_OUT" "$LOG_ERR"' EXIT

"$LLAMA_BIN/bin/llama-completion" -m "$V4_GGUF" -ngl 999 \
  -p "Tell me a short story about a dog." -n 50 --no-warmup --temp 0 -no-cnv \
  < /dev/null > "$LOG_OUT" 2> "$LOG_ERR" || true

tail -10 "$LOG_OUT" "$LOG_ERR" 2>/dev/null

# -ngl 999 only requests offload; CPU fallback could still pass the TPS bar.
# Require explicit Metal backend markers in output before trusting tok/s.
grep -qaE "Metal|MTL backend|ggml_metal_init|using device.*Metal" "$LOG_OUT" "$LOG_ERR" \
  || { echo "FAIL: Metal backend not detected in output"; exit 1; }

# Verify layers actually offloaded (not just Metal init).
OFFLOAD=$(grep -aoE "offloaded [0-9]+/[0-9]+ layers to GPU" "$LOG_OUT" "$LOG_ERR" \
  | head -1 | sed 's/^[^:]*://')
OFFLOAD=$(printf '%s' "$OFFLOAD" | sed 's/^ *//')
if [ -z "$OFFLOAD" ]; then
  echo "FAIL: no GPU layer offload detected (Metal init alone is not sufficient)"
  exit 1
fi
N_OFF=$(printf '%s' "$OFFLOAD" | awk '{print $2}' | cut -d/ -f1)
if [ "$N_OFF" -lt 1 ]; then
  echo "FAIL: 0 layers offloaded to GPU ($OFFLOAD)"
  exit 1
fi
echo "GPU offload: $OFFLOAD"

# common_perf_print emits two "tokens per second" lines:
#   "prompt eval time = ... / N tokens (... tokens per second)"  [prefill]
#   "       eval time = ... / N runs   (... tokens per second)"  [decode]
# We want the decode (eval) line. Anchor on the eval (not prompt-eval) prefix.
TPS=$(grep -aE 'common_perf_print:[[:space:]]+eval time' "$LOG_ERR" \
  | grep -oE '[0-9.]+ tokens per second' | tail -1 | awk '{print $1}')
if [ -z "$TPS" ]; then
  # Fallback: any "tokens per second" line, prefer the last (decode is reported after prefill)
  TPS=$(grep -aoE "[0-9.]+ tokens per second" "$LOG_ERR" "$LOG_OUT" \
    | tail -1 | awk '{print $1}')
fi
if [ -z "$TPS" ]; then echo "FAIL: no tokens-per-second metric in output"; exit 1; fi

echo "Decode tok/s: $TPS (min: $MIN_TPS)"
awk "BEGIN { if ($TPS < $MIN_TPS) exit 1 }" || { echo "FAIL: $TPS < $MIN_TPS tok/s"; exit 1; }

echo "PASS: speed (NGL=999, $TPS tok/s)"
