#!/usr/bin/env bash
# Stream C speed gate: Metal decode > 10 tok/s on a 50-token generation
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"
MIN_TPS="${MIN_TPS:-10}"

OUT=$("$LLAMA_BIN/bin/llama-cli" -m "$V4_GGUF" -ngl 999 \
  -p "Tell me a short story about a dog." -n 50 --temp 0 2>&1) || true
echo "$OUT" | tail -10

# -ngl 999 only requests offload; CPU fallback could still pass the TPS bar.
# Require explicit Metal backend markers in llama-cli output before trusting tok/s.
echo "$OUT" | grep -qE "Metal|MTL backend|ggml_metal_init|using device.*Metal" \
  || { echo "FAIL: Metal backend not detected in output"; exit 1; }

TPS=$(echo "$OUT" | grep -oE "[0-9.]+ tokens per second" | tail -1 | awk '{print $1}')
if [ -z "$TPS" ]; then echo "FAIL: no tokens-per-second metric in output"; exit 1; fi

echo "Decode tok/s: $TPS (min: $MIN_TPS)"
awk "BEGIN { if ($TPS < $MIN_TPS) exit 1 }" || { echo "FAIL: $TPS < $MIN_TPS tok/s"; exit 1; }

echo "PASS: speed (NGL=999, $TPS tok/s)"
