#!/usr/bin/env bash
# Stream A gate: model loader recognizes V4 GGUF, prints metadata, exits 0
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"

OUT=$("$LLAMA_BIN/bin/llama-cli" -m "$V4_GGUF" --no-warmup -p "" -n 0 2>&1) || true
echo "$OUT" | tail -20

# Required metadata markers
echo "$OUT" | grep -qE "arch\s*=\s*deepseek4|arch[: ]+deepseek4" || { echo "FAIL: arch not deepseek4"; exit 1; }
echo "$OUT" | grep -qE "n_params|llm_load_print_meta" || { echo "FAIL: no model metadata block"; exit 1; }

echo "PASS: loader recognizes V4 GGUF"
