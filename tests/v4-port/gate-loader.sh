#!/usr/bin/env bash
# Stream A gate: model loader recognizes V4 GGUF, prints metadata, exits 0
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"

STATUS=0
OUT=$("$LLAMA_BIN/bin/llama-cli" -m "$V4_GGUF" --no-warmup -p "" -n 0 2>&1) || STATUS=$?
STATUS=${STATUS:-0}
echo "$OUT" | tail -20

# Required metadata markers (must run BEFORE the STATUS check so we get a useful
# "FAIL: arch not deepseek4" message when V4 isn't ported yet, rather than just
# "FAIL: llama-cli exited 1").
echo "$OUT" | grep -qE "arch\s*=\s*deepseek4|arch[: ]+deepseek4" || { echo "FAIL: arch not deepseek4"; exit 1; }
echo "$OUT" | grep -qE "n_params|llm_load_print_meta" || { echo "FAIL: no model metadata block"; exit 1; }

# Now that metadata looks right, also require llama-cli itself to have exited cleanly.
[ "$STATUS" = "0" ] || { echo "FAIL: llama-cli exited $STATUS"; exit 1; }

echo "PASS: loader recognizes V4 GGUF"
