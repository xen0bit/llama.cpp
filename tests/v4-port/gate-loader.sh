#!/usr/bin/env bash
# Stream A gate: model loader recognizes V4 GGUF, prints metadata, exits 0
set -euo pipefail
LLAMA_BIN="${LLAMA_BIN:-build}"
V4_GGUF="${V4_GGUF:?V4_GGUF must be set}"

STATUS=0
# Capture llama-cli output through a temp file rather than $(...) — V4 GGUFs
# emit tens of megabytes of tokenizer / control-token chatter, which can blow
# bash 3.2's command-substitution buffer ("xrealloc: cannot allocate ...").
# Close stdin: when llama-cli detects a chat template it auto-enables
# conversation mode, which blocks waiting for user input even with -n 0.
LOG=$(mktemp -t v4-port-gate-loader.XXXXXX)
trap 'rm -f "$LOG"' EXIT

# -st (single-turn) plus a non-empty prompt forces llama-cli to run one
# pass and exit. Without -st, conversation mode keeps polling stdin even
# with -n 0 + < /dev/null, hanging the gate indefinitely.
# -lv 3 (info) is needed because the default verbosity (1=error) suppresses
# the loader's `print_info:` metadata block we assert against below.
"$LLAMA_BIN/bin/llama-cli" -m "$V4_GGUF" --no-warmup -p "ping" -n 0 -st -lv 3 \
    < /dev/null > "$LOG" 2>&1 || STATUS=$?
STATUS=${STATUS:-0}
tail -20 "$LOG"

# Required metadata markers (must run BEFORE the STATUS check so we get a useful
# "FAIL: arch not deepseek4" message when V4 isn't ported yet, rather than just
# "FAIL: llama-cli exited 1").
grep -qE "arch\s*=\s*deepseek4|arch[: ]+deepseek4" "$LOG" || { echo "FAIL: arch not deepseek4"; exit 1; }
# Newer llama-cli prints the metadata block with `print_info:` (e.g.
# `print_info: model params = 284.33 B`) and no longer emits the legacy
# `llm_load_print_meta` / `n_params` markers.
grep -qE "n_params|llm_load_print_meta|print_info: model params" "$LOG" || { echo "FAIL: no model metadata block"; exit 1; }

# Now that metadata looks right, also require llama-cli itself to have exited cleanly.
[ "$STATUS" = "0" ] || { echo "FAIL: llama-cli exited $STATUS"; exit 1; }

echo "PASS: loader recognizes V4 GGUF"
