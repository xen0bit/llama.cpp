#!/usr/bin/env bash
# gate-imatrix: regression test that llama-imatrix runs end-to-end on
# V4 without crashing and collects activations from at least the
# expected minimum number of weight tensors.
#
# Background: V4's KV-cache layout (n_stream==1 hard-coded in the
# compressed-attention graph) and V4-specific ops (LIGHTNING_INDEXER,
# DSV4_HC_*, DSV4_FP8_KV_QUANTIZE, DSV4_ROPE_TAIL) can trip the imatrix
# collector. This gate catches regressions of the fix landed in
# feat/v4-port-I-imatrix.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"

V4_GGUF="${V4_GGUF:-$HOME/models/DeepSeek-V4-Flash-Q8_0.gguf}"
BIN="${LLAMA_IMATRIX_BIN:-$ROOT/build/bin/llama-imatrix}"
SAMPLE="$DIR/calibration/wikitext-tiny.txt"
OUT=$(mktemp -t imatrix-gate.XXXXXX.dat)

if [ ! -f "$V4_GGUF" ]; then
    echo "FAIL: V4 GGUF not found at $V4_GGUF"
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "FAIL: llama-imatrix not found at $BIN"
    exit 1
fi
if [ ! -f "$SAMPLE" ]; then
    echo "FAIL: calibration sample not found at $SAMPLE"
    exit 1
fi

trap 'rm -f "$OUT"' EXIT

echo "Running llama-imatrix against V4 (chunks=2)..."
"$BIN" \
    -m "$V4_GGUF" \
    -f "$SAMPLE" \
    -o "$OUT" \
    --chunks 2 \
    -ngl 999

if [ ! -s "$OUT" ]; then
    echo "FAIL: imatrix output file missing or empty: $OUT"
    exit 1
fi

# Minimum tensor count check. V4 has 43 layers x 5 attention + 3 expert
# = 8 tensor classes. At chunks=2 we expect coverage to be partial but
# not minimal: at least 100 distinct tensor entries indicates the fix
# isn't accidentally skipping whole tensor classes. The full per-class
# coverage check (~344 expected) lives in the Task 5 production run.
TENSOR_COUNT=$(strings "$OUT" | grep -cE '^blk\.[0-9]+\.' || true)
MIN_TENSORS=100
if [ "$TENSOR_COUNT" -lt "$MIN_TENSORS" ]; then
    echo "FAIL: imatrix output has only $TENSOR_COUNT tensors, expected >= $MIN_TENSORS"
    exit 1
fi

echo "PASS: gate-imatrix ($TENSOR_COUNT tensors collected)"
