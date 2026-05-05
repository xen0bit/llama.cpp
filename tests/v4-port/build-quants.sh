#!/usr/bin/env bash
# Build a quant ladder for DeepSeek V4 Flash from the Q8_0 intermediate
# produced by Task H (`convert_hf_to_gguf.py --outtype q8_0`).
#
# Usage:
#   ./tests/v4-port/build-quants.sh                           # default ladder (see DEFAULT_TARGETS)
#   ./tests/v4-port/build-quants.sh Q4_K_M Q6_K               # specific targets
#   SRC=~/models/foo.gguf OUTDIR=~/quants ./build-quants.sh Q4_K_M
#
# Env overrides:
#   LLAMA_BIN     — path to the llama.cpp build dir (default: $REPO_ROOT/build)
#   SRC           — source GGUF (default: $HOME/models/DeepSeek-V4-Flash-Q8_0.gguf)
#   OUTDIR        — output directory (default: same dir as $SRC)
#   NAME_PREFIX   — output filename prefix (default: derived from SRC, replacing -Q8_0 with -<TARGET>)
#   SKIP_GATE     — set to 1 to skip the post-quant gate-loader smoke check

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA_BIN="${LLAMA_BIN:-$REPO_ROOT/build}"
SRC="${SRC:-$HOME/models/DeepSeek-V4-Flash-Q8_0.gguf}"
OUTDIR="${OUTDIR:-$(dirname "$SRC")}"
NAME_PREFIX="${NAME_PREFIX:-$(basename "$SRC" | sed -E 's/-(Q8_0|F16|BF16)\.gguf$//')}"

DEFAULT_TARGETS=(Q6_K Q5_K_M Q4_K_M IQ4_XS)

if [ "$#" -eq 0 ]; then
  TARGETS=("${DEFAULT_TARGETS[@]}")
else
  TARGETS=("$@")
fi

if [ ! -x "$LLAMA_BIN/bin/llama-quantize" ]; then
  echo "FAIL: llama-quantize not found at $LLAMA_BIN/bin/llama-quantize" >&2
  echo "Build llama.cpp first: cd $REPO_ROOT && cmake --build build -j" >&2
  exit 1
fi

if [ ! -f "$SRC" ]; then
  echo "FAIL: source GGUF not found at $SRC" >&2
  echo "Run Task H to produce the Q8_0 intermediate first, or set SRC=/path/to/source.gguf" >&2
  exit 1
fi

echo "Source:  $SRC ($(du -h "$SRC" | awk '{print $1}'))"
echo "Outdir:  $OUTDIR"
echo "Targets: ${TARGETS[*]}"
echo

mkdir -p "$OUTDIR"

declare -a results
for target in "${TARGETS[@]}"; do
  out="$OUTDIR/${NAME_PREFIX}-${target}.gguf"

  if [ -f "$out" ]; then
    echo "===> $target: skipping, already exists at $out"
    results+=("$target  EXISTS  $(du -h "$out" | awk '{print $1}')")
    continue
  fi

  echo "===> $target: $SRC -> $out"
  start=$(date +%s)
  "$LLAMA_BIN/bin/llama-quantize" "$SRC" "$out" "$target"
  end=$(date +%s)
  elapsed=$((end - start))
  size=$(du -h "$out" | awk '{print $1}')
  echo "     done in ${elapsed}s, size $size"

  if [ -z "${SKIP_GATE:-}" ]; then
    if ! V4_GGUF="$out" "$REPO_ROOT/tests/v4-port/gate-loader.sh" > /tmp/build-quants-gate-$target.log 2>&1; then
      echo "     FAIL: gate-loader failed for $target — see /tmp/build-quants-gate-$target.log" >&2
      results+=("$target  GATE-FAIL  $size")
      continue
    fi
    echo "     gate-loader: PASS"
  fi

  results+=("$target  OK  $size  ${elapsed}s")
done

echo
echo "=== Summary ==="
printf '%s\n' "${results[@]}"
