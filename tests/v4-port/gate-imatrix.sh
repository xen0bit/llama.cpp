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

# --no-repack: defensively disabled here for the same reason it's
# disabled in gate-coherence.sh — when V4's Q8 layers spill CPU-side
# (full GPU offload but VRAM tight, or test machines without GPU),
# the loader allocates CPU_Mapped + CPU_REPACK and the source mmap
# is not released, doubling resident memory and OOM-killing the
# process on a 512 GB host. With -ngl 999 + sufficient GPU VRAM
# the flag is a no-op; on machines that fall back to CPU layers
# it's the difference between PASS and SIGKILL.
echo "Running llama-imatrix against V4 (chunks=2)..."
"$BIN" \
    -m "$V4_GGUF" \
    -f "$SAMPLE" \
    -o "$OUT" \
    --chunks 2 \
    -ngl 999 \
    --no-repack

if [ ! -s "$OUT" ]; then
    echo "FAIL: imatrix output file missing or empty: $OUT"
    exit 1
fi

# Per-class coverage check. V4 has 43 layers; each weight tensor class
# (5 attention projections + 3 expert MoE matrices = 8 classes) should
# appear in most layers even at chunks=2. An aggregate count alone
# could pass a degraded fix that skips an entire class (e.g. all
# attention or all expert tensors) as long as enough names from other
# classes remain — the check below catches that by requiring each
# of the 8 classes to appear in at least PER_CLASS_MIN distinct layers.
#
# Threshold rationale: at chunks=2 (the wikitext-tiny.txt calibration
# the gate uses) we don't expect every layer to fire on every class,
# but the fix should produce broad enough coverage that each class
# appears in at least 25 of 43 layers. The exhaustive Task 5 check
# (1000 chunks, threshold 38/43 per class) lives in the plan; this
# gate is the cheaper smoke test that still catches whole-class drops.
python3 - "$OUT" <<'PYEOF'
import re, sys
content = open(sys.argv[1], 'rb').read()
# Capture (layer, class) pairs so we can count DISTINCT layers per class.
# A naive set of tensor-name strings would conflate e.g. "attn_q_b" with
# "attn_q_b.weight" / "attn_q_b.bias" (multiple tensor names per layer)
# and inflate the count well past the 43-layer ceiling.
pairs = set()
for m in re.finditer(rb'blk\.(\d+)\.([a-z_.]+)', content):
    layer = int(m.group(1))
    rest = m.group(2).decode()
    pairs.add((layer, rest))

ATTN_CLASSES = ['attn_q_a', 'attn_q_b', 'attn_kv', 'attn_output_a', 'attn_output_b']
EXPERT_CLASSES = ['ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps']
PER_CLASS_MIN = 25  # of 43 layers

def layers_for_class(cls):
    return {layer for (layer, rest) in pairs
            if rest == cls or rest.startswith(f'{cls}.')}

print(f'Per-class layer coverage (V4 has 43 layers; threshold >= {PER_CLASS_MIN}):')
fail = False
for cls in ATTN_CLASSES + EXPERT_CLASSES:
    layers = layers_for_class(cls)
    n = len(layers)
    status = 'PASS' if n >= PER_CLASS_MIN else 'FAIL'
    if n < PER_CLASS_MIN:
        fail = True
    print(f'  {cls:25s} {n:3d} layers  {status}')

if fail:
    print('FAIL: at least one tensor class has insufficient layer coverage.')
    print('      The imatrix fix is skipping tensors that should be calibrated.')
    sys.exit(1)
print(f'PASS: gate-imatrix ({len(pairs)} distinct (layer, tensor) pairs, all 8 classes >= {PER_CLASS_MIN} layers)')
PYEOF
