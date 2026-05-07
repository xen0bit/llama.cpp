# V4 Imatrix Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose and fix the V4 imatrix segfault in `tools/imatrix/imatrix.cpp` so `llama-imatrix` runs end-to-end on V4 Q8 source. Produce a calibration `imatrix.dat` artifact and a regression gate so future llama.cpp upstream changes don't silently break V4 imatrix.

**Architecture:** V4's tool/op set (`LIGHTNING_INDEXER`, `DSV4_HC_*`, `DSV4_FP8_KV_QUANTIZE`) and integer lookup tensors (`ffn_gate_tid2eid.weight`, I32) likely trip imatrix's collector hook in ways that don't occur on V3-class models. The fix is the smallest patch that makes the collector skip whatever V4-specific tensor or op it doesn't understand, while still recording activations from all standard `MUL_MAT` and `MUL_MAT_ID` operations. Diagnose first via address-sanitizer; do not pre-commit to a fix strategy.

**Tech Stack:** llama.cpp C++ runtime + ggml; `tools/imatrix/imatrix.cpp` is the patch site; bash gate scripts under `tests/v4-port/`; ASan-enabled debug build for diagnosis; wikitext-103 for calibration corpus.

---

## Worktree / branch setup

This task runs in a worktree off `feat/v4-port` (parent branch on remote `mine`):

```bash
cd /Users/cchuter/work/llama.cpp
git fetch mine
git worktree add ../llama.cpp.I-imatrix -b feat/v4-port-I-imatrix mine/feat/v4-port
cd ../llama.cpp.I-imatrix
```

All steps below run from `../llama.cpp.I-imatrix`. Validation requires the V4 Q8 GGUF on disk:

```bash
export V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q8_0.gguf
ls -la "$V4_GGUF"   # must exist; ~282 GiB
```

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `tools/imatrix/imatrix.cpp` | Modify | Apply targeted skip / passthrough fix at the crash site identified by diagnosis |
| `tests/v4-port/gate-imatrix.sh` | Create | Regression gate: runs imatrix with `--chunks 2`, asserts no crash + minimum tensor count |
| `tests/v4-port/run-all-gates.sh` | Modify | Wire new gate into the standard suite |
| `tests/v4-port/calibration/imatrix-v4-flash.dat` | Create | Calibration data artifact for downstream IQ-quant builds (Task J) |
| `docs/plans/v4-port-imatrix-diagnosis.md` | Create | Diagnosis writeup: backtrace, root cause, why this fix and not others |

---

### Task 1: Build sanitized debug binary for diagnosis

**Files:**
- No source changes yet; cmake build only

- [ ] **Step 1: Configure ASan debug build**

```bash
cmake -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLAMA_SANITIZE_ADDRESS=ON \
  -DLLAMA_FATAL_WARNINGS=OFF
```

Expected: cmake configures without errors, prints "Generating done" at the end.

- [ ] **Step 2: Build llama-imatrix only (faster than full build)**

```bash
cmake --build build-asan --target llama-imatrix -j
```

Expected: build completes, produces `build-asan/bin/llama-imatrix`. ASan symbols make this binary slow but instrumented.

- [ ] **Step 3: Verify the binary runs against a non-V4 model first (sanity check)**

```bash
./build-asan/bin/llama-imatrix --help 2>&1 | head -20
```

Expected: prints usage, no segfault. Confirms the binary itself is sound.

- [ ] **Step 4: Commit the build configuration note**

```bash
git add -A docs/ 2>/dev/null || true
git commit --allow-empty -m "v4-port-I: ASan debug build configured for imatrix diagnosis"
```

---

### Task 2: Reproduce the segfault and capture backtrace

**Files:**
- Create: `docs/plans/v4-port-imatrix-diagnosis.md`

- [ ] **Step 1: Extract a small wikitext sample for reproduction**

Use a small extract from the same `wikitext-103-raw-v1` test split that Task 5's full calibration uses. Using wikitext (not synthetic text) ensures the diagnosis hits the same activation distribution as the production calibration; only the chunk count differs.

```bash
mkdir -p tests/v4-port/calibration

# Pull the test split parquet (same source as Task 5)
hf download wikitext --repo-type dataset \
  --include "wikitext-103-raw-v1/test-*.parquet" \
  --local-dir /tmp/wikitext-test/

# Extract first ~50 non-empty paragraphs
python3 <<'PYEOF'
import pandas as pd, glob
files = sorted(glob.glob('/tmp/wikitext-test/wikitext-103-raw-v1/test-*.parquet'))
out_path = 'tests/v4-port/calibration/wikitext-tiny.txt'
written = 0
with open(out_path, 'w') as out:
    for f in files:
        df = pd.read_parquet(f)
        for line in df['text']:
            if not line or not line.strip():
                continue
            out.write(line)
            written += 1
            if written >= 50:
                break
        if written >= 50:
            break
print(f'wrote {written} paragraphs to {out_path}')
PYEOF

ls -la tests/v4-port/calibration/wikitext-tiny.txt
```

Expected: file exists, ≥10 KB.

- [ ] **Step 2: Run llama-imatrix and capture full output**

```bash
./build-asan/bin/llama-imatrix \
  -m "$V4_GGUF" \
  -f tests/v4-port/calibration/wikitext-tiny.txt \
  --chunks 5 \
  -ngl 999 \
  2>&1 | tee /tmp/imatrix-asan.log
```

Expected: process crashes during first chunk's forward pass. ASan should print a "AddressSanitizer:" or "SEGV" report with stack trace before the process exits.

- [ ] **Step 3: Extract the actionable parts of the crash report**

```bash
# ASan crash header + backtrace + last log line before crash
grep -B1 -A40 "AddressSanitizer\|SEGV\|signal SIGSEGV" /tmp/imatrix-asan.log | head -80 \
  > /tmp/imatrix-crash.txt
echo "---"
tail -30 /tmp/imatrix-asan.log >> /tmp/imatrix-crash.txt
cat /tmp/imatrix-crash.txt
```

Expected: A backtrace pointing to a line in `tools/imatrix/imatrix.cpp` (or a callee from there), plus the V4 tensor name being processed at the time.

- [ ] **Step 4: Write the diagnosis writeup**

```bash
cat > docs/plans/v4-port-imatrix-diagnosis.md <<'EOF'
# V4 imatrix segfault diagnosis

## Reproducer
- Binary: build-asan/bin/llama-imatrix (ASan debug build of feat/v4-port HEAD)
- Model: ~/models/DeepSeek-V4-Flash-Q8_0.gguf
- Calibration: tests/v4-port/calibration/wikitext-tiny.txt (extracted from wikitext-103-raw-v1 test split)
- Chunks: 5
- NGL: 999

## Crash report

(paste content of /tmp/imatrix-crash.txt here)

## Root cause

(one paragraph: which line crashes, what was the immediate cause -
e.g. dereferencing a null pointer in the activation collector,
out-of-bounds access on an I32 tensor, missing case for a V4-specific op)

## Tensor at fault

(e.g. blk.0.ffn_gate_tid2eid.weight, type=I32, shape=[256, 1, 1, 1] -
or blk.0.attn_compressor_*, op=DSV4_HC_SPLIT_SINKHORN, etc.)

## Fix strategy chosen

(I32-passthrough OR op-class skip OR MUL_MAT_ID layout fix.
Pick exactly one. Justify in 1-2 sentences why this is the smallest
patch that satisfies the spec's coverage criterion.)

## Strategies rejected

(brief: why the other two strategies are wrong for this crash)
EOF
```

The implementer fills in the parenthesized sections from the crash report. **Do NOT continue to Task 3 until this writeup is complete and the fix strategy is chosen.**

- [ ] **Step 5: Commit the diagnosis**

```bash
git add docs/plans/v4-port-imatrix-diagnosis.md tests/v4-port/calibration/wikitext-tiny.txt
git commit -m "v4-port-I: diagnosis - imatrix segfault root cause + chosen fix strategy"
```

---

### Task 3: Apply the chosen fix

**Files:**
- Modify: `tools/imatrix/imatrix.cpp` (exact lines depend on diagnosis)

The diagnosis at `docs/plans/v4-port-imatrix-diagnosis.md` selected one of three fix strategies. Apply only that one. Reference implementations for each:

#### Strategy 1: I32-passthrough (most likely)

This mirrors our existing `src/llama-quant.cpp::tensor_allows_quantization` fix where I32 lookup tensors are skipped before any operations that assume floating-point types.

**Where:** `tools/imatrix/imatrix.cpp`, in the `IMatrixCollector::collect_imatrix` callback (around line 240, exact line per diagnosis). The check goes BEFORE any `t->src[N]` dereference or any code path that reads tensor data.

**Code (only apply if Strategy 1 selected):**

```cpp
// Skip integer lookup tensors (V4: ffn_gate_tid2eid is I32).
// imatrix collects floating-point activation statistics; integer
// tensors have no meaningful imatrix data and dereferencing them
// here will segfault.
if (t->type == GGML_TYPE_I8  ||
    t->type == GGML_TYPE_I16 ||
    t->type == GGML_TYPE_I32 ||
    t->type == GGML_TYPE_I64) {
    return false;
}
```

#### Strategy 2: Op-class skip

If diagnosis shows the crash is inside a V4-specific op (`LIGHTNING_INDEXER`, `DSV4_HC_*`, `DSV4_FP8_KV_QUANTIZE`, `DSV4_ROPE_TAIL`) when collector tries to read its output.

**Where:** `tools/imatrix/imatrix.cpp`, same callback, in the `t->op` dispatch (around lines 241-242).

**Code (only apply if Strategy 2 selected):**

```cpp
// imatrix records calibration statistics from matrix multiplications.
// V4-specific ops (lightning indexer, hyper-connection ops, etc.) do
// not consume weight tensors that need calibration; their outputs
// are not consumed by anything that benefits from imatrix data.
if (t->op == GGML_OP_MUL_MAT_ID) return true;
if (t->op != GGML_OP_MUL_MAT)    return false;
```

(This is what's already there in spirit; the patch may just be moving the check earlier or adding to existing logic. Confirm with diagnosis where the actual misbehavior is.)

#### Strategy 3: MUL_MAT_ID layout fix

If diagnosis shows the crash is inside the expert-routing unpacker at lines 263-306 with V4-specific shape mismatch.

**Where:** `tools/imatrix/imatrix.cpp`, the `if (t->op == GGML_OP_MUL_MAT_ID) { ... }` block.

This strategy is more involved. If it's selected, the implementer must read the existing block carefully and patch the indexing/loop logic to match V4's `ffn_gate_inp` shape. The diagnosis writeup must document the exact shape mismatch before this is attempted. **If Strategy 3 is selected and the patch grows beyond ~30 lines, stop and re-brainstorm — that's outside spec scope.**

- [ ] **Step 1: Apply the patch chosen by diagnosis**

Edit `tools/imatrix/imatrix.cpp` per the strategy above. The diff should be small (≤10 lines for Strategy 1 or 2; ≤30 for Strategy 3).

- [ ] **Step 2: Configure release `build/` if not already (fresh worktree may not have it)**

```bash
# Idempotent: skips if build/ is already configured.
if [ ! -f build/CMakeCache.txt ]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Release
fi
```

Expected: cmake configures or reports "Skipping" if already done.

- [ ] **Step 3: Rebuild llama-imatrix (release build, not ASan)**

```bash
cmake --build build --target llama-imatrix -j
```

Expected: build completes cleanly. Use the release `build/` not `build-asan/` from now on; ASan was just for diagnosis.

- [ ] **Step 4: Verify the fix unblocks the reproducer**

```bash
./build/bin/llama-imatrix \
  -m "$V4_GGUF" \
  -f tests/v4-port/calibration/wikitext-tiny.txt \
  -o /tmp/imatrix-test.dat \
  --chunks 2 \
  -ngl 999 \
  2>&1 | tail -20
```

Expected: completes without segfault, prints "save_imatrix" line, exits 0. The output `/tmp/imatrix-test.dat` exists.

- [ ] **Step 5: Commit the fix**

```bash
git add tools/imatrix/imatrix.cpp
git commit -m "v4-port-I: fix imatrix segfault on V4 (strategy: <one of 1/2/3 from diagnosis>)"
```

---

### Task 4: Add the regression gate

**Files:**
- Create: `tests/v4-port/gate-imatrix.sh`
- Modify: `tests/v4-port/run-all-gates.sh`

- [ ] **Step 1: Write the gate script**

```bash
cat > tests/v4-port/gate-imatrix.sh <<'EOF'
#!/usr/bin/env bash
# gate-imatrix: regression test that llama-imatrix runs end-to-end on
# V4 without crashing and collects activations from at least the
# expected minimum number of weight tensors.
#
# Background: V4's ffn_gate_tid2eid (I32) and V4-specific ops can
# trip the imatrix collector. This gate catches regressions of the
# fix landed in feat/v4-port-I-imatrix.

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

# Minimum tensor count check. V4 has 43 layers × 5 attention + 3 expert
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
EOF
chmod +x tests/v4-port/gate-imatrix.sh
```

- [ ] **Step 2: Run the gate to confirm it passes**

```bash
./tests/v4-port/gate-imatrix.sh
```

Expected: prints `PASS: gate-imatrix (N tensors collected)` where N >= 50.

- [ ] **Step 3: Wire the gate into `run-all-gates.sh`**

Read the current contents of `tests/v4-port/run-all-gates.sh` to find the right insertion point (just before the `echo "ALL GATES PASS"` line):

```bash
cat tests/v4-port/run-all-gates.sh
```

Expected output structure:
```bash
#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
"$DIR/gate-loader.sh"
NGL=0   "$DIR/gate-coherence.sh"
NGL=999 "$DIR/gate-coherence.sh"
"$DIR/gate-speed.sh"
"$DIR/gate-tools.sh"
"$DIR/gate-server-chat.sh"
"$DIR/gate-server-chat-q8.sh"
MODE=warn-fa-off "$DIR/gate-server-chat-q8.sh"
echo "ALL GATES PASS"
```

Insert the new gate before `echo "ALL GATES PASS"`:

```bash
# Use Edit-style replacement (or sed -i)
python3 <<'EOF'
import pathlib
p = pathlib.Path("tests/v4-port/run-all-gates.sh")
content = p.read_text()
new = content.replace(
    'echo "ALL GATES PASS"',
    '"$DIR/gate-imatrix.sh"\necho "ALL GATES PASS"'
)
assert new != content, "insertion point not found"
p.write_text(new)
EOF
```

- [ ] **Step 4: Run the full gate suite to confirm no regression**

```bash
./tests/v4-port/run-all-gates.sh
```

Expected: all gates pass (loader, coherence×2, speed, tools, server-chat, server-chat-q8 ×2, imatrix). Final line: `ALL GATES PASS`.

- [ ] **Step 5: Commit the gate**

```bash
git add tests/v4-port/gate-imatrix.sh tests/v4-port/run-all-gates.sh
git commit -m "v4-port-I: add gate-imatrix.sh regression gate, wire into run-all-gates"
```

---

### Task 5: Produce the calibration data artifact

**Files:**
- Create: `tests/v4-port/calibration/imatrix-v4-flash.dat`

This is the artifact Task J consumes. Use the standard wikitext-103 corpus.

- [ ] **Step 1: Source wikitext-103**

The implementer follows the existing llama.cpp imatrix workflow for sourcing wikitext-103. Two common paths:

```bash
# Option A: HF datasets (preferred)
mkdir -p tests/v4-port/calibration/wikitext
hf download wikitext --repo-type dataset \
  --include "wikitext-103-raw-v1/test-*.parquet" \
  --local-dir tests/v4-port/calibration/wikitext/

# Convert parquet to text
python3 -c "
import pandas as pd, glob
files = sorted(glob.glob('tests/v4-port/calibration/wikitext/wikitext-103-raw-v1/test-*.parquet'))
with open('tests/v4-port/calibration/wikitext.txt', 'w') as out:
    for f in files:
        df = pd.read_parquet(f)
        for line in df['text']:
            out.write(line)
"
```

If `pandas` is unavailable:

```bash
# Option B: well-known mirror used by other llama.cpp imatrix workflows
# (e.g. wget the canonical wikitext-103-raw-v1 from a mirror; the
# implementer picks whichever is available)
```

Expected output: `tests/v4-port/calibration/wikitext.txt` exists, ≥1MB.

- [ ] **Step 2: Run full imatrix calibration (1000 chunks)**

```bash
./build/bin/llama-imatrix \
  -m "$V4_GGUF" \
  -f tests/v4-port/calibration/wikitext.txt \
  -o tests/v4-port/calibration/imatrix-v4-flash.dat \
  --chunks 1000 \
  -ngl 999 \
  2>&1 | tee /tmp/imatrix-full.log
```

Expected: runs ~30-60 minutes on M3 Ultra, prints periodic chunk progress, exits 0. Final output: `tests/v4-port/calibration/imatrix-v4-flash.dat` (~few MB).

- [ ] **Step 3: Verify tensor coverage (per-class assertions)**

The spec's success criterion #1 requires coverage of all five attention projection classes (each appearing once per layer × 43 layers) and all three MoE expert tensor classes (each appearing once per layer × 43 layers; 256 experts is the internal dimension of each tensor, not a tensor multiplier).

This check enforces per-class minimums rather than just an aggregate count, so a fix that accidentally skips one whole tensor class (e.g. all `attn_kv`) still fails the check.

```bash
python3 <<'EOF'
import re, sys
content = open('tests/v4-port/calibration/imatrix-v4-flash.dat', 'rb').read()
unique = sorted({b.decode() for b in re.findall(rb'blk\.\d+\.[a-z_.]+', content)})

# Per-class expected: each tensor class should appear in ~all 43 layers.
# Allow a 5-layer slack for layers where the calibration corpus may not
# have exercised some path (rare but possible at long-context-only ops).
ATTN_CLASSES = ['attn_q_a', 'attn_q_b', 'attn_kv', 'attn_output_a', 'attn_output_b']
EXPERT_CLASSES = ['ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps']
PER_CLASS_MIN = 38   # 43 layers minus 5-layer slack

def count_class(cls):
    return sum(1 for t in unique if f'.{cls}.' in t or t.endswith(f'.{cls}'))

print('Per-class coverage (V4 has 43 layers; threshold ≥ %d each):' % PER_CLASS_MIN)
fail = False
for cls in ATTN_CLASSES + EXPERT_CLASSES:
    n = count_class(cls)
    status = 'PASS' if n >= PER_CLASS_MIN else 'FAIL'
    if n < PER_CLASS_MIN:
        fail = True
    print(f'  {cls:25s}  {n:3d}  {status}')

if fail:
    print('FAIL: at least one tensor class has insufficient layer coverage.')
    print('      The fix is skipping tensors that should be calibrated. Re-evaluate Task 3.')
    sys.exit(1)
print('PASS: per-class tensor coverage acceptable')
EOF
```

Expected: PASS for all 8 classes. If any class fails, the fix is too aggressive — it's skipping a tensor class that the calibration needs. Re-evaluate Task 3.

- [ ] **Step 4: Add the calibration data file to git (it's small)**

```bash
ls -lh tests/v4-port/calibration/imatrix-v4-flash.dat
# verify size: should be a few MB, not gigabytes
git add tests/v4-port/calibration/imatrix-v4-flash.dat
```

- [ ] **Step 5: Add wikitext source to .gitignore (large, transient)**

```bash
cat >> .gitignore <<'EOF'

# imatrix calibration source data (large, regenerable)
tests/v4-port/calibration/wikitext/
tests/v4-port/calibration/wikitext.txt
EOF
git add .gitignore
```

- [ ] **Step 6: Commit the calibration data**

```bash
git commit -m "v4-port-I: produce imatrix-v4-flash.dat from wikitext-103 (1000 chunks)"
```

---

### Task 6: Final integration check + push

**Files:**
- None modified

- [ ] **Step 1: Run the full gate suite one more time**

```bash
./tests/v4-port/run-all-gates.sh
```

Expected: `ALL GATES PASS`.

- [ ] **Step 2: Push the branch to remote**

```bash
git push -u mine feat/v4-port-I-imatrix
```

Expected: push succeeds. Branch URL printed.

- [ ] **Step 3: Mark task complete in dev-team task tracker**

```bash
mkdir -p tasks/done
mv tasks/v4-port-I-imatrix.json tasks/done/ 2>/dev/null || true
git add -A tasks/
git commit --allow-empty -m "v4-port-I: mark done; move task JSON to tasks/done/"
git push mine feat/v4-port-I-imatrix
```

---

## Definition of done

- [ ] `tools/imatrix/imatrix.cpp` patched with smallest viable fix per diagnosis
- [ ] `docs/plans/v4-port-imatrix-diagnosis.md` written, names exact crash site + chosen strategy + rejected alternatives
- [ ] `tests/v4-port/gate-imatrix.sh` created, executable, passes locally
- [ ] `tests/v4-port/run-all-gates.sh` includes the new gate, full suite passes
- [ ] `tests/v4-port/calibration/imatrix-v4-flash.dat` exists and committed (≥200 attention + ≥100 expert tensors covered)
- [ ] Branch `feat/v4-port-I-imatrix` pushed to `mine`
- [ ] Codex plan-review and code-review both APPROVE (per dev-team workflow)

## What this task explicitly does NOT do

- Build IQ1 quants — that's Task J's deliverable
- Validate IQ1 quality on agent benchmarks — out of scope per spec
- Refactor imatrix beyond the V4-specific crash fix
- Upstream the imatrix fix to ggml-org/llama.cpp (gated on V3.2 PR #21149)
