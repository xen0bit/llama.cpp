# V4 Q4_K_M Quants Implementation Plan (v3 — surgical antirez port)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Q4_K_M GGUF for `deepseek-ai/DeepSeek-V4-Flash` from base safetensors and validate it against the V4 gate suite. Land V4 support in `convert_hf_to_gguf.py` and `gguf-py/gguf/constants.py` by porting only the V4-additive code from antirez's fork — without disturbing the V3.2/DSA support that already lives on `feat/v4-port`.

**Architecture:** Antirez's fork (`antirez/main`) branched before V3.2/DSA landed in upstream/fairydreaming, so his diff *renames* DEEPSEEK32 → DEEPSEEK4 (and DeepseekV32Model → DeepseekV4Model) rather than adding alongside. A wholesale `git checkout antirez/main -- file` would silently delete our V3.2 support. v3 instead does a **surgical port**: read antirez's V4-additive code as a reference, manually add it to our files preserving V3.2, then verify both archs coexist. Conversion uses `--outtype q8_0` (the highest precision allowed by antirez's V4 expert path; FP4 routed experts cannot be written as f16).

**Tech Stack:** Python 3.13 (`convert_hf_to_gguf.py`, `gguf-py/gguf/`), llama.cpp C++ (`llama-quantize` already built), bash gates under `tests/v4-port/`.

**Why this is v3:** v1 attempted to design the converter from scratch (REVISE'd by codex twice for missing prerequisites and wrong tensor naming). v2 attempted wholesale `git checkout` of antirez's files (REVISE'd by codex once for two new fatal issues: the diff is non-additive so V3.2 would be silently deleted, AND `--outtype f16` is unreachable because V4's FP4 routed experts force a compact output type). v3 fixes both: surgical V4-only port + q8_0 intermediate.

---

## Worktree / branch setup

```bash
cd /Users/cchuter/work/llama.cpp
git fetch mine antirez
git worktree add ../llama.cpp.H-quants -b feat/v4-port-H-quants mine/feat/v4-port
cd ../llama.cpp.H-quants
cmake --build build -j 32   # rebuild against worktree HEAD
```

All steps run from `../llama.cpp.H-quants`.

---

## File structure

Files modified by this plan:

- **Modify** `gguf-py/gguf/constants.py` — surgical additions of V4-only code, preserving all V3.2 entries:
  - Add `MODEL_ARCH.DEEPSEEK4 = auto()` adjacent to `DEEPSEEK32`
  - Add V4 tensor enums (`ATTN_COMPRESSOR_*`, `INDEXER_COMPRESSOR_*`, `INDEXER_*`, `HC_ATTN_*`, `HC_FFN_*`, `OUTPUT_HC_*`)
  - Add `MODEL_ARCH_NAMES[MODEL_ARCH.DEEPSEEK4] = "deepseek4"` adjacent to V3.2
  - Add `MODEL_TENSORS[MODEL_ARCH.DEEPSEEK4]` entry adjacent to V3.2
  - Add V4 entries to `TENSOR_NAMES` adjacent to V3.2
  - Add V4 KV constants (`HASH_LAYER_COUNT`, `HYPER_CONNECTION_*`, `ATTENTION_INDEXER_*`, `ATTENTION_COMPRESS_*`, `ATTENTION_OUTPUT_*`, `NEXTN_PREDICT_LAYERS`)
- **Modify** `gguf-py/gguf/gguf_writer.py` (likely) — add V4 writer helpers (`add_attention_compress_ratios`, `add_attention_output_lora_rank`, `add_attention_output_group_count`, `add_attention_compress_rope_freq_base`, `add_hash_layer_count`, `add_hyper_connection_*`)
- **Modify** `convert_hf_to_gguf.py` — add `DeepseekV4Model` class adjacent to `DeepseekV32Model` (preserve V3.2 class). Also add any infrastructure antirez introduced (FP8/I8/FP4 dequant helpers, ftype-to-qtype mapping, `_strip_model_prefix`, `_skip_layer_tensor`, `_write_deepseek4_expert_tensors`, etc.)
- **Create** `docs/plans/v4-port-quants-completion.md` — followup with recipe + sha256s + perf
- **Build artifacts on disk (NOT committed):**
  - `~/models/DeepSeek-V4-Flash-Q8_0.gguf` (~290 GiB intermediate; **note: q8_0 not f16**)
  - `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` (~150 GiB primary deliverable)

---

## Conflict resolution rule

When antirez's converter and our `feat/v4-port` both touch the same line, antirez's "rename" is misleading because his branch lacks V3.2. Treat his V4 code as **additive over** our V3.2 baseline:

- **V4-specific code** (`DeepseekV4Model`, `MODEL_ARCH.DEEPSEEK4`, V4 tensor enums, V4 KV constants, V4 writer helpers) → **add alongside V3.2**, never replace
- **DSA / V3.2 / sparse-attention / lightning-indexer code** that already exists on `feat/v4-port` → **keep as-is**
- **Shared infrastructure** (e.g. `_qtype_aliases` table, FP8 helper functions antirez introduces) → **add antirez's; merge with any upstream additions**

After every port operation: explicitly verify BOTH `DEEPSEEK32` AND `DEEPSEEK4` coexist via Python import test (see Task 2 Step 4 and Task 3 Step 4).

---

## Task 1: Verify preconditions (fail-fast)

**Files:** none modified

- [ ] **Step 1: Confirm base weights cloned and complete**

```bash
ls ~/models/DeepSeek-V4-Flash/config.json
ls ~/models/DeepSeek-V4-Flash/*.safetensors | wc -l
find ~/models/DeepSeek-V4-Flash -name "*.lock" -o -name "*.tmp" 2>/dev/null
```

Expected: `config.json` present, ≥40 safetensors shards, no `.lock`/`.tmp`.

- [ ] **Step 2: Confirm cloned config.json arch class**

```bash
python3 -c "import json; c=json.load(open('$HOME/models/DeepSeek-V4-Flash/config.json')); print('architectures:', c['architectures']); print('model_type:', c['model_type'])"
```

Expected: `['DeepseekV4ForCausalLM']`, `deepseek_v4`.

- [ ] **Step 3: Confirm V3.2 entries currently present (must NOT be deleted by the port)**

```bash
grep -nE "DEEPSEEK32|DeepseekV32Model" gguf-py/gguf/constants.py convert_hf_to_gguf.py | head -10
```

Expected output includes at minimum:
- `gguf-py/gguf/constants.py:NNN: DEEPSEEK32 = auto()`
- `gguf-py/gguf/constants.py:NNN: MODEL_ARCH.DEEPSEEK32: "deepseek32",`
- `gguf-py/gguf/constants.py:NNN: MODEL_ARCH.DEEPSEEK32: [` (twice, in MODEL_TENSORS and TENSOR_NAMES)
- `convert_hf_to_gguf.py:NNN: class DeepseekV32Model(TextModel):`
- `convert_hf_to_gguf.py:NNN: model_arch = gguf.MODEL_ARCH.DEEPSEEK32`

Record the exact line numbers — we'll re-grep at the end of each port task to verify these survived.

- [ ] **Step 4: Confirm V4 entries currently absent**

```bash
python3 -c "import sys; sys.path.insert(0,'gguf-py'); import gguf; print('has DEEPSEEK4:', hasattr(gguf.MODEL_ARCH,'DEEPSEEK4'))"
```

Expected: `has DEEPSEEK4: False`.

- [ ] **Step 5: Confirm disk space (revised lower for q8_0 intermediate)**

```bash
df -h ~/models | tail -1
```

Required: ≥**500 GiB** free (290 GiB q8_0 intermediate + 150 GiB Q4_K_M + 60 GiB headroom). Lower than v2's 800 GiB requirement because we no longer build an f16 intermediate.

- [ ] **Step 6: Confirm antirez remote**

```bash
git remote get-url antirez
git fetch antirez
git log --oneline antirez/main | head -3
git show antirez/main:convert_hf_to_gguf.py | head -1 > /dev/null && echo "antirez convert_hf accessible"
git show antirez/main:gguf-py/gguf/constants.py | head -1 > /dev/null && echo "antirez constants accessible"
```

Expected: remote URL ends in `antirez/llama.cpp-deepseek-v4-flash.git`, both file accesses succeed.

- [ ] **Step 7: No commit (verification step only)**

---

## Task 2: Surgical port of antirez's gguf-py V4 additions

**Files:**
- Modify: `gguf-py/gguf/constants.py` (add V4 entries, preserve V3.2)
- Modify: `gguf-py/gguf/gguf_writer.py` (add V4 writer helpers if any)

**Why:** Foundation for Task 3. Without `MODEL_ARCH.DEEPSEEK4`, the converter class fails at import. Must NOT delete V3.2 entries.

- [ ] **Step 1: Save antirez's gguf-py files for reference (do NOT overlay)**

```bash
mkdir -p /tmp/v4-antirez-ref
git show antirez/main:gguf-py/gguf/constants.py    > /tmp/v4-antirez-ref/constants.py
git show antirez/main:gguf-py/gguf/gguf_writer.py  > /tmp/v4-antirez-ref/gguf_writer.py
```

These are reference copies. We will read them and manually add V4-specific code to our `feat/v4-port` files. Do NOT `git checkout` them.

- [ ] **Step 2: Identify antirez's V4-additive lines in `gguf-py/gguf/constants.py`**

```bash
# Find all lines mentioning DEEPSEEK4 or V4 tensor types in antirez's constants.py
grep -nE "DEEPSEEK4|ATTN_COMPRESSOR|INDEXER_COMPRESSOR|INDEXER_(K|PROJ|ATTN)|HC_ATTN|HC_FFN|OUTPUT_HC|HASH_LAYER|HYPER_CONNECTION|COMPRESS_RATIOS|COMPRESS_ROPE|OUTPUT_LORA_RANK|OUTPUT_GROUP_COUNT|NEXTN_PREDICT|deepseek4" /tmp/v4-antirez-ref/constants.py | head -80
```

This produces a "shopping list" of lines to add. Note their structure (enum entry, dict entry, list entry, etc.) and which section of `constants.py` each belongs in.

- [ ] **Step 3: Manually add V4-additive lines to our `gguf-py/gguf/constants.py`**

For each section antirez touched, add the V4 entry **adjacent to** the existing V3.2 entry. Concrete order:

1. **`MODEL_ARCH` enum** (around line 445): add `DEEPSEEK4 = auto()` immediately after `DEEPSEEK32 = auto()`
2. **`MODEL_TENSOR` enum** (around line 600+): add V4 tensor enum entries (`ATTN_COMPRESSOR_APE`, `ATTN_COMPRESSOR_KV`, `ATTN_COMPRESSOR_GATE`, `ATTN_COMPRESSOR_NORM`, `INDEXER_COMPRESSOR_*`, `INDEXER_K_NORM`, `INDEXER_PROJ`, `INDEXER_ATTN_K`, `INDEXER_ATTN_Q_B`, `HC_ATTN_BASE/FN/SCALE`, `HC_FFN_BASE/FN/SCALE`, `OUTPUT_HC_BASE/FN/SCALE`)
3. **`MODEL_ARCH_NAMES` dict** (around line 932): add `MODEL_ARCH.DEEPSEEK4: "deepseek4"` immediately after the `DEEPSEEK32` entry
4. **`MODEL_TENSORS` dict** (around line 2821): add `MODEL_ARCH.DEEPSEEK4: [...]` block with all V4 tensors. Copy the structure from antirez's reference file
5. **`TENSOR_NAMES` dict** (around line 3990): add the V4 mapping block adjacent to V3.2's
6. **GGUF KV key constants** (search `class Keys`): add V4-specific keys antirez added — `HASH_LAYER_COUNT`, `HYPER_CONNECTION_COUNT/SINKHORN_ITERS/EPS`, `ATTENTION_INDEXER_HEAD_COUNT/KEY_LENGTH/TOP_K`, `ATTENTION_COMPRESS_RATIOS`, `ATTENTION_COMPRESS_ROPE_FREQ_BASE`, `ATTENTION_OUTPUT_LORA_RANK`, `ATTENTION_OUTPUT_GROUP_COUNT`, `NEXTN_PREDICT_LAYERS`. Place adjacent to existing similar constants.

After each block, save and run a parse check:

```bash
python3 -c "import ast; ast.parse(open('gguf-py/gguf/constants.py').read()); print('parses')"
```

- [ ] **Step 4: Verify both DEEPSEEK32 AND DEEPSEEK4 coexist after the additions**

```bash
python3 - <<'PY'
import sys; sys.path.insert(0,'gguf-py')
import gguf
checks = [
    ('DEEPSEEK32 enum', hasattr(gguf.MODEL_ARCH,'DEEPSEEK32')),
    ('DEEPSEEK4 enum',  hasattr(gguf.MODEL_ARCH,'DEEPSEEK4')),
    ('DEEPSEEK32 named', gguf.MODEL_ARCH_NAMES.get(gguf.MODEL_ARCH.DEEPSEEK32) == 'deepseek32'),
    ('DEEPSEEK4 named',  gguf.MODEL_ARCH_NAMES.get(gguf.MODEL_ARCH.DEEPSEEK4)  == 'deepseek4'),
    ('DEEPSEEK32 in MODEL_TENSORS', gguf.MODEL_ARCH.DEEPSEEK32 in gguf.MODEL_TENSORS),
    ('DEEPSEEK4 in MODEL_TENSORS',  gguf.MODEL_ARCH.DEEPSEEK4  in gguf.MODEL_TENSORS),
    ('ATTN_COMPRESSOR_APE',  hasattr(gguf.MODEL_TENSOR,'ATTN_COMPRESSOR_APE')),
    ('INDEXER_COMPRESSOR_KV', hasattr(gguf.MODEL_TENSOR,'INDEXER_COMPRESSOR_KV')),
    ('HC_ATTN_BASE',         hasattr(gguf.MODEL_TENSOR,'HC_ATTN_BASE')),
    ('OUTPUT_HC_BASE',       hasattr(gguf.MODEL_TENSOR,'OUTPUT_HC_BASE')),
]
for name, ok in checks:
    print(f"  {'OK' if ok else 'MISSING'}  {name}")
assert all(ok for _, ok in checks), 'V4 port incomplete OR V3.2 was clobbered'
print('OK: both V3.2 and V4 enums coexist')
PY
```

Expected: all `OK`. If any V3.2 check fails, the port deleted V3.2 code — back out and try again.

- [ ] **Step 5: Add V4 writer helpers to `gguf_writer.py` (if missing)**

```bash
python3 - <<'PY'
import sys; sys.path.insert(0,'gguf-py')
import gguf
methods = [
    'add_attention_compress_ratios',
    'add_attention_compress_rope_freq_base',
    'add_attention_output_lora_rank',
    'add_attention_output_group_count',
    'add_hash_layer_count',
    'add_hyper_connection_count',
    'add_hyper_connection_sinkhorn_iters',
    'add_hyper_connection_eps',
]
w = gguf.GGUFWriter.__dict__
for m in methods:
    print(f"  {'OK' if m in w else 'MISSING'}  {m}")
PY
```

For any `MISSING`: read the corresponding implementation in `/tmp/v4-antirez-ref/gguf_writer.py` and add it to our `gguf-py/gguf/gguf_writer.py`. These are typically thin wrappers around `add_uint32` / `add_array` / `add_float32`, e.g.:

```python
def add_attention_compress_ratios(self, value: list[int]) -> None:
    self.add_array(Keys.Attention.COMPRESS_RATIOS.format(arch=self.arch), value)
```

After adding, re-run the verification.

- [ ] **Step 6: Final coexistence test**

```bash
python3 - <<'PY'
import sys; sys.path.insert(0,'gguf-py')
import gguf
print('DEEPSEEK32 named:', gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.DEEPSEEK32])
print('DEEPSEEK4 named: ', gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.DEEPSEEK4])
print('DEEPSEEK32 tensors:', len(gguf.MODEL_TENSORS[gguf.MODEL_ARCH.DEEPSEEK32]))
print('DEEPSEEK4 tensors: ', len(gguf.MODEL_TENSORS[gguf.MODEL_ARCH.DEEPSEEK4]))
PY
```

Both must print non-zero tensor counts.

- [ ] **Step 7: Commit**

```bash
git add gguf-py/gguf/constants.py
git add gguf-py/gguf/gguf_writer.py 2>/dev/null || true
git commit -m "v4-port-H: add gguf-py V4 enums, KV constants, writer helpers (V3.2 preserved)"
```

---

## Task 3: Surgical port of antirez's `DeepseekV4Model` class

**Files:**
- Modify: `convert_hf_to_gguf.py` (add `DeepseekV4Model` adjacent to `DeepseekV32Model`)

**Why:** The actual converter for V4. Must add as a sibling class, not replace V3.2.

- [ ] **Step 1: Save antirez's converter for reference**

```bash
git show antirez/main:convert_hf_to_gguf.py > /tmp/v4-antirez-ref/convert_hf_to_gguf.py
```

- [ ] **Step 2: Identify the full antirez V4-additive code surface**

```bash
# DeepseekV4Model class boundaries
grep -nE "^class DeepseekV4Model|^@ModelBase\.register.*DeepseekV4|^class [A-Z]" /tmp/v4-antirez-ref/convert_hf_to_gguf.py | grep -A1 "DeepseekV4" | head -10

# Top-of-file imports antirez added
diff <(head -50 convert_hf_to_gguf.py) <(head -50 /tmp/v4-antirez-ref/convert_hf_to_gguf.py) | head -30

# Helper functions/methods antirez added outside DeepseekV4Model (e.g. FP4 dequant infrastructure)
git diff feat/v4-port..antirez/main -- convert_hf_to_gguf.py | grep -E "^[+]def |^[+]class |^[+]@" | head -30
```

This produces three lists: (a) class boundaries, (b) new imports to add, (c) helper functions outside the class. The architect should plan around all three categories — antirez's class depends on infrastructure that lives outside it.

- [ ] **Step 3: Add V4-additive code to our `convert_hf_to_gguf.py`**

Concrete order:

1. **Imports** (top of file): merge antirez's additions with our existing imports. Common additions: `ctypes`, possibly extra typing/numpy helpers. Do NOT remove imports we already have.

2. **Helper functions / utilities** added by antirez outside `DeepseekV4Model`: read the diff for `+def ` and `+class ` lines outside the class block; add the new helpers preserving any existing function definitions. If a helper has the same name as one we already have, **DO NOT replace** — examine why; either rename antirez's or skip if the existing version works.

3. **`DeepseekV4Model` class** (around our line 9221, immediately after `DeepseekV32Model` closes): paste antirez's class wholesale, preserving the `@ModelBase.register("DeepseekV4ForCausalLM")` decorator. The class includes ~245 lines: `__init__`, `set_vocab`, `set_gguf_parameters`, `modify_tensors`, expert handling (`_write_deepseek4_expert_tensors`, `_qtype_for_ftype`, `_parse_expert_outtype_spec`, `_strip_model_prefix`, `_skip_layer_tensor`), and the FP4 expert decode lookup table.

After the class is added, save and parse-check:

```bash
python3 -c "import ast; ast.parse(open('convert_hf_to_gguf.py').read()); print('parses')"
```

- [ ] **Step 4: Verify both DeepseekV32Model AND DeepseekV4Model are registered**

```bash
python3 - <<'PY'
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'gguf-py')
import convert_hf_to_gguf as m
from convert_hf_to_gguf import ModelBase
v32 = ModelBase.from_model_architecture("DeepseekV32ForCausalLM")
v4  = ModelBase.from_model_architecture("DeepseekV4ForCausalLM")
print(f"V3.2 class: {v32.__name__} (model_arch={v32.model_arch})")
print(f"V4 class:   {v4.__name__} (model_arch={v4.model_arch})")
assert v32.__name__ == 'DeepseekV32Model', 'V3.2 class lost'
assert v4.__name__  == 'DeepseekV4Model',  'V4 class missing'
print('OK: both V3.2 and V4 model classes registered')
PY
```

Expected: both classes present and distinct.

- [ ] **Step 5: Verify our existing V3.2 architecture path still works**

```bash
# Quick spot-check: V3.2 entries still grep-able at the same line counts (give or take)
grep -nE "DeepseekV32Model|MODEL_ARCH.DEEPSEEK32" convert_hf_to_gguf.py | head -5
```

Expected: similar to Task 1 Step 3's output (line numbers may have shifted by the added V4 code, but the entries must exist).

- [ ] **Step 6: Commit**

```bash
git add convert_hf_to_gguf.py
git commit -m "v4-port-H: add DeepseekV4Model converter class (FP8/I8/FP4 dequant + V4 tensor remap; V3.2 class preserved)"
```

---

## Task 4: Smoke-test the converter against the cloned model

**Files:** none modified (read-only)

**Why:** Validate the port by running just the metadata-emit path without committing to the long conversion.

- [ ] **Step 1: Walk the cloned safetensors patterns and verify antirez's class handles each**

```bash
python3 - <<'PY'
import struct, json, glob, os, re
shards = sorted(glob.glob(os.path.expanduser('~/models/DeepSeek-V4-Flash/*.safetensors')))
patterns = set()
for shard in shards[:6]:
    with open(shard, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    for k in h:
        if k == '__metadata__': continue
        patterns.add(re.sub(r'\.\d+\.', '.N.', k))
for p in sorted(patterns): print(p)
PY
```

Compare the printed list against `DeepseekV4Model.modify_tensors` and `_write_deepseek4_expert_tensors`. Every pattern should map to a GGUF tensor, be consumed as an expert weight/scale, or be explicitly skipped (e.g. `*.scale` companions). If a pattern is unhandled, codex code-review will catch it; for this step, just record the inventory for later reference.

- [ ] **Step 2: Run the converter for ONE layer in dry-mode (uses --deepseek4-max-layers)**

Antirez's class supports `--deepseek4-max-layers=N` for partial conversion. Use it for a fast smoke test:

```bash
python3 convert_hf_to_gguf.py \
  ~/models/DeepSeek-V4-Flash \
  --outfile /tmp/v4-smoke.gguf \
  --outtype q8_0 \
  --deepseek4-max-layers 2 \
  2>&1 | tee /tmp/v4-smoke-convert.log
```

Expected: clean run in ~1-2 min, file at `/tmp/v4-smoke.gguf` (~10-15 GiB). If errors occur, they're visible immediately without 30+ min of compute wasted.

If `--deepseek4-max-layers` isn't recognized, antirez's CLI plumbing didn't get ported in Task 3 — check the imports / argparse setup.

- [ ] **Step 3: Smoke-load via gate-loader**

```bash
V4_GGUF=/tmp/v4-smoke.gguf ./tests/v4-port/gate-loader.sh
```

Expected: `PASS: loader recognizes V4 GGUF`. If FAIL with "tensor count mismatch" or "missing kv key", the converter has a bug — fix at the converter level before proceeding to full conversion.

- [ ] **Step 4: Clean up smoke artifact**

```bash
rm /tmp/v4-smoke.gguf
```

- [ ] **Step 5: No commit (verification step)**

---

## Task 5: Run the full Q8_0 conversion

**Files:** none modified (produces `~/models/DeepSeek-V4-Flash-Q8_0.gguf`)

**IMPORTANT:** `--outtype q8_0`, NOT `f16`. V4's FP4 routed expert weights cannot be written as f16 — antirez's `_write_deepseek4_expert_tensors` raises `NotImplementedError` unless `--outtype` is one of `iq2_xxs, iq2_xs, q2_k, tq2_0, tq1_0, q8_0`. Q8_0 is the highest-precision allowed option.

- [ ] **Step 1: Run the full conversion**

```bash
python3 convert_hf_to_gguf.py \
  ~/models/DeepSeek-V4-Flash \
  --outfile ~/models/DeepSeek-V4-Flash-Q8_0.gguf \
  --outtype q8_0 \
  2>&1 | tee /tmp/v4-convert.log
```

Wall-clock: ~30-60 min on the M3 Ultra.

- [ ] **Step 2: Verify artifact**

```bash
ls -la ~/models/DeepSeek-V4-Flash-Q8_0.gguf
du -sh ~/models/DeepSeek-V4-Flash-Q8_0.gguf
```

Expected: ~290 GiB.

- [ ] **Step 3: Smoke-load via gate-loader**

```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q8_0.gguf ./tests/v4-port/gate-loader.sh
```

Expected: `PASS: loader recognizes V4 GGUF`.

- [ ] **Step 4: No commit (artifact step)**

---

## Task 6: Run llama-quantize Q8_0 → Q4_K_M

**Files:** none modified (produces `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf`)

`llama-quantize` accepts any GGUF as source and dequantizes block-by-block to fp32 internally before re-quantizing to the target. Q8_0 source → Q4_K_M target is a standard supported path.

- [ ] **Step 1: Run quantization**

```bash
./build/bin/llama-quantize \
  ~/models/DeepSeek-V4-Flash-Q8_0.gguf \
  ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  Q4_K_M \
  2>&1 | tee /tmp/v4-quantize.log
```

Wall-clock: ~30-90 min.

- [ ] **Step 2: Verify artifact**

```bash
ls -la ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf
du -sh ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf
```

Expected: ~150 GiB.

- [ ] **Step 3: Capture sha256 of both artifacts**

```bash
shasum -a 256 ~/models/DeepSeek-V4-Flash-Q8_0.gguf > /tmp/v4-sha256.txt
shasum -a 256 ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf >> /tmp/v4-sha256.txt
cat /tmp/v4-sha256.txt
```

- [ ] **Step 4: No commit**

---

## Task 7: Validate Q4_K_M via the V4 gate suite

**Files:** none modified

- [ ] **Step 1: Run all gates against Q4_K_M**

```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  ./tests/v4-port/run-all-gates.sh 2>&1 | tee /tmp/v4-gates.log
```

Expected: `ALL GATES PASS`. Specifically:
- `gate-loader`: `PASS: loader recognizes V4 GGUF`
- `gate-coherence` (NGL=0 and NGL=999): `PASS`
- `gate-speed`: `PASS: speed (NGL=999, ~XX tok/s)` — record the number
- `gate-tools`: `PASS: tool calling (5/5 with tool_calls)`
- `gate-server-chat`: `PASS: server-chat (3/3 tests coherent)`

(Task G's `gate-server-chat-q8.sh` is not in the suite yet — that's Task G's deliverable, scheduled after H merges.)

- [ ] **Step 2: Capture decode tok/s**

```bash
grep -E "Decode tok/s|gate-speed" /tmp/v4-gates.log
```

Record for the completion doc. IQ2XXS baseline: 25.91 tok/s.

- [ ] **Step 3: If any gate FAILed**

Likely causes (in order of probability):
1. Conflict-resolution bug in Tasks 2-3 (V3.2 entry accidentally shifted, or a V4 entry placed in the wrong section)
2. Helper function missing from Task 3 Step 3
3. Q4_K_M re-quantization issue (rare; report to user if all else checks out)

Compare against IQ2XXS to localize:

```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  ./tests/v4-port/<failing-gate>.sh
```

If IQ2XXS still passes, the converter or quantize is at fault — inspect the new GGUF's metadata in the gate's loader log against the IQ2XXS reference.

- [ ] **Step 4: No commit (verification)**

---

## Task 8: Write the completion doc

**Files:**
- Create: `docs/plans/v4-port-quants-completion.md`

- [ ] **Step 1: Write the doc**

```markdown
# V4 Port — Q4_K_M Quants Completion

Follow-up to `docs/superpowers/specs/2026-05-05-v4-quants-kvcache-design.md`
(Task H). Closes the deferred "build our own quants" item.

## TL;DR

`convert_hf_to_gguf.py` now recognizes `DeepseekV4ForCausalLM`. A Q4_K_M
GGUF is at `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` (~150 GiB) and passes
the V4 gate suite.

The converter is a surgical port of antirez's `DeepseekV4Model` from
`antirez/llama.cpp-deepseek-v4-flash`, added alongside our existing V3.2
support without disturbing it.

## Recipe (reproducible)

```bash
# 1. Q8_0 conversion (NOT f16 — V4 FP4 routed experts force a compact type)
python3 convert_hf_to_gguf.py ~/models/DeepSeek-V4-Flash \
  --outfile ~/models/DeepSeek-V4-Flash-Q8_0.gguf --outtype q8_0

# 2. Q4_K_M re-quantization
./build/bin/llama-quantize \
  ~/models/DeepSeek-V4-Flash-Q8_0.gguf \
  ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  Q4_K_M

# 3. Validate
V4_GGUF=~/models/DeepSeek-V4-Flash-Q4_K_M.gguf ./tests/v4-port/run-all-gates.sh
```

## Artifact identities

```
[paste contents of /tmp/v4-sha256.txt]
```

Sizes:
- `DeepSeek-V4-Flash-Q8_0.gguf`: [paste from `du -sh`] (~290 GiB nominal)
- `DeepSeek-V4-Flash-Q4_K_M.gguf`: [paste from `du -sh`] (~150 GiB nominal)

## Performance vs. IQ2XXS

| Metric | IQ2XXS (baseline) | Q4_K_M (this) |
|---|---|---|
| BPW | 2.44 | ~4.5 |
| File size | ~80 GiB | [paste] |
| Decode tok/s (NGL=999) | 25.91 | [paste from gate-speed] |

Targeted terminal-bench retry on Q4_K_M is recommended as a separate
followup; agentic validation is out-of-scope for this task per spec.

## What landed

- `gguf-py/gguf/constants.py`: V4 enums (`MODEL_ARCH.DEEPSEEK4`, V4 tensor
  enums, V4 KV constants), V4 entries in `MODEL_ARCH_NAMES` /
  `MODEL_TENSORS` / `TENSOR_NAMES`. Added alongside V3.2.
- `gguf-py/gguf/gguf_writer.py`: V4 writer helpers
  (`add_attention_compress_ratios`, `add_attention_output_lora_rank`, etc.)
- `convert_hf_to_gguf.py`: `DeepseekV4Model` class (~245 lines) with FP8
  e4m3 / I8 / FP4 dequantization, V4 tensor remap, expert handling. Added
  alongside `DeepseekV32Model` (V3.2 untouched).

## Why q8_0 intermediate (not f16)

V4 base safetensors store routed experts as FP4 with FP8 e8m0 scales.
Antirez's converter raises `NotImplementedError` if FP4 routed experts
are present and `--outtype` is not one of {iq2_xxs, iq2_xs, q2_k, tq2_0,
tq1_0, q8_0}. Q8_0 is the highest-precision allowed option, and
`llama-quantize` cleanly re-quantizes Q8_0 → Q4_K_M (block-by-block
dequant to fp32, then re-quant to target).

## Conflict resolution notes

[Document any non-trivial conflicts encountered during Tasks 2-3, e.g.
"helper function `_strip_model_prefix` collided with our existing
`_strip_prefix` — renamed antirez's to `_strip_dsv4_prefix`." If no
conflicts: write "Trivial port — antirez's V4 code is fully additive over
our V3.2 baseline once placed in the correct sections."]

## What is NOT included

- Other quant levels (Q6_K, Q5_K_M, IQ4_XS). Trivial follow-ups: re-quantize
  from `~/models/DeepSeek-V4-Flash-Q8_0.gguf` with `llama-quantize`.
- Imatrix calibration.
- terminal-bench end-to-end validation under Q4_K_M.
```

- [ ] **Step 2: Fill in placeholders**

Replace `[paste ...]` markers with actual values.

- [ ] **Step 3: Commit**

```bash
git add docs/plans/v4-port-quants-completion.md
git commit -m "v4-port-H: completion doc — Q4_K_M GGUF + recipe + sha256 + perf"
```

---

## Task 9: Push to mine

- [ ] **Step 1: Push the feature branch**

```bash
git push mine feat/v4-port-H-quants
```

- [ ] **Step 2: Confirm DoD is met**

- [x] `convert_hf_to_gguf.py` recognizes `DeepseekV4ForCausalLM` (and `DeepseekV32ForCausalLM` still works)
- [x] `gguf.MODEL_ARCH.DEEPSEEK4` and all V4 tensor enums present (and `DEEPSEEK32` still present)
- [x] `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` exists, valid (passes `gate-loader.sh`)
- [x] `run-all-gates.sh` (loader, coherence×2, speed, tools, server-chat) all PASS against Q4_K_M
- [x] Decode tok/s recorded
- [x] Followup writeup committed
- [x] Branch pushed to `mine`
- [ ] Codex plan-review and code-review APPROVE — handled by the dev-team orchestrator

This plan ends here. The orchestrator will run codex code-review and, on APPROVE, mark `tasks/active/v4-port-H-quants.json` state="done" and move it to `tasks/done/`.
