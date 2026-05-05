# V4 Q4_K_M Quants Implementation Plan (v2 — antirez port)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Q4_K_M GGUF for `deepseek-ai/DeepSeek-V4-Flash` from base safetensors and validate it against the existing V4 gate suite. Land V4 support in `convert_hf_to_gguf.py` and `gguf-py/gguf/constants.py` by **porting antirez's existing V4 converter**, the same way we ported his runtime in Tasks A–F.

**Architecture:** antirez's fork (`antirez/main`) already contains a complete, working V4 converter that produced the IQ2XXS GGUF we've been using. Port his `DeepseekV4Model` class (~245 lines) plus the V4 enums and writer helper methods in `gguf-py/gguf/constants.py` (~235 lines of diff). Resolve conflicts with upstream evolutions (V4-port already lives on top of upstream + fairydreaming/dsa, so the conflict surface mirrors the runtime port). Then run convert → quantize → validate.

**Tech Stack:** Python 3.13 (`convert_hf_to_gguf.py`, `gguf-py/gguf/`), llama.cpp C++ (`llama-quantize` already built), bash gates under `tests/v4-port/`, `~/models/` filesystem layout for artifacts.

**Why this is v2:** v1 of this plan was REVISE'd twice by codex (high reasoning effort) at the plan-review gate. v1 attempted to design the converter from scratch based on inferred tensor mappings; codex found 4 fatal defects (gguf-py prerequisites missing, wrong GGUFWriter API, wrong KV key names, wrong HF tensor naming, missing required tensors). The v2 plan replaces that bottom-up design with a top-down port of antirez's already-working code, eliminating the entire design-from-scratch error class.

---

## Worktree / branch setup

This task runs first (Task G follows after H merges). Worktree off `feat/v4-port`:

```bash
cd /Users/cchuter/work/llama.cpp
git fetch mine antirez
git worktree add ../llama.cpp.H-quants -b feat/v4-port-H-quants mine/feat/v4-port
cd ../llama.cpp.H-quants
cmake --build build -j 32   # rebuild against worktree HEAD; needed only if any C++ changes land
```

All steps below run from `../llama.cpp.H-quants`. The base weights:

```bash
ls -la ~/models/DeepSeek-V4-Flash/config.json   # must exist; clone complete
```

---

## File structure

Files created or modified by this plan:

- **Modify** `gguf-py/gguf/constants.py` — port antirez's V4 additions (~235 lines):
  - `MODEL_ARCH.DEEPSEEK4` enum entry
  - `MODEL_TENSOR.{ATTN_COMPRESSOR_*, INDEXER_COMPRESSOR_*, INDEXER_*, HC_ATTN_*, HC_FFN_*, OUTPUT_HC_*}` entries
  - V4 KV constants (`HASH_LAYER_COUNT`, `HYPER_CONNECTION_*`, `ATTENTION_INDEXER_*`, `ATTENTION_COMPRESS_*`, `ATTENTION_OUTPUT_*`, `NEXTN_PREDICT_LAYERS`, etc.)
  - `MODEL_ARCH_NAMES[MODEL_ARCH.DEEPSEEK4] = "deepseek4"`
  - V4 entries in `MODEL_TENSORS[MODEL_ARCH.DEEPSEEK4]`
  - V4 entries in `TENSOR_NAMES`
  - GGUFWriter helper methods (e.g. `add_attention_compress_ratios`, `add_attention_output_lora_rank`, `add_attention_output_group_count`, `add_attention_compress_rope_freq_base`, `add_hash_layer_count`, `add_hyper_connection_*`)
- **Modify** `convert_hf_to_gguf.py` — port antirez's `DeepseekV4Model` class (~245 lines + minor infrastructure: ~1082 lines of diff total, but most of that is the class body and FP8 dequant helpers). Includes:
  - `@ModelBase.register("DeepseekV4ForCausalLM")` registration
  - FP8 e4m3 + I8 dequantization for the cloned safetensors (weights are FP8/I8 with FP8 e8m0 scales — antirez's class already handles this)
  - V4-specific `set_gguf_parameters` and `modify_tensors`
  - V4 expert handling (256 experts × 6 active)
  - Hyper-connection tensor remapping
  - Attention compressor / indexer compressor remapping
- **Create** `docs/plans/v4-port-quants-completion.md` — followup with recipe, sha256s, perf numbers.
- **Build artifacts on disk (NOT committed):**
  - `~/models/DeepSeek-V4-Flash-F16.gguf` (~570 GiB intermediate)
  - `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` (~150 GiB primary deliverable)

`tests/v4-port/run-all-gates.sh` and the gate scripts already honor the `V4_GGUF` env var; no modification expected.

---

## Conflict resolution rule (mirrors Task A's rule)

When antirez's converter and our `feat/v4-port` (which already includes upstream + fairydreaming evolutions) both touched the same line:

- **V4-specific code** (`DeepseekV4Model` class, V4 tensor enums, V4 KV constants, V4 writer helpers) → **keep antirez's**
- **DSA / V3.2 / sparse-attention / lightning-indexer code** in surrounding context (e.g. `DeepseekV32Model` class, V3.2 KV constants) → **keep our `feat/v4-port`'s**, which already has the fairydreaming-derived V3.2 work
- **Shared infrastructure** (e.g. `GGUFWriter` class additions where antirez added `add_attention_compress_ratios` adjacent to upstream additions for other archs) → **MERGE BOTH** so all writer methods coexist

This is the same rule the runtime port (Task A-loader) used. The architect's plan-review-loop noticed two HIGH-correctness issues in antirez's runtime source; expect the same level of scrutiny on his converter and be prepared to fix anything codex flags during code-review.

---

## Task 1: Verify preconditions (fail-fast)

**Files:** none modified

**Why:** The task burns 30-90 min of compute on the M3 Ultra later. Confirm upfront that the inputs are present and complete.

- [ ] **Step 1: Confirm base weights cloned and complete**

```bash
ls -la ~/models/DeepSeek-V4-Flash/config.json
ls ~/models/DeepSeek-V4-Flash/*.safetensors | wc -l
find ~/models/DeepSeek-V4-Flash -name "*.lock" -o -name "*.tmp" 2>/dev/null
```

Expected:
- `config.json` exists
- ≥40 safetensors shards (cloned model has 46 shards; some may still be filling)
- No `.lock` / `.tmp` files (clone is complete)

- [ ] **Step 2: Confirm cloned config.json arch class name**

```bash
python3 -c "import json; c=json.load(open('$HOME/models/DeepSeek-V4-Flash/config.json')); print('architectures:', c['architectures']); print('model_type:', c['model_type'])"
```

Expected: `architectures: ['DeepseekV4ForCausalLM']`, `model_type: deepseek_v4`. Antirez's converter registers this exact class name; if it differs, abort and ask the human.

- [ ] **Step 3: Confirm disk space**

```bash
df -h ~/models | tail -1
```

Required: ≥800 GiB free.

- [ ] **Step 4: Confirm antirez remote is fetched**

```bash
git remote -v | grep antirez
git fetch antirez
git log --oneline antirez/main | head -3
```

Expected: antirez remote exists, fetch succeeds, and `git show antirez/main:convert_hf_to_gguf.py | head -1` returns content.

- [ ] **Step 5: Confirm gguf-py is currently MISSING V4 entries**

```bash
python3 -c "import sys; sys.path.insert(0,'gguf-py'); import gguf; print('has DEEPSEEK4:', hasattr(gguf.MODEL_ARCH,'DEEPSEEK4'))"
```

Expected: `has DEEPSEEK4: False` (this confirms the v1 plan's assumption that gguf-py was ready was WRONG; the port is necessary).

- [ ] **Step 6: No commit (verification step only)**

---

## Task 2: Port antirez's gguf-py V4 additions

**Files:**
- Modify: `gguf-py/gguf/constants.py` (~235 lines added/changed)

**Why:** Foundation for Task 3. Without these, `DeepseekV4Model.model_arch = gguf.MODEL_ARCH.DEEPSEEK4` raises `AttributeError`.

- [ ] **Step 1: Inspect antirez's full diff for gguf-py/gguf/constants.py**

```bash
git diff feat/v4-port..antirez/main -- gguf-py/gguf/constants.py | tee /tmp/v4-gguf-py-diff.patch
wc -l /tmp/v4-gguf-py-diff.patch
```

Expected: ~235 lines. Read it end-to-end. Note which sections are pure additions (V4 enums) vs. mid-file insertions (writer helper methods adjacent to existing helpers).

- [ ] **Step 2: Apply antirez's gguf-py constants diff**

The cleanest mechanic is `git checkout antirez/main -- gguf-py/gguf/constants.py` followed by inspection, OR the manual `apply` approach below if upstream evolved that file too. Try the cherry-pick approach first; fall back if conflicts:

```bash
# Approach A (preferred): direct file overlay if upstream/feat-v4-port hasn't touched gguf-py/constants.py
git checkout antirez/main -- gguf-py/gguf/constants.py
# Verify still parses + V4 enums present
python3 -c "import sys; sys.path.insert(0,'gguf-py'); import gguf; print('DEEPSEEK4:', gguf.MODEL_ARCH.DEEPSEEK4)"
```

If approach A's import works AND `git diff feat/v4-port -- gguf-py/gguf/constants.py | head` shows reasonable additions (not destructive), proceed.

If approach A breaks something (e.g. removed an upstream-only enum), fall back to approach B:

```bash
# Approach B: apply only the V4-additive hunks via patch
git diff feat/v4-port..antirez/main -- gguf-py/gguf/constants.py > /tmp/v4-gguf-py.patch
git apply --3way /tmp/v4-gguf-py.patch
# Resolve conflicts manually following the conflict resolution rule above
```

- [ ] **Step 3: Verify gguf-py imports cleanly with V4 enums**

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, 'gguf-py')
import gguf
checks = [
    ('MODEL_ARCH.DEEPSEEK4', hasattr(gguf.MODEL_ARCH, 'DEEPSEEK4')),
    ('MODEL_TENSOR.ATTN_COMPRESSOR_APE', hasattr(gguf.MODEL_TENSOR, 'ATTN_COMPRESSOR_APE')),
    ('MODEL_TENSOR.INDEXER_COMPRESSOR_KV', hasattr(gguf.MODEL_TENSOR, 'INDEXER_COMPRESSOR_KV')),
    ('MODEL_TENSOR.HC_ATTN_BASE', hasattr(gguf.MODEL_TENSOR, 'HC_ATTN_BASE')),
    ('MODEL_TENSOR.OUTPUT_HC_BASE', hasattr(gguf.MODEL_TENSOR, 'OUTPUT_HC_BASE')),
]
for name, ok in checks:
    print(f'  {"OK" if ok else "MISSING"}  {name}')
assert all(ok for _, ok in checks), 'V4 enums incomplete'
print('all V4 gguf-py enums present')
PY
```

Expected: all `OK`, exit 0. If any `MISSING`, the diff didn't apply cleanly — revisit Step 2.

- [ ] **Step 4: Verify writer helper methods exist**

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, 'gguf-py')
import gguf
methods = [
    'add_attention_compress_ratios',
    'add_attention_compress_rope_freq_base',
    'add_attention_output_lora_rank',
    'add_attention_output_group_count',
    'add_hash_layer_count',
]
w = gguf.GGUFWriter.__dict__
for m in methods:
    print(f'  {"OK" if m in w else "MISSING"}  {m}')
assert all(m in w for m in methods), 'V4 writer helpers incomplete'
print('all V4 writer helpers present')
PY
```

Expected: all `OK`. If any are MISSING, antirez's diff also includes additions to `gguf-py/gguf/gguf_writer.py` — read his diff for that file too:

```bash
git diff feat/v4-port..antirez/main -- gguf-py/gguf/gguf_writer.py
git checkout antirez/main -- gguf-py/gguf/gguf_writer.py
```

Then re-run Step 4.

- [ ] **Step 5: Commit**

```bash
git add gguf-py/gguf/constants.py
# If gguf_writer.py was also touched in Step 4 fallback, include it:
git add gguf-py/gguf/gguf_writer.py 2>/dev/null || true
git commit -m "v4-port-H: port antirez's gguf-py V4 additions (MODEL_ARCH.DEEPSEEK4, V4 tensor enums, V4 KV constants, writer helpers)"
```

---

## Task 3: Port antirez's `DeepseekV4Model` class

**Files:**
- Modify: `convert_hf_to_gguf.py` (~1082 lines diff — antirez adds the class plus FP8/I8 dequant infrastructure)

**Why:** The actual converter for V4. With Task 2's gguf-py support in place, this class can register and run.

- [ ] **Step 1: Inspect antirez's full diff for convert_hf_to_gguf.py**

```bash
git diff feat/v4-port..antirez/main -- convert_hf_to_gguf.py | tee /tmp/v4-convert-diff.patch
wc -l /tmp/v4-convert-diff.patch
git diff feat/v4-port..antirez/main -- convert_hf_to_gguf.py --stat
```

Expected: ~1082 lines of diff. Read at least the first few hundred lines to understand:
- Top-of-file imports (antirez adds `ctypes` and possibly others — pull these in)
- The `DeepseekV4Model` class body starting at antirez's line ~9206
- Any infrastructure additions (FP8 dequant helpers, e8m0 unpacking, expert outtype parsing)

- [ ] **Step 2: Apply antirez's convert diff**

Try direct overlay first:

```bash
git checkout antirez/main -- convert_hf_to_gguf.py
python3 -c "import ast; ast.parse(open('convert_hf_to_gguf.py').read()); print('parses')"
```

If approach A parses AND `git diff feat/v4-port -- convert_hf_to_gguf.py | git apply --check` shows no conflicts with our V3.2/DSA work (i.e. nothing removed that we need), proceed.

If approach A removes our DSA-related code (likely, since antirez branched before fairydreaming/dsa landed), fall back to approach B:

```bash
# Approach B: surgical patch of just antirez's V4-additive hunks
git diff feat/v4-port..antirez/main -- convert_hf_to_gguf.py > /tmp/v4-convert.patch
git apply --3way /tmp/v4-convert.patch
# Resolve conflicts manually per the conflict resolution rule
```

When resolving conflicts:
- antirez's `DeepseekV4Model` class is purely additive — keep all of it
- antirez's `DeepseekV32Model`-related code that conflicts with our `feat/v4-port` versions: **keep ours** (we already have the up-to-date V3.2/DSA code from fairydreaming)
- Top-of-file imports: **merge both** so `ctypes` and any other antirez additions coexist with upstream's existing imports

- [ ] **Step 3: Verify the file parses + class is registered**

```bash
python3 -c "import ast; ast.parse(open('convert_hf_to_gguf.py').read()); print('parses')"
python3 - <<'PY'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'gguf-py')
# Import the converter module; it registers all model classes at import time
import convert_hf_to_gguf as m
# Look up the V4 class by registered architecture name
from convert_hf_to_gguf import ModelBase
v4_class = ModelBase.from_model_architecture("DeepseekV4ForCausalLM")
print(f'registered class: {v4_class.__name__}')
print(f'model_arch: {v4_class.model_arch}')
PY
```

Expected:
```
parses
registered class: DeepseekV4Model
model_arch: MODEL_ARCH.DEEPSEEK4
```

If either step fails, the import or registration is broken — revisit Step 2's conflict resolution.

- [ ] **Step 4: Commit**

```bash
git add convert_hf_to_gguf.py
git commit -m "v4-port-H: port antirez's DeepseekV4Model converter class (FP8/I8 dequant, V4 tensor remap, expert handling)"
```

---

## Task 4: Smoke-test the converter on shard 0

**Files:** none modified (read-only test)

**Why:** Catch any conflict-resolution bugs before kicking off a 30-60 min full conversion.

- [ ] **Step 1: Walk the cloned model's tensor names**

```bash
python3 - <<'PY'
import struct, json, glob, os
shards = sorted(glob.glob(os.path.expanduser('~/models/DeepSeek-V4-Flash/*.safetensors')))
unique_patterns = set()
import re
for shard in shards[:6]:  # first few shards cover all per-layer patterns
    with open(shard, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    for k in header.keys():
        if k == '__metadata__': continue
        p = re.sub(r'\.\d+\.', '.N.', k)
        unique_patterns.add(p)
print(f'unique HF tensor patterns across first 6 shards: {len(unique_patterns)}')
for p in sorted(unique_patterns):
    print(p)
PY
```

Compare the printed list against antirez's `DeepseekV4Model.modify_tensors` remap branches. Every printed pattern should map to a GGUF target tensor (or be explicitly skipped, e.g. `.scale` companion tensors that antirez's class consumes alongside their `.weight`).

If a pattern appears that antirez's class doesn't handle, that's a code-review issue — note it for the codex code-review step. Don't try to add tensor handling here; that's not in scope for a port task.

- [ ] **Step 2: No commit (verification step only)**

---

## Task 5: Run the f16 conversion

**Files:** none modified (produces artifact at `~/models/DeepSeek-V4-Flash-F16.gguf`)

- [ ] **Step 1: Run the converter**

```bash
python3 convert_hf_to_gguf.py \
  ~/models/DeepSeek-V4-Flash \
  --outfile ~/models/DeepSeek-V4-Flash-F16.gguf \
  --outtype f16 \
  2>&1 | tee /tmp/v4-convert.log
```

Wall-clock: ~30-60 min on the M3 Ultra. Watch for:
- `Tensor not found in mapping` — a remap is missing; halt, capture the offending tensor name, fall back to fix-loop
- Out-of-memory errors — RAM headroom; converter loads shards iteratively but FP8 dequant temporary buffers can spike

Expected ending: a `INFO:gguf.gguf_writer:Wrote ...` line and exit 0.

- [ ] **Step 2: Verify artifact size**

```bash
ls -la ~/models/DeepSeek-V4-Flash-F16.gguf
du -sh ~/models/DeepSeek-V4-Flash-F16.gguf
```

Expected: ~570 GiB.

- [ ] **Step 3: Smoke-load via gate-loader**

```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-F16.gguf \
  ./tests/v4-port/gate-loader.sh
```

Expected: `PASS: loader recognizes V4 GGUF`.

If FAIL: examine error, identify which tensor/hparam is malformed. Fix at the converter level, blow away the f16 artifact, re-run conversion.

- [ ] **Step 4: No commit (artifact step only)**

---

## Task 6: Run llama-quantize to Q4_K_M

**Files:** none modified (produces artifact at `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf`)

- [ ] **Step 1: Run quantization**

```bash
./build/bin/llama-quantize \
  ~/models/DeepSeek-V4-Flash-F16.gguf \
  ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  Q4_K_M \
  2>&1 | tee /tmp/v4-quantize.log
```

Wall-clock: ~30-90 min.

- [ ] **Step 2: Verify artifact size**

```bash
ls -la ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf
du -sh ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf
```

Expected: ~150 GiB.

- [ ] **Step 3: Capture sha256 of both artifacts**

```bash
shasum -a 256 ~/models/DeepSeek-V4-Flash-F16.gguf > /tmp/v4-sha256.txt
shasum -a 256 ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf >> /tmp/v4-sha256.txt
cat /tmp/v4-sha256.txt
```

- [ ] **Step 4: No commit (artifact step only)**

---

## Task 7: Validate via the V4 gate suite

**Files:** none modified

**Why:** End-to-end validation. Gates exercise loader, coherence (NGL=0 and 999), decode speed, tools, and chat. (Task G's `gate-server-chat-q8.sh` is not in run-all-gates.sh yet; that's G's deliverable.)

- [ ] **Step 1: Run all gates against Q4_K_M**

```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  ./tests/v4-port/run-all-gates.sh 2>&1 | tee /tmp/v4-gates.log
```

Expected: `ALL GATES PASS`. Specifically:
- `gate-loader`: `PASS: loader recognizes V4 GGUF`
- `gate-coherence` (NGL=0 and NGL=999): `PASS: coherence`
- `gate-speed`: `PASS: speed (NGL=999, ~XX tok/s)`
- `gate-tools`: `PASS: tool calling (5/5 with tool_calls)`
- `gate-server-chat`: `PASS: server-chat (3/3 tests coherent)`

- [ ] **Step 2: Capture decode tok/s**

```bash
grep -E "Decode tok/s|gate-speed" /tmp/v4-gates.log
```

Record for the completion doc. Compare against the IQ2XXS baseline (25.91 tok/s on Metal per the original handoff).

- [ ] **Step 3: If any gate FAILed**

Most likely cause is a converter issue (wrong tensor name, missing hparam, FP8 dequant bug). Less likely: the runtime expects a hparam antirez's converter doesn't emit (would mean upstream-after-antirez introduced new requirements; rare but possible).

Compare against IQ2XXS:

```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  ./tests/v4-port/<failing-gate>.sh
```

If IQ2XXS still passes the same gate, our converter is at fault. Inspect the new GGUF's metadata via the gate's loader log (`llama_model_loader: - kv ...` lines) against the IQ2XXS reference and identify the diff.

If IQ2XXS also fails (unlikely), the gate might have regressed — escalate to the human.

- [ ] **Step 4: No commit (verification step only)**

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
GGUF is available at `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` (~150 GiB)
and passes the full V4 gate suite.

The converter is a port of antirez's working V4 converter from
`antirez/llama.cpp-deepseek-v4-flash`, the same fork that produced the
IQ2XXS GGUF we'd been running. Mirrors the structure of Task A's runtime
port: cherry-pick antirez's V4-specific code, resolve conflicts with our
DSA/V3.2 baseline, validate.

## Recipe (reproducible)

```bash
# Prereqs: HF clone of deepseek-ai/DeepSeek-V4-Flash at ~/models/DeepSeek-V4-Flash/
#          ≥800 GiB free disk
#          llama.cpp built at HEAD of feat/v4-port-H-quants

# 1. f16 conversion (~30-60 min)
python3 convert_hf_to_gguf.py ~/models/DeepSeek-V4-Flash \
  --outfile ~/models/DeepSeek-V4-Flash-F16.gguf --outtype f16

# 2. Q4_K_M quantization (~30-90 min)
./build/bin/llama-quantize \
  ~/models/DeepSeek-V4-Flash-F16.gguf \
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
- `DeepSeek-V4-Flash-F16.gguf`: [paste from `du -sh`] (~570 GiB nominal)
- `DeepSeek-V4-Flash-Q4_K_M.gguf`: [paste from `du -sh`] (~150 GiB nominal)

## Performance vs. IQ2XXS

| Metric | IQ2XXS (baseline) | Q4_K_M (this) |
|---|---|---|
| BPW | 2.44 | ~4.5 |
| File size | ~80 GiB | [paste] |
| Decode tok/s (NGL=999) | 25.91 | [paste from gate-speed] |
| API recall (anecdotal) | weak — confused numpy.linalg.eig and scipy.linalg.eig in terminal-bench | TBD per agentic test |

A targeted terminal-bench retry on Q4_K_M is recommended as a separate
followup; the spec explicitly leaves end-to-end agentic validation
out-of-scope for this task.

## What landed

Two commits on `feat/v4-port-H-quants`:
1. `gguf-py/gguf/constants.py` (and possibly `gguf_writer.py`) — V4 enums,
   KV constants, writer helper methods. Direct port of antirez's diff.
2. `convert_hf_to_gguf.py` — `DeepseekV4Model` class, FP8/I8 dequant
   infrastructure. Direct port of antirez's diff with conflict resolution
   for our V3.2/DSA baseline.

## What is NOT included

- Other quant levels (Q6_K, Q5_K_M, Q8_0, IQ4_XS). Future tasks can
  re-quantize from `~/models/DeepSeek-V4-Flash-F16.gguf` without re-running
  the converter.
- Imatrix calibration.
- terminal-bench end-to-end validation under Q4_K_M.

## Conflict resolution notes

[Document any non-trivial conflicts encountered during Tasks 2-3 — e.g.
"upstream introduced X in the same hunk where antirez added V4 writer Y;
merged both to keep X functional." If no conflicts: write "Trivial port —
no conflicts."]
```

- [ ] **Step 2: Fill in placeholders from captured outputs**

Replace `[paste ...]` markers with the actual sha256s, file sizes, decode tok/s, and conflict-resolution notes.

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

- [x] `convert_hf_to_gguf.py` recognizes `DeepseekV4ForCausalLM`
- [x] `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` exists, valid (passes `gate-loader.sh`)
- [x] `run-all-gates.sh` (loader, coherence×2, speed, tools, server-chat) all PASS against Q4_K_M
- [x] Decode tok/s recorded
- [x] Followup writeup committed
- [x] Branch pushed to `mine`
- [ ] Codex plan-review and code-review APPROVE — handled by the dev-team orchestrator

This plan ends here. The orchestrator will run codex code-review and, on APPROVE, mark `tasks/active/v4-port-H-quants.json` state="done" and move it to `tasks/done/`.
