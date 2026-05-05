# V4 Q4_K_M Quants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Q4_K_M GGUF for `deepseek-ai/DeepSeek-V4-Flash` from base safetensors and validate it against the existing V4 gate suite. Land V4 support in `convert_hf_to_gguf.py` so the recipe is reproducible.

**Architecture:** Subclass the existing `DeepseekV32Model` (V3.2 is the immediate ancestor — V4 inherits its DSA / sparse attention pipeline) and add V4-specific tensor name mappings (hyper-connection weights, attention/indexer compressors, output LoRA groups, KV compressors) and hparams (per-layer `attn_compress_ratio[]`, indexer head/top_k). Conversion produces an f16 GGUF; `llama-quantize` then produces Q4_K_M. Validation is the existing `run-all-gates.sh` (which now includes G's `gate-server-chat-q8.sh`) pointed at the new GGUF via `V4_GGUF=`.

**Tech Stack:** Python 3.13 (`convert_hf_to_gguf.py`), llama.cpp C++ (`llama-quantize` already built), bash gates under `tests/v4-port/`, `~/models/` filesystem layout for artifacts.

---

## Worktree / branch setup

This task runs after Task G has merged into `feat/v4-port`. Worktree off `feat/v4-port`:

```bash
cd /Users/cchuter/work/llama.cpp
git fetch mine
git worktree add ../llama.cpp.H-quants -b feat/v4-port-H-quants mine/feat/v4-port
cd ../llama.cpp.H-quants
cmake --build build -j 32   # rebuild against worktree HEAD
```

All steps below run from `../llama.cpp.H-quants`.

---

## File structure

Files created or modified by this plan:

- **Modify** `convert_hf_to_gguf.py` — add `DeepseekV4ForCausalLM` class (subclass of `DeepseekV32Model`); ~200-400 line addition.
- **Possibly modify** `gguf-py/gguf/constants.py` — only if V4-specific tensor enum entries are missing.
- **Possibly modify** `gguf-py/gguf/tensor_mapping.py` — only if V4-specific tensor name mappings are missing.
- **Create** `docs/plans/v4-port-quants-completion.md` — followup doc with recipe, sha256s, perf numbers.
- **Build artifacts on disk (NOT committed):**
  - `~/models/DeepSeek-V4-Flash-F16.gguf` (~570 GiB, intermediate)
  - `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` (~150 GiB, primary deliverable)

`tests/v4-port/run-all-gates.sh` and the individual gate scripts already honor the `V4_GGUF` env var; no modification expected.

---

## Task 1: Verify preconditions (fail-fast)

**Files:** none modified

**Why:** This task burns ~30-90 min of compute on the M3 Ultra later. Confirm upfront that the inputs are present and complete before starting the conversion.

- [ ] **Step 1: Confirm base weights cloned and complete**

```bash
ls -la ~/models/DeepSeek-V4-Flash/ | head -10
ls ~/models/DeepSeek-V4-Flash/*.safetensors 2>/dev/null | wc -l
ls ~/models/DeepSeek-V4-Flash/config.json && echo "config present"
ls ~/models/DeepSeek-V4-Flash/.git/lfs 2>/dev/null && echo "git-lfs present"
```

Expected: directory exists, ≥1 `.safetensors` file (V4 is sharded; expect dozens), `config.json` present, no `.lock` files.

```bash
find ~/models/DeepSeek-V4-Flash -name "*.lock" -o -name "*.tmp" 2>/dev/null
```

Expected: empty output. If any `.lock` files: clone is still in progress; abort and ask the user to wait.

- [ ] **Step 2: Confirm disk space**

```bash
df -h ~/models | tail -1 | awk '{print "Avail: "$4}'
```

Required: ≥800 GiB free (570 GiB f16 + 150 GiB Q4_K_M + 80 GiB headroom for intermediate buffers and OS). If less than 800 GiB free, abort and ask the user to free space.

- [ ] **Step 3: Inspect HF model class name**

```bash
python3 -c "import json; c=json.load(open('$HOME/models/DeepSeek-V4-Flash/config.json')); print('architectures:', c.get('architectures'))"
```

Record the architecture class name (e.g. `DeepseekV4ForCausalLM` or `DeepseekV4FlashForCausalLM`). This is the registration name we use in `convert_hf_to_gguf.py` in Task 2.

- [ ] **Step 4: Confirm V4 GGUF tensor naming already exists in gguf-py**

```bash
grep -nE "INDEXER_COMPRESSOR|ATTN_COMPRESSOR|HYPER_CONN" gguf-py/gguf/constants.py gguf-py/gguf/tensor_mapping.py 2>&1 | head -10
```

Expected: matches present (these were added during the V4 port). If missing: Task 2 will need to add them.

- [ ] **Step 5: No commit (verification step only)**

---

## Task 2: Add V4 model class skeleton to `convert_hf_to_gguf.py`

**Files:**
- Modify: `convert_hf_to_gguf.py` — add new class around line 9219 (next to `DeepseekV32Model`)

- [ ] **Step 1: Read `DeepseekV32Model` for the pattern**

```bash
sed -n '9219,9380p' convert_hf_to_gguf.py
```

Note: `set_gguf_parameters`, `modify_tensors`, the registry decorator (`@ModelBase.register("DeepseekV32ForCausalLM")`), and how DSA hparams are surfaced.

- [ ] **Step 2: Read V4's runtime hparam loader**

```bash
sed -n '15,70p' src/models/deepseek4.cpp
```

This is the source of truth for what V4 hparams llama.cpp expects. Match the converter's GGUF KV writes against these `ml.get_key(...)` calls.

- [ ] **Step 3: Read V4's tensor loader**

```bash
sed -n '72,160p' src/models/deepseek4.cpp
```

This enumerates the tensors V4 expects: `attn_compressor_*`, `indexer_compressor_*`, `attn_kv_a_norm`, `attn_kv`, `attn_wo_a`, `attn_wo_b`, indexer-specific tensors. The converter must map HF safetensors keys to these GGUF tensor names.

- [ ] **Step 4: Add the V4 class skeleton**

Insert after the `DeepseekV32Model` class (after the closing of its block, before the next class definition). Use the architecture name(s) recorded in Task 1, Step 3:

```python
@ModelBase.register("DeepseekV4ForCausalLM", "DeepseekV4FlashForCausalLM")
class DeepseekV4Model(DeepseekV32Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK4

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # V4-specific hparams
        # attn_compress_ratio is per-layer; emit as an int-list KV
        compress_ratios = self.hparams.get("attn_compress_ratio")
        if compress_ratios is None:
            raise ValueError("V4 config missing attn_compress_ratio")
        self.gguf_writer.add_uint32_list(
            "deepseek4.attention.compress_ratio", compress_ratios
        )
        # Indexer params
        self.gguf_writer.add_uint32(
            "deepseek4.attention.indexer_head_count",
            self.hparams["indexer_n_head"],
        )
        self.gguf_writer.add_uint32(
            "deepseek4.attention.indexer_key_length",
            self.hparams["indexer_head_size"],
        )
        self.gguf_writer.add_uint32(
            "deepseek4.attention.indexer_top_k",
            self.hparams["indexer_top_k"],
        )

    def modify_tensors(self, data_torch, name, bid):
        # V4-specific tensor remapping; fall through to V3.2's modify_tensors
        # for tensors V4 inherits unchanged. Remap branches added in Task 3.
        return super().modify_tensors(data_torch, name, bid)
```

(This stub is functional — it just falls through to the parent class. Task 3 replaces the body with the V4-specific remap logic.)

- [ ] **Step 5: Verify the file parses**

```bash
python3 -c "import ast; ast.parse(open('convert_hf_to_gguf.py').read()); print('parses')"
```

Expected: `parses`. (No conversion run yet — that's Task 4 once the tensor remaps are in.)

- [ ] **Step 6: Commit**

```bash
git add convert_hf_to_gguf.py
git commit -m "v4-port-H: add DeepseekV4Model class skeleton (subclass of V3.2)"
```

---

## Task 3: Implement V4 tensor remaps in `modify_tensors`

**Files:**
- Modify: `convert_hf_to_gguf.py` — flesh out `DeepseekV4Model.modify_tensors`

**Why:** V4 has tensors V3.2 doesn't: hyper-connection weights, attention/indexer compressors (each with `ape`, `kv`, `gate`, `norm` sub-tensors), output LoRA groups (`attn_wo_a`, `attn_wo_b`), and the V4-specific KV compressor. Map HF naming to GGUF naming.

- [ ] **Step 1: List the HF tensor keys present in the cloned model**

```bash
python3 - <<'PY'
import safetensors, glob, json
keys = set()
for f in sorted(glob.glob(f'$HOME/models/DeepSeek-V4-Flash/*.safetensors')):
    with safetensors.safe_open(f, framework='pt') as sf:
        for k in sf.keys(): keys.add(k)
# Group by suffix for legibility
from collections import Counter
suffixes = Counter('.'.join(k.split('.')[-3:]) for k in keys)
for s, n in suffixes.most_common(40):
    print(f'{n:4d} {s}')
PY
```

Record the V4-specific suffixes you'll need to remap (e.g. `compressor.kv.weight`, `compressor.ape.weight`, `hyper_conn.weight`, `wo_a.weight`, `wo_b.weight`).

- [ ] **Step 2: Map each V4 tensor name pattern to its GGUF target**

The mapping table (cross-checked against `src/llama-arch.cpp` LLM_TENSOR_INDEXER_COMPRESSOR_* and LLM_TENSOR_ATTN_COMPRESSOR_* enums, and `gguf-py/gguf/tensor_mapping.py`):

| HF safetensors suffix | GGUF tensor name (per-layer N) |
|---|---|
| `model.layers.N.self_attn.compressor.ape.weight` | `blk.N.attn_compressor_ape.weight` |
| `model.layers.N.self_attn.compressor.kv.weight` | `blk.N.attn_compressor_kv.weight` |
| `model.layers.N.self_attn.compressor.gate.weight` | `blk.N.attn_compressor_gate.weight` |
| `model.layers.N.self_attn.compressor.norm.weight` | `blk.N.attn_compressor_norm.weight` |
| `model.layers.N.self_attn.indexer.compressor.ape.weight` | `blk.N.indexer_compressor_ape.weight` |
| `model.layers.N.self_attn.indexer.compressor.kv.weight` | `blk.N.indexer_compressor_kv.weight` |
| `model.layers.N.self_attn.indexer.compressor.gate.weight` | `blk.N.indexer_compressor_gate.weight` |
| `model.layers.N.self_attn.indexer.compressor.norm.weight` | `blk.N.indexer_compressor_norm.weight` |
| `model.layers.N.self_attn.indexer.q_b_proj.weight` | `blk.N.indexer_attn_q_b.weight` |
| `model.layers.N.self_attn.indexer.proj.weight` | `blk.N.indexer_proj.weight` |
| `model.layers.N.self_attn.kv_a_norm.weight` | `blk.N.attn_kv_a_norm.weight` |
| `model.layers.N.self_attn.kv.weight` | `blk.N.attn_kv.weight` |
| `model.layers.N.self_attn.o_a.weight` | `blk.N.attn_wo_a.weight` |
| `model.layers.N.self_attn.o_b.weight` | `blk.N.attn_wo_b.weight` |
| `model.layers.N.hyper_conn.weight` | `blk.N.hyper_conn.weight` |

**Important:** before writing this table into code, verify the exact HF suffixes against the actual safetensors keys from Step 1. The spec's mapping is best-effort; the cloned model is authoritative. Adjust the table to match what's in the file.

- [ ] **Step 3: Implement the remap in `modify_tensors`**

Replace the placeholder body added in Task 2:

```python
def modify_tensors(self, data_torch, name, bid):
    # V4-specific tensor remaps. Order: most-specific first; fall through
    # to V3.2's modify_tensors for tensors V4 inherits unchanged.
    import re
    m = re.match(r"model\.layers\.(\d+)\.self_attn\.(.+)$", name)
    if m:
        layer = int(m.group(1))
        rest  = m.group(2)
        # Compressors (attn and indexer share the same {ape,kv,gate,norm} structure)
        comp = re.match(r"(indexer\.)?compressor\.(ape|kv|gate|norm)\.weight$", rest)
        if comp:
            is_indexer = comp.group(1) is not None
            sub        = comp.group(2)
            prefix     = "indexer_compressor" if is_indexer else "attn_compressor"
            new_name   = f"blk.{layer}.{prefix}_{sub}.weight"
            return [(new_name, data_torch)]
        # Indexer Q-B and proj
        if rest == "indexer.q_b_proj.weight":
            return [(f"blk.{layer}.indexer_attn_q_b.weight", data_torch)]
        if rest == "indexer.proj.weight":
            return [(f"blk.{layer}.indexer_proj.weight", data_torch)]
        # KV compressor
        if rest == "kv_a_norm.weight":
            return [(f"blk.{layer}.attn_kv_a_norm.weight", data_torch)]
        if rest == "kv.weight":
            return [(f"blk.{layer}.attn_kv.weight", data_torch)]
        # Output LoRA groups
        if rest == "o_a.weight":
            return [(f"blk.{layer}.attn_wo_a.weight", data_torch)]
        if rest == "o_b.weight":
            return [(f"blk.{layer}.attn_wo_b.weight", data_torch)]
    # Hyper-connection weights (per-layer, not under self_attn)
    m = re.match(r"model\.layers\.(\d+)\.hyper_conn\.weight$", name)
    if m:
        return [(f"blk.{m.group(1)}.hyper_conn.weight", data_torch)]
    # Fall through for everything else (token embeddings, MLP, etc.)
    return super().modify_tensors(data_torch, name, bid)
```

- [ ] **Step 4: Smoke-test the remap on the actual safetensors**

```bash
python3 - <<'PY'
import sys
sys.path.insert(0, '.')
from convert_hf_to_gguf import DeepseekV4Model
import safetensors, glob

# Sanity: instantiate the class and walk one shard's keys through modify_tensors.
# We don't need a full model — just confirm the remap maps known V4 suffixes.
test_keys = [
    "model.layers.0.self_attn.compressor.ape.weight",
    "model.layers.0.self_attn.compressor.kv.weight",
    "model.layers.0.self_attn.indexer.compressor.kv.weight",
    "model.layers.0.self_attn.kv.weight",
    "model.layers.0.self_attn.o_a.weight",
    "model.layers.0.hyper_conn.weight",
    "model.embed_tokens.weight",   # falls through to parent
]
# Bare-bones invocation; modify_tensors is a method but should be callable
# with self=None for this remap-only check (or use a real instance).
import torch
class _FakeSelf(DeepseekV4Model):
    def __init__(self): pass  # skip parent init for smoke test
    def __getattr__(self, name): raise AttributeError(name)
fake = _FakeSelf()
for k in test_keys:
    try:
        result = DeepseekV4Model.modify_tensors(fake, torch.tensor([0.0]), k, 0)
        print(f'{k}  ->  {[r[0] for r in result] if result else "(fall-through)"}')
    except AttributeError:
        print(f'{k}  ->  (fall-through to parent)')
    except Exception as e:
        print(f'{k}  ->  ERROR: {e}')
PY
```

Expected: each V4-specific key maps to a `blk.N.*` GGUF name; the embed_tokens key falls through. If a V4 key falls through unexpectedly, the regex doesn't match — refine.

- [ ] **Step 5: Commit**

```bash
git add convert_hf_to_gguf.py
git commit -m "v4-port-H: implement DeepseekV4Model.modify_tensors remaps"
```

---

## Task 4: Run the f16 conversion

**Files:** none modified (produces artifact at `~/models/DeepSeek-V4-Flash-F16.gguf`)

**Why:** Converts HF safetensors → GGUF f16. This is the input to llama-quantize.

- [ ] **Step 1: Run the converter**

```bash
python3 convert_hf_to_gguf.py \
  ~/models/DeepSeek-V4-Flash \
  --outfile ~/models/DeepSeek-V4-Flash-F16.gguf \
  --outtype f16 \
  2>&1 | tee /tmp/v4-convert.log
```

Wall-clock: ~30-60 min on the M3 Ultra. Watch for:
- "Tensor not found in mapping" — a remap is missing; abort and refine Task 3.
- Out-of-memory errors — RAM headroom check; converter loads shards iteratively but can spike.

Expected ending: a line like `INFO:gguf.gguf_writer:Wrote ~/models/DeepSeek-V4-Flash-F16.gguf` and process exit 0.

- [ ] **Step 2: Verify artifact**

```bash
ls -la ~/models/DeepSeek-V4-Flash-F16.gguf
du -sh ~/models/DeepSeek-V4-Flash-F16.gguf
```

Expected: ~570 GiB. If much smaller (<400 GiB) something was skipped.

- [ ] **Step 3: Smoke-load via gate-loader**

```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-F16.gguf \
  ./tests/v4-port/gate-loader.sh
```

Expected: `PASS: loader recognizes V4 GGUF`.

If FAIL: the converter shipped a malformed GGUF (missing tensors, wrong types, hparam mismatch). Inspect the loader error against the spec's hparam list; fix the converter; re-run conversion.

- [ ] **Step 4: No commit (artifact step only — GGUFs are not committed)**

---

## Task 5: Run llama-quantize to Q4_K_M

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

Expected ending: a summary like `[1234/1234] f16 -> Q4_K_M, 567.89 MiB -> 234.56 MiB` plus an exit-0.

- [ ] **Step 2: Verify artifact**

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

Save this for the completion doc (Task 7).

- [ ] **Step 4: No commit (artifact step only)**

---

## Task 6: Validate via the full V4 gate suite

**Files:** none modified

**Why:** End-to-end validation that the converter produced a correct GGUF and the quantization didn't break anything. Gates exercise loader, coherence, decode speed, tools, chat, and (post-G-merge) q8 KV behavior.

- [ ] **Step 1: Run all gates against Q4_K_M**

```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  ./tests/v4-port/run-all-gates.sh 2>&1 | tee /tmp/v4-gates.log
```

Expected: `ALL GATES PASS` at the end. Specifically:
- `gate-loader`: recognizes V4 arch
- `gate-coherence` (NGL=0 and NGL=999): coherent text generation, ≥80% printable, no degenerate decode
- `gate-speed`: ≥10 tok/s decode (M3 Ultra should easily exceed)
- `gate-tools`: 5/5 successful tool_calls
- `gate-server-chat`: 3/3 coherent chat completions
- `gate-server-chat-q8`: coherent + WARN observed (from Task G)

- [ ] **Step 2: Capture Q4_K_M decode speed for the completion doc**

```bash
grep "Decode tok/s" /tmp/v4-gates.log
```

Record the exact tok/s number. Compare against the IQ2XXS baseline (25.91 tok/s on Metal per the original handoff).

- [ ] **Step 3: If any gate FAILed, diagnose**

Most likely cause is a converter bug (wrong tensor name/type/hparam). Less likely: quantization issue (Q4_K_M doesn't typically corrupt models built on a working f16 source).

Compare the failing gate's behavior against the IQ2XXS baseline:

```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  ./tests/v4-port/<failing-gate>.sh
```

If IQ2XXS still PASSes the same gate, the converter is the suspect. Inspect the offending tensor name(s) in `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` via `gguf-dump` (or the `llama_model_loader: - tensor` lines in the gate's server log) against the IQ2XXS reference.

If IQ2XXS also FAILs (unlikely), the gate itself is the bug — but our existing gates have been stable for both IQ2XXS and prior V4 work, so investigate carefully.

- [ ] **Step 4: No commit (verification step only)**

---

## Task 7: Write the completion doc

**Files:**
- Create: `docs/plans/v4-port-quants-completion.md`

- [ ] **Step 1: Write the doc**

```markdown
# V4 Port — Q4_K_M Quants Completion

Follow-up to `docs/superpowers/specs/2026-05-05-v4-quants-kvcache-design.md`
(Task H). Closes the deferred "build our own quants" item.

## TL;DR

`convert_hf_to_gguf.py` now recognizes `DeepseekV4ForCausalLM` /
`DeepseekV4FlashForCausalLM`. A Q4_K_M GGUF is available at
`~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` (~150 GiB) and passes the full
V4 gate suite, including the new q8-KV gate from Task G.

## Recipe (reproducible)

```bash
# Prereqs: HF clone of deepseek-ai/DeepSeek-V4-Flash at ~/models/DeepSeek-V4-Flash/
#          ≥800 GiB free disk, llama.cpp built at HEAD of feat/v4-port-H-quants

# 1. f16 conversion
python3 convert_hf_to_gguf.py ~/models/DeepSeek-V4-Flash \
  --outfile ~/models/DeepSeek-V4-Flash-F16.gguf --outtype f16

# 2. Q4_K_M quantization
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

## What landed in the converter

`DeepseekV4Model` subclass of `DeepseekV32Model` in `convert_hf_to_gguf.py`,
with:
- `set_gguf_parameters` adding V4 hparams: `attn_compress_ratio[]`,
  `indexer_n_head`, `indexer_head_size`, `indexer_top_k`
- `modify_tensors` mapping V4-specific HF safetensors keys to GGUF tensor
  names: attention/indexer compressors (ape/kv/gate/norm), KV compressor,
  output LoRA groups (wo_a, wo_b), indexer q_b_proj/proj, hyper-connections

Total addition: ~[N] lines in `convert_hf_to_gguf.py` (no changes to
`gguf-py/gguf/`; V4 tensor enums were already present from antirez's port).

## What is NOT included

- Other quant levels (Q6_K, Q5_K_M, Q8_0, IQ4_XS). Future tasks can
  re-quantize from `~/models/DeepSeek-V4-Flash-F16.gguf` without re-running
  the converter.
- Imatrix calibration. Future quality work could add this.
- terminal-bench end-to-end validation under Q4_K_M. Queue separately.

## What we explicitly did NOT commit

- Either GGUF artifact (~150 GiB and ~570 GiB respectively). `~/models/`
  is not in git. The recipe above plus the sha256s document recoverability.
```

- [ ] **Step 2: Fill in the placeholders from your captured outputs**

Before committing: replace `[paste ...]` markers with the actual sha256s, file sizes, decode tok/s, and line count from the converter diff.

- [ ] **Step 3: Commit**

```bash
git add docs/plans/v4-port-quants-completion.md
git commit -m "v4-port-H: completion doc — Q4_K_M GGUF + recipe + sha256 + perf"
```

---

## Task 8: Push to mine

- [ ] **Step 1: Push the feature branch**

```bash
git push mine feat/v4-port-H-quants
```

- [ ] **Step 2: Confirm DoD is met**

Verify each item in the spec's DoD section:
- [x] `convert_hf_to_gguf.py` recognizes `DeepseekV4ForCausalLM` (and `DeepseekV4FlashForCausalLM` if applicable)
- [x] `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` exists, valid (passes `gate-loader.sh`)
- [x] `run-all-gates.sh` (incl. `gate-server-chat-q8.sh` from G) all PASS against Q4_K_M
- [x] Decode tok/s recorded for the new quant
- [x] Followup writeup committed
- [x] Branch pushed to `mine`
- [ ] Codex plan-review and code-review APPROVE — handled by the dev-team orchestrator

This plan ends here. The orchestrator will run codex code-review and, on APPROVE, mark `tasks/active/v4-port-H-quants.json` state="done" and move it to `tasks/done/`.
