# V4 IQ1 Quant Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the smallest practical V4-Flash GGUF artifacts (IQ1_S ~52 GiB and IQ1_M ~60 GiB) using the imatrix calibration data produced by Task I, validate them through the standard V4 gates, and publish to the existing HF repo.

**Architecture:** Re-quantize the existing V4 Q8_0 source GGUF using `llama-quantize` with the `--imatrix` flag pointing at `tests/v4-port/calibration/imatrix-v4-flash.dat` (landed by Task I). Pin `output_tensor` and `token_embd` at Q5_K to protect special-token discrimination — IQ1 body alone collapses tool-call grammar otherwise. No XL/XXL-style attention/indexer/hc pinning; that's a Q2_K-specific failure mode that doesn't apply at IQ1's 1.5–1.75 BPW.

**Tech Stack:** llama.cpp `llama-quantize` (release build), `llama-gguf-split` for shard packaging, the V4 validation gate scripts under `tests/v4-port/`, the `hf` CLI for HF uploads.

---

## Worktree / branch setup

Runs in a worktree off `feat/v4-port` (which by this point contains Task I's merged work):

```bash
cd /Users/cchuter/work/llama.cpp
git fetch mine
git worktree add ../llama.cpp.J-iq1 -b feat/v4-port-J-iq1-quants mine/feat/v4-port
cd ../llama.cpp.J-iq1

# Verify Task I's calibration data is present (this is the only hard prerequisite)
ls -la tests/v4-port/calibration/imatrix-v4-flash.dat
```

Expected: file exists, size ~few MB. If missing, Task I has not been merged yet and this task cannot proceed.

```bash
# Verify Q8 source is on disk (re-quantize input)
export V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q8_0.gguf
ls -la "$V4_GGUF"   # ~282 GiB

# Configure release build/ if fresh worktree (idempotent)
if [ ! -f build/CMakeCache.txt ]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Release
fi

# Build llama-quantize and llama-gguf-split
cmake --build build --target llama-quantize llama-gguf-split -j
```

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `~/models/DeepSeek-V4-Flash-IQ1_S.gguf` | Create | Single-file IQ1_S build (intermediate, deleted after split) |
| `~/models/DeepSeek-V4-Flash-IQ1_M.gguf` | Create | Single-file IQ1_M build (intermediate, deleted after split) |
| `~/models/DeepSeek-V4-Flash-GGUF/IQ1_S/` | Create | Split shards for IQ1_S |
| `~/models/DeepSeek-V4-Flash-GGUF/IQ1_M/` | Create | Split shards for IQ1_M |
| HF: `teamblobfish/DeepSeek-V4-Flash-GGUF/IQ1_S/*.gguf` | Upload | Public IQ1_S artifact |
| HF: `teamblobfish/DeepSeek-V4-Flash-GGUF/IQ1_M/*.gguf` | Upload | Public IQ1_M artifact |
| HF: `teamblobfish/DeepSeek-V4-Flash-GGUF/README.md` | Modify | Add IQ1 rows to quant table |

No source code changes. This task is pure artifact production.

---

### Task 1: Build IQ1_S

**Files:**
- Create: `~/models/DeepSeek-V4-Flash-IQ1_S.gguf`

- [ ] **Step 1: Verify disk space (need ~52 GiB headroom)**

```bash
df -h "$HOME/models" | head -2
```

Expected: free space ≥ 80 GiB (52 for the build + buffer). If less, free space first; do NOT delete the imatrix calibration file or the Q8 source.

- [ ] **Step 2: Run llama-quantize with imatrix and Q5_K pins**

```bash
./build/bin/llama-quantize --allow-requantize \
  --imatrix tests/v4-port/calibration/imatrix-v4-flash.dat \
  --output-tensor-type q5_K \
  --token-embedding-type q5_K \
  "$V4_GGUF" \
  ~/models/DeepSeek-V4-Flash-IQ1_S.gguf \
  IQ1_S \
  2>&1 | tee /tmp/iq1s-build.log
```

Expected: runs ~10-15 minutes. Progress prints per tensor. Final lines should report:
```
llama_model_quantize_impl: model size  = 288259.78 MiB (8.50 BPW)
llama_model_quantize_impl: quant size  = ~53000 MiB (~1.55 BPW)
```

Exact size will vary; ~52 GiB output is the target.

- [ ] **Step 3: Verify the file exists and has reasonable size**

```bash
ls -lh ~/models/DeepSeek-V4-Flash-IQ1_S.gguf
# Expected: 50-58 GiB. Anything <40 GiB or >70 GiB indicates a problem.
```

- [ ] **Step 4: Sanity-load the quant**

```bash
./build/bin/llama-cli \
  --model ~/models/DeepSeek-V4-Flash-IQ1_S.gguf \
  --jinja --reasoning off \
  -p "What is 2+2?" \
  -n 16 \
  -ngl 999 \
  --temp 0 \
  2>&1 | tail -20
```

Expected: produces some output (correctness optional at IQ1 — coherent text is the bar). If it segfaults or produces only special tokens, the quant is broken; do not proceed.

- [ ] **Step 5: Commit progress note (no source changes, just log)**

```bash
git commit --allow-empty -m "v4-port-J: IQ1_S build complete (~52 GiB, $(stat -f%z ~/models/DeepSeek-V4-Flash-IQ1_S.gguf) bytes)"
```

---

### Task 2: Validate IQ1_S through gates

**Files:**
- None modified; runs existing gates with overridden `V4_GGUF`

- [ ] **Step 1: Run gate-loader against IQ1_S**

```bash
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_S.gguf \
  ./tests/v4-port/gate-loader.sh
```

Expected: `PASS` line. Confirms architecture is recognized.

- [ ] **Step 2: Run gate-coherence at NGL=999**

```bash
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_S.gguf NGL=999 \
  ./tests/v4-port/gate-coherence.sh
```

Expected: `PASS` line; coherent text generated at temp 0.

- [ ] **Step 3: Run gate-tools (the make-or-break for agent use)**

```bash
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_S.gguf \
  ./tests/v4-port/gate-tools.sh
```

Expected: `5/5 tool calls succeeded` or equivalent. **If this fails, IQ1_S is unviable for agent use at Q5_K output pin.** Two options:

1. **Bump output pin to Q6_K and rebuild.** Replace `q5_K` with `q6_K` in Task 1 Step 2 and rerun. Document the bump in the commit message.
2. **Declare IQ1_S unviable.** Document in the PR/commit; ship anyway as a research artifact. Continue to IQ1_M (Task 3) — it may still work.

- [ ] **Step 4: Record validation result**

```bash
cat > /tmp/iq1s-gates.txt <<EOF
IQ1_S validation results:
- gate-loader:    $(V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_S.gguf ./tests/v4-port/gate-loader.sh 2>&1 | tail -1)
- gate-coherence: $(V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_S.gguf NGL=999 ./tests/v4-port/gate-coherence.sh 2>&1 | tail -1)
- gate-tools:     $(V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_S.gguf ./tests/v4-port/gate-tools.sh 2>&1 | tail -1)
EOF
cat /tmp/iq1s-gates.txt
```

- [ ] **Step 5: Commit the validation note**

```bash
mkdir -p tests/v4-port/results
cp /tmp/iq1s-gates.txt tests/v4-port/results/iq1s-gates.txt
git add tests/v4-port/results/iq1s-gates.txt
git commit -m "v4-port-J: IQ1_S gate results recorded"
```

---

### Task 3: Build IQ1_M

**Files:**
- Create: `~/models/DeepSeek-V4-Flash-IQ1_M.gguf`

- [ ] **Step 1: Verify disk space (need ~60 GiB headroom)**

```bash
df -h "$HOME/models" | head -2
```

If headroom is tight after IQ1_S build, the IQ1_S single-file can be split + deleted before this step (do steps from Task 5 first, then return here). Otherwise proceed.

- [ ] **Step 2: Run llama-quantize for IQ1_M**

```bash
./build/bin/llama-quantize --allow-requantize \
  --imatrix tests/v4-port/calibration/imatrix-v4-flash.dat \
  --output-tensor-type q5_K \
  --token-embedding-type q5_K \
  "$V4_GGUF" \
  ~/models/DeepSeek-V4-Flash-IQ1_M.gguf \
  IQ1_M \
  2>&1 | tee /tmp/iq1m-build.log
```

Expected: ~10-15 min build. Output ~60 GiB / ~1.75 BPW.

- [ ] **Step 3: Verify file**

```bash
ls -lh ~/models/DeepSeek-V4-Flash-IQ1_M.gguf
```

- [ ] **Step 4: Sanity-load**

```bash
./build/bin/llama-cli \
  --model ~/models/DeepSeek-V4-Flash-IQ1_M.gguf \
  --jinja --reasoning off \
  -p "What is 2+2?" \
  -n 16 \
  -ngl 999 \
  --temp 0 \
  2>&1 | tail -20
```

- [ ] **Step 5: Commit progress note**

```bash
git commit --allow-empty -m "v4-port-J: IQ1_M build complete ($(stat -f%z ~/models/DeepSeek-V4-Flash-IQ1_M.gguf) bytes)"
```

---

### Task 4: Validate IQ1_M through gates

Identical to Task 2 but with `IQ1_M` GGUF.

- [ ] **Step 1: Run all three gates**

```bash
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_M.gguf ./tests/v4-port/gate-loader.sh
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_M.gguf NGL=999 ./tests/v4-port/gate-coherence.sh
V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_M.gguf ./tests/v4-port/gate-tools.sh
```

Same fallback rule: if `gate-tools` fails at Q5_K, bump to Q6_K and rebuild OR declare unviable.

- [ ] **Step 2: Record validation result**

```bash
cat > /tmp/iq1m-gates.txt <<EOF
IQ1_M validation results:
- gate-loader:    $(V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_M.gguf ./tests/v4-port/gate-loader.sh 2>&1 | tail -1)
- gate-coherence: $(V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_M.gguf NGL=999 ./tests/v4-port/gate-coherence.sh 2>&1 | tail -1)
- gate-tools:     $(V4_GGUF=~/models/DeepSeek-V4-Flash-IQ1_M.gguf ./tests/v4-port/gate-tools.sh 2>&1 | tail -1)
EOF
cp /tmp/iq1m-gates.txt tests/v4-port/results/iq1m-gates.txt
git add tests/v4-port/results/iq1m-gates.txt
git commit -m "v4-port-J: IQ1_M gate results recorded"
```

- [ ] **Step 3: Decide ship/no-ship per quant**

If both pass: ship both. Continue to Task 5.
If only one passes: ship the one that passes, document the other as unviable in the README update (Task 8). Continue to Task 5 with the passing quant only.
If neither passes: stop. The fix may need to be Q6_K output pin (rebuild both). If neither works at Q6_K either, declare 1.5-1.75 BPW the floor below which V4 tool-calling collapses; ship anyway as documented research artifacts and continue to Task 5.

---

### Task 5: Split IQ1_S into shards

**Files:**
- Create: `~/models/DeepSeek-V4-Flash-GGUF/IQ1_S/DeepSeek-V4-Flash-IQ1_S-NNNNN-of-NNNNN.gguf`

(Skip this task if Task 4 declared IQ1_S unviable.)

- [ ] **Step 1: Make output directory**

```bash
mkdir -p ~/models/DeepSeek-V4-Flash-GGUF/IQ1_S
```

- [ ] **Step 2: Run gguf-split**

```bash
./build/bin/llama-gguf-split \
  --split --split-max-size 50G \
  ~/models/DeepSeek-V4-Flash-IQ1_S.gguf \
  ~/models/DeepSeek-V4-Flash-GGUF/IQ1_S/DeepSeek-V4-Flash-IQ1_S
```

Expected: 1-2 shards (52 GiB / 50 GiB max-size). Output filenames include `-NNNNN-of-NNNNN`.

- [ ] **Step 3: Verify shards**

```bash
ls -lh ~/models/DeepSeek-V4-Flash-GGUF/IQ1_S/
du -sh ~/models/DeepSeek-V4-Flash-GGUF/IQ1_S/
```

Expected: total size ≈ original single file size.

- [ ] **Step 4: Sanity-load via the split shards (confirms split is well-formed)**

```bash
./build/bin/llama-cli \
  --model ~/models/DeepSeek-V4-Flash-GGUF/IQ1_S/DeepSeek-V4-Flash-IQ1_S-00001-of-*.gguf \
  --jinja --reasoning off \
  -p "Hi" \
  -n 8 -ngl 999 --temp 0 \
  2>&1 | tail -10
```

Expected: loads, generates text. Llama.cpp auto-loads remaining shards from the first.

- [ ] **Step 5: Delete the single-file IQ1_S to free disk**

```bash
rm ~/models/DeepSeek-V4-Flash-IQ1_S.gguf
df -h "$HOME/models" | head -2
```

---

### Task 6: Split IQ1_M into shards

(Skip this task if Task 4 declared IQ1_M unviable.)

- [ ] **Step 1: Make output directory**

```bash
mkdir -p ~/models/DeepSeek-V4-Flash-GGUF/IQ1_M
```

- [ ] **Step 2: Run gguf-split**

```bash
./build/bin/llama-gguf-split \
  --split --split-max-size 50G \
  ~/models/DeepSeek-V4-Flash-IQ1_M.gguf \
  ~/models/DeepSeek-V4-Flash-GGUF/IQ1_M/DeepSeek-V4-Flash-IQ1_M
```

- [ ] **Step 3: Verify shards**

```bash
ls -lh ~/models/DeepSeek-V4-Flash-GGUF/IQ1_M/
du -sh ~/models/DeepSeek-V4-Flash-GGUF/IQ1_M/
```

- [ ] **Step 4: Sanity-load via shards**

```bash
./build/bin/llama-cli \
  --model ~/models/DeepSeek-V4-Flash-GGUF/IQ1_M/DeepSeek-V4-Flash-IQ1_M-00001-of-*.gguf \
  --jinja --reasoning off \
  -p "Hi" \
  -n 8 -ngl 999 --temp 0 \
  2>&1 | tail -10
```

- [ ] **Step 5: Delete the single-file IQ1_M to free disk**

```bash
rm ~/models/DeepSeek-V4-Flash-IQ1_M.gguf
df -h "$HOME/models" | head -2
```

---

### Task 7: Upload both quants to Hugging Face

**Files (remote):**
- `teamblobfish/DeepSeek-V4-Flash-GGUF/IQ1_S/*.gguf`
- `teamblobfish/DeepSeek-V4-Flash-GGUF/IQ1_M/*.gguf`

The HF token must be available via the `HF_TOKEN` env var. The reference recipe and gotchas are documented at `~/work/blobfish/docs/hf-quant-uploads.md`.

- [ ] **Step 1: Verify HF auth and org access**

```bash
HF_TOKEN="$HF_TOKEN" hf auth whoami
```

Expected: `user=cchuter orgs=...,teamblobfish`. If `teamblobfish` is missing, refresh the token per `docs/hf-quant-uploads.md`.

- [ ] **Step 2: Upload IQ1_S shards (skip if unviable)**

```bash
HF_TOKEN="$HF_TOKEN" hf upload \
  teamblobfish/DeepSeek-V4-Flash-GGUF \
  ~/models/DeepSeek-V4-Flash-GGUF/IQ1_S \
  IQ1_S \
  --commit-message "Add IQ1_S shards (imatrix-calibrated, ~52 GiB)"
```

Expected: progress lines, final `url=https://huggingface.co/teamblobfish/DeepSeek-V4-Flash-GGUF/commit/<sha>`.

- [ ] **Step 3: Upload IQ1_M shards (skip if unviable)**

```bash
HF_TOKEN="$HF_TOKEN" hf upload \
  teamblobfish/DeepSeek-V4-Flash-GGUF \
  ~/models/DeepSeek-V4-Flash-GGUF/IQ1_M \
  IQ1_M \
  --commit-message "Add IQ1_M shards (imatrix-calibrated, ~60 GiB)"
```

- [ ] **Step 4: Verify upload via API**

```bash
HF_TOKEN="$HF_TOKEN" uv run --with huggingface_hub python3 -c "
from huggingface_hub import HfApi
info = HfApi().repo_info('teamblobfish/DeepSeek-V4-Flash-GGUF', repo_type='model', files_metadata=True)
total = 0
for f in sorted(info.siblings, key=lambda s: s.rfilename):
    if f.rfilename.startswith(('IQ1_S/', 'IQ1_M/')):
        sz = f.size or 0
        total += sz
        print(f'  {f.rfilename:60s}  {sz/1024**3:7.2f} GiB')
print(f'IQ1 total: {total/1024**3:.2f} GiB')
"
```

Expected: lists the new files; total IQ1 size matches local.

---

### Task 8: Update HF README with IQ1 rows

**Files (remote):**
- `teamblobfish/DeepSeek-V4-Flash-GGUF/README.md`

- [ ] **Step 1: Download the current README**

```bash
HF_TOKEN="$HF_TOKEN" hf download \
  teamblobfish/DeepSeek-V4-Flash-GGUF \
  --include "README.md" \
  --local-dir /tmp/v4-readme
```

- [ ] **Step 2: Edit the quant table**

Open `/tmp/v4-readme/README.md` in `$EDITOR`. Find the `## Available quants` section. Add two rows after the existing rows:

```markdown
| `IQ1_S/` | ~52 GiB (1-2 shards) | 1.5  | Smallest practical V4 quant; imatrix-calibrated against wikitext-103. Output/embed pinned at Q5_K to preserve tool-call grammar. **<insert: PASS or UNVIABLE per Task 4>** |
| `IQ1_M/` | ~60 GiB (1-2 shards) | 1.75 | Slight quality bump over IQ1_S; same Q5_K output/embed pin. **<insert: PASS or UNVIABLE per Task 4>** |
```

Replace `<insert: ...>` with the actual gate-tools result for each variant.

If only one quant shipped: only add that row; document the missing variant in a paragraph below the table:

> "IQ1_M was attempted but failed `gate-tools` at both Q5_K and Q6_K output pins, indicating tool-call grammar collapses below ~2 BPW for V4 even with calibrated imatrix data. Not shipped."

(Adjust phrasing to match what actually happened.)

- [ ] **Step 3: Add a calibration provenance note**

Find the "Provenance" section. Add a sentence about the imatrix corpus:

```markdown
IQ1_S and IQ1_M quants additionally use an imatrix calibration produced from the
`wikitext-103-raw-v1` test split (1000 chunks, ~1M tokens). Calibration data is
checked into the fork at `tests/v4-port/calibration/imatrix-v4-flash.dat`.
```

- [ ] **Step 4: Upload the updated README**

```bash
HF_TOKEN="$HF_TOKEN" hf upload \
  teamblobfish/DeepSeek-V4-Flash-GGUF \
  /tmp/v4-readme/README.md \
  README.md \
  --commit-message "Update quant table for IQ1_S + IQ1_M"
```

- [ ] **Step 5: Verify on the HF web UI**

Open `https://huggingface.co/teamblobfish/DeepSeek-V4-Flash-GGUF` in a browser. Confirm the quant table renders with the new rows and the provenance note shows. If the table is misformatted, fix and re-upload.

---

### Task 9: Final integration check + push branch

**Files:**
- None modified

- [ ] **Step 1: Confirm local validation results are committed**

```bash
git log --oneline -5
ls tests/v4-port/results/
```

Expected: recent commits show IQ1_S and IQ1_M gate results; results directory has both files.

- [ ] **Step 2: Push branch to remote**

```bash
git push -u mine feat/v4-port-J-iq1-quants
```

- [ ] **Step 3: Mark task complete in dev-team task tracker**

```bash
mkdir -p tasks/done
mv tasks/v4-port-J-iq1-quants.json tasks/done/ 2>/dev/null || true
git add -A tasks/
git commit --allow-empty -m "v4-port-J: mark done; move task JSON to tasks/done/"
git push mine feat/v4-port-J-iq1-quants
```

---

## Definition of done

- [ ] `IQ1_S` and `IQ1_M` GGUFs built from V4 Q8 source + Task I's imatrix data (or one declared unviable with documented reason)
- [ ] Both pass `gate-loader.sh`, `gate-coherence.sh`, `gate-tools.sh` (or unviable + documented)
- [ ] Local single-file artifacts split into 50 GiB shards under `~/models/DeepSeek-V4-Flash-GGUF/IQ1_S/` and `IQ1_M/`
- [ ] Both uploaded to `teamblobfish/DeepSeek-V4-Flash-GGUF` HF repo, total size matches local
- [ ] HF model card updated with IQ1 rows in the quant table + calibration provenance note
- [ ] Branch `feat/v4-port-J-iq1-quants` pushed to `mine`
- [ ] Codex plan-review and code-review APPROVE (per dev-team workflow)

## What this task explicitly does NOT do

- Validate IQ1 quality on Terminal-Bench or any agent benchmark — out of scope per spec
- Build IQ2_XXS / IQ2_M variants — separate future work
- Modify any source code in llama.cpp itself
- Re-run the imatrix calibration — that's Task I's deliverable
