# v4-port-H-quants: produce Q4_K_M GGUF for DeepSeek V4 Flash

## Goal
Port antirez's working V4 converter from `antirez/main` onto our `feat/v4-port`, the same way Task A-loader ported antirez's V4 runtime. Then run the converter on the cloned base safetensors at `~/models/DeepSeek-V4-Flash/`, quantize to Q4_K_M via `llama-quantize`, and validate against the existing V4 gate suite. Commit the converter changes; **do NOT commit the GGUF artifacts** (they live in `~/models/`, not in git).

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-05-v4-port-H-quants.md` — this is v4. Follow it as your starting point.**

The plan was pre-written, but **the architect is allowed to revise it** if codex plan-review finds issues. Standard dev-team pipeline: REVISE → architect updates plan → codex re-reviews → APPROVE or escalate after 2 rounds. Use the existing v1/v2/v3 codex history (preserved in the task JSON) as context for what's already been ruled out.

## What changed across plan versions
- **v1** designed the V4 converter from scratch (REVISE'd twice).
- **v2** wholesale-overlaid antirez's files (REVISE'd: would delete V3.2; `--outtype f16` unreachable).
- **v3** surgical V4-additive port + `--outtype q8_0` intermediate (REVISE'd for 3 underspecified pieces of the antirez touch surface).
- **v4** (this plan) enumerates the full antirez touch surface as explicit Task 3 substeps with line references: imports + `TORCH_FLOAT8_E8M0FNU` (3.A), `LazyTorchTensor` F8_E8M0 dtype maps (3.B), base-class `TextModel.set_gguf_parameters()` `sqrtsoftplus` branch (3.C), argparse `--deepseek4-*` flags (3.D), `ModelBase.__init__` signature (3.E), `ftype_map` iq2_xxs/iq2_xs/q2_k (3.F), `model_class(...)` call-site threading (3.G), `DeepseekV4Model` class (3.H), helper functions (3.I). Task 2 also expanded to include `MODEL_TENSOR.{ATTN_KV, ATTN_OUT_A, ATTN_OUT_B, FFN_GATE_TID2EID}` and `ExpertGatingFuncType.SQRTSOFTPLUS`.

## Gate (must pass before code-review)
After conversion + quantization, run:
```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  ./tests/v4-port/run-all-gates.sh
```
Expected: `ALL GATES PASS` (loader, coherence×2, speed, tools, server-chat). `gate-server-chat-q8.sh` is added by Task G later — it is NOT part of this task's gate.

## Conflict resolution rule (refined for v3)
Antirez branched before V3.2/DSA landed, so his diff *renames* DEEPSEEK32 → DEEPSEEK4 rather than adding alongside. Treat his V4 code as **additive over** our V3.2 baseline:
- **V4-specific code** (DeepseekV4Model, MODEL_ARCH.DEEPSEEK4, V4 tensor enums, V4 KV constants, V4 writer helpers) → **add alongside V3.2**, never replace
- **DSA / V3.2 / sparse-attention code** that already exists on `feat/v4-port` → **keep as-is**
- **Shared infrastructure** (e.g. helper functions, `_qtype_aliases` table, FP8 helpers antirez introduces) → **add antirez's; merge with any upstream additions**

After every port operation: explicitly verify BOTH `DEEPSEEK32` AND `DEEPSEEK4` coexist via Python import test. The plan's Task 2 Step 4 and Task 3 Step 4 enforce this.

## Preconditions (architect must verify before any work)
- `~/models/DeepSeek-V4-Flash/` exists and clone is complete (no `.lock` files, `config.json` present, `architectures: ['DeepseekV4ForCausalLM']`)
- ≥**500 GiB** free on the volume holding `~/models/` (lower than v2's 800 GiB because q8_0 intermediate is ~290 GiB, not f16's ~570 GiB)
- `cmake --build build -j` produces a clean `./build/bin/llama-quantize`
- `git remote get-url antirez` returns `https://github.com/antirez/llama.cpp-deepseek-v4-flash.git` and `git fetch antirez` succeeds
- V3.2 entries currently present in our files (`DEEPSEEK32` enum, `DeepseekV32Model` class) — they MUST survive the port

## Files (touched by surgical V4 port)
- `gguf-py/gguf/constants.py`: V4-only additions adjacent to V3.2 entries — MODEL_ARCH.DEEPSEEK4, V4 tensor enums (ATTN_COMPRESSOR_*, INDEXER_COMPRESSOR_*, INDEXER_*, HC_ATTN_*, HC_FFN_*, OUTPUT_HC_*), V4 KV constants, V4 entries in MODEL_ARCH_NAMES + MODEL_TENSORS + TENSOR_NAMES. V3.2 entries preserved.
- `gguf-py/gguf/gguf_writer.py` (likely): V4 writer helpers (`add_attention_compress_ratios`, `add_attention_output_lora_rank`, `add_attention_output_group_count`, `add_attention_compress_rope_freq_base`, `add_hash_layer_count`, `add_hyper_connection_*`)
- `convert_hf_to_gguf.py`: `DeepseekV4Model` class added adjacent to `DeepseekV32Model`, plus FP8/I8/FP4 dequant infrastructure for cloned safetensors (FP8 e4m3 attention weights with FP8 e8m0 scales, I8 + e8m0 for non-routed MoE shared-expert weights, FP4 routed-expert decode for the 256-expert path)

## What this task explicitly does NOT do
- Other quant levels (Q6_K, Q8_0, IQ4_XS). Trivial follow-ups once the f16 GGUF exists.
- Imatrix calibration.
- Terminal-bench end-to-end validation under Q4_K_M.
- Any V4-runtime code changes — converter only.

## Iteration budget
- Up to 8 builder fix rounds for conflict resolution / dequant bugs
- Up to 8 codex review rounds for plan-review and code-review (high reasoning effort, fall back to default on stalls per spec)

## Branch
- `feat/v4-port-H-quants` off `feat/v4-port`
- Worktree isolation required

## Definition of done
- `DeepseekV4Model` class registered and importable in `convert_hf_to_gguf.py` AND `DeepseekV32Model` still registered (V3.2 not regressed)
- `gguf.MODEL_ARCH.DEEPSEEK4` and all V4 tensor enums present AND `gguf.MODEL_ARCH.DEEPSEEK32` still present
- Q8_0 intermediate at `~/models/DeepSeek-V4-Flash-Q8_0.gguf` (≈290 GiB) — produced via `--outtype q8_0`
- `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` exists (≈150 GiB) — produced via `llama-quantize Q4_K_M` from the q8_0 intermediate
- `run-all-gates.sh` PASSES against the new Q4_K_M
- Decode tok/s recorded for the new quant
- `docs/plans/v4-port-quants-completion.md` committed with recipe (using `--outtype q8_0`) + sha256s + perf + conflict-resolution notes
- Codex code-review APPROVED
- Pushed to `mine`

## Cross-cutting ground rules
- Never push to `origin` (= ggml-org/llama.cpp upstream)
- Never run `gh pr create`
- Don't amend existing commits
- Don't skip git hooks
- 8+8 exhaustion → mark `needs-human`
