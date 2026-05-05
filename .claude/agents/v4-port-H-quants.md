# v4-port-H-quants: produce Q4_K_M GGUF for DeepSeek V4 Flash

## Goal
Port antirez's working V4 converter from `antirez/main` onto our `feat/v4-port`, the same way Task A-loader ported antirez's V4 runtime. Then run the converter on the cloned base safetensors at `~/models/DeepSeek-V4-Flash/`, quantize to Q4_K_M via `llama-quantize`, and validate against the existing V4 gate suite. Commit the converter changes; **do NOT commit the GGUF artifacts** (they live in `~/models/`, not in git).

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-05-v4-port-H-quants.md` — this is v2 of the plan after the v1 attempt was REVISE'd twice by codex at high reasoning effort. Follow it exactly.**

The plan was pre-written and corrected based on codex's findings. Per dev-team convention, the architect should adopt this plan as-is (set state directly to `plan-review` with `plan_path` pointing at the file above) rather than rewriting it. Codex plan-review will catch any remaining issues.

## What changed from v1
v1 of the plan attempted to design the V4 converter from scratch by inferring tensor mappings from the runtime. Codex's plan-review caught 4 fatal defects: missing gguf-py prerequisites, wrong GGUFWriter API, wrong GGUF KV key names, and wrong HF tensor naming patterns. v2 replaces the design-from-scratch approach with a port of antirez's already-working V4 converter — same playbook used for the runtime port.

## Gate (must pass before code-review)
After conversion + quantization, run:
```bash
V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  ./tests/v4-port/run-all-gates.sh
```
Expected: `ALL GATES PASS` (loader, coherence×2, speed, tools, server-chat). `gate-server-chat-q8.sh` is added by Task G later — it is NOT part of this task's gate.

## Conflict resolution rule (mirrors Task A)
When antirez's converter and our `feat/v4-port` (which already includes upstream + fairydreaming/dsa) both touched the same line:
- **V4-specific code** (DeepseekV4Model, V4 enums, V4 KV constants, V4 writer helpers) → **keep antirez's**
- **DSA / V3.2 / sparse-attention code** in surrounding context → **keep our `feat/v4-port`'s**
- **Shared infrastructure** (e.g. GGUFWriter helpers added near upstream additions) → **MERGE BOTH**

This is the same rule Task A used for the runtime cherry-pick.

## Preconditions (architect must verify before any work)
- `~/models/DeepSeek-V4-Flash/` exists and clone is complete (no `.lock` files, `config.json` present)
- ≥800 GiB free on the volume holding `~/models/`
- `cmake --build build -j` produces a clean `./build/bin/llama-quantize`
- `git remote get-url antirez` returns `https://github.com/antirez/llama.cpp-deepseek-v4-flash.git` and `git fetch antirez` succeeds

## Files (touched by antirez's V4 converter port)
- `gguf-py/gguf/constants.py` (~235 lines diff): MODEL_ARCH.DEEPSEEK4 enum, V4 tensor enums (ATTN_COMPRESSOR_*, INDEXER_COMPRESSOR_*, INDEXER_*, HC_ATTN_*, HC_FFN_*, OUTPUT_HC_*), V4 KV constants, MODEL_ARCH_NAMES + MODEL_TENSORS + TENSOR_NAMES entries
- Possibly `gguf-py/gguf/gguf_writer.py`: V4 writer helpers (`add_attention_compress_ratios`, `add_attention_output_lora_rank`, `add_attention_output_group_count`, `add_attention_compress_rope_freq_base`, `add_hash_layer_count`, `add_hyper_connection_*`)
- `convert_hf_to_gguf.py` (~1082 lines diff): `DeepseekV4Model` class (~245 lines) plus FP8/I8 dequant infrastructure for the cloned safetensors (which use FP8 e4m3 + FP8 e8m0 scales for weights and I8 + e8m0 scales for MoE experts)

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
- `DeepseekV4Model` class registered and importable in `convert_hf_to_gguf.py`
- `gguf.MODEL_ARCH.DEEPSEEK4` and all V4 tensor enums present in gguf-py
- `~/models/DeepSeek-V4-Flash-Q4_K_M.gguf` exists (≈150 GiB)
- `run-all-gates.sh` PASSES against the new Q4_K_M
- Decode tok/s recorded for the new quant
- `docs/plans/v4-port-quants-completion.md` committed with recipe + sha256s + perf + conflict-resolution notes
- Codex code-review APPROVED
- Pushed to `mine`

## Cross-cutting ground rules
- Never push to `origin` (= ggml-org/llama.cpp upstream)
- Never run `gh pr create`
- Don't amend existing commits
- Don't skip git hooks
- 8+8 exhaustion → mark `needs-human`
