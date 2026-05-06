---
license: other
license_name: deepseek-license
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4-Flash
base_model_relation: quantized
library_name: gguf
quantized_by: cchuter
language:
  - en
  - zh
tags:
  - gguf
  - llama.cpp
  - mixture-of-experts
  - moe
  - deepseek-v4
  - q2_k
  - q4_k_m
  - q8_0
  - sparse-attention
  - hyper-connections
pipeline_tag: text-generation
---

# DeepSeek V4 Flash — GGUF Quants (Q2_K, Q4_K_M, Q8_0)

Multiple quantizations of [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) for use with [llama.cpp](https://github.com/ggml-org/llama.cpp). DeepSeek V4 support is **not yet in upstream** — see "Required runtime" below.

## Files

| File | Quant | Size | BPW | Notes |
|---|---|---|---|---|
| `DeepSeek-V4-Flash-Q2_K-OutQ8.gguf` | Q2_K body, Q8_0 output + token-embed | 97 GiB | 2.92 | **Smallest available**. Tool calls verified working. Best for speed. |
| `DeepSeek-V4-Flash-Q4_K_M.gguf` | Q4_K_M body, Q6_K output | 160 GiB | 4.84 | Balanced quality/size. Standard recommendation. |
| `DeepSeek-V4-Flash-Q8_0.gguf` | Q8_0 | 282 GiB | 8.50 | High-precision reference. Required intermediate when re-quantizing to other targets via `llama-quantize --allow-requantize`. |

All files have the V4 chat template embedded (5015 chars) so `--jinja` works out of the box. Output and token-embedding tensors are kept at Q8_0 in the Q2_K mix (matching antirez's "OutQ8" recipe) to preserve special-token discrimination needed by V4's `<｜DSML｜tool_calls>` grammar trigger.

### Why no IQ-series quants (IQ2_M, IQ2_XXS, IQ1_M)?

Antirez's reference IQ2_XXS recipe uses `IQ2_XXS` for `ffn_{gate,up}_exps` and `Q2_K` for `ffn_down_exps`. The IQ2_XXS path requires an importance matrix (`imatrix`) for calibration — without it, `llama-quantize` refuses with `ERROR: this quantization requires an importance matrix!`.

Building an imatrix for V4 currently fails: `llama-imatrix` segfaults during the first chunk's forward pass on V4. Likely cause is a runtime bug where imatrix's per-tensor activation-collection hooks collide with V4's recurrent memory + hyper-connection graph (the scheduler reports `fused Gated Delta Net (autoregressive) enabled` even though V4 doesn't use GDN — suggesting tensor misclassification). Fixing this is tracked as a separate followup.

Until imatrix works, **Q2_K-OutQ8 is the smallest practical quant** — about 20% larger than antirez's 80 GiB IQ2_XXS recipe, but produced cleanly without imatrix.

SHA-256 checksums:

```
[FILL: paste from /tmp/v4-sha256.txt]
```

## Model architecture

DeepSeek V4 Flash is a 284B-parameter Mixture-of-Experts (MoE) language model:

- **Total parameters:** 284.33 B
- **Active per token:** ~32 B (out of 256 experts, top-6 routed + 1 shared)
- **Layers:** 43 transformer layers (40 with MoE; 3 standard)
- **Hidden size:** 4096
- **Attention heads:** 64 (head dim 512)
- **KV heads:** 1 (latent KV via MLA-style compression)
- **Indexer:** 64 heads × 128 dim, top-512 sparse attention via lightning indexer
- **Hyper-connections:** count=4, sinkhorn iterations=20, ε=1e-6
- **Vocabulary:** 129,280 tokens (GPT-2 style BPE)
- **Trained context:** 1,048,576 tokens (1M, with YaRN scaling beyond 65,536)
- **RoPE:** YaRN scaling factor 16, original max 65,536, base 10,000 (compressed RoPE base 160,000)
- **Expert gating:** sqrtsoftplus
- **Source dtype:** weights in FP8 e4m3 with FP8 e8m0 scales; routed experts in FP4

The architecture extends DeepSeek V3.2's Direct Sparse Attention (DSA) with V4-specific compressed attention, indexer compressors, hyper-connections, and FP4 routed experts.

## Required runtime

V4 support is **not yet in upstream `ggml-org/llama.cpp`**. You need a build from a fork that includes the V4 architecture port. The reference build for these GGUFs is [`cchuter/llama.cpp` at `feat/v4-port`](https://github.com/cchuter/llama.cpp/tree/feat/v4-port). The runtime port is itself derived from [`antirez/llama.cpp-deepseek-v4-flash`](https://github.com/antirez/llama.cpp-deepseek-v4-flash), restructured onto the upstream + V3.2/DSA baseline.

Build:

```bash
git clone https://github.com/cchuter/llama.cpp.git
cd llama.cpp
git checkout feat/v4-port
cmake -B build -DGGML_METAL=ON   # or -DGGML_CUDA=ON, -DGGML_BLAS=OFF, etc.
cmake --build build -j
```

## Hardware requirements

| Quant | Min unified RAM (loadable) | Recommended (with KV cache @ long ctx) | Notes |
|---|---|---|---|
| Q4_K_M | ~150 GiB | 192 GiB+ | Practical inference target; fits comfortably with 32k–128k context |
| Q8_0 | ~290 GiB | 384 GiB+ | Reference quality; needs more headroom for KV cache |

KV cache footprint scales with `n_ctx`. At 32k context, V4's compressed/indexer KV is small (~1.5 GiB total). At 1M context, expect ~45 GiB of KV cache.

These GGUFs were validated on an Apple M3 Ultra (512 GiB unified). Other hardware will work but performance will vary.

## Usage

### Single-prompt completion

```bash
./build/bin/llama-completion \
  -m /path/to/DeepSeek-V4-Flash-Q4_K_M.gguf \
  -ngl 999 \
  -p "The capital of France is" \
  -n 50 --temp 0.7 -no-cnv
```

### OpenAI-compatible server

```bash
./build/bin/llama-server \
  --model /path/to/DeepSeek-V4-Flash-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --jinja --reasoning-budget 0 \
  --ctx-size 131072 -ngl 999 --parallel 1 --flash-attn on \
  --threads-batch 32 \
  --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05
```

Then:

```bash
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"v4","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200}'
```

A reference launcher script is at `claude-cache-proxy/start-server-v4.sh` in the same fork.

### Recommended sampling

```
--temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.05
```

These match the values used during gate validation. The model also works at greedy (`--temp 0`) for deterministic output.

## Known caveats

### `--cache-type-k|v q8_0` is silently overridden to fp16

V4's compressed-attention K cache (`cache_dsv4_attn_k_l*`) and indexer K cache (`cache_dsv4_index_k_l*`) store latent (post-compression) representations whose value distribution doesn't match q8_0's per-block stationarity assumption. Quantizing them to q8_0 produces degenerate output (`=`-loops, single-character completions, or `"Mirror …"` garbage).

The fork's runtime detects DeepSeek V4 and pins these caches to fp16 unconditionally, with a one-time `LLAMA_LOG_WARN` on first allocation. The `--cache-type-k|v q8_0` flag is accepted for compatibility but ignored for V4-specific caches. KV memory is small enough that this is not a practical limitation.

### Reasoning-budget 0 starts an empty `<think></think>` block

The V4 chat template emits `<think>` as the assistant generation prompt. When `--reasoning-budget 0` is set, the runtime forces `</think>` to close it immediately. With recommended flags (above) this works correctly; if you observe the model emitting reasoning content before the `</think>`, the parser may attribute it to `content` rather than `reasoning_content`. Set `--reasoning-budget -1` to keep thinking enabled if you want full chain-of-thought output.

### Requires the V4-aware fork

This GGUF will not load with upstream `ggml-org/llama.cpp` until V4 lands there. Loading produces:

```
error loading model architecture: unknown model architecture: 'deepseek4'
```

Use the fork build instructed above.

## Performance

Decode speed measured via `tests/v4-port/gate-speed.sh` on Apple M3 Ultra (Metal, NGL=999):

| Quant | Decode tok/s |
|---|---|
| Q4_K_M | [FILL: paste from gate-speed output] |
| Q8_0 | [FILL: paste if measured] |
| IQ2XXS (reference) | 25.91 |

Performance is expert-routing-bound at decode time. Prefill scales with hardware bandwidth.

## Reproducibility

Build the same Q4_K_M from the original `deepseek-ai/DeepSeek-V4-Flash` safetensors:

```bash
# 1. Clone the V4-aware fork
git clone https://github.com/cchuter/llama.cpp.git && cd llama.cpp
git checkout feat/v4-port
cmake -B build -DGGML_METAL=ON && cmake --build build -j

# 2. Set up Python venv with FP8-aware torch + transformers <5
uv venv .venv --python 3.12
./.venv/bin/pip install --index-strategy unsafe-best-match \
  -r requirements/requirements-convert_hf_to_gguf.txt
./.venv/bin/pip install "transformers<5"

# 3. Clone base safetensors (~600 GiB)
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash ~/models/DeepSeek-V4-Flash

# 4. Convert HF -> GGUF Q8_0 intermediate (~30-60 min)
#    NOTE: --outtype q8_0, NOT f16. V4's FP4 routed experts force a compact intermediate;
#    f16 conversion is rejected by `_write_deepseek4_expert_tensors`.
./.venv/bin/python convert_hf_to_gguf.py ~/models/DeepSeek-V4-Flash \
  --outfile ~/models/DeepSeek-V4-Flash-Q8_0.gguf --outtype q8_0

# 5. Re-quantize Q8_0 -> Q4_K_M (~30-60 min)
./build/bin/llama-quantize --allow-requantize \
  ~/models/DeepSeek-V4-Flash-Q8_0.gguf \
  ~/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
  Q4_K_M

# 6. Validate
V4_GGUF=~/models/DeepSeek-V4-Flash-Q4_K_M.gguf ./tests/v4-port/run-all-gates.sh
```

For other quant levels (Q6_K, Q5_K_M, IQ4_XS, etc.), use `./tests/v4-port/build-quants.sh` after step 4 — re-quantizes the Q8_0 intermediate to any target.

## Acknowledgments

- **Base model:** [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- **Reference V4 inference / converter port:** [`antirez/llama.cpp-deepseek-v4-flash`](https://github.com/antirez/llama.cpp-deepseek-v4-flash) — produced the original IQ2XXS GGUF this work is built on top of
- **DSA / V3.2 baseline runtime:** [`fairydreaming/llama.cpp:deepseek-dsa`](https://github.com/fairydreaming/llama.cpp/tree/deepseek-dsa) (upstream PR [#21149](https://github.com/ggml-org/llama.cpp/pull/21149))
- **Upstream:** [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)

## License

Inherits the license of [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/LICENSE). Confirm the upstream license terms before commercial use.

## Issues

For runtime issues, file at [`cchuter/llama.cpp` issues](https://github.com/cchuter/llama.cpp/issues) (the V4-port fork) until V4 lands in upstream. Bug reports should include:
- The exact `llama-server` / `llama-completion` command
- The fork commit SHA (`git rev-parse HEAD` from your build dir)
- The first ~30 lines of `--verbose` output (model load + chat template detection)
- A minimal reproducer prompt
