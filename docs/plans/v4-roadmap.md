# DeepSeek V3.2 / V4 Roadmap

**Status:** planning artifact
**Branch this audits:** `deepseek-dsa` (fairydreaming PR #21149)
**Target hardware:** Apple M3 Ultra (Metal). NVIDIA boxes not available.
**Goal:** make V3.2 (DSA) usable on Metal, then add V4 (Flash) on top.

This doc has five sections (per `.claude/agents/v4-roadmap.md`):
1. V3.2 inventory
2. Metal gap analysis
3. V4 delta over V3.2
4. Phased task breakdown
5. Risks and unknowns

---

## 1. V3.2 inventory

`git diff --stat origin/master...HEAD` reports **34 files**, **+2376 / -103 lines**, on top of upstream master. The branch consists of ~72 commits, mostly fairydreaming's, with the most recent being upstream merges plus indexer-kernel WMMA optimizations (`ec083ceb4`, `cdf27d2f4`).

Grouped by component:

### 1a.1 Architecture / model registry

| File | Lines | What it adds |
|---|---|---|
| `src/llama-arch.h:79-83` | +1 | `LLM_ARCH_DEEPSEEK32` enum value (between `DEEPSEEK2OCR` and `CHATGLM`). |
| `src/llama-arch.cpp:75-80` | +1 | Registers `LLM_ARCH_DEEPSEEK32` → `"deepseek32"` arch name. |
| `src/llama-arch.cpp:887-895` | +1 | Adds `LLM_ARCH_DEEPSEEK32` to `llm_arch_supports_sm_tensor()` (shared-MoE-bias support). |
| `src/llama-model.h:130-145` | +1 | New `LLM_TYPE_685B_A37B` (DeepSeek V3.2 size class). |
| `src/llama-model.cpp:170-176` | +2 | `llama_model_mapping` dispatches `LLM_ARCH_DEEPSEEK32 → llama_model_deepseek32`. |
| `src/llama-model.cpp:777-781` | +1 | Pretty name `"685B.A37B"`. |
| `src/llama-model.cpp:1745-1755` | +1 | Pulls V3.2 into the existing DeepSeek2 print-info block. |
| `src/llama-model.cpp:1933-1957` | +18 | `create_memory()` branch for V3.2: builds a `llama_kv_cache_dsa`. |
| `src/llama-model.cpp:2244-2252` | +1 | Adds V3.2 to NEOX-rope arch list. |
| `src/CMakeLists.txt:24-31` | +1 | Compiles `llama-kv-cache-dsa.cpp` into `llama`. |

The arch is structurally a **DeepSeek V2 + DSA bolt-on**. V3.2's loading code reuses every DeepSeek2 hparam (`q_lora_rank`, `kv_lora_rank`, MLA dims, MoE expert counts, e-score correction bias, etc.) and adds *only* the indexer:

| GGUF key | C++ field | Source |
|---|---|---|
| `<arch>.attention.indexer_head_count` | `hparams.indexer_n_head` | `src/llama-hparams.h:206` |
| `<arch>.attention.indexer_key_length` | `hparams.indexer_head_size` | `src/llama-hparams.h:207` |
| `<arch>.attention.indexer_top_k` | `hparams.indexer_top_k` | `src/llama-hparams.h:208` |

### 1a.2 Conversion script

| File | Lines | What it adds |
|---|---|---|
| `convert_hf_to_gguf.py:830-836` | +2 | Forces `INDEXER_PROJ` to F32 (indexer is bf16 in HF, must stay full precision). |
| `convert_hf_to_gguf.py:9215-9358` | +143 | New `DeepseekV32Model` class (`@ModelBase.register("DeepseekV32ForCausalLM")`). |
| `gguf-py/gguf/constants.py:442-444` | +1 | `MODEL_ARCH.DEEPSEEK32` enum. |
| `gguf-py/gguf/constants.py:929-931` | +1 | Arch name string. |
| `gguf-py/gguf/constants.py:708-712` | +4 | Four new tensor IDs: `INDEXER_K_NORM`, `INDEXER_PROJ`, `INDEXER_ATTN_K`, `INDEXER_ATTN_Q_B`. |
| `gguf-py/gguf/constants.py:1194-1198` | +4 | Tensor name templates `blk.{bid}.indexer.*`. |
| `gguf-py/gguf/constants.py:2818-2860` | +37 | Tensor list for `MODEL_ARCH.DEEPSEEK32` (full DeepSeek2 set + indexer + NextN/MTP placeholders). |
| `gguf-py/gguf/constants.py:3987-3990` | +4 | `MODEL_TENSOR_SKIP[DEEPSEEK32]`. |

`DeepseekV32Model` differs from `DeepseekV2Model` in three places:
- it asserts `add_bos_token=true` in `tokenizer_config.json` (V3.2 requires explicit BOS),
- it writes `add_indexer_*` keys + `add_nextn_predict_layers` (MTP layers exist in the checkpoint but are skipped via `skip_mtp = True`),
- it does the same MLA `kv_b_proj → k_b_proj/v_b_proj` split as V2,
- experts are stacked by `(down|gate|up)_proj` exactly like V2.

### 1a.3 Forward pass / model graph

| File | Lines | What it adds |
|---|---|---|
| `src/models/deepseek32.cpp` | +510 (new) | Full V3.2 forward graph: MLA attention + DSA indexer path + MoE FFN. Key call sites: hparam read at line 31, `build_attn_inp_k_dsa()` call at line 204, indexer Q at line 228, `ggml_lightning_indexer` at line 309, `ggml_top_k` at line 358, DSA-aware `build_attn(...)` invocation at line 437. |
| `src/models/models.h:996-1006` | +14 | Defines `struct llama_model_deepseek32 : public llama_model_base` (full struct definition, not a forward decl) with a `build_arch_graph()` override (the project's standard graph entry point — see other `llama_model_*` overrides in the same file). |
| `src/llama-graph.h:22-26` | +1 | Forward-declares `llama_kv_cache_dsa_context`. |
| `src/llama-graph.h:355-393` | +38 | New `llm_graph_input_attn_k_dsa` class (mask + rope inputs for both MLA cache and indexer cache). |
| `src/llama-graph.h:901-1010` | +33 | Adds `top_k` parameter to existing `build_attn_mha()` and a new DSA-aware `build_attn(...)` overload. |
| `src/llama-graph.cpp` | +136 / -8 | Implements `build_attn_inp_k_dsa()` (definition at line 2589, `set_input` at line 498, `can_reuse` at line 510) and the DSA `build_attn(...)` overload (line 2367). The flow is: compute indexer Q+K, run `ggml_lightning_indexer` to get scores, take top-K, build a synthetic KQ mask via `ggml_set_rows` that unmasks the top-K rows, then call `ggml_flash_attn_ext` on the MLA path with the top-K indices attached via `ggml_flash_attn_ext_add_top_k` (`build_attn_mha` invocation with `top_k` at line 2428). |

The DSA path is gated *inside* the existing flash-attn op rather than being a separate sparse-attention op — flash-attn loads K/V using `top_k[i]` for row indexing when `src[5] != NULL`.

### 1a.4 New GGML ops (header + core)

Despite the high commit count for "GGML_OP_SCATTER" / "GGML_OP_HADAMARD" / "GGML_OP_WHERE_ID" in the history, all of those were **removed before merge**. The actual surface delta is concentrated in two files:

| File | Lines | What it adds |
|---|---|---|
| `ggml/include/ggml.h` | +13 | Adds `GGML_OP_LIGHTNING_INDEXER` enum (line 564), `ggml_flash_attn_ext_add_top_k(a, top_k)` mutator (lines 2408-2411), and `ggml_lightning_indexer(ctx, q, k, weights, scale_embd, scale_heads)` constructor (lines 2547-2553). |
| `ggml/src/ggml.c` | +59 / -2 | Implements the new ops. `GGML_OP_NAME` table entry at line 1066, `GGML_OP_SYMBOL` table entry at line 1177, both `static_assert(GGML_OP_COUNT == 97)` lines at 1084 and 1195. Extends `ggml_fill_impl` to accept `GGML_TYPE_F16` (line 5213, with the new assertion at line 5218). Implements `ggml_flash_attn_ext_add_top_k` to hang an i32 index tensor on `src[5]` of an existing `GGML_OP_FLASH_ATTN_EXT` node (line 5400). Implements `ggml_lightning_indexer` constructor (line 6244) that fuses `mul_mat + relu + weighted sum_rows` and outputs `[n_kv, n_head, 1, n_stream] f32`. |
| `ggml/include/ggml-rpc.h` | +2 / -2 | Bumps `GGML_RPC_PROTO_*` to reflect the new op (commit `5715c365a`). |

Removed (and not landed):
- `GGML_OP_SCATTER` — replaced by `ggml_set_rows()` with 1-element rows (commit `81209f9ba`).
- `GGML_OP_HADAMARD` — replaced by an explicit Hadamard-rotation matmul (commit `a7820f6db`).
- `GGML_OP_WHERE_ID` — replaced by the lightning-indexer fused op (commit `014e63cd9`).

So the **net new op count is one** (`LIGHTNING_INDEXER`) plus **one new feature on an existing op** (top_k on flash-attn).

### 1a.5 CPU backend implementations

| File | Lines | What it adds |
|---|---|---|
| `ggml/src/ggml-cpu/ops.cpp` | +109 | (a) `ggml_compute_forward_fill_f16` (line 2238) and dispatch update at line 2267; (b) `ggml_compute_forward_lightning_indexer` (line 11252; multi-stream, multi-batch, dequantizes K rows on the fly via `ggml_get_type_traits()->to_float`). |
| `ggml/src/ggml-cpu/ops.h` | +1 | `void ggml_compute_forward_lightning_indexer(...)` declaration. |
| `ggml/src/ggml-cpu/ggml-cpu.c` | +11 | Dispatch case `GGML_OP_LIGHTNING_INDEXER` in `ggml_compute_forward()` at line 2040; `n_tasks = n_threads` mapping at line 2363; per-thread `wdata` scratch sizing for K dequant in `ggml_graph_plan` at line 2947. |

The CPU path supports K-cache types: F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0 (via `ggml_get_type_traits()->to_float`).

### 1a.6 CUDA backend implementations

| File | Lines | What it adds |
|---|---|---|
| `ggml/src/ggml-cuda/lightning-indexer.cu` | +560 (new) | Lightning-indexer CUDA kernel suite. Three code paths: (1) WMMA Ampere+ kernel for prompt processing, (2) vector kernel for token generation with reduced shared mem (commit `ca6fc8eb1`), (3) F32 fallback. |
| `ggml/src/ggml-cuda/lightning-indexer.cuh` | +3 (new) | Header. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +5 | Includes the new header; dispatch case for `GGML_OP_LIGHTNING_INDEXER` at line 2956 → `ggml_cuda_op_lightning_indexer` at line 2957; adds the op to the `ggml_backend_cuda_device_supports_op` allow-list at line 5219. |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | +198 / -23 | Adds `use_top_k` template parameter to flash-attn MMA kernel. K/V loads use `KV + top_k[i]*stride` when enabled (`fattn-mma-f16.cuh:387-389`). Disables `cp_async` path under top_k (incompatible). |
| `ggml/src/ggml-cuda/fattn-common.cuh` | +20 / -1 | Plumbs `top_k` argument through the common flash-attn host-side launch helpers (`launch_fattn` etc.). |
| `ggml/src/ggml-cuda/fattn-tile.cuh` | +2 / -1 | Adds `use_top_k = false` parameter to the tile-kernel template signature for ABI alignment with MMA. |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | +2 / -1 | Same as tile.cuh — `use_top_k = false` template parameter on the vector flash-attn variant. |
| `ggml/src/ggml-cuda/fattn-wmma-f16.cu` | +2 / -1 | Same as tile.cuh — passes `use_top_k=false` from WMMA dispatch to keep the unified type signature. |
| `ggml/src/ggml-cuda/fattn.cu` | +17 | `ggml_cuda_fattn_ext_mma_f16_shall_use_top_k()` heuristic that opts into the slower-but-sparse path when DSA is active, plus dispatch wiring. |

### 1a.7 KV cache changes

| File | Lines | What it adds |
|---|---|---|
| `src/llama-kv-cache-dsa.h` | +138 (new) | Two classes: `llama_kv_cache_dsa` (composes two `llama_kv_cache` instances — one for the MLA latent, one for the indexer K), and `llama_kv_cache_dsa_context`. |
| `src/llama-kv-cache-dsa.cpp` | +261 (new) | Implements both. Key trick (lines 22-44): a private `hparams_lid` copy with `n_head_kv=1`, `n_embd_head_k_full=indexer_head_size`, `rope_type=NEOX` is fed to the second `llama_kv_cache` so it sizes its tensors for the indexer key cache without changing the public `llama_hparams`. |
| `src/llama-kv-cache-iswa.cpp` | +4 / -2 | Minor signature update for the new `const llama_hparams &` parameter on `llama_kv_cache::llama_kv_cache`. |
| `src/llama-kv-cache.cpp` | +9 / -1 | Constructor signature gains `const llama_hparams & hparams` (line 81), and the initializer-list now stores the passed hparams (line 94: `hparams(hparams)`) instead of `model.hparams`, so DSA can pass a tweaked copy. |
| `src/llama-kv-cache.h` | +1 / -1 | Public-API ripple of the constructor signature change (added `const llama_hparams &` parameter). |
| `src/llama-memory-hybrid.cpp` | +1 | Pass-through of the new `model.hparams` argument to the `llama_kv_cache` constructor. |

### 1a.8 Tests / examples

| File | Lines | What it adds |
|---|---|---|
| `tests/test-backend-ops.cpp` | +79 | `test_lightning_indexer` test case (8 K-types: F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, fixed shape `q=[128,64,128,1]`, `k=[128,1,256,1]`, `w=[64,128,1,1]`); F16 variants of `test_fill`. |
| `tests/test-llama-archs.cpp` | +7 / -2 | Adds `LLM_ARCH_DEEPSEEK32` to the parameterized arch tests (incl. `moe_mandatory()`); bumps fake indexer head dims to realistic 64×128 to avoid CUDA crashes (commit `eea1a6e1e`). |

---

## 2. Metal gap analysis

The Metal backend lives in `ggml/src/ggml-metal/`. Its op support is enumerated in `ggml-metal-device.m:1000-1300` (`ggml_metal_device_supports_op`).

### 2a. New op coverage

| Op | Metal kernel? | Closest existing analog | Complexity to port |
|---|---|---|---|
| `GGML_OP_LIGHTNING_INDEXER` | **No.** Searched `ggml-metal.metal`, `ggml-metal-device.m`, `ggml-metal-ops.cpp` — no occurrence. | `kernel_mul_mm_*` (heavy GEMM) chained with `kernel_relu` and `kernel_sum_rows`. The fused kernel is conceptually `relu(Q · Kᵀ) · weights` with a sum reduction over heads. | **M.** F32 reference path is straightforward; achieving CUDA-WMMA-class performance on Apple GPUs requires `simdgroup_matrix` (Apple's analog of WMMA). Quantized K dequant on the fly mirrors what `kernel_mul_mv_*` already does. |
| `ggml_flash_attn_ext_add_top_k` (sparse K/V indexing on `GGML_OP_FLASH_ATTN_EXT`) | **No** — Metal flash-attn does not consult `src[5]`. Two relevant code paths: (a) **non-vec** (prompt processing, batch ≥ 20): `kernel_flash_attn_ext` + `kernel_flash_attn_ext_impl` at `ggml/src/ggml-metal/ggml-metal.metal:5801-7235`, dispatched from `ggml_metal_op_flash_attn_ext` at `ggml/src/ggml-metal/ggml-metal-ops.cpp:2631-2956`; (b) **vec** (decode, batch < 20 — see `ggml_metal_op_flash_attn_ext_use_vec` at `ggml-metal-ops.cpp:2507-2515`): `kernel_flash_attn_ext_vec` + `_vec_reduce` dispatched at `ggml-metal-ops.cpp:2958-3045`. There is no separate "MLA" kernel — MLA reuses the same kernels with different head-dim template parameters. | Template instantiations covering head-dims 32/64/80/96/112/128/192/256 (and the MLA-typical 576) are emitted starting at `ggml-metal.metal:6498`. Each loads `K/V` linearly via `KV + i*stride`. | **L.** Two-step: (1) extend the **vec** kernel only — DSA's hot path is single-token decode, which routes to vec (`ne01 < 20`). Add an optional `top_k` buffer to `ggml_metal_kargs_flash_attn_ext_vec` (`ggml-metal-ops.cpp:2958`) and replace the linear K/V row index with `top_k[i]` when the buffer is bound. (2) Add the same to the non-vec path for prompt-processing correctness. The CUDA work in `fattn-mma-f16.cuh` disables `cp_async` under top_k for coalescing reasons; Metal has no `cp_async`, but threadgroup-memory prefetch via `simdgroup_load` has analogous bank-conflict considerations to verify. |

### 2b. Already covered by Metal

These do not require kernel work — V3.2's graph builds entirely on top of them:

| Op used in V3.2 forward | Metal status |
|---|---|
| `GGML_OP_FILL` (F32) | Supported (`ggml-metal-device.m:1019`). |
| `GGML_OP_FILL` (F16) | Already supported — the unary kernel in `ggml/src/ggml-metal/ggml-metal.metal:1064-1066` writes `dst_ptr[i0] = (T) args.val` where `T` is the destination tensor's element type. The host-side `args.val` setter in `ggml/src/ggml-metal/ggml-metal-ops.cpp:785-787` casts F32 → dst type. **Roadmap action: a verification task only — confirm `test_fill(*, GGML_TYPE_F16, …)` passes on Metal.** |
| `GGML_OP_SET_ROWS` (used to build the synthetic KQ mask from top-K) | Supported with F32/F16/BF16/Q* dst types (`ggml-metal-device.m:1257-1279`). |
| `GGML_OP_TOP_K` | Supported (`ggml-metal-device.m:1143`). |
| `GGML_OP_FLASH_ATTN_EXT` | Supported for F16/BF16/Q* K/V (`ggml-metal-device.m:1147-1186`); MLA variant exists. |
| `GGML_OP_MUL_MAT` / `GGML_OP_MUL_MAT_ID` | Supported (`ggml-metal-device.m:1195-1197`). |
| `GGML_OP_ROPE` (NEOX) | Supported (`ggml-metal-device.m:1115`). |
| `GGML_OP_ARGSORT` | Supported (`ggml-metal-device.m:1142`). |
| `GGML_OP_SUM_ROWS` | Supported (`ggml-metal-device.m:1098`). |

### 2c. Inferred kernel skeletons

For the lightning indexer Metal kernel, the closest reference is `kernel_mul_mm_*<f32, half>` in `ggml-metal.metal` (looped via `simdgroup_load`/`simdgroup_multiply_accumulate`). The CUDA WMMA tile size in `lightning-indexer.cu` is `32×8×16`; Apple's `simdgroup_matrix<float, 8, 8>` is the natural analog. A first-pass port should:
1. Implement an F32-K/F32-Q kernel correctness-first (matches `test_lightning_indexer(F32,F32,F32,…)`).
2. Add the F16 K-cache kernel (matches `test_lightning_indexer(F32,F16,F32,…)`).
3. Add Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 K-cache kernels via the existing `dequantize_q*_*` helpers.
4. Optimise the prompt-processing path with `simdgroup_matrix` once correctness lands.

For top-K flash-attn, **both** the vec and non-vec paths must be extended — the V3.2 graph computes `top_k` unconditionally and attaches it to every `ggml_flash_attn_ext` call (see `src/models/deepseek32.cpp:224, 358, 437` and `src/llama-graph.cpp:2002, 2428`), so prompt processing (which routes to the non-vec kernel via `ggml_metal_op_flash_attn_ext_use_vec` returning false for `ne01 ≥ 20`) is broken without it. The work is: (a) add an optional `top_k` buffer to both `ggml_metal_kargs_flash_attn_ext` (non-vec) and `ggml_metal_kargs_flash_attn_ext_vec` (vec) kernel-args structs, (b) gate it on `op->src[5] != NULL` in `ggml_metal_op_flash_attn_ext`, and (c) replace `K/V + i*stride` with `K/V + top_k[i]*stride` in both kernels.

---

## 3. V4 (Flash) delta over V3.2

**Confidence levels** are explicit because the only reference implementations are antirez's fork (rejected upstream) and unsloth's HuggingFace card. Every V4 claim below is **unverified** until checked against the official DeepSeek V4 release.

### 3a. Confirmed-against-HF-card

The following come directly from https://huggingface.co/unsloth/DeepSeek-V4-Flash:

- **284B total, 13B activated** for V4-Flash. 1.6T total / 49B active for V4-Pro.
- **1M-token context** target. V4-Pro reports 27% of V3.2 single-token FLOPs and 10% of V3.2 KV at 1M ctx — i.e. the new attention is ~3-10× more aggressive in compression than V3.2's DSA.
- **Hybrid Attention = Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA).** The card asserts CSA + HCA; their internal mechanics (V3.2-style top-K, compressed cache, ratios) are *not* spelled out in the card and are inferred in §3b.
- **FP4 + FP8 mixed precision:** routed experts in FP4, other params in FP8. Base versions FP8-only. The card does not specify whether FP4 is NVFP4 / E2M1 / shared-exponent — confirm at release.
- **Custom Python encoder is canonical.** The HF card includes a code sample that calls `encoding_dsv4.encode_messages(messages, thinking_mode=...)`. A Jinja template ships for compatibility, but the canonical path is the Python encoder. Specific encoder behaviour (special tokens, role separators) is hidden inside `encoding_dsv4.py` and must be ported by reading that file at release time — treat the function signature as confirmed and the implementation as TBD.
- **Manifold-Constrained Hyper-Connections (mHC).** Named by the card. Specific tensor names and hparams are *not* on the card — see §3b.
- **Muon optimizer at training time.** No inference-time impact.

### 3b. Inferred-from-antirez-fork (treat as hypothesis until checked)

All references in this subsection point to the *external* antirez fork (https://github.com/antirez/llama.cpp-deepseek-v4-flash) — none of those files exist in our tree. Inspected via a temporary local clone at `/tmp/antirez-v4/` during this audit. **Treat everything below as a hypothesis until verified against the official DeepSeek V4 release.**

- **mHC tensor and hparam names (hypothesis).** The external fork surfaces mHC as three new tensors per layer (`output_hc_base`, `output_hc_fn`, `output_hc_scale`) plus three GGUF hparams (`hyper_connection.count`, `hyper_connection.sinkhorn_iterations`, `hyper_connection.epsilon`). We will adopt these names by default; flag for re-mapping if the official release names differ.
- **HCA compress ratios (hypothesis).** External fork uses `compress_ratios = [2, 4]` (vector hparam). Specific ratios may differ at release.
- **No new GGML ops are required.** The external `src/models/deepseek4.cpp` (1392 lines) builds V4 entirely from existing primitives: `ggml_argsort_top_k`, `ggml_set_rows`, `ggml_fill`, `ggml_sum_rows`, `ggml_mul_mat`, `ggml_flash_attn_ext`. The new compression and hyper-connection logic lives in C++ helpers (`dsv4_hc_pre`, `dsv4_make_state_layout`, `dsv4_decode_compressor`).
- **HCA needs two parallel KV caches:** a raw-window cache (small, exact) and a compressed cache (larger context window, lossy). The external fork models this with `dsv4_state_pair { kv, score }` and three custom mask kinds (`RAW_WINDOW`, `COMPRESS_CAUSAL`, `ATTN_STATIC`). Likely needs a new `llama_kv_cache_v4` (analogous to `llama_kv_cache_dsa`) composed of *three* sub-caches: raw, compressed, indexer.
- **Indexer is the same as V3.2.** The external fork reuses `add_indexer_head_count / key_length / top_k`. So the lightning-indexer Metal kernel from Phase 1 is reusable for V4.
- **New GGUF keys (hypothesised)** beyond V3.2: `attention.compress_ratios` (vector), `attention.compress_rope_freq_base` (scalar), `attention.output_lora_rank`, `attention.output_group_count`, `hash.layer_count` (hash-routing), `hyper_connection.*`, `swiglu.clamp_exp`. Each will need an `LLM_KV_*` enum + a `gguf_writer.add_*` helper. Final names depend on the official release.
- **No new tokenizer model.** GPT-2 tokenizer with custom encoder logic. `set_vocab` is expected to call `_set_vocab_gpt2()` — same as V3.2.

### 3c. Open questions for V4 (must answer before Phase 3 starts)

| Question | How to answer |
|---|---|
| Does the official V4 release match antirez's fork on tensor names / hparams? | Diff antirez's `tensor_mapping.py` against the official `config.json` + `model.safetensors.index.json` once published. |
| Is FP4 a new GGUF quant type, or is it pre-dequantized at conversion time? | The external antirez fork uses an `_fp4_table` to dequantize FP4 → F32 at conversion time and writes IQ2/Q2_K. Native GGUF FP4 (E2M1 with shared exp) would be a separate, larger task. **For our roadmap: assume pre-dequant in Phase 2; native FP4 GGUF is a stretch goal.** |
| Does Mistral-style Jinja work for tool-calling, or is `encoding_dsv4` mandatory? | Test against the released model: encode the same multi-turn conversation both ways and diff token IDs. If they diverge, port `encoding_dsv4.py` into `examples/` or `tools/main`. |
| What chat template does `llama-server` ship by default for V4? | Likely needs a custom `chat_template` builder (the project uses `LLM_CHAT_TEMPLATE_DEEPSEEK_3` today — see `src/llama-chat.h:30`, `src/llama-chat.cpp:51`). Must add `LLM_CHAT_TEMPLATE_DEEPSEEK_4` following that convention. |
| Does mHC need a new GGML op? | **Probably not.** The external fork implements mHC purely as `mul_mat + scale + add` (see its `dsv4_hc_pre` helper). If perf is bad, a fused kernel becomes a Phase-3 stretch. |
| Is the compressed-cache compaction step graphable, or does it need a dedicated kernel? | The external fork does it via `ggml_set_rows` + `ggml_sum_rows`. Stays graphable. |

---

## 4. Phased task breakdown

Five phases. Within a phase, tasks can run in parallel; across phases, the dependency arrow is ordered. **Each task is independently shippable and lands its own PR/branch.**

#### Builder hand-off contract

For **every row** in the per-phase tables below, the builder phase MUST produce **two files**:

1. `.claude/agents/<task-id>.md` — the per-task spec. Should embed the row's title + scope verbatim, plus an "Acceptance criteria" section listing the test cases / file paths the task is expected to touch.
2. `tasks/active/<task-id>.json` — the dev-team task tracker, initialised with the schema below (state `roadmap`, no plan yet).

```json
{
  "id": "<task-id>",                                    // e.g. "v4-p1-metal-lightning-indexer"
  "title": "<row title>",                               // copy verbatim from the table
  "state": "roadmap",
  "branch": "feat/<task-id>",                           // always feat/ prefix
  "spec_path": ".claude/agents/<task-id>.md",
  "plan_path": null,
  "review_rounds": 0,
  "fix_attempts": 0,
  "history": [],
  "test_report": null,
  "errors": [],
  "in_progress": false
}
```

**Naming rule:** `<task-id>` is the exact id in the leftmost column of the tables. **Branch rule:** always `feat/<task-id>`. The builder must NOT omit a row, rename ids, or merge rows. If a row is judged inappropriate during builder phase, the builder must add a note to `errors[]` instead of silently dropping it.

### Phase 1 — Metal kernels for V3.2 (gets us a Metal-capable V3.2 first)

| id | title | scope | dependencies | complexity |
|---|---|---|---|---|
| `v4-p1-metal-lightning-indexer` | Implement `GGML_OP_LIGHTNING_INDEXER` on Metal | F32-Q F32-K and F32-Q F16-K reference kernels in `ggml-metal.metal`; dispatch in `ggml-metal-ops.cpp`; supports-list in `ggml-metal-device.m`. Mirrors `ggml/src/ggml-cuda/lightning-indexer.cu` semantics but no WMMA/simdgroup-matrix optimization yet. Adds the op to `ggml_metal_device_supports_op` for those two K-types. **Out:** quantized K paths (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/BF16), perf optimisation. | none | M |
| `v4-p1-metal-lightning-indexer-quant` | Quantized K-cache paths for lightning-indexer Metal kernel | Add Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/BF16 K-cache support by templating the kernel on the K-type and reusing the existing `dequantize_q*_*` / `dequantize_bf16` helpers in `ggml-metal.metal`. Each format needs its own template instantiation, supports-list entry, and dispatch case. Match all eight `test_lightning_indexer(F32, *, F32, …)` cases in `tests/test-backend-ops.cpp:8881-8895`. **Out:** simdgroup-matrix opt. | `v4-p1-metal-lightning-indexer` | M |
| `v4-p1-metal-lightning-indexer-simdgroup` | simdgroup-matrix optimisation of lightning-indexer | Promote the prompt-processing path to `simdgroup_matrix<float,8,8>` (Apple's WMMA analog). Target ≥50% of CPU baseline tok/s on M3 Ultra. **Out:** correctness changes. | `v4-p1-metal-lightning-indexer` | M |
| `v4-p1-metal-fattn-top-k` | Top-K / sparse K-V indexing in Metal flash-attn (vec **and** non-vec) | Extend **both** the vec and non-vec paths of Metal flash-attn. V3.2's graph (`src/models/deepseek32.cpp:224, 358, 437`; `src/llama-graph.cpp:2428`) attaches `top_k` to every `ggml_flash_attn_ext` call unconditionally, so prompt processing (which routes to non-vec when `ne01 ≥ 20` per `ggml-metal-ops.cpp:2507-2515`) is *broken* without non-vec coverage. Work: (1) add an optional `top_k` buffer to `ggml_metal_kargs_flash_attn_ext` and `ggml_metal_kargs_flash_attn_ext_vec` kernel-args structs; (2) plumb it from the dispatcher in `ggml_metal_op_flash_attn_ext` (`ggml-metal-ops.cpp:2631-3045`) when `op->src[5] != NULL`; (3) replace linear `K/V + i*stride` with `K/V + top_k[i]*stride` in `kernel_flash_attn_ext_impl` and `kernel_flash_attn_ext_vec` (`ggml-metal.metal:5801-…`); (4) add a `src[5] != NULL` clause to the flash-attn entry of `ggml_metal_device_supports_op` (`ggml-metal-device.m:1147-1186`). Mirrors `ggml/src/ggml-cuda/fattn-mma-f16.cuh:382-405` semantically. Test against `tests/test-llama-archs.cpp` for the `LLM_ARCH_DEEPSEEK32` arch on Metal. **Out:** changing op-args layout for non-DSA flash-attn callers. | none | L |
| `v4-p1-metal-fill-f16-verify` | Verify F16 `GGML_OP_FILL` on Metal | The unary kernel in `ggml-metal.metal:1064-1066` already writes `(T) args.val`, so F16 destinations should work without code changes. Task is purely a regression test: run `test_fill(*, GGML_TYPE_F16, …)` (4 cases at `test-backend-ops.cpp:8700-8703`) on the Metal backend and confirm green. If it fails, file a follow-up — do **not** rewrite the kernel inside this task. | none | S |
| `v4-p1-test-v32-end-to-end` | V3.2 Metal validation harness | Two-part validation. **Part A (synthetic, always-on):** extend `tests/test-llama-archs.cpp`'s `LLM_ARCH_DEEPSEEK32` test to run on Metal (today it only tests model loading; needs a token of forward pass to exercise the new kernels). The arch test machinery already builds an in-process tiny GGUF for the arch via `get_gguf_ctx` (`tests/test-llama-archs.cpp:99-200`), so no external checkpoint is needed. Compares Metal logits against CPU baseline to within `1e-3` RMS. **Part B (model-conditioned, skipped if absent):** add a `tests/test-deepseek32-cli.sh` harness that, given a real DeepSeek V3.2 GGUF in a known location (path passed via env var, e.g. `LLAMA_TEST_DEEPSEEK32_GGUF`), runs `./build/bin/llama-cli` on a fixed prompt with both `--device CPU` and `--device Metal` and `diff`s the first 32 token IDs. Skipped on CI / when env var unset. **Out:** any kernel work. | `v4-p1-metal-lightning-indexer-quant`, `v4-p1-metal-fattn-top-k`, `v4-p1-metal-fill-f16-verify` | M |

### Phase 2 — V4 architecture skeleton (loadable, even if forward pass is stubbed)

| id | title | scope | dependencies | complexity |
|---|---|---|---|---|
| `v4-p2-arch-enum` | Add `LLM_ARCH_DEEPSEEK4` enum and registration | New entries in `src/llama-arch.h` (enum), `src/llama-arch.cpp` (`LLM_ARCH_NAMES`, `llm_arch_supports_sm_tensor`), `gguf-py/gguf/constants.py` (`MODEL_ARCH.DEEPSEEK4`, name, tensor list — start with V3.2's set). New `LLM_TYPE_*` constants for V4-Flash (`284B.A13B`) and V4-Pro (`1.6T.A49B`). **Out:** any model logic. (No dep on Phase 1: arch-registry plumbing is independent of Metal kernel work.) | none | S |
| `v4-p2-tensor-ids` | New tensor IDs and GGUF keys for V4-specific weights | `gguf-py/gguf/constants.py`: add `OUTPUT_HC_BASE / FN / SCALE`, `*COMPRESS*` tensor names. Add `LLM_KV_HYPER_CONNECTION_*`, `LLM_KV_ATTENTION_COMPRESS_RATIOS`, `LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE`, `LLM_KV_ATTENTION_OUTPUT_LORA_RANK`, `LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT`, `LLM_KV_HASH_LAYER_COUNT`, `LLM_KV_SWIGLU_CLAMP_EXP`. Wire each to `add_*` helpers in `gguf-py/gguf/gguf_writer.py`. **Out:** runtime use of these — Phase 3. | `v4-p2-arch-enum` | M |
| `v4-p2-conversion-script` | `DeepseekV4Model` in `convert_hf_to_gguf.py` | Port the `DeepseekV4Model` class from the *external* antirez fork (https://github.com/antirez/llama.cpp-deepseek-v4-flash, `convert_hf_to_gguf.py` `DeepseekV4Model`): register `DeepseekV4ForCausalLM`, copy the FP4-dequant table (`_fp4_table`) and tensor renaming, write all the new hparams. **Pre-dequant FP4 → F32 at conversion time** (no native FP4 GGUF). **Smoke test input:** convert a single-layer slice (`--deepseek4-max-layers 1`) of the official HF DeepSeek V4 release once available (https://huggingface.co/unsloth/DeepSeek-V4-Flash), or a synthetic mock HF checkpoint built from `test-llama-archs` fixtures. `convert_hf_to_gguf.py` consumes HF safetensors, not pre-converted GGUFs — the antirez 80GB GGUF on huggingface.co/antirez is *not* a valid input. **Out:** quantization beyond what V3.2 supports. | `v4-p2-tensor-ids` | M |
| `v4-p2-model-loader-stub` | `llama_model_deepseek4` skeleton that loads tensors but throws on forward | New `src/models/deepseek4.cpp` and the corresponding `struct llama_model_deepseek4` definition in `src/models/models.h` (mirroring V3.2's pattern at lines 996-1006). Loads all V4 tensors; the `build_arch_graph()` override immediately throws "V4 forward not yet implemented". Unblocks gguf-conversion roundtrip testing. | `v4-p2-conversion-script` | S |

### Phase 3 — V4 forward pass (CPU + Metal)

| id | title | scope | dependencies | complexity |
|---|---|---|---|---|
| `v4-p3-hc-residual` | Manifold-Constrained Hyper-Connections forward path | Implement `dsv4_hc_pre`/`dsv4_hc_post` analogs as plain `mul_mat + scale + add` chains. No new ggml ops. Add `hyper_connection.sinkhorn_iters` loop. Tested on CPU first; Metal needs no new kernels. | `v4-p2-model-loader-stub` | M |
| `v4-p3-csa-attention` | Compressed Sparse Attention (V3.2-DSA-equivalent) | Reuse `llama_kv_cache_dsa` from V3.2 + Phase-1 Metal kernels. Wire indexer Q/K to V4's tensor names. **Out:** HCA path. | `v4-p3-hc-residual`, `v4-p1-metal-lightning-indexer-quant`, `v4-p1-metal-fattn-top-k`, `v4-p1-test-v32-end-to-end` | M |
| `v4-p3-hca-cache` | New `llama_kv_cache_v4_hca` for compressed-K/V cache | New header/source pair under `src/`. Two sub-caches (raw window + compressed) with a compaction step that merges old raw-window entries into the compressed cache via `ggml_set_rows + ggml_sum_rows`. **Out:** indexer cache (delegate to V3.2's). | `v4-p3-csa-attention` | L |
| `v4-p3-hca-attention` | Heavily Compressed Attention forward path | New mask kinds (RAW_WINDOW, COMPRESS_CAUSAL, ATTN_STATIC) inspired by the external antirez fork's `dsv4_mask_kind` enum (https://github.com/antirez/llama.cpp-deepseek-v4-flash, `src/models/deepseek4.cpp`). Two flash-attn calls fused via `ggml_concat` + `ggml_soft_max` over both raw + compressed K/V. **Out:** training. | `v4-p3-hca-cache` | L |
| `v4-p3-test-backend-ops` | Add V4 graph blocks to `tests/test-backend-ops.cpp` | Tests for the compaction op (set_rows + sum_rows roundtrip), HC residual block, and a small V4 attention block. **Out:** end-to-end model load. | `v4-p3-hca-attention` | S |
| `v4-p3-metal-perf-pass` | Profile V4 forward on Metal and address kernel hot spots | Run `llama-bench` on a converted V4 layer subset; identify which existing Metal kernels need rework (likely flash-attn's static cache size assumptions). **Out:** new ops — only optimisation of existing ones. | `v4-p3-test-backend-ops` | M |
| `v4-p3-output-routing` | V4 output projection (LoRA / grouped-output) and hash-routing | Wire the hypothesised `attention.output_lora_rank`, `attention.output_group_count`, and `hash.layer_count` keys (added in `v4-p2-tensor-ids`) into the runtime forward path: split the output projection into LoRA-A/LoRA-B mat-muls when `output_lora_rank > 0`, and gate the per-layer hash routing on `hash.layer_count`. Mirror the external antirez fork's handling. **Out:** non-router parts of FFN. If the official V4 release proves these keys are stale-from-antirez, the task is closed as no-op (its existence ensures we don't ship Phase-2 IDs that nothing reads). | `v4-p3-hc-residual` | M |
| `v4-p3-quant-pipeline` | End-to-end FP4/FP8 → GGUF quantization recipe for V4 on M3 Ultra | Define the quantization pipeline that takes the HF FP4+FP8 release and lands a GGUF that fits in 192 GB unified memory with KV cache headroom. Two stages: (a) `convert_hf_to_gguf.py` writes routed-expert tensors as F32 (post-FP4-dequant) and other tensors as F16 (post-FP8-dequant) — produces a ~568 GB intermediate that is **not** the shippable artifact; (b) the user runs `llama-quantize` to re-quantize routed experts to IQ2_XXS (or Q2_K / IQ2_XS — antirez's Recipe), other tensors to Q8_0, producing an ~80 GB final GGUF. Document the exact `llama-quantize` invocation and target sizes in `docs/plans/v4-quant-recipe.md`. **Out:** native FP4 GGUF type. | `v4-p2-conversion-script` | M |

### Phase 4 — Chat template / tool-calling

| id | title | scope | dependencies | complexity |
|---|---|---|---|---|
| `v4-p4-chat-template` | Add `LLM_CHAT_TEMPLATE_DEEPSEEK_4` | New entry in `src/llama-chat.h` (enum at line 28-32) and `src/llama-chat.cpp` (string mapping at line 49-52, plus matching block in `llm_chat_template_from_str` and `llm_chat_apply_template_impl`). Reproduces `encoding_dsv4.encode_messages()` semantics (system / user / assistant / `reasoning_content` interleaving). Add a fixture in `tests/test-chat.cpp` that compares a fixed message list against tokens produced by HF's `transformers.AutoTokenizer.encode(encoding_dsv4.encode_messages(...))`. **Independent of forward-pass and indexer test work** — chat-template plumbing only needs the V3.2-style tokenizer registration that ships with V3.2 (`_set_vocab_gpt2()`). | `v4-p2-conversion-script` | M |
| `v4-p4-thinking-mode` | Reasoning / `<think>` mode plumbing | Surface `thinking_mode` as a CLI flag for `llama-cli` and `llama-server`. Map to template variant in `LLM_CHAT_TEMPLATE_DEEPSEEK_4`. **Out:** changing the sampling pipeline. | `v4-p4-chat-template` | S |
| `v4-p4-tool-calling` | Function-calling format for V4 | Investigate whether DeepSeek V4 reuses V3.2's `<｜tool_calls_begin｜>` framing or introduces a new one. Implement in `tools/main` and `tools/server` (`tools/server/server.cpp` JSON parsing). Ship a `tests/test-tool-call.cpp` fixture against a known-good completion. | `v4-p4-chat-template` | M |

### Phase 5 — Validation against reference outputs

| id | title | scope | dependencies | complexity |
|---|---|---|---|---|
| `v4-p5-reference-logits` | Capture reference logits from a known-good runtime | Two-tier strategy because running 284B V4-Flash via HF `transformers` on FP16 CPU is not feasible on the M3 Ultra (would need ≈568 GB host RAM). **Tier A (preferred):** query the official DeepSeek V4 inference API for top-K logits / log-probs on a fixed plain-text prompt (no tool-calls; the API exposes log-probs in its OpenAI-compatible endpoint). Save the per-token top-K to a JSON fixture. **Tier B (fallback):** run a single-layer V4 slice via HF `transformers` on the same M3 Ultra (single-layer ≈ 600 MB FP16) and compare against our converted single-layer GGUF — covers correctness of the per-layer math even without a full-model reference. The reference prompt deliberately exercises only the chat-template path, **not** tool-calling, so this task depends on the chat template (so the encoded tokens are stable) but **not** on the V4 forward path (this is data collection from a *different* runtime). **Out:** any code in our tree. | `v4-p4-chat-template` | S |
| `v4-p5-logit-diff-test` | `tests/test-deepseek4-logits.cpp` against the reference blob | New test that loads a converted V4 GGUF, runs the same plain-text prompt as `v4-p5-reference-logits`, compares logits to within `1e-2` RMS. Skipped if the GGUF is absent. Requires the full V4 forward path to be working end-to-end (HCA + CSA + mHC), which means depending on the entire Phase-3 forward stack — not just the test-ops landing. Does **not** depend on tool-calling because the reference prompt does not exercise tool-call framing. | `v4-p5-reference-logits`, `v4-p3-hca-attention`, `v4-p3-test-backend-ops` | M |
| `v4-p5-tool-call-fixture` | End-to-end tool-call regression test | Once `v4-p4-tool-calling` lands, capture a reference tool-call completion from the official V4 API and add a fixture in `tests/test-tool-call.cpp`. Separated from `v4-p5-logit-diff-test` so the latter can land before tool-calling is stable. | `v4-p5-logit-diff-test`, `v4-p4-tool-calling` | S |
| `v4-p5-eval-suite` | Run a small eval (HumanEval-100 subset) and compare pass-rate | Verifies semantic correctness on top of the per-token logit check. **Out:** anything beyond a single eval. | `v4-p5-logit-diff-test` | M |
| `v4-p5-readme-and-howto` | `docs/deepseek4-howto.md` for users | Conversion + Metal-build + sample-run instructions. **Out:** marketing — this is engineering docs. | `v4-p5-eval-suite` | S |

---

## 5. Risks and unknowns

### 5a. Kernel-side risks (Phase 1 + 3)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Metal lacks a primitive equivalent to CUDA's `cp_async`. Sparse top-K loads on flash-attn could be slow. | High | The CUDA path *also* disables `cp_async` under top_k (`fattn-mma-f16.cuh:301`), so Metal isn't behind on this axis. Use threadgroup-memory prefetch via `simdgroup_load`. |
| `simdgroup_matrix` doesn't expose all the WMMA fragment shapes used in `lightning-indexer.cu` (32×8×16, 8×8×16). | Medium | Phase 1 ships an unoptimised simdgroup-matrix-free kernel first (`v4-p1-metal-lightning-indexer`); the simdgroup optimisation is a separate task (`v4-p1-metal-lightning-indexer-simdgroup`) so it can slip without blocking V3.2 functionality. |
| Compute-buffer sizing changes for Metal under DSA (top_k indices, indexer scores). | Medium | `ggml-metal-context.m` already auto-sizes from `ggml_graph_plan`; the new `n_tasks=n_threads` rule for `LIGHTNING_INDEXER` (`ggml-cpu.c:2360`) has no Metal counterpart — verify by running the test suite. |

### 5b. V4 architecture risks (Phase 2-3)

| Risk | Likelihood | Mitigation |
|---|---|---|
| antirez's fork diverges from official V4 release on tensor names / hparams. | High | Mark every Phase-2 task as "needs revisit on official drop." Don't commit to FP4-table values until confirmed against `model.safetensors.index.json`. |
| FP4 quantization scheme is essential for fitting V4 on M3 Ultra (192GB unified memory). | High | Pre-dequant to F16 (~568GB raw) is a non-starter. Use IQ2_XXS for routed experts following antirez's recipe. Stretch goal: native NVFP4 GGUF type. |
| HCA-cache compaction step has bugs that only surface at long context (>64k tokens). | Medium | Phase-3 tests should include a 256k+ token regression run (synthetic `aaaa...` tokens are fine). |
| mHC's sinkhorn iteration count or epsilon differs from antirez's choices. | Medium | These are GGUF hparams (`hyper_connection.*`), so they round-trip from the source checkpoint. Validate by logit-diff against HF reference (Phase 5). |

### 5c. Tooling / hardware gaps

| Gap | Impact | Workaround |
|---|---|---|
| No NVIDIA GPU on the dev box. | Can't regression-test that Phase-1 work doesn't break CUDA. | Rely on upstream CI on the eventual upstream PR. Until then, keep CUDA paths untouched in Phase-1 commits. |
| No CI for this fork. | Bugs caught only locally. | Run `ctest --test-dir build -L main --timeout 120` before every push. Document the smoke-test in `dev-team.json`. |
| No DeepSeek V4 weights yet (only antirez's repack). | Can't fully validate Phase 2-5 until release. | Phase 1 (V3.2 Metal) is independent and can ship now. Phases 2-5 are speculative until weights drop. |
| 192GB M3 Ultra is borderline for V4-Flash (284B in IQ2 ≈ 80GB; with KV cache and OS overhead, prompt processing at long ctx may OOM). | High at long context. | Bench memory before committing to a release-quality V4 build. May need to ship "V4-Flash but only up to 128k ctx" as the supported config. |

### 5d. Where we should expect to be wrong

- **The Phase-1 timeline.** Even with antirez's fork as a reference, Metal flash-attn forks tend to expose alignment bugs (Q/K/V stride mismatches between simdgroup and threadgroup memory) that aren't visible in CUDA. Expect 2× the estimated complexity on `v4-p1-metal-fattn-top-k`.
- **The "no new GGML ops for V4" claim.** Holds for antirez's fork but the official V4 might tighten HCA into a fused kernel, in which case we'd inherit that op. Treat the V4 forward graph as architecturally fluid until release.
- **The Mistral-template-works hypothesis.** Tool-calling formats have been the single biggest source of post-launch user pain across previous DeepSeek releases. Plan for Phase 4 slipping by a release cycle.
- **The V3.2-CPU-baseline number.** Lightning-indexer CPU code is single-threaded inside each row (`ops.cpp:11293-11315`), so its floor is low — Metal can beat it without much effort, which may make perf comparisons misleading. Compare against CUDA where possible.

---

*End of roadmap. Builder phase converts §1d into individual `tasks/active/v4-p*.json` + `.claude/agents/v4-p*.md` files.*
