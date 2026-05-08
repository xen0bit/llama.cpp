# V4 imatrix segfault diagnosis

## Reproducer
- Binary: `build-asan/bin/llama-imatrix` (ASan debug build of feat/v4-port-I-imatrix HEAD a6f3579cb)
- Model: `~/models/DeepSeek-V4-Flash-Q8_0.gguf` (~282 GiB)
- Calibration: `tests/v4-port/calibration/wikitext-tiny.txt` (50 paragraphs from wikitext-103-raw-v1 test split)
- Args: `--chunks 5 -ngl 999`
- Hardware: M3 Ultra (Metal backend)

## Crash report

```
common_init_result: fitting params to device memory, for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on
common_params_fit_impl: getting device memory data for initial parameters:
/Users/cchuter/work/llama.cpp.I-imatrix/ggml/src/ggml.c:3650: GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2) failed
0   ggml_print_backtrace + 160
1   ggml_abort + 240
2   ggml_reshape_3d + 184
3   llama_model_deepseek4::graph::graph(...) + 9208     <-- ASan symbol
4   llama_model_deepseek4::build_arch_graph(...) + 300
5   llama_model::build_graph(...) + 424
6   llama_context::graph_reserve(...) + 2644
7   llama_context::sched_reserve() + 2372
8   llama_context::llama_context(...) + 24776
9   llama_init_from_model + 5060
10  common_get_device_memory_data(...) + 1048
11  common_params_fit_impl(...) + 2128
12  common_fit_params(...) + 112
13  common_init_result(...) + 1416
14  common_init_from_params(...) + 560
15  llama-imatrix main + 2752
```

**Resolved source line (atos against build-asan dSYM):**

```
$ atos -o build-asan/bin/libllama.0.0.9159.dylib -arch arm64 0x66f9c8
llama_model_deepseek4::graph::graph(...) (deepseek4.cpp:1162)
```

`src/models/deepseek4.cpp:1162`:
```cpp
ggml_tensor * k_cache = mctx_swa->get_k(ctx0, il);
k_cache = ggml_reshape_3d(ctx0, k_cache, n_embd_head_k, 1, k_cache->ne[2]); // <- crashes here
```

The reshape requires `nelements(k_cache) == n_embd_head_k * 1 * k_cache->ne[2]`, but
`k_cache` actually has 4× more elements because the SWA KV cache is allocated
across 4 streams.

## Root cause

The crash is **NOT in the imatrix activation collector** as the spec hypothesized.
It is in V4's graph builder during the sched-reserve / fit-params probe, which
runs **before** any imatrix collection happens.

Causal chain:

1. `tools/imatrix/imatrix.cpp:1235-1243` (main()) sets
   `params.n_parallel = max(1, n_batch / n_ctx)`. With the imatrix defaults
   (`n_ctx=512`, `n_batch=2048`), this yields `n_parallel = 4`.
2. `common/common.cpp:1492` maps `cparams.n_seq_max = params.n_parallel = 4`.
3. `common/common.cpp:1147-1156` invokes `common_fit_params` which calls
   `common_get_device_memory_data` → `llama_init_from_model` to probe
   memory usage.
4. `src/llama-context.cpp:llama_context::sched_reserve()` calls `graph_reserve`
   to build a worst-case forward graph for budget estimation.
5. The V4 graph builder at `src/models/deepseek4.cpp:1162` calls
   `mctx_swa->get_k(ctx0, il)`, which (`src/llama-kv-cache.cpp:1162`) returns
   a 4-D view of shape `[n_embd_head_k, n_head_kv=1, n_kv, ns]`.
6. With `cparams.kv_unified == false` (imatrix default) and
   `cparams.n_seq_max == 4`, the SWA KV cache is allocated with `n_stream = 4`
   (`src/llama-kv-cache.cpp:95: n_stream(unified ? 1 : n_seq_max)`), so
   `ns = 4` and `nelements(k_cache) = head_dim * 1 * n_kv * 4`.
7. The reshape target `[head_dim, 1, n_kv]` only accounts for 1 stream, so the
   element-count assertion fails.

Stock `llama-cli` does **not** hit this path because its default
`params.n_parallel = 1`, so `n_seq_max = 1` → `n_stream = 1` → reshape OK.
Imatrix is the only tool whose `main()` overrides `n_parallel` to 4.

## Tensor at fault

Not a specific weight tensor — the crash site is the SWA K-cache **view**
returned by `llama_kv_cache::get_k(...)` for a V4 layer with
`compress_ratio == 0` (full-attention layers; ratio table contains zeroes).
Op = `GGML_OP_VIEW` (4-D), then `GGML_OP_RESHAPE` is what aborts.

Shape mismatch (with imatrix defaults `n_ctx=512, n_batch=2048`):

- `k_cache` shape:    `[n_embd_head_k, 1, n_kv, 4]`  (4 streams)
- reshape target:     `[n_embd_head_k, 1, n_kv]`     (assumes 1 stream)
- ratio: 4× off

## Fix strategy chosen

**Strategy 4 (NEW): force `kv_unified = true` in `tools/imatrix/imatrix.cpp::main()`.**

Rationale: the V4 graph builder at `deepseek4.cpp:1162` is correct under the
contract that `n_stream == 1` (which already holds for the rest of llama.cpp's
V4 callers — see the unconditional `n_seqs = 1` for `LLM_ARCH_DEEPSEEK4` at
`src/llama-context.cpp:402`). The bug is that imatrix raises `n_seq_max` to
keep its multi-chunk batching strategy (`n_parallel = n_batch / n_ctx`) but
never sets `kv_unified`, so it accidentally ends up with `n_stream = 4` for a
graph that hard-codes `n_stream = 1`.

`kv_unified = true` makes the KV cache use a single shared buffer regardless of
`n_seq_max`, which both:
1. Restores the `n_stream == 1` invariant the V4 graph builder requires.
2. Preserves imatrix's multi-chunk batching throughput (n_parallel still 4 →
   ubatch parallelism for collection still works; only the KV layout changes).

The patch is a single line in `imatrix.cpp::main()`:

```cpp
// V4's compressed-attention graph (src/models/deepseek4.cpp:1162) hard-codes
// n_stream == 1 for the SWA KV reshape; without this the multi-chunk
// batching strategy (n_parallel = n_batch / n_ctx) crashes graph_reserve.
// kv_unified=true is benign for non-V4 archs (single shared KV buffer).
params.kv_unified = true;
```

Smallest patch satisfying spec: 4 lines (1 line of code + 3 lines of comment).
No collector logic changes; activation coverage is unaffected.

## Strategies rejected

**Strategy 1 (I32-passthrough in collector):** Wrong layer. The crash is in
graph construction during `sched_reserve`, before any collector callback fires.
Adding `GGML_TYPE_I32` skips to `IMatrixCollector::collect_imatrix` would not
prevent the abort. The collector hook is registered via `params.cb_eval` at
`imatrix.cpp:1279`, which runs only during graph **execution**, not graph
**construction**. (The plan's hypothesis that V4 I32 lookup tensors trip the
collector may still be true at run time, but it is not the cause of *this*
crash, so a guard there is unnecessary and would be untested.)

**Strategy 2 (op-class skip in collector):** Same wrong-layer argument as 1.
The V4-specific ops (`LIGHTNING_INDEXER`, `DSV4_HC_*`, `DSV4_FP8_KV_QUANTIZE`)
do appear in the V4 graph, but the collector never sees them in this
reproducer because graph construction aborts before the graph is executed.

**Strategy 3 (MUL_MAT_ID layout fix in collector):** Same wrong-layer argument.
Also, V4's expert routing produces MUL_MAT_ID nodes whose `ids` shape matches
upstream (`[n_experts_used, n_tokens]`); inspection of the collector code
(lines 263-306) shows no V4-specific shape mismatch. The hypothesised crash
inside the expert unpacker does not occur.

**Alternative considered: force `n_parallel = 1` in imatrix for V4.** Works,
but reduces collection throughput by 4× on V4 (n_parallel controls ubatch
batching, which is the unit of parallel forward passes). `kv_unified = true`
preserves throughput at the cost of a slightly larger KV buffer.

**Alternative considered: detect arch via `llama_model_load_from_file` first,
then conditionally set `kv_unified`.** Adds a redundant load round-trip;
`kv_unified = true` is benign for non-V4 archs (same KV layout, just one
shared buffer instead of per-seq buffers — no correctness or perf delta for
the imatrix workload).

## Per-class coverage check (Task 5 implication)

Setting `kv_unified` does not skip any tensor classes — the activation
collection callback still runs over every `MUL_MAT` and `MUL_MAT_ID` node in
the V4 graph. The Task 5 per-class coverage gate (≥38 layers each for
attn_q_a, attn_q_b, attn_kv, attn_output_a, attn_output_b, ffn_gate_exps,
ffn_up_exps, ffn_down_exps) is expected to pass without modification.
