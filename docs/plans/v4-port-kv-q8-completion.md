# V4 Port — Q8 KV Cache Fix Completion

Follow-up to `docs/plans/v4-port-debug-completion.md`. Closes the deferred
"Underlying q8 KV bug" item.

## TL;DR

For DeepSeek4 (V4) models, the KV cache type is now pinned to
`GGML_TYPE_F16` regardless of the user's `--cache-type-k|v` flags. The
override is applied early in `llama_init_from_model`
(`src/llama-context.cpp`), immediately after the existing Grok flash-attn
override and **before** any of the shared cache-type validations
(SPLIT_MODE_TENSOR + KV-quant rejection at `:3047-3050`,
flash-attn + V-cache-quant block-size checks at `:3053-3073`,
"V cache quantization requires flash_attn" at `:3075-3078`, and the
constructor's "quantized V cache requires Flash Attention" at `:351-355`).
A single `LLAMA_LOG_WARN` ("DeepSeek4: forcing fp16 KV cache ...") is
emitted at the early site if the user requested anything other than fp16.

The original coercion in `src/llama-model.cpp::create_memory()` is kept as
a defense-in-depth safety net (without the WARN, since it now fires once
at the earlier site) for any direct callers of `create_memory` that might
bypass `llama_init_from_model`.

The model now produces coherent output even when invoked with
`--cache-type-k q8_0 --cache-type-v q8_0`, AND -- per codex round-1 finding
-- requests like `--cache-type-k f16 --cache-type-v q8_0 --flash-attn off`
or under `SPLIT_MODE_TENSOR` no longer trip the shared validators with
misleading FA / quantization-not-implemented errors.

Two `tests/v4-port/gate-server-chat-q8.sh` modes enforce this:
* `MODE=warn` (default) -- `--cache-type-k|v q8_0 --flash-attn on` boots
  coherent and emits the WARN line.
* `MODE=warn-fa-off` -- `--cache-type-k f16 --cache-type-v q8_0
  --flash-attn off` boots coherent, emits the WARN line, and does NOT
  trigger the "V cache quantization requires flash_attn" diagnostic. This
  is the codex round-1 regression check.

Both are wired into `tests/v4-port/run-all-gates.sh`.

## Bisection (Phase 1 of the design spec)

V4 has three K caches in play under the `llama_memory_hybrid_iswa` umbrella
(see `src/llama-memory-hybrid-iswa.cpp` and `src/models/deepseek4.cpp`):

1. `mem_attn` — standard SWA K cache (allocated by `llama_kv_cache_iswa`,
   uses `params.type_k` from `--cache-type-k`).
2. `cache.attn_k` — V4-specific compressed-attention K cache
   (`src/llama-memory-hybrid-iswa.cpp:227`).
3. `cache.index_k` — V4-specific indexer K cache, ratio==4 layers only
   (`src/llama-memory-hybrid-iswa.cpp:231`).

In `src/models/deepseek4.cpp:1352`, the decode path concatenates the SWA K
cache view with the V4 `attn_k` cache view:

```cpp
k_all = ggml_concat(ctx0, k_raw, kv_comp_cache, 2);
```

`ggml_concat` asserts both inputs share the same dtype
(`ggml/src/ggml.c:2601` `GGML_ASSERT(a->type == b->type)`).

Bisect results (all against `$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf`,
`--cache-type-k q8_0 --cache-type-v q8_0`, three chat completions):

| Configuration | Outcome |
|---|---|
| `attn_k` F16, `index_k` q8_0, `mem_attn` q8_0 | ABORT: `ggml_concat` type mismatch (mem_attn q8_0 vs attn_k F16). Server never reaches main loop. |
| `attn_k` q8_0, `index_k` F16, `mem_attn` q8_0 | FAIL: degenerate decode (2 unique chars, e.g. `.\n.\n.`). `index_k` is not the operand of the SWA-vs-compressed concat (it's used in the indexer scoring path only); pinning it alone fixes nothing. |
| `attn_k` F16, `index_k` F16, `mem_attn` q8_0 | ABORT: same `ggml_concat` mismatch as case 1. |
| `attn_k` F16, `index_k` F16, `mem_attn` F16 | PASS: 3/3 coherent (`OK[tiny-no-tools-q8]`, `OK[medium-no-tools-q8]`, `OK[tools-q8]`). |

**Conclusion:** all three K caches must share the same fp16 dtype. The
plan's documented "fork point" is triggered: the standard SWA K cache
(`mem_attn`) is also implicated, not just the V4-specific compressed and
indexer caches. The fix must therefore live at the construction call site,
where one `type_k` argument flows to `mem_attn` and is re-used inside the
V4-specific allocation loop. Pinning at the call site subsumes the
in-allocator pin: the V4-specific caches naturally also become F16 because
they take the same `type_k` argument.

Working notes from the bisection are preserved at `/tmp/v4-G-bisect-summary.txt`.

## Root cause (Phase 2)

V4 stores K activations that have already been pushed through
`ggml_dsv4_fp8_kv_quantize` (see `src/models/deepseek4.cpp:1152` and
`:1229`). That fp8 representation is not equivalent to a per-block linear
scale: q8_0's single fp16 scale per 32-element block cannot faithfully
reconstruct fp8-quantized values. q8_0's quantize picks the scale
`d = amax / 127` from the absolute max in each block; if the fp8
distribution has rare large outliers (typical for fp8 quantization-aware
tensors) the scale fits the bulk values poorly and the int8 round() loses
information that the model's attention pathway relies on.

For V4 specifically, both `kv_comp` (the compressed/gated K state) and
`kv` (the post-fp8 K activation) store latent representations that violate
q8_0's per-block stationarity assumption. Two failure modes follow:

1. **Silent corruption** when all caches happen to share the same dtype:
   the model boots fine but decode produces `=` loops, single-character
   output, or scrambled multilingual gibberish.
2. **Hard abort** when caches are mixed dtypes: the in-graph
   `ggml_concat(ctx, k_raw, kv_comp_cache, 2)` fires
   `GGML_ASSERT(a->type == b->type)`.

Block alignment (`QK8_0 == 32`) is not the issue: typical V4 head dimensions
(`n_embd_head_k = 128/256` and `indexer_head_size`) are all multiples of 32.

## Fix landed (Phase 3 — chosen path)

**Primary location:** `src/llama-context.cpp`, in `llama_init_from_model()`
right after the existing `LLM_ARCH_GROK` flash-attn override (around line
3038, just before the `SPLIT_MODE_TENSOR` block).

**Change:** detect `model->arch == LLM_ARCH_DEEPSEEK4` immediately after
the model is loaded, coerce `params.type_k` and `params.type_v` to
`GGML_TYPE_F16`, and emit a single `LLAMA_LOG_WARN` if the user requested
anything else. The coercion runs **before** any of the shared cache-type
validations (`:3047-3050`, `:3053-3073`, `:3075-3078`, `:351-355`), so
those validators see the effective fp16 types for V4 and accept the
request.

```cpp
// V4 (DeepSeek4) requires fp16 KV cache: V4's standard SWA K cache,
// compressed-attention K cache (cache.attn_k), and indexer K cache
// (cache.index_k) all share the same `type_k` and must agree in dtype
// because src/models/deepseek4.cpp concatenates the SWA K view with the
// compressed K view via ggml_concat (which asserts a->type == b->type).
// Furthermore, V4's K activations are post-fp8-quantized
// (ggml_dsv4_fp8_kv_quantize), and q8_0's single fp16 scale per 32-element
// block cannot faithfully reproduce fp8-quantized value distributions --
// pinning to q8_0 corrupts decode silently ("=" loops, "Mirror ..."
// garbage). Coerce here, before the SPLIT_MODE_TENSOR / FA / V-quant
// shared validations below and before the constructor's flash_attn check,
// so those validations see the effective fp16 types and won't reject V4
// requests with --cache-type-k|v q8_0.
if (model->arch == LLM_ARCH_DEEPSEEK4) {
    if (params.type_k != GGML_TYPE_F16 || params.type_v != GGML_TYPE_F16) {
        LLAMA_LOG_WARN("DeepSeek4: forcing fp16 KV cache (--cache-type-k|v are ignored for V4 because compressed/indexer K caches require fp16; "
                       "see docs/plans/v4-port-kv-q8-completion.md)\n");
        params.type_k = GGML_TYPE_F16;
        params.type_v = GGML_TYPE_F16;
    }
}
```

**Why this location, not `create_memory()`?** Per codex round-1 review,
the prior placement in `src/llama-model.cpp::create_memory()` only coerced
the values *passed into* `llama_memory_hybrid_iswa`; it did **not**
normalize the user-facing `params.type_k`/`type_v` that the rest of the
shared context init still reads. As a result, V4 still behaved as
"quantized KV requested" in:

* `src/llama-context.cpp:351-353` — constructor: with `--flash-attn off`
  and a quantized `type_v`, the request was rejected with
  *"quantized V cache was requested, but this requires Flash Attention"*
  even though V4 would have pinned to fp16.
* `src/llama-context.cpp:3047-3050` — `SPLIT_MODE_TENSOR` + KV
  quantization combination was rejected with
  *"simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not
  implemented"*.
* `src/llama-context.cpp:3075-3078` — analogous V-cache + no-FA rejection.

Moving the coercion into `llama_init_from_model` ensures those shared
validations operate on the effective fp16 types for V4, so the
"--cache-type-k|v are ignored for V4" behaviour is propagated cleanly
to both `llama-completion` and `llama-server`.

**Defense-in-depth:** the original coercion in
`src/llama-model.cpp::create_memory()` is kept as a safety net (the local
`v4_type_k` / `v4_type_v` are still pinned to fp16 inside the
`LLM_ARCH_DEEPSEEK4` case before the `llama_memory_hybrid_iswa`
constructor). The duplicate WARN there has been removed because it now
fires once at the earlier site. This protects any direct caller of
`create_memory` that bypasses `llama_init_from_model`.

This is the *correct* fix, not a workaround:

- A single arch-guarded point covers all three V4 K caches because the
  `type_k` argument flows through `llama_memory_hybrid_iswa` to `mem_attn`
  (the standard SWA cache) AND is re-used inside the V4-specific
  allocation loop for `attn_k` / `index_k`.
- Both `llama-server` and `llama-completion` invoke `llama_init_from_model`
  → `llama_context` → the memory factory → this code path, so a single
  point covers both binaries (per the spec's bail-guard placement note).
- The coercion now runs **before** the shared validators, so
  `--cache-type-k f16 --cache-type-v q8_0 --flash-attn off` (and
  `SPLIT_MODE_TENSOR` with a quantized cache type request) on V4 are
  accepted with a WARN instead of being rejected with misleading error
  messages.
- No changes to `llama_memory_hybrid_iswa` internals, no kernel changes, no
  scattered checks per cache.

Memory cost: at 32k context, fp16 ups the per-layer cache size vs q8_0 by
~2x, but the V4 cache footprint is small relative to weights — negligible
on the M3 Ultra used for validation.

### Alternatives considered

- **Kernel-level fix** (Phase 3 option A): would require either changing
  q8_0's per-block stationarity assumption (broader llama.cpp change,
  rejected) or a V4-specific quantization scheme that handles fp8-shaped
  inputs (out of scope; would also need new metal kernels).
- **CLI bail-guard** (Phase 3 option C): error fast at startup when
  `--cache-type-k` or `--cache-type-v` is non-fp16 for V4 GGUFs. Rejected
  because it forces users to know about an internal detail; the silent
  override + WARN is friendlier and matches what the runtime is already
  doing for `params.flash_attn` defaults.
- **In-allocator pin** (`src/llama-memory-hybrid-iswa.cpp`): would only fix
  `attn_k` / `index_k`, leaving `mem_attn` (standard SWA K) at the user's
  `type_k`. As bisect step 3 shows, this still aborts on `ggml_concat`
  type mismatch.

## Validation

```text
$ V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
    ./tests/v4-port/gate-server-chat-q8.sh
OK[tiny-no-tools-q8]: 45 tokens, 164 chars, 35 unique
OK[medium-no-tools-q8]: 13 tokens, 56 chars, 26 unique
OK[tools-q8]: 1 tool_call(s)
PASS: server-chat-q8 (warn; coherent + override WARN observed)

$ V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
    MODE=warn-fa-off ./tests/v4-port/gate-server-chat-q8.sh
OK[tiny-no-tools-q8]: 45 tokens, 164 chars, 35 unique
OK[medium-no-tools-q8]: 13 tokens, 58 chars, 26 unique
OK[tools-q8]: 1 tool_call(s)
PASS: server-chat-q8 (warn-fa-off; coherent + override WARN + no shared-validator rejection)

$ V4_GGUF=$HOME/models/DeepSeek-V4-Flash-Q4_K_M.gguf \
    ./tests/v4-port/gate-server-chat.sh
OK[tiny-no-tools]: 45 tokens, 164 chars, 35 unique
OK[medium-no-tools]: 13 tokens, 56 chars, 26 unique
OK[tools]: 1 tool_call(s)
PASS: server-chat (3/3 tests coherent)
```

The `warn-fa-off` mode is the codex round-1 regression check: it asserts
the V4 coercion runs **before** the shared "V cache quantization requires
flash_attn" validator, so `--cache-type-v q8_0 --flash-attn off` is
accepted (server boots, WARN fires, decode is coherent) instead of being
rejected.

`gate-loader.sh`, `gate-tools.sh`, `gate-speed.sh`, and
`gate-coherence.sh NGL=999` also pass against this branch.

### Pre-existing baseline issue noted (not introduced by this fix)

`gate-coherence.sh NGL=0` (CPU-only decode) FAILS on Q4_K_M with the same
"repetitive decode (top word 4/7)" output (`The capital of France is Paris.,
Paris., Paris., Paris.,`). This failure reproduces on the parent branch
`feat/v4-port` HEAD before any of this PR's commits, so it is **pre-existing
behavior of the Q4_K_M GGUF on CPU-only decode** and is not a regression
introduced by the q8 KV fix. The original V4 debug validation
(`docs/plans/v4-port-debug-completion.md`) was performed against the IQ2XXS
GGUF (`...chat-v2.gguf`); on Q4_K_M the CPU decode at temp=0 lands in a
local repetition mode for the "capital of France" prompt. NGL=999 (Metal)
on Q4_K_M coherence still PASSes both before and after the fix.

This is documented for visibility — addressing it (by tweaking the prompt,
relaxing the threshold, or fixing the underlying CPU-decode Q4_K_M
repetition behaviour) is out of scope for the q8 KV fix.

## What we explicitly did NOT touch

- `ggml/src/ggml-cpu/quants.c` and `ggml/src/ggml-metal/ggml-metal.metal`
  — q8_0 kernels are correct for tensors that meet per-block stationarity;
  V4's compressed K simply doesn't.
- The `--cache-type-k|v` CLI surface — still accepted, still applies to
  non-V4 archs; only V4 overrides silently with WARN.
- `src/llama-memory-hybrid-iswa.cpp` itself — the in-allocator allocation
  is unchanged; it still uses the `type_k` parameter, which the call site
  now coerces to fp16 for V4.

## Future work

If a future variant of V4 ships with quantization-friendly compressed KV
(e.g. with explicit per-block normalization), revisit the pin. Until then,
fp16 is the right answer.
