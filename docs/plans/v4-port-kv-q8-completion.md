# V4 Port — Q8 KV Cache Fix Completion

Follow-up to `docs/plans/v4-port-debug-completion.md`. Closes the deferred
"Underlying q8 KV bug" item.

## TL;DR

For DeepSeek4 (V4) models, the KV cache type is now pinned to
`GGML_TYPE_F16` regardless of the user's `--cache-type-k|v` flags. The
override is applied at the V4 case in `src/llama-model.cpp` where the
`llama_memory_hybrid_iswa` is constructed: both `params.type_k` and
`params.type_v` are replaced with `GGML_TYPE_F16` before construction, and a
single `LLAMA_LOG_WARN` is emitted on first allocation if the user
requested anything other than fp16. The model now produces coherent output
even when invoked with `--cache-type-k q8_0 --cache-type-v q8_0`.

A new gate `tests/v4-port/gate-server-chat-q8.sh` enforces this in `warn`
mode (asserts coherent output AND the WARN line in server stderr); wired
into `tests/v4-port/run-all-gates.sh`.

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

**Location:** `src/llama-model.cpp`, case `LLM_ARCH_DEEPSEEK4` in
`llama_model::create_memory()` (around line 1989).

**Change:** before constructing `llama_memory_hybrid_iswa`, override
`params.type_k` and `params.type_v` to `GGML_TYPE_F16`, log a single
`LLAMA_LOG_WARN` if the user requested anything else.

```cpp
// V4's standard SWA K cache, compressed-attention K cache (cache.attn_k),
// and indexer K cache (cache.index_k) all share the same `type_k` and
// must agree in dtype because src/models/deepseek4.cpp concatenates the
// SWA K view with the compressed K view via ggml_concat (which asserts
// a->type == b->type). Furthermore, V4's K activations are post-fp8-
// quantized (ggml_dsv4_fp8_kv_quantize), and q8_0's single fp16 scale
// per 32-element block cannot faithfully reproduce fp8-quantized value
// distributions -- pinning to q8_0 corrupts decode silently ("=" loops,
// "Mirror ..." garbage). Force fp16 unconditionally for V4 KV caches and
// log once if the user requested anything different.
ggml_type v4_type_k = GGML_TYPE_F16;
ggml_type v4_type_v = GGML_TYPE_F16;
if (params.type_k != v4_type_k || params.type_v != v4_type_v) {
    LLAMA_LOG_WARN("DeepSeek4: forcing fp16 KV cache (--cache-type-k|v are ignored for V4 because compressed/indexer K caches require fp16; "
                   "see docs/plans/v4-port-kv-q8-completion.md)\n");
}
```

This is the *correct* fix, not a workaround:

- A single arch-guarded point (case `LLM_ARCH_DEEPSEEK4`) covers all three
  V4 K caches because the `type_k` argument flows through
  `llama_memory_hybrid_iswa` to `mem_attn` (the standard SWA cache) AND is
  re-used inside the V4-specific allocation loop for `attn_k` / `index_k`.
- Both `llama-server` and `llama-completion` invoke `llama_init_from_model`
  → `llama_context` → the memory factory → this code path, so a single
  point covers both binaries (per the spec's bail-guard placement note).
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
    ./tests/v4-port/gate-server-chat.sh
OK[tiny-no-tools]: 45 tokens, 164 chars, 35 unique
OK[medium-no-tools]: 13 tokens, 56 chars, 26 unique
OK[tools]: 1 tool_call(s)
PASS: server-chat (3/3 tests coherent)
```

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
