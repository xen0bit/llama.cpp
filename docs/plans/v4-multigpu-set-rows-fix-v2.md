# V4 multi-GPU SET_ROWS fix — Plan v2 (second attempt, V4-graph-only)

Spec: `.claude/agents/v4-multigpu-set-rows-fix.md` (rewritten 2026-05-13)
Branch: `fix/v4-cuda-multigpu-supports-op` (baseline commit `19c3f8fd9`, kept in place)
Hard constraint: changes confined to `src/models/deepseek4.cpp`. No edits under `ggml/src/`.

## Why this attempt is different

Attempt 1 tried to patch ggml-cuda's `supports_op` to walk the view-src chain and read the destination buffer. That was a no-op because the scheduler invokes `supports_op` *before* it has allocated buffers for the intermediate K-cache state tensors, so `op->view_src->buffer` (and its chain root) is `nullptr` at query time. The commit (`19c3f8fd9`) is harmless and stays as defensive logic for other failure modes.

This attempt avoids the scheduler entirely by sidestepping the problematic op at the V4-graph level. `ggml_set_rows` is routed by source-device affinity; `ggml_cpy`-into-view is routed by destination-buffer affinity. The latter already works correctly in production via the `dsv4_store_state_segment` precedent (which has never been implicated in any multi-GPU crash log). We substitute `cpy`-into-view for `set_rows` at the two V4 sites that touch recurrent state / K-cache.

## Site inventory

`grep -n ggml_set_rows src/models/deepseek4.cpp` will turn up three call sites. Only two are in scope:

| # | Function                                       | Line(s) | In scope? |
|---|------------------------------------------------|---------|-----------|
| 1 | `dsv4_store_cache_rows`                        | 405-406 | YES       |
| 2 | `dsv4_build_compressor_decode_projected`       | 864-866 | YES       |
| 3 | Lightning indexer top-k write (`mask = ggml_set_rows(...)`) | ~1053 | NO — different pattern (sparse), not implicated in crash, spec explicitly excludes |

## Change 1 — `dsv4_store_cache_rows` (lines 391-407)

Replace the trailing two statements:
```cpp
ggml_tensor * rows = dsv4_arange_i32(ctx, row_start, row_start + n_rows);
ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache, src, rows));
```

with a single contiguous CPY-into-view, modeled on `dsv4_store_state_segment` (lines 375-389):
```cpp
// Avoid ggml_set_rows here: on multi-GPU, sched routes set_rows by SOURCE
// device, but the cache destination has its own device affinity → illegal
// memory access when those differ. ggml_cpy into a contiguous view of
// cache routes correctly by dst affinity (same pattern as
// dsv4_store_state_segment, which works in production multi-GPU).
ggml_tensor * cache_view = ggml_view_2d(ctx, cache,
        cache->ne[0], n_rows,
        cache->nb[1],
        row_start * cache->nb[1]);
ggml_build_forward_expand(gf, ggml_cpy(ctx, src, cache_view));
```

Semantics: `src` has already been made contiguous and reshaped to `[cache->ne[0], n_rows]` two lines above. The view spans rows `[row_start, row_start + n_rows)` of `cache` with stride `cache->nb[1]`. CPY of the matching-shape tensor into that view writes exactly the same bytes that `set_rows` would have, into the same destination addresses. Because we're writing CONTIGUOUS rows here (the `rows` argument was just `arange(row_start, row_start + n_rows)`), the row-stride view is equivalent — no indirection needed.

Drop the now-unused `rows` local. The `dsv4_arange_i32` import call disappears from this call site (still used elsewhere — site 2, indexer).

## Change 2 — `dsv4_build_compressor_decode_projected` (lines 864-866)

Single-row write at row index `row` (a scalar known at graph-build time). Replace:
```cpp
ggml_tensor * row_idx = dsv4_arange_i32(ctx, row, row + 1);
ggml_tensor * kv_state    = ggml_set_rows(ctx, prev_kv_state,    kv_cur, row_idx);
ggml_tensor * score_state = ggml_set_rows(ctx, prev_score_state, sc_cur, row_idx);
```

with:
```cpp
// Single-row write via cpy-into-view. set_rows would crash on multi-GPU
// (see dsv4_store_cache_rows for the same problem and fix).
auto cpy_into_row = [&](ggml_tensor * dst, ggml_tensor * src) -> ggml_tensor * {
    ggml_tensor * view = ggml_view_2d(ctx, dst,
            dst->ne[0], 1,
            dst->nb[1],
            row * dst->nb[1]);
    return ggml_cpy(ctx, src, view);  // returns a view-of-dst with op=CPY, src[0]=src
};

ggml_tensor * kv_state    = cpy_into_row(prev_kv_state,    kv_cur);
ggml_tensor * score_state = cpy_into_row(prev_score_state, sc_cur);
```

### Dependency-chain correctness

`ggml_cpy(ctx, src, dst)` returns a tensor whose `op = GGML_OP_CPY`, `src[0] = src`, `view_src = dst`. Downstream code at lines 869-894 reads from `kv_state` / `score_state` via `dsv4_view_cols(...)` (`ggml_view_2d`). Those new views have `view_src = kv_state` (the cpy result), and their own descendants ultimately bottom out in the cpy node when ggml walks the dataflow during scheduling.

Because `kv_state` IS the cpy result (not a fresh view of `prev_kv_state`), any consumer of `kv_state` in the forward graph creates a transitive dependency on the cpy node. The scheduler will therefore order the cpy before the consumer. This matches the dependency semantics of the original `ggml_set_rows` (which also returned a view-tagged-as-set_rows-op).

NOTE: The orchestrator's earlier draft attempted to wrap the cpy in `ggml_build_forward_expand(gf, ...)` and return a fresh view of `prev_kv_state`. That approach is broken on TWO counts:
1. The cpy result tensor would have no in-graph consumer, so the dependency would rely on the explicit forward_expand call.
2. `gf` is NOT plumbed into `dsv4_build_compressor_decode_projected` — it's not in the function signature. Trying to use it would not compile.

Returning the cpy result directly resolves both issues. The cpy will be visited via the normal graph traversal that the caller's `ggml_build_forward_expand(gf, ...)` already does on whatever consumes `kv_state` downstream.

### Edge cases checked

- `row_start = 0`: view offset is 0, valid for `ggml_view_2d`.
- `n_rows = 1`: same shape as site 2 case, valid.
- `src not yet contiguous`: site 1 already calls `ggml_cont` + `ggml_reshape_2d` before the write — no change. Site 2's `kv_cur`/`sc_cur` are mul_mat outputs, which are F32 contiguous `[width, 1]` tensors and match the view shape exactly. No additional `ggml_cont` needed.
- Non-contiguous row stride: cache->nb[1] is the natural row stride; the view we create has the same nb[1]. CPY preserves the dst layout, so this is fine for any cache->nb[1] (contiguous or padded).

## What is NOT changing

- Commit `19c3f8fd9` (view-chain walk in supports_op) — kept in place as harmless defense.
- Commits `13df7dfe3`, `5db52f6d9`, `1aed9b5e9`, `ca8734ab6` — kept (split-buffer source allowance + diagnostics + earlier dst-device check). They cover different failure modes from this bug.
- Line 1053 lightning-indexer set_rows — different pattern (sparse top-k mask write), not in the crash logs.
- `dsv4_store_state_segment` — already uses the correct cpy-into-view pattern.
- Any file under `ggml/src/`.

## Build & verify locally (Mac/Metal)

```
cmake --build build --target llama-server -j
```

Mac build only validates that the C++ compiles. CUDA-path correctness needs gpudual.

## Validation on gpudual (matches spec §"Validation plan")

1. `git fetch && git reset --hard origin/fix/v4-cuda-multigpu-supports-op` then `cmake --build build-cuda -j --target llama-cli test-backend-ops`. Expect a clean build.
2. `./build-cuda/bin/test-backend-ops -o DSV4_ROPE_TAIL,DSV4_HC_SPLIT_SINKHORN,DSV4_HC_WEIGHTED_SUM,DSV4_HC_EXPAND,DSV4_FP8_KV_QUANTIZE`. Expect 19/19 pass.
3. The repro script with `--model …Q2_K-XL…00001-of-00003.gguf`, `-ngl 999 -cmoe -ub 128 --ctx-size 8192 -p "hi" -n 20`, kill-timer `perl -e "alarm 600; exec @ARGV"`. Expect 20 generated tokens, no `SET_ROWS failed` / `CUDA error` / `Aborted` in output.

## Failure handling

If check 3 fails with a CPY-related error (e.g., `CPY failed` + illegal memory access), per the spec we DON'T retry V4-graph variations. We update the JSON state to `blocked-on-ggml-core-change` with the empirical finding "cpy also misrouted by sched" and stop. The user must then decide whether to relax the no-ggml-core constraint.

If check 3 fails with the same `SET_ROWS failed` somewhere we didn't expect, that means there's a third site we missed. Grep for `ggml_set_rows` in deepseek4.cpp again, double-check, and if it really is one of these two sites, mark blocked and report.

If check 1 or 2 fails (build broken or per-op regression), this is a coding error — fix the code, recommit, rerun. Don't escalate to ggml-core constraint.

## Commit message draft

```
v4: replace ggml_set_rows with ggml_cpy-into-view in compressor decode/store paths

On multi-GPU CUDA, ggml_set_rows is routed by the device affinity of its
SOURCES (kv_cur, row indices) after peer-copy, while the K-cache destination
is anchored to a different device, producing an illegal memory access when
they differ. ggml_cpy into a contiguous view of the destination routes by
dst affinity and works in production today (cf. dsv4_store_state_segment).

Substitute the working pattern at the two implicated V4 sites:
- dsv4_store_cache_rows: contiguous N-row write via ggml_view_2d + ggml_cpy.
- dsv4_build_compressor_decode_projected: single-row write via the same.

No ggml/src/ changes; this is a graph-construction fix only.

Co-Authored-By: cchuter <cchuter@yahoo.com>
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```
