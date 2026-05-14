# Plan: v4-rowsplit-perf — Per-device MUL_MAT_ID for V4 decode

## Problem

PR #2 added a gather-to-main-device fallback for cuda_split MUL_MAT_ID (`ggml-cuda.cu:2660-2755`). It is correct but transfers ~80 GB of expert weights over PCIe per token, dropping decode from 19.4 t/s (layer-split) to 0.5 t/s (row-split).

## Approach (chosen from three design consults with codex)

After three rounds with codex (`/tmp/v4-rowsplit-perf/design-consult-{1,2,3}-output.txt`), the chosen approach is **targeted MMVQ split fast path**:

1. Detect `cuda_split src0` + V4-decode-shaped MMID (quantized, `dst->ne[2] == 1`, `dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)`, hits `ggml_cuda_mul_mat_vec_q`).
2. Validate the split lies on whole-expert boundaries (`row_low % ne01 == 0 && row_high % ne01 == 0`). If not, fall through to the existing gather path.
3. Per-device dispatch:
   - Each GPU launches `mul_mat_vec_q` over its local slab.
   - Local slab interpretation: `ne01` unchanged (rows per expert), `ne02 = expert_high - expert_low` (subset of experts).
   - Kernel-side change: accept `expert_low/expert_high` params. Skip `(channel_dst, token_idx)` cells whose `ids[...]` is outside the local range; for in-range cells, use `channel_x = ids[...] - expert_low`.
   - Output: write to a per-device dst temp (full shape `[ne0, n_expert_used, n_tokens]`), pre-zeroed. In-range cells contribute the correct value; out-of-range cells are skipped (temp stays zero there).
4. Aggregation on `ctx.device`:
   - Zero `dst`.
   - Each device's temp is peer-copied to `ctx.device` scratch, then added into `dst` with a small F32 add kernel.
5. Keep gather fallback for MMQ/MMF/MMVF paths and for non-aligned splits.

Why this scope:
- V4 single-token decode hits `ggml_cuda_mul_mat_vec_q` (codex verified: `ncols_dst = ne2 = n_tokens`, generic kernel for `ncols_dst == 1`). Optimizing only this path covers the t/s decode metric.
- Prefill (multi-token) goes to MMQ or `mul_mat_vec_q_moe`. Acceptable to leave on the gather fallback for this PR — prefill perf isn't the decode metric, and the gather fallback was already validated by codex on PR #2.
- Total touched code: ~150-200 LOC in `mmvq.cu` + `ggml-cuda.cu`. No allocator changes, no buffer extensions, no MMQ/MMF/MMVF kernel mods.

Codex Q5 verdict from third consult: "Choose (a) in spirit, but implement it as a targeted MMVQ expert-range guard, not dst_row_offset."

## Files touched

### 1. `ggml/src/ggml-cuda/mmvq.cu`

#### A. `mul_mat_vec_q` kernel (~line 396)
Add template/runtime params `uint32_t expert_low, expert_high` (use sentinel `0xFFFFFFFF` or runtime branch to keep the non-split fast path identical). For `ids != nullptr`:
```cpp
const uint32_t global_id = ids[channel_dst]; // ncols_dst==1 path uses ids[channel_dst]
if (expert_high != 0 /*split active*/) {
    if (global_id < expert_low || global_id >= expert_high) return;
    channel_x = global_id - expert_low;
} else {
    channel_x = global_id;
}
```

Caveat: kernel is templated and the non-split path is hot. Use a `bool split` template param or branchless arithmetic, validated by codex review. Default to non-split for ABI compat.

#### B. `mul_mat_vec_q_switch_fusion`, `mul_mat_vec_q_switch_ncols_dst`, `mul_mat_vec_q_switch_type` (lines 671-1030)
Thread `expert_low, expert_high` through. Default to `(0, 0)` meaning "no split".

#### C. `ggml_cuda_mul_mat_vec_q` (~line 1032)
Add optional `expert_low, expert_high` params (default 0). The split orchestrator calls this with non-zero values per device.

### 2. `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_cuda_mul_mat_id` (line 2653)

Replace the gather block (lines 2660-2755) with branching logic:

```cpp
if (ggml_backend_buft_is_cuda_split(src0->buffer->buft)) {
    // Try fast path: per-device dispatch if eligible
    if (can_use_split_mmid_fast_path(src0, src1, ids, ctx)) {
        ggml_cuda_mul_mat_id_split_fast(ctx, src0, src1, ids, dst);
        return;
    }
    // Else: existing gather fallback (unchanged)
    ... existing gather code ...
}
```

`can_use_split_mmid_fast_path` checks (revised per codex plan-review blockers):
- `src0` is quantized (`ggml_is_quantized(src0->type)`), so dispatch lands in MMVQ.
- **`dst->ne[2] == 1`** — guarantees generic `mul_mat_vec_q` (not `_moe`), per `mmvq.cu:788` `has_ids && ncols_dst > 1` branch.
- **`dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)`** — guarantees dispatch reaches the MMVQ branch in `ggml_cuda_mul_mat_id` (line 2773).
- **`src0->ne[3] == 1`** — defer multi-sample MMID to gather fallback; not needed for V4.
- For the **root** split tensor (`root = src0` walked through `view_src`), each device's `[row_low_root, row_high_root)` is a whole-expert range: `row_low_root % ne01_root == 0 && row_high_root % ne01_root == 0`. Coordinates are in the **root's flat-row space** because `get_row_split` is called on the root and `data_device[id]` points to that root's slab.
- All devices participate (no empty slabs after rounding).

When ANY check fails, fall through to the existing gather path (preserved unchanged).

`ggml_cuda_mul_mat_id_split_fast`:
1. Allocate `dst_main` (zero on ctx.device, full shape).
2. For each device id:
   - `ggml_cuda_set_device(id)`.
   - Allocate per-device temp = `ggml_nbytes(dst)` (full F32 shape) on device `id`, zero it.
   - Allocate per-device ids translation buffer if needed (initially: use the original ids; the kernel does the range check; no separate translated ids needed — the kernel subtracts `expert_low` itself).
   - Copy src1 (small) to device `id` if `id != src1's device`.
   - Build a local `src0` ggml_tensor metadata clone:
     - `data = src0_extra->data_device[id]`
     - `ne[2] = expert_high - expert_low`
     - `nb` unchanged (already in row-major per-expert layout).
   - Build a local `dst` ggml_tensor metadata clone with `data = per_device_dst_temp`.
   - Call `ggml_cuda_mul_mat_vec_q(local_ctx, &local_src0, &local_src1, ids, &local_dst, expert_low, expert_high)`.
3. Peer-copy each per-device temp back to ctx.device scratch.
4. Add scratches into `dst_main` (small F32 element-wise sum kernel).
5. `dst->data` was `dst_main` all along, so we're done.

Synchronization: events between devices to ensure src1 peer-copy completes before kernel; main device waits on all per-device kernels before aggregation.

### 3. Optional: New tiny kernel `ggml_cuda_op_sum_into` (or reuse existing add)

`dst += src` element-wise on F32. We can synthesize this with a launch of `add` or use a one-line custom kernel. Codex suggested binbcast's `ggml_cuda_op_add` — but constructing a `ggml_tensor` to call that is annoying. A tiny custom kernel `__global__ void add_into_kernel(float * dst, const float * src, size_t n) { ... }` is cleaner.

## Implementation order

1. Write the kernel-side change in `mul_mat_vec_q` (add expert_low/expert_high). Add a runtime `is_split` branch (NOT template — avoid binary bloat); the branch is single-uniform per kernel launch so warp divergence is zero.
2. Thread the params through the `switch_*` wrappers and `ggml_cuda_mul_mat_vec_q`. Default values: `expert_low=0, expert_high=0` (meaning "split off" — equivalent to original behavior).
3. Verify single-GPU regression: `test-backend-ops -o MUL_MAT_ID` should still pass with the new default-zero params.
4. Add `ggml_cuda_mul_mat_id_split_fast` orchestrator in `ggml-cuda.cu`.
5. Add eligibility check; gate it. If not eligible, fall through to existing gather.
6. **Add a verification hook (env-gated GGML_CUDA_DSV4_MMID_TRACE=1) that prints a one-line summary on each MMID call**: which path (fast or gather), shapes, device count, alignment status. This lets us prove on gpudual that V4 decode actually exercises the fast path. Remove or downgrade to debug-only before opening the PR.
7. Build clean on gpudual.
8. Validate on gpudual: `test-backend-ops -o DSV4_*` 19/19 on both GPUs, layer-split chat ≥18 t/s, row-split chat ≥19 t/s (target ≥40 t/s).
9. With the trace hook, confirm rowsplit decode calls into the FAST path (not gather) for the V4 MMID layers.

## Risk register

| Risk | Mitigation |
|------|------------|
| Adding template param doubles kernel compile time | Use runtime branch with predictable code path; codex review can decide |
| Non-aligned splits fall back to slow gather | Detected at orchestrator; user knows fast path is conditional |
| dst temp allocation cost dominates for tiny ops | Pool-allocate; reuse temps across calls within a graph |
| Out-of-range kernel still touches src0 memory for OOB reads (e.g. ids prefetch) | Codex flagged this. Range check happens BEFORE first src0 read in the modified kernel |
| Event sync misses, causing race | Mirror the event scheme in `ggml_cuda_op_mul_mat` (lines 2009-2150) |
| CUDA graphs incompatible | Existing logic already disables CUDA graphs whenever any node has a split `src0` (see ggml-cuda.cu:3471 area). Therefore the row-split decode path is already non-graph-captured; the fast path inherits this and does not regress graph behavior. Do NOT claim graph-friendliness; this gate stays as-is. |
| Per-call temp allocation overhead | Use `ggml_cuda_pool_alloc` (same pool used by gather fallback) so temps recycle within the graph; for 43 layers × ~224 KiB the pool churn is negligible. |
| HIP/MUSA compatibility | Use only `cudaMemcpyPeerAsync`, `cudaEventRecord`, `cudaStreamWaitEvent`, `cudaMemsetAsync` — all wrapped by ggml-cuda's vendor.h. No `cudaMemcpy3DPeerAsync` needed for this path (we don't strided-copy across devices). Compile under HIP must be verified before merging. |
| Bandwidth math correction (codex review) | Per layer single-token = 224 KiB / non-main peer. Per token across 43 layers = ~9.4 MiB / peer = ~18.8 MiB at 2 peers. At 32 GB/s PCIe ~0.6 ms/token — within budget. (Previous plan miswrote this as 9.4 MiB per call.) |

## Validation plan (mandatory, on gpudual)

1. `test-backend-ops -o DSV4_*` on both GPUs (0 and 1, set CUDA_VISIBLE_DEVICES): 19/19 pass.
2. Layer-split chat: `--split-mode layer -ngl 999 --flash-attn on` → `≥18 t/s` decode (regression gate).
3. Single-GPU: `CUDA_VISIBLE_DEVICES=0 -ngl 999 -cmoe -ub 128` → works, regression gate.
4. Row-split chat: `--split-mode row -ngl 999 --flash-attn on` → coherent text + decode t/s. PRIMARY METRIC.

Pass thresholds:
- ≥40 t/s: SUCCESS, open PR (target is 2× layer-split).
- 19-40 t/s: MARGINAL, open PR with caveat.
- <19 t/s: FAIL, no PR, document and stop.

## Root-coordinate / view handling

The gather fallback at lines 2674-2752 walks `src0->view_src` to find the root cuda_split tensor, then uses **root coordinates** for per-device slab access (`src0_extra` is the root's extra; `root->nb[1]` is the per-flat-row stride; `get_row_split` is called on the root). The fast path must follow the same convention:

- Walk `view_src` chain to the root.
- Compute `row_low_root, row_high_root` for each device via `get_row_split(root, tensor_split, id)`.
- Alignment check: `row_low_root % root->ne[1] == 0 && row_high_root % root->ne[1] == 0` (whole-expert in root coordinates).
- Per-device local src0 metadata: derive `ne[2] = (row_high_root - row_low_root) / root->ne[1]` (expert count), keep `ne[0], ne[1]` (cols, rows-per-expert) unchanged, set `data = src0_extra->data_device[id]`.
- The view (`src0` we were called with) is Tier-1 = full-cover contiguous (asserted in the gather path); the fast path inherits this contract via the same eligibility check (presence of Tier-1 view through `view_src`).
- If `src0->ne[3] > 1` (root or view): fall back to gather. Out of scope for this PR.

## Out of scope

- Optimizing prefill (MMQ multi-token MMID path).
- Modifying `mul_mat_vec_q_moe` for `ncols_dst > 1` (separate follow-up).
- Touching `src/models/deepseek4.cpp` (V4 graph is already correct).
- Optimizing the gather fallback (it's the safety net).
- Changing cuda_split allocation/rounding semantics.
