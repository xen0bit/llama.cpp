# V4-port CUDA Stream B2: dsv4_hc_split_sinkhorn CUDA kernel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a CUDA kernel + dispatch for `ggml_dsv4_hc_split_sinkhorn`. Pass the `test_dsv4_hc_split_sinkhorn` cases that Stream A added.

**Architecture:** New `.cu`/`.cuh` pair under `ggml/src/ggml-cuda/`, kernel translated from Metal. One new `case GGML_OP_DSV4_HC_SPLIT_SINKHORN:` block in `ggml-cuda.cu`. The kernel performs iterative row/column normalization (Sinkhorn algorithm); each thread handles one row, with shared memory used for per-row reductions across iterations.

**Tech Stack:** CUDA C++ (kernel), C++ (dispatch), CMake.

**Spec:** `docs/superpowers/specs/2026-05-10-v4-port-cuda-design.md`

**Branch:** `feat/v4-port-cuda-B-sinkhorn` off `feat/v4-port-cuda`. **Prerequisite:** Stream A merged.

**Reference sources:**
- Metal kernel: `ggml/src/ggml-metal/ggml-metal.metal:2076-2245` (`kernel_dsv4_hc_split_sinkhorn`)
- Metal dispatch: `ggml/src/ggml-metal/ggml-metal-ops.cpp:1392-1438`
- CPU reference: `ggml/src/ggml-cpu/ops.cpp:10990+`
- Public API: `ggml/include/ggml.h:2563-2570`
- ggml-cuda style reference: any reduction kernel, e.g., `ggml/src/ggml-cuda/softmax.cu` (softmax also does row-wise reductions).

---

## Task 1: Branch + verify prerequisites

- [ ] **Step 1.1: Confirm Stream A merged into parent**

Run: `git log feat/v4-port-cuda --oneline | grep -i "v4-port-cuda-A" | head -3`
Expected: at least one commit. If empty, stop.

- [ ] **Step 1.2: Create branch + verify CUDA build**

```bash
cd ~/work/llama.cpp
git checkout feat/v4-port-cuda
git pull --ff-only origin feat/v4-port-cuda 2>/dev/null || true
git checkout -b feat/v4-port-cuda-B-sinkhorn
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89;120" 2>&1 | tail -5
cmake --build build-cuda -j --target test-backend-ops 2>&1 | tail -5
```
Expected: configures and builds successfully.

- [ ] **Step 1.3: Baseline test run**

Run: `./build-cuda/bin/test-backend-ops -o DSV4_HC_SPLIT_SINKHORN 2>&1 | tail -20`
Expected: PASSES via CPU fallback. Becomes a real comparison once CUDA kernel registered.

**Note on `-b` flag (plan v2 fix):** `test-backend-ops` parses `-b` via exact `strcmp` (test-backend-ops.cpp:9882) — it does NOT support comma-separated lists. The CUDA device name is `CUDA0` / `CUDA1` (per `ggml-cuda.cu:4680` + `common.cuh:1429`), not bare `CUDA`. With no `-b` flag, the harness iterates every device, auto-skipping CPU and using it internally as the reference for comparison (test-backend-ops.cpp:9615-9620). That's what we want: every CUDA device runs against the CPU reference. Stream A's `-b CPU,CUDA` shorthand was incorrect — it would silently skip everything and the COUNT assertion would catch it as 0/0.

---

## Task 2: Read references

- `ggml/src/ggml-metal/ggml-metal.metal:2076-2245` — `kernel_dsv4_hc_split_sinkhorn`. The kernel does:
  1. Split mixes [mix_hc, n_rows] into three sub-ranges (pre / post / comb), where the split is by op_params or implicit by index ranges.
  2. For each row, iteratively normalize: row sums = 1, col sums = 1, repeat `sinkhorn_iters` times (typically 4).
  3. Combine the normalized region with `scale` and `base` tensors.
- `ggml/src/ggml-cpu/ops.cpp:10990+` — CPU reference. Note: the CPU version is monolithic per row; CUDA can parallelize across rows.
- `ggml/src/ggml-cuda/softmax.cu` — reference for row-wise reductions using shared memory.

- [ ] **Step 2.1: Note the launch shape**

From Metal dispatch (`ggml-metal-ops.cpp:1425-1426`):
```cpp
const int nth = std::min<int64_t>(256, std::max<int64_t>(1, args.n_rows));
const int n_tg = (args.n_rows + nth - 1) / nth;
```
That's 256 threads per block, n_rows split across blocks. CUDA equivalent:
- One block per chunk of rows; one thread per row inside the block.
- Or, one block per row, threads cooperating on the row's reductions (better for large mix_hc).

Choose **one block per row** if `mix_hc > 32`; otherwise the per-row work fits in a single warp.

---

## Task 3: Create the .cuh header

**Files:** Create: `ggml/src/ggml-cuda/dsv4-hc-split-sinkhorn.cuh`

- [ ] **Step 3.1: Write the header**

```cpp
#pragma once

// V4 hyperconnection splitter with Sinkhorn normalization.
// Splits [mix_hc, n_rows] mixes into pre/post/comb regions, applies
// Sinkhorn doubly-stochastic normalization (alternating row + column
// normalization for `sinkhorn_iters` iterations), then combines with
// scale and base inputs to produce the final hyperconnection weights.
//
// Reference Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2076-2245
// CPU reference:          ggml/src/ggml-cpu/ops.cpp:10990+
// Public API:             ggml/include/ggml.h:2563 (ggml_dsv4_hc_split_sinkhorn)
//
// The kernel uses one CUDA block per row of the output. Threads within a
// block cooperate on the per-row reductions (sum across columns) via
// shared memory. Each iteration of Sinkhorn requires two reductions
// (row-normalize, then col-normalize). For typical mix_hc values
// (<256), one warp's reduce intrinsics suffice; larger mix_hc uses a
// shared-memory tree reduce.

#include "common.cuh"

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

- [ ] **Step 3.2: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-hc-split-sinkhorn.cuh
git commit -m "v4-port-cuda-B-sinkhorn: header for dsv4_hc_split_sinkhorn dispatch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Create the .cu source — kernel skeleton

**Files:** Create: `ggml/src/ggml-cuda/dsv4-hc-split-sinkhorn.cu`

- [ ] **Step 4.1: Write the skeleton with dispatch + empty kernel**

```cpp
#include "dsv4-hc-split-sinkhorn.cuh"

// Per-row Sinkhorn kernel. One block per row; threads cooperate across
// the row's mix_hc dimension via shared memory.
static __global__ void dsv4_hc_split_sinkhorn_f32(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float       * __restrict__ dst,
        const int n_hc,
        const int sinkhorn_iters,
        const int n_rows,
        const int mix_hc,
        const int nb01,            // input row stride (bytes)
        const int nb1,             // output row stride (bytes)
        const float eps) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    // TODO: implement Sinkhorn iterations. Translation of
    // ggml-metal.metal:2076-2245 lines that correspond to the algorithm body.
    //
    // High-level pseudocode (verify against Metal source before relying on this):
    //   load mixes row into shared mem (mix_hc floats)
    //   for it in 0..sinkhorn_iters:
    //     row_sum = sum(shared[:])
    //     shared /= (row_sum + eps)         // row-normalize
    //     col_sum = sum across threads      // col-normalize (cooperative)
    //     shared /= (col_sum + eps)
    //   combine with scale[hc, row] and base[hc, row] per Metal lines XXX-YYY
    //   write to dst[..., row]
    (void) mixes; (void) scale; (void) base; (void) dst;
    (void) n_hc; (void) sinkhorn_iters; (void) mix_hc;
    (void) nb01; (void) nb1; (void) eps;
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(mixes->nb[0] == sizeof(float));
    GGML_ASSERT(scale->nb[0] == sizeof(float));
    GGML_ASSERT(base->nb[0]  == sizeof(float));
    GGML_ASSERT(dst->nb[0]   == sizeof(float));

    const int n_hc           = ggml_get_op_params_i32(dst, 0);
    const int sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps          = ggml_get_op_params_f32(dst, 2);

    GGML_ASSERT(n_hc > 0 && n_hc <= 16);
    GGML_ASSERT(sinkhorn_iters > 0);

    const int n_rows = (int)ggml_nrows(mixes);
    const int mix_hc = (int)(mixes->ne[0]);
    const int nb01   = (int)(mixes->nb[1]);
    const int nb1    = (int)(dst->nb[1]);

    GGML_ASSERT(mix_hc == (2 + n_hc) * n_hc);
    GGML_ASSERT((int)ggml_nrows(dst) == n_rows);

    // Block size MUST be a multiple of WARP_SIZE (32) and at least 32.
    // ggml-cuda's block_reduce (common.cuh:599) asserts `block_size % WARP_SIZE == 0`
    // and the warp_reduce_sum shuffle uses the full 0xffffffff mask, which is UB
    // when inactive lanes participate. Stream A's cases have mix_hc in {24, 80};
    // both must round up: 24 -> 32, 80 -> 96. Kernel guards loop bounds with
    // `i < mix_hc` so the extra threads do no useful work but stay active for
    // the reductions.
    constexpr int WARP_SIZE_PLAN = 32;
    constexpr int MAX_BLOCK = 256;
    const int rounded = ((mix_hc + WARP_SIZE_PLAN - 1) / WARP_SIZE_PLAN) * WARP_SIZE_PLAN;
    const int threads_per_block = std::min(MAX_BLOCK, std::max(WARP_SIZE_PLAN, rounded));
    const dim3 grid(n_rows);
    const dim3 block(threads_per_block);
    const size_t shared = mix_hc * sizeof(float);

    cudaStream_t stream = ctx.stream();
    dsv4_hc_split_sinkhorn_f32<<<grid, block, shared, stream>>>(
        (const float *) mixes->data,
        (const float *) scale->data,
        (const float *) base->data,
        (float *)       dst->data,
        n_hc, sinkhorn_iters, n_rows, mix_hc,
        nb01, nb1, eps);
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 4.2: Add to CMakeLists (if not GLOB)**

If `ggml/src/ggml-cuda/CMakeLists.txt` uses an explicit source list, add `dsv4-hc-split-sinkhorn.cu`. If it uses `GLOB`, skip.

- [ ] **Step 4.3: Build skeleton, expect compile success**

Run: `cmake --build build-cuda -j --target ggml-cuda 2>&1 | tail -10`
Expected: builds (kernel body is empty but compiles).

- [ ] **Step 4.4: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-hc-split-sinkhorn.cu ggml/src/ggml-cuda/CMakeLists.txt
git commit -m "v4-port-cuda-B-sinkhorn: kernel skeleton + dispatch wrapper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Implement the kernel body

**Files:** Modify: `ggml/src/ggml-cuda/dsv4-hc-split-sinkhorn.cu`

- [ ] **Step 5.1: Implement the kernel body (faithful translation of CPU reference)**

**CRITICAL ALGORITHM NOTE:** The CPU reference (`ggml/src/ggml-cpu/ops.cpp:11037-11117`) and Metal kernel split the output into THREE distinct sections — only the third is Sinkhorn-normalized. Do NOT normalize the entire mix_hc vector.

Layout per row (mix_hc = (2 + n_hc) * n_hc; scale is [3] = pre/post/comb; base is [mix_hc]):

1. **pre slice** out[0..n_hc): `out[i] = sigmoid(mix[i] * pre_scale + base[i]) + eps`
2. **post slice** out[n_hc..2*n_hc): `out[n_hc+i] = 2 * sigmoid(mix[n_hc+i] * post_scale + base[n_hc+i])`
3. **comb matrix** out[2*n_hc..mix_hc): n_hc × n_hc matrix indexed `c[src_hc + dst_hc*n_hc]`:
   a. Apply scale+base to get logits: `c[idx] = mix[2*n_hc+idx] * comb_scale + base[2*n_hc+idx]`.
   b. First-iter softmax over src_hc per dst_hc row (max-subtract + exp + normalize), then add eps.
   c. Column-normalize (sum over dst_hc per src_hc, divide by sum+eps).
   d. For iter in [1, sinkhorn_iters): row-normalize (sum over src_hc, divide by sum+eps) then column-normalize.
4. Copy c[] to out[2*n_hc..].

Pseudocode (faithful to CPU lines 11037-11117; n_hc <= 16 per assert at line 11014):

```cpp
extern __shared__ float shmem[];  // sized to hold the n_hc*n_hc comb matrix; mix_hc floats suffices since mix_hc >= n_hc*n_hc + 2*n_hc.

const int tid    = threadIdx.x;
const int blksz  = blockDim.x;  // warp multiple, possibly > mix_hc

const float pre_scale  = scale[0];
const float post_scale = scale[1];
const float comb_scale = scale[2];

const float * row_in  = (const float *)((const char *)mixes + row * nb01);
float       * row_out = (float *)((char *)dst + row * nb1);

// Section 1 — pre slice: out[0..n_hc) = sigmoid(z) + eps
for (int i = tid; i < n_hc; i += blksz) {
    const float z = row_in[i] * pre_scale + base[i];
    row_out[i] = 1.0f / (1.0f + expf(-z)) + eps;
}

// Section 2 — post slice: out[n_hc..2*n_hc) = 2 * sigmoid(z)
for (int i = tid; i < n_hc; i += blksz) {
    const int off = n_hc + i;
    const float z = row_in[off] * post_scale + base[off];
    row_out[off] = 2.0f / (1.0f + expf(-z));
}

// Section 3 — comb matrix into shared mem (c laid out src_hc + dst_hc*n_hc)
float * c = shmem;  // n_hc*n_hc floats
for (int i = tid; i < n_hc*n_hc; i += blksz) {
    const int off = 2*n_hc + i;
    c[i] = row_in[off] * comb_scale + base[off];
}
__syncthreads();

// Iter 0: per-dst_hc softmax (max-subtract, exp, normalize), + eps
// Parallelize across dst_hc rows. n_hc is tiny (<=16), so one thread per row works.
// All threads in the warp participate in the reduce; out-of-bounds threads
// contribute identity values (-INFINITY for max, 0 for sum). Since n_hc<=16
// and we round block size to >=32, all reductions fit in a single warp_reduce
// using __shfl_xor or block_reduce on the full warp.
//
// Simpler: serialize each dst_hc row across the warp (n_hc <=16 is too small
// to benefit from parallel reduce per row). Pseudocode:
if (tid == 0) {
    for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            row_max = fmaxf(row_max, c[src_hc + dst_hc*n_hc]);
        }
        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc*n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc*n_hc;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }

    // First column-normalize (per src_hc, sum over dst_hc, divide by sum+eps).
    for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            sum += c[src_hc + dst_hc*n_hc];
        }
        const float inv_denom = 1.0f / (sum + eps);
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            c[src_hc + dst_hc*n_hc] *= inv_denom;
        }
    }

    // Remaining sinkhorn_iters - 1 iterations: row-normalize then column-normalize.
    for (int it = 1; it < sinkhorn_iters; ++it) {
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                sum += c[src_hc + dst_hc*n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                c[src_hc + dst_hc*n_hc] *= inv_denom;
            }
        }
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc*n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc*n_hc] *= inv_denom;
            }
        }
    }
}
__syncthreads();

// Copy comb matrix to output section 3.
for (int i = tid; i < n_hc*n_hc; i += blksz) {
    row_out[2*n_hc + i] = c[i];
}
```

**Performance note:** The single-thread serial inner loop is acceptable because n_hc <= 16 (asserted at CPU ops.cpp:11014); total comb-matrix work per row is O(n_hc^2 * sinkhorn_iters) = at most 16*16*N = ~1000 ops. Optimizing this further (e.g., warp-parallel reductions per dst_hc row) is out of scope per agent spec L52 ("Optimize beyond functional parity"). The performance-critical part is the n_rows-level parallelism (one block per row), which scales.

**Why this is correct:** Direct line-by-line translation of CPU reference at ops.cpp:11037-11117. The kernel is functional-parity, not optimized; correctness is the bar.

Block size is still computed as in Step 4.1 (rounded up to warp multiple from mix_hc) — Sections 1, 2, and the final copy benefit from blksz-wise parallelism even though Section 3 is serial on tid==0.

- [ ] **Step 5.2: (deleted — block reduction no longer needed)**

The v5 algorithm rewrite (Step 5.1) does Section 3's Sinkhorn iterations serially on `tid == 0` because n_hc <= 16 makes the inner work trivial. No block-wide reductions required. Sections 1, 2, and the final copy use simple `for (i = tid; i < n; i += blksz)` patterns with no inter-thread communication. The block_reduce concern from r4 is now moot for this kernel — but the warp-multiple block-size requirement from Step 4.1 is retained for correctness of any future expansion AND to keep the `__syncthreads()` calls valid (`__syncthreads` requires consistent thread participation, which warp-multiple blocks guarantee).

- [ ] **Step 5.3: Build**

Run: `cmake --build build-cuda -j --target test-backend-ops 2>&1 | tail -10`
Expected: builds.

- [ ] **Step 5.4: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-hc-split-sinkhorn.cu
git commit -m "v4-port-cuda-B-sinkhorn: kernel body — Sinkhorn iterations

Per-row Sinkhorn normalization with shared-memory cooperative reductions.
Translation of Metal kernel at ggml-metal.metal:2076-2245.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Register in CUDA dispatcher

**Files:** Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`

- [ ] **Step 6.1: Add include + case**

Add `#include "dsv4-hc-split-sinkhorn.cuh"` near the other op headers. Add to the dispatch switch:
```cpp
        case GGML_OP_DSV4_HC_SPLIT_SINKHORN:
            ggml_cuda_op_dsv4_hc_split_sinkhorn(ctx, dst);
            break;
```

- [ ] **Step 6.2: Add to supports_op switch (if present)**

Mirror however the rope path handles it (`grep -n "supports_op\|GGML_OP_ROPE" ggml/src/ggml-cuda/ggml-cuda.cu`). Likely:
```cpp
        case GGML_OP_DSV4_HC_SPLIT_SINKHORN:
            return op->src[0]->type == GGML_TYPE_F32;
```

- [ ] **Step 6.3: Build full**

Run: `cmake --build build-cuda -j 2>&1 | tail -10`
Expected: build succeeds.

- [ ] **Step 6.4: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "v4-port-cuda-B-sinkhorn: register dsv4_hc_split_sinkhorn

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Validate against test-backend-ops

- [ ] **Step 7.1: Run dsv4_hc_split_sinkhorn test on CUDA with count assertion**

The harness reports success on `0/0` so SKIPPED/NOT_SUPPORTED would silently pass. Stream A registered **4 dsv4_hc_split_sinkhorn cases**.

```bash
# Run to log; preserve exit status (don't pipe through tee|tail which masks $?).
./build-cuda/bin/test-backend-ops -o DSV4_HC_SPLIT_SINKHORN > /tmp/v4-cuda-B-sinkhorn-test.log 2>&1
RC=$?
tail -50 /tmp/v4-cuda-B-sinkhorn-test.log
test $RC -eq 0 || { echo "FAIL: test-backend-ops exited $RC"; exit 1; }

# Parse every backend summary line; require each to report >= 4 tests passed,
# AND require at least one CUDA-named backend in the log (guards against silent
# CPU-only runs if CUDA device init failed). Bash 3.2-compatible (no mapfile).
COUNTS=$(awk '/^[[:space:]]+[0-9]+\/[0-9]+ tests passed/ { split($1,a,"/"); print a[1] }' /tmp/v4-cuda-B-sinkhorn-test.log)
set -- $COUNTS
echo "Per-backend counts: ${COUNTS:-(none)}"
test "$#" -ge 1 || { echo "FAIL: no backend summaries found"; exit 1; }
grep -q "Backend.*CUDA" /tmp/v4-cuda-B-sinkhorn-test.log || { echo "FAIL: no CUDA backend ran"; exit 1; }
for c in "$@"; do
    test "$c" -ge 4 || { echo "FAIL: a backend reported only $c (need >= 4)"; exit 1; }
done
echo "PASS: all backends ran >= 4 sinkhorn cases, includes CUDA"
```

Expected: every backend reports >= 4 tests passed, the log contains at least one `Backend ... CUDA` init line, and `test-backend-ops` exited 0. CPU is auto-skipped as reference; CUDA (and any other non-CPU device) runs the 4 sinkhorn cases against the CPU reference within `max_nmse_err` (1e-3 per Stream A).

**Why no `-b` flag (plan v2 fix):** see Step 1.3 note. Omitting `-b` runs every non-CPU device against the CPU reference; this is the canonical comparison mode. `-b CPU,CUDA` is invalid syntax that silently no-ops to 0/0.

**Why preserve exit + check all summaries (plan v2 fix):** `tee | tail` masks the binary's exit status, and `tail -1` only reads the LAST backend summary — on a multi-GPU host a no-op first backend would be invisible. The gate now (a) saves exit status, (b) iterates every summary line and asserts each >= 4, and (c) requires a CUDA backend init line in the log so a degraded CPU-only run can't silently pass.

Common failure modes:
- Wrong index decomposition (Metal uses col-major, your CUDA might assume row-major) → re-check `nb01` vs `nb1` semantics.
- Reduction off-by-one when mix_hc is not a multiple of 32 → check the warp tail handling.
- Sinkhorn convergence diverges if eps is missing from the denominator.

- [ ] **Step 7.2: Run with compute-sanitizer**

If available: `compute-sanitizer ./build-cuda/bin/test-backend-ops -o DSV4_HC_SPLIT_SINKHORN 2>&1 | tail -20`
Expected: no warnings.

- [ ] **Step 7.3: Commit any debugging fixes**

If 7.1 required kernel fixes, those commits should already be in place from iterating.

---

## Task 8: Push + prepare for merge

- [ ] **Step 8.1: Diff review**

Run: `git diff --stat feat/v4-port-cuda..HEAD`
Expected:
- New: `ggml/src/ggml-cuda/dsv4-hc-split-sinkhorn.cuh`
- New: `ggml/src/ggml-cuda/dsv4-hc-split-sinkhorn.cu`
- Modified: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modified: `ggml/src/ggml-cuda/CMakeLists.txt` (if not GLOB)
- Nothing else.

- [ ] **Step 8.2: Push**

```bash
git push -u origin feat/v4-port-cuda-B-sinkhorn
```

---

## Definition of done (Stream B2)

- `.cu` + `.cuh` exist; kernel implements per-row Sinkhorn with cooperative reductions.
- Op registered in `ggml-cuda.cu` (dispatch + supports_op).
- `test-backend-ops -o DSV4_HC_SPLIT_SINKHORN` PASSES with `max_nmse_err = 1e-3`.
- No files outside `ggml/src/ggml-cuda/` modified.
- Branch pushed.

## Out of scope (Stream B2)

- Performance tuning beyond functional correctness.
- F16 / BF16 variants — F32 only.
- Multi-pass kernels for very large mix_hc (>1024) — current scope assumes mix_hc fits in shared memory.
