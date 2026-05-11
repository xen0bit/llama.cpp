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

Run: `./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_HC_SPLIT_SINKHORN 2>&1 | tail -20`
Expected: PASSES via CPU fallback. Becomes a real comparison once CUDA kernel registered.

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
    GGML_ASSERT(mixes->ne[2] == 1 && mixes->ne[3] == 1);

    const int n_hc           = ggml_get_op_params_i32(dst, 0);
    const int sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps          = ggml_get_op_params_f32(dst, 2);

    const int n_rows = (int)(mixes->ne[1] * mixes->ne[2] * mixes->ne[3]);
    const int mix_hc = (int)(mixes->ne[0]);
    const int nb01   = (int)(mixes->nb[1]);
    const int nb1    = (int)(dst->nb[1]);

    constexpr int blk = 256;
    const int threads_per_block = std::min(blk, mix_hc);  // one thread per element in the row
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

- [ ] **Step 5.1: Implement the Sinkhorn iteration loop**

Replace the TODO with a translation of Metal's kernel body. Key structure:

```cpp
extern __shared__ float shmem[];  // size = mix_hc floats

const int tid = threadIdx.x;
const int blksz = blockDim.x;

// Load mixes row into shared memory
const float * row_in = (const float *)((const char *)mixes + row * nb01);
for (int i = tid; i < mix_hc; i += blksz) {
    shmem[i] = row_in[i];
}
__syncthreads();

for (int it = 0; it < sinkhorn_iters; ++it) {
    // Row normalize: sum across the mix_hc dimension via shared-mem reduce
    float local = 0.0f;
    for (int i = tid; i < mix_hc; i += blksz) {
        local += shmem[i];
    }
    // Reduce within block
    float row_sum = blockReduceSum(local);  // helper from common.cuh or write inline
    __shared__ float s_row_sum;
    if (tid == 0) s_row_sum = row_sum + eps;
    __syncthreads();
    for (int i = tid; i < mix_hc; i += blksz) {
        shmem[i] /= s_row_sum;
    }
    __syncthreads();

    // Column normalize (per element, with normalization factor across rows)
    // NOTE: column normalization here is across the mix_hc dimension internally
    // — verify against Metal source whether this is a separate dimension or
    // an inner detail of the Sinkhorn loop.
    // If column-normalization is across rows, this requires inter-block
    // sync which isn't possible in one kernel launch — Metal may use a
    // two-pass or per-row-only normalization. Re-read Metal source carefully.
}

// Combine with scale + base, write to output
for (int i = tid; i < mix_hc; i += blksz) {
    // Translation of the combination logic from Metal lines ~2200-2245
    // (look up exact lines and translate the arithmetic verbatim)
    // dst[hc, row] = scale[hc, row] * shmem[i] + base[hc, row]
    // (or whatever the Metal kernel does)
}
```

**Critical:** Re-read `ggml-metal.metal:2076-2245` carefully to understand whether "Sinkhorn" here means doubly-stochastic across rows AND columns of the [mix_hc, n_rows] matrix, OR a per-row normalization that the name "Sinkhorn" is being slightly misused for. The latter is a single-kernel-launch operation; the former needs two-pass or atomic-based cooperative reduction across blocks.

- [ ] **Step 5.2: Add blockReduceSum helper if not present**

If `common.cuh` doesn't expose a block-wide reduction, write inline:
```cpp
__device__ __forceinline__ float blockReduceSum(float v) {
    // Standard warp shuffle + shared-mem reduce. ggml-cuda likely already
    // has this; check common.cuh first.
    __shared__ float buf[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, o);
    }
    if (lane == 0) buf[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x + 31) / 32) ? buf[lane] : 0.0f;
    if (wid == 0) {
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, o);
        }
    }
    return v;
}
```

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
./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_HC_SPLIT_SINKHORN 2>&1 | tee /tmp/v4-cuda-B-sinkhorn-test.log | tail -30
COUNT=$(grep -E "^\s+[0-9]+/[0-9]+ tests passed" /tmp/v4-cuda-B-sinkhorn-test.log | tail -1 | grep -oE "^\s+[0-9]+" | tr -d ' ')
echo "Tests passed: ${COUNT:-0}"
test "${COUNT:-0}" -ge 4 || { echo "FAIL: only ${COUNT:-0} of 4+ expected tests ran"; exit 1; }
```

Expected: tests PASS with `${COUNT:-0}` >= 4, CPU and CUDA matching within `max_nmse_err` (1e-3 per Stream A).

Common failure modes:
- Wrong index decomposition (Metal uses col-major, your CUDA might assume row-major) → re-check `nb01` vs `nb1` semantics.
- Reduction off-by-one when mix_hc is not a multiple of 32 → check the warp tail handling.
- Sinkhorn convergence diverges if eps is missing from the denominator.

- [ ] **Step 7.2: Run with compute-sanitizer**

If available: `compute-sanitizer ./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_HC_SPLIT_SINKHORN 2>&1 | tail -20`
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
