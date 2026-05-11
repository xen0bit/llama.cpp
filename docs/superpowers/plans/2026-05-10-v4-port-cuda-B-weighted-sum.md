# V4-port CUDA Stream B3: dsv4_hc_weighted_sum CUDA kernel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CUDA for `ggml_dsv4_hc_weighted_sum`. Pass the `test_dsv4_hc_weighted_sum` cases from Stream A.

**Architecture:** New `.cu`/`.cuh` pair. Kernel: per-output-element, each thread accumulates `sum_hc weights[hc, token] * x[embd, hc, token]`. Trivially parallel; no reductions across blocks needed.

**Tech Stack:** CUDA C++, CMake.

**Spec:** `docs/superpowers/specs/2026-05-10-v4-port-cuda-design.md`

**Branch:** `feat/v4-port-cuda-B-weighted-sum` off `feat/v4-port-cuda`. **Prerequisite:** Stream A merged.

**Reference sources:**
- Metal kernel: `ggml/src/ggml-metal/ggml-metal.metal:2278-2327`
- Metal dispatch: `ggml/src/ggml-metal/ggml-metal-ops.cpp:1440-1486`
- CPU reference: `ggml/src/ggml-cpu/ops.cpp:11100+`
- Public API: `ggml/include/ggml.h:2574-2577`

---

## Task 1: Branch + verify prerequisites

- [ ] **Step 1.1: Check Stream A merged**

Run: `git log feat/v4-port-cuda --oneline | grep -i "v4-port-cuda-A" | head -3`
Expected: at least one commit. Otherwise stop.

- [ ] **Step 1.2: Create branch + verify CUDA build**

```bash
cd ~/work/llama.cpp
git checkout feat/v4-port-cuda
git pull --ff-only origin feat/v4-port-cuda 2>/dev/null || true
git checkout -b feat/v4-port-cuda-B-weighted-sum
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89;120" 2>&1 | tail -5
cmake --build build-cuda -j --target test-backend-ops 2>&1 | tail -5
```
Expected: builds.

- [ ] **Step 1.3: Baseline test**

Run: `./build-cuda/bin/test-backend-ops -o DSV4_HC_WEIGHTED_SUM 2>&1 | tail -10`
Expected: PASSES via CPU fallback.

---

## Task 2: Read references

- Metal kernel at `ggml-metal.metal:2278-2327` — straightforward: launch grid = (n_embd × n_tokens) elements; each thread loops over n_hc to accumulate.
- Dispatch at `ggml-metal-ops.cpp:1440-1486` — shows the strides to use. `n_elem = ne0 * ne1` where ne0=n_embd, ne1=n_tokens.
- CPU ref at `ggml-cpu/ops.cpp:11100+` — verify the accumulation order. FP32 sum so order matters slightly for last-bit accuracy.

---

## Task 3: Create the .cuh header

**Files:** Create: `ggml/src/ggml-cuda/dsv4-hc-weighted-sum.cuh`

- [ ] **Step 3.1: Write the header**

```cpp
#pragma once

// V4 hyperconnection weighted-sum: collapses the hc dimension.
// Computes: out[embd, token] = sum over hc of weights[hc, token] * x[embd, hc, token]
//
// Reference Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2278-2327
// CPU reference:          ggml/src/ggml-cpu/ops.cpp:11100+
// Public API:             ggml/include/ggml.h:2574 (ggml_dsv4_hc_weighted_sum)
//
// Embarrassingly parallel: one thread per output element, each thread
// loops over n_hc to accumulate.

#include "common.cuh"

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

- [ ] **Step 3.2: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-hc-weighted-sum.cuh
git commit -m "v4-port-cuda-B-weighted-sum: header

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Create the .cu source

**Files:** Create: `ggml/src/ggml-cuda/dsv4-hc-weighted-sum.cu`

- [ ] **Step 4.1: Write the kernel + dispatch**

```cpp
#include "dsv4-hc-weighted-sum.cuh"

static __global__ void dsv4_hc_weighted_sum_f32(
        const float * __restrict__ x,
        const float * __restrict__ weights,
        float       * __restrict__ dst,
        const int n_embd, const int n_hc, const int n_tokens,
        const int nb_x0, const int nb_x1, const int nb_x2,   // x strides (bytes)
        const int nb_w0, const int nb_w1,                    // weights strides (bytes)
        const int nb0,   const int nb1) {                    // dst strides (bytes)
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_embd * n_tokens;
    if (gid >= total) return;

    const int i_embd = gid % n_embd;
    const int i_tok  = gid / n_embd;

    float acc = 0.0f;
    for (int i_hc = 0; i_hc < n_hc; ++i_hc) {
        const float * x_ptr = (const float *)((const char *)x
            + i_tok * nb_x2 + i_hc * nb_x1 + i_embd * nb_x0);
        const float * w_ptr = (const float *)((const char *)weights
            + i_tok * nb_w1 + i_hc * nb_w0);
        acc += (*x_ptr) * (*w_ptr);
    }

    float * d_ptr = (float *)((char *)dst + i_tok * nb1 + i_embd * nb0);
    *d_ptr = acc;
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type       == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);

    const int n_embd   = (int) dst->ne[0];
    const int n_hc     = (int) x->ne[1];
    const int n_tokens = (int) dst->ne[1];

    const int nb_x0 = (int) x->nb[0];
    const int nb_x1 = (int) x->nb[1];
    const int nb_x2 = (int) x->nb[2];
    const int nb_w0 = (int) weights->nb[0];
    const int nb_w1 = (int) weights->nb[1];
    const int nb0   = (int) dst->nb[0];
    const int nb1   = (int) dst->nb[1];

    const int64_t total = (int64_t)n_embd * n_tokens;
    constexpr int blk = 256;
    const dim3 grid((total + blk - 1) / blk);
    const dim3 block(blk);

    cudaStream_t stream = ctx.stream();
    dsv4_hc_weighted_sum_f32<<<grid, block, 0, stream>>>(
        (const float *) x->data,
        (const float *) weights->data,
        (float *)       dst->data,
        n_embd, n_hc, n_tokens,
        nb_x0, nb_x1, nb_x2,
        nb_w0, nb_w1,
        nb0, nb1);
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 4.2: Add to CMakeLists** (skip if GLOB)

`ggml/src/ggml-cuda/CMakeLists.txt:104` uses `file(GLOB GGML_SOURCES_CUDA "*.cu")` WITHOUT `CONFIGURE_DEPENDS`. The glob is evaluated at cmake-configure time, so the new `.cu` will NOT be picked up by an already-configured `build-cuda/`. There is no source-list edit to make, but we MUST force a reconfigure before building.

- [ ] **Step 4.3: Reconfigure + build**

```bash
# Force CMake to re-run the GLOB so the new .cu is included in GGML_SOURCES_CUDA.
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89;120" 2>&1 | tail -5
cmake --build build-cuda -j --target ggml-cuda 2>&1 | tail -10
# Sanity-check the new translation unit actually compiled:
test -f build-cuda/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/dsv4-hc-weighted-sum.cu.o || { echo "FAIL: dsv4-hc-weighted-sum.cu.o missing"; exit 1; }
```
Expected: builds, .o exists.

- [ ] **Step 4.4: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-hc-weighted-sum.cu ggml/src/ggml-cuda/CMakeLists.txt
git commit -m "v4-port-cuda-B-weighted-sum: kernel + dispatch

Per-output-element accumulation over n_hc dimension. Direct translation
of Metal kernel at ggml-metal.metal:2278-2327.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Register in CUDA dispatcher

**Files:** Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`

- [ ] **Step 5.1: Add include + case + supports_op**

Add `#include "ggml-cuda/dsv4-hc-weighted-sum.cuh"` (match the existing include style at the top of `ggml-cuda.cu`, e.g. `#include "ggml-cuda/dsv4-hc-expand.cuh"`). Add:
```cpp
        case GGML_OP_DSV4_HC_WEIGHTED_SUM:
            ggml_cuda_op_dsv4_hc_weighted_sum(ctx, dst);
            break;
```
And in supports_op (if present):
```cpp
        case GGML_OP_DSV4_HC_WEIGHTED_SUM:
            return op->type           == GGML_TYPE_F32
                && op->src[0]->type   == GGML_TYPE_F32
                && op->src[1]->type   == GGML_TYPE_F32;
```

- [ ] **Step 5.2: Build full**

Run: `cmake --build build-cuda -j 2>&1 | tail -10`
Expected: builds.

- [ ] **Step 5.3: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "v4-port-cuda-B-weighted-sum: register dsv4_hc_weighted_sum

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Validate

- [ ] **Step 6.1: Run test with count assertion**

The harness reports success on `0/0` so SKIPPED/NOT_SUPPORTED would silently pass. Stream A registered **3 dsv4_hc_weighted_sum cases**.

```bash
./build-cuda/bin/test-backend-ops -o DSV4_HC_WEIGHTED_SUM 2>&1 | tee /tmp/v4-cuda-B-weighted-sum-test.log | tail -30
# Sum tests-passed counts across all backend summaries so a CUDA-only build
# and a CPU+CUDA build are both accepted. Stream A registers 3 cases; CUDA
# must report >= 3 on a CUDA build (CPU mirror is fine but not required).
COUNT=$(grep -E "^\s+[0-9]+/[0-9]+ tests passed" /tmp/v4-cuda-B-weighted-sum-test.log | grep -oE "^\s+[0-9]+" | tr -d ' ' | awk '{s+=$1} END {print s+0}')
echo "Tests passed (aggregate across backends): ${COUNT:-0}"
test "${COUNT:-0}" -ge 3 || { echo "FAIL: only ${COUNT:-0} of 3+ expected tests ran"; exit 1; }
```

Expected: tests PASS with `${COUNT:-0}` >= 3, CPU vs CUDA within `max_nmse_err = 1e-5` (the tolerance Stream A actually set in `test-backend-ops.cpp` for `test_dsv4_hc_weighted_sum`; matches the design spec's "1e-5 abs / 1e-4 rel" line for B3).

Common failures:
- Wrong stride decomposition (bytes vs elements) → check `nb_x0` semantics. `nb[0]` is bytes per element; for F32 row-major it's `sizeof(float) = 4`.
- Off-by-one on hc iteration.

- [ ] **Step 6.2: compute-sanitizer if available**

Run: `compute-sanitizer ./build-cuda/bin/test-backend-ops -o DSV4_HC_WEIGHTED_SUM 2>&1 | tail -20`

---

## Task 7: Push

- [ ] **Step 7.1: Diff review**

Run: `git diff --stat feat/v4-port-cuda..HEAD`
Expected: 2 new files in ggml-cuda, modifications to ggml-cuda.cu and CMakeLists.

- [ ] **Step 7.2: Push**

```bash
git push -u origin feat/v4-port-cuda-B-weighted-sum
```

---

## Definition of done (Stream B3)

- `dsv4-hc-weighted-sum.{cu,cuh}` exist.
- Registered in dispatcher + supports_op.
- `test-backend-ops -o DSV4_HC_WEIGHTED_SUM` PASSES on CUDA backend.
- Branch pushed.

## Out of scope (Stream B3)

- Performance tuning. Trivial parallelism; expect bandwidth-bound.
- F16/BF16. F32 only.
