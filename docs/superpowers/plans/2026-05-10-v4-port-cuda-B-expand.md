# V4-port CUDA Stream B4: dsv4_hc_expand CUDA kernel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CUDA for `ggml_dsv4_hc_expand`. Pass the `test_dsv4_hc_expand` cases from Stream A.

**Architecture:** New `.cu`/`.cuh` pair. Kernel computes `out[i, hc, tok] = post[hc, tok] * block_out[i, hc, tok] + sum_{hc'} comb[hc, hc', tok] * residual[i, hc', tok]`. Per-thread one output element; each thread does an n_hc accumulation for the `comb @ residual` term.

**Tech Stack:** CUDA C++, CMake.

**Spec:** `docs/superpowers/specs/2026-05-10-v4-port-cuda-design.md`

**Branch:** `feat/v4-port-cuda-B-expand` off `feat/v4-port-cuda`. **Prerequisite:** Stream A merged.

**Reference sources:**
- Metal kernel: `ggml/src/ggml-metal/ggml-metal.metal:2247-2276`
- Metal dispatch: `ggml/src/ggml-metal/ggml-metal-ops.cpp:1488-1548`
- CPU reference: `ggml/src/ggml-cpu/ops.cpp:11200+`
- Public API: `ggml/include/ggml.h:2581-2586`

---

## Task 1: Branch + verify prerequisites

- [ ] **Step 1.1: Stream A merged check**

Run: `git log feat/v4-port-cuda --oneline | grep -i "v4-port-cuda-A" | head -3`
Expected: at least one commit. Otherwise stop.

- [ ] **Step 1.2: Create branch + build**

```bash
cd ~/work/llama.cpp
git checkout feat/v4-port-cuda
git pull --ff-only origin feat/v4-port-cuda 2>/dev/null || true
git checkout -b feat/v4-port-cuda-B-expand
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89;120" 2>&1 | tail -5
cmake --build build-cuda -j --target test-backend-ops 2>&1 | tail -5
```

- [ ] **Step 1.3: Baseline**

Run: `./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_HC_EXPAND 2>&1 | tail -10`
Expected: PASSES via CPU fallback.

---

## Task 2: Read references

- Metal kernel `ggml-metal.metal:2247-2276` — note the exact stride semantics and the 4-input fusion.
- Metal dispatch `ggml-metal-ops.cpp:1488-1548` — 16 params total; lots of strides to track.
- CPU ref `ggml-cpu/ops.cpp:11200+` — use to verify accumulation order.

The shapes (from the public API and test_dsv4_hc_expand):
- `block_out` [n_embd, n_hc, n_tokens]
- `residual`  [n_embd, n_hc, n_tokens]
- `post`      [n_hc, n_tokens]
- `comb`      [n_hc, n_hc, n_tokens]      (the second n_hc index is `hc'`)
- `out`       [n_embd, n_hc, n_tokens]

Per-element math:
```
out[i_embd, i_hc, i_tok] = post[i_hc, i_tok] * block_out[i_embd, i_hc, i_tok]
                         + sum_{hc'} comb[i_hc, hc', i_tok] * residual[i_embd, hc', i_tok]
```

---

## Task 3: Create the .cuh header

**Files:** Create: `ggml/src/ggml-cuda/dsv4-hc-expand.cuh`

- [ ] **Step 3.1: Write the header**

```cpp
#pragma once

// V4 hyperconnection expand: per-token mix of block_out and residual.
//
// out[i, hc, tok] = post[hc, tok] * block_out[i, hc, tok]
//                 + sum_{hc'} comb[hc, hc', tok] * residual[i, hc', tok]
//
// Reference Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2247-2276
// CPU reference:          ggml/src/ggml-cpu/ops.cpp:11200+
// Public API:             ggml/include/ggml.h:2581 (ggml_dsv4_hc_expand)
//
// Embarrassingly parallel: one thread per output element (i_embd, i_hc, i_tok).
// Each thread does an n_hc-wide accumulation for the comb·residual term plus
// one fused multiply-add for the post·block_out term.

#include "common.cuh"

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

- [ ] **Step 3.2: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-hc-expand.cuh
git commit -m "v4-port-cuda-B-expand: header

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Create the .cu source

**Files:** Create: `ggml/src/ggml-cuda/dsv4-hc-expand.cu`

- [ ] **Step 4.1: Write the kernel + dispatch**

```cpp
#include "dsv4-hc-expand.cuh"

static __global__ void dsv4_hc_expand_f32(
        const float * __restrict__ block_out,
        const float * __restrict__ residual,
        const float * __restrict__ post,
        const float * __restrict__ comb,
        float       * __restrict__ dst,
        const int n_embd, const int n_hc, const int n_tokens,
        // block_out strides
        const int nb_b0, const int nb_b1, const int nb_b2,
        // residual strides
        const int nb_r0, const int nb_r1, const int nb_r2,
        // post strides (2D)
        const int nb_p0, const int nb_p1,
        // comb strides (3D)
        const int nb_c0, const int nb_c1, const int nb_c2,
        // dst strides
        const int nb0,   const int nb1,   const int nb2) {
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_embd * n_hc * n_tokens;
    if (gid >= total) return;

    const int i_embd = gid % n_embd;
    const int rest   = gid / n_embd;
    const int i_hc   = rest % n_hc;
    const int i_tok  = rest / n_hc;

    // post * block_out
    const float p = *(const float *)((const char *)post + i_tok * nb_p1 + i_hc * nb_p0);
    const float b = *(const float *)((const char *)block_out
        + i_tok * nb_b2 + i_hc * nb_b1 + i_embd * nb_b0);
    float acc = p * b;

    // comb @ residual: sum over hc'
    for (int hc_p = 0; hc_p < n_hc; ++hc_p) {
        const float c = *(const float *)((const char *)comb
            + i_tok * nb_c2 + hc_p * nb_c1 + i_hc * nb_c0);
        const float r = *(const float *)((const char *)residual
            + i_tok * nb_r2 + hc_p * nb_r1 + i_embd * nb_r0);
        acc += c * r;
    }

    float * d = (float *)((char *)dst + i_tok * nb2 + i_hc * nb1 + i_embd * nb0);
    *d = acc;
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int n_embd   = (int) dst->ne[0];
    const int n_hc     = (int) dst->ne[1];
    const int n_tokens = (int) dst->ne[2];

    const int64_t total = (int64_t)n_embd * n_hc * n_tokens;
    constexpr int blk = 256;
    const dim3 grid((total + blk - 1) / blk);
    const dim3 block(blk);

    cudaStream_t stream = ctx.stream();
    dsv4_hc_expand_f32<<<grid, block, 0, stream>>>(
        (const float *) block_out->data,
        (const float *) residual->data,
        (const float *) post->data,
        (const float *) comb->data,
        (float *)       dst->data,
        n_embd, n_hc, n_tokens,
        (int) block_out->nb[0], (int) block_out->nb[1], (int) block_out->nb[2],
        (int) residual->nb[0],  (int) residual->nb[1],  (int) residual->nb[2],
        (int) post->nb[0],      (int) post->nb[1],
        (int) comb->nb[0],      (int) comb->nb[1],      (int) comb->nb[2],
        (int) dst->nb[0],       (int) dst->nb[1],       (int) dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 4.2: Add to CMakeLists** (skip if GLOB)

- [ ] **Step 4.3: Build**

Run: `cmake --build build-cuda -j --target ggml-cuda 2>&1 | tail -10`
Expected: builds.

- [ ] **Step 4.4: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-hc-expand.cu ggml/src/ggml-cuda/CMakeLists.txt
git commit -m "v4-port-cuda-B-expand: kernel + dispatch

Per-element fused multiply-add (post·block_out) plus n_hc-wide
accumulation (comb @ residual). Translation of
ggml-metal.metal:2247-2276.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Register in CUDA dispatcher

- [ ] **Step 5.1: Add include + case + supports_op**

In `ggml-cuda.cu`:
```cpp
#include "dsv4-hc-expand.cuh"
// ...
        case GGML_OP_DSV4_HC_EXPAND:
            ggml_cuda_op_dsv4_hc_expand(ctx, dst);
            break;
// and in supports_op:
        case GGML_OP_DSV4_HC_EXPAND:
            return op->src[0]->type == GGML_TYPE_F32;
```

- [ ] **Step 5.2: Build full**

Run: `cmake --build build-cuda -j 2>&1 | tail -10`

- [ ] **Step 5.3: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "v4-port-cuda-B-expand: register dsv4_hc_expand

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Validate

- [ ] **Step 6.1: Run test with count assertion**

The harness reports success on `0/0` so SKIPPED/NOT_SUPPORTED would silently pass. Stream A registered **3 dsv4_hc_expand cases**.

```bash
./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_HC_EXPAND 2>&1 | tee /tmp/v4-cuda-B-expand-test.log | tail -30
COUNT=$(grep -E "^\s+[0-9]+/[0-9]+ tests passed" /tmp/v4-cuda-B-expand-test.log | tail -1 | grep -oE "^\s+[0-9]+" | tr -d ' ')
echo "Tests passed: ${COUNT:-0}"
test "${COUNT:-0}" -ge 3 || { echo "FAIL: only ${COUNT:-0} of 3+ expected tests ran"; exit 1; }
```

Expected: tests PASS with `${COUNT:-0}` >= 3, CUDA vs CPU within `max_nmse_err = 1e-4`.

Common failures:
- Wrong index ordering for `comb`: the kernel reads `comb[i_hc, hc_p, i_tok]` not `comb[hc_p, i_hc, i_tok]`. Re-check Metal source — `comb` may be transposed relative to what the public API suggests.
- Stride bytes vs elements confusion — `nb[0]` for contiguous F32 is `4`.

- [ ] **Step 6.2: compute-sanitizer**

Run: `compute-sanitizer ./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_HC_EXPAND 2>&1 | tail -20`

---

## Task 7: Push

- [ ] **Step 7.1: Diff review**

Run: `git diff --stat feat/v4-port-cuda..HEAD`

- [ ] **Step 7.2: Push**

```bash
git push -u origin feat/v4-port-cuda-B-expand
```

---

## Definition of done (Stream B4)

- `dsv4-hc-expand.{cu,cuh}` exist.
- Registered in dispatcher + supports_op.
- `test-backend-ops -o DSV4_HC_EXPAND` PASSES.
- Branch pushed.

## Out of scope (Stream B4)

- Performance tuning. The kernel is bandwidth-bound for typical sizes.
- F16/BF16. F32 only.
- Tensorized comb·residual (matmul intrinsics) — keep the simple loop for clarity.
