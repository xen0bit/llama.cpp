# V4-port CUDA Stream B1: dsv4_rope_tail CUDA kernel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a CUDA kernel + dispatch for `ggml_dsv4_rope_tail`. Register it in the CUDA backend so V4 inference on CUDA stops falling back to CPU for this op. Pass the `test_dsv4_rope_tail` cases that Stream A added.

**Architecture:** New `.cu`/`.cuh` pair under `ggml/src/ggml-cuda/`, kernel translated from the Metal reference, dispatch follows the `rope.cu` pattern. One new `case GGML_OP_DSV4_ROPE_TAIL:` block in `ggml-cuda.cu`. One CMakeLists entry.

**Tech Stack:** CUDA C++ (kernel), C++ (dispatch), CMake.

**Spec:** `docs/superpowers/specs/2026-05-10-v4-port-cuda-design.md`

**Branch:** `feat/v4-port-cuda-B-rope-tail` off `feat/v4-port-cuda`. **Prerequisite:** Stream A must be merged into `feat/v4-port-cuda` first.

**Reference sources (read these before writing code):**
- Metal kernel: `ggml/src/ggml-metal/ggml-metal.metal:4906-4997` (`kernel_dsv4_rope_tail_f32`)
- Metal dispatch: `ggml/src/ggml-metal/ggml-metal-ops.cpp:1596-1673` (`ggml_metal_op_dsv4_rope_tail`)
- CPU reference: `ggml/src/ggml-cpu/ops.cpp:5961+`
- ggml-cuda convention: `ggml/src/ggml-cuda/rope.cu` and `rope.cuh`
- Public API: `ggml/include/ggml.h:2599-2613`

---

## Task 1: Branch + verify prerequisites

**Files:** none (git only)

- [ ] **Step 1.1: Check Stream A is merged**

Run: `git log feat/v4-port-cuda --oneline | grep -i "v4-port-cuda-A" | head -3`
Expected: at least one commit from Stream A is present. If empty, stop and wait for Stream A to merge before proceeding.

- [ ] **Step 1.2: Create branch**

```bash
cd ~/work/llama.cpp
git checkout feat/v4-port-cuda
git pull --ff-only origin feat/v4-port-cuda 2>/dev/null || true
git checkout -b feat/v4-port-cuda-B-rope-tail
```

- [ ] **Step 1.3: Verify CUDA build works**

Run: `cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89;120" 2>&1 | tail -10`
Expected: CMake configures successfully. If not, the dev environment is missing CUDA toolkit — stop and report.

Run: `cmake --build build-cuda -j --target test-backend-ops 2>&1 | tail -10`
Expected: build succeeds. (test-backend-ops binary will be at `build-cuda/bin/test-backend-ops`.)

- [ ] **Step 1.4: Confirm baseline behavior**

Run: `./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_ROPE_TAIL 2>&1 | tail -20`
Expected: test runs. Output likely PASSES because CUDA backend has no kernel → falls back to CPU → CPU vs CPU is trivially equal. The test becomes a real comparison once the CUDA kernel exists.

---

## Task 2: Read references

**Files (read-only):**
- `ggml/src/ggml-cuda/rope.cu` — full file. Note the structure: helper device functions for theta interpolation, `__global__` kernel, host-side dispatch function `ggml_cuda_op_rope`.
- `ggml/src/ggml-cuda/rope.cuh` — header pattern.
- `ggml/src/ggml-metal/ggml-metal.metal:4906-4997` — `kernel_dsv4_rope_tail_f32`. Note: this is "partial RoPE" — leaves the non-RoPE prefix unchanged, applies RoPE only to the last `n_dims` elements per row.
- `ggml/src/ggml-cpu/ops.cpp:5961+` — CPU reference for `dsv4_rope_tail`. Use this to validate your CUDA kernel matches scalar semantics.

- [ ] **Step 2.1: Identify the kernel's core math**

Make a mental note of (or write down):
- Where the RoPE rotation applies (only to indices in `[ne0 - n_dims, ne0)`)
- The frequency computation: `theta = pos * freq_base^(-2*i/n_dims) * freq_scale`
- Yarn extrapolation (when `ext_factor != 0`): blend between linear and NTK-aware scaling using `beta_fast` / `beta_slow` ramp
- Per-pair rotation: `(x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)`
- Inverse mode: same math but with `theta = -theta`

This is the same math as `ggml_rope_ext`, restricted to the tail.

---

## Task 3: Create the .cuh header

**Files:**
- Create: `ggml/src/ggml-cuda/dsv4-rope-tail.cuh`

- [ ] **Step 3.1: Write the header**

```cpp
#pragma once

// V4 partial-RoPE: applies RoPE rotation to the last n_dims elements of each
// row, leaving the non-RoPE prefix unchanged. Matches the math of
// ggml_rope_ext restricted to the tail (indices ne0 - n_dims .. ne0 - 1).
//
// Reference Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:4906-4997
// CPU reference:          ggml/src/ggml-cpu/ops.cpp:5961
// Public API:             ggml/include/ggml.h:2599 (ggml_dsv4_rope_tail)
//
// The dispatch function extracts op_params (n_dims, mode, n_ctx_orig, RoPE
// floats, inverse-flag) from the destination tensor and launches the
// __global__ kernel. The kernel's parameter list matches the Metal kargs
// struct in ggml-metal-ops.cpp:1596+ verbatim (function-arg ABI per
// ggml-cuda convention; no on-device kargs struct).

#include "common.cuh"

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

- [ ] **Step 3.2: Verify header parses**

Run: `cd ~/work/llama.cpp && clang++ -fsyntax-only -I ggml/src/ggml-cuda -I ggml/include -I ggml/src ggml/src/ggml-cuda/dsv4-rope-tail.cuh 2>&1 | head -5` (best-effort; if clang isn't installed, skip — the real check is the CUDA compile.)

- [ ] **Step 3.3: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-rope-tail.cuh
git commit -m "v4-port-cuda-B-rope-tail: header for dsv4_rope_tail dispatch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Create the .cu source — kernel skeleton

**Files:**
- Create: `ggml/src/ggml-cuda/dsv4-rope-tail.cu`

- [ ] **Step 4.1: Write the skeleton with dispatch + empty kernel**

```cpp
#include "dsv4-rope-tail.cuh"
#include "ggml.h"           // for ggml_rope_yarn_corr_dims
#include <algorithm>        // std::min / std::max in dispatch

// YaRN helper: matches the existing CUDA rope_yarn pattern at
// ggml/src/ggml-cuda/rope.cu:22-41. Reuses rope_corr_dims (computed
// host-side via ggml_rope_yarn_corr_dims) and rope_yarn_ramp.
// Output: cos_theta/sin_theta both scaled by mscale (attn_factor adjusted).
//
// Translates the same math as ggml-metal.metal:4906-4997's call to
// rope_yarn(theta/freq_factor, freq_scale, corr_dims, rel_i0, ext_factor,
// attn_factor, &cos_theta, &sin_theta).
//
// Implementation note: rope.cu's rope_yarn is already a template<bool forward>
// device function. Stream B1 may either (a) #include the relevant helpers
// from rope.cuh (if exposed) and reuse them, OR (b) duplicate the small
// rope_yarn_ramp + rope_yarn body verbatim into dsv4-rope-tail.cu. Option (b)
// is preferred to avoid coupling Stream B1 to changes in rope.cuh; the
// duplication is ~15 lines and keeps the kernel self-contained.

struct rope_corr_dims {
    float v[2];
};

static __device__ __forceinline__ float dsv4_rope_yarn_ramp(
        const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// forward=true: standard rotation; forward=false: inverse (sin flipped).
template<bool forward>
static __device__ __forceinline__ void dsv4_rope_yarn(
        const float theta_extrap, const float freq_scale,
        const rope_corr_dims corr_dims, const int i0,
        const float ext_factor, float mscale,
        float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = dsv4_rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta = -sin_theta;
    }
}

// Main kernel: launch shape mirrors Metal — grid = (ne01, ne02, ne03),
// block.x walks the ne00 dim. This matches ggml-metal.metal:4906-4997's
// dispatch pattern (tgpig = (i1, i2, i3); tid loops over ne00).
//
// Template params:
//   is_neox    — NEOX layout (rotate (j0, j0+n_half)) vs NORMAL adjacent-pair.
//   forward    — false flips sin sign (inverse RoPE).
//   has_ff     — whether freq_factors src2 is present.
static __global__ void dsv4_rope_tail_f32_kernel(
        const float * __restrict__ src0,
        const int   * __restrict__ pos,
        const float * __restrict__ freq_factors,
        float       * __restrict__ dst,
        const int    ne00,
        const int    nb00, const int nb01, const int nb02, const int nb03,
        const int    nb0,  const int nb1,  const int nb2,  const int nb3,
        const int    n_dims,
        const float  freq_base, const float freq_scale,
        const float  ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const bool   is_neox, const bool inverse) {
    // See Step 5.2 for the body. Indexing pattern follows the Metal
    // reference at ggml-metal.metal:4906-4997.
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * pos  = dst->src[1];
    const ggml_tensor * ff   = dst->src[2];  // optional; may be NULL

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(pos->type  == GGML_TYPE_I32);

    // op_params layout — MUST match the Metal dispatch in
    // ggml/src/ggml-metal/ggml-metal-ops.cpp:1606-1623 verbatim:
    //   [0] = n_dims      (i32)
    //   [1] = mode        (i32)
    //   [2] = n_ctx_orig  (i32)
    //   [3] = inverse     (i32, treated as bool)
    //   [4] = freq_base   (f32)
    //   [5] = freq_scale  (f32)
    //   [6] = ext_factor  (f32)
    //   [7] = attn_factor (f32)
    //   [8] = beta_fast   (f32)
    //   [9] = beta_slow   (f32)
    const int32_t n_dims     = ggml_get_op_params_i32(dst, 0);
    const int32_t mode       = ggml_get_op_params_i32(dst, 1);
    const int32_t n_ctx_orig = ggml_get_op_params_i32(dst, 2);
    const int32_t inverse_i  = ggml_get_op_params_i32(dst, 3);
    const bool    inverse    = inverse_i != 0;

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (const int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) dst->op_params + 9, sizeof(float));

    const bool is_neox = (mode == 2);  // GGML_ROPE_TYPE_NEOX

    // Precompute corr_dims host-side (matches Metal's rope_yarn_corr_dims
    // call at ggml-metal.metal:4927). The CUDA equivalent is exposed in
    // ggml.c as ggml_rope_yarn_corr_dims.
    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // Launch shape: grid = (ne01, ne02, ne03); block.x walks ne00.
    // Mirrors the Metal launch at ggml-metal-ops.cpp:1670
    // (dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1) with
    //  nth = min(256, ne00)).
    const int ne00 = (int) src0->ne[0];
    const int ne01 = (int) src0->ne[1];
    const int ne02 = (int) src0->ne[2];
    const int ne03 = (int) src0->ne[3];

    const int nth = std::min(256, std::max(1, ne00));
    const dim3 grid(ne01, ne02, ne03);
    const dim3 block(nth, 1, 1);

    cudaStream_t stream = ctx.stream();
    dsv4_rope_tail_f32_kernel<<<grid, block, 0, stream>>>(
        (const float *) src0->data,
        (const int *)   pos->data,
        ff ? (const float *) ff->data : nullptr,
        (float *)       dst->data,
        ne00,
        (int) src0->nb[0], (int) src0->nb[1], (int) src0->nb[2], (int) src0->nb[3],
        (int) dst->nb[0],  (int) dst->nb[1],  (int) dst->nb[2],  (int) dst->nb[3],
        n_dims,
        freq_base, freq_scale, ext_factor, attn_factor,
        corr_dims,
        is_neox, inverse);

    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 4.2: Add to CMakeLists**

Edit `ggml/src/ggml-cuda/CMakeLists.txt`. Find the existing list of `.cu` source files (likely a `file(GLOB ...)` or explicit list). Add `dsv4-rope-tail.cu` to that list. If the existing list uses `GLOB`, no edit needed.

- [ ] **Step 4.3: Build, expect compile errors only from the kernel TODO**

Run: `cmake --build build-cuda -j --target ggml-cuda 2>&1 | tail -20`
Expected: compiles successfully (the kernel body is empty but the signature compiles). If errors about `ggml_backend_cuda_context` or `CUDA_CHECK`, missing `#include "common.cuh"` is the likely cause.

- [ ] **Step 4.4: Commit skeleton**

```bash
git add ggml/src/ggml-cuda/dsv4-rope-tail.cu ggml/src/ggml-cuda/CMakeLists.txt
git commit -m "v4-port-cuda-B-rope-tail: kernel skeleton + dispatch wrapper

Empty kernel body (TODO); op_params extraction and launch shape match
the Metal dispatch at ggml-metal-ops.cpp:1596-1673.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Implement the kernel body

**Files:**
- Modify: `ggml/src/ggml-cuda/dsv4-rope-tail.cu`

- [ ] **Step 5.1: Implement the YaRN helper (already declared in Step 4.1)**

The helper signature was finalized in Step 4.1 as a template
`dsv4_rope_yarn<bool forward>(theta_extrap, freq_scale, corr_dims, i0,
ext_factor, mscale, &cos_theta, &sin_theta)`. Its body (also in Step 4.1)
is a direct port of the CUDA `rope_yarn` at `ggml/src/ggml-cuda/rope.cu:22-41`
and matches Metal's call at `ggml-metal.metal:4956` / `4979` line-for-line.

No additional implementation work needed in this step — verify the
function body from Step 4.1 against `rope.cu:22-41`:
```
ramp = clamp((i0/2 - corr_dims.v[0]) / max(corr_dims.v[1] - corr_dims.v[0], 0.001), 0, 1)
ramp_mix = (1 - ramp) * ext_factor      // (note: rope_yarn_ramp returns 1 - ramp)
theta_interp = freq_scale * theta_extrap
theta_blend = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix   (when ext_factor != 0)
mscale *= 1 + 0.1 * log(1/freq_scale)                                    (when ext_factor != 0)
cos_theta = cos(theta_blend) * mscale; sin_theta = sin(theta_blend) * mscale
if (!forward) sin_theta = -sin_theta
```

- [ ] **Step 5.2: Implement the main kernel body**

Translate `kernel_dsv4_rope_tail_f32` from `ggml-metal.metal:4906-4997`
line-for-line. Two branches: `is_neox` and the default NORMAL.

```cpp
const int i1 = blockIdx.x;
const int i2 = blockIdx.y;
const int i3 = blockIdx.z;
const int tid = threadIdx.x;
const int ntg = blockDim.x;

const int n_nope = ne00 - n_dims;
if (n_nope < 0) return;

const float theta_base_pos = (float) pos[i2];
const float inv_ndims = -1.0f / (float) n_dims;

const char * src_base = (const char *) src0 + i3*nb03 + i2*nb02 + i1*nb01;
char       * dst_base = (char *)       dst  + i3*nb3  + i2*nb2  + i1*nb1;

for (int i0 = tid; i0 < ne00; i0 += ntg) {
    // Pass-through prefix.
    if (i0 < n_nope) {
        *((float *)(dst_base + i0*nb0)) = *((const float *)(src_base + i0*nb00));
        continue;
    }

    const int r = i0 - n_nope;

    if (is_neox) {
        const int n_half = n_dims / 2;
        if (r >= n_half) continue;

        const int ic = r;
        const int rel_i0 = 2 * ic;
        const float theta = theta_base_pos * powf(freq_base, inv_ndims * (float) rel_i0);
        const float freq_factor = freq_factors ? freq_factors[ic] : 1.0f;

        float cos_theta, sin_theta;
        // Use the forward template; inverse is applied below as a sign flip
        // on sin_theta (matches Metal's "if (args.inverse) sin_theta = -sin_theta").
        dsv4_rope_yarn<true>(theta / freq_factor, freq_scale, corr_dims,
                             rel_i0, ext_factor, attn_factor,
                             cos_theta, sin_theta);
        if (inverse) sin_theta = -sin_theta;

        const int j0 = n_nope + ic;
        const int j1 = n_nope + ic + n_half;
        const float x0 = *((const float *)(src_base + j0*nb00));
        const float x1 = *((const float *)(src_base + j1*nb00));
        *((float *)(dst_base + j0*nb0)) = x0*cos_theta - x1*sin_theta;
        *((float *)(dst_base + j1*nb0)) = x0*sin_theta + x1*cos_theta;
    } else {
        if ((r & 1) != 0) continue;

        const int ic = r / 2;
        const float theta = theta_base_pos * powf(freq_base, inv_ndims * (float) r);
        const float freq_factor = freq_factors ? freq_factors[ic] : 1.0f;

        float cos_theta, sin_theta;
        dsv4_rope_yarn<true>(theta / freq_factor, freq_scale, corr_dims,
                             r, ext_factor, attn_factor,
                             cos_theta, sin_theta);
        if (inverse) sin_theta = -sin_theta;

        const int j0 = n_nope + r;
        const int j1 = j0 + 1;
        const float x0 = *((const float *)(src_base + j0*nb00));
        const float x1 = *((const float *)(src_base + j1*nb00));
        *((float *)(dst_base + j0*nb0)) = x0*cos_theta - x1*sin_theta;
        *((float *)(dst_base + j1*nb0)) = x0*sin_theta + x1*cos_theta;
    }
}
```

Cross-check against the Metal kernel at `ggml-metal.metal:4906-4997`:
the index variables (`i1, i2, i3, tid, ntg, n_nope, r, ic, rel_i0, j0, j1`)
and pointer math (`src_base + i0*nb00`, `dst_base + i0*nb0`) are the
direct translation. The only difference is CUDA's `__global__` + `blockIdx`
in place of Metal's `kernel void` + `tgpig` — same semantics.

- [ ] **Step 5.3: Build CUDA**

Run: `cmake --build build-cuda -j --target test-backend-ops 2>&1 | tail -10`
Expected: builds without errors.

- [ ] **Step 5.4: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-rope-tail.cu
git commit -m "v4-port-cuda-B-rope-tail: kernel body — RoPE rotation on tail

Per-thread element decompose; non-RoPE prefix passes through; RoPE tail
rotates pairs (x0, x1) via sincosf(theta). Yarn extrapolation and
freq_factors handling match the Metal reference at
ggml-metal.metal:4906-4997 line-for-line.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Register in the CUDA dispatcher

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`

- [ ] **Step 6.1: Add the include**

Near the top of `ggml-cuda.cu`, where other op headers are included (search for `#include "rope.cuh"`), add:
```cpp
#include "dsv4-rope-tail.cuh"
```

- [ ] **Step 6.2: Add the registry case in `ggml_cuda_compute_forward`**

Locate the main op-dispatch switch (around line 2866 — search for `case GGML_OP_ROPE:` in `ggml_cuda_compute_forward`). Insert:

```cpp
        case GGML_OP_DSV4_ROPE_TAIL:
            ggml_cuda_op_dsv4_rope_tail(ctx, dst);
            break;
```

Place it adjacent to the existing `GGML_OP_ROPE` case (alphabetical-by-DSV4 ordering can be deferred to later streams).

- [ ] **Step 6.3: Add to `ggml_backend_cuda_device_supports_op`**

The supports-op switch lives in `ggml_backend_cuda_device_supports_op` at `ggml/src/ggml-cuda/ggml-cuda.cu:4842`. Locate the `GGML_OP_ROPE` / `GGML_OP_ROPE_BACK` case (around line 5162-5165) that returns a contiguity predicate. Insert immediately after that case:

```cpp
        case GGML_OP_DSV4_ROPE_TAIL: {
            // Kernel supports mode == 0 (NORMAL) and mode == 2 (NEOX). Other
            // modes are not exercised by V4 and would silently produce wrong
            // output; refuse them so the framework falls back to CPU.
            const int32_t mode = ggml_get_op_params_i32(op, 1);
            if (mode != 0 && mode != 2) return false;
            // Same contiguity requirement as GGML_OP_ROPE.
            return op->src[0]->type == GGML_TYPE_F32
                && op->src[0]->nb[0] == ggml_type_size(op->src[0]->type)
                && ggml_is_contiguous_2(op->src[0]);
        }
```

Do NOT add `GGML_OP_DSV4_ROPE_TAIL` to the `get_op_batch_size` switch at line 5234 — V4's row layout for this op matches the default `ggml_nrows(op)` semantics, so omitting it is correct.

- [ ] **Step 6.4: Build full CUDA**

Run: `cmake --build build-cuda -j 2>&1 | tail -10`
Expected: full CUDA library + test-backend-ops both build successfully.

- [ ] **Step 6.5: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "v4-port-cuda-B-rope-tail: register dsv4_rope_tail in CUDA dispatcher

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Validate against test-backend-ops

**Files:** none (test execution)

- [ ] **Step 7.1: Run the dsv4_rope_tail test on CUDA with count assertion**

The harness reports success on `0/0` (`tests/test-backend-ops.cpp:9310-9326`) so `SKIPPED`/`NOT_SUPPORTED` would silently pass otherwise. Pin the expected count: Stream A registered **5 dsv4_rope_tail cases** (4 from the inverse×ff loop + 1 NEOX edge). The harness runs each case per enabled backend.

```bash
./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_ROPE_TAIL 2>&1 | tee /tmp/v4-cuda-B-rope-tail-test.log | tail -30
COUNT=$(grep -E "^\s+[0-9]+/[0-9]+ tests passed" /tmp/v4-cuda-B-rope-tail-test.log | tail -1 | grep -oE "^\s+[0-9]+" | tr -d ' ')
echo "Tests passed: ${COUNT:-0}"
test "${COUNT:-0}" -ge 5 || { echo "FAIL: only ${COUNT:-0} of 5+ expected tests ran (SKIPPED-counts-as-pass would mask this)"; exit 1; }
```

Expected: tests PASS on both CPU and CUDA backends with `${COUNT:-0}` >= 5. CUDA results must match CPU within `max_nmse_err` (1e-4 per Stream A).

If FAIL with large numerical error (e.g., `nmse = 0.5`):
- Common bug: index decomposition wrong → re-check `i0 / ne0 / ...` math against Metal source.
- Common bug: theta sign flip on inverse mode.
- Common bug: freq_factors index off-by-one (Metal uses `i_pair` not `2*i_pair` or vice versa).

If FAIL with kernel launch error:
- Add `CUDA_CHECK(cudaDeviceSynchronize())` after launch in the dispatch.
- Check for `out-of-bounds` via `cuda-memcheck` if available.

- [ ] **Step 7.2: Run with compute-sanitizer if available**

Run: `compute-sanitizer ./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_ROPE_TAIL 2>&1 | tail -50` (if `compute-sanitizer` is in PATH; otherwise skip)
Expected: no out-of-bounds reads, no uninitialized-memory accesses.

- [ ] **Step 7.3: Run V4 gate-loader to confirm full architecture still loads**

```bash
V4_GGUF=~/models/DeepSeek-V4-Flash-GGUF/IQ1_S-XL/DeepSeek-V4-Flash-IQ1_S-XL-00001-of-00002.gguf \
LLAMA_BIN=build-cuda \
./tests/v4-port/gate-loader.sh 2>&1 | tail -20
```
Expected: gate-loader PASSES. (If model isn't on disk, skip this step — it's the final-validation session's job. The test-backend-ops case is the per-stream gate.)

- [ ] **Step 7.4: Commit any debug fixes**

If Step 7.1 required kernel fixes, the fix commits should already exist from iterating. Otherwise, no commit needed.

---

## Task 8: Push + prepare for merge

**Files:** none

- [ ] **Step 8.1: Final review of the diff**

Run: `git log --oneline feat/v4-port-cuda..HEAD`
Expected: 4-5 commits, all on `feat/v4-port-cuda-B-rope-tail`.

Run: `git diff --stat feat/v4-port-cuda..HEAD`
Expected diff:
- 1 new file: `ggml/src/ggml-cuda/dsv4-rope-tail.cuh`
- 1 new file: `ggml/src/ggml-cuda/dsv4-rope-tail.cu`
- modified: `ggml/src/ggml-cuda/ggml-cuda.cu` (registration)
- modified: `ggml/src/ggml-cuda/CMakeLists.txt` (if not GLOB)
- No other files modified.

If anything outside these is touched, fix it (likely accidental from debugging).

- [ ] **Step 8.2: Push branch (internal-merge workflow)**

```bash
git push -u mine feat/v4-port-cuda-B-rope-tail
```

This pushes to remote `mine` (the personal fork at `github.com/cchuter/llama.cpp`). The upstream `origin` (`ggml-org/llama.cpp`) is read-only at this stage — per `AGENTS.md:88-92`, automated pushes to upstream are prohibited; the final upstream PR is a separate, human-driven step after all 5 streams merge.

- [ ] **Step 8.3: Note for parent merger**

This branch is ready to fast-forward into `feat/v4-port-cuda`. The orchestrator's testing-phase gate runs `gate-loader.sh` after the merge to confirm the build still produces a V4-capable binary.

---

## Definition of done (Stream B1)

- `ggml/src/ggml-cuda/dsv4-rope-tail.cu` and `.cuh` exist; kernel implements partial-RoPE matching the Metal reference.
- Op registered via `case GGML_OP_DSV4_ROPE_TAIL` in `ggml-cuda.cu`.
- CMakeLists includes the new `.cu` (or GLOB covers it).
- `./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_ROPE_TAIL` PASSES with CUDA backend numerically matching CPU within `max_nmse_err`.
- No files outside `ggml/src/ggml-cuda/` touched.
- Branch `feat/v4-port-cuda-B-rope-tail` pushed.

## Out of scope (Stream B1)

- Performance optimization (TMA, shared-memory tiling). Functional correctness only.
- Multi-stream / async overlap. Single-stream dispatch matches the rest of ggml-cuda.
- FP16 / BF16 kernel variants. F32 only — the V4 graph emits F32 for this op.
- The full architecture validation on real models — that's the final on-hardware session, after all 5 streams merge.
