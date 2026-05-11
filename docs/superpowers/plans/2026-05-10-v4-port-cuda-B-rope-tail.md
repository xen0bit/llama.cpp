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

Run: `./build-cuda/bin/test-backend-ops -o DSV4_ROPE_TAIL 2>&1 | tail -20`
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

// Helper: RoPE frequency computation (theta scaling, Yarn ext_factor).
// Translates the corresponding portion of ggml-metal.metal:4906-4997.
static __device__ __forceinline__ float rope_yarn_freq(
        float theta_base, float freq_scale, float ext_factor, float attn_factor,
        int i0, int n_dims) {
    // TODO: copy the yarn ramp formula from rope.cu / Metal reference verbatim.
    // Identical math; only the host-side wrapping differs.
    (void) theta_base; (void) freq_scale; (void) ext_factor;
    (void) attn_factor; (void) i0; (void) n_dims;
    return 0.0f;
}

// Main kernel: one thread per output element. Element offset and index logic
// follows the Metal kernel at ggml-metal.metal:4906-4997.
static __global__ void dsv4_rope_tail_f32(
        const float * __restrict__ src0,
        const int   * __restrict__ pos,
        const float * __restrict__ freq_factors,
        float       * __restrict__ dst,
        const int    ne0, const int ne1, const int ne2, const int ne3,
        const int    nb00, const int nb01, const int nb02, const int nb03,
        const int    nb0,  const int nb1,  const int nb2,  const int nb3,
        const int    n_dims, const int mode, const int n_ctx_orig,
        const float  freq_base, const float freq_scale,
        const float  ext_factor, const float attn_factor,
        const float  beta_fast, const float beta_slow,
        const bool   inverse) {
    // TODO: translate ggml-metal.metal:4906-4997 to CUDA semantics.
    // Indexing: gid = blockIdx.x * blockDim.x + threadIdx.x;
    // Decompose to (i0, i1, i2, i3) using ne0..ne3 strides.
    // If i0 < ne0 - n_dims: pass through (dst[...] = src0[...])
    // Else: rotate (x_pair, x_pair+1) using theta from pos[i2] and freq_factors.
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * pos  = dst->src[1];
    const ggml_tensor * ff   = dst->src[2];  // optional; may be NULL

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(pos->type  == GGML_TYPE_I32);

    const int n_dims      = ggml_get_op_params_i32(dst, 0);
    const int mode        = ggml_get_op_params_i32(dst, 1);
    const int n_ctx_orig  = ggml_get_op_params_i32(dst, 2);
    const float freq_base   = ggml_get_op_params_f32(dst, 3);
    const float freq_scale  = ggml_get_op_params_f32(dst, 4);
    const float ext_factor  = ggml_get_op_params_f32(dst, 5);
    const float attn_factor = ggml_get_op_params_f32(dst, 6);
    const float beta_fast   = ggml_get_op_params_f32(dst, 7);
    const float beta_slow   = ggml_get_op_params_f32(dst, 8);
    const int   inverse_i   = ggml_get_op_params_i32(dst, 9);
    const bool  inverse     = inverse_i != 0;

    // (Verify the op_params indices against the Metal dispatch at
    //  ggml-metal-ops.cpp:1596-1673 before relying on these.)

    const int64_t n_total = ggml_nelements(dst);
    constexpr int blk = 256;
    const dim3 grid((n_total + blk - 1) / blk);
    const dim3 block(blk);

    cudaStream_t stream = ctx.stream();
    dsv4_rope_tail_f32<<<grid, block, 0, stream>>>(
        (const float *) src0->data,
        (const int *)   pos->data,
        ff ? (const float *) ff->data : nullptr,
        (float *)       dst->data,
        (int) src0->ne[0], (int) src0->ne[1], (int) src0->ne[2], (int) src0->ne[3],
        (int) src0->nb[0], (int) src0->nb[1], (int) src0->nb[2], (int) src0->nb[3],
        (int) dst->nb[0],  (int) dst->nb[1],  (int) dst->nb[2],  (int) dst->nb[3],
        n_dims, mode, n_ctx_orig,
        freq_base, freq_scale, ext_factor, attn_factor,
        beta_fast, beta_slow, inverse);

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

- [ ] **Step 5.1: Implement rope_yarn_freq**

Open `ggml/src/ggml-cuda/rope.cu` and locate the existing `rope_yarn` device function. Copy the Yarn-ramp computation verbatim (ramp from `beta_fast` to `beta_slow`, interpolate between linear and NTK-aware scaling). Adapt the function signature to match the helper declared in Step 4.1.

Reference math (also in Metal kernel lines 4906-4997):
```
ramp = clamp((|i0/2 - beta_low| / max(beta_high - beta_low, 0.001)), 0.0, 1.0)
theta_interp = theta_base * freq_scale
theta_extrap = theta_base
theta_blend = mix(theta_interp, theta_extrap, ramp) * attn_factor (when ext_factor != 0)
```

- [ ] **Step 5.2: Implement the main kernel body**

Within `dsv4_rope_tail_f32`, replace the TODO with the indexing + rotation logic. Direct translation of `kernel_dsv4_rope_tail_f32` in Metal source:

```cpp
const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
if (gid >= (int64_t)ne0 * ne1 * ne2 * ne3) return;

// Decompose linear index to (i0, i1, i2, i3)
const int i0 = gid % ne0;
const int rest1 = gid / ne0;
const int i1 = rest1 % ne1;
const int rest2 = rest1 / ne1;
const int i2 = rest2 % ne2;
const int i3 = rest2 / ne2;

const int prefix = ne0 - n_dims;

const float * src_ptr = (const float *)((const char *)src0
    + i3*nb03 + i2*nb02 + i1*nb01);
float * dst_ptr = (float *)((char *)dst
    + i3*nb3 + i2*nb2 + i1*nb1);

if (i0 < prefix) {
    // Non-RoPE prefix: pass through.
    dst_ptr[i0] = src_ptr[i0];
    return;
}

// RoPE tail: rotate pairs (x0, x1) at indices (prefix + 2k, prefix + 2k + 1).
const int j = i0 - prefix;       // index within the rotated tail
if (j & 1) return;               // each thread handles a pair; odd indices skipped

const int i_pair = j / 2;
const int pos_i  = pos[i2];

// Compute theta using Yarn helper (or vanilla NTK if ext_factor == 0).
float theta_base = (float)pos_i * powf(freq_base, -((float)(2*i_pair)) / (float)n_dims);
if (ff) {
    theta_base = theta_base / ff[i_pair];
}
const float theta = rope_yarn_freq(theta_base, freq_scale, ext_factor,
                                   attn_factor, 2*i_pair, n_dims);

float cos_th, sin_th;
sincosf(inverse ? -theta : theta, &sin_th, &cos_th);

const float x0 = src_ptr[i0];
const float x1 = src_ptr[i0 + 1];
dst_ptr[i0]     = x0 * cos_th - x1 * sin_th;
dst_ptr[i0 + 1] = x0 * sin_th + x1 * cos_th;
```

(Cross-check the index decomposition and indices against the Metal kernel before relying on this. The Metal kernel may use a slightly different launch shape — adapt the thread-to-element mapping to match.)

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

- [ ] **Step 6.2: Add the registry case**

In the main `switch` inside `ggml_cuda_compute_forward` (or whichever function handles op-to-kernel dispatch — find via `grep -n "case GGML_OP_ROPE:" ggml/src/ggml-cuda/ggml-cuda.cu`), insert:

```cpp
        case GGML_OP_DSV4_ROPE_TAIL:
            ggml_cuda_op_dsv4_rope_tail(ctx, dst);
            break;
```

Insert it alphabetically-adjacent to other DSV4 ops if any exist already; otherwise place it near the existing ROPE case.

- [ ] **Step 6.3: Also add to ggml_backend_cuda_supports_op (if present)**

Run: `grep -n "GGML_OP_ROPE" ggml/src/ggml-cuda/ggml-cuda.cu | head -10`
For each location the GGML_OP_ROPE case appears in a "supports_op"-style switch (where backends advertise which ops they support), add a corresponding `case GGML_OP_DSV4_ROPE_TAIL: return true;` (or `return ...` with the appropriate predicate matching how ROPE handles its support check).

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

- [ ] **Step 7.1: Run the dsv4_rope_tail test on CUDA**

Run: `./build-cuda/bin/test-backend-ops -o DSV4_ROPE_TAIL 2>&1 | tail -30`
Expected: tests PASS on both CPU and CUDA backends. The CUDA results should match CPU within `max_nmse_err` (1e-4 per Stream A).

If FAIL with large numerical error (e.g., `nmse = 0.5`):
- Common bug: index decomposition wrong → re-check `i0 / ne0 / ...` math against Metal source.
- Common bug: theta sign flip on inverse mode.
- Common bug: freq_factors index off-by-one (Metal uses `i_pair` not `2*i_pair` or vice versa).

If FAIL with kernel launch error:
- Add `CUDA_CHECK(cudaDeviceSynchronize())` after launch in the dispatch.
- Check for `out-of-bounds` via `cuda-memcheck` if available.

- [ ] **Step 7.2: Run with compute-sanitizer if available**

Run: `compute-sanitizer ./build-cuda/bin/test-backend-ops -o DSV4_ROPE_TAIL 2>&1 | tail -50` (if `compute-sanitizer` is in PATH; otherwise skip)
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

- [ ] **Step 8.2: Push branch**

```bash
git push -u origin feat/v4-port-cuda-B-rope-tail
```

- [ ] **Step 8.3: Note for parent merger**

This branch is ready to merge into `feat/v4-port-cuda`. The parent merger runs `gate-loader.sh` after the merge to confirm the build still produces a V4-capable binary.

---

## Definition of done (Stream B1)

- `ggml/src/ggml-cuda/dsv4-rope-tail.cu` and `.cuh` exist; kernel implements partial-RoPE matching the Metal reference.
- Op registered via `case GGML_OP_DSV4_ROPE_TAIL` in `ggml-cuda.cu`.
- CMakeLists includes the new `.cu` (or GLOB covers it).
- `./build-cuda/bin/test-backend-ops -o DSV4_ROPE_TAIL` PASSES with CUDA backend numerically matching CPU within `max_nmse_err`.
- No files outside `ggml/src/ggml-cuda/` touched.
- Branch `feat/v4-port-cuda-B-rope-tail` pushed.

## Out of scope (Stream B1)

- Performance optimization (TMA, shared-memory tiling). Functional correctness only.
- Multi-stream / async overlap. Single-stream dispatch matches the rest of ggml-cuda.
- FP16 / BF16 kernel variants. F32 only — the V4 graph emits F32 for this op.
- The full architecture validation on real models — that's the final on-hardware session, after all 5 streams merge.
