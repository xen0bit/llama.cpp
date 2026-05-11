# V4-port CUDA Stream C: dsv4_fp8_kv_quantize CUDA kernel (dual path)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CUDA for `ggml_dsv4_fp8_kv_quantize`. Pass the four `test_dsv4_fp8_kv_quantize` cases registered by Stream A (`tests/test-backend-ops.cpp:8868-8871`). Supports SM_89+ (native FP8) AND SM_70+ (software emulation) via compile-time `__CUDA_ARCH__` dispatch.

**Architecture:** New `.cu`/`.cuh` pair. The non-RoPE prefix (first `ne00 - n_rot` elements per row) is split into 64-element blocks; each block is **quantized with a per-block scale**: reduce `amax` across the 64 elements, compute `scale = 2^ceil(log2(max(amax, 1e-4) / 448))`, quantize/dequantize `clamp(v/scale, -448, 448)` through E4M3FN, multiply back by `scale`. The RoPE tail (`n_rot` trailing elements per row) is copied through unchanged. This block-scaled quantization is the actual algorithm used by both the CPU reference (`ggml/src/ggml-cpu/ops.cpp:11264-11313`) and the Metal kernel (`ggml/src/ggml-metal/ggml-metal.metal:2328-2376`). A naive raw-value round-trip would fail the NMSE test by orders of magnitude.

Native path uses NVIDIA's `__nv_fp8_e4m3` class wrapper (E4M3 finite encoding, round-to-nearest-even, saturate-to-finite via the constructor; explicit `float()` to dequantize); software path performs the same nearest-even E4M3FN code search as the CPU reference.

**Tech Stack:** CUDA C++ (with `__CUDA_ARCH__` conditional compilation), CMake.

**Spec:** `docs/superpowers/specs/2026-05-10-v4-port-cuda-design.md`

**Branch:** `feat/v4-port-cuda-C-fp8-kv` off `feat/v4-port-cuda`. **Prerequisite:** Stream A merged. Can run parallel to all B streams.

**Reference sources (authoritative):**
- CPU reference (this is the bit-exact spec the test compares against): `ggml/src/ggml-cpu/ops.cpp:11235-11313`
- Metal kernel: `ggml/src/ggml-metal/ggml-metal.metal:2302-2376`
- Metal dispatch: `ggml/src/ggml-metal/ggml-metal-ops.cpp:1550-1594`
- Public API: `ggml/include/ggml.h:2591-2594`
- Stream A test cases: `tests/test-backend-ops.cpp:5008-5050` (struct), `8862-8871` (registrations)
- NVIDIA FP8 intrinsics: `cuda_fp8.h` (header shipped with CUDA toolkit ≥11.8)

---

## Task 1: Branch + verify prerequisites

- [ ] **Step 1.1: Stream A merged**

Run: `git log feat/v4-port-cuda --oneline | grep -i "v4-port-cuda-A" | head -3`
Expected: at least one commit. Otherwise stop.

- [ ] **Step 1.2: Create branch + initial build**

```bash
cd ~/work/llama.cpp
git checkout feat/v4-port-cuda
git pull --ff-only mine feat/v4-port-cuda 2>/dev/null || true
git checkout -b feat/v4-port-cuda-C-fp8-kv
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89;120" 2>&1 | tail -5
cmake --build build-cuda -j --target test-backend-ops 2>&1 | tail -5
```

- [ ] **Step 1.3: Verify cuda_fp8.h is available**

Run: `find /usr/local/cuda* /opt/cuda* -name cuda_fp8.h 2>/dev/null | head -3`
Expected: at least one match. If empty, the CUDA toolkit is too old (need ≥11.8) — stop and report.

- [ ] **Step 1.4: Baseline test (no kernel registered yet)**

```bash
./build-cuda/bin/test-backend-ops -o DSV4_FP8_KV_QUANTIZE 2>&1 | tail -10
```

Expected: passes for the CPU backend. CUDA may report `not supported` and skip — that's fine pre-registration. The `-b` flag is intentionally omitted: `test-backend-ops` parses `-b` via exact `strcmp` at `tests/test-backend-ops.cpp:9882`, so any comma-list silently no-ops; with `-b` omitted, the harness auto-iterates every registered non-CPU backend using CPU as internal reference (see `tests/test-backend-ops.cpp:9615-9620`).

---

## Task 2: Read references (do not skip)

Read these in order — Step 5 (software-path implementation) requires understanding them.

1. **CPU reference** `ggml/src/ggml-cpu/ops.cpp:11264-11313` (function `ggml_compute_forward_dsv4_fp8_kv_quantize` and helper `ggml_dsv4_e4m3fn_dequant` at `11235-11262`). This is the bit-exact specification the `test-backend-ops` framework compares CUDA output against. Note three structural points:
   - Per-row iteration over `n_rows = ne01 * ne02 * ne03`.
   - For each row, the non-RoPE prefix `n_nope = ne00 - n_rot` is processed in 64-element blocks (`for off = 0; off < n_nope; off += 64`).
   - Per block: pass 1 reduces `amax = max |v|`; `amax = max(amax, 1e-4f)`; `scale = ldexp(1.0f, ceil(log2(amax / 448.0f)))`. Pass 2 quantizes each element as `dequant(clamp(v / scale, -448, 448)) * scale`.
   - The E4M3FN encoder (`ggml_dsv4_e4m3fn_dequant`) enumerates codes `1..126`, computes each code's float value, and picks the one with smallest absolute difference to `|x|`, breaking ties toward the even code. The tail (`i in [n_nope, ne00)`) is copied through unchanged.

2. **Metal kernel** `ggml/src/ggml-metal/ggml-metal.metal:2302-2376` (helpers `dsv4_e4m3fn_value`, `dsv4_e4m3fn_dequant`, kernel `kernel_dsv4_fp8_kv_quantize_f32`). The Metal layout is one threadgroup per row, 64 threads per threadgroup, threadgroup-scratch reduction across the 64 elements per block. This is the layout the CUDA kernel will mirror.

3. **Metal dispatch** `ggml/src/ggml-metal/ggml-metal-ops.cpp:1550-1594`. Confirms `n_rot = op_params_i32(0)` and the 13-arg ABI. We pass the same scalars to the CUDA kernel.

4. **Stream A test cases** `tests/test-backend-ops.cpp:5008-5050` and `8862-8871`. Note `max_nmse_err = 1e-3` (line 5034). The four registered shapes use `n_rot = 64` or `128`, and `n_nope` is always a multiple of 64 (asserted at construction).

5. **NVIDIA FP8 docs**: the supported public API is the class wrapper `__nv_fp8_e4m3` from `<cuda_fp8.h>`. The constructor `__nv_fp8_e4m3(float)` round-to-nearest-even with saturate-to-finite to ±448; explicit `float(q)` dequantizes back to F32. (The lower-level `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)` returns `__nv_fp8_storage_t`, but there is no symmetric `__nv_cvt_fp8_to_float`; only `__nv_cvt_fp8_to_halfraw`. The class wrapper is the right level of abstraction here and avoids a half hop.)

---

## Task 3: Create the .cuh header

**Files:** Create: `ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cuh`

- [ ] **Step 3.1: Write the header**

```cpp
#pragma once

// V4 FP8 KV-cache simulation: quantizes/dequantizes the non-RoPE prefix
// of each row in 64-element blocks through E4M3FN representation with
// per-block scaling; leaves the RoPE tail unchanged.
//
// Block-scaled algorithm (must match CPU reference for the
// test-backend-ops NMSE check):
//   for each row (n_rows = ne01 * ne02 * ne03):
//     for off in [0, n_nope) step 64:
//       amax  = max(|src[off..off+64)|, 1e-4)
//       scale = 2^ceil(log2(amax / 448))
//       dst[off+i] = dequant_e4m3fn(clamp(src[off+i]/scale, -448, 448)) * scale
//     copy src[n_nope..ne00) to dst unchanged   // RoPE tail
//
// References:
//   CPU reference:        ggml/src/ggml-cpu/ops.cpp:11235-11313
//   Metal kernel:         ggml/src/ggml-metal/ggml-metal.metal:2302-2376
//   Metal dispatch:       ggml/src/ggml-metal/ggml-metal-ops.cpp:1550-1594
//   Public API:           ggml/include/ggml.h:2591 (ggml_dsv4_fp8_kv_quantize)
//
// Dual-path implementation:
//   - __CUDA_ARCH__ >= 890 (Ada/Hopper/Blackwell): native FP8 via the
//     __nv_fp8_e4m3 class wrapper from <cuda_fp8.h> (round-to-nearest-even,
//     saturate-to-finite to +/-448).
//   - __CUDA_ARCH__ <  890 (Volta/Turing/Ampere): software emulation by
//     nearest-even E4M3FN code search, mirroring the CPU reference.
//
// Both paths produce numerically equivalent output (subject to FP8's
// inherent lossiness). The four test_dsv4_fp8_kv_quantize cases from
// Stream A (tests/test-backend-ops.cpp:8868-8871) validate with
// max_nmse_err = 1e-3.

#include "common.cuh"

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

- [ ] **Step 3.2: Add to CMakeLists** (only if `ggml/src/ggml-cuda/CMakeLists.txt` does NOT use a glob over `*.cu`)

Check with `head -40 ggml/src/ggml-cuda/CMakeLists.txt`. If it uses `file(GLOB ... *.cu)` pattern, no change is needed (this is the upstream pattern). If it lists each `.cu` explicitly, add `dsv4-fp8-kv-quantize.cu` alongside the other dsv4 entries.

- [ ] **Step 3.3: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cuh
git add ggml/src/ggml-cuda/CMakeLists.txt  # only if modified
git commit
```

Use a concise human-written commit message. Do not use the `Co-Authored-By` line at this stage (AGENTS.md policy).

---

## Task 4: Create the .cu source — block-scaled kernel skeleton

**Files:** Create: `ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cu`

The kernel maps **one CUDA block per row**, **64 threads per block** (matching Metal's threadgroup layout). Each block iterates the row's prefix in 64-element chunks: thread `tid` loads one element, all 64 threads cooperate on a warp-shuffle (or shared-mem) max-reduction for `amax`, then each thread quantizes and stores its element. The RoPE tail is copied through after the prefix loop.

- [ ] **Step 4.1: Write the skeleton**

```cpp
#include "dsv4-fp8-kv-quantize.cuh"

#if __CUDA_ARCH__ >= 890
#include <cuda_fp8.h>
#endif

#include <cstdint>

// E4M3FN code value: 0..127.
// Format: 1 sign + 4 exponent + 3 mantissa, bias 7, no inf/nan reserved.
// (i >> 3) & 0xf = exponent, i & 7 = mantissa. Code 0 is +0.
// Mirrors the CPU helper dsv4_e4m3fn_value / Metal dsv4_e4m3fn_value.
static __device__ __forceinline__ float dsv4_e4m3fn_value(int i) {
    const int e = (i >> 3) & 0x0f;
    const int m = i & 0x07;
    return e == 0
        ? float(m) * 0.001953125f                              // 2^-9 * m  (subnormal)
        : (1.0f + float(m) * 0.125f) * exp2f(float(e - 7));    // normal
}

// Round |x| to the nearest E4M3FN positive code value, breaking ties
// toward the EVEN code (matches CPU reference ops.cpp:11242-11253 exactly).
// Returns the dequantized F32, sign-preserved.
static __device__ __forceinline__ float dsv4_e4m3fn_dequant_sw(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax   = fminf(fabsf(x), 448.0f);

    int   best      = 0;
    float best_diff = ax;
    #pragma unroll
    for (int i = 1; i < 127; ++i) {
        const float val  = dsv4_e4m3fn_value(i);
        const float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best      = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e4m3fn_value(best);
}

// Dual-path E4M3FN quantize+dequantize round-trip with saturation.
//
// Native path uses NVIDIA's documented FP8 class API. The constructor
// __nv_fp8_e4m3(float) applies round-to-nearest-even and saturates to
// the finite E4M3 range (±448). The explicit float() conversion expands
// the FP8 storage back to F32. This is the supported public API per
// NVIDIA's cuda_fp8.h headers (CUDA toolkit >= 11.8).
//
// (We intentionally avoid the lower-level __nv_cvt_fp8_to_halfraw +
// __half2float chain: the class wrapper is clearer and avoids a half
// hop on F32-only data.)
static __device__ __forceinline__ float dsv4_e4m3fn_roundtrip(float x) {
#if __CUDA_ARCH__ >= 890
    const __nv_fp8_e4m3 q(x);
    return float(q);
#else
    // Software emulation: matches CPU reference bit-for-bit.
    return dsv4_e4m3fn_dequant_sw(x);
#endif
}

// Warp-level (32 threads) max-reduction via __shfl_xor_sync.
static __device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// One block per row. blockDim.x == 64 (two warps).
static __global__ void dsv4_fp8_kv_quantize_f32(
        const char * __restrict__ src,
        char       * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb0,  const int64_t nb1,  const int64_t nb2,  const int64_t nb3,
        const int     n_rot) {

    const int64_t n_rows = ne01 * ne02 * ne03;
    const int64_t row    = blockIdx.x;
    if (row >= n_rows) return;

    const int     tid = threadIdx.x;                    // 0..63
    const int     warp_id = tid >> 5;                   // 0 or 1
    const int     lane    = tid & 31;

    const int64_t i1 = row % ne01;
    const int64_t i2 = (row / ne01) % ne02;
    const int64_t i3 = row / (ne01 * ne02);

    const char * src_base = src + i1*nb01 + i2*nb02 + i3*nb03;
    char       * dst_base = dst + i1*nb1  + i2*nb2  + i3*nb3;

    const int64_t n_nope = ne00 - (int64_t) n_rot;

    // Shared-mem slot for the two warps' partial max.
    __shared__ float warp_max[2];

    // Prefix loop: 64-element blocks.
    for (int64_t off = 0; off < n_nope; off += 64) {
        const float v = *(const float *)(src_base + (off + tid) * nb00);

        // Two-stage block-max reduction across 64 threads.
        // Stage 1: each warp reduces its 32 lanes via shfl_xor; lane 0 stores
        //          the warp's max to shared memory.
        // Stage 2: a single thread (warp 0, lane 0) combines the two warp maxes
        //          and writes the final block max back to warp_max[0].
        float m = warp_reduce_max(fabsf(v));
        if (lane == 0) warp_max[warp_id] = m;
        __syncthreads();
        if (warp_id == 0 && lane == 0) {
            warp_max[0] = fmaxf(warp_max[0], warp_max[1]);
        }
        __syncthreads();

        const float amax  = fmaxf(warp_max[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));

        const float q = dsv4_e4m3fn_roundtrip(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
        *(float *)(dst_base + (off + tid) * nb0) = q;

        __syncthreads();  // protect warp_max for the next block
    }

    // Tail loop: copy n_rot elements per row through unchanged.
    // 64 threads stride through the tail.
    for (int64_t i = n_nope + tid; i < ne00; i += 64) {
        *(float *)(dst_base + i * nb0) = *(const float *)(src_base + i * nb00);
    }
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src, dst));

    const int n_rot = ggml_get_op_params_i32(dst, 0);
    const int64_t head_dim = src->ne[0];
    const int64_t n_nope = head_dim - (int64_t) n_rot;

    GGML_ASSERT(n_rot >= 0);
    GGML_ASSERT(n_nope > 0);
    GGML_ASSERT(n_nope % 64 == 0);

    const int64_t n_rows = src->ne[1] * src->ne[2] * src->ne[3];

    const dim3 grid((unsigned) n_rows, 1, 1);
    const dim3 block(64, 1, 1);

    cudaStream_t stream = ctx.stream();
    dsv4_fp8_kv_quantize_f32<<<grid, block, 0, stream>>>(
        (const char *) src->data,
        (      char *) dst->data,
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        (int64_t) src->nb[0], (int64_t) src->nb[1], (int64_t) src->nb[2], (int64_t) src->nb[3],
        (int64_t) dst->nb[0], (int64_t) dst->nb[1], (int64_t) dst->nb[2], (int64_t) dst->nb[3],
        n_rot);
    CUDA_CHECK(cudaGetLastError());
}
```

Key correctness points (all required to pass the NMSE test):
- **One block per row, 64 threads.** Block-scoped `__syncthreads()` is valid only because we use one block per row, and a row's prefix is processed serially across 64-element windows by the same block.
- **Byte-stride indexing on the innermost dim** using `nb00`/`nb0` rather than `s[i0]`/`d[i0]`. CPU and Metal references both index with `nb[0]`, and `test-backend-ops` does not guarantee `nb[0] == sizeof(float)` for all cases.
- **Block-max reduction** uses warp shuffle (`__shfl_xor_sync`) for the two warps and a 2-slot shared array to combine them. No external library dependency.
- **Per-block scale** matches the CPU reference exactly. `amax = max(amax, 1e-4)` floors the scale and prevents `log2(0) = -inf`. `scale = exp2(ceil(log2(amax/448)))` snaps to the next power of two (allows full E4M3 dynamic range without clipping).
- **Quantize-then-dequantize** uses `clamp(v/scale, -448, 448)`, NOT raw `v`. Multiplying the dequantized result by `scale` recovers the original magnitude. Skipping the scale is the bug codex flagged.

- [ ] **Step 4.2: Build for SM_89/SM_120**

```bash
cmake --build build-cuda -j --target ggml-cuda 2>&1 | tail -10
```

Expected: compiles. (The kernel won't be reached until Task 6 registers the op; we just need the TU to build.)

- [ ] **Step 4.3: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cu
git commit
```

Use a concise human-written message (no AI co-author line).

---

## Task 5: Verify the software path compiles on SM_70

The software path is already implemented inline in `dsv4_e4m3fn_dequant_sw` (Task 4 Step 4.1). This task just confirms it compiles for the SM_70 target — required because the `#if __CUDA_ARCH__ >= 890` block must NOT reference `__nv_*` intrinsics on the SM_70 device pass.

- [ ] **Step 5.1: Build for SM_70 (forces the `#else` branch)**

```bash
cmake -B build-cuda-sm70 -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70" 2>&1 | tail -5
cmake --build build-cuda-sm70 -j --target ggml-cuda 2>&1 | tail -10
```

Expected: builds successfully. The preprocessor evaluates `__CUDA_ARCH__` per device pass; on SM_70 the `<cuda_fp8.h>` include is skipped and the software path is taken. If this fails (e.g. compiler still tries to instantiate `__nv_cvt_float_to_fp8`), the includes/macros are mis-guarded.

- [ ] **Step 5.2: Inspect the SM_70 PTX (sanity check, not a gate)**

```bash
cuobjdump --dump-ptx build-cuda-sm70/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/dsv4-fp8-kv-quantize.cu.o 2>&1 \
    | head -20 || true
```

Optional. If `__nv_cvt_float_to_fp8` symbols appear in the SM_70 PTX, the guards are wrong.

- [ ] **Step 5.3: Commit if any fixes were needed**

If the SM_70 build required corrections (e.g. moving an include inside the guard), commit them with a concise message.

---

## Task 6: Register in CUDA dispatcher

**Files:** Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`

- [ ] **Step 6.1: Add include + dispatch case + supports_op arm**

In the top-of-file include block (near the other `dsv4-*` includes if any, otherwise alongside existing op headers):

```cpp
#include "dsv4-fp8-kv-quantize.cuh"
```

In `ggml_cuda_compute_forward` (the big switch around line 2942 today), add an arm beside the existing `case GGML_OP_DSV4_HC_EXPAND:`:

```cpp
        case GGML_OP_DSV4_FP8_KV_QUANTIZE:
            ggml_cuda_op_dsv4_fp8_kv_quantize(ctx, dst);
            break;
```

In `ggml_backend_cuda_device_supports_op` (around line 5205, beside the existing `case GGML_OP_DSV4_HC_EXPAND` arm):

```cpp
        case GGML_OP_DSV4_FP8_KV_QUANTIZE:
            return op->type == GGML_TYPE_F32
                && op->src[0]->type == GGML_TYPE_F32;
```

Both `op->type` and `op->src[0]->type` are checked, matching CPU/Metal assertions (`ops.cpp:11269-11270`, `ggml-metal-ops.cpp:1556-1557`).

- [ ] **Step 6.2: Full SM_89/SM_120 build**

```bash
cmake --build build-cuda -j 2>&1 | tail -10
```

Expected: builds clean.

- [ ] **Step 6.3: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu
git commit
```

Concise human-written message; no AI co-author line.

---

## Task 7: Validate

- [ ] **Step 7.1: Run test on SM_89/SM_120 build (native path) with count assertion**

The harness reports "0/0 tests passed" as a pass, so a silently-skipped op would slip through. Stream A registered **4** dsv4_fp8_kv_quantize cases (`tests/test-backend-ops.cpp:8868-8871`); we assert `COUNT >= 4`.

```bash
./build-cuda/bin/test-backend-ops -o DSV4_FP8_KV_QUANTIZE 2>&1 | tee /tmp/v4-cuda-C-fp8-kv-test.log | tail -30
COUNT=$(grep -E "^\s+[0-9]+/[0-9]+ tests passed" /tmp/v4-cuda-C-fp8-kv-test.log | tail -1 | grep -oE "^\s+[0-9]+" | tr -d ' ')
echo "Tests passed: ${COUNT:-0}"
test "${COUNT:-0}" -ge 4 || { echo "FAIL: only ${COUNT:-0} of 4 expected tests ran"; exit 1; }
```

Expected: PASSES with `${COUNT}` >= 4 and `max_nmse_err = 1e-3`. CUDA result matches CPU within FP8's representable precision.

**Why no `-b` flag with a backend list:** `tests/test-backend-ops.cpp:9882` parses `-b` via exact `strcmp` — comma-separated lists silently no-op. The harness already auto-iterates every registered non-CPU backend using CPU as the internal reference (see `tests/test-backend-ops.cpp:9615-9620`). Omitting `-b` is the documented invocation and exercises CUDA-vs-CPU correctly.

Debugging hints if the test fails:
- **NMSE ≈ 0.5–1.0** → likely raw-value round-trip (forgot the per-block scale); recompute against the CPU loop at `ops.cpp:11293-11306`.
- **NMSE ≈ 0.1–0.3** → likely the block-max reduction is reading stale shared memory (missing `__syncthreads()` between blocks) or the scale formula is using the wrong base (must be `log2` and `exp2`, both base 2).
- **NMSE ≈ 0.05** with off-by-one in element count → byte-stride indexing on the innermost dim is wrong; double-check `(off + tid) * nb00` vs `(off + tid) * sizeof(float)`.
- **NMSE just over 1e-3** → rounding tie-break disagrees with CPU (`(i & 1) == 0 && (best & 1) != 0` condition); verify the `dsv4_e4m3fn_dequant_sw` loop matches CPU verbatim.
- **NaN output** → likely `log2(0)` from a forgotten `max(amax, 1e-4)` floor, or `__shfl_xor_sync` mask wrong on the second warp.

- [ ] **Step 7.2: Run test on SM_70 build (software path)**

```bash
./build-cuda-sm70/bin/test-backend-ops -o DSV4_FP8_KV_QUANTIZE 2>&1 | tail -30
```

Expected: PASSES with the same tolerance — but only if the validation box has an actual SM_70 device. If the dev box's GPU is Ada/Blackwell, the SM_70 binary won't load any kernel and the test will report "not supported" or skip; in that case, the SM_70 *build success* from Step 5.1 is the gate, and runtime validation of the software path is deferred to an older-GPU CI box. Note this outcome in the task history.

- [ ] **Step 7.3: compute-sanitizer**

```bash
compute-sanitizer ./build-cuda/bin/test-backend-ops -o DSV4_FP8_KV_QUANTIZE 2>&1 | tail -20
```

Expected: clean — no races, no out-of-bounds. The `__syncthreads()` between block iterations is essential here; without it, `warp_max` collides across iterations.

- [ ] **Step 7.4: Commit any fixes**

If debugging required code changes, commit them.

---

## Task 8: Push

- [ ] **Step 8.1: Diff review**

```bash
git diff --stat feat/v4-port-cuda..HEAD
```

Expected:
- New: `ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cuh`
- New: `ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cu`
- Modified: `ggml/src/ggml-cuda/ggml-cuda.cu` (one dispatch case, one supports_op arm, one include)
- Modified: `ggml/src/ggml-cuda/CMakeLists.txt` (only if it doesn't use a `*.cu` glob)

- [ ] **Step 8.2: Push to `mine`**

```bash
git push -u mine feat/v4-port-cuda-C-fp8-kv
```

Push target is `mine` because PR_BEHAVIOR=internal-merge for this project. Do NOT run `gh pr create` — the orchestrator handles the merge into `feat/v4-port-cuda`.

---

## Definition of done (Stream C)

- `dsv4-fp8-kv-quantize.{cu,cuh}` exist with dual native+software path.
- Algorithm matches the CPU reference exactly: per-row, 64-element blocks, per-block `amax`/`scale` derivation, `clamp(v/scale, -448, 448)` quantize-dequantize, multiply by `scale`. RoPE tail copied through.
- Software path verified to compile under `-DCMAKE_CUDA_ARCHITECTURES=70`.
- Op registered in dispatcher + `supports_op` (both `op->type` and `op->src[0]->type` checked).
- `test-backend-ops -o DSV4_FP8_KV_QUANTIZE` PASSES on SM_89/SM_120 build with `COUNT >= 4` and `max_nmse_err = 1e-3`.
- Branch pushed to `mine`.

## Out of scope (Stream C)

- Performance tuning of the native FP8 path. Functional correctness only.
- Vectorized fp8x2 / fp8x4 packed conversions. Single-element conversions are sufficient at KV-cache simulation bandwidth.
- Runtime architecture detection / fallback selection. Compile-time `__CUDA_ARCH__` dispatch is the convention.
- Runtime validation of the SM_70 path on hardware without an SM_70 GPU — build-only validation is the per-stream gate.
- Editing `tests/test-backend-ops.cpp` — Stream A already added the four cases.
