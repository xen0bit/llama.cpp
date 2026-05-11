# V4-port CUDA Stream C: dsv4_fp8_kv_quantize CUDA kernel (dual path)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CUDA for `ggml_dsv4_fp8_kv_quantize`. Pass the `test_dsv4_fp8_kv_quantize` cases from Stream A. Supports SM_89+ (native FP8) AND SM_70+ (software emulation) via compile-time `__CUDA_ARCH__` dispatch.

**Architecture:** New `.cu`/`.cuh` pair with dual-path kernel. The non-RoPE prefix (first `ne00 - n_rot` elements per row) is round-tripped through E4M3 FP8 quantization; the RoPE tail is passed through unchanged. Native path uses `__nv_cvt_float_to_fp8` / `__nv_cvt_fp8_to_float` intrinsics; software path emulates E4M3 via bit-pattern manipulation.

**Tech Stack:** CUDA C++ (with `__CUDA_ARCH__` conditional compilation), CMake.

**Spec:** `docs/superpowers/specs/2026-05-10-v4-port-cuda-design.md`

**Branch:** `feat/v4-port-cuda-C-fp8-kv` off `feat/v4-port-cuda`. **Prerequisite:** Stream A merged. Can run parallel to all B streams.

**Reference sources:**
- Metal kernel: `ggml/src/ggml-metal/ggml-metal.metal:2328-2403`
- Metal dispatch: `ggml/src/ggml-metal/ggml-metal-ops.cpp:1550-1594`
- CPU reference: `ggml/src/ggml-cpu/ops.cpp:11305+`
- Public API: `ggml/include/ggml.h:2591-2594`
- NVIDIA FP8 intrinsics: `cuda_fp8.h` (header shipped with CUDA toolkit 11.8+)

---

## Task 1: Branch + verify prerequisites

- [ ] **Step 1.1: Stream A merged**

Run: `git log feat/v4-port-cuda --oneline | grep -i "v4-port-cuda-A" | head -3`
Expected: at least one commit. Otherwise stop.

- [ ] **Step 1.2: Create branch + build**

```bash
cd ~/work/llama.cpp
git checkout feat/v4-port-cuda
git pull --ff-only origin feat/v4-port-cuda 2>/dev/null || true
git checkout -b feat/v4-port-cuda-C-fp8-kv
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89;120" 2>&1 | tail -5
cmake --build build-cuda -j --target test-backend-ops 2>&1 | tail -5
```

- [ ] **Step 1.3: Verify cuda_fp8.h is available**

Run: `find /usr/local/cuda* /opt/cuda* -name cuda_fp8.h 2>/dev/null | head -3`
Expected: at least one match. If empty, the CUDA toolkit is too old (need ≥11.8) — stop and report.

- [ ] **Step 1.4: Baseline test**

Run: `./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_FP8_KV_QUANTIZE 2>&1 | tail -10`
Expected: PASSES via CPU fallback.

---

## Task 2: Read references

- Metal kernel `ggml-metal.metal:2328-2403` — note the FP8 conversion semantics on Metal (uses `bfloat` or `half` packing; Metal does NOT have native FP8 intrinsics, so it's already a software emulation). Use the Metal source as a math reference even though the CUDA fast path will be faster.
- Metal dispatch `ggml-metal-ops.cpp:1550-1594` — 13 params, mostly tensor dims and strides.
- CPU ref `ggml-cpu/ops.cpp:11305+` — authoritative for the exact rounding behavior; the CUDA result MUST match the CPU result within FP8's representable precision.
- Optional reading: NVIDIA's "Using FP8 with Transformer Engine" docs explain E4M3FN format (1 sign + 4 exponent + 3 mantissa, with the all-1s exponent NOT reserved for inf/nan — that's the "FN" suffix for "finite"). Knowing this is essential for the software path.

---

## Task 3: Create the .cuh header

**Files:** Create: `ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cuh`

- [ ] **Step 3.1: Write the header**

```cpp
#pragma once

// V4 FP8 KV-cache simulation: quantizes/dequantizes the non-RoPE prefix
// in E4M3FN blocks, leaves the RoPE tail unchanged.
//
// The output is F32 with values that have been round-tripped through
// E4M3 FP8 representation, simulating the lossy storage of the V4 KV
// cache at compute-graph time.
//
// Reference Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2328-2403
// CPU reference:          ggml/src/ggml-cpu/ops.cpp:11305+
// Public API:             ggml/include/ggml.h:2591 (ggml_dsv4_fp8_kv_quantize)
//
// Dual-path implementation:
//   - __CUDA_ARCH__ >= 890 (Ada/Hopper/Blackwell): native FP8 intrinsics
//     via __nv_cvt_float_to_fp8 / __nv_cvt_fp8_to_float (cuda_fp8.h).
//   - __CUDA_ARCH__ < 890 (Volta/Turing/Ampere): software emulation via
//     bit-pattern manipulation of the F32 representation.
//
// Both paths produce numerically equivalent output (subject to FP8's
// inherent lossiness). The test_dsv4_fp8_kv_quantize cases in Stream A
// validate this with max_nmse_err = 1e-2.

#include "common.cuh"

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

- [ ] **Step 3.2: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cuh
git commit -m "v4-port-cuda-C-fp8-kv: header

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Create the .cu source — skeleton with both paths

**Files:** Create: `ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cu`

- [ ] **Step 4.1: Write the skeleton**

```cpp
#include "dsv4-fp8-kv-quantize.cuh"

#if __CUDA_ARCH__ >= 890
#include <cuda_fp8.h>
#endif

// Round-trip a single F32 value through E4M3FN representation.
// E4M3FN format: 1 sign + 4 exponent + 3 mantissa = 8 bits, no inf/nan
// reservation in the all-1s exponent (the "FN" suffix = "finite").
// Max representable: ±448. Subnormals supported.
static __device__ __forceinline__ float fp8_e4m3_roundtrip(float x) {
#if __CUDA_ARCH__ >= 890
    // Native path: convert to FP8 then back to F32.
    const __nv_fp8_e4m3 q = __nv_fp8_e4m3(x);
    return float(q);
#else
    // Software emulation. Translation of the bit-fiddling that the
    // Metal kernel performs at ggml-metal.metal:2328-2403, or
    // equivalently the CPU reference at ggml-cpu/ops.cpp:11305+.
    //
    // High-level algorithm:
    //   1. Clamp |x| to E4M3 max (~448).
    //   2. Extract F32 sign/exp/mantissa via bit-cast to uint32.
    //   3. Re-bias the exponent (F32 bias = 127, E4M3 bias = 7).
    //   4. Round the mantissa to 3 bits using round-to-nearest-even.
    //   5. Handle subnormals where the rebiased exponent goes negative.
    //   6. Re-expand to F32 by zero-extending the mantissa and adjusting bias.
    //
    // CRITICAL: match the CPU reference's rounding mode exactly. CPU uses
    // round-to-nearest-even (banker's rounding). Most FP8 implementations
    // do; verify with the CPU source before relying on this.
    //
    // The implementation is ~30 lines of bit manipulation; lift it from
    // the CPU reference (ggml-cpu/ops.cpp:11305+) and adapt.
    (void) x;
    return 0.0f;  // TODO: implement
#endif
}

static __global__ void dsv4_fp8_kv_quantize_f32(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int ne00, const int ne01, const int ne02, const int ne03,
        const int nb00, const int nb01, const int nb02, const int nb03,
        const int nb0,  const int nb1,  const int nb2,  const int nb3,
        const int n_rot) {
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)ne00 * ne01 * ne02 * ne03;
    if (gid >= total) return;

    // Decompose to (i0, i1, i2, i3)
    const int i0 = gid % ne00;
    const int rest1 = gid / ne00;
    const int i1 = rest1 % ne01;
    const int rest2 = rest1 / ne01;
    const int i2 = rest2 % ne02;
    const int i3 = rest2 / ne02;

    const float * s = (const float *)((const char *)src
        + i3 * nb03 + i2 * nb02 + i1 * nb01);
    float * d = (float *)((char *)dst
        + i3 * nb3 + i2 * nb2 + i1 * nb1);

    const int prefix = ne00 - n_rot;
    if (i0 < prefix) {
        // Quantize/dequantize through FP8.
        d[i0] = fp8_e4m3_roundtrip(s[i0]);
    } else {
        // RoPE tail: pass through.
        d[i0] = s[i0];
    }
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int n_rot = ggml_get_op_params_i32(dst, 0);

    const int64_t total = ggml_nelements(dst);
    constexpr int blk = 256;
    const dim3 grid((total + blk - 1) / blk);
    const dim3 block(blk);

    cudaStream_t stream = ctx.stream();
    dsv4_fp8_kv_quantize_f32<<<grid, block, 0, stream>>>(
        (const float *) src->data,
        (float *)       dst->data,
        (int) src->ne[0], (int) src->ne[1], (int) src->ne[2], (int) src->ne[3],
        (int) src->nb[0], (int) src->nb[1], (int) src->nb[2], (int) src->nb[3],
        (int) dst->nb[0], (int) dst->nb[1], (int) dst->nb[2], (int) dst->nb[3],
        n_rot);
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 4.2: Add to CMakeLists** (skip if GLOB)

- [ ] **Step 4.3: Build skeleton**

Run: `cmake --build build-cuda -j --target ggml-cuda 2>&1 | tail -10`
Expected: builds. On SM_89+ the native path compiles; on lower archs the software path stubs to `0.0f` until Task 5 implements it.

- [ ] **Step 4.4: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cu ggml/src/ggml-cuda/CMakeLists.txt
git commit -m "v4-port-cuda-C-fp8-kv: skeleton + dispatch + native FP8 path

Native path (__CUDA_ARCH__ >= 890) uses __nv_fp8_e4m3 intrinsics.
Software path stubbed for next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Implement the software emulation path

**Files:** Modify: `ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cu`

- [ ] **Step 5.1: Read the CPU reference**

Open `ggml/src/ggml-cpu/ops.cpp:11305+` and locate the F32→E4M3→F32 round-trip function. Note the exact bit-fiddling: typically a function called something like `fp32_to_e4m3_to_fp32` or inline at the call site. Note the rounding mode used.

- [ ] **Step 5.2: Implement the software path**

Replace the `#else` branch's TODO with a direct translation. Common idiom:

```cpp
    // Software emulation for SM < 89.
    // E4M3FN format: sign(1) + exp(4, bias 7) + mantissa(3), no inf/nan.
    // Max magnitude: 448 (= S.1111.110, exponent 15, mantissa 110)
    // Min normal:    2^-6 = 0.015625
    // Smallest subnormal: 2^-9 ≈ 0.00195

    union { float f; uint32_t u; } in = { x };
    const uint32_t sign = in.u >> 31;
    const int32_t  exp_f32 = (int32_t)((in.u >> 23) & 0xff) - 127;
    const uint32_t mant_f32 = in.u & 0x007fffff;

    // Handle NaN/Inf input: clamp to E4M3 max (matches IEEE no-NaN-output convention).
    if (exp_f32 == 128) {  // F32 inf or nan
        return sign ? -448.0f : 448.0f;
    }

    // Clamp to E4M3 max
    const float ax = fabsf(x);
    if (ax > 448.0f) {
        return sign ? -448.0f : 448.0f;
    }
    if (ax < ldexpf(1.0f, -9)) {  // smaller than smallest subnormal
        return 0.0f;
    }

    int32_t exp_e4m3 = exp_f32 + 7;  // re-bias
    uint32_t mant_e4m3;
    if (exp_e4m3 <= 0) {
        // Subnormal in E4M3
        const int shift = 1 - exp_e4m3;
        mant_e4m3 = (mant_f32 | 0x00800000) >> (20 + shift);
        exp_e4m3 = 0;
    } else if (exp_e4m3 >= 15) {
        // Saturate at max (binary: 0.1111.110 = 0x7e for positive)
        return sign ? -448.0f : 448.0f;
    } else {
        // Normal in E4M3: take top 3 bits of F32 mantissa with round-to-nearest-even
        const uint32_t rounding_bit = (mant_f32 >> 19) & 1;
        const uint32_t sticky_bits  = mant_f32 & 0x0007ffff;
        mant_e4m3 = mant_f32 >> 20;
        // Round-to-nearest-even
        if (rounding_bit && (sticky_bits != 0 || (mant_e4m3 & 1))) {
            mant_e4m3 += 1;
            if (mant_e4m3 >= 8) {
                mant_e4m3 = 0;
                exp_e4m3 += 1;
                if (exp_e4m3 >= 15) {
                    return sign ? -448.0f : 448.0f;
                }
            }
        }
    }

    // Now reconstruct as F32
    union { float f; uint32_t u; } out;
    if (exp_e4m3 == 0) {
        // Subnormal in E4M3 → reconstruct as F32 normal/subnormal
        if (mant_e4m3 == 0) {
            return sign ? -0.0f : 0.0f;
        }
        // Find leading bit of mant_e4m3 to normalize
        int lz = __clz(mant_e4m3) - (32 - 3);  // bits beyond the 3-bit mantissa
        const int exp_f32_out = -6 - lz;
        const uint32_t mant_f32_out = (mant_e4m3 << (20 + lz + 1)) & 0x007fffff;
        out.u = (sign << 31) | (((uint32_t)(exp_f32_out + 127)) << 23) | mant_f32_out;
    } else {
        // Normal in E4M3
        const int32_t exp_f32_out = exp_e4m3 - 7 + 127;
        out.u = (sign << 31) | (((uint32_t)exp_f32_out) << 23) | (mant_e4m3 << 20);
    }
    return out.f;
```

**Critical:** Verify this against the CPU reference before committing. The exact rounding behavior must match — a single ULP difference will fail the test tolerance check.

- [ ] **Step 5.3: Build for SM_70 (force the software path)**

To verify the software path compiles on older arches:

```bash
cmake -B build-cuda-sm70 -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70" 2>&1 | tail -5
cmake --build build-cuda-sm70 -j --target ggml-cuda 2>&1 | tail -10
```
Expected: builds successfully (validates the `#else` path compiles).

- [ ] **Step 5.4: Commit**

```bash
git add ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cu
git commit -m "v4-port-cuda-C-fp8-kv: software emulation path for SM < 89

E4M3FN bit-fiddling matching the CPU reference at
ggml-cpu/ops.cpp:11305+. Builds clean on -DCMAKE_CUDA_ARCHITECTURES=70.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Register in CUDA dispatcher

**Files:** Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`

- [ ] **Step 6.1: Add include + case + supports_op**

```cpp
#include "dsv4-fp8-kv-quantize.cuh"
// ...
        case GGML_OP_DSV4_FP8_KV_QUANTIZE:
            ggml_cuda_op_dsv4_fp8_kv_quantize(ctx, dst);
            break;
// in supports_op:
        case GGML_OP_DSV4_FP8_KV_QUANTIZE:
            return op->src[0]->type == GGML_TYPE_F32;
```

- [ ] **Step 6.2: Build full SM_89/SM_120**

```bash
cmake --build build-cuda -j 2>&1 | tail -10
```
Expected: builds.

- [ ] **Step 6.3: Commit**

```bash
git add ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "v4-port-cuda-C-fp8-kv: register dsv4_fp8_kv_quantize

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Validate both paths

- [ ] **Step 7.1: Run test on SM_89+ (native path) with count assertion**

The harness reports success on `0/0` so SKIPPED/NOT_SUPPORTED would silently pass. Stream A registered **4 dsv4_fp8_kv_quantize cases**.

```bash
./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_FP8_KV_QUANTIZE 2>&1 | tee /tmp/v4-cuda-C-fp8-kv-test.log | tail -30
COUNT=$(grep -E "^\s+[0-9]+/[0-9]+ tests passed" /tmp/v4-cuda-C-fp8-kv-test.log | tail -1 | grep -oE "^\s+[0-9]+" | tr -d ' ')
echo "Tests passed: ${COUNT:-0}"
test "${COUNT:-0}" -ge 4 || { echo "FAIL: only ${COUNT:-0} of 4+ expected tests ran"; exit 1; }
```

Expected: PASSES with `${COUNT:-0}` >= 4 and `max_nmse_err = 1e-2`. CUDA result matches CPU within FP8's representable precision.

If FAIL with NMSE around 0.1-0.5:
- Likely sign bit mishandling in the native path or wrong target FP8 format (E5M2 vs E4M3FN).
- Verify `__nv_fp8_e4m3` (E4M3 finite, not E4M3-with-NaN).

If FAIL with NMSE near 1.0:
- Likely a complete misroute — the wrong elements are being quantized (off-by-one on `n_rot`).

- [ ] **Step 7.2: Run test on SM_70 build (software path)**

```bash
./build-cuda-sm70/bin/test-backend-ops -b CPU,CUDA -o DSV4_FP8_KV_QUANTIZE 2>&1 | tail -30
```
Expected: PASSES with the same tolerance. If the software path doesn't match the native path within ~1 ULP, debug the rounding logic.

(Note: if you don't have an SM_70 GPU on your dev box, the kernel won't execute on it. The build-time check from Step 5.3 is the minimum gate; runtime validation of the software path requires an older GPU or a JIT-recompile flag.)

- [ ] **Step 7.3: compute-sanitizer**

```bash
compute-sanitizer ./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_FP8_KV_QUANTIZE 2>&1 | tail -20
```
Expected: clean.

- [ ] **Step 7.4: Commit any fixes**

If debugging required code changes, commit them.

---

## Task 8: Push

- [ ] **Step 8.1: Diff review**

Run: `git diff --stat feat/v4-port-cuda..HEAD`
Expected:
- New: `ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cuh`
- New: `ggml/src/ggml-cuda/dsv4-fp8-kv-quantize.cu`
- Modified: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modified: `ggml/src/ggml-cuda/CMakeLists.txt` (if not GLOB)

- [ ] **Step 8.2: Push**

```bash
git push -u origin feat/v4-port-cuda-C-fp8-kv
```

---

## Definition of done (Stream C)

- `dsv4-fp8-kv-quantize.{cu,cuh}` exist with dual native+software path.
- Software path verified to compile under `-DCMAKE_CUDA_ARCHITECTURES=70`.
- Op registered in dispatcher + supports_op.
- `test-backend-ops -o DSV4_FP8_KV_QUANTIZE` PASSES on SM_89/SM_120 build (`max_nmse_err = 1e-2`).
- Branch pushed.

## Out of scope (Stream C)

- Performance tuning of the native FP8 path. Functional correctness only.
- Vectorized fp8x2 / fp8x4 packed conversions. Single-element conversions are sufficient at the bandwidth required for KV-cache simulation.
- Runtime architecture detection / fallback selection. Compile-time `__CUDA_ARCH__` dispatch is the convention.
- Comparing software path output against the CPU reference on hardware that doesn't have native FP8 — runtime validation of the SM_70 path requires that GPU. Build-only validation is the per-stream gate.
