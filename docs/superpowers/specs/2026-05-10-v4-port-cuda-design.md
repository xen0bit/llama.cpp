# V4-port CUDA Port — Design Spec

**Status:** approved 2026-05-10
**Parent branch:** `feat/v4-port`
**Target branch:** `feat/v4-port-cuda`

## Purpose

V4-Flash currently runs on Apple Silicon (Metal) and CPU. The 5 V4-specific
custom ops have no CUDA kernels — running V4 on a CUDA host today triggers a
fallback to the CPU compute path for those ops, which is too slow to be usable.
This spec defines the work to add CUDA implementations for all 5 ops with a
quality bar suitable for upstream merge into `ggml-org/llama.cpp` as part of the
V4 architecture upstreaming bundle.

The 5 ops are:

- `ggml_dsv4_rope_tail`
- `ggml_dsv4_hc_split_sinkhorn`
- `ggml_dsv4_hc_weighted_sum`
- `ggml_dsv4_hc_expand`
- `ggml_dsv4_fp8_kv_quantize`

## Definition of done

"Functional parity" — *not* performance parity with Metal. Specifically:

- All 5 ops are registered on the CUDA backend (no more CPU fallback for these
  ops on a CUDA-enabled build).
- Numerical output matches the CPU reference within per-op tolerance (FP32 baseline
  1e-4 absolute / 1e-3 relative; tighter where the op admits it).
- `test-backend-ops` includes 5 new test classes covering the V4 ops; all pass on
  CUDA backend.
- `tests/v4-port/gate-loader.sh`, `gate-coherence.sh`, and `gate-tools.sh` pass on
  a CUDA build (`-DGGML_CUDA=ON`).
- `gate-speed.sh` records a t/s number but no floor is enforced for this phase.
- Code style and structure are upstream-merge-ready: no scratch code, no dev-only
  diagnostics in the merged diff, no compute-capability runtime checks that
  belong at compile time.

## Hardware constraints

**Primary dev target:** RTX 5090 (Blackwell, SM_120, 32 GB). Native FP8 + FP4.

**Validation target:** 2× RTX 6000 Ada (Lovelace, SM_89, 48 GB each = 96 GB
combined). Native FP8 e4m3 instructions via `__nv_cvt_*_to_fp8`.

**Test-session scarcity:** Both machines have limited availability. Validation
sessions must be batched — all 5 streams must be merge-ready before the session
opens. The CPU-derived gold fixtures (Stream A) exist specifically to let the
other streams iterate without GPU time.

**Upstream targets:** ggml-org CI runs on older NVIDIA hardware (down to SM_70
Volta). The CUDA port must compile on SM_70+. FP8 native intrinsics are gated
behind `__CUDA_ARCH__ >= 890`; older architectures get a software emulation path
in the FP8 KV op.

## Branch topology

```
feat/v4-port (current parent)
└── feat/v4-port-cuda (port parent)
    ├── feat/v4-port-cuda-A-fixtures    (Stream 1; lands first, blocks B/C)
    ├── feat/v4-port-cuda-B-rope-tail   (Stream 2a)
    ├── feat/v4-port-cuda-B-sinkhorn    (Stream 2b)
    ├── feat/v4-port-cuda-B-weighted-sum(Stream 2c)
    ├── feat/v4-port-cuda-B-expand      (Stream 2d)
    └── feat/v4-port-cuda-C-fp8-kv      (Stream 3; parallel to B but isolated)
```

**Stream order:** Stream A (fixtures) must merge into `feat/v4-port-cuda` first
because every other stream's `test-backend-ops` cases depend on the harness it
ships. Streams B1-B4 and C can run in parallel after A lands.

**Conflict-resolution rule:** Each per-op stream modifies exactly one
`case GGML_OP_DSV4_*:` block in `ggml/src/ggml-cuda/ggml-cuda.cu`. Per-op edits
are physically disjoint and merge without conflict in any order. The parent-
branch merger runs `gate-loader.sh` after each per-op merge to confirm the
build still produces a V4-capable binary.

## Stream A: Test infrastructure

**Branch:** `feat/v4-port-cuda-A-fixtures`

**Deliverables:**

1. `tests/v4-port/fixtures/gen-dsv4-fixtures.cpp` — CLI utility that loads the
   CPU reference path, generates seeded random input tensors of representative
   shape per op, runs the op, dumps `{inputs, outputs, op_params}` as little-
   endian binary to `tests/v4-port/fixtures/{op_name}.bin`.
2. 5 new test classes in `tests/test-backend-ops.cpp` (one per op), following
   the pattern of the existing ROPE tests at lines 2382-2526. Each loads its
   `.bin` fixture, runs the op on every enabled backend, asserts the output
   matches the dumped reference within tolerance.
3. Per-op tolerance constants documented in-source with one-sentence rationale
   each (e.g., "1e-4 abs — sinkhorn iterates 4× FP32 multiplies, max
   accumulated rounding error empirical 3.2e-5 on CPU reference").
4. `.gitignore` entry for `tests/v4-port/fixtures/*.bin` (binaries are
   regenerated, never committed).
5. README snippet in `tests/v4-port/fixtures/` explaining how to regenerate
   fixtures (one command), so upstream maintainers can reproduce.

**Size:** ~600 lines (utility + 5 test classes + tolerance docs).

**Acceptance:** `cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_METAL=OFF`,
`./build-cpu/bin/test-backend-ops -o DSV4_ROPE_TAIL` (and equivalents) all pass.

## Streams B1-B4: Four "easy" ops

Each stream produces identical artifact shape:

- `ggml/src/ggml-cuda/dsv4-{op_name}.cu` — kernel + dispatch function
  `ggml_cuda_op_dsv4_{op_name}(ctx, dst)`.
- `ggml/src/ggml-cuda/dsv4-{op_name}.cuh` — header with dispatch function
  declaration and a 1-paragraph kernel design note in the file-header comment
  (what the op does, what shape it expects, reference to Metal kernel
  file/lines and CPU reference file/lines).
- One new `case GGML_OP_DSV4_*:` block in `ggml/src/ggml-cuda/ggml-cuda.cu`
  (existing switch starting at line 2629).
- One CMakeLists entry (`ggml/src/ggml-cuda/CMakeLists.txt`) adding the new
  `.cu` source to the build.

**Tolerance values below are initial estimates.** Stream A measures the
actual CPU-reference rounding error per op and updates the constants. The
B/C streams use the finalized values from Stream A.

### B1: `dsv4_rope_tail`

22-param kernel ABI lifted from `ggml-metal-ops.cpp:1596-1673`:
- tensor strides (ne0-ne3, nb0-nb3)
- scalars: n_dims, mode, n_ctx_orig, inverse
- ROPE params: freq_base, freq_scale, ext_factor, attn_factor, beta_fast,
  beta_slow
- src2_flag

Reference: Metal kernel at `ggml-metal.metal:4906-4997`, CPU reference at
`ggml/src/ggml-cpu/ops.cpp:5961`.

Tolerance: 1e-5 abs / 1e-4 rel (ROPE is mostly trig + multiply; tight is fine).

Size: ~350 lines.

### B2: `dsv4_hc_split_sinkhorn`

7-param ABI from `ggml-metal-ops.cpp:1392-1438`:
- n_hc, sinkhorn_iters, n_rows, mix_hc, nb01, nb1, eps

Reference: Metal `ggml-metal.metal:2076-2245`, CPU `ggml-cpu/ops.cpp:10990+`.

Tolerance: 1e-4 abs / 1e-3 rel (4 iterations of normalization compound rounding).

Size: ~450 lines.

### B3: `dsv4_hc_weighted_sum`

10-param ABI from `ggml-metal-ops.cpp:1440-1486`:
- n_embd, n_hc, n_tokens, 5 input strides, 2 output strides

Reference: Metal `ggml-metal.metal:2278-2327`, CPU `ggml-cpu/ops.cpp:11100+`.

Tolerance: 1e-5 abs / 1e-4 rel (pure weighted sum, minimal accumulation).

Size: ~250 lines.

### B4: `dsv4_hc_expand`

16-param ABI from `ggml-metal-ops.cpp:1488-1548`:
- n_embd, n_hc, n_tokens, 12 input strides, 3 output strides

Reference: Metal `ggml-metal.metal:2247-2276`, CPU `ggml-cpu/ops.cpp:11200+`.

Tolerance: 1e-5 abs / 1e-4 rel.

Size: ~400 lines.

## Stream C: `dsv4_fp8_kv_quantize`

**Branch:** `feat/v4-port-cuda-C-fp8-kv`

The hardest op. Same artifact shape as B-streams, plus dual-path FP8 dispatch.

13-param ABI from `ggml-metal-ops.cpp:1550-1594`:
- input dims (ne00-ne03)
- input strides (nb00-nb03)
- output strides (nb0-nb3)
- n_rot

Reference: Metal `ggml-metal.metal:2328-2403`, CPU `ggml-cpu/ops.cpp:11305`.

**Dual-path dispatch:**

```cpp
#if __CUDA_ARCH__ >= 890
    // Native path: __nv_cvt_float_to_fp8 + __nv_fp8_e4m3 storage
#else
    // Software emulation: bit-pattern conversion, per-row scale derivation
    // in shared memory, manual sign/exp/mantissa packing
#endif
```

Both paths must produce numerically equivalent output (subject to FP8's inherent
quantization noise). The `test-backend-ops` case for this op must run both
paths and compare against the CPU reference.

Tolerance: 1e-3 abs (FP8 e4m3 quantization is inherently lossy; the test
asserts dequantize(quantize(x)) ≈ x within FP8's representable precision).

Size: ~700 lines (kernel + dual paths + extra test surface).

## Per-op kernel ABI: function-arg convention

CUDA kernels accept parameters as direct `__global__` function arguments, NOT a
kargs struct buffer. This follows the existing ggml-cuda pattern (see
`ggml/src/ggml-cuda/rope.cu`). The kargs struct used by Metal is an
implementation detail of the Metal backend; the ABI contract is the parameter
list, which is identical across both backends.

## Data flow: gold-fixture protocol

```
[Stream A, one-time setup]
  Build CPU-only:
    cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_METAL=OFF
    cmake --build build-cpu --target gen-dsv4-fixtures
  Generate fixtures:
    ./build-cpu/bin/gen-dsv4-fixtures --seed 42 --out tests/v4-port/fixtures/

[Streams B/C development]
  test-backend-ops loads .bin files, runs op on CUDA backend,
  compares CUDA output against the dumped CPU reference output.
  No model load required; runs in seconds.

[Validation session]
  Build CUDA:
    cmake -B build-cuda -DGGML_CUDA=ON
    cmake --build build-cuda
  Run per-op:
    ./build-cuda/bin/test-backend-ops -o DSV4_ROPE_TAIL,DSV4_HC_SPLIT_SINKHORN,...
  Run gate suite:
    V4_GGUF=~/models/.../IQ1_S-XL/.../-00001-of-00002.gguf \
    LLAMA_BIN=build-cuda \
    tests/v4-port/run-all-gates.sh
```

CPU is the reference (not Metal) because: (a) bit-deterministic, (b)
reproducible on any dev machine without an Apple device, (c) upstream
maintainers can regenerate fixtures locally for review.

## Error handling

1. **Op unregistered (any backend):** existing ggml convention preserved —
   `ggml_cuda_compute_forward` returns `false`, llama.cpp falls back to CPU,
   warns once. After this port lands, none of the 5 V4 ops trigger this path
   on a CUDA build. We do not modify the fallback path itself.

2. **Compute-capability fallback (FP8 KV):** compile-time `__CUDA_ARCH__`
   dispatch. No runtime check.

3. **Kernel launch error:** `CUDA_CHECK(cudaGetLastError())` after launch,
   propagate via `GGML_ASSERT`. Standard ggml-cuda pattern.

4. **Numerical drift in tests:** existing `test-backend-ops` failure
   semantics — prints `max_err`, `eps`, op name; non-zero exit.

**Explicitly excluded** (to minimize upstream-review surface):

- No new logging subsystem.
- No debug-dump flags in merged code.
- No kernel-side printf in merged code.
- No runtime warnings about backend selection beyond what ggml already logs.

## Testing strategy

Three layers, top-down:

1. **`test-backend-ops` (per-op):** 5 new test classes from Stream A.
   Runs in seconds. No model load. This is the validation upstream maintainers
   inspect first. Must pass on CPU, Metal, and CUDA backends.

2. **Gate suite (architecture-level):** `tests/v4-port/run-all-gates.sh`
   already exists, runs against any backend. With `LLAMA_BIN=build-cuda`,
   the same gates exercise the new CUDA kernels in the context of full
   model inference. Mandatory passes: `gate-loader`, `gate-coherence`,
   `gate-tools`. `gate-speed` records a number but no floor enforced for
   this phase.

3. **End-to-end coding/tool-call session (manual):** One on-hardware
   validation session with all 5 streams merged. Run on RTX 6000 Ada with
   IQ1_S-XL (single-card fit). Real prompt through llama-server,
   verify a multi-turn coding workflow and a tool-call invocation
   both work.

**Hardware fit for validation:**

| Card | Whole-fit quants | Notes |
|---|---|---|
| 1× RTX 6000 Ada (48 GB) | IQ1_S-XL, IQ1_M | Single-card validation; cleanest baseline |
| 2× RTX 6000 Ada (96 GB row split) | + IQ1_M-XL, IQ2_XXS-XL, IQ2_XS-XL | Multi-card stress; optional |
| RTX 5090 (32 GB) | none whole | Partial-offload only; secondary target |

## Upstream-mergeability requirements

This is the constraint shaping all the above. Concrete rules:

- **Compute capability:** kernels compile on SM_70+. FP8 fast path gated behind
  `__CUDA_ARCH__ >= 890`; software path covers everything else.
- **Style:** match existing ggml-cuda conventions. .cu/.cuh pair per op,
  `static __global__` kernels, `ggml_cuda_op_*` dispatch naming, registry
  case in main switch. No new patterns invented.
- **CMakeLists:** minimal — one entry per `.cu` per stream. No new build
  options.
- **Test coverage:** every op has at least one `test-backend-ops` case. The
  FP8 KV op has cases for both the native and the software-emulation paths
  (the test compiles both paths on SM_89+ by forcing the software path via
  a compile-time flag in the test source).
- **Documentation:** each `.cuh` has a file-header comment describing the
  op, the shape it expects, and references to Metal and CPU reference files.
  This is what the upstream PR description quotes from.
- **No dead code, no scratch comments, no TODOs in merged diff.** Streams may
  keep TODOs in dev-only branches that get stripped before the per-stream
  merge.
- **Code review surface:** target ~2,750 LOC across the 6 streams. Each per-op
  stream is independently reviewable in one sitting (≤700 LOC).

## Out of scope (this phase)

- **Performance optimization beyond functional parity.** No TMA, no wgmma,
  no shared-memory micro-tuning beyond what the kernel naturally requires.
- **Multi-GPU sharding tested as primary target.** Single-GPU (1× RTX 6000 Ada)
  is the validation baseline; multi-GPU is exercised opportunistically.
- **Q4/Q8 quant validation on CUDA.** They don't fit in available CUDA memory;
  the test session uses IQ-class quants.
- **FP4 routed-expert support on CUDA.** Routed experts are already FP4 on
  the safetensors source; their CUDA dequant path is existing ggml-cuda work,
  unrelated to the 5 V4-specific ops.
- **Backend-selection UX changes.** Users still use `-ngl N` and standard
  llama.cpp flags. No new CLI surface.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| FP8 e4m3 native intrinsic API differs across CUDA toolkit versions | Medium | Pin tested toolkit version in CMakeLists comment; verify against CUDA 12.4+ which is the upstream baseline |
| Test-window scarcity blocks final integration | Medium | Stream A's fixtures let all streams reach merge-ready without GPU; on-hardware session only blocks the final validation, not development |
| Numerical drift between CPU and CUDA exceeds tolerance for sinkhorn | Low | Sinkhorn is iterative; if tolerances need widening, document why in the test |
| Registry merge conflicts if streams race | Low | Conflict-resolution rule physically isolates per-op edits; merger runs gate-loader after each |
| FP8 software emulation path is slower than expected and bottlenecks IQ-class on Ampere | Low-Medium | Out of scope for "functional parity"; would be a follow-up perf phase |

## What lands when

Stream A merges first into `feat/v4-port-cuda`. Streams B1-B4 and C run in
parallel after A lands, merge in any order. Parent merger runs `gate-loader`
after each per-stream merge. When all 5 op streams are merged and the
`test-backend-ops` cases pass on a CPU build, the on-hardware validation
session is scheduled. After validation passes, `feat/v4-port-cuda` merges
into `feat/v4-port`. After that, the full V4 architecture upstreaming bundle
(V4 ops + Metal + CUDA + CPU reference) becomes the basis for the eventual
upstream PR.
