# V4-port CUDA Stream A: test-backend-ops cases for 5 V4 ops

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan version:** v3 (2026-05-10). v2 revised after codex round-2 plan-review found 3 verified mechanical gaps; v3 addresses each:
1. Every CPU-only test invocation now passes `-b CPU` (per `tests/test-backend-ops.cpp:9572`: CPU backend is skipped by default in test mode unless explicitly selected).
2. Task 8.1 now pins an explicit expected test count and asserts it; otherwise `SKIPPED`/`NOT_SUPPORTED` (per `tests/test-backend-ops.cpp:9310-9326`) would silently report success on 0 tests run.
3. The F16 variant of `test_dsv4_rope_tail` is dropped (Metal kernel at `ggml/src/ggml-metal/ggml-metal-device.m:1226` requires F32 src0; F16 would surface as NOT_SUPPORTED on Metal — silent pass without exercising the kernel).

v1→v2 change log (5 blockers; addressed in v2 and preserved here):
- `-o "DSV4_*"` is not glob (exact comma-separated matching only).
- Sinkhorn `mix_dim` violated `ggml/src/ggml.c:6306` assertion.
- `hc_expand` `block_out` must be 2D per `ggml/src/ggml.c:6363-6366`.
- FP8 KV requires `(ne[0] - n_rot) % 64 == 0` per `ggml/src/ggml.c:6396`.
- `max_nmse_err()` is NMSE, not absolute/relative error.

**Goal:** Add 5 new `test_case` structs to `tests/test-backend-ops.cpp` — one for each V4-specific custom op — so that every backend (CPU, Metal, CUDA) can be validated against the CPU reference via the existing framework. This stream lands first; Streams B/C use these tests to validate their CUDA kernels.

**Architecture:** Each op gets its own `test_*` struct following the `test_rope` pattern at `tests/test-backend-ops.cpp:4645`. `build_graph()` allocates input tensors via `ggml_new_tensor_*`, calls the public `ggml_dsv4_*` constructor (which packs op_params internally — see `ggml/src/ggml.c:6314`), and returns the output node. test-backend-ops handles seeded random inputs and per-backend output comparison automatically. No separate binary-fixture generator — that was an earlier design alternative simplified out in favor of upstream convention. Tests do **not** manually call `ggml_set_op_params*`; the public constructors do that for us.

**Tech Stack:** C++ (test-backend-ops), ggml/llama.cpp build, CMake (only existing test target updated).

**Spec:** `docs/superpowers/specs/2026-05-10-v4-port-cuda-design.md`

**Branch:** `feat/v4-port-cuda-A-fixtures` off `feat/v4-port-cuda` (parent off `feat/v4-port`).

## Critical constraints from the constructors (verified against ggml/src/ggml.c)

These constraints come from `GGML_ASSERT`s in the public constructors. Violating them causes the test to abort during `build_graph()`. Plan v1 violated three of them; v2 fixes them.

- **`ggml_dsv4_hc_split_sinkhorn`** (ggml.c:6306): `mixes->ne[0] == (2 + n_hc) * n_hc`. For `n_hc=4` → `ne[0]=24`. For `n_hc=8` → `ne[0]=80`. Also `mixes->ne[2]==1`, `mixes->ne[3]==1`; `ggml_nelements(scale) >= 3`; `ggml_nelements(base) >= mixes->ne[0]`.
- **`ggml_dsv4_hc_weighted_sum`** (ggml.c:6335-6339): `x->ne[1] == weights->ne[0]`, `x->ne[2] == weights->ne[1]`, `x->ne[3] == 1`, `weights->ne[2] == 1`, `weights->ne[3] == 1`. So `x` is `{n_embd, n_hc, n_tokens, 1}` and `weights` is `{n_hc, n_tokens, 1, 1}`.
- **`ggml_dsv4_hc_expand`** (ggml.c:6363-6374): `block_out` is **2D** `{n_embd, n_tokens, 1, 1}` (NOT 3D); `residual` is 3D `{n_embd, n_hc, n_tokens, 1}`; `post` is `{n_hc, n_tokens, 1, 1}`; `comb` is `{n_hc, n_hc, n_tokens, 1}`. All ne[2]/ne[3] constraints tight; see assertions.
- **`ggml_dsv4_fp8_kv_quantize`** (ggml.c:6395-6396): `a->ne[0] > n_rot` AND `(a->ne[0] - n_rot) % 64 == 0`. So `{128, ...}` requires `n_rot=64`; `{192, ...}` allows `n_rot=64` or `n_rot=128`; `{256, ...}` allows `n_rot=64, 128, or 192`.
- **`ggml_dsv4_rope_tail`** (ggml.c:6425-6438): `mode == GGML_ROPE_TYPE_NORMAL` or `GGML_ROPE_TYPE_NEOX` only; `a->ne[2] == pos->ne[0]`; `n_dims > 0 && n_dims <= a->ne[0] && n_dims % 2 == 0`; if `freq_factors`, `freq_factors->ne[0] >= n_dims/2`.

## Test filter syntax (verified against tests/test-backend-ops.cpp:1251)

`matches_filter` does **exact comma-separated op-name matching**, NOT glob/regex. `-o "DSV4_*"` matches NOTHING and silently passes. The correct invocation is:

```bash
-o DSV4_HC_SPLIT_SINKHORN,DSV4_HC_WEIGHTED_SUM,DSV4_HC_EXPAND,DSV4_FP8_KV_QUANTIZE,DSV4_ROPE_TAIL
```

Every test command in this plan uses the full comma-separated list. A shell variable `V4_FILTER` is defined once and reused.

## Tolerance semantics (verified against tests/test-backend-ops.cpp:1126-1149)

`max_nmse_err()` returns an **NMSE** (normalized mean squared error) threshold, not absolute or relative error. The default `err()` returns `nmse(a, b, n)`. The default `max_nmse_err` is `1e-7`. Tolerances below are NMSE values, justified inline per op. The spec's "1e-4 abs / 1e-3 rel" guidance has been translated to NMSE equivalents.

---

## Task 1: Create branches

**Files:** none (git only)

- [ ] **Step 1.1: Create the cuda port parent branch from feat/v4-port**

```bash
cd ~/work/llama.cpp
git checkout feat/v4-port
git pull --ff-only origin feat/v4-port 2>/dev/null || true
# Only create if it doesn't already exist:
git show-ref --verify --quiet refs/heads/feat/v4-port-cuda || git checkout -b feat/v4-port-cuda
git push -u origin feat/v4-port-cuda 2>/dev/null || true
```

- [ ] **Step 1.2: Create the per-stream branch**

```bash
git checkout feat/v4-port-cuda
git checkout -b feat/v4-port-cuda-A-fixtures
```

- [ ] **Step 1.3: Verify current state**

Run: `git status && git log --oneline -1`
Expected: clean working tree, head at the most recent feat/v4-port commit.

- [ ] **Step 1.4: Define V4_FILTER once for all subsequent invocations**

```bash
export V4_FILTER=DSV4_HC_SPLIT_SINKHORN,DSV4_HC_WEIGHTED_SUM,DSV4_HC_EXPAND,DSV4_FP8_KV_QUANTIZE,DSV4_ROPE_TAIL
```

---

## Task 2: Read existing pattern + identify insertion points

**Files (read-only):**
- `tests/test-backend-ops.cpp:4645-4900` — `test_rope` reference pattern.
- `tests/test-backend-ops.cpp` — search for the main() registration block (where new test ops get added).
- `ggml/include/ggml.h:2563-2613` — public API for all 5 V4 ops.
- `ggml/src/ggml.c:6280-6457` — constructor assertions for each op (authoritative source of shape constraints).
- `ggml/src/ggml-metal/ggml-metal-ops.cpp:1392-1673` — Metal kargs structs (informational only; the public ggml.h constructor is the contract).

- [ ] **Step 2.1: Locate the registration block in main()**

Run: `grep -n "test_rope(" tests/test-backend-ops.cpp | head -20`
Expected: lines around 8800-9200 showing existing `test_rope(...)` calls being pushed into `test_cases`.

- [ ] **Step 2.2: Identify the exact line range to insert new tests near**

Run: `grep -n "// rope\|GGML_OP_ROPE" tests/test-backend-ops.cpp`
Note the location — V4 dsv4 tests will be added after the rope group.

---

## Task 3: test_dsv4_rope_tail struct + registration

**Files:**
- Modify: `tests/test-backend-ops.cpp` (insert struct near other rope tests; insert registration calls in main())

- [ ] **Step 3.1: Add the test_dsv4_rope_tail struct**

Insert this immediately after the `test_rope` struct (i.e., after the closing `};` of `test_rope`):

```cpp
// V4 partial-RoPE: leaves the non-RoPE prefix unchanged, applies RoPE to the tail.
// Reference: ggml/include/ggml.h:2599 (ggml_dsv4_rope_tail).
// Metal kernel:  ggml/src/ggml-metal/ggml-metal.metal:4906-4997.
// CPU fallback:  ggml/src/ggml-cpu/ops.cpp:5961.
// Constraints (ggml.c:6425-6438): mode in {NORMAL, NEOX}; a->ne[2] == pos->ne[0];
// n_dims > 0 && n_dims <= a->ne[0] && n_dims % 2 == 0; if freq_factors,
// freq_factors->ne[0] >= n_dims/2.
struct test_dsv4_rope_tail : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int n_dims;
    int mode;
    int n_ctx;
    float fs;       // freq_scale
    float ef;       // ext_factor
    float af;       // attn_factor
    bool ff;        // use freq_factors
    bool inverse;

    std::string vars() override {
        return VARS_TO_STR10(type, ne_a, n_dims, mode, n_ctx, fs, ef, af, ff, inverse);
    }

    test_dsv4_rope_tail(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {64, 8, 4, 1},
            int n_dims = 32, int mode = GGML_ROPE_TYPE_NORMAL, int n_ctx = 128,
            float fs = 1.0f, float ef = 0.0f, float af = 0.0f,
            bool ff = false, bool inverse = false)
        : type(type), ne_a(ne_a), n_dims(n_dims), mode(mode), n_ctx(n_ctx),
          fs(fs), ef(ef), af(af), ff(ff), inverse(inverse) {}

    // NMSE tolerance: 1e-5. Rationale: RoPE is trig + multiply, no
    // accumulation. Matches test_rope's de-facto behavior on this backend pair.
    double max_nmse_err() override {
        return 1e-5;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        // Constraint: a->ne[2] == pos->ne[0].
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2]);
        ggml_set_name(pos, "pos");

        ggml_tensor * freq = nullptr;
        if (ff) {
            // Constraint: freq_factors->ne[0] >= n_dims/2.
            freq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims / 2);
            ggml_set_name(freq, "freq");
        }

        ggml_tensor * out = ggml_dsv4_rope_tail(
            ctx, a, pos, freq,
            n_dims, mode, n_ctx,
            10000.0f, fs, ef, af, 1.0f, 1.0f,
            inverse);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // Match test_rope's pattern (tests/test-backend-ops.cpp:4752): positions
        // are random within [0, n_ctx) so the test exercises a representative
        // distribution of RoPE phases on every run, not just sequential 0..N-1.
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                std::vector<int> data(ggml_nelements(t));
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = rand() % n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int));
            } else {
                init_tensor_uniform(t, -1.0f, 1.0f);
            }
        }
    }
};
```

- [ ] **Step 3.2: Register the new tests in main()**

Find the line that pushes `test_rope` cases (look for a contiguous block of `test_cases.emplace_back(new test_rope(`). Immediately after that block, add:

```cpp
    // V4-port: dsv4_rope_tail (partial-RoPE) test cases
    for (bool inverse : {false, true}) {
        for (bool ff : {false, true}) {
            // F32, default shape
            test_cases.emplace_back(new test_dsv4_rope_tail(
                GGML_TYPE_F32, {64, 8, 4, 1}, 32, GGML_ROPE_TYPE_NORMAL, 128,
                1.0f, 0.0f, 0.0f, ff, inverse));
        }
    }
    // Edge: larger head_dim, NEOX mode (exercises the second supported mode path).
    test_cases.emplace_back(new test_dsv4_rope_tail(
        GGML_TYPE_F32, {128, 16, 8, 1}, 64, GGML_ROPE_TYPE_NEOX, 256,
        1.0f, 0.0f, 0.0f, false, false));
    // (F16 dtype variant intentionally NOT registered: the Metal kernel
    //  at ggml/src/ggml-metal/ggml-metal-device.m:1226 requires F32 src0,
    //  so an F16 case would surface as NOT_SUPPORTED on Metal — silently
    //  passing without exercising the kernel. F32-only here.)
```

- [ ] **Step 3.3: Build CPU-only**

Run: `cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpu -j --target test-backend-ops 2>&1 | tail -20`
Expected: build succeeds. If compile errors, fix them (most likely a `VARS_TO_STRn` arity mismatch — adjust the N in the macro to match the number of fields).

- [ ] **Step 3.4: Run the new tests**

Run: `./build-cpu/bin/test-backend-ops -b CPU -o DSV4_ROPE_TAIL 2>&1 | tail -30`
Expected: tests execute on the CPU backend, all PASS (because there's only one backend enabled, the comparison is trivially CPU-vs-CPU). The harness validates that the op is callable through the public API.

- [ ] **Step 3.5: Commit**

```bash
git add tests/test-backend-ops.cpp
git commit -m "v4-port-cuda-A: test-backend-ops case for dsv4_rope_tail

Adds test_dsv4_rope_tail following the test_rope pattern. CPU-only run
trivially passes; case becomes useful once Stream B1 registers a CUDA
kernel for the op.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: test_dsv4_hc_split_sinkhorn

**Files:**
- Modify: `tests/test-backend-ops.cpp`

- [ ] **Step 4.1: Add the test_dsv4_hc_split_sinkhorn struct**

Insert immediately after the `test_dsv4_rope_tail` struct:

```cpp
// V4 hyper-connection splitter with Sinkhorn normalization.
// Reference:    ggml/include/ggml.h:2563 (ggml_dsv4_hc_split_sinkhorn).
// Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2076-2245.
// CPU fallback: ggml/src/ggml-cpu/ops.cpp:10990+.
// Constraints (ggml.c:6306-6310): mixes->ne[0] == (2 + n_hc) * n_hc;
// mixes->ne[2] == 1; mixes->ne[3] == 1; nelements(scale) >= 3;
// nelements(base) >= mixes->ne[0].
struct test_dsv4_hc_split_sinkhorn : public test_case {
    const int n_hc;
    const int64_t n_rows;
    const int sinkhorn_iters;
    const float eps;

    std::string vars() override {
        return VARS_TO_STR4(n_hc, n_rows, sinkhorn_iters, eps);
    }

    test_dsv4_hc_split_sinkhorn(int n_hc = 4, int64_t n_rows = 16,
                                int sinkhorn_iters = 4, float eps = 1e-6f)
        : n_hc(n_hc), n_rows(n_rows), sinkhorn_iters(sinkhorn_iters), eps(eps) {}

    // NMSE tolerance: 1e-3. Rationale: 4 iterations of normalization compound
    // floating-point rounding; per-iteration eps division amplifies relative
    // error on near-zero entries. Spec calls for "1e-3 rel"; NMSE 1e-3 is the
    // matching budget.
    double max_nmse_err() override {
        return 1e-3;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // Hard constraint: mixes->ne[0] MUST equal (2 + n_hc) * n_hc.
        const int64_t mix_dim = (int64_t)(2 + n_hc) * n_hc;

        ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mix_dim, n_rows);
        ggml_set_param(mixes);
        ggml_set_name(mixes, "mixes");

        // scale: nelements(scale) >= 3. Constructor uses scale as a 1D
        // parameter buffer. Use a 1D tensor of size 3 (the minimum).
        ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
        ggml_set_param(scale);
        ggml_set_name(scale, "scale");

        // base: nelements(base) >= mixes->ne[0]. Use a 1D tensor of size mix_dim.
        ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mix_dim);
        ggml_set_param(base);
        ggml_set_name(base, "base");

        ggml_tensor * out = ggml_dsv4_hc_split_sinkhorn(ctx, mixes, scale, base, n_hc, sinkhorn_iters, eps);
        ggml_set_name(out, "out");
        return out;
    }
};
```

- [ ] **Step 4.2: Register**

Add to main(), right after the rope_tail registrations:

```cpp
    // V4-port: dsv4_hc_split_sinkhorn test cases.
    // For n_hc=4 → mix_dim = (2+4)*4 = 24.
    // For n_hc=8 → mix_dim = (2+8)*8 = 80.
    test_cases.emplace_back(new test_dsv4_hc_split_sinkhorn(4, 16, 4, 1e-6f));
    test_cases.emplace_back(new test_dsv4_hc_split_sinkhorn(4, 32, 4, 1e-6f));
    test_cases.emplace_back(new test_dsv4_hc_split_sinkhorn(4, 16, 8, 1e-6f));
    test_cases.emplace_back(new test_dsv4_hc_split_sinkhorn(8, 16, 4, 1e-6f));
```

- [ ] **Step 4.3: Build + run**

```bash
cmake --build build-cpu -j --target test-backend-ops 2>&1 | tail -10
./build-cpu/bin/test-backend-ops -b CPU -o DSV4_HC_SPLIT_SINKHORN 2>&1 | tail -30
```
Expected: build succeeds, 4 tests PASS on CPU backend.

- [ ] **Step 4.4: Commit**

```bash
git add tests/test-backend-ops.cpp
git commit -m "v4-port-cuda-A: test-backend-ops case for dsv4_hc_split_sinkhorn

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: test_dsv4_hc_weighted_sum

**Files:**
- Modify: `tests/test-backend-ops.cpp`

- [ ] **Step 5.1: Add the test_dsv4_hc_weighted_sum struct**

Insert after `test_dsv4_hc_split_sinkhorn`:

```cpp
// V4 hyper-connection weighted-sum: out[embd, token] = sum_hc weights[hc, token] * x[embd, hc, token].
// Reference:    ggml/include/ggml.h:2574 (ggml_dsv4_hc_weighted_sum).
// Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2278-2327.
// CPU fallback: ggml/src/ggml-cpu/ops.cpp:11100+.
// Constraints (ggml.c:6335-6339):
//   x       shape {n_embd, n_hc, n_tokens, 1}
//   weights shape {n_hc,   n_tokens, 1, 1}
struct test_dsv4_hc_weighted_sum : public test_case {
    const int64_t n_embd;
    const int64_t n_hc;
    const int64_t n_tokens;

    std::string vars() override {
        return VARS_TO_STR3(n_embd, n_hc, n_tokens);
    }

    test_dsv4_hc_weighted_sum(int64_t n_embd = 128, int64_t n_hc = 4, int64_t n_tokens = 16)
        : n_embd(n_embd), n_hc(n_hc), n_tokens(n_tokens) {}

    // NMSE tolerance: 1e-5. Rationale: weighted sum with n_hc≤16 terms;
    // accumulation error is small; pure F32 multiply-add.
    double max_nmse_err() override {
        return 1e-5;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, n_tokens);
        ggml_set_param(x);
        ggml_set_name(x, "x");

        ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, n_tokens);
        ggml_set_param(weights);
        ggml_set_name(weights, "weights");

        ggml_tensor * out = ggml_dsv4_hc_weighted_sum(ctx, x, weights);
        ggml_set_name(out, "out");
        return out;
    }
};
```

- [ ] **Step 5.2: Register**

```cpp
    // V4-port: dsv4_hc_weighted_sum test cases (n_embd, n_hc, n_tokens).
    test_cases.emplace_back(new test_dsv4_hc_weighted_sum(128, 4, 16));
    test_cases.emplace_back(new test_dsv4_hc_weighted_sum(512, 4, 32));
    test_cases.emplace_back(new test_dsv4_hc_weighted_sum(64,  8, 8));
```

- [ ] **Step 5.3: Build + run**

```bash
cmake --build build-cpu -j --target test-backend-ops 2>&1 | tail -10
./build-cpu/bin/test-backend-ops -b CPU -o DSV4_HC_WEIGHTED_SUM 2>&1 | tail -30
```
Expected: build succeeds, 3 tests PASS on CPU.

- [ ] **Step 5.4: Commit**

```bash
git add tests/test-backend-ops.cpp
git commit -m "v4-port-cuda-A: test-backend-ops case for dsv4_hc_weighted_sum

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: test_dsv4_hc_expand

**Files:**
- Modify: `tests/test-backend-ops.cpp`

- [ ] **Step 6.1: Add the test_dsv4_hc_expand struct**

Insert after `test_dsv4_hc_weighted_sum`:

```cpp
// V4 hyper-connection expand: out[embd, hc, token] = post[hc, token] * block_out[embd, token]
//                                                    + (comb[:, :, token]^T @ residual[:, :, token])[embd, hc].
// Reference:    ggml/include/ggml.h:2581 (ggml_dsv4_hc_expand).
// Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2247-2276.
// CPU fallback: ggml/src/ggml-cpu/ops.cpp:11200+.
// Constraints (ggml.c:6363-6374):
//   block_out shape {n_embd, n_tokens, 1, 1}    (2D, NOT 3D)
//   residual  shape {n_embd, n_hc,    n_tokens, 1}
//   post      shape {n_hc,   n_tokens, 1, 1}
//   comb      shape {n_hc,   n_hc,    n_tokens, 1}
struct test_dsv4_hc_expand : public test_case {
    const int64_t n_embd;
    const int64_t n_hc;
    const int64_t n_tokens;

    std::string vars() override {
        return VARS_TO_STR3(n_embd, n_hc, n_tokens);
    }

    test_dsv4_hc_expand(int64_t n_embd = 128, int64_t n_hc = 4, int64_t n_tokens = 16)
        : n_embd(n_embd), n_hc(n_hc), n_tokens(n_tokens) {}

    // NMSE tolerance: 1e-5. Rationale: one matmul along n_hc (small) plus a
    // pointwise scale; minimal accumulation noise in F32.
    double max_nmse_err() override {
        return 1e-5;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // block_out is 2D: {n_embd, n_tokens}. ne[2]==1, ne[3]==1.
        ggml_tensor * block_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_param(block_out);
        ggml_set_name(block_out, "block_out");

        ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, n_tokens);
        ggml_set_param(residual);
        ggml_set_name(residual, "residual");

        ggml_tensor * post = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, n_tokens);
        ggml_set_param(post);
        ggml_set_name(post, "post");

        ggml_tensor * comb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_hc, n_hc, n_tokens);
        ggml_set_param(comb);
        ggml_set_name(comb, "comb");

        ggml_tensor * out = ggml_dsv4_hc_expand(ctx, block_out, residual, post, comb);
        ggml_set_name(out, "out");
        return out;
    }
};
```

- [ ] **Step 6.2: Register**

```cpp
    // V4-port: dsv4_hc_expand test cases (n_embd, n_hc, n_tokens).
    test_cases.emplace_back(new test_dsv4_hc_expand(128, 4, 16));
    test_cases.emplace_back(new test_dsv4_hc_expand(512, 4, 32));
    test_cases.emplace_back(new test_dsv4_hc_expand(64,  8, 8));
```

- [ ] **Step 6.3: Build + run**

```bash
cmake --build build-cpu -j --target test-backend-ops 2>&1 | tail -10
./build-cpu/bin/test-backend-ops -b CPU -o DSV4_HC_EXPAND 2>&1 | tail -30
```
Expected: build succeeds, 3 tests PASS on CPU.

- [ ] **Step 6.4: Commit**

```bash
git add tests/test-backend-ops.cpp
git commit -m "v4-port-cuda-A: test-backend-ops case for dsv4_hc_expand

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: test_dsv4_fp8_kv_quantize

**Files:**
- Modify: `tests/test-backend-ops.cpp`

- [ ] **Step 7.1: Add the test_dsv4_fp8_kv_quantize struct**

Insert after `test_dsv4_hc_expand`:

```cpp
// V4 FP8 KV-cache simulation: quantizes/dequantizes the non-RoPE prefix
// in E4M3FN blocks, leaves the RoPE tail unchanged.
// Reference:    ggml/include/ggml.h:2591 (ggml_dsv4_fp8_kv_quantize).
// Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2328-2403.
// CPU fallback: ggml/src/ggml-cpu/ops.cpp:11305.
// Constraints (ggml.c:6394-6396): n_rot >= 0; a->ne[0] > n_rot;
// (a->ne[0] - n_rot) % 64 == 0  (block size is 64 for the FP8 prefix).
struct test_dsv4_fp8_kv_quantize : public test_case {
    const std::array<int64_t, 4> ne_a;
    const int n_rot;

    std::string vars() override {
        return VARS_TO_STR2(ne_a, n_rot);
    }

    test_dsv4_fp8_kv_quantize(std::array<int64_t, 4> ne_a = {192, 8, 4, 1},
                              int n_rot = 64)
        : ne_a(ne_a), n_rot(n_rot) {}

    // NMSE tolerance: 1e-3. Rationale: FP8 e4m3 represents ~7 bits of mantissa;
    // the quantize-dequantize round-trip's NMSE is dominated by representable
    // precision, not by accumulation. The spec's "1e-3 abs (FP8 inherently
    // lossy)" maps to NMSE 1e-3 because each sample's squared error is bounded
    // by the FP8 ULP² at the local scale, normalized by signal power yields
    // roughly the same order.
    double max_nmse_err() override {
        return 1e-3;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // Constraint check at construction time so test fails fast on a bad shape.
        GGML_ASSERT(ne_a[0] > n_rot && "(ne_a[0] > n_rot) required");
        GGML_ASSERT((ne_a[0] - n_rot) % 64 == 0 && "(ne_a[0]-n_rot) %% 64 == 0 required");

        ggml_tensor * a = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_a.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_dsv4_fp8_kv_quantize(ctx, a, n_rot);
        ggml_set_name(out, "out");
        return out;
    }
};
```

- [ ] **Step 7.2: Register**

```cpp
    // V4-port: dsv4_fp8_kv_quantize test cases.
    // Constraint: (ne_a[0] - n_rot) % 64 == 0. Valid examples:
    //   ne_a[0]=128, n_rot=64   → prefix=64  (1 block)
    //   ne_a[0]=192, n_rot=64   → prefix=128 (2 blocks)
    //   ne_a[0]=256, n_rot=64   → prefix=192 (3 blocks)
    //   ne_a[0]=192, n_rot=128  → prefix=64  (1 block)
    test_cases.emplace_back(new test_dsv4_fp8_kv_quantize({128, 8, 4, 1}, 64));
    test_cases.emplace_back(new test_dsv4_fp8_kv_quantize({192, 8, 4, 1}, 64));
    test_cases.emplace_back(new test_dsv4_fp8_kv_quantize({256, 16, 8, 1}, 64));
    test_cases.emplace_back(new test_dsv4_fp8_kv_quantize({192, 16, 8, 1}, 128));
```

- [ ] **Step 7.3: Build + run**

```bash
cmake --build build-cpu -j --target test-backend-ops 2>&1 | tail -10
./build-cpu/bin/test-backend-ops -b CPU -o DSV4_FP8_KV_QUANTIZE 2>&1 | tail -30
```
Expected: build succeeds, 4 tests PASS on CPU.

- [ ] **Step 7.4: Commit**

```bash
git add tests/test-backend-ops.cpp
git commit -m "v4-port-cuda-A: test-backend-ops case for dsv4_fp8_kv_quantize

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Run full V4 suite + verify all 5 ops registered

**Files:** none (validation only)

- [ ] **Step 8.1: Run all 5 dsv4 tests in one invocation, assert non-zero count**

The harness's `-o` flag does exact comma-separated matching, not glob. The harness also treats `SKIPPED` and `NOT_SUPPORTED` as non-failures, and reports success on `0/0` — so a filter that matches no tests silently passes. Pin the expected count explicitly.

Expected count: **19 tests** (5 rope_tail [4 from the inverse×ff loop + 1 NEOX edge] + 4 sinkhorn + 3 weighted_sum + 3 expand + 4 fp8_kv). If the printed total deviates, the registration block in main() is wrong — fix and re-run.

```bash
./build-cpu/bin/test-backend-ops -b CPU -o DSV4_HC_SPLIT_SINKHORN,DSV4_HC_WEIGHTED_SUM,DSV4_HC_EXPAND,DSV4_FP8_KV_QUANTIZE,DSV4_ROPE_TAIL 2>&1 | tee /tmp/v4-cuda-A-task8-1.log | tail -60

# Assert non-zero AND matches expected
COUNT=$(grep -E "^\s+[0-9]+/[0-9]+ tests passed" /tmp/v4-cuda-A-task8-1.log | tail -1 | grep -oE "^\s+[0-9]+" | tr -d ' ')
echo "Tests passed: ${COUNT:-0}"
test "${COUNT:-0}" -ge 19 || { echo "FAIL: only ${COUNT:-0} of 19 expected tests ran (SKIPPED-counts-as-pass would mask this)"; exit 1; }
```

Expected: `Tests passed: 19` (or more if extra variants got added during implementation). If 0 or under, the harness silently no-op'd — the filter syntax or backend selection is wrong.

- [ ] **Step 8.2: Run full test-backend-ops to ensure no regressions**

Run: `./build-cpu/bin/test-backend-ops -b CPU 2>&1 | tail -20`
Expected: existing tests still pass; new dsv4 tests added to the count.

- [ ] **Step 8.3: Push the stream branch**

```bash
git push -u origin feat/v4-port-cuda-A-fixtures
```

- [ ] **Step 8.4: Final stream commit (summary)**

If any cleanup needed, commit it. Otherwise this task is just verification.

---

## Task 9: Merge into the parent feat/v4-port-cuda branch

**Files:** none (git only)

- [ ] **Step 9.1: Fast-forward merge**

```bash
git checkout feat/v4-port-cuda
git merge --ff-only feat/v4-port-cuda-A-fixtures
```
Expected: fast-forward merge succeeds (no other streams have landed yet).

- [ ] **Step 9.2: Run gate-loader as the merge-discipline check**

Build a CUDA test if hardware available; otherwise CPU-only build is enough for Stream A since no CUDA kernels were added.

Run: `cmake --build build-cpu --target llama-cli 2>&1 | tail -5`
Expected: builds successfully.

- [ ] **Step 9.3: Push parent**

```bash
git push origin feat/v4-port-cuda
```

---

## Definition of done (Stream A)

- 5 new test_case structs in `tests/test-backend-ops.cpp`, one per V4 op.
- All 5 test classes have a `max_nmse_err()` override with a documented NMSE-based tolerance rationale in a comment immediately above the override.
- Registration entries added in main() with at least 2 size/shape variants per op; shapes obey the constructor assertions documented at the top of this plan.
- `./build-cpu/bin/test-backend-ops -b CPU -o DSV4_HC_SPLIT_SINKHORN,DSV4_HC_WEIGHTED_SUM,DSV4_HC_EXPAND,DSV4_FP8_KV_QUANTIZE,DSV4_ROPE_TAIL` passes on a CPU-only build (NOT `-o "DSV4_*"` — wildcard does not work).
- Branch `feat/v4-port-cuda-A-fixtures` pushed; merged into `feat/v4-port-cuda`.
- No new files created; only `tests/test-backend-ops.cpp` modified.

## What's deferred to other streams

- CUDA kernel implementations (Streams B1-B4, C).
- Native FP8 path for fp8_kv_quantize (Stream C).
- Registration in `ggml/src/ggml-cuda/ggml-cuda.cu` (each B/C stream adds its own `case` block).
- Multi-card / partial-offload validation runs (final on-hardware validation session, after all streams merged).
- Tolerance re-tuning once CUDA backends are present and the CPU-vs-CUDA NMSE values are empirically measured. Initial tolerances above are documented best-estimates; Stream B/C may submit per-op tolerance adjustments as part of their own PRs if measurements warrant.

## Change log

- **v2 (2026-05-10):** Address codex round-1 plan-review:
  - Fix sinkhorn shape (`mix_dim = (2+n_hc)*n_hc`).
  - Fix expand `block_out` shape (2D, not 3D).
  - Fix fp8_kv shape constraint examples (`n_rot=64`, prefix multiple of 64).
  - Replace all `-o "DSV4_*"` wildcard invocations with comma-separated exact list.
  - Reframe tolerances as NMSE with per-op rationale.
  - Add randomized position init for rope_tail to match test_rope's pattern.
  - Remove misleading "sets op_params" language from architecture section — constructors handle that internally.
  - Add Critical Constraints section citing ggml.c line numbers as authoritative source.
- **v1 (2026-05-10):** Initial plan from brainstorming + writing-plans skills.
