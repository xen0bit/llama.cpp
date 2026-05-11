# V4-port CUDA Stream A: test-backend-ops cases for 5 V4 ops

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 new `test_case` structs to `tests/test-backend-ops.cpp` — one for each V4-specific custom op — so that every backend (CPU, Metal, CUDA) can be validated against the CPU reference via the existing framework. This stream lands first; Streams B/C use these tests to validate their CUDA kernels.

**Architecture:** Each op gets its own `test_*` struct following the `test_rope` pattern at `tests/test-backend-ops.cpp:4645`. `build_graph()` allocates input tensors, sets op_params where the op needs them, calls the `ggml_dsv4_*` constructor, returns the output node. test-backend-ops handles seeded random inputs and per-backend output comparison automatically. No separate binary-fixture generator — that was an earlier design alternative simplified out in favor of upstream convention.

**Tech Stack:** C++ (test-backend-ops), ggml/llama.cpp build, CMake (only existing test target updated).

**Spec:** `docs/superpowers/specs/2026-05-10-v4-port-cuda-design.md`

**Branch:** `feat/v4-port-cuda-A-fixtures` off `feat/v4-port-cuda` (parent off `feat/v4-port`).

---

## Task 1: Create branches

**Files:** none (git only)

- [ ] **Step 1.1: Create the cuda port parent branch from feat/v4-port**

```bash
cd ~/work/llama.cpp
git checkout feat/v4-port
git pull --ff-only origin feat/v4-port 2>/dev/null || true
git checkout -b feat/v4-port-cuda
git push -u origin feat/v4-port-cuda 2>/dev/null || true
```

- [ ] **Step 1.2: Create the per-stream branch**

```bash
git checkout -b feat/v4-port-cuda-A-fixtures
```

- [ ] **Step 1.3: Verify current state**

Run: `git status && git log --oneline -1`
Expected: clean working tree, head at the most recent feat/v4-port commit.

---

## Task 2: Read existing pattern + identify insertion points

**Files (read-only):**
- `tests/test-backend-ops.cpp:4645-4900` — `test_rope` reference pattern.
- `tests/test-backend-ops.cpp` — search for the main() registration block (where new test ops get added).
- `ggml/include/ggml.h:2563-2613` — public API for all 5 V4 ops.
- `ggml/src/ggml-metal/ggml-metal-ops.cpp:1392-1673` — op_params extraction patterns per op (kargs structs show what op_params each op reads).

- [ ] **Step 2.1: Locate the registration block in main()**

Run: `grep -n "test_rope(" tests/test-backend-ops.cpp | head -20`
Expected: lines around 8800-9200 (depending on file size) showing existing `test_rope(...)` calls being pushed into the test vector.

- [ ] **Step 2.2: Identify the exact line range to insert new tests near**

Run: `grep -n "// rope" tests/test-backend-ops.cpp`
Note the location — V4 dsv4 tests will be added in the same registration section, alphabetically after the rope group.

---

## Task 3: test_dsv4_rope_tail struct + registration

**Files:**
- Modify: `tests/test-backend-ops.cpp` (insert struct near other rope tests; insert registration calls in main())

- [ ] **Step 3.1: Add the test_dsv4_rope_tail struct**

Insert this immediately after the `test_rope` struct (i.e., after the closing `};` of `test_rope`):

```cpp
// V4 partial-RoPE: leaves the non-RoPE prefix unchanged, applies RoPE to the tail.
// Reference: ggml/include/ggml.h:2599 (ggml_dsv4_rope_tail).
// Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:4906-4997.
// CPU fallback: ggml/src/ggml-cpu/ops.cpp:5961.
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

    double max_nmse_err() override {
        return 1e-4;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2]);
        ggml_set_name(pos, "pos");

        ggml_tensor * freq = nullptr;
        if (ff) {
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
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // position tensor: fill with sequential positions
                std::vector<int> pos(ggml_nelements(t));
                for (size_t i = 0; i < pos.size(); ++i) pos[i] = i;
                ggml_backend_tensor_set(t, pos.data(), 0, pos.size() * sizeof(int));
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
            test_cases.emplace_back(new test_dsv4_rope_tail(
                GGML_TYPE_F32, {64, 8, 4, 1}, 32, GGML_ROPE_TYPE_NORMAL, 128,
                1.0f, 0.0f, 0.0f, ff, inverse));
        }
    }
```

- [ ] **Step 3.3: Build CPU-only**

Run: `cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpu -j --target test-backend-ops 2>&1 | tail -20`
Expected: build succeeds. If compile errors, fix them (most likely a `VARS_TO_STRn` arity mismatch — adjust the N in the macro to match the number of fields).

- [ ] **Step 3.4: Run the new tests**

Run: `./build-cpu/bin/test-backend-ops -o DSV4_ROPE_TAIL 2>&1 | tail -30`
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
// Reference: ggml/include/ggml.h:2563 (ggml_dsv4_hc_split_sinkhorn).
// Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2076-2245.
// CPU fallback: ggml/src/ggml-cpu/ops.cpp:10990+.
struct test_dsv4_hc_split_sinkhorn : public test_case {
    const std::array<int64_t, 2> ne_mixes;   // {mix_hc, n_rows}
    const int n_hc;
    const int sinkhorn_iters;
    const float eps;

    std::string vars() override {
        return VARS_TO_STR4(ne_mixes, n_hc, sinkhorn_iters, eps);
    }

    test_dsv4_hc_split_sinkhorn(std::array<int64_t, 2> ne_mixes = {8, 16},
                                int n_hc = 4, int sinkhorn_iters = 4,
                                float eps = 1e-6f)
        : ne_mixes(ne_mixes), n_hc(n_hc), sinkhorn_iters(sinkhorn_iters), eps(eps) {}

    double max_nmse_err() override {
        return 1e-3;  // iterative; rounding compounds over 4 iterations
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_mixes[0], ne_mixes[1]);
        ggml_set_param(mixes);
        ggml_set_name(mixes, "mixes");

        ggml_tensor * scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, ne_mixes[1]);
        ggml_set_param(scale);
        ggml_set_name(scale, "scale");

        ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, ne_mixes[1]);
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
    // V4-port: dsv4_hc_split_sinkhorn test cases
    test_cases.emplace_back(new test_dsv4_hc_split_sinkhorn({8, 16}, 4, 4, 1e-6f));
    test_cases.emplace_back(new test_dsv4_hc_split_sinkhorn({16, 32}, 4, 4, 1e-6f));
    test_cases.emplace_back(new test_dsv4_hc_split_sinkhorn({8, 16}, 4, 8, 1e-6f));
```

- [ ] **Step 4.3: Build + run**

```bash
cmake --build build-cpu -j --target test-backend-ops 2>&1 | tail -10
./build-cpu/bin/test-backend-ops -o DSV4_HC_SPLIT_SINKHORN 2>&1 | tail -30
```
Expected: build succeeds, 3 tests PASS on CPU backend.

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
// V4 hyper-connection weighted-sum: sum_hc weights[hc, token] * x[embd, hc, token].
// Reference: ggml/include/ggml.h:2574 (ggml_dsv4_hc_weighted_sum).
// Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2278-2327.
// CPU fallback: ggml/src/ggml-cpu/ops.cpp:11100+.
struct test_dsv4_hc_weighted_sum : public test_case {
    const std::array<int64_t, 3> ne_x;   // {n_embd, n_hc, n_tokens}

    std::string vars() override {
        return VARS_TO_STR1(ne_x);
    }

    test_dsv4_hc_weighted_sum(std::array<int64_t, 3> ne_x = {128, 4, 16})
        : ne_x(ne_x) {}

    double max_nmse_err() override {
        return 1e-4;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne_x[0], ne_x[1], ne_x[2]);
        ggml_set_param(x);
        ggml_set_name(x, "x");

        ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_x[1], ne_x[2]);
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
    // V4-port: dsv4_hc_weighted_sum test cases
    test_cases.emplace_back(new test_dsv4_hc_weighted_sum({128, 4, 16}));
    test_cases.emplace_back(new test_dsv4_hc_weighted_sum({512, 4, 32}));
    test_cases.emplace_back(new test_dsv4_hc_weighted_sum({64, 8, 8}));
```

- [ ] **Step 5.3: Build + run**

```bash
cmake --build build-cpu -j --target test-backend-ops 2>&1 | tail -10
./build-cpu/bin/test-backend-ops -o DSV4_HC_WEIGHTED_SUM 2>&1 | tail -30
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
// V4 hyper-connection expand: post * block_out + comb^T @ residual per token.
// Reference: ggml/include/ggml.h:2581 (ggml_dsv4_hc_expand).
// Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2247-2276.
// CPU fallback: ggml/src/ggml-cpu/ops.cpp:11200+.
struct test_dsv4_hc_expand : public test_case {
    const std::array<int64_t, 3> ne_block;   // {n_embd, n_hc, n_tokens}

    std::string vars() override {
        return VARS_TO_STR1(ne_block);
    }

    test_dsv4_hc_expand(std::array<int64_t, 3> ne_block = {128, 4, 16})
        : ne_block(ne_block) {}

    double max_nmse_err() override {
        return 1e-4;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_embd = ne_block[0];
        const int64_t n_hc   = ne_block[1];
        const int64_t n_tok  = ne_block[2];

        ggml_tensor * block_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, n_tok);
        ggml_set_param(block_out);
        ggml_set_name(block_out, "block_out");

        ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, n_tok);
        ggml_set_param(residual);
        ggml_set_name(residual, "residual");

        ggml_tensor * post = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, n_tok);
        ggml_set_param(post);
        ggml_set_name(post, "post");

        ggml_tensor * comb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_hc, n_hc, n_tok);
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
    // V4-port: dsv4_hc_expand test cases
    test_cases.emplace_back(new test_dsv4_hc_expand({128, 4, 16}));
    test_cases.emplace_back(new test_dsv4_hc_expand({512, 4, 32}));
```

- [ ] **Step 6.3: Build + run**

```bash
cmake --build build-cpu -j --target test-backend-ops 2>&1 | tail -10
./build-cpu/bin/test-backend-ops -o DSV4_HC_EXPAND 2>&1 | tail -30
```
Expected: build succeeds, 2 tests PASS on CPU.

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
// Reference: ggml/include/ggml.h:2591 (ggml_dsv4_fp8_kv_quantize).
// Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2328-2403.
// CPU fallback: ggml/src/ggml-cpu/ops.cpp:11305.
struct test_dsv4_fp8_kv_quantize : public test_case {
    const std::array<int64_t, 4> ne_a;
    const int n_rot;

    std::string vars() override {
        return VARS_TO_STR2(ne_a, n_rot);
    }

    test_dsv4_fp8_kv_quantize(std::array<int64_t, 4> ne_a = {128, 8, 4, 1},
                              int n_rot = 32)
        : ne_a(ne_a), n_rot(n_rot) {}

    double max_nmse_err() override {
        return 1e-2;  // FP8 e4m3 is lossy by design; tolerance reflects FP8 representable precision
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
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
    // V4-port: dsv4_fp8_kv_quantize test cases
    test_cases.emplace_back(new test_dsv4_fp8_kv_quantize({128, 8, 4, 1}, 32));
    test_cases.emplace_back(new test_dsv4_fp8_kv_quantize({256, 16, 8, 1}, 64));
```

- [ ] **Step 7.3: Build + run**

```bash
cmake --build build-cpu -j --target test-backend-ops 2>&1 | tail -10
./build-cpu/bin/test-backend-ops -o DSV4_FP8_KV_QUANTIZE 2>&1 | tail -30
```
Expected: build succeeds, 2 tests PASS on CPU.

- [ ] **Step 7.4: Commit**

```bash
git add tests/test-backend-ops.cpp
git commit -m "v4-port-cuda-A: test-backend-ops case for dsv4_fp8_kv_quantize

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Run full V4 suite + verify all 5 ops registered

**Files:** none (validation only)

- [ ] **Step 8.1: Run all 5 dsv4 tests in one invocation**

Run: `./build-cpu/bin/test-backend-ops -o "DSV4_*" 2>&1 | tail -50`
Expected: all dsv4 cases PASS on CPU backend. Count: ~14 tests (4 rope_tail + 3 sinkhorn + 3 weighted_sum + 2 expand + 2 fp8_kv).

- [ ] **Step 8.2: Run full test-backend-ops to ensure no regressions**

Run: `./build-cpu/bin/test-backend-ops 2>&1 | tail -20`
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
- All 5 test classes have a `max_nmse_err()` override with documented tolerance.
- Registration entries added in main() with at least 2 size variants per op.
- `./build-cpu/bin/test-backend-ops -o "DSV4_*"` passes on a CPU-only build.
- Branch `feat/v4-port-cuda-A-fixtures` pushed; merged into `feat/v4-port-cuda`.
- No new files created; only `tests/test-backend-ops.cpp` modified.

## What's deferred to other streams

- CUDA kernel implementations (Streams B1-B4, C).
- Native FP8 path for fp8_kv_quantize (Stream C).
- Registration in `ggml/src/ggml-cuda/ggml-cuda.cu` (each B/C stream adds its own `case` block).
- Multi-card / partial-offload validation runs (final on-hardware validation session, after all streams merged).
