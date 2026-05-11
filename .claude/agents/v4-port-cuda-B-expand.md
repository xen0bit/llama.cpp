# v4-port-cuda-B-expand: CUDA kernel for ggml_dsv4_hc_expand

## Goal

Implement a CUDA kernel + dispatch for `ggml_dsv4_hc_expand`. The `test_dsv4_hc_expand` cases (3 cases registered by Stream A) must pass with CUDA backend matching CPU within `max_nmse_err = 1e-4`.

Success criterion: `./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_HC_EXPAND` PASSES with `${COUNT:-0}` >= 3.

## Plan reference

**Implementation plan at `docs/superpowers/plans/2026-05-10-v4-port-cuda-B-expand.md`. Follow it task-by-task.**

Plan incorporates the 2 mechanical fixes Stream A surfaced. Standard dev-team pipeline applies.

## Spec reference

`docs/superpowers/specs/2026-05-10-v4-port-cuda-design.md`, "Stream B4: hc_expand".

**Note on shape constraints (verified via Stream A's R1 findings):** `block_out` is a 2D tensor `{n_embd, n_tokens, 1, 1}` (NOT 3D), per `ggml/src/ggml.c:6363-6366`. `residual` is 3D `{n_embd, n_hc, n_tokens, 1}`; `post` is `{n_hc, n_tokens, 1, 1}`; `comb` is `{n_hc, n_hc, n_tokens, 1}`. Plan v2 (post plan-review R1) corrects the `block_out` indexing to 2D `(i_embd, i_tok)` and drops the bogus `nb_b2` stride; verify the kernel reads `block_out + i_embd*nb_b0 + i_tok*nb_b1` only.

## Phased plan summary

1. **Branch setup** — `feat/v4-port-cuda-B-expand` off `feat/v4-port-cuda`.
2. **Read references** — Metal kernel at `ggml/src/ggml-metal/ggml-metal.metal:2247-2276`, CPU reference at `ggml/src/ggml-cpu/ops.cpp:11200+`.
3. **Create `.cuh` + `.cu`** — kernel computes `out[i, hc, tok] = post[hc, tok] * block_out[i, tok] + sum_{hc'} comb[hc, hc', tok] * residual[i, hc', tok]`. One thread per output element; each thread does n_hc-wide accumulation for the comb·residual term.
4. **Register in ggml-cuda.cu** — `case GGML_OP_DSV4_HC_EXPAND` + supports_op.
5. **Validate** — count assertion >= 3.
6. **Push + merge** — fast-forward into `feat/v4-port-cuda`.

## Reference sources

- Plan: `docs/superpowers/plans/2026-05-10-v4-port-cuda-B-expand.md`
- Metal kernel: `ggml/src/ggml-metal/ggml-metal.metal:2247-2276`
- Metal dispatch: `ggml/src/ggml-metal/ggml-metal-ops.cpp:1488-1548`
- CPU reference: `ggml/src/ggml-cpu/ops.cpp:11200+`
- Public API: `ggml/include/ggml.h:2581-2586`
- Constructor shape constraints: `ggml/src/ggml.c:6363-6374`

## Gate (must pass before code-review)

```bash
cmake --build build-cuda -j --target test-backend-ops
./build-cuda/bin/test-backend-ops -b CPU,CUDA -o DSV4_HC_EXPAND 2>&1 | tee /tmp/v4-cuda-B-expand-test.log | tail -30
COUNT=$(grep -E "^\s+[0-9]+/[0-9]+ tests passed" /tmp/v4-cuda-B-expand-test.log | tail -1 | grep -oE "^\s+[0-9]+" | tr -d ' ')
test "${COUNT:-0}" -ge 3 || { echo "FAIL: ${COUNT:-0} of 3"; exit 1; }
```

## Test scope

CUDA kernel only.

## What this task explicitly does NOT do

- Modify `tests/test-backend-ops.cpp`.
- Modify other streams' files.
- Tensorize the comb·residual matmul — keep the simple loop for clarity.

## Iteration budget

- Plan review: 2 codex rounds max.
- Code review: 2 codex rounds max.
- Total fix attempts: 3.

## Upstream-mergeability note

Match ggml-cuda conventions.
