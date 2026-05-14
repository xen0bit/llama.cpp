# Plan: v4-rowsplit-fix

Unblock `--split-mode row` (tensor parallel) for DeepSeek V4 on multi-GPU CUDA by
extending the cuda_split exemption in `ggml_backend_cuda_device_supports_op` to
allow metadata-only ops on cuda_split-buffer sources.

## Problem recap

`ggml-backend.cpp:908` aborts at model load:

```
pre-allocated tensor (blk.0.attn_output_a.weight (reshaped))
in a buffer (CUDA0_Split)
that cannot run the operation (RESHAPE)
```

- V4's `dsv4_grouped_out` (src/models/deepseek4.cpp:567-595) does
  `ggml_reshape_3d(wo_a, ...)` to prep the weight for `ggml_mul_mat_id`.
- Under `-sm row`, `attn_output_a.weight` lives in a `CUDA*_Split` buffer.
- `ggml_backend_cuda_device_supports_op` in `ggml-cuda.cu:5126` rejects any
  non-MUL_MAT / non-DSV4-custom op whose `src[i]` is in a cuda_split buffer.
- Pre-allocated → sched can't relocate → abort.

## (a) Initial ops to add to the exemption

Four ops, all confirmed metadata-only on CUDA (ggml-cuda.cu:3050-3055,
5429-5443 — the dispatch is a fall-through `break;`; supports_op returns true
unconditionally; no kernel is launched).

- `GGML_OP_RESHAPE`
- `GGML_OP_VIEW`
- `GGML_OP_PERMUTE`
- `GGML_OP_TRANSPOSE`

These ops reinterpret shape/stride of the underlying storage without touching
data — safe to allow on a cuda_split source because the split-buffer's
per-device slices remain untouched. Downstream consumers (the reshaped tensor
inherits the cuda_split buffer) will themselves be queried via supports_op and
must independently be cuda_split-compatible. Today, that set is `MUL_MAT` plus
the DSV4 custom ops we already exempt. `MUL_MAT_ID` is **not** in that set —
its CUDA kernel asserts non-split at ggml-cuda.cu:2635, so a reshape whose
downstream consumer is `mul_mat_id` will still abort (see section (b)).

Explicitly **not** added:

- `GGML_OP_CONT` — real memcpy kernel (ggml-cuda.cu:5466-5467 returns true via
  the regular path, not the metadata fall-through). Adding it would launch a
  copy kernel against a split buffer and crash.
- `GGML_OP_CPY`, `GGML_OP_DUP`, arithmetic ops — all run kernels.

## (b) Iteration strategy

After RESHAPE/VIEW/PERMUTE/TRANSPOSE are allowed, the very next blocker is
expected to be `GGML_OP_MUL_MAT_ID` at deepseek4.cpp:590
(`ggml_mul_mat_id(ctx, wo_a_g, o, ids)`):

- `wo_a_g` is the reshaped view of `wo_a`, which inherits the cuda_split
  buffer.
- `ggml_cuda_mul_mat_id` asserts at ggml-cuda.cu:2635:
  `GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft) &&
   "mul_mat_id does not support split buffers")`.
- That op is also not in the cuda_split exemption — supports_op will return
  false, and since `wo_a_g`'s storage is pre-allocated split, sched cannot
  relocate it, so we get the same abort class.

This is a **kernel-level** issue, not a metadata gate. Per the spec, we STOP
and report when we hit a real-kernel op on cuda_split. We will not add
MUL_MAT_ID to the exemption — that would just move the crash from
ggml-backend.cpp to the kernel's own assert.

Iteration loop (bounded to 3):

1. Build on gpudual.
2. Launch llama-server with `-sm row`.
3. If it aborts:
   - If the new op is metadata-only on CUDA (dispatch is a bare `break;`):
     add to the exemption and rebuild.
   - If the new op has a real kernel (MUL_MAT_ID, GET_ROWS, CONT, CPY,
     CONCAT, SET, SET_ROWS, REPEAT, CAST, ADD, MUL, SUM_ROWS, RMS_NORM, ...):
     STOP, set state="blocked-on-kernel-work", report details.
4. Cap at 3 iterations.

## (c) Stopping conditions

STOP and report (state="blocked-on-kernel-work") if:

- The next blocking op has a real CUDA kernel (e.g. `MUL_MAT_ID`, `CONT`,
  `CPY`, `GET_ROWS`, `ADD`, `MUL`, anything with a kernel dispatch in the
  ggml-cuda.cu switch other than a bare `break;`).
- The fix would require modifying a kernel to be split-buffer-aware (e.g.
  teaching `MUL_MAT_ID` to handle row-split weights). That's an independent
  design task, not in scope here.
- 3 iterations of metadata-op additions still don't unblock end-to-end
  inference.

Note: per codex review (plan-review round 1), `MUL_MAT_ID` is expected to be
the first real-kernel blocker after the metadata gate is opened, because
`dsv4_grouped_out` immediately consumes the reshaped `wo_a_g` (still on
cuda_split) via `ggml_mul_mat_id`. Adding split-buffer support to the
mul_mat_id kernel is a separate task.

## (d) Validation

Local (Mac, Metal — host-side compilation of ggml-cuda.cu is guarded out, so
this is mainly a sanity check that surrounding files still build):

```bash
cmake --build build --target llama-server -j
```

Remote (gpudual):

```bash
ssh gpudual 'cd ~/work/llama.cpp && git fetch origin && \
  git checkout fix/v4-rowsplit && \
  git reset --hard origin/fix/v4-rowsplit && \
  PATH=/usr/local/cuda-12.8/bin:$PATH \
  cmake --build build-cuda -j --target llama-server test-backend-ops'
```

Validation gates:

1. **Build clean** on gpudual (must pass).
2. **DSV4 ops regression**: `test-backend-ops -o DSV4_*` returns 19/19 (must
   pass — proves we didn't break the layer-split path).
3. **Row-split launch**: `llama-server --split-mode row` is launched.

   3a. **Best case** — server boots, chat-completion returns coherent text:
       record decode t/s, compare to 19.4 t/s layer-split baseline, open PR.

   3b. **Expected case** — server aborts at `MUL_MAT_ID` (or another
       real-kernel op) on a cuda_split source: this is the
       `blocked-on-kernel-work` outcome described in section (c). Capture
       the abort message verbatim, stop, report back. Gate 3 is **not a
       pass-fail criterion for the PR** — the PR's value is unblocking the
       metadata-op gate and exposing the next real blocker. Gates 1 and 2
       are the regression gates that must pass.

Success metric (best case): gen t/s vs layer-split baseline of **19.4 t/s**
on 2x RTX 6000 Ada with V4 Flash IQ2_XS-XL. Target: meaningfully faster.

All invocations capped via `perl -e "alarm <secs>; exec @ARGV"`.

## Rollout

1. Branch `fix/v4-rowsplit` off `feat/v4-port-cuda`.
2. Edit ggml/src/ggml-cuda/ggml-cuda.cu lines 5134-5145, add the four ops.
3. Commit, push to `mine`, validate on gpudual.
4. If validation green: open PR against `feat/v4-port-cuda`. Do not merge.
5. If validation hits new metadata op: extend exemption, repeat.
6. If validation hits kernel op: stop, report.
