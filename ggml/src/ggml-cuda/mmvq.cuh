#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// Standard MUL_MAT / MUL_MAT_ID entry point.
//
// expert_low/expert_high: when non-zero (expert_high > 0), enables the V4
// row-split MUL_MAT_ID fast path. The kernel treats `src0` as a local slab
// covering experts [expert_low, expert_high) of the original split tensor,
// and skips ids cells outside that range so per-device dst temps can be
// sum-aggregated on the main device. Default (0, 0) preserves all existing
// behavior (no split).
void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr,
    uint32_t expert_low = 0, uint32_t expert_high = 0);

// [TAG_MMVQ_SPLIT_MMID] V4 row-split MUL_MAT_ID fast path entry. The caller
// (ggml_cuda_mul_mat_id_split_fast) provides:
//   - stream/pool on the device that owns this expert range
//   - dst_override: per-device dst temp (full dst shape) on that device
//   - expert_low/expert_high: this device's whole-expert range in the root
//     split tensor's expert-index space
// src0 is the device-local slab (data = src0_extra->data_device[id]) with
// ne[2] = expert_high - expert_low.
void ggml_cuda_mul_mat_vec_q_split(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    cudaStream_t stream, ggml_cuda_pool & pool, void * dst_override,
    uint32_t expert_low, uint32_t expert_high);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
