#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// MoE routed-expert selection profiler. No-op unless LLAMA_EXPERT_PROFILE is set
// in the environment. Used to measure expert hotness/skew so we can decide
// whether an explicit hotness-aware expert cache beats the kernel page-cache LRU
// (see perf/TIER2-DESIGN.md). Call from a single thread (ith == 0).
//
// tensor_name : src0 name of the mul_mat_id, e.g. "blk.5.ffn_gate_exps.weight"
//               (only "*ffn_gate_exps*" tensors are counted, so gate/up/down
//               sharing the same routing are not triple-counted).
// n_expert    : total experts for this tensor (src0->ne[2]).
// ids_*       : the selected-expert index tensor (I32): ne0 = experts/token,
//               ne1 = tokens, nb0/nb1 = element/row byte strides.
void ggml_expert_profile_record(const char * tensor_name, int n_expert,
                                const void * ids_data,
                                int64_t ne0, int64_t ne1,
                                size_t nb0, size_t nb1);

#ifdef __cplusplus
}
#endif
