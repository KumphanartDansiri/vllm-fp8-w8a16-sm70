// SPDX-License-Identifier: Apache-2.0
// Our thin torch binding for the vendored TurboMind SM70 FP8 engine.
// Dense (prepare + gemm) + grouped MoE (moe_gemm + build_strided_ptrs).
// NEVER registers `_auto` (removed from the vendored awq_sm70_gemm.cu) — the required
// contract is prepare -> meta(k_ld,q_ld) -> gemm. See docs/FP8_ENGINE_STAGE_A_*.md.
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

// defined in binding/awq_sm70_gemm.cu (global namespace)
std::vector<torch::Tensor> fp8_sm70_prepare(torch::Tensor qweight,
                                            torch::Tensor scales,
                                            int64_t group_size);
void fp8_gemm_sm70_out(torch::Tensor out,
                       torch::Tensor in_feats,
                       torch::Tensor kernel,
                       torch::Tensor scaling_factors,
                       int64_t group_size,
                       int64_t k_ld,
                       int64_t q_ld);
std::vector<torch::Tensor> awq_moe_build_strided_ptrs(torch::Tensor tm_weights,
                                                      torch::Tensor tm_scales,
                                                      int64_t k_ld,
                                                      int64_t q_ld,
                                                      int64_t num_experts);
void fp8_moe_gemm_sm70_out(torch::Tensor out,
                           torch::Tensor sorted_input,
                           torch::Tensor expert_offsets,
                           torch::Tensor strided_ptrs_w,
                           torch::Tensor strided_ptrs_s,
                           int64_t num_experts,
                           int64_t k,
                           int64_t n,
                           int64_t group_size,
                           bool gated_silu);

TORCH_LIBRARY(turbomind_fp8_sm70, m) {
  m.def("fp8_sm70_prepare(Tensor qweight, Tensor scales, int group_size) -> Tensor[]");
  m.def("fp8_gemm_sm70_out(Tensor(a!) out, Tensor in_feats, Tensor kernel, "
        "Tensor scaling_factors, int group_size, int k_ld, int q_ld) -> ()");
  m.def("awq_moe_build_strided_ptrs(Tensor tm_weights, Tensor tm_scales, "
        "int k_ld, int q_ld, int num_experts) -> Tensor[]");
  m.def("fp8_moe_gemm_sm70_out(Tensor(a!) out, Tensor sorted_input, Tensor expert_offsets, "
        "Tensor strided_ptrs_w, Tensor strided_ptrs_s, int num_experts, int k, int n, "
        "int group_size, bool gated_silu) -> ()");
}

TORCH_LIBRARY_IMPL(turbomind_fp8_sm70, CUDA, m) {
  m.impl("fp8_sm70_prepare", &fp8_sm70_prepare);
  m.impl("fp8_gemm_sm70_out", &fp8_gemm_sm70_out);
  m.impl("awq_moe_build_strided_ptrs", &awq_moe_build_strided_ptrs);
  m.impl("fp8_moe_gemm_sm70_out", &fp8_moe_gemm_sm70_out);
}
