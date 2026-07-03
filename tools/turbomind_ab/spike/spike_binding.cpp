// Stage-F build spike :: minimal torch binding for the SM70 FP8 dense path.
// Registers ONLY fp8_sm70_prepare + fp8_gemm_sm70_out (enough for the Stage-D gate).
// The implementations are the global-namespace wrappers defined in 1catai's
// awq_sm70_gemm.cu (which call vllm::awq_sm70::...). We never register `_auto`.
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

// defined in awq_sm70_gemm.cu (global namespace)
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

TORCH_LIBRARY(fp8sm70_spike, m) {
  m.def("fp8_sm70_prepare(Tensor qweight, Tensor scales, int group_size) -> Tensor[]");
  m.def("fp8_gemm_sm70_out(Tensor(a!) out, Tensor in_feats, Tensor kernel, "
        "Tensor scaling_factors, int group_size, int k_ld, int q_ld) -> ()");
}

TORCH_LIBRARY_IMPL(fp8sm70_spike, CUDA, m) {
  m.impl("fp8_sm70_prepare", &fp8_sm70_prepare);
  m.impl("fp8_gemm_sm70_out", &fp8_gemm_sm70_out);
}
