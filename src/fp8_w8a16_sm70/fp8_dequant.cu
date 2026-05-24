// FP8 (E4M3-FN) -> FP16 conversion kernel for V100 (sm_70)
//
// "Hello world" correctness test: prove that a CUDA-core dequant
// produces bit-exact results against PyTorch's reference CPU conversion.
//
// E4M3-FN bit layout (1 byte):
//   bit 7    : sign
//   bits 3-6 : exponent (4 bits, bias = 7)
//   bits 0-2 : mantissa (3 bits)
//   Special values (FN convention = "Finite, No Inf"):
//     0x00 / 0x80 : +0 / -0
//     0x7F / 0xFF : NaN  (exp=1111, mant=111)  -- no infinity encoding
//     exp=0, mant!=0  : subnormal
//     exp=15, mant<7  : normal (max magnitude = 2^8 * 1.75 = 448)
//
// FP16 bit layout (2 bytes):
//   bit 15   : sign
//   bits 10-14: exponent (5 bits, bias = 15)
//   bits 0-9 : mantissa (10 bits)

#include <cuda_fp16.h>
#include <cstdint>
#include <mma.h>                       // nvcuda::wmma — V100 HMMA.884 tensor cores
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>   // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <c10/cuda/CUDAStream.h>       // at::cuda::getCurrentCUDAStream

// All kernel launches in this file must use PyTorch's current CUDA stream,
// not the default stream 0. Inside vLLM at TP>1, NCCL and other ops run on
// non-default streams; if we launch on stream 0, our writes are not
// stream-ordered with subsequent torch reads → downstream layers consume
// half-written kernel output → cascading numerical garbage. (This is what
// caused the 27B+TP=4 "!!!!!" bug; offline tests and TP=1 happen to use
// stream 0 throughout so the race never manifested there.)
#define V100_FP8_STREAM at::cuda::getCurrentCUDAStream()

__device__ inline uint16_t fp8_e4m3_to_fp16_bits(uint8_t x) {
    uint16_t sign = (uint16_t)(x & 0x80) << 8;   // 0x8000 if negative, else 0
    uint16_t exp  = (x >> 3) & 0x0F;             // 4-bit FP8 exponent
    uint16_t mant = x & 0x07;                    // 3-bit FP8 mantissa

    // Zero
    if (exp == 0 && mant == 0) {
        return sign;                             // +0 or -0
    }

    // NaN: E4M3-FN reserves exp=1111 AND mant=111 (0x7F/0xFF) as the only NaN.
    // Use the same canonical qNaN payload PyTorch's CPU conversion produces,
    // so bit-exact comparison against the reference passes.
    if (exp == 0x0F && mant == 0x07) {
        return sign | 0x7F80;                    // qNaN, mant = 0x380
    }

    // Subnormal in FP8: value = (mant/8) * 2^(1-bias) = mant * 2^-9
    // Re-normalize into FP16's normal range. If highest set bit of mant is
    // at position k (0-indexed), then value = (1 + lower/2^k) * 2^(k-9).
    // FP16 stored exp = (k - 9) + 15 = k + 6.
    if (exp == 0) {
        // Find leading 1 in 3-bit mantissa (mant in {1..7}).
        int shift = 0;
        while ((mant & 0x04) == 0) {             // until bit 2 of mant is set
            mant <<= 1;
            shift++;
        }
        // After loop: leading 1 is at bit 2, so k = 2 - shift.
        // fp16_exp = k + 6 = 8 - shift.
        mant &= 0x03;                            // drop the leading 1 (implicit)
        uint16_t fp16_exp  = (uint16_t)(8 - shift);
        uint16_t fp16_mant = (uint16_t)mant << 8; // 2-bit -> upper 2 bits of 10-bit mantissa
        return sign | (fp16_exp << 10) | fp16_mant;
    }

    // Normal: exp in [1..15], unbiased exp = exp - 7
    // FP16 stored exp = (exp - 7) + 15 = exp + 8
    uint16_t fp16_exp  = exp + 8;
    uint16_t fp16_mant = (uint16_t)mant << 7;    // 3-bit -> upper 3 bits of 10-bit mantissa
    return sign | (fp16_exp << 10) | fp16_mant;
}

__global__ void fp8_to_fp16_kernel(const uint8_t* __restrict__ in,
                                   __half*        __restrict__ out,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t bits = fp8_e4m3_to_fp16_bits(in[i]);
    out[i] = __ushort_as_half(bits);
}

// Phase 2: dequant + per-group scale broadcast.
//   out[i] = fp8_to_fp16(in[i]) * scales[i / group_size]
__global__ void fp8_to_fp16_scaled_kernel(const uint8_t* __restrict__ in,
                                          const __half*  __restrict__ scales,
                                          __half*        __restrict__ out,
                                          int n,
                                          int group_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half v = __ushort_as_half(fp8_e4m3_to_fp16_bits(in[i]));
    __half s = scales[i / group_size];
    out[i]   = __hmul(v, s);
}

// Phase 4: dequant + 2D-block scale broadcast (DeepSeek-style FP8 W8A16).
//   weight  : [N, K] row-major FP8 bytes
//   scales  : [Nb, Kb] row-major FP16, where Nb = ceil(N/block_h), Kb = ceil(K/block_w)
//   out[i, j] = fp8_to_fp16(weight[i, j]) * scales[i / block_h, j / block_w]
//
// 2D thread grid for coalesced loads: threads in a warp span 32 consecutive K
// values so the FP8 byte read is perfectly coalesced.
__global__ void fp8_to_fp16_block_scaled_kernel(
        const uint8_t* __restrict__ in,
        const __half*  __restrict__ scales,
        __half*        __restrict__ out,
        int N, int K,
        int block_h, int block_w,
        int Kb) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // K dim (fast)
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // N dim (slow)
    if (i >= N || j >= K) return;
    int idx = i * K + j;
    int scale_idx = (i / block_h) * Kb + (j / block_w);
    __half v = __ushort_as_half(fp8_e4m3_to_fp16_bits(in[idx]));
    out[idx] = __hmul(v, scales[scale_idx]);
}

torch::Tensor fp8_e4m3_to_fp16(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(),                  "input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kUInt8,   "input must be uint8 (raw FP8 bytes)");
    TORCH_CHECK(x.is_contiguous(),            "input must be contiguous");

    auto out = torch::empty({x.numel()},
                            torch::TensorOptions().dtype(torch::kFloat16).device(x.device()));

    const int n = x.numel();
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    fp8_to_fp16_kernel<<<blocks, threads, 0, V100_FP8_STREAM>>>(
        x.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor fp8_e4m3_to_fp16_scaled(torch::Tensor x,
                                       torch::Tensor scales,
                                       int64_t       group_size) {
    TORCH_CHECK(x.is_cuda() && scales.is_cuda(),         "inputs must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kUInt8,              "x must be uint8 (raw FP8 bytes)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16,       "scales must be float16");
    TORCH_CHECK(x.is_contiguous() && scales.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(group_size > 0,                          "group_size must be > 0");
    TORCH_CHECK(x.numel() % group_size == 0,             "numel must be multiple of group_size");
    TORCH_CHECK(scales.numel() == x.numel() / group_size,
                "scales.numel() must equal x.numel() / group_size");

    auto out = torch::empty({x.numel()},
                            torch::TensorOptions().dtype(torch::kFloat16).device(x.device()));

    const int n = x.numel();
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    fp8_to_fp16_scaled_kernel<<<blocks, threads, 0, V100_FP8_STREAM>>>(
        x.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        n,
        (int)group_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// Phase 5: naive fused FP8 W8A16 GEMM on CUDA cores (V100-compatible).
//
//   A:       FP16 [M, K]
//   W:       FP8 (uint8)  [N, K], row-major; row n = output channel n
//   scales:  FP16 [Nb, Kb] block scales, broadcast pattern from Phase 4
//   C:       FP16 [M, N], C[m, n] = sum_k A[m, k] * dequant(W[n, k]) * scale[n/block_h, k/block_w]
//
// One CTA owns one (m, n_tile) output tile of shape [1, FP8_GEMM_BLOCK_N].
// 128 threads per CTA = one thread per output column in the tile.
// FP32 accumulator per thread, cast to FP16 only on final write.
//
// Memory pattern per CTA:
//   A:      load FP8_GEMM_BLOCK_K halfs to shared per K-chunk
//   W:      each thread reads K bytes (one column of W); no redundancy across threads
//   scales: 1 read per K-chunk (since BLOCK_K == block_w typically)
//
// This is correctness-first; performance pass comes later.

constexpr int FP8_GEMM_BLOCK_N = 128;
constexpr int FP8_GEMM_BLOCK_K = 128;

__global__ void fp8_w8a16_gemm_kernel(
        const __half*  __restrict__ A,
        const uint8_t* __restrict__ W,
        const __half*  __restrict__ scales,
        __half*        __restrict__ C,
        int M, int N, int K,
        int block_h, int block_w,
        int Kb) {
    const int m      = blockIdx.y;
    const int n_base = blockIdx.x * FP8_GEMM_BLOCK_N;
    const int tid    = threadIdx.x;
    const int n      = n_base + tid;
    if (m >= M || n >= N) return;

    const int scale_row = n / block_h;
    __shared__ __half a_shared[FP8_GEMM_BLOCK_K];
    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += FP8_GEMM_BLOCK_K) {
        // Cooperatively load A[m, k_base:k_base+BLOCK_K] into shared memory.
        const int k_load = k_base + tid;
        a_shared[tid] = (k_load < K) ? A[m * K + k_load] : __float2half(0.f);
        __syncthreads();

        // One scale value covers this K-chunk (BLOCK_K == block_w for this model).
        const int scale_col = k_base / block_w;
        const float scale_f = __half2float(scales[scale_row * Kb + scale_col]);

        // Inner FMA loop: each thread accumulates its output column n.
        const int kk_max = min(FP8_GEMM_BLOCK_K, K - k_base);
        #pragma unroll
        for (int kk = 0; kk < FP8_GEMM_BLOCK_K; ++kk) {
            if (kk >= kk_max) break;
            const uint8_t wb = W[n * K + k_base + kk];
            const float w_f = __half2float(__ushort_as_half(fp8_e4m3_to_fp16_bits(wb)));
            const float a_f = __half2float(a_shared[kk]);
            acc += a_f * w_f * scale_f;
        }
        __syncthreads();
    }

    C[m * N + n] = __float2half(acc);
}

// ─── Phase A.1: vectorized W loads ────────────────────────────────────────
//
// Single optimization vs the naive kernel: read 16 FP8 bytes per memory
// transaction via uint4 (= 4 uint32 = 16 bytes) instead of one byte at a time.
//
// Everything else (one thread per output, no M-tiling, FP32 accumulator,
// same shared-memory A preload, same scale handling, same FMA loop body)
// is unchanged from `fp8_w8a16_gemm_kernel` so the isolated speedup of A.1
// is measurable.
//
// Requirement: K must be a multiple of 16 bytes for the uint4 cast to be
// well-aligned. True for all 128-aligned FP8 weight shapes (our model).

constexpr int K_VEC_A1 = 16;   // bytes per vectorized W load

__global__ void fp8_w8a16_gemm_a1_kernel(
        const __half*  __restrict__ A,
        const uint8_t* __restrict__ W,
        const __half*  __restrict__ scales,
        __half*        __restrict__ C,
        int M, int N, int K,
        int block_h, int block_w,
        int Kb) {
    const int m      = blockIdx.y;
    const int n_base = blockIdx.x * FP8_GEMM_BLOCK_N;
    const int tid    = threadIdx.x;
    const int n      = n_base + tid;
    if (m >= M || n >= N) return;

    const int scale_row = n / block_h;
    __shared__ __half a_shared[FP8_GEMM_BLOCK_K];
    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += FP8_GEMM_BLOCK_K) {
        const int k_load = k_base + tid;
        a_shared[tid] = (k_load < K) ? A[m * K + k_load] : __float2half(0.f);
        __syncthreads();

        const int scale_col = k_base / block_w;
        const float scale_f = __half2float(scales[scale_row * Kb + scale_col]);

        // Vectorized W load: process the K-chunk in groups of K_VEC_A1=16 bytes.
        // BLOCK_K / K_VEC_A1 = 128 / 16 = 8 outer iterations per chunk.
        #pragma unroll
        for (int kk_outer = 0; kk_outer < FP8_GEMM_BLOCK_K / K_VEC_A1; ++kk_outer) {
            const int k_off = k_base + kk_outer * K_VEC_A1;
            const uint4 wv  = *reinterpret_cast<const uint4*>(&W[n * K + k_off]);
            const uint8_t* wbytes = reinterpret_cast<const uint8_t*>(&wv);

            #pragma unroll
            for (int kv = 0; kv < K_VEC_A1; ++kv) {
                const int kk = kk_outer * K_VEC_A1 + kv;
                const float w_f = __half2float(__ushort_as_half(
                    fp8_e4m3_to_fp16_bits(wbytes[kv])));
                const float a_f = __half2float(a_shared[kk]);
                acc += a_f * w_f * scale_f;
            }
        }
        __syncthreads();
    }

    C[m * N + n] = __float2half(acc);
}

// ─── Phase A.2: M-tiling on top of A.1 ───────────────────────────────────
//
// Builds on A.1 by having each CTA produce BLOCK_M_A2 output rows
// simultaneously instead of just one. The same W byte read serves
// BLOCK_M_A2 output values via broadcast-FMA — at BLOCK_M_A2 = 8, that's
// an 8x reduction in total W bytes pulled from HBM when M >= 8.
//
// Per-CTA workload: one [BLOCK_M_A2, BLOCK_N_A2] output tile.
// Per-thread: owns one column of that tile, holds BLOCK_M_A2 FP32 accumulators.
// Shared memory: BLOCK_M_A2 x BLOCK_K_A2 halfs (2 KB at BLOCK_M_A2=8).
//
// At M < BLOCK_M_A2 (e.g. decode M=1), the tail is masked — kernel still
// works but loses the amortization benefit; A.1 may match or beat A.2 there.

constexpr int BLOCK_N_A2 = 128;
constexpr int BLOCK_K_A2 = 128;
constexpr int BLOCK_M_A2 = 8;
constexpr int K_VEC_A2   = 16;

__global__ void fp8_w8a16_gemm_a2_kernel(
        const __half*  __restrict__ A,
        const uint8_t* __restrict__ W,
        const __half*  __restrict__ scales,
        __half*        __restrict__ C,
        int M, int N, int K,
        int block_h, int block_w,
        int Kb) {
    const int n_base = blockIdx.x * BLOCK_N_A2;
    const int m_base = blockIdx.y * BLOCK_M_A2;
    const int tid    = threadIdx.x;
    const int n      = n_base + tid;
    if (n >= N) return;

    const int scale_row = n / block_h;

    __shared__ __half a_shared[BLOCK_M_A2][BLOCK_K_A2];

    float acc[BLOCK_M_A2];
    #pragma unroll
    for (int m = 0; m < BLOCK_M_A2; ++m) acc[m] = 0.0f;

    for (int k_base = 0; k_base < K; k_base += BLOCK_K_A2) {
        // Cooperative A load: thread tid loads column tid for all BLOCK_M_A2 rows.
        // Zero-pad rows past M so the tail tile still computes safely.
        #pragma unroll
        for (int m = 0; m < BLOCK_M_A2; ++m) {
            const int row = m_base + m;
            const int col = k_base + tid;
            a_shared[m][tid] = (row < M && col < K)
                ? A[row * K + col]
                : __float2half(0.f);
        }
        __syncthreads();

        const int scale_col = k_base / block_w;
        const float scale_f = __half2float(scales[scale_row * Kb + scale_col]);

        // Vectorized W load: 8 uint4 transactions per K-chunk, 16 bytes each.
        #pragma unroll
        for (int kk_outer = 0; kk_outer < BLOCK_K_A2 / K_VEC_A2; ++kk_outer) {
            const int k_off = k_base + kk_outer * K_VEC_A2;
            const uint4 wv  = *reinterpret_cast<const uint4*>(&W[n * K + k_off]);
            const uint8_t* wbytes = reinterpret_cast<const uint8_t*>(&wv);

            #pragma unroll
            for (int kv = 0; kv < K_VEC_A2; ++kv) {
                const int kk = kk_outer * K_VEC_A2 + kv;
                const float w_f = __half2float(__ushort_as_half(
                    fp8_e4m3_to_fp16_bits(wbytes[kv])));
                const float w_scaled = w_f * scale_f;

                // Broadcast this scaled weight across BLOCK_M_A2 rows.
                #pragma unroll
                for (int m = 0; m < BLOCK_M_A2; ++m) {
                    acc[m] += __half2float(a_shared[m][kk]) * w_scaled;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < BLOCK_M_A2; ++m) {
        const int row = m_base + m;
        if (row < M) {
            C[row * N + n] = __float2half(acc[m]);
        }
    }
}

// ─── Phase A.3: K-axis splitting for low-M decode ────────────────────────
//
// At low M (especially M=1 decode), the A.1/A.2 grids spawn very few CTAs
// (N/128 at M=1 = 20 CTAs vs V100's 80 SMs), leaving most of the GPU idle.
//
// A.3 fixes this by splitting K across K_SPLIT CTAs that cooperate on the
// same output cell via atomic-add. At K_SPLIT=4, M=1 spawns 80 CTAs = full
// SM occupancy.
//
// Trade-offs:
//   + Many CTAs at low M -> better SM occupancy -> faster decode
//   + Each CTA does K/K_SPLIT work -> finishes K_SPLIT× sooner
//   - Atomic-add output contention (K_SPLIT writers per cell, serialized)
//   - Needs FP32 intermediate accumulator buffer (atomicAdd on FP16 is slower
//     and less precise; FP32 atomicAdd is always available and fast on V100)
//   - One extra FP32 -> FP16 conversion pass after the kernel
//
// Constraint: K must be divisible by K_SPLIT * block_w (= 128) so each slice
// is scale-aligned. True for our model's K ∈ {5120, 6144, 9216, 17408} for
// K_SPLIT ∈ {4, 8}.

constexpr int BLOCK_N_A3 = 128;
constexpr int BLOCK_K_A3 = 128;
constexpr int K_VEC_A3   = 16;

__global__ void fp8_w8a16_gemm_a3_kernel(
        const __half*  __restrict__ A,
        const uint8_t* __restrict__ W,
        const __half*  __restrict__ scales,
        float*         __restrict__ C_fp32,    // [M, N] FP32, pre-zeroed
        int M, int N, int K,
        int block_h, int block_w,
        int Kb,
        int k_slice_size) {
    const int n_base   = blockIdx.x * BLOCK_N_A3;
    const int m        = blockIdx.y;
    const int slice_id = blockIdx.z;
    const int tid      = threadIdx.x;
    const int n        = n_base + tid;
    if (m >= M || n >= N) return;

    const int scale_row = n / block_h;
    const int k_start   = slice_id * k_slice_size;
    const int k_end     = min(k_start + k_slice_size, K);

    __shared__ __half a_shared[BLOCK_K_A3];
    float acc = 0.0f;

    for (int k_base = k_start; k_base < k_end; k_base += BLOCK_K_A3) {
        const int k_load = k_base + tid;
        a_shared[tid] = (k_load < K) ? A[m * K + k_load] : __float2half(0.f);
        __syncthreads();

        const int scale_col = k_base / block_w;
        const float scale_f = __half2float(scales[scale_row * Kb + scale_col]);

        #pragma unroll
        for (int kk_outer = 0; kk_outer < BLOCK_K_A3 / K_VEC_A3; ++kk_outer) {
            const int k_off = k_base + kk_outer * K_VEC_A3;
            const uint4 wv  = *reinterpret_cast<const uint4*>(&W[n * K + k_off]);
            const uint8_t* wbytes = reinterpret_cast<const uint8_t*>(&wv);

            #pragma unroll
            for (int kv = 0; kv < K_VEC_A3; ++kv) {
                const int kk = kk_outer * K_VEC_A3 + kv;
                const float w_f = __half2float(__ushort_as_half(
                    fp8_e4m3_to_fp16_bits(wbytes[kv])));
                acc += __half2float(a_shared[kk]) * w_f * scale_f;
            }
        }
        __syncthreads();
    }

    // Atomic-add this slice's partial sum into the shared output cell.
    // FP32 atomicAdd is well-supported on V100, low overhead at K_SPLIT
    // levels of contention (≤ 8 writers per cell).
    atomicAdd(&C_fp32[m * N + n], acc);
}

// ─── Stage 2B: grouped routed A.3 GEMM for MoE decode ─────────────────────
//
// Same low-M math as A.3, but each output row can read from a different expert
// weight matrix. This turns the Python MoE fallback's per-active-expert GEMM
// launches into one launch for all routed rows.
//
// Layouts:
//   A:          [R, K] FP16 routed activations
//   expert_ids: [R]    local expert id for each routed row
//   W:          [E, N, K] FP8 bytes, contiguous
//   scales:     [E, ceil(N/block_h), ceil(K/block_w)] FP16
//   C:          [R, N]

__global__ void fp8_w8a16_grouped_routed_gemm_a3_kernel(
        const __half*  __restrict__ A,
        const int64_t* __restrict__ expert_ids,
        const uint8_t* __restrict__ W,
        const __half*  __restrict__ scales,
        float*         __restrict__ C_fp32,
        int R, int E, int N, int K,
        int block_h, int block_w,
        int Nb, int Kb,
        int k_slice_size) {
    const int n_base   = blockIdx.x * BLOCK_N_A3;
    const int r        = blockIdx.y;
    const int slice_id = blockIdx.z;
    const int tid      = threadIdx.x;
    const int n        = n_base + tid;
    if (r >= R || n >= N) return;

    const int64_t expert64 = expert_ids[r];
    if (expert64 < 0 || expert64 >= E) return;
    const int expert = (int)expert64;

    const int scale_row = n / block_h;
    const int k_start   = slice_id * k_slice_size;
    const int k_end     = min(k_start + k_slice_size, K);
    const int64_t w_base = ((int64_t)expert * N + n) * K;
    const int64_t s_base = ((int64_t)expert * Nb + scale_row) * Kb;

    __shared__ __half a_shared[BLOCK_K_A3];
    float acc = 0.0f;

    for (int k_base = k_start; k_base < k_end; k_base += BLOCK_K_A3) {
        const int k_load = k_base + tid;
        a_shared[tid] = (k_load < K) ? A[r * K + k_load] : __float2half(0.f);
        __syncthreads();

        const int scale_col = k_base / block_w;
        const float scale_f = __half2float(scales[s_base + scale_col]);

        #pragma unroll
        for (int kk_outer = 0; kk_outer < BLOCK_K_A3 / K_VEC_A3; ++kk_outer) {
            const int k_off = k_base + kk_outer * K_VEC_A3;
            const uint4 wv  = *reinterpret_cast<const uint4*>(&W[w_base + k_off]);
            const uint8_t* wbytes = reinterpret_cast<const uint8_t*>(&wv);

            #pragma unroll
            for (int kv = 0; kv < K_VEC_A3; ++kv) {
                const int kk = kk_outer * K_VEC_A3 + kv;
                const float w_f = __half2float(__ushort_as_half(
                    fp8_e4m3_to_fp16_bits(wbytes[kv])));
                acc += __half2float(a_shared[kk]) * w_f * scale_f;
            }
        }
        __syncthreads();
    }

    atomicAdd(&C_fp32[r * N + n], acc);
}

// ─── Phase A.4 POC: WMMA (V100 HMMA.884) FP8 W8A16 GEMM ──────────────────
//
// Goal: prove naive WMMA beats the CUDA-core A.1/A.2 kernels by 2-5× at
// prefill M values (64..512). If yes, justifies full optimized version
// (double-buffering, smem swizzle, scale fusion in epilogue, dispatch).
//
// Design choices (deliberately naive):
//   - Block tile: 64×64 output. 4 warps per block in 2×2 layout. 128 threads.
//   - Each warp owns a 32×32 output region = 2×2 fragments of 16×16.
//   - K step: 16 per iteration (one HMMA.884 K-slice). K loops K/16 times.
//   - Single-buffered smem (no ping-pong). __syncthreads after load, after mma.
//   - FP8 dequant in registers, scale applied per K-iteration (one scalar value
//     per iteration since N_tile=64 < block_h=128, so scale_row is constant).
//   - FP32 accumulator (Volta HMMA.884.F32 variant). Cast to FP16 on store.
//   - C accumulator goes to smem first, then cooperative FP32→FP16 to global.
//
// Constraints (POC limits):
//   - M, N divisible by WMMA_BLOCK_M/N (64)
//   - K divisible by WMMA_BLOCK_K (16)
//   - block_h == block_w == 128 (matches all our target shapes)

constexpr int WMMA_BLOCK_M = 64;
constexpr int WMMA_BLOCK_N = 64;
constexpr int WMMA_BLOCK_K = 16;
constexpr int WMMA_THREADS_PER_BLOCK = 128;   // 4 warps × 32 lanes
constexpr int WMMA_FRAG = 16;                 // Volta only supports 16×16×16 FP16
// Smem tile stride with +8 halves padding to break bank conflicts.
// Naive stride=16 halves (32 B) means every 4th row hits the same 32 banks
// → 4-way conflict on cooperative dequant writes AND on wmma::load_matrix_sync.
// Stride=24 halves (48 B) shifts row-to-row offset to 12 banks → conflict-free
// for the warp's write/load pattern. Costs +1 KB smem per tile (3 KB vs 2 KB);
// total per block goes from 4 KB tiles + 16 KB C_smem → 6 KB tiles + 16 KB C_smem.
// Still well under V100's 96 KB/SM limit.
constexpr int WMMA_TILE_STRIDE = WMMA_BLOCK_K + 8;   // 24 halves

__global__ void fp8_w8a16_gemm_wmma_kernel(
        const __half*  __restrict__ A,
        const uint8_t* __restrict__ W,
        const __half*  __restrict__ scales,
        __half*        __restrict__ C,
        int M, int N, int K,
        int block_h, int block_w,
        int Kb) {
    using namespace nvcuda;

    const int n_start = blockIdx.x * WMMA_BLOCK_N;
    const int m_start = blockIdx.y * WMMA_BLOCK_M;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m  = warp_id / 2;   // 0..1
    const int warp_n  = warp_id % 2;   // 0..1

    // Double-buffered smem: while mma consumes buf[cur], threads load buf[nxt]
    // for the next K-iter. Without cp.async on Volta, the "overlap" is implicit
    // — hardware schedules in-flight ld.global from one warp alongside HMMA
    // from another. The classic Volta dual-buffer pattern.
    __shared__ __half A_tile[2][WMMA_BLOCK_M][WMMA_TILE_STRIDE];   // 2×64×24 = 6 KB
    __shared__ __half B_tile[2][WMMA_BLOCK_N][WMMA_TILE_STRIDE];   // 2×64×24 = 6 KB
    // Per-warp 16×16 FP32 staging buffer for fragment FP32→FP16 cast at the
    // epilogue. 1 KB per warp × 4 warps = 4 KB. Smem total: 16 KB per block.
    __shared__ float  warp_stage[4][WMMA_FRAG][WMMA_FRAG];          // 4 KB

    // 2×2 = 4 accumulator fragments per warp
    wmma::fragment<wmma::accumulator, WMMA_FRAG, WMMA_FRAG, WMMA_FRAG, float> c_frag[2][2];
    #pragma unroll
    for (int fm = 0; fm < 2; ++fm) {
        #pragma unroll
        for (int fn = 0; fn < 2; ++fn) {
            wmma::fill_fragment(c_frag[fm][fn], 0.0f);
        }
    }

    // scale_row is constant across the block (N_tile=64 ≤ block_h=128).
    const int scale_row = n_start / block_h;
    const int K_ITERS = K / WMMA_BLOCK_K;

    // load-to-smem and mma bodies are inlined (no helper lambdas) so the
    // kernel compiles under the existing JIT flags without --extended-lambda.
    // Two clear sections: LOAD_TILE writes A_tile[buf]/B_tile[buf]; MMA_TILE
    // reads them and accumulates into c_frag[].

    // ─── Prologue: load buf[0] for k=0 ───
    {
        const int row     = tid / 2;          // 0..63
        const int col_off = (tid % 2) * 8;    // 0 or 8
        // A
        const __half* a_src = &A[(m_start + row) * K + 0 + col_off];
        *reinterpret_cast<uint4*>(&A_tile[0][row][col_off]) =
            *reinterpret_cast<const uint4*>(a_src);
        // W + dequant + per-block scale
        const int   scale_col_l = 0 / block_w;
        const float scale_f_l   = __half2float(scales[scale_row * Kb + scale_col_l]);
        const uint8_t* w_src    = &W[(n_start + row) * K + 0 + col_off];
        const uint2 wv = *reinterpret_cast<const uint2*>(w_src);
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv);
        #pragma unroll
        for (int kv = 0; kv < 8; ++kv) {
            const float w_dq = __half2float(__ushort_as_half(
                fp8_e4m3_to_fp16_bits(wb[kv])));
            B_tile[0][row][col_off + kv] = __float2half(w_dq * scale_f_l);
        }
    }
    __syncthreads();

    // ─── Main loop: load buf[nxt] for k=k_next while mma uses buf[cur].
    //     One __syncthreads per iter (down from 2) — next-iter load writes
    //     to the OTHER buffer than mma reads from, so no read-after-write race.
    for (int it = 0; it < K_ITERS - 1; ++it) {
        const int cur = it & 1;
        const int nxt = cur ^ 1;
        const int k_next = (it + 1) * WMMA_BLOCK_K;

        // ── Load buf[nxt] for k=k_next
        {
            const int row     = tid / 2;
            const int col_off = (tid % 2) * 8;
            const __half* a_src = &A[(m_start + row) * K + k_next + col_off];
            *reinterpret_cast<uint4*>(&A_tile[nxt][row][col_off]) =
                *reinterpret_cast<const uint4*>(a_src);
            const int   scale_col_l = k_next / block_w;
            const float scale_f_l   = __half2float(scales[scale_row * Kb + scale_col_l]);
            const uint8_t* w_src    = &W[(n_start + row) * K + k_next + col_off];
            const uint2 wv = *reinterpret_cast<const uint2*>(w_src);
            const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv);
            #pragma unroll
            for (int kv = 0; kv < 8; ++kv) {
                const float w_dq = __half2float(__ushort_as_half(
                    fp8_e4m3_to_fp16_bits(wb[kv])));
                B_tile[nxt][row][col_off + kv] = __float2half(w_dq * scale_f_l);
            }
        }

        // ── MMA using buf[cur]
        {
            wmma::fragment<wmma::matrix_a, WMMA_FRAG, WMMA_FRAG, WMMA_FRAG, __half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, WMMA_FRAG, WMMA_FRAG, WMMA_FRAG, __half, wmma::col_major> b_frag[2];
            #pragma unroll
            for (int fm = 0; fm < 2; ++fm) {
                const int m_off = warp_m * 32 + fm * 16;
                wmma::load_matrix_sync(a_frag[fm], &A_tile[cur][m_off][0], WMMA_TILE_STRIDE);
            }
            #pragma unroll
            for (int fn = 0; fn < 2; ++fn) {
                const int n_off = warp_n * 32 + fn * 16;
                wmma::load_matrix_sync(b_frag[fn], &B_tile[cur][n_off][0], WMMA_TILE_STRIDE);
            }
            #pragma unroll
            for (int fm = 0; fm < 2; ++fm) {
                #pragma unroll
                for (int fn = 0; fn < 2; ++fn) {
                    wmma::mma_sync(c_frag[fm][fn], a_frag[fm], b_frag[fn], c_frag[fm][fn]);
                }
            }
        }

        __syncthreads();
    }

    // ─── Epilogue mma: final tile (loaded by the last main-loop iter) ───
    {
        const int last_buf = (K_ITERS - 1) & 1;
        wmma::fragment<wmma::matrix_a, WMMA_FRAG, WMMA_FRAG, WMMA_FRAG, __half, wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b, WMMA_FRAG, WMMA_FRAG, WMMA_FRAG, __half, wmma::col_major> b_frag[2];
        #pragma unroll
        for (int fm = 0; fm < 2; ++fm) {
            const int m_off = warp_m * 32 + fm * 16;
            wmma::load_matrix_sync(a_frag[fm], &A_tile[last_buf][m_off][0], WMMA_TILE_STRIDE);
        }
        #pragma unroll
        for (int fn = 0; fn < 2; ++fn) {
            const int n_off = warp_n * 32 + fn * 16;
            wmma::load_matrix_sync(b_frag[fn], &B_tile[last_buf][n_off][0], WMMA_TILE_STRIDE);
        }
        #pragma unroll
        for (int fm = 0; fm < 2; ++fm) {
            #pragma unroll
            for (int fn = 0; fn < 2; ++fn) {
                wmma::mma_sync(c_frag[fm][fn], a_frag[fm], b_frag[fn], c_frag[fm][fn]);
            }
        }
    }

    // ─── Epilogue: per-warp fragment store → FP16 cast → direct global write ───
    // Each warp writes its own 32×32 output region (= 4 fragments) one
    // fragment at a time. wmma::store_matrix_sync is warp-synchronous, so no
    // __syncwarp is needed before lanes read from warp_stage. Each warp uses
    // its own staging slot, so no __syncthreads either.
    const int lane = tid & 31;
    #pragma unroll
    for (int fm = 0; fm < 2; ++fm) {
        #pragma unroll
        for (int fn = 0; fn < 2; ++fn) {
            wmma::store_matrix_sync(&warp_stage[warp_id][0][0],
                                    c_frag[fm][fn],
                                    WMMA_FRAG,
                                    wmma::mem_row_major);
            // 32 lanes × 8 elements = 256 = 16×16. Cast and write to global.
            const int m_off = m_start + warp_m * 32 + fm * 16;
            const int n_off = n_start + warp_n * 32 + fn * 16;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int idx = lane + i * 32;       // 0..255
                const int r   = idx >> 4;            // / 16
                const int c   = idx & 15;            // % 16
                C[(m_off + r) * N + (n_off + c)] =
                    __float2half(warp_stage[warp_id][r][c]);
            }
        }
    }
}

torch::Tensor fp8_w8a16_gemm_wmma_poc(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor scales,
                                       int64_t       N,
                                       int64_t       K,
                                       int64_t       block_h,
                                       int64_t       block_w) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scales.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(input.dtype()  == torch::kFloat16, "input must be float16");
    TORCH_CHECK(weight.dtype() == torch::kUInt8,   "weight must be uint8 (raw FP8 bytes)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && scales.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == K,
                "input must be [M, K] with K matching");
    TORCH_CHECK(weight.numel() == N * K, "weight.numel() must equal N*K");
    TORCH_CHECK(block_h == 128 && block_w == 128,
                "WMMA POC requires block_h=block_w=128 (scale layout assumption)");

    const int M  = (int)input.size(0);
    TORCH_CHECK(M % WMMA_BLOCK_M == 0,
                "WMMA POC requires M divisible by 64");
    TORCH_CHECK(N % WMMA_BLOCK_N == 0,
                "WMMA POC requires N divisible by 64");
    TORCH_CHECK(K % WMMA_BLOCK_K == 0,
                "WMMA POC requires K divisible by 16");

    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.numel() == (int64_t)Nb * Kb,
                "scales.numel() must equal ceil(N/block_h) * ceil(K/block_w)");

    auto C = torch::empty({(int64_t)M, (int64_t)N},
                          torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));

    dim3 block(WMMA_THREADS_PER_BLOCK);
    dim3 grid(((int)N) / WMMA_BLOCK_N, M / WMMA_BLOCK_M);

    fp8_w8a16_gemm_wmma_kernel<<<grid, block, 0, V100_FP8_STREAM>>>(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, (int)N, (int)K, (int)block_h, (int)block_w, Kb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

torch::Tensor fp8_w8a16_gemm_a3(torch::Tensor input,
                                 torch::Tensor weight,
                                 torch::Tensor scales,
                                 int64_t       N,
                                 int64_t       K,
                                 int64_t       block_h,
                                 int64_t       block_w,
                                 int64_t       k_split) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scales.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(input.dtype()  == torch::kFloat16, "input must be float16");
    TORCH_CHECK(weight.dtype() == torch::kUInt8,   "weight must be uint8 (raw FP8 bytes)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && scales.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == K,
                "input must be [M, K] with K matching");
    TORCH_CHECK(weight.numel() == N * K, "weight.numel() must equal N*K");
    TORCH_CHECK(k_split >= 1, "k_split must be >= 1");
    TORCH_CHECK(K % (k_split * block_w) == 0,
                "K must be divisible by k_split * block_w for scale alignment");
    TORCH_CHECK(K % K_VEC_A3 == 0,
                "K must be divisible by 16 for vectorized W loads");

    const int M  = (int)input.size(0);
    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.numel() == (int64_t)Nb * Kb,
                "scales.numel() must equal ceil(N/block_h) * ceil(K/block_w)");

    const int k_slice_size = (int)K / (int)k_split;

    // FP32 accumulator, pre-zeroed for atomic-add.
    auto C_fp32 = torch::zeros({(int64_t)M, (int64_t)N},
                                torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    dim3 block(BLOCK_N_A3);
    dim3 grid(((int)N + BLOCK_N_A3 - 1) / BLOCK_N_A3, M, (int)k_split);

    fp8_w8a16_gemm_a3_kernel<<<grid, block, 0, V100_FP8_STREAM>>>(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        C_fp32.data_ptr<float>(),
        M, (int)N, (int)K, (int)block_h, (int)block_w, Kb,
        k_slice_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Convert FP32 accumulator -> FP16 result.
    return C_fp32.to(torch::kFloat16);
}

torch::Tensor fp8_w8a16_grouped_routed_gemm_a3(
                                 torch::Tensor input,
                                 torch::Tensor expert_ids,
                                 torch::Tensor weight,
                                 torch::Tensor scales,
                                 int64_t       N,
                                 int64_t       K,
                                 int64_t       block_h,
                                 int64_t       block_w,
                                 int64_t       k_split) {
    TORCH_CHECK(input.is_cuda() && expert_ids.is_cuda() &&
                weight.is_cuda() && scales.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(input.dtype()  == torch::kFloat16, "input must be float16");
    TORCH_CHECK(expert_ids.dtype() == torch::kInt64, "expert_ids must be int64");
    TORCH_CHECK(weight.dtype() == torch::kUInt8,   "weight must be uint8 (raw FP8 bytes)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(input.is_contiguous() && expert_ids.is_contiguous() &&
                weight.is_contiguous() && scales.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == K,
                "input must be [R, K] with K matching");
    TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.size(0) == input.size(0),
                "expert_ids must be [R]");
    TORCH_CHECK(weight.dim() == 3 && weight.size(1) == N && weight.size(2) == K,
                "weight must be [E, N, K]");
    TORCH_CHECK(scales.dim() == 3, "scales must be [E, Nb, Kb]");
    TORCH_CHECK(scales.size(0) == weight.size(0), "scales E must match weight E");
    TORCH_CHECK(k_split >= 1, "k_split must be >= 1");
    TORCH_CHECK(K % (k_split * block_w) == 0,
                "K must be divisible by k_split * block_w for scale alignment");
    TORCH_CHECK(K % K_VEC_A3 == 0,
                "K must be divisible by 16 for vectorized W loads");

    const int R  = (int)input.size(0);
    const int E  = (int)weight.size(0);
    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.size(1) == Nb && scales.size(2) == Kb,
                "scales must be [E, ceil(N/block_h), ceil(K/block_w)]");

    const int k_slice_size = (int)K / (int)k_split;
    auto C_fp32 = torch::zeros({(int64_t)R, (int64_t)N},
                                torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    if (R == 0) {
        return C_fp32.to(torch::kFloat16);
    }

    dim3 block(BLOCK_N_A3);
    dim3 grid(((int)N + BLOCK_N_A3 - 1) / BLOCK_N_A3, R, (int)k_split);

    fp8_w8a16_grouped_routed_gemm_a3_kernel<<<grid, block, 0, V100_FP8_STREAM>>>(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        expert_ids.data_ptr<int64_t>(),
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        C_fp32.data_ptr<float>(),
        R, E, (int)N, (int)K, (int)block_h, (int)block_w, Nb, Kb,
        k_slice_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C_fp32.to(torch::kFloat16);
}

torch::Tensor fp8_w8a16_gemm_a2(torch::Tensor input,
                                 torch::Tensor weight,
                                 torch::Tensor scales,
                                 int64_t       N,
                                 int64_t       K,
                                 int64_t       block_h,
                                 int64_t       block_w) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scales.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(input.dtype()  == torch::kFloat16, "input must be float16");
    TORCH_CHECK(weight.dtype() == torch::kUInt8,   "weight must be uint8 (raw FP8 bytes)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && scales.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == K,
                "input must be [M, K] with K matching");
    TORCH_CHECK(weight.numel() == N * K, "weight.numel() must equal N*K");
    TORCH_CHECK(K % K_VEC_A2 == 0,
                "K must be divisible by 16 for vectorized W loads");

    const int M  = (int)input.size(0);
    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.numel() == (int64_t)Nb * Kb,
                "scales.numel() must equal ceil(N/block_h) * ceil(K/block_w)");

    auto C = torch::empty({(int64_t)M, (int64_t)N},
                          torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));

    dim3 block(BLOCK_N_A2);
    dim3 grid(((int)N + BLOCK_N_A2 - 1) / BLOCK_N_A2,
              (M       + BLOCK_M_A2 - 1) / BLOCK_M_A2);

    fp8_w8a16_gemm_a2_kernel<<<grid, block, 0, V100_FP8_STREAM>>>(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, (int)N, (int)K, (int)block_h, (int)block_w, Kb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

torch::Tensor fp8_w8a16_gemm_a1(torch::Tensor input,
                                 torch::Tensor weight,
                                 torch::Tensor scales,
                                 int64_t       N,
                                 int64_t       K,
                                 int64_t       block_h,
                                 int64_t       block_w) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scales.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(input.dtype()  == torch::kFloat16, "input must be float16");
    TORCH_CHECK(weight.dtype() == torch::kUInt8,   "weight must be uint8 (raw FP8 bytes)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && scales.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == K,
                "input must be [M, K] with K matching");
    TORCH_CHECK(weight.numel() == N * K, "weight.numel() must equal N*K");
    TORCH_CHECK(K % K_VEC_A1 == 0,
                "K must be divisible by 16 for vectorized W loads");

    const int M  = (int)input.size(0);
    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.numel() == (int64_t)Nb * Kb,
                "scales.numel() must equal ceil(N/block_h) * ceil(K/block_w)");

    auto C = torch::empty({(int64_t)M, (int64_t)N},
                          torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));

    dim3 block(FP8_GEMM_BLOCK_N);
    dim3 grid(((int)N + FP8_GEMM_BLOCK_N - 1) / FP8_GEMM_BLOCK_N, M);

    fp8_w8a16_gemm_a1_kernel<<<grid, block, 0, V100_FP8_STREAM>>>(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, (int)N, (int)K, (int)block_h, (int)block_w, Kb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

torch::Tensor fp8_w8a16_gemm(torch::Tensor input,
                             torch::Tensor weight,
                             torch::Tensor scales,
                             int64_t       N,
                             int64_t       K,
                             int64_t       block_h,
                             int64_t       block_w) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scales.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(input.dtype()  == torch::kFloat16, "input must be float16");
    TORCH_CHECK(weight.dtype() == torch::kUInt8,   "weight must be uint8 (raw FP8 bytes)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && scales.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == K,
                "input must be [M, K] with K matching");
    TORCH_CHECK(weight.numel() == N * K, "weight.numel() must equal N*K");

    const int M  = (int)input.size(0);
    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.numel() == (int64_t)Nb * Kb,
                "scales.numel() must equal ceil(N/block_h) * ceil(K/block_w)");

    auto C = torch::empty({(int64_t)M, (int64_t)N},
                          torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));

    dim3 block(FP8_GEMM_BLOCK_N);
    dim3 grid(((int)N + FP8_GEMM_BLOCK_N - 1) / FP8_GEMM_BLOCK_N, M);

    fp8_w8a16_gemm_kernel<<<grid, block, 0, V100_FP8_STREAM>>>(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, (int)N, (int)K, (int)block_h, (int)block_w, Kb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

torch::Tensor fp8_e4m3_to_fp16_block_scaled(torch::Tensor weight,
                                             torch::Tensor scales,
                                             int64_t       N,
                                             int64_t       K,
                                             int64_t       block_h,
                                             int64_t       block_w) {
    TORCH_CHECK(weight.is_cuda() && scales.is_cuda(),     "inputs must be CUDA");
    TORCH_CHECK(weight.dtype() == torch::kUInt8,          "weight must be uint8 (raw FP8 bytes)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16,        "scales must be float16");
    TORCH_CHECK(weight.is_contiguous() && scales.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(N > 0 && K > 0 && block_h > 0 && block_w > 0, "sizes must be positive");
    TORCH_CHECK(weight.numel() == N * K,                  "weight.numel() != N*K");

    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.numel() == (int64_t)Nb * Kb,
                "scales.numel() must equal ceil(N/block_h) * ceil(K/block_w)");

    auto out = torch::empty({N * K},
                            torch::TensorOptions().dtype(torch::kFloat16).device(weight.device()));

    dim3 block(32, 8);   // 32 along K (coalesced byte loads), 8 along N
    dim3 grid(((int)K + block.x - 1) / block.x,
              ((int)N + block.y - 1) / block.y);

    fp8_to_fp16_block_scaled_kernel<<<grid, block, 0, V100_FP8_STREAM>>>(
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        (int)N, (int)K, (int)block_h, (int)block_w, Kb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_e4m3_to_fp16",              &fp8_e4m3_to_fp16,
          "Convert raw FP8 E4M3-FN bytes (uint8) to FP16 on GPU");
    m.def("fp8_e4m3_to_fp16_scaled",       &fp8_e4m3_to_fp16_scaled,
          "Dequant FP8 E4M3-FN bytes and multiply by per-group FP16 scales");
    m.def("fp8_e4m3_to_fp16_block_scaled", &fp8_e4m3_to_fp16_block_scaled,
          "Dequant FP8 E4M3-FN with 2D block scales [N/block_h, K/block_w] "
          "(DeepSeek-style FP8 W8A16)");
    m.def("fp8_w8a16_gemm",                &fp8_w8a16_gemm,
          "Naive fused FP8 W8A16 GEMM on CUDA cores (V100). "
          "C[M,N] = A[M,K] @ dequant(W[N,K]).T with 2D block scales.");
    m.def("fp8_w8a16_gemm_a1",             &fp8_w8a16_gemm_a1,
          "Phase A.1 GEMM: same as fp8_w8a16_gemm but with vectorized "
          "16-byte W loads via uint4. No M-tiling yet.");
    m.def("fp8_w8a16_gemm_a2",             &fp8_w8a16_gemm_a2,
          "Phase A.2 GEMM: A.1 + M-tiling. Each CTA produces BLOCK_M=8 "
          "output rows simultaneously, amortizing W reads across rows.");
    m.def("fp8_w8a16_gemm_a3",             &fp8_w8a16_gemm_a3,
          "Phase A.3 GEMM: A.1 + K-axis CTA splitting via atomic-add. "
          "K_SPLIT extra CTAs share each output cell. Targets low-M decode "
          "where naive/A.1/A.2 grids under-utilize SMs.");
    m.def("fp8_w8a16_grouped_routed_gemm_a3",
          &fp8_w8a16_grouped_routed_gemm_a3,
          "Stage 2B grouped routed A.3 GEMM for MoE decode. Each routed row "
          "selects its local expert via expert_ids[R], reducing per-expert "
          "Python launch fanout.");
    m.def("fp8_w8a16_gemm_wmma_poc",       &fp8_w8a16_gemm_wmma_poc,
          "Phase A.4 POC: WMMA-based GEMM on V100 HMMA.884 tensor cores. "
          "Naive 64×64 tile, 4 warps, no double-buffering. POC to gauge "
          "whether full WMMA optimization is worth pursuing.");
}
