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
#include <cstdlib>
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

// Branchless E4M3-FN -> FP16, bit-identical to the original branchy converter
// for all 256 bytes (normal/subnormal/zero/NaN verified). Shift the 7 magnitude
// bits into FP16 position and rebias via a single *2^8 multiply; FP16's wider
// exponent renormalizes E4M3 subnormals automatically -> no branches, no while-loop,
// no warp divergence. Targets the ALU/dequant bottleneck the NCU found
// (coal_m M=8 large-K: ALU 40% >> FMA 26% >> LSU 13%, DRAM 11%).
__device__ inline uint16_t fp8_e4m3_to_fp16_bits(uint8_t x) {
    const uint16_t sign = (uint16_t)(x & 0x80) << 8;
    const uint16_t mag  = (uint16_t)(x & 0x7F);
    const half h = __hmul(__ushort_as_half((uint16_t)(mag << 7)), __float2half(256.0f));
    const uint16_t val = (mag == 0x7F) ? (uint16_t)0x7F80 : __half_as_ushort(h);
    return (uint16_t)(sign | val);
}

// Compatibility alias for experiments that explicitly call the fast converter.
__device__ inline uint16_t fp8_e4m3_to_fp16_bits_fast(uint8_t x) {
    return fp8_e4m3_to_fp16_bits(x);
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
    // Partial-CTA safety: m = blockIdx.y is uniform across the CTA, so an m>=M
    // early-return is non-divergent and safe. n>=N is NOT uniform (the last
    // N-tile is partial), so we must NOT early-return on it — threads with n>=N
    // still have to reach the tid-indexed a_shared load and both __syncthreads()
    // below, or surviving lanes read uninitialized a_shared slots (the
    // garbage-output bug on non-128-aligned N). Mask only the n-dependent work.
    if (m >= M) return;
    const bool active = (n < N);

    const int scale_row = active ? (n / block_h) : 0;
    __shared__ __half a_shared[FP8_GEMM_BLOCK_K];
    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += FP8_GEMM_BLOCK_K) {
        const int k_load = k_base + tid;
        a_shared[tid] = (k_load < K) ? A[m * K + k_load] : __float2half(0.f);
        __syncthreads();

        if (active) {
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
        }
        __syncthreads();
    }

    if (active) C[m * N + n] = __float2half(acc);
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
    // Partial-CTA safety: do NOT early-return on n>=N. The cooperative A load
    // below is tid-indexed (a_shared[m][tid]) and n-independent, so every lane
    // must run it and both __syncthreads(); a returning lane leaves its
    // a_shared column uninitialized -> garbage for the surviving lanes (the
    // non-128-aligned-N bug). Mask only the n-dependent W-read/compute/write.
    const bool active = (n < N);

    const int scale_row = active ? (n / block_h) : 0;

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

        if (active) {
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
        }
        __syncthreads();
    }

    if (active) {
        #pragma unroll
        for (int m = 0; m < BLOCK_M_A2; ++m) {
            const int row = m_base + m;
            if (row < M) {
                C[row * N + n] = __float2half(acc[m]);
            }
        }
    }
}

// ─── Phase 4 Stage 1.5: FUSED grouped-tiled GEMM (one launch for ALL routed rows)
//
// The per-expert Python loop (128 a2 launches + 3x .tolist() per layer x45) was
// ~17.5s of prefill WALL (CPU/sync serialization), per the profiler — far more
// than the 12s of actual w13 GPU. This kernel does ALL experts in ONE launch using
// GPU-side offsets, so NO host loop and NO .tolist() on the hot path.
//
// Layout: A/C are pre-sorted by expert (caller's argsort). Each CTA owns
// (n-tile, tile_idx); a tile = BLOCK_M_A2 contiguous sorted rows of ONE expert.
// The tile's expert is found by binary search over expert_tile_off (GPU [E]).
// Padding tiles (grid rounds up) self-exit via the per-expert range guard, so the
// caller needs no total_tiles sync. Inner math == a2 (weight reused across rows).
__global__ void fp8_w8a16_grouped_tiled_gemm_kernel(
        const __half*  __restrict__ A,                 // [R, K] sorted by expert
        const int*     __restrict__ expert_tile_off,   // [E] first tile idx of expert
        const int*     __restrict__ tiles_per_e,       // [E] tiles for expert
        const int*     __restrict__ expert_row_off,    // [E] first sorted-row of expert
        const int*     __restrict__ counts,            // [E] rows for expert
        const uint8_t* __restrict__ W,                 // [E, N, K] fp8 bytes
        const __half*  __restrict__ scales,            // [E, Nb, Kb]
        __half*        __restrict__ C,                 // [R, N] sorted
        int R, int E, int N, int K,
        int block_h, int block_w, int Nb, int Kb) {
    const int n_base   = blockIdx.x * BLOCK_N_A2;
    const int tile_idx = blockIdx.y;                   // uniform across CTA
    const int tid      = threadIdx.x;
    const int n        = n_base + tid;

    // Find expert: rightmost e with expert_tile_off[e] <= tile_idx (skips 0-tile
    // experts since they share their successor's offset). Uniform -> safe to return.
    int lo = 0, hi = E - 1, e = 0;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (expert_tile_off[mid] <= tile_idx) { e = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    if (tile_idx >= expert_tile_off[e] + tiles_per_e[e]) return;   // padding tile

    const int local     = tile_idx - expert_tile_off[e];
    const int row_start = expert_row_off[e] + local * BLOCK_M_A2;
    const int n_rows    = min(BLOCK_M_A2, counts[e] - local * BLOCK_M_A2);

    const bool active    = (n < N);
    const int  scale_row = active ? ((block_h == 1) ? n : (n / block_h)) : 0;
    const int64_t w_base = (int64_t)e * N * K;
    const int64_t s_base = (int64_t)e * Nb * Kb;

    __shared__ __half a_shared[BLOCK_M_A2][BLOCK_K_A2];
    float acc[BLOCK_M_A2];
    #pragma unroll
    for (int m = 0; m < BLOCK_M_A2; ++m) acc[m] = 0.0f;

    for (int k_base = 0; k_base < K; k_base += BLOCK_K_A2) {
        // Cooperative A load: thread tid loads col tid for all BLOCK_M_A2 rows.
        // Every lane runs this + both __syncthreads (partial-CTA safety).
        #pragma unroll
        for (int m = 0; m < BLOCK_M_A2; ++m) {
            const int col = k_base + tid;
            a_shared[m][tid] = (m < n_rows && col < K)
                ? A[(int64_t)(row_start + m) * K + col] : __float2half(0.f);
        }
        __syncthreads();

        if (active) {
            const int   scale_col = k_base / block_w;
            const float scale_f   = __half2float(scales[s_base + scale_row * Kb + scale_col]);
            #pragma unroll
            for (int kk_outer = 0; kk_outer < BLOCK_K_A2 / K_VEC_A2; ++kk_outer) {
                const int k_off = k_base + kk_outer * K_VEC_A2;
                const uint4 wv = *reinterpret_cast<const uint4*>(&W[w_base + (int64_t)n * K + k_off]);
                const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv);
                #pragma unroll
                for (int kv = 0; kv < K_VEC_A2; ++kv) {
                    const int kk = kk_outer * K_VEC_A2 + kv;
                    const float w_scaled = __half2float(__ushort_as_half(
                        fp8_e4m3_to_fp16_bits(wb[kv]))) * scale_f;
                    #pragma unroll
                    for (int m = 0; m < BLOCK_M_A2; ++m)
                        acc[m] += __half2float(a_shared[m][kk]) * w_scaled;
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        #pragma unroll
        for (int m = 0; m < BLOCK_M_A2; ++m)
            if (m < n_rows)
                C[(int64_t)(row_start + m) * N + n] = __float2half(acc[m]);
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
    // Partial-CTA safety: m (blockIdx.y) and slice_id (blockIdx.z) are uniform
    // across the CTA, so an m>=M early-return is safe. n>=N is partial, so mask
    // it instead of returning — all lanes must reach the tid-indexed a_shared
    // load and both __syncthreads() (non-128-aligned-N correctness).
    if (m >= M) return;
    const bool active = (n < N);

    const int scale_row = active ? (n / block_h) : 0;
    const int k_start   = slice_id * k_slice_size;
    const int k_end     = min(k_start + k_slice_size, K);

    __shared__ __half a_shared[BLOCK_K_A3];
    float acc = 0.0f;

    for (int k_base = k_start; k_base < k_end; k_base += BLOCK_K_A3) {
        const int k_load = k_base + tid;
        a_shared[tid] = (k_load < K) ? A[m * K + k_load] : __float2half(0.f);
        __syncthreads();

        if (active) {
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
        }
        __syncthreads();
    }

    // Atomic-add this slice's partial sum into the shared output cell.
    // FP32 atomicAdd is well-supported on V100, low overhead at K_SPLIT
    // levels of contention (≤ 8 writers per cell).
    if (active) atomicAdd(&C_fp32[m * N + n], acc);
}

// ─── Prototype: coalesced M=1 decode GEMV ────────────────────────────────
//
// A.3 maps one thread to one output n, so neighboring warp lanes read W rows
// separated by K bytes. This prototype flips the mapping for decode:
// row. Each lane loads 4 adjacent FP8 values per 128-wide scale block, then the
// warp reduces the partial dot product. The loop handles two 128-wide scale
// blocks per iteration so each warp can issue multiple W loads before doing the
// dequant/FMA work. It is intentionally M==1 only and block_w==128 only so it
// can be tested as a separate fast-path candidate.

constexpr int GEMV_COAL_WARPS_PER_CTA = 8;
constexpr int GEMV_COAL_THREADS       = GEMV_COAL_WARPS_PER_CTA * 32;
constexpr int GEMV_COAL_BLOCK_K       = 128;
constexpr int GEMV_COAL_ELEMS_PER_LANE = 4;  // 32 lanes * 4 bytes = 128 FP8s
constexpr int GEMV_COAL_DEFAULT_UNROLL_K = 2;
constexpr int GEMV_COAL_M_MAX         = 8;

__inline__ __device__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<int UNROLL_K>
__global__ void fp8_w8a16_gemv_coalesced_kernel(
        const __half*  __restrict__ A,       // [1, K]
        const uint8_t* __restrict__ W,       // [N, K], row-major
        const __half*  __restrict__ scales,  // [ceil(N/block_h), ceil(K/128)]
        __half*        __restrict__ C,       // [1, N]
        int N, int K,
        int block_h,
        int Kb) {
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n       = blockIdx.x * GEMV_COAL_WARPS_PER_CTA + warp_id;
    const bool active = (n < N);

    __shared__ __half a_shared[UNROLL_K][GEMV_COAL_BLOCK_K];
    float acc = 0.0f;
    const int scale_row = active ? (n / block_h) : 0;

    for (int k_base = 0; k_base < K; k_base += UNROLL_K * GEMV_COAL_BLOCK_K) {
        if (tid < GEMV_COAL_BLOCK_K) {
            #pragma unroll
            for (int u = 0; u < UNROLL_K; ++u) {
                const int kk = k_base + u * GEMV_COAL_BLOCK_K + tid;
                a_shared[u][tid] = (kk < K) ? A[kk] : __float2half(0.f);
            }
        }
        __syncthreads();

        if (active) {
            uint32_t wv[UNROLL_K];
            float scale_f[UNROLL_K];
            bool has_block[UNROLL_K];

            #pragma unroll
            for (int u = 0; u < UNROLL_K; ++u) {
                const int block_base = k_base + u * GEMV_COAL_BLOCK_K;
                has_block[u] = block_base < K;
                wv[u] = 0;
                scale_f[u] = 0.0f;
                if (has_block[u]) {
                    const int scale_col = block_base / GEMV_COAL_BLOCK_K;
                    scale_f[u] = __half2float(scales[scale_row * Kb + scale_col]);
                    const int k_lane = block_base + lane * GEMV_COAL_ELEMS_PER_LANE;
                    wv[u] = *reinterpret_cast<const uint32_t*>(&W[(int64_t)n * K + k_lane]);
                }
            }

            #pragma unroll
            for (int u = 0; u < UNROLL_K; ++u) {
                if (has_block[u]) {
                    const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv[u]);
                    #pragma unroll
                    for (int j = 0; j < GEMV_COAL_ELEMS_PER_LANE; ++j) {
                        const float w_f = __half2float(__ushort_as_half(
                            fp8_e4m3_to_fp16_bits(wb[j])));
                        const float a_f = __half2float(
                            a_shared[u][lane * GEMV_COAL_ELEMS_PER_LANE + j]);
                        acc += a_f * w_f * scale_f[u];
                    }
                }
            }
        }
        __syncthreads();
    }

    acc = warp_sum(acc);
    if (active && lane == 0) {
        C[n] = __float2half(acc);
    }
}

template<int UNROLL_K>
__global__ void fp8_w8a16_gemv_coalesced_m_kernel(
        const __half*  __restrict__ A,       // [M, K]
        const uint8_t* __restrict__ W,       // [N, K], row-major
        const __half*  __restrict__ scales,  // [ceil(N/block_h), ceil(K/128)]
        __half*        __restrict__ C,       // [M, N]
        int M, int N, int K,
        int block_h,
        int Kb) {
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n       = blockIdx.x * GEMV_COAL_WARPS_PER_CTA + warp_id;
    const bool active = (n < N);

    __shared__ __half a_shared[GEMV_COAL_M_MAX][UNROLL_K][GEMV_COAL_BLOCK_K];
    float acc[GEMV_COAL_M_MAX];
    #pragma unroll
    for (int m = 0; m < GEMV_COAL_M_MAX; ++m) {
        acc[m] = 0.0f;
    }
    const int scale_row = active ? (n / block_h) : 0;

    for (int k_base = 0; k_base < K; k_base += UNROLL_K * GEMV_COAL_BLOCK_K) {
        for (int idx = tid; idx < GEMV_COAL_M_MAX * UNROLL_K * GEMV_COAL_BLOCK_K;
             idx += GEMV_COAL_THREADS) {
            const int k_in_block = idx % GEMV_COAL_BLOCK_K;
            const int u = (idx / GEMV_COAL_BLOCK_K) % UNROLL_K;
            const int m = idx / (UNROLL_K * GEMV_COAL_BLOCK_K);
            const int kk = k_base + u * GEMV_COAL_BLOCK_K + k_in_block;
            a_shared[m][u][k_in_block] = (m < M && kk < K)
                    ? A[(int64_t)m * K + kk]
                    : __float2half(0.f);
        }
        __syncthreads();

        if (active) {
            uint32_t wv[UNROLL_K];
            float scale_f[UNROLL_K];
            bool has_block[UNROLL_K];

            #pragma unroll
            for (int u = 0; u < UNROLL_K; ++u) {
                const int block_base = k_base + u * GEMV_COAL_BLOCK_K;
                has_block[u] = block_base < K;
                wv[u] = 0;
                scale_f[u] = 0.0f;
                if (has_block[u]) {
                    const int scale_col = block_base / GEMV_COAL_BLOCK_K;
                    scale_f[u] = __half2float(scales[scale_row * Kb + scale_col]);
                    const int k_lane = block_base + lane * GEMV_COAL_ELEMS_PER_LANE;
                    wv[u] = *reinterpret_cast<const uint32_t*>(&W[(int64_t)n * K + k_lane]);
                }
            }

            #pragma unroll
            for (int u = 0; u < UNROLL_K; ++u) {
                if (has_block[u]) {
                    const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv[u]);
                    #pragma unroll
                    for (int j = 0; j < GEMV_COAL_ELEMS_PER_LANE; ++j) {
                        const float w_f = __half2float(__ushort_as_half(
                            fp8_e4m3_to_fp16_bits(wb[j])));
                        const float w_scaled = w_f * scale_f[u];
                        #pragma unroll
                        for (int m = 0; m < GEMV_COAL_M_MAX; ++m) {
                            const float a_f = __half2float(
                                a_shared[m][u][lane * GEMV_COAL_ELEMS_PER_LANE + j]);
                            acc[m] += a_f * w_scaled;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < GEMV_COAL_M_MAX; ++m) {
        acc[m] = warp_sum(acc[m]);
    }
    if (active && lane == 0) {
        #pragma unroll
        for (int m = 0; m < GEMV_COAL_M_MAX; ++m) {
            if (m < M) {
                C[(int64_t)m * N + n] = __float2half(acc[m]);
            }
        }
    }
}

// ─── Stage G1: GROUPED coalesced routed GEMV (MoE w13 decode) ─────────────────
//
// The grouped routed decode kernel (fp8_w8a16_grouped_routed_gemm_a3) maps one
// THREAD to one output column n, so a warp reads W rows K bytes apart (NCU: 28
// sectors/request, 7.16% DRAM — the A.3 pathology). This flips it the same way
// the dense coalesced GEMV did: one WARP owns one (routed-row r, column n), lanes
// stride consecutive K, warp-reduce. Each routed row picks its expert via
// expert_ids[r]; the R dimension (gridDim.y) supplies occupancy even at the small
// MoE N (=352), so no k-slicing is needed. Channel scale (block_h=1) or block
// (block_h=128). M==1 per routed row (decode). Falls back to the grouped a3
// kernel for prefill (large R) via the Python gate.
template<int UNROLL_K>
__global__ void fp8_w8a16_grouped_gemv_coalesced_kernel(
        const __half*  __restrict__ A,           // [R, K] routed activations
        const int64_t* __restrict__ expert_ids,  // [R]
        const uint8_t* __restrict__ W,           // [E, N, K] fp8 bytes
        const __half*  __restrict__ scales,      // [E, Nb, Kb]
        __half*        __restrict__ C,           // [R, N]
        int R, int E, int N, int K, int block_h, int Nb, int Kb) {
    const int r       = blockIdx.y;              // uniform across CTA
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n       = blockIdx.x * GEMV_COAL_WARPS_PER_CTA + warp_id;

    const int64_t e64 = expert_ids[r];
    if (e64 < 0 || e64 >= E) {                   // invalid route -> zero the row (uniform)
        if (n < N && lane == 0) C[(int64_t)r * N + n] = __float2half(0.f);
        return;
    }
    const int     e      = (int)e64;
    const bool    active = (n < N);
    const int     scale_row = active ? (n / block_h) : 0;
    const int64_t w_base = (int64_t)e * N * K;
    const int64_t s_base = (int64_t)e * Nb * Kb;
    const int64_t a_base = (int64_t)r * K;

    __shared__ __half a_shared[UNROLL_K][GEMV_COAL_BLOCK_K];
    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += UNROLL_K * GEMV_COAL_BLOCK_K) {
        if (tid < GEMV_COAL_BLOCK_K) {
            #pragma unroll
            for (int u = 0; u < UNROLL_K; ++u) {
                const int kk = k_base + u * GEMV_COAL_BLOCK_K + tid;
                a_shared[u][tid] = (kk < K) ? A[a_base + kk] : __float2half(0.f);
            }
        }
        __syncthreads();

        if (active) {
            uint32_t wv[UNROLL_K];
            float    scale_f[UNROLL_K];
            bool     has_block[UNROLL_K];
            #pragma unroll
            for (int u = 0; u < UNROLL_K; ++u) {
                const int block_base = k_base + u * GEMV_COAL_BLOCK_K;
                has_block[u] = block_base < K;
                wv[u] = 0; scale_f[u] = 0.0f;
                if (has_block[u]) {
                    const int scale_col = block_base / GEMV_COAL_BLOCK_K;
                    scale_f[u] = __half2float(scales[s_base + scale_row * Kb + scale_col]);
                    const int k_lane = block_base + lane * GEMV_COAL_ELEMS_PER_LANE;
                    wv[u] = *reinterpret_cast<const uint32_t*>(
                        &W[w_base + (int64_t)n * K + k_lane]);
                }
            }
            #pragma unroll
            for (int u = 0; u < UNROLL_K; ++u) {
                if (has_block[u]) {
                    const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv[u]);
                    #pragma unroll
                    for (int j = 0; j < GEMV_COAL_ELEMS_PER_LANE; ++j) {
                        const float w_f = __half2float(__ushort_as_half(
                            fp8_e4m3_to_fp16_bits(wb[j])));
                        const float a_f = __half2float(
                            a_shared[u][lane * GEMV_COAL_ELEMS_PER_LANE + j]);
                        acc += a_f * w_f * scale_f[u];
                    }
                }
            }
        }
        __syncthreads();
    }

    acc = warp_sum(acc);
    if (active && lane == 0) C[(int64_t)r * N + n] = __float2half(acc);
}

static int gemv_coalesced_unroll_from_env() {
    const char* env = std::getenv("VLLM_V100_FP8_COALESCED_UNROLL");
    if (!env) {
        return GEMV_COAL_DEFAULT_UNROLL_K;
    }
    const int requested = std::atoi(env);
    if (requested >= 8) {
        return 8;
    }
    if (requested >= 4) {
        return 4;
    }
    return GEMV_COAL_DEFAULT_UNROLL_K;
}

static int gemv_coalesced_m_unroll_from_env() {
    const char* env = std::getenv("VLLM_V100_FP8_COALESCED_M_UNROLL");
    if (!env) {
        return gemv_coalesced_unroll_from_env();
    }
    const int requested = std::atoi(env);
    if (requested >= 8) {
        return 8;
    }
    if (requested >= 4) {
        return 4;
    }
    return GEMV_COAL_DEFAULT_UNROLL_K;
}

static void launch_fp8_w8a16_gemv_coalesced(
        const __half* A,
        const uint8_t* W,
        const __half* scales,
        __half* C,
        int N, int K, int block_h, int Kb,
        dim3 grid, dim3 block) {
    const int unroll = gemv_coalesced_unroll_from_env();
    if (unroll >= 8) {
        fp8_w8a16_gemv_coalesced_kernel<8><<<grid, block, 0, V100_FP8_STREAM>>>(
            A, W, scales, C, N, K, block_h, Kb);
    } else if (unroll >= 4) {
        fp8_w8a16_gemv_coalesced_kernel<4><<<grid, block, 0, V100_FP8_STREAM>>>(
            A, W, scales, C, N, K, block_h, Kb);
    } else {
        fp8_w8a16_gemv_coalesced_kernel<2><<<grid, block, 0, V100_FP8_STREAM>>>(
            A, W, scales, C, N, K, block_h, Kb);
    }
}

static void launch_fp8_w8a16_gemv_coalesced_m(
        const __half* A,
        const uint8_t* W,
        const __half* scales,
        __half* C,
        int M, int N, int K, int block_h, int Kb,
        dim3 grid, dim3 block) {
    const int unroll = gemv_coalesced_m_unroll_from_env();
    if (unroll >= 8) {
        fp8_w8a16_gemv_coalesced_m_kernel<8><<<grid, block, 0, V100_FP8_STREAM>>>(
            A, W, scales, C, M, N, K, block_h, Kb);
    } else if (unroll >= 4) {
        fp8_w8a16_gemv_coalesced_m_kernel<4><<<grid, block, 0, V100_FP8_STREAM>>>(
            A, W, scales, C, M, N, K, block_h, Kb);
    } else {
        fp8_w8a16_gemv_coalesced_m_kernel<2><<<grid, block, 0, V100_FP8_STREAM>>>(
            A, W, scales, C, M, N, K, block_h, Kb);
    }
}

static void launch_fp8_w8a16_grouped_gemv_coalesced(
        const __half* A, const int64_t* expert_ids, const uint8_t* W,
        const __half* scales, __half* C,
        int R, int E, int N, int K, int block_h, int Nb, int Kb,
        dim3 grid, dim3 block) {
    const int unroll = gemv_coalesced_unroll_from_env();
    if (unroll >= 8) {
        fp8_w8a16_grouped_gemv_coalesced_kernel<8><<<grid, block, 0, V100_FP8_STREAM>>>(
            A, expert_ids, W, scales, C, R, E, N, K, block_h, Nb, Kb);
    } else if (unroll >= 4) {
        fp8_w8a16_grouped_gemv_coalesced_kernel<4><<<grid, block, 0, V100_FP8_STREAM>>>(
            A, expert_ids, W, scales, C, R, E, N, K, block_h, Nb, Kb);
    } else {
        fp8_w8a16_grouped_gemv_coalesced_kernel<2><<<grid, block, 0, V100_FP8_STREAM>>>(
            A, expert_ids, W, scales, C, R, E, N, K, block_h, Nb, Kb);
    }
}

torch::Tensor fp8_w8a16_grouped_gemv_coalesced(
        torch::Tensor input,        // [R, K] fp16
        torch::Tensor expert_ids,   // [R] int64
        torch::Tensor weight,       // [E, N, K] uint8
        torch::Tensor scales,       // [E, Nb, Kb] fp16
        int64_t N, int64_t K, int64_t block_h, int64_t block_w) {
    TORCH_CHECK(input.is_cuda() && expert_ids.is_cuda() && weight.is_cuda() && scales.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(expert_ids.dtype() == torch::kInt64, "expert_ids must be int64");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "weight must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(input.is_contiguous() && expert_ids.is_contiguous()
                && weight.is_contiguous() && scales.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == K, "input must be [R, K]");
    TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.size(0) == input.size(0), "expert_ids [R]");
    TORCH_CHECK(weight.dim() == 3 && weight.size(1) == N && weight.size(2) == K, "weight [E,N,K]");
    TORCH_CHECK(block_w == GEMV_COAL_BLOCK_K, "block_w must be 128");
    TORCH_CHECK(K % GEMV_COAL_BLOCK_K == 0, "K must be divisible by 128");
    TORCH_CHECK(block_h == 1 || block_h == 128, "block_h must be 1 (channel) or 128 (block)");

    const int R = (int)input.size(0);
    const int E = (int)weight.size(0);
    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.numel() == (int64_t)E * Nb * Kb, "scales must be [E, Nb, Kb]");

    auto C = torch::empty({(int64_t)R, (int64_t)N},
                          torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));
    if (R == 0) return C;

    dim3 block(GEMV_COAL_THREADS);
    dim3 grid(((int)N + GEMV_COAL_WARPS_PER_CTA - 1) / GEMV_COAL_WARPS_PER_CTA, R);
    launch_fp8_w8a16_grouped_gemv_coalesced(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        expert_ids.data_ptr<int64_t>(),
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        R, E, (int)N, (int)K, (int)block_h, Nb, Kb, grid, block);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

torch::Tensor fp8_w8a16_gemv_coalesced_m(torch::Tensor input,
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
                "coalesced GEMV-M kernel requires input [M, K]");
    TORCH_CHECK(input.size(0) <= GEMV_COAL_M_MAX,
                "coalesced GEMV-M kernel is scoped to M<=8 decode");
    TORCH_CHECK(weight.numel() == N * K, "weight.numel() must equal N*K");
    TORCH_CHECK(block_w == GEMV_COAL_BLOCK_K,
                "coalesced GEMV-M kernel requires block_w=128");
    TORCH_CHECK(K % GEMV_COAL_BLOCK_K == 0,
                "coalesced GEMV-M kernel requires K divisible by 128");
    TORCH_CHECK(block_h > 0, "block_h must be positive");

    const int M = (int)input.size(0);
    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.numel() == (int64_t)Nb * Kb,
                "scales.numel() must equal ceil(N/block_h) * ceil(K/block_w)");

    auto C = torch::empty({M, N},
                          torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));

    dim3 block(GEMV_COAL_THREADS);
    dim3 grid(((int)N + GEMV_COAL_WARPS_PER_CTA - 1) / GEMV_COAL_WARPS_PER_CTA);

    launch_fp8_w8a16_gemv_coalesced_m(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, (int)N, (int)K, (int)block_h, Kb, grid, block);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
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
    // Partial-CTA safety (same fix as A.1/A.2/A.3): r (blockIdx.y) is uniform
    // across the CTA, and the expert id derives from r, so both early-returns
    // below are non-divergent and safe. n>=N IS divergent (partial last N-tile),
    // so we mask it instead of returning — every lane must reach the tid-indexed
    // a_shared load + both __syncthreads() (else surviving lanes read garbage,
    // the non-128-aligned-N bug; confirmed on GLM expert w13 N=352/2736).
    if (r >= R) return;

    const int64_t expert64 = expert_ids[r];
    if (expert64 < 0 || expert64 >= E) return;
    const int expert = (int)expert64;
    const bool active = (n < N);

    const int scale_row = active ? (n / block_h) : 0;
    const int k_start   = slice_id * k_slice_size;
    const int k_end     = min(k_start + k_slice_size, K);
    const int64_t w_base = ((int64_t)expert * N + n) * K;   // read only when active
    const int64_t s_base = ((int64_t)expert * Nb + scale_row) * Kb;

    __shared__ __half a_shared[BLOCK_K_A3];
    float acc = 0.0f;

    for (int k_base = k_start; k_base < k_end; k_base += BLOCK_K_A3) {
        const int k_load = k_base + tid;
        a_shared[tid] = (k_load < K) ? A[r * K + k_load] : __float2half(0.f);
        __syncthreads();

        if (active) {
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
        }
        __syncthreads();
    }

    if (active) atomicAdd(&C_fp32[r * N + n], acc);
}

// Grouped routed GEMM for already-dequantized FP16 expert weights.
//
// Layouts:
//   A:          [R, K] FP16 routed activations
//   expert_ids: [R]    local expert id for each routed row
//   W:          [E, N, K] FP16, contiguous
//   C:          [R, N]
//
// This is intentionally shaped like the FP8 grouped A.3 path above so CT-MoE can
// remove the per-expert Python loop for GLM w2, whose K is small (e.g. 176) and
// not block-FP8 aligned. Accumulate in FP32 and cast once, matching torch matmul
// closely enough for the existing mixed-path tolerance.
__global__ void fp16_grouped_routed_gemm_kernel(
        const __half*  __restrict__ A,
        const int64_t* __restrict__ expert_ids,
        const __half*  __restrict__ W,
        float*         __restrict__ C_fp32,
        int R, int E, int N, int K,
        int k_slice_size) {
    const int n_base   = blockIdx.x * BLOCK_N_A3;
    const int r        = blockIdx.y;
    const int slice_id = blockIdx.z;
    const int tid      = threadIdx.x;
    const int n        = n_base + tid;
    if (r >= R) return;

    const int64_t expert64 = expert_ids[r];
    if (expert64 < 0 || expert64 >= E) return;
    const int expert = (int)expert64;
    const bool active = (n < N);

    const int k_start = slice_id * k_slice_size;
    const int k_end   = min(k_start + k_slice_size, K);
    const int64_t w_base = ((int64_t)expert * N + n) * K;

    __shared__ __half a_shared[BLOCK_K_A3];
    float acc = 0.0f;

    for (int k_base = k_start; k_base < k_end; k_base += BLOCK_K_A3) {
        const int k_load = k_base + tid;
        a_shared[tid] = (k_load < k_end) ? A[(int64_t)r * K + k_load]
                                         : __float2half(0.f);
        __syncthreads();

        if (active) {
            const int kk_count = min(BLOCK_K_A3, k_end - k_base);
            for (int kk = 0; kk < kk_count; ++kk) {
                const __half a = a_shared[kk];
                const __half w = W[w_base + k_base + kk];
                acc += __half2float(a) * __half2float(w);
            }
        }
        __syncthreads();
    }

    if (active) atomicAdd(&C_fp32[(int64_t)r * N + n], acc);
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
        // W + dequant + scale. block_h==1 => CHANNEL scale: per-output-row n
        // (scale_row = n_start+row), NOT the per-128-block n_start/block_h. The
        // scale is baked into B_tile per row, so C[m,n] = sum_k A[m,k]W[n,k]*scale[n]
        // = channel scale exactly. block_h==128 keeps the original block behavior.
        const int   scale_col_l = 0 / block_w;
        const int   sr          = (block_h == 1) ? (n_start + row) : scale_row;
        const float scale_f_l   = __half2float(scales[sr * Kb + scale_col_l]);
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
            const int   sr          = (block_h == 1) ? (n_start + row) : scale_row;
            const float scale_f_l   = __half2float(scales[sr * Kb + scale_col_l]);
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
    TORCH_CHECK((block_h == 128 || block_h == 1) && block_w == 128,
                "WMMA requires block_w=128 and block_h in {128 (block scale), "
                "1 (channel scale: per-output-row)}");

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

torch::Tensor fp8_w8a16_gemv_coalesced(torch::Tensor input,
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
    TORCH_CHECK(input.dim() == 2 && input.size(0) == 1 && input.size(1) == K,
                "coalesced GEMV proof kernel requires input [1, K]");
    TORCH_CHECK(weight.numel() == N * K, "weight.numel() must equal N*K");
    TORCH_CHECK(block_w == GEMV_COAL_BLOCK_K,
                "coalesced GEMV proof kernel requires block_w=128");
    TORCH_CHECK(K % GEMV_COAL_BLOCK_K == 0,
                "coalesced GEMV proof kernel requires K divisible by 128");
    TORCH_CHECK(block_h > 0, "block_h must be positive");

    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.numel() == (int64_t)Nb * Kb,
                "scales.numel() must equal ceil(N/block_h) * ceil(K/block_w)");

    auto C = torch::empty({1, N},
                          torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));

    dim3 block(GEMV_COAL_THREADS);
    dim3 grid(((int)N + GEMV_COAL_WARPS_PER_CTA - 1) / GEMV_COAL_WARPS_PER_CTA);

    launch_fp8_w8a16_gemv_coalesced(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        (int)N, (int)K, (int)block_h, Kb, grid, block);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
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

torch::Tensor fp16_grouped_routed_gemm(torch::Tensor input,
                                       torch::Tensor expert_ids,
                                       torch::Tensor weight,
                                       int64_t       k_split) {
    TORCH_CHECK(input.is_cuda() && expert_ids.is_cuda() && weight.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(expert_ids.dtype() == torch::kInt64, "expert_ids must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "weight must be float16");
    TORCH_CHECK(input.is_contiguous() && expert_ids.is_contiguous() &&
                weight.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be [R, K]");
    TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.size(0) == input.size(0),
                "expert_ids must be [R]");
    TORCH_CHECK(weight.dim() == 3, "weight must be [E, N, K]");
    TORCH_CHECK(weight.size(2) == input.size(1),
                "weight K must match input K");
    TORCH_CHECK(k_split >= 1, "k_split must be >= 1");
    TORCH_CHECK(input.size(1) % k_split == 0,
                "K must be divisible by k_split");

    const int R = (int)input.size(0);
    const int E = (int)weight.size(0);
    const int N = (int)weight.size(1);
    const int K = (int)input.size(1);
    const int k_slice_size = K / (int)k_split;
    auto C_fp32 = torch::zeros({(int64_t)R, (int64_t)N},
                                torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    if (R == 0) {
        return C_fp32.to(torch::kFloat16);
    }

    dim3 block(BLOCK_N_A3);
    dim3 grid((N + BLOCK_N_A3 - 1) / BLOCK_N_A3, R, (int)k_split);

    fp16_grouped_routed_gemm_kernel<<<grid, block, 0, V100_FP8_STREAM>>>(
        reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
        expert_ids.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(weight.data_ptr<at::Half>()),
        C_fp32.data_ptr<float>(),
        R, E, N, K, k_slice_size);
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

// Phase 4 Stage 1.5 host wrapper. Caller passes A pre-sorted by expert and the
// GPU-side route layout (all sync-free to build: scatter_add counts, cumsums).
// grid.y is bounded host-side by ceil(R/BM)+E (worst-case padding tiles); extra
// tiles self-exit, so NO total_tiles sync is needed.
torch::Tensor fp8_w8a16_grouped_tiled_gemm(
        torch::Tensor A,                 // [R, K] fp16, sorted by expert
        torch::Tensor expert_tile_off,   // [E] int32
        torch::Tensor tiles_per_e,       // [E] int32
        torch::Tensor expert_row_off,    // [E] int32
        torch::Tensor counts,            // [E] int32
        torch::Tensor weight,            // [E, N, K] uint8
        torch::Tensor scales,            // [E, Nb, Kb] fp16
        int64_t N, int64_t K,
        int64_t block_h, int64_t block_w) {
    TORCH_CHECK(A.is_cuda() && weight.is_cuda() && scales.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "weight must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(A.is_contiguous() && weight.is_contiguous() && scales.is_contiguous(),
                "A/weight/scales must be contiguous");
    TORCH_CHECK(A.dim() == 2 && A.size(1) == K, "A must be [R, K]");
    TORCH_CHECK(weight.dim() == 3 && weight.size(1) == N && weight.size(2) == K,
                "weight must be [E, N, K]");
    TORCH_CHECK(K % K_VEC_A2 == 0, "K must be divisible by 16");
    TORCH_CHECK(block_w == 128 && (block_h == 1 || block_h == 128),
                "block_w=128 and block_h in {1,128}");

    const int R = (int)A.size(0);
    const int E = (int)weight.size(0);
    for (auto& t : {expert_tile_off, tiles_per_e, expert_row_off, counts})
        TORCH_CHECK(t.dtype() == torch::kInt32 && t.is_cuda() && t.is_contiguous()
                    && t.numel() == E,
                    "route-layout tensors must be int32 CUDA contiguous [E]");
    const int Nb = (int)((N + block_h - 1) / block_h);
    const int Kb = (int)((K + block_w - 1) / block_w);
    TORCH_CHECK(scales.numel() == (int64_t)E * Nb * Kb, "scales must be [E, Nb, Kb]");

    auto C = torch::empty({(int64_t)R, (int64_t)N},
                          torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));
    if (R == 0) return C;

    const int max_tiles = (R + BLOCK_M_A2 - 1) / BLOCK_M_A2 + E;   // host bound, no sync
    dim3 block(BLOCK_N_A2);
    dim3 grid(((int)N + BLOCK_N_A2 - 1) / BLOCK_N_A2, max_tiles);
    fp8_w8a16_grouped_tiled_gemm_kernel<<<grid, block, 0, V100_FP8_STREAM>>>(
        reinterpret_cast<__half*>(A.data_ptr<at::Half>()),
        expert_tile_off.data_ptr<int>(), tiles_per_e.data_ptr<int>(),
        expert_row_off.data_ptr<int>(), counts.data_ptr<int>(),
        weight.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        R, E, (int)N, (int)K, (int)block_h, (int)block_w, Nb, Kb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
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
    m.def("fp8_w8a16_gemv_coalesced",      &fp8_w8a16_gemv_coalesced,
          "Prototype M=1 FP8 W8A16 GEMV with coalesced K-lane weight loads. "
          "Separate proof kernel; block_w=128 only.");
    m.def("fp8_w8a16_gemv_coalesced_m",    &fp8_w8a16_gemv_coalesced_m,
          "Prototype M<=8 FP8 W8A16 GEMV with coalesced K-lane weight loads. "
          "Separate proof kernel; block_w=128 only.");
    m.def("fp8_w8a16_grouped_gemv_coalesced", &fp8_w8a16_grouped_gemv_coalesced,
          "Stage G1: GROUPED coalesced routed GEMV for MoE w13 decode. Warp owns "
          "one (routed-row, column); lanes coalesce over K; expert via "
          "expert_ids[R]. Fixes the grouped-a3 N-strided read (28 sectors/req).");
    m.def("fp8_w8a16_grouped_routed_gemm_a3",
          &fp8_w8a16_grouped_routed_gemm_a3,
          "Stage 2B grouped routed A.3 GEMM for MoE decode. Each routed row "
          "selects its local expert via expert_ids[R], reducing per-expert "
          "Python launch fanout.");
    m.def("fp16_grouped_routed_gemm",
          &fp16_grouped_routed_gemm,
          "Grouped routed GEMM for FP16 expert weights. Each routed row "
          "selects its local expert via expert_ids[R].");
    m.def("fp8_w8a16_grouped_tiled_gemm",
          &fp8_w8a16_grouped_tiled_gemm,
          "Phase 4 Stage 1.5: fused grouped-tiled FP8 GEMM. ONE launch for all "
          "routed rows (A pre-sorted by expert) using GPU-side per-expert tile "
          "offsets — removes the per-expert Python loop + .tolist() syncs.");
    m.def("fp8_w8a16_gemm_wmma_poc",       &fp8_w8a16_gemm_wmma_poc,
          "Phase A.4 POC: WMMA-based GEMM on V100 HMMA.884 tensor cores. "
          "Naive 64×64 tile, 4 warps, no double-buffering. POC to gauge "
          "whether full WMMA optimization is worth pursuing.");
}
