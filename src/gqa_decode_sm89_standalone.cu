/*
 * FlashInfer GQA Decode Standalone Implementation
 * Supports: SM89 (Ada/RTX 40xx) and above
 *
 * GQA (Grouped Query Attention) Decode with Paged KV Cache:
 * - Multiple query heads share each KV head (GQA)
 * - Paged KV cache for efficient memory management
 * - FP8 KV cache support (E4M3 format)
 * - Llama-style RoPE support
 *
 * Based on FlashInfer's implementation:
 * - flashinfer/include/flashinfer/attention/decode.cuh
 * - flashinfer/flashinfer/decode.py
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <tuple>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// ============================================================================
// FP8 Utilities (E4M3 format for KV cache)
// ============================================================================

// FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
// Range: [-448, 448], approximately 0.06% precision

// Raw conversion from FP8 bits to float (bypasses CUDA's built-in conversion)
__device__ __forceinline__ float fp8_e4m3_to_float_raw(uint8_t fp8_bits) {
    unsigned int sign = (fp8_bits >> 7) & 0x1;
    unsigned int exp = (fp8_bits >> 3) & 0xF;
    unsigned int mant = fp8_bits & 0x7;

    if (exp == 0) {
        // Zero or subnormal (treat as zero for simplicity)
        return 0.0f;
    }

    if (exp == 15) {
        // Saturated value: exp=15 indicates overflow during quantization
        // Host uses clamp value of 448, so we return the same here
        float abs_max = 448.0f;
        return sign ? -abs_max : abs_max;
    }

    // FP8 E4M3 exponent bias is 7, FP32 exponent bias is 127
    int new_exp = static_cast<int>(exp) - 7 + 127;

    // Construct FP32 value
    unsigned int result_bits = (sign << 31) | (new_exp << 23) | (mant << 20);
    return __uint_as_float(result_bits);
}

// Direct conversion from uint8_t (for FP8 stored as raw bytes)
__device__ __forceinline__ float fp8_to_float(uint8_t fp8_bits) {
    return fp8_e4m3_to_float_raw(fp8_bits);
}

// Legacy overload for __nv_fp8_e4m3 (for compatibility, but not recommended)
__device__ __forceinline__ float fp8_to_float(const __nv_fp8_e4m3& fp8_val) {
    // This is still needed for places where __nv_fp8_e4m3 is used directly
    uint8_t fp8_bits;
    memcpy(&fp8_bits, &fp8_val, sizeof(uint8_t));
    return fp8_e4m3_to_float_raw(fp8_bits);
}

template <typename T>
__device__ __forceinline__ float fp8_to_float(T fp8_val) {
    if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
        uint8_t bits = reinterpret_cast<const uint8_t&>(fp8_val);
        unsigned int sign = (bits >> 7) & 0x1;
        unsigned int exp = (bits >> 2) & 0x1F;
        unsigned int mant = bits & 0x3;

        if (exp == 0) {
            return 0.0f;
        }

        // FP8 E5M2 exponent bias is 15, FP32 exponent bias is 127
        int new_exp = static_cast<int>(exp) - 15 + 127;

        unsigned int result_bits = (sign << 31) | (new_exp << 23) | (mant << 21);
        return __uint_as_float(result_bits);
    } else {
        return static_cast<float>(fp8_val);
    }
}

template <typename T>
__device__ __forceinline__ __nv_fp8_e4m3 float_to_fp8_e4m3(float val) {
    // Simplified FP8 E4M3 conversion for CUDA 12.2
    // Clamp to FP8 E4M3 range
    if (val > 448.0f) val = 448.0f;
    if (val < -448.0f) val = -448.0f;

    // Use CUDA's built-in conversion if available
    #if defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4))
        return __float2_fp8_e4m3(val);
    #else
        // Use round-trip conversion through fp16 (available on SM89+)
        // This is less accurate but more reliable than manual implementation
        __half fp16_val = __float2half(val);
        // Manual FP16 to FP8 E4M3 conversion
        uint16_t fp16_bits = __half_as_ushort(fp16_val);
        unsigned int sign = (fp16_bits >> 15) & 0x1;
        unsigned int exp = ((fp16_bits >> 10) & 0x1F);
        unsigned int mant = fp16_bits & 0x3FF;

        if (exp == 0) {
            return __nv_fp8_e4m3{0};
        }
        if (exp == 31) {
            return __nv_fp8_e4m3{static_cast<uint8_t>(sign ? 0x80 : 0x7F)};
        }

        // FP16 exponent bias is 15, FP8 E4M3 exponent bias is 7
        int new_exp = static_cast<int>(exp) - 15 + 7;

        if (new_exp >= 15) {
            // Overflow: saturate to max value
            // 0x78 = 0 1111 1000 = max positive (approx 448)
            // 0xF8 = 1 1111 1000 = max negative (approx -448)
            // Use exp=14, mant=7 for max positive
            // For negative, use same pattern
            return __nv_fp8_e4m3{static_cast<uint8_t>((sign << 7) | 0x78)};
        }
        if (new_exp <= 0) {
            return __nv_fp8_e4m3{0};
        }

        // Round mantissa from 10 bits to 3 bits
        unsigned int new_mant = (mant >> 7) & 0x7;
        if ((mant & 0x7F) >= 0x40) {
            new_mant++;
            if (new_mant >= 8) {
                new_mant = 0;
                new_exp++;
            }
        }

        uint8_t result = static_cast<uint8_t>((sign << 7) | (new_exp << 3) | new_mant);
        return __nv_fp8_e4m3{result};
    #endif
}

// Host-side FP8 E4M3 conversion (for test data preparation)
// Proper implementation following FP8 E4M3 format specification
// Returns uint8_t directly to bypass CUDA's built-in __nv_fp8_e4m3 conversion
inline uint8_t float_to_fp8_e4m3_host(float val) {
    // Handle special cases
    if (val == 0.0f || fabsf(val) < 1e-9f) {
        return 0;
    }

    // Clamp to FP8 E4M3 range
    if (val > 448.0f) return 0x7F;  // max positive (Inf-like)
    if (val < -448.0f) return 0x80;  // min negative (negative zero-like)

    // Extract components
    unsigned int bits;
    memcpy(&bits, &val, sizeof(float));

    uint8_t sign = (bits >> 31) & 0x1;
    uint8_t exp = ((bits >> 23) & 0xFF);
    uint32_t mant = bits & 0x7FFFFF;

    // Handle subnormal numbers
    if (exp == 0) {
        return 0;
    }
    // Handle Inf/NaN
    if (exp == 255) {
        return static_cast<uint8_t>(sign ? 0x80 : 0x7F);
    }

    // Convert from FP32 to FP8 E4M3
    // FP32: 1 sign, 8 exponent (bias 127), 23 mantissa
    // FP8:  1 sign, 4 exponent (bias 7),   3 mantissa
    int8_t new_exp = static_cast<int8_t>(exp) - 127 + 7;

    // Handle overflow/underflow
    if (new_exp >= 15) {
        // Saturate to max value using exp=15 (reserved for overflow/Inf)
        // Dequantization will detect exp=15 and return ±448.0
        return static_cast<uint8_t>((sign << 7) | (15 << 3));
    }
    if (new_exp <= 0) {
        // Map to subnormal or zero
        // For simplicity, return zero
        return 0;
    }

    // Round mantissa from 23 bits to 3 bits
    uint8_t new_mant = (mant >> 20) & 0x7;
    // Round half to even
    uint32_t mant_fraction = mant & 0xFFFFF;  // lower 20 bits
    if (mant_fraction > 0x80000 || (mant_fraction == 0x80000 && new_mant % 2 == 1)) {
        new_mant++;
        if (new_mant >= 8) {
            new_mant = 0;
            new_exp++;
        }
    }

    // Handle overflow from rounding
    if (new_exp >= 15) {
        // Saturate to max value using exp=15
        return static_cast<uint8_t>((sign << 7) | (15 << 3));
    }

    return static_cast<uint8_t>((sign << 7) | (new_exp << 3) | new_mant);
}

// Per-channel quantization: each head dimension has its own scale
// Dequantize: fp32_value = fp8_value * scale
template <typename FP8T>
__device__ __forceinline__ void fp8_dequantize(
    const FP8T* fp8_data,
    float* float_data,
    int n,
    const float* scale
) {
    int tx = threadIdx.x;
    float s = scale[0];  // Per-tensor scale (could be extended to per-channel)

    for (int i = tx; i < n; i += blockDim.x) {
        float_data[i] = fp8_to_float(fp8_data[i]) * s;
    }
}

// Quantize with scaling: fp8_value = fp32_value / scale
template <typename FP8T>
__device__ __forceinline__ void fp8_quantize(
    const float* float_data,
    FP8T* fp8_data,
    int n,
    const float* scale
) {
    int tx = threadIdx.x;
    float s = scale[0];

    for (int i = tx; i < n; i += blockDim.x) {
        fp8_data[i] = float_to_fp8_e4m3<FP8T>(float_data[i] / s);
    }
}

// Quantize with scaling (host side)
template <typename FP8T>
void fp8_quantize_host(
    const float* float_data,
    FP8T* fp8_data,
    int n,
    float scale
) {
    for (int i = 0; i < n; ++i) {
        fp8_data[i] = float_to_fp8_e4m3_host(float_data[i] / scale);
    }
}

// Dequantize with scaling (host side)
template <typename FP8T>
void fp8_dequantize_host(
    const FP8T* fp8_data,
    float* float_data,
    int n,
    float scale
) {
    for (int i = 0; i < n; ++i) {
        // Use CUDA runtime for host-side conversion
        float_data[i] = __nv_fp8_e4m3_to_fp32(fp8_data[i]) * scale;
    }
}

// Host-side FP8 E4M3 dequantization (for uint8_t storage)
// Same as device-side fp8_e4m3_to_float_raw but for host
inline float fp8_e4m3_to_float_host(uint8_t fp8_bits) {
    unsigned int sign = (fp8_bits >> 7) & 0x1;
    unsigned int exp = (fp8_bits >> 3) & 0xF;
    unsigned int mant = fp8_bits & 0x7;

    if (exp == 0) {
        return 0.0f;
    }

    if (exp == 15) {
        // Saturated value
        float abs_max = 448.0f;
        return sign ? -abs_max : abs_max;
    }

    // FP8 E4M3 exponent bias is 7, FP32 exponent bias is 127
    int new_exp = static_cast<int>(exp) - 7 + 127;
    unsigned int result_bits = (sign << 31) | (new_exp << 23) | (mant << 20);

    float result;
    memcpy(&result, &result_bits, sizeof(float));
    return result;
}

// Specialization for uint8_t FP8 storage
inline void fp8_dequantize_host(
    const uint8_t* fp8_data,
    float* float_data,
    int n,
    float scale
) {
    for (int i = 0; i < n; ++i) {
        float_data[i] = fp8_e4m3_to_float_host(fp8_data[i]) * scale;
    }
}

// Compute quantization scale (max abs value)
float compute_quant_scale(const float* data, int n, float max_fp8 = 448.0f) {
    float max_val = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_val = fmaxf(max_val, fabsf(data[i]));
    }
    // Add small epsilon to avoid division by zero
    return fmaxf(max_val / max_fp8, 1e-6f);
}

// ============================================================================
// Vector types and utilities
// ============================================================================

template <typename T, int n>
struct vec_t {
    T data[n];

    __device__ __forceinline__ T& operator[](int i) { return data[i]; }
    __device__ __forceinline__ const T& operator[](int i) const { return data[i]; }
};

// Specializations for CUDA vector types
template <>
struct vec_t<float, 1> {
    float data[1];
    __device__ __forceinline__ void load(const float* ptr) { data[0] = ptr[0]; }
    __device__ __forceinline__ void store(float* ptr) const { ptr[0] = data[0]; }
};

template <>
struct vec_t<float, 2> {
    float2 data;
    __device__ __forceinline__ void load(const float* ptr) { data = *reinterpret_cast<const float2*>(ptr); }
    __device__ __forceinline__ void store(float* ptr) const { *reinterpret_cast<float2*>(ptr) = data; }
};

template <>
struct vec_t<float, 4> {
    float4 data;
    __device__ __forceinline__ void load(const float* ptr) { data = *reinterpret_cast<const float4*>(ptr); }
    __device__ __forceinline__ void store(float* ptr) const { *reinterpret_cast<float4*>(ptr) = data; }
};

// FP16 vector conversions
template <>
struct vec_t<__half, 1> {
    __half data[1];
    __device__ __forceinline__ float to_float(int i) const { return __half2float(data[i]); }
    __device__ __forceinline__ void from_float(int i, float val) { data[i] = __float2half(val); }
};

template <>
struct vec_t<__half, 2> {
    __half2 data;
    __device__ __forceinline__ float to_float(int i) const {
        float2 f2 = __half22float2(data);
        return i == 0 ? f2.x : f2.y;
    }
    __device__ __forceinline__ void from_float(int i, float val) {
        float2 f2 = __half22float2(data);
        if (i == 0) f2.x = val; else f2.y = val;
        data = __floats2half2_rn(f2.x, f2.y);
    }
};

template <>
struct vec_t<__half, 4> {
    uint4 data;  // 4 FP16 = 8 bytes = 64 bits, stored in uint4 (128 bits) for alignment
    __device__ __forceinline__ float to_float(int i) const {
        const __half* h = reinterpret_cast<const __half*>(&data);
        return __half2float(h[i]);
    }
    __device__ __forceinline__ void from_float(int i, float val) {
        __half* h = reinterpret_cast<__half*>(&data);
        h[i] = __float2half(val);
    }
};

// BF16 vector conversions
template <>
struct vec_t<__nv_bfloat16, 1> {
    __nv_bfloat16 data[1];
    __device__ __forceinline__ float to_float(int i) const { return __bfloat162float(data[i]); }
    __device__ __forceinline__ void from_float(int i, float val) { data[i] = __float2bfloat16(val); }
};

template <>
struct vec_t<__nv_bfloat16, 2> {
    __nv_bfloat162 data;
    __device__ __forceinline__ float to_float(int i) const {
        float2 f2 = __bfloat1622float2(data);
        return i == 0 ? f2.x : f2.y;
    }
    __device__ __forceinline__ void from_float(int i, float val) {
        float2 f2 = __bfloat1622float2(data);
        if (i == 0) f2.x = val; else f2.y = val;
        data = __floats2bfloat162_rn(f2.x, f2.y);
    }
};

template <>
struct vec_t<__nv_bfloat16, 4> {
    uint4 data;
    __device__ __forceinline__ float to_float(int i) const {
        const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&data);
        return __bfloat162float(bf[i]);
    }
    __device__ __forceinline__ void from_float(int i, float val) {
        __nv_bfloat16* bf = reinterpret_cast<__nv_bfloat16*>(&data);
        bf[i] = __float2bfloat16(val);
    }
};

// ============================================================================
// Warp-level reductions
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// ============================================================================
// Attention state for online softmax
// ============================================================================

struct AttentionState {
    float m;  // Running maximum
    float d;  // Running normalizer
    float o[128];  // Output (max head_dim = 128)

    __device__ __forceinline__ void init() {
        m = -INFINITY;
        d = 0.0f;
        for (int i = 0; i < 128; ++i) {
            o[i] = 0.0f;
        }
    }

    __device__ __forceinline__ void update(float* scores, float* v_tile, int tile_size, int head_dim) {
        float m_prev = m;

        // Find new maximum
        for (int i = 0; i < tile_size; ++i) {
            m = fmaxf(m, scores[i]);
        }

        // Update normalizer (online softmax)
        float scale = expf(m_prev - m);
        d *= scale;
        for (int i = 0; i < tile_size; ++i) {
            d += expf(scores[i] - m);
        }

        // Update output
        for (int i = 0; i < head_dim; ++i) {
            o[i] *= scale;
        }
        for (int j = 0; j < tile_size; ++j) {
            float weight = expf(scores[j] - m);
            for (int i = 0; i < head_dim; ++i) {
                o[i] += weight * v_tile[j * head_dim + i];
            }
        }
    }
};

// ============================================================================
// RoPE (Rotary Position Embedding) - Llama style
// ============================================================================

template <typename T, int vec_size>
__device__ __forceinline__ void apply_llama_rope(
    const T* q_ptr,
    float* q_vec,
    float* freq,
    int rope_offset,
    int head_dim
) {
    // Load and apply RoPE
    #pragma unroll
    for (int i = 0; i < vec_size; ++i) {
        float val;

        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(q_ptr[i]);
        } else {
            val = __bfloat162float(q_ptr[i]);
        }

        int dim = i;  // Simplified: assume vec_size <= head_dim/2
        if (dim < head_dim / 2) {
            float freq_val = freq[dim];
            float cos_val = cosf(rope_offset * freq_val);
            float sin_val = sinf(rope_offset * freq_val);

            // Pair with dim + head_dim/2
            float val2;

            if constexpr (std::is_same_v<T, __half>) {
                val2 = __half2float(q_ptr[i + vec_size]);
            } else {
                val2 = __bfloat162float(q_ptr[i + vec_size]);
            }

            q_vec[i] = val * cos_val - val2 * sin_val;
            q_vec[i + vec_size] = val * sin_val + val2 * cos_val;
        } else {
            q_vec[i] = val;
        }
    }
}

template <typename T>
__device__ __forceinline__ void compute_rope_freq(float* freq, int head_dim, float theta = 10000.0f) {
    for (int i = 0; i < head_dim / 2; ++i) {
        freq[i] = 1.0f / powf(theta, float(2 * i) / float(head_dim));
    }
}

// ============================================================================
// Version 1: Simple GQA Decode (no paged KV cache, single batch)
// ============================================================================

template <typename T>
__global__ void gqa_decode_simple_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    T* __restrict__ output,
    float sm_scale,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len
) {
    // Compute group size (number of QO heads per KV head)
    const int group_size = num_qo_heads / num_kv_heads;

    // Block indices
    const int kv_head_idx = blockIdx.y;  // KV head index
    const int qo_head_idx = blockIdx.x;  // QO head index (direct mapping)

    // Only process first num_qo_heads blocks
    if (qo_head_idx >= num_qo_heads) return;

    // Find which KV head this QO head belongs to
    const int kv_head_for_qo = qo_head_idx / group_size;
    if (kv_head_for_qo != kv_head_idx) return;

    // Thread index
    const int tx = threadIdx.x;

    // Shared memory for K/V tiles
    __shared__ float k_smem[16 * 128];  // tile_size * head_dim
    __shared__ float v_smem[16 * 128];

    // Load query vector (tx=0 does the computation, so let it load everything)
    float q_vec[128];
    const T* q_ptr = q + qo_head_idx * head_dim;

    if (tx == 0) {
        for (int i = 0; i < head_dim; ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                q_vec[i] = __half2float(q_ptr[i]);
            } else {
                q_vec[i] = __bfloat162float(q_ptr[i]);
            }
        }
    }
    __syncthreads();

    // Initialize attention state
    AttentionState state;
    if (tx == 0) {
        state.init();
    }
    __syncthreads();

    // Process KV cache in tiles
    constexpr int tile_size = 16;
    const int num_tiles = (kv_len + tile_size - 1) / tile_size;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int kv_start = tile * tile_size;
        int kv_end = min(kv_start + tile_size, kv_len);
        int current_tile_size = kv_end - kv_start;

        // Load K tile cooperatively
        for (int j = 0; j < current_tile_size; ++j) {
            for (int i = tx; i < head_dim; i += blockDim.x) {
                const T* k_ptr = k_cache + (kv_start + j) * num_kv_heads * head_dim + kv_head_idx * head_dim + i;
                float k_val;

                if constexpr (std::is_same_v<T, __half>) {
                    k_val = __half2float(*k_ptr);
                } else {
                    k_val = __bfloat162float(*k_ptr);
                }

                k_smem[j * head_dim + i] = k_val;
            }
        }
        __syncthreads();

        // Compute QK scores (only tx=0 needs to compute)
        float scores[16];
        if (tx == 0) {
            for (int j = 0; j < current_tile_size; ++j) {
                float score = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    score += q_vec[i] * k_smem[j * head_dim + i];
                }
                scores[j] = score * sm_scale;
            }
        }
        __syncthreads();

        // Load V tile cooperatively
        for (int j = 0; j < current_tile_size; ++j) {
            for (int i = tx; i < head_dim; i += blockDim.x) {
                const T* v_ptr = v_cache + (kv_start + j) * num_kv_heads * head_dim + kv_head_idx * head_dim + i;
                if constexpr (std::is_same_v<T, __half>) {
                    v_smem[j * head_dim + i] = __half2float(*v_ptr);
                } else {
                    v_smem[j * head_dim + i] = __bfloat162float(*v_ptr);
                }
            }
        }
        __syncthreads();

        // Update attention state (only tx=0)
        if (tx == 0) {
            state.update(scores, v_smem, current_tile_size, head_dim);
        }
        __syncthreads();
    }

    // Write output (only tx=0)
    if (tx == 0) {
        T* out_ptr = output + qo_head_idx * head_dim;
        float d_safe = fmaxf(state.d, 1e-6f);  // Prevent division by zero
        for (int i = 0; i < head_dim; ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                out_ptr[i] = __float2half(state.o[i] / d_safe);
            } else {
                out_ptr[i] = __float2bfloat16(state.o[i] / d_safe);
            }
        }
    }
}

// ============================================================================
// Version 2: GQA Decode with Paged KV Cache
// ============================================================================

template <typename T>
__global__ void gqa_decode_paged_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_data,
    const T* __restrict__ v_data,
    const int* __restrict__ page_indices,
    const int* __restrict__ page_indptr,
    T* __restrict__ output,
    float sm_scale,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int batch_idx,
    int kv_len  // Actual sequence length
) {
    // Compute group size
    const int group_size = num_qo_heads / num_kv_heads;

    // Block indices - similar to simple kernel
    const int kv_head_idx = blockIdx.y;
    const int qo_head_idx = blockIdx.x;

    // Only process first num_qo_heads blocks
    if (qo_head_idx >= num_qo_heads) return;

    // Find which KV head this QO head belongs to
    const int kv_head_for_qo = qo_head_idx / group_size;
    if (kv_head_for_qo != kv_head_idx) return;

    // Thread index
    const int tx = threadIdx.x;

    // Get page start for this batch
    int page_start = page_indptr[batch_idx];

    if (kv_len <= 0) return;

    // Shared memory
    __shared__ float k_smem[128 * 16];  // head_dim * tile_size
    __shared__ float v_smem[128 * 16];

    // Load query vector
    float q_vec[128];
    const T* q_ptr = q + batch_idx * num_qo_heads * head_dim + qo_head_idx * head_dim;

    if (tx == 0) {
        for (int i = 0; i < head_dim; ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                q_vec[i] = __half2float(q_ptr[i]);
            } else {
                q_vec[i] = __bfloat162float(q_ptr[i]);
            }
        }
    }
    __syncthreads();

    // Initialize attention state
    AttentionState state;
    if (tx == 0) {
        state.init();
    }
    __syncthreads();

    // Process pages in tiles
    constexpr int tile_size = 16;
    const int num_tiles = (kv_len + tile_size - 1) / tile_size;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int kv_start = tile * tile_size;
        int kv_end = min(kv_start + tile_size, kv_len);
        int current_tile_size = kv_end - kv_start;

        // Compute page and offset
        int page_idx = kv_start / page_size;
        int offset_in_page = kv_start % page_size;
        int physical_page = page_indices[page_start + page_idx];

        // Load K tile from paged KV cache
        for (int j = 0; j < current_tile_size; ++j) {
            int pos = offset_in_page + j;
            for (int i = tx; i < head_dim; i += blockDim.x) {
                // NHD layout: [page, head, pos, dim]
                size_t offset = (physical_page * num_kv_heads + kv_head_idx) * page_size * head_dim
                              + pos * head_dim + i;
                float k_val;

                if constexpr (std::is_same_v<T, __half>) {
                    k_val = __half2float(k_data[offset]);
                } else {
                    k_val = __bfloat162float(k_data[offset]);
                }

                k_smem[j * head_dim + i] = k_val;
            }
        }
        __syncthreads();

        // Compute QK scores (only tx=0 needs to compute)
        float scores[16];
        if (tx == 0) {
            for (int j = 0; j < current_tile_size; ++j) {
                float score = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    score += q_vec[i] * k_smem[j * head_dim + i];
                }
                scores[j] = score * sm_scale;
            }
        }
        __syncthreads();

        // Load V tile from paged KV cache
        for (int j = 0; j < current_tile_size; ++j) {
            int pos = offset_in_page + j;
            for (int i = tx; i < head_dim; i += blockDim.x) {
                size_t offset = (physical_page * num_kv_heads + kv_head_idx) * page_size * head_dim
                              + pos * head_dim + i;
                if constexpr (std::is_same_v<T, __half>) {
                    v_smem[j * head_dim + i] = __half2float(v_data[offset]);
                } else {
                    v_smem[j * head_dim + i] = __bfloat162float(v_data[offset]);
                }
            }
        }
        __syncthreads();

        // Update attention state (only tx=0)
        if (tx == 0) {
            state.update(scores, v_smem, current_tile_size, head_dim);
        }
        __syncthreads();
    }

    // Write output
    if (tx == 0) {
        T* out_ptr = output + batch_idx * num_qo_heads * head_dim + qo_head_idx * head_dim;
        float d_safe = fmaxf(state.d, 1e-6f);  // Prevent division by zero
        for (int i = 0; i < head_dim; ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                out_ptr[i] = __float2half(state.o[i] / d_safe);
            } else {
                out_ptr[i] = __float2bfloat16(state.o[i] / d_safe);
            }
        }
    }
}

// ============================================================================
// Version 3: GQA Decode with FP8 KV Cache (Simple)
// ============================================================================

template <typename T>
__global__ void gqa_decode_simple_fp8_kernel(
    const T* __restrict__ q,              // FP16/BF16 query
    const uint8_t* __restrict__ k_cache_fp8,  // FP8 K cache (raw bytes)
    const uint8_t* __restrict__ v_cache_fp8,  // FP8 V cache (raw bytes)
    const float* __restrict__ k_scale,     // K quantization scale
    const float* __restrict__ v_scale,     // V quantization scale
    T* __restrict__ output,
    float sm_scale,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len
) {
    const int group_size = num_qo_heads / num_kv_heads;
    const int kv_head_idx = blockIdx.y;
    const int qo_head_idx = blockIdx.x;

    if (qo_head_idx >= num_qo_heads) return;
    const int kv_head_for_qo = qo_head_idx / group_size;
    if (kv_head_for_qo != kv_head_idx) return;

    const int tx = threadIdx.x;
    __shared__ float k_smem[16 * 128];
    __shared__ float v_smem[16 * 128];

    // Load query vector
    float q_vec[128];
    const T* q_ptr = q + qo_head_idx * head_dim;

    if (tx == 0) {
        for (int i = 0; i < head_dim; ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                q_vec[i] = __half2float(q_ptr[i]);
            } else {
                q_vec[i] = __bfloat162float(q_ptr[i]);
            }
        }
    }
    __syncthreads();

    // Initialize attention state
    AttentionState state;
    if (tx == 0) {
        state.init();
    }
    __syncthreads();

    // Process KV cache in tiles
    constexpr int tile_size = 16;
    const int num_tiles = (kv_len + tile_size - 1) / tile_size;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int kv_start = tile * tile_size;
        int kv_end = min(kv_start + tile_size, kv_len);
        int current_tile_size = kv_end - kv_start;

        // Load K tile (FP8 -> FP32 with dequantization)
        for (int j = 0; j < current_tile_size; ++j) {
            for (int i = tx; i < head_dim; i += blockDim.x) {
                uint8_t fp8_val = k_cache_fp8[(kv_start + j) * num_kv_heads * head_dim + kv_head_idx * head_dim + i];
                float k_val = fp8_to_float(fp8_val) * k_scale[kv_head_idx];
                k_smem[j * head_dim + i] = k_val;
            }
        }
        __syncthreads();

        // Compute QK scores
        float scores[16];
        if (tx == 0) {
            for (int j = 0; j < current_tile_size; ++j) {
                float score = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    score += q_vec[i] * k_smem[j * head_dim + i];
                }
                scores[j] = score * sm_scale;
            }
        }
        __syncthreads();

        // Load V tile (FP8 -> FP32 with dequantization)
        for (int j = 0; j < current_tile_size; ++j) {
            for (int i = tx; i < head_dim; i += blockDim.x) {
                uint8_t fp8_val = v_cache_fp8[(kv_start + j) * num_kv_heads * head_dim + kv_head_idx * head_dim + i];
                float v_val = fp8_to_float(fp8_val) * v_scale[kv_head_idx];
                v_smem[j * head_dim + i] = v_val;
            }
        }
        __syncthreads();

        // Update attention state
        if (tx == 0) {
            state.update(scores, v_smem, current_tile_size, head_dim);
        }
        __syncthreads();
    }

    // Write output
    if (tx == 0) {
        T* out_ptr = output + qo_head_idx * head_dim;
        float d_safe = fmaxf(state.d, 1e-6f);
        for (int i = 0; i < head_dim; ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                out_ptr[i] = __float2half(state.o[i] / d_safe);
            } else {
                out_ptr[i] = __float2bfloat16(state.o[i] / d_safe);
            }
        }
    }
}

// ============================================================================
// Version 4: GQA Decode with FP8 Paged KV Cache
// ============================================================================

template <typename T>
__global__ void gqa_decode_paged_fp8_kernel(
    const T* __restrict__ q,
    const uint8_t* __restrict__ k_data_fp8,  // FP8 K data (raw bytes)
    const uint8_t* __restrict__ v_data_fp8,  // FP8 V data (raw bytes)
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int* __restrict__ page_indices,
    const int* __restrict__ page_indptr,
    T* __restrict__ output,
    float sm_scale,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int batch_idx,
    int kv_len
) {
    const int group_size = num_qo_heads / num_kv_heads;
    const int kv_head_idx = blockIdx.y;
    const int qo_head_idx = blockIdx.x;

    if (qo_head_idx >= num_qo_heads) return;
    const int kv_head_for_qo = qo_head_idx / group_size;
    if (kv_head_for_qo != kv_head_idx) return;

    const int tx = threadIdx.x;
    int page_start = page_indptr[batch_idx];

    if (kv_len <= 0) return;

    __shared__ float k_smem[128 * 16];
    __shared__ float v_smem[128 * 16];

    // Load query vector
    float q_vec[128];
    const T* q_ptr = q + batch_idx * num_qo_heads * head_dim + qo_head_idx * head_dim;

    if (tx == 0) {
        for (int i = 0; i < head_dim; ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                q_vec[i] = __half2float(q_ptr[i]);
            } else {
                q_vec[i] = __bfloat162float(q_ptr[i]);
            }
        }
    }
    __syncthreads();

    // Initialize attention state
    AttentionState state;
    if (tx == 0) {
        state.init();
    }
    __syncthreads();

    // Process pages in tiles
    constexpr int tile_size = 16;
    const int num_tiles = (kv_len + tile_size - 1) / tile_size;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int kv_start = tile * tile_size;
        int kv_end = min(kv_start + tile_size, kv_len);
        int current_tile_size = kv_end - kv_start;

        int page_idx = kv_start / page_size;
        int offset_in_page = kv_start % page_size;
        int physical_page = page_indices[page_start + page_idx];

        // Load K tile from FP8 paged KV cache
        for (int j = 0; j < current_tile_size; ++j) {
            int pos = offset_in_page + j;
            for (int i = tx; i < head_dim; i += blockDim.x) {
                size_t offset = (physical_page * num_kv_heads + kv_head_idx) * page_size * head_dim
                              + pos * head_dim + i;
                float k_val = fp8_to_float(k_data_fp8[offset]) * k_scale[kv_head_idx];
                k_smem[j * head_dim + i] = k_val;
            }
        }
        __syncthreads();

        // Compute QK scores
        float scores[16];
        if (tx == 0) {
            for (int j = 0; j < current_tile_size; ++j) {
                float score = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    score += q_vec[i] * k_smem[j * head_dim + i];
                }
                scores[j] = score * sm_scale;
            }
        }
        __syncthreads();

        // Load V tile from FP8 paged KV cache
        for (int j = 0; j < current_tile_size; ++j) {
            int pos = offset_in_page + j;
            for (int i = tx; i < head_dim; i += blockDim.x) {
                size_t offset = (physical_page * num_kv_heads + kv_head_idx) * page_size * head_dim
                              + pos * head_dim + i;
                float v_val = fp8_to_float(v_data_fp8[offset]) * v_scale[kv_head_idx];
                v_smem[j * head_dim + i] = v_val;
            }
        }
        __syncthreads();

        // Update attention state
        if (tx == 0) {
            state.update(scores, v_smem, current_tile_size, head_dim);
        }
        __syncthreads();
    }

    // Write output
    if (tx == 0) {
        T* out_ptr = output + batch_idx * num_qo_heads * head_dim + qo_head_idx * head_dim;
        float d_safe = fmaxf(state.d, 1e-6f);
        for (int i = 0; i < head_dim; ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                out_ptr[i] = __float2half(state.o[i] / d_safe);
            } else {
                out_ptr[i] = __float2bfloat16(state.o[i] / d_safe);
            }
        }
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

enum class KernelVersion {
    SIMPLE = 0,      // Simple contiguous KV cache
    PAGED = 1,       // Paged KV cache
    FP8_SIMPLE = 2,  // FP8 KV cache (simple)
    FP8_PAGED = 3    // FP8 KV cache (paged)
};

template <typename T>
void gqa_decode_launch(
    const T* q,
    const T* k_cache,
    const T* v_cache,
    T* output,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,
    int batch_size,
    const int* page_indices,
    const int* page_indptr,
    int page_size,
    KernelVersion version,
    float sm_scale = 1.0f / sqrtf(128.0f),
    cudaStream_t stream = 0
) {
    dim3 block(128);  // Fixed block size for simplicity

    if (version == KernelVersion::PAGED) {
        // Paged KV cache version
        // For each batch in the batch_size, launch grids
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            // Grid: x=num_qo_heads (one block per QO head), y=num_kv_heads (for KV head matching)
            dim3 grid(num_qo_heads, num_kv_heads);
            gqa_decode_paged_kernel<T><<<grid, block, 0, stream>>>(
                q, k_cache, v_cache, page_indices, page_indptr, output,
                sm_scale, num_qo_heads, num_kv_heads, head_dim, page_size, batch_idx, kv_len
            );
        }
    } else {
        // Simple contiguous KV cache version
        // Grid: x=num_qo_heads (one block per QO head), y=num_kv_heads (for KV head matching)
        dim3 grid(num_qo_heads, num_kv_heads);
        gqa_decode_simple_kernel<T><<<grid, block, 0, stream>>>(
            q, k_cache, v_cache, output,
            sm_scale, num_qo_heads, num_kv_heads, head_dim, kv_len
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
template void gqa_decode_launch<__half>(
    const __half*, const __half*, const __half*, __half*,
    int, int, int, int, int,
    const int*, const int*, int,
    KernelVersion, float, cudaStream_t);
template void gqa_decode_launch<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int,
    const int*, const int*, int,
    KernelVersion, float, cudaStream_t);

// FP8 launch wrapper
template <typename T>
void gqa_decode_fp8_launch(
    const T* q,
    const uint8_t* k_cache_fp8,  // Changed from KV_T to uint8_t
    const uint8_t* v_cache_fp8,  // Changed from KV_T to uint8_t
    const float* k_scale,
    const float* v_scale,
    T* output,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,
    int batch_size,
    const int* page_indices,
    const int* page_indptr,
    int page_size,
    KernelVersion version,
    float sm_scale = 1.0f / sqrtf(128.0f),
    cudaStream_t stream = 0
) {
    dim3 block(128);

    if (version == KernelVersion::FP8_PAGED) {
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            dim3 grid(num_qo_heads, num_kv_heads);
            gqa_decode_paged_fp8_kernel<T><<<grid, block, 0, stream>>>(
                q, k_cache_fp8, v_cache_fp8, k_scale, v_scale,
                page_indices, page_indptr, output,
                sm_scale, num_qo_heads, num_kv_heads, head_dim, page_size, batch_idx, kv_len
            );
        }
    } else {  // FP8_SIMPLE
        dim3 grid(num_qo_heads, num_kv_heads);
        gqa_decode_simple_fp8_kernel<T><<<grid, block, 0, stream>>>(
            q, k_cache_fp8, v_cache_fp8, k_scale, v_scale, output,
            sm_scale, num_qo_heads, num_kv_heads, head_dim, kv_len
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations for FP8 (no KV_T parameter since we use uint8_t)
template void gqa_decode_fp8_launch<__half>(
    const __half*, const uint8_t*, const uint8_t*, const float*, const float*, __half*,
    int, int, int, int, int,
    const int*, const int*, int,
    KernelVersion, float, cudaStream_t);
template void gqa_decode_fp8_launch<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const uint8_t*, const float*, const float*, __nv_bfloat16*,
    int, int, int, int, int,
    const int*, const int*, int,
    KernelVersion, float, cudaStream_t);

// ============================================================================
// GPU info and reference implementation
// ============================================================================

void print_gpu_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        exit(1);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("\n");
}

// Reference GQA decode (CPU) - non-template version
void gqa_decode_reference(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* output,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,
    float sm_scale
) {
    int group_size = num_qo_heads / num_kv_heads;

    for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
        for (int g = 0; g < group_size; ++g) {
            int qo_head = kv_head * group_size + g;

            // Compute attention scores
            std::vector<float> scores(kv_len);
            for (int kv = 0; kv < kv_len; ++kv) {
                float score = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    score += q[qo_head * head_dim + i] * k_cache[kv * num_kv_heads * head_dim + kv_head * head_dim + i];
                }
                scores[kv] = score * sm_scale;
            }

            // Softmax
            float max_score = *std::max_element(scores.begin(), scores.end());
            float sum_exp = 0.0f;
            for (int kv = 0; kv < kv_len; ++kv) {
                scores[kv] = expf(scores[kv] - max_score);
                sum_exp += scores[kv];
            }
            for (int kv = 0; kv < kv_len; ++kv) {
                scores[kv] /= sum_exp;
            }

            // Compute weighted sum of V
            for (int i = 0; i < head_dim; ++i) {
                output[qo_head * head_dim + i] = 0.0f;
                for (int kv = 0; kv < kv_len; ++kv) {
                    output[qo_head * head_dim + i] += scores[kv] * v_cache[kv * num_kv_heads * head_dim + kv_head * head_dim + i];
                }
            }
        }
    }
}

// ============================================================================
// Test function
// ============================================================================

template <typename T>
void test_gqa_decode(
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int kv_len,
    int batch_size,
    KernelVersion version,
    bool verify = true
) {
    bool use_fp8 = (version == KernelVersion::FP8_SIMPLE || version == KernelVersion::FP8_PAGED);
    bool use_paged = (version == KernelVersion::PAGED || version == KernelVersion::FP8_PAGED);

    const char* version_str = use_fp8 ? (use_paged ? "fp8_paged" : "fp8_simple") : (use_paged ? "paged" : "simple");

    printf("\n");
    printf("===============================================================\n");
    printf("Test: num_qo_heads=%d, num_kv_heads=%d, head_dim=%d, kv_len=%d, batch=%d, version=%s\n",
           num_qo_heads, num_kv_heads, head_dim, kv_len, batch_size, version_str);
    printf("===============================================================\n\n");

    // Group size
    int group_size = num_qo_heads / num_kv_heads;
    printf("Group size: %d (%d QO heads per KV head)\n", group_size, group_size);

    // Allocate memory
    size_t q_size = batch_size * num_qo_heads * head_dim;
    size_t k_size, v_size;

    if (use_paged) {
        int page_size = 16;
        int num_pages = (kv_len + page_size - 1) / page_size * batch_size + 4;
        k_size = num_pages * num_kv_heads * page_size * head_dim;
        v_size = k_size;
    } else {
        k_size = kv_len * num_kv_heads * head_dim;
        v_size = kv_len * num_kv_heads * head_dim;
    }

    size_t output_size = batch_size * num_qo_heads * head_dim;
    size_t page_indptr_size = batch_size + 1;
    size_t page_indices_size = batch_size * (kv_len / 16 + 2) * 2;

    T *d_q, *d_output;
    int *d_page_indices, *d_page_indptr;

    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_page_indices, page_indices_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_page_indptr, page_indptr_size * sizeof(int)));

    // K/V pointers (could be FP16 or FP8)
    T *d_k = nullptr, *d_v = nullptr;           // For FP16/BF16
    uint8_t *d_k_fp8 = nullptr, *d_v_fp8 = nullptr;  // For FP8 (use uint8_t to bypass CUDA conversion)
    float *d_k_scale = nullptr, *d_v_scale = nullptr;      // FP8 scales

    if (use_fp8) {
        CUDA_CHECK(cudaMalloc(&d_k_fp8, k_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_v_fp8, v_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_k_scale, num_kv_heads * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_scale, num_kv_heads * sizeof(float)));
    } else {
        CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_v, v_size * sizeof(T)));
    }

    // Prepare test data
    std::vector<float> h_q_float(q_size);
    std::vector<float> h_k_float(k_size);
    std::vector<float> h_v_float(v_size);
    std::vector<int> h_page_indptr(page_indptr_size);
    std::vector<int> h_page_indices(page_indices_size);
    std::vector<uint8_t> h_k_fp8, h_v_fp8;  // Use uint8_t to bypass CUDA conversion
    std::vector<float> h_k_scale(num_kv_heads), h_v_scale(num_kv_heads);

    // Generate test data
    for (size_t i = 0; i < q_size; ++i) {
        h_q_float[i] = -1.0f + (float)i / 1000.0f;
    }

    if (use_paged) {
        int page_size = 16;
        int num_pages = (kv_len + page_size - 1) / page_size;
        for (int batch = 0; batch < batch_size; ++batch) {
            h_page_indptr[batch] = batch * num_pages;
            int page_start = h_page_indptr[batch];

            for (int kv = 0; kv < kv_len; ++kv) {
                int page_idx = page_start + kv / page_size;
                int offset = kv % page_size;
                h_page_indices[page_idx] = page_idx;

                for (int head = 0; head < num_kv_heads; ++head) {
                    for (int d = 0; d < head_dim; ++d) {
                        size_t idx = (page_idx * num_kv_heads + head) * page_size * head_dim
                                   + offset * head_dim + d;
                        h_k_float[idx] = static_cast<float>((kv + head + d) % 256) / 256.0f - 0.5f;
                        h_v_float[idx] = static_cast<float>((kv * head + d) % 256) / 256.0f - 0.5f;
                    }
                }
            }
        }
        h_page_indptr[batch_size] = batch_size * num_pages;
    } else {
        for (int kv = 0; kv < kv_len; ++kv) {
            for (int head = 0; head < num_kv_heads; ++head) {
                for (int d = 0; d < head_dim; ++d) {
                    size_t idx = kv * num_kv_heads * head_dim + head * head_dim + d;
                    h_k_float[idx] = static_cast<float>((kv + head + d) % 256) / 256.0f - 0.5f;
                    h_v_float[idx] = static_cast<float>((kv * head + d) % 256) / 256.0f - 0.5f;
                }
            }
        }
    }

    // Convert Q to FP16/BF16
    std::vector<T> h_q(q_size);
    for (size_t i = 0; i < q_size; ++i) {
        if constexpr (std::is_same_v<T, __half>) {
            h_q[i] = __float2half(h_q_float[i]);
        } else {
            h_q[i] = __float2bfloat16(h_q_float[i]);
        }
    }

    // Copy Q and page table
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
    if (use_paged) {
        CUDA_CHECK(cudaMemcpy(d_page_indptr, h_page_indptr.data(), page_indptr_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_page_indices, h_page_indices.data(), page_indices_size * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Prepare K/V based on whether using FP8
    if (use_fp8) {
        // Compute quantization scales per KV head
        h_k_fp8.resize(k_size);
        h_v_fp8.resize(v_size);

        if (use_paged) {
            // Paged version: data layout is [num_pages, num_kv_heads, page_size, head_dim]
            for (int head = 0; head < num_kv_heads; ++head) {
                std::vector<float> k_head_values, v_head_values;

                // Collect all values for this head across all pages
                for (size_t i = 0; i < k_size; ++i) {
                    size_t page_idx = i / (num_kv_heads * 16 * head_dim);
                    size_t head_in_page = (i / (16 * head_dim)) % num_kv_heads;
                    if (head_in_page == static_cast<size_t>(head)) {
                        k_head_values.push_back(h_k_float[i]);
                    }
                }
                for (size_t i = 0; i < v_size; ++i) {
                    size_t page_idx = i / (num_kv_heads * 16 * head_dim);
                    size_t head_in_page = (i / (16 * head_dim)) % num_kv_heads;
                    if (head_in_page == static_cast<size_t>(head)) {
                        v_head_values.push_back(h_v_float[i]);
                    }
                }

                // Compute scales
                h_k_scale[head] = compute_quant_scale(k_head_values.data(), k_head_values.size());
                h_v_scale[head] = compute_quant_scale(v_head_values.data(), v_head_values.size());

                // Quantize K/V for this head
                for (size_t i = 0; i < k_size; ++i) {
                    size_t head_in_page = (i / (16 * head_dim)) % num_kv_heads;
                    if (head_in_page == static_cast<size_t>(head)) {
                        h_k_fp8[i] = float_to_fp8_e4m3_host(h_k_float[i] / h_k_scale[head]);
                    }
                }
                for (size_t i = 0; i < v_size; ++i) {
                    size_t head_in_page = (i / (16 * head_dim)) % num_kv_heads;
                    if (head_in_page == static_cast<size_t>(head)) {
                        h_v_fp8[i] = float_to_fp8_e4m3_host(h_v_float[i] / h_v_scale[head]);
                    }
                }
            }
        } else {
            // Simple version: data layout is [kv_len, num_kv_heads, head_dim]
            // idx = kv * num_kv_heads * head_dim + head * head_dim + d
            for (int head = 0; head < num_kv_heads; ++head) {
                std::vector<float> k_head_values, v_head_values;
                k_head_values.reserve(kv_len * head_dim);
                v_head_values.reserve(kv_len * head_dim);

                // Collect all values for this head
                for (int kv = 0; kv < kv_len; ++kv) {
                    for (int d = 0; d < head_dim; ++d) {
                        size_t idx = kv * num_kv_heads * head_dim + head * head_dim + d;
                        k_head_values.push_back(h_k_float[idx]);
                        v_head_values.push_back(h_v_float[idx]);
                    }
                }

                // Compute scales
                h_k_scale[head] = compute_quant_scale(k_head_values.data(), k_head_values.size());
                h_v_scale[head] = compute_quant_scale(v_head_values.data(), v_head_values.size());

                // Debug: print scale for head 0
                if (head == 0 && verify) {
                    printf("Debug: head 0 - k_scale=%.6f, v_scale=%.6f\n", h_k_scale[head], h_v_scale[head]);
                    printf("  First 5 K values: %.4f, %.4f, %.4f, %.4f, %.4f\n",
                           k_head_values[0], k_head_values[1], k_head_values[2], k_head_values[3], k_head_values[4]);

                    // Test FP8 conversion for first value
                    float test_val = -0.5f;
                    float test_normalized = test_val / h_k_scale[head];
                    uint8_t test_fp8 = float_to_fp8_e4m3_host(test_normalized);
                    printf("  FP8 test: val=%.4f -> normalized=%.4f -> fp8=0x%02x (scale=%.6f)\n",
                           test_val, test_normalized, test_fp8, h_k_scale[head]);
                }

                // Quantize K/V for this head
                for (int kv = 0; kv < kv_len; ++kv) {
                    for (int d = 0; d < head_dim; ++d) {
                        size_t idx = kv * num_kv_heads * head_dim + head * head_dim + d;
                        h_k_fp8[idx] = float_to_fp8_e4m3_host(h_k_float[idx] / h_k_scale[head]);
                        h_v_fp8[idx] = float_to_fp8_e4m3_host(h_v_float[idx] / h_v_scale[head]);
                    }
                }
            }
        }

        // Copy FP8 K/V and scales
        CUDA_CHECK(cudaMemcpy(d_k_fp8, h_k_fp8.data(), k_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_fp8, h_v_fp8.data(), v_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_scale, h_k_scale.data(), num_kv_heads * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_scale, h_v_scale.data(), num_kv_heads * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        // Convert K/V to FP16/BF16
        std::vector<T> h_k(k_size), h_v(v_size);
        for (size_t i = 0; i < k_size; ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                h_k[i] = __float2half(h_k_float[i]);
            } else {
                h_k[i] = __float2bfloat16(h_k_float[i]);
            }
        }
        for (size_t i = 0; i < v_size; ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                h_v[i] = __float2half(h_v_float[i]);
            } else {
                h_v[i] = __float2bfloat16(h_v_float[i]);
            }
        }

        CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Warmup
    for (int i = 0; i < 10; ++i) {
        if (use_fp8) {
            gqa_decode_fp8_launch(d_q, d_k_fp8, d_v_fp8, d_k_scale, d_v_scale, d_output,
                num_qo_heads, num_kv_heads, head_dim, kv_len, batch_size,
                d_page_indices, d_page_indptr, 16, version);
        } else {
            gqa_decode_launch(d_q, d_k, d_v, d_output,
                num_qo_heads, num_kv_heads, head_dim, kv_len, batch_size,
                d_page_indices, d_page_indptr, 16, version);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_iters = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; ++i) {
        if (use_fp8) {
            gqa_decode_fp8_launch(d_q, d_k_fp8, d_v_fp8, d_k_scale, d_v_scale, d_output,
                num_qo_heads, num_kv_heads, head_dim, kv_len, batch_size,
                d_page_indices, d_page_indptr, 16, version);
        } else {
            gqa_decode_launch(d_q, d_k, d_v, d_output,
                num_qo_heads, num_kv_heads, head_dim, kv_len, batch_size,
                d_page_indices, d_page_indptr, 16, version);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_latency_ms = elapsed_ms / num_iters;

    // Compute statistics
    size_t kv_bytes = use_fp8 ? (k_size + v_size) * sizeof(__nv_fp8_e4m3) : (k_size + v_size) * sizeof(T);
    size_t total_bytes = q_size * sizeof(T) + kv_bytes + output_size * sizeof(T);
    float bandwidth_gb_per_sec = total_bytes / (avg_latency_ms / 1000.0f) / 1e9f;
    float ops = static_cast<float>(batch_size * num_qo_heads * kv_len * head_dim * 4);
    float gflops = ops / (avg_latency_ms / 1000.0f) / 1e9f;

    printf("Performance:\n");
    printf("  Average latency: %.4f ms\n", avg_latency_ms);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gb_per_sec);
    printf("  Throughput: %.2f GFLOPS\n", gflops);
    if (use_fp8) {
        printf("  Memory savings: %.1f%% (FP8 KV cache)\n", 50.0f);
    }

    // Verify output
    if (verify) {
        std::vector<T> h_output(output_size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost));

        if (use_fp8) {
            // FP8 verification: compare with FP16 reference
            // Compute FP16 reference output
            std::vector<float> h_output_ref(output_size);
            gqa_decode_reference(h_q_float.data(), h_k_float.data(), h_v_float.data(),
                               h_output_ref.data(), num_qo_heads, num_kv_heads, head_dim, kv_len,
                               1.0f / sqrtf(float(head_dim)));

            bool correct = true;
            float max_error = 0.0f;
            float max_rel_error = 0.0f;
            float sum_squared_error = 0.0f;
            float sum_squared_ref = 0.0f;
            int count = 0;

            // Compare FP8 output with FP16 reference
            for (int head = 0; head < std::min(num_qo_heads, 4); ++head) {
                for (int i = 0; i < std::min(head_dim, 8); ++i) {
                    float ref = h_output_ref[head * head_dim + i];
                    float actual;

                    if constexpr (std::is_same_v<T, __half>) {
                        actual = __half2float(h_output[head * head_dim + i]);
                    } else {
                        actual = __bfloat162float(h_output[head * head_dim + i]);
                    }

                    float abs_error = fabs(ref - actual);
                    float rel_error = (fabs(ref) > 1e-6f) ? abs_error / fabs(ref) : abs_error;

                    max_error = fmax(max_error, abs_error);
                    max_rel_error = fmax(max_rel_error, rel_error);
                    sum_squared_error += abs_error * abs_error;
                    sum_squared_ref += ref * ref;
                    count++;

                    // Relaxed threshold for FP8 (allow larger error due to quantization)
                    if (abs_error > 5e-2f) {
                        correct = false;
                    }
                }
            }

            float mse = sum_squared_error / count;
            float rmse = sqrtf(mse);
            float ref_rmse = sqrtf(sum_squared_ref / count);
            float relative_rmse = (ref_rmse > 1e-6f) ? rmse / ref_rmse : 0.0f;

            printf("\nFP8 Verification (vs FP16 reference):\n");
            printf("  Result: %s\n", correct ? "PASSED" : "FAILED");
            printf("  Max absolute error: %.6f\n", max_error);
            printf("  Max relative error: %.6f%%\n", max_rel_error * 100);
            printf("  RMSE: %.6f\n", rmse);
            printf("  Relative RMSE: %.6f%%\n", relative_rmse * 100);

            // Print first few outputs for head 0
            printf("\nFirst 8 output values (head 0):\n");
            for (int i = 0; i < std::min(head_dim, 8); ++i) {
                float ref = h_output_ref[i];
                float actual;

                if constexpr (std::is_same_v<T, __half>) {
                    actual = __half2float(h_output[i]);
                } else {
                    actual = __bfloat162float(h_output[i]);
                }

                float abs_error = fabs(ref - actual);
                float rel_error = (fabs(ref) > 1e-6f) ? abs_error / fabs(ref) : abs_error;

                printf("  [%d] ref=%.4f, fp8=%.4f, abs_err=%.6f, rel_err=%.4f%%\n",
                       i, ref, actual, abs_error, rel_error * 100);
            }

        } else {
            // FP16 verification: compare with CPU reference
            std::vector<float> h_output_ref(output_size);
            gqa_decode_reference(h_q_float.data(), h_k_float.data(), h_v_float.data(),
                               h_output_ref.data(), num_qo_heads, num_kv_heads, head_dim, kv_len,
                               1.0f / sqrtf(float(head_dim)));

            bool correct = true;
            float max_error = 0.0f;

            for (int head = 0; head < std::min(num_qo_heads, 4); ++head) {
                for (int i = 0; i < std::min(head_dim, 8); ++i) {
                    float expected = h_output_ref[head * head_dim + i];
                    float actual;

                    if constexpr (std::is_same_v<T, __half>) {
                        actual = __half2float(h_output[head * head_dim + i]);
                    } else {
                        actual = __bfloat162float(h_output[head * head_dim + i]);
                    }

                    float error = fabs(expected - actual);
                    max_error = fmax(max_error, error);

                    if (error > 1e-2f) {
                        correct = false;
                    }
                }
            }

            printf("\nFP16 Verification (vs CPU reference):\n");
            printf("  Result: %s\n", correct ? "PASSED" : "FAILED");
            printf("  Max error: %.6f\n", max_error);

            // Print first few outputs
            printf("\nFirst 8 output values (head 0):\n");
            for (int i = 0; i < std::min(head_dim, 8); ++i) {
                float q_val = h_q_float[i];
                float expected = h_output_ref[i];
                float actual;

                if constexpr (std::is_same_v<T, __half>) {
                    actual = __half2float(h_output[i]);
                } else {
                    actual = __bfloat162float(h_output[i]);
                }

                printf("  [%d] q=%.4f, expected=%.4f, actual=%.4f, error=%.6f\n",
                       i, q_val, expected, actual, fabs(expected - actual));
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_q));
    if (use_fp8) {
        CUDA_CHECK(cudaFree(d_k_fp8));
        CUDA_CHECK(cudaFree(d_v_fp8));
        CUDA_CHECK(cudaFree(d_k_scale));
        CUDA_CHECK(cudaFree(d_v_scale));
    } else {
        CUDA_CHECK(cudaFree(d_k));
        CUDA_CHECK(cudaFree(d_v));
    }
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_page_indices));
    CUDA_CHECK(cudaFree(d_page_indptr));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("\n");
    printf("===============================================================\n");
    printf("GQA Decode Standalone Implementation Test\n");
    printf("Supports: SM89 (Ada/RTX 40xx) and above\n");
    printf("===============================================================\n\n");

    print_gpu_info();

    // Get kernel version from command line
    KernelVersion version = KernelVersion::SIMPLE;
    bool use_fp8 = false;

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "paged") {
            version = KernelVersion::PAGED;
            printf("Using PAGED KV cache version\n");
        } else if (arg == "fp8") {
            version = KernelVersion::FP8_SIMPLE;
            use_fp8 = true;
            printf("Using FP8 KV cache version (simple)\n");
        } else if (arg == "fp8_paged") {
            version = KernelVersion::FP8_PAGED;
            use_fp8 = true;
            printf("Using FP8 KV cache version (paged)\n");
        } else {
            printf("Using SIMPLE (contiguous) KV cache version\n");
        }
    } else {
        printf("Using SIMPLE (contiguous) KV cache version\n");
    }

    // Test different configurations
    struct Config {
        int num_qo_heads, num_kv_heads, head_dim, kv_len;
    };
    std::vector<Config> configs = {
        {32, 8, 128, 128},    // GQA: 32 QO heads, 8 KV heads, head_dim=128, kv_len=128
        {32, 4, 128, 256},    // GQA: 8-way grouping
        {32, 32, 128, 128},   // MHA: 1-way grouping (baseline)
        {32, 8, 64, 512},     // Smaller head_dim, longer sequence
    };

    for (const auto& cfg : configs) {
        test_gqa_decode<__half>(cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim, cfg.kv_len, 1, version);
    }

    // Test batch processing with paged versions
    if (version == KernelVersion::PAGED) {
        printf("\n");
        printf("===============================================================\n");
        printf("Batch processing test (paged version)\n");
        printf("===============================================================\n\n");
        test_gqa_decode<__half>(32, 8, 128, 256, 4, KernelVersion::PAGED, false);
        test_gqa_decode<__half>(32, 8, 128, 256, 8, KernelVersion::PAGED, false);
    } else if (version == KernelVersion::FP8_PAGED) {
        printf("\n");
        printf("===============================================================\n");
        printf("Batch processing test (FP8 paged version)\n");
        printf("===============================================================\n\n");
        test_gqa_decode<__half>(32, 8, 128, 256, 4, KernelVersion::FP8_PAGED, false);
        test_gqa_decode<__half>(32, 8, 128, 256, 8, KernelVersion::FP8_PAGED, false);
    } else if (version == KernelVersion::FP8_SIMPLE) {
        printf("\n");
        printf("===============================================================\n");
        printf("Batch processing test (FP8 simple version)\n");
        printf("===============================================================\n\n");
        test_gqa_decode<__half>(32, 8, 128, 256, 4, KernelVersion::FP8_SIMPLE, false);
        test_gqa_decode<__half>(32, 8, 128, 256, 8, KernelVersion::FP8_SIMPLE, false);
    }

    printf("\n");
    printf("===============================================================\n");
    printf("All tests completed!\n");
    printf("===============================================================\n");
    printf("\n");

    return 0;
}
