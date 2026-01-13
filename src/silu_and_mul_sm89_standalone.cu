/*
 * FlashInfer SiLU_and_Mul Standalone Implementation
 * Supports: SM89 (Ada/RTX 40xx) and above
 *
 * This is a standalone implementation of the SiLU_and_Mul fused activation operation.
 * SiLU (Sigmoid Linear Unit): silu(x) = x / (1 + exp(-x))
 * Operation: output = silu(x) * y where input = [x || y]
 *
 * Based on FlashInfer's implementation:
 * - flashinfer/include/flashinfer/activation.cuh
 * - flashinfer/flashinfer/activation.py
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <utility>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
// SiLU Activation Function (both device and host versions)
// ============================================================================

__device__ __forceinline__ float silu_device(const float& x) {
    return x / (1.0f + __expf(-x));
}

__host__ __forceinline__ float silu_host(const float& x) {
    return x / (1.0f + std::exp(-x));
}

// ============================================================================
// Version 1: Simple implementation (scalar operations)
// ============================================================================

template <typename T>
__global__ void silu_and_mul_simple_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int hidden_dim
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int offset = token_idx * 2 * hidden_dim;

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x, y;

        if constexpr (std::is_same_v<T, __half>) {
            x = __half2float(input[offset + i]);
            y = __half2float(input[offset + hidden_dim + i]);
        } else {
            // bfloat16
            __nv_bfloat16 bf16_val = input[offset + i];
            x = __bfloat162float(bf16_val);
            bf16_val = input[offset + hidden_dim + i];
            y = __bfloat162float(bf16_val);
        }

        float result = silu_device(x) * y;

        if constexpr (std::is_same_v<T, __half>) {
            output[token_idx * hidden_dim + i] = __float2half(result);
        } else {
            output[token_idx * hidden_dim + i] = __float2bfloat16(result);
        }
    }
}

// ============================================================================
// Version 2: Vectorized implementation (16-byte aligned)
// ============================================================================

template <typename T>
__global__ void silu_and_mul_vectorized_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int hidden_dim
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int offset = token_idx * 2 * hidden_dim;

    // For FP16/BF16: 8 elements = 16 bytes (use uint4 for raw 16-byte load/store)
    constexpr int vec_size = 8;
    int vec_hidden_dim = hidden_dim / vec_size;

    // Vectorized loop
    for (int i = tid; i < vec_hidden_dim; i += blockDim.x) {
        // Raw 16-byte loads (treat data as raw bytes)
        uint4 x_raw = reinterpret_cast<const uint4*>(input + offset + i * vec_size)[0];
        uint4 y_raw = reinterpret_cast<const uint4*>(input + offset + hidden_dim + i * vec_size)[0];

        // Convert to FP32 and process
        float x_vals[8], y_vals[8], out_vals[8];

        if constexpr (std::is_same_v<T, __half>) {
            const __half* x_half = reinterpret_cast<const __half*>(&x_raw);
            const __half* y_half = reinterpret_cast<const __half*>(&y_raw);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                x_vals[j] = __half2float(x_half[j]);
                y_vals[j] = __half2float(y_half[j]);
                out_vals[j] = silu_device(x_vals[j]) * y_vals[j];
            }
        } else {
            // bfloat16 - need to handle differently
            const __nv_bfloat16* x_bf16 = reinterpret_cast<const __nv_bfloat16*>(&x_raw);
            const __nv_bfloat16* y_bf16 = reinterpret_cast<const __nv_bfloat16*>(&y_raw);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                x_vals[j] = __bfloat162float(x_bf16[j]);
                y_vals[j] = __bfloat162float(y_bf16[j]);
                out_vals[j] = silu_device(x_vals[j]) * y_vals[j];
            }
        }

        // Convert back and store
        if constexpr (std::is_same_v<T, __half>) {
            __half out_half[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                out_half[j] = __float2half(out_vals[j]);
            }
            uint4 out_raw = *reinterpret_cast<const uint4*>(out_half);
            reinterpret_cast<uint4*>(output + token_idx * hidden_dim + i * vec_size)[0] = out_raw;
        } else {
            __nv_bfloat16 out_bf16[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                out_bf16[j] = __float2bfloat16(out_vals[j]);
            }
            uint4 out_raw = *reinterpret_cast<const uint4*>(out_bf16);
            reinterpret_cast<uint4*>(output + token_idx * hidden_dim + i * vec_size)[0] = out_raw;
        }
    }

    // Handle remaining elements (scalar processing)
    int scalar_offset = vec_hidden_dim * vec_size;
    for (int i = tid + scalar_offset; i < hidden_dim; i += blockDim.x) {
        float x, y;

        if constexpr (std::is_same_v<T, __half>) {
            x = __half2float(input[offset + i]);
            y = __half2float(input[offset + hidden_dim + i]);
        } else {
            __nv_bfloat16 bf16_val = input[offset + i];
            x = __bfloat162float(bf16_val);
            bf16_val = input[offset + hidden_dim + i];
            y = __bfloat162float(bf16_val);
        }

        float result = silu_device(x) * y;

        if constexpr (std::is_same_v<T, __half>) {
            output[token_idx * hidden_dim + i] = __float2half(result);
        } else {
            output[token_idx * hidden_dim + i] = __float2bfloat16(result);
        }
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

enum class KernelVersion {
    SIMPLE = 0,
    VECTORIZED = 1
};

template <typename T>
void silu_and_mul_launch(
    const T* input,
    T* output,
    int num_tokens,
    int hidden_dim,
    KernelVersion version,
    cudaStream_t stream = 0
) {
    // Check alignment
    size_t alignment_check = hidden_dim * sizeof(T);
    if (alignment_check % 16 != 0) {
        fprintf(stderr, "Warning: hidden_dim * sizeof(T) = %zu is not 16-byte aligned\n", alignment_check);
    }

    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_dim, 1024));

    if (version == KernelVersion::VECTORIZED && hidden_dim % 8 == 0) {
        silu_and_mul_vectorized_kernel<T><<<grid, block, 0, stream>>>(
            input, output, hidden_dim
        );
    } else {
        silu_and_mul_simple_kernel<T><<<grid, block, 0, stream>>>(
            input, output, hidden_dim
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
template void silu_and_mul_launch<__half>(
    const __half*, __half*, int, int, KernelVersion, cudaStream_t);
template void silu_and_mul_launch<__nv_bfloat16>(
    const __nv_bfloat16*, __nv_bfloat16*, int, int, KernelVersion, cudaStream_t);

// ============================================================================
// Main test function
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

template <typename T>
void test_silu_and_mul(
    int num_tokens,
    int hidden_dim,
    KernelVersion version,
    bool verify = true
) {
    printf("\n");
    printf("===============================================================\n");
    printf("Test: num_tokens=%d, hidden_dim=%d, version=%s\n",
           num_tokens, hidden_dim,
           version == KernelVersion::VECTORIZED ? "vectorized" : "simple");
    printf("===============================================================\n\n");

    // Allocate memory
    size_t input_size = num_tokens * 2 * hidden_dim;
    size_t output_size = num_tokens * hidden_dim;

    T *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(T)));

    // Prepare test data (simple pattern for verification)
    std::vector<float> h_input_float(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        int val = (static_cast<int>(i) % 256) - 128;
        h_input_float[i] = static_cast<float>(val) / 128.0f;  // Range [-1, 1]
    }

    // Convert to target type
    std::vector<T> h_input(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        if constexpr (std::is_same_v<T, __half>) {
            h_input[i] = __float2half(h_input_float[i]);
        } else {
            h_input[i] = __float2bfloat16(h_input_float[i]);
        }
    }

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(T),
                         cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        silu_and_mul_launch(d_input, d_output, num_tokens, hidden_dim, version);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_iters = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; ++i) {
        silu_and_mul_launch(d_input, d_output, num_tokens, hidden_dim, version);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_latency_ms = elapsed_ms / num_iters;

    // Compute statistics
    size_t total_bytes = (input_size + output_size) * sizeof(T);
    float bandwidth_gb_per_sec = total_bytes / (avg_latency_ms / 1000.0f) / 1e9f;
    float ops = static_cast<float>(num_tokens * hidden_dim * 5);  // approx 5 ops per element
    float gflops = ops / (avg_latency_ms / 1000.0f) / 1e9f;

    printf("Performance:\n");
    printf("  Average latency: %.4f ms\n", avg_latency_ms);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gb_per_sec);
    printf("  Throughput: %.2f GFLOPS\n", gflops);

    // Verify output
    if (verify) {
        std::vector<T> h_output(output_size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(T),
                             cudaMemcpyDeviceToHost));

        bool correct = true;
        float max_error = 0.0f;

        for (int t = 0; t < std::min(num_tokens, 4); ++t) {
            for (int i = 0; i < std::min(hidden_dim, 8); ++i) {
                float x = h_input_float[t * 2 * hidden_dim + i];
                float y = h_input_float[t * 2 * hidden_dim + hidden_dim + i];

                float expected = silu_host(x) * y;
                float actual;

                if constexpr (std::is_same_v<T, __half>) {
                    actual = __half2float(h_output[t * hidden_dim + i]);
                } else {
                    actual = __bfloat162float(h_output[t * hidden_dim + i]);
                }

                float error = fabs(expected - actual);
                max_error = fmax(max_error, error);

                if (error > 1e-2f) {  // Relaxed tolerance for bfloat16
                    correct = false;
                }
            }
        }

        printf("\nVerification:\n");
        printf("  Result: %s\n", correct ? "PASSED" : "FAILED");
        printf("  Max error: %.6f\n", max_error);

        // Print first few outputs
        printf("\nFirst 8 output values (token 0):\n");
        for (int i = 0; i < std::min(hidden_dim, 8); ++i) {
            float x = h_input_float[i];
            float y = h_input_float[hidden_dim + i];
            float expected = silu_host(x) * y;
            float actual;

            if constexpr (std::is_same_v<T, __half>) {
                actual = __half2float(h_output[i]);
            } else {
                actual = __bfloat162float(h_output[i]);
            }

            printf("  [%d] x=%.4f, y=%.4f, expected=%.4f, actual=%.4f, error=%.6f\n",
                   i, x, y, expected, actual, fabs(expected - actual));
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    printf("\n");
    printf("===============================================================\n");
    printf("SiLU_and_Mul Standalone Implementation Test\n");
    printf("Supports: SM89 (Ada/RTX 40xx) and above\n");
    printf("===============================================================\n\n");

    print_gpu_info();

    // Get kernel version from command line
    KernelVersion version = KernelVersion::VECTORIZED;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "simple") {
            version = KernelVersion::SIMPLE;
        }
    }

    // Test different configurations
    std::vector<std::pair<int, int>> configs = {
        {1, 128},     // Small
        {1, 512},     // Medium
        {1, 2048},    // Large
        {1, 4096},    // XLarge (hidden size of LLaMA-7B)
        {4, 2048},    // Batch
        {16, 4096},   // Large batch
    };

    for (const auto& [num_tokens, hidden_dim] : configs) {
        test_silu_and_mul<__half>(num_tokens, hidden_dim, version);
    }

    printf("\n");
    printf("===============================================================\n");
    printf("All tests completed!\n");
    printf("===============================================================\n");
    printf("\n");

    return 0;
}
