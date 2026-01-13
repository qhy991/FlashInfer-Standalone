/*
 * FlashInfer RMSNorm Standalone Implementation
 * Supports: SM89 (Ada/RTX 40xx) and above
 *
 * RMSNorm (Root Mean Square Normalization):
 * rms = sqrt(mean(input^2) + eps)
 * output = (input / rms) * weight
 *
 * Based on FlashInfer's implementation:
 * - flashinfer/include/flashinfer/norm.cuh
 * - flashinfer/flashinfer/norm.py
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
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
// Block-level reduction utilities
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // First, reduce within each warp
    float warp_sum = warp_reduce_sum(val);

    // Store warp sum in shared memory
    if (lane_id == 0) {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    // Reduce all warp sums (only first warp needs to participate)
    float block_sum = 0.0f;
    if (threadIdx.x < 32) {
        block_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }

    return block_sum;
}

// ============================================================================
// Version 1: Simple implementation (scalar operations)
// ============================================================================

template <typename T>
__global__ void rmsnorm_simple_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    float eps,
    int hidden_dim
) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int offset = row_idx * hidden_dim;

    // Shared memory for block-level reduction
    __shared__ float shared_sums[32];  // Max warps per block: 1024/32 = 32

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val;

        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(input[offset + i]);
        } else {
            __nv_bfloat16 bf16_val = input[offset + i];
            val = __bfloat162float(bf16_val);
        }

        sum_sq += val * val;
    }

    // Block-level reduction
    sum_sq = block_reduce_sum(sum_sq, shared_sums);

    // Broadcast RMS to all threads (only first thread has the correct sum)
    __shared__ float shared_rms;
    if (tid == 0) {
        shared_rms = sqrtf(sum_sq / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();

    float rms = shared_rms;

    // Normalize and scale
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val, w;

        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(input[offset + i]);
            w = __half2float(weight[i]);
        } else {
            __nv_bfloat16 bf16_val = input[offset + i];
            val = __bfloat162float(bf16_val);
            bf16_val = weight[i];
            w = __bfloat162float(bf16_val);
        }

        float result = (val / rms) * w;

        if constexpr (std::is_same_v<T, __half>) {
            output[offset + i] = __float2half(result);
        } else {
            output[offset + i] = __float2bfloat16(result);
        }
    }
}

// ============================================================================
// Version 2: Vectorized implementation (16-byte aligned)
// ============================================================================

template <typename T>
__global__ void rmsnorm_vectorized_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    float eps,
    int hidden_dim
) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int offset = row_idx * hidden_dim;

    // Shared memory for block-level reduction
    __shared__ float shared_sums[32];  // Max warps per block: 1024/32 = 32

    // For FP16/BF16: 8 elements = 16 bytes
    constexpr int vec_size = 8;
    int vec_hidden_dim = hidden_dim / vec_size;

    // Compute sum of squares
    float sum_sq = 0.0f;

    // Vectorized loop
    for (int i = tid; i < vec_hidden_dim; i += blockDim.x) {
        // Raw 16-byte loads
        uint4 data_raw = reinterpret_cast<const uint4*>(input + offset + i * vec_size)[0];

        // Convert and process
        if constexpr (std::is_same_v<T, __half>) {
            const __half* data_half = reinterpret_cast<const __half*>(&data_raw);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float val = __half2float(data_half[j]);
                sum_sq += val * val;
            }
        } else {
            const __nv_bfloat16* data_bf16 = reinterpret_cast<const __nv_bfloat16*>(&data_raw);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float val = __bfloat162float(data_bf16[j]);
                sum_sq += val * val;
            }
        }
    }

    // Handle remaining elements
    int scalar_offset = vec_hidden_dim * vec_size;
    for (int i = tid + scalar_offset; i < hidden_dim; i += blockDim.x) {
        float val;

        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(input[offset + i]);
        } else {
            __nv_bfloat16 bf16_val = input[offset + i];
            val = __bfloat162float(bf16_val);
        }

        sum_sq += val * val;
    }

    // Block-level reduction
    sum_sq = block_reduce_sum(sum_sq, shared_sums);

    // Broadcast RMS to all threads
    __shared__ float shared_rms;
    if (tid == 0) {
        shared_rms = sqrtf(sum_sq / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();

    float rms = shared_rms;

    // Normalize and scale
    for (int i = tid; i < vec_hidden_dim; i += blockDim.x) {
        uint4 data_raw = reinterpret_cast<const uint4*>(input + offset + i * vec_size)[0];
        uint4 weight_raw = reinterpret_cast<const uint4*>(weight + i * vec_size)[0];

        if constexpr (std::is_same_v<T, __half>) {
            const __half* data_half = reinterpret_cast<const __half*>(&data_raw);
            const __half* weight_half = reinterpret_cast<const __half*>(&weight_raw);
            __half out_half[8];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float val = __half2float(data_half[j]);
                float w = __half2float(weight_half[j]);
                out_half[j] = __float2half((val / rms) * w);
            }

            uint4 out_raw = *reinterpret_cast<const uint4*>(out_half);
            reinterpret_cast<uint4*>(output + offset + i * vec_size)[0] = out_raw;
        } else {
            const __nv_bfloat16* data_bf16 = reinterpret_cast<const __nv_bfloat16*>(&data_raw);
            const __nv_bfloat16* weight_bf16 = reinterpret_cast<const __nv_bfloat16*>(&weight_raw);
            __nv_bfloat16 out_bf16[8];

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float val = __bfloat162float(data_bf16[j]);
                float w = __bfloat162float(weight_bf16[j]);
                out_bf16[j] = __float2bfloat16((val / rms) * w);
            }

            uint4 out_raw = *reinterpret_cast<const uint4*>(out_bf16);
            reinterpret_cast<uint4*>(output + offset + i * vec_size)[0] = out_raw;
        }
    }

    // Handle remaining elements
    for (int i = tid + scalar_offset; i < hidden_dim; i += blockDim.x) {
        float val, w;

        if constexpr (std::is_same_v<T, __half>) {
            val = __half2float(input[offset + i]);
            w = __half2float(weight[i]);
            output[offset + i] = __float2half((val / rms) * w);
        } else {
            __nv_bfloat16 bf16_val = input[offset + i];
            val = __bfloat162float(bf16_val);
            bf16_val = weight[i];
            w = __bfloat162float(bf16_val);
            output[offset + i] = __float2bfloat16((val / rms) * w);
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
void rmsnorm_launch(
    const T* input,
    const T* weight,
    T* output,
    int num_rows,
    int hidden_dim,
    KernelVersion version,
    float eps = 1e-6f,
    cudaStream_t stream = 0
) {
    // Check alignment
    size_t alignment_check = hidden_dim * sizeof(T);
    if (version == KernelVersion::VECTORIZED && alignment_check % 16 != 0) {
        fprintf(stderr, "Warning: hidden_dim * sizeof(T) = %zu is not 16-byte aligned, falling back to simple\n",
                alignment_check);
        version = KernelVersion::SIMPLE;
    }

    dim3 grid(num_rows);
    dim3 block(std::min(hidden_dim, 1024));

    if (version == KernelVersion::VECTORIZED && hidden_dim % 8 == 0) {
        rmsnorm_vectorized_kernel<T><<<grid, block, 0, stream>>>(
            input, weight, output, eps, hidden_dim
        );
    } else {
        rmsnorm_simple_kernel<T><<<grid, block, 0, stream>>>(
            input, weight, output, eps, hidden_dim
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
template void rmsnorm_launch<__half>(
    const __half*, const __half*, __half*, int, int, KernelVersion, float, cudaStream_t);
template void rmsnorm_launch<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, KernelVersion, float, cudaStream_t);

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

// Reference RMSNorm implementation (CPU)
void rmsnorm_reference(
    const float* input,
    const float* weight,
    float* output,
    int num_rows,
    int hidden_dim,
    float eps
) {
    for (int row = 0; row < num_rows; ++row) {
        int offset = row * hidden_dim;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            float val = input[offset + i];
            sum_sq += val * val;
        }

        // Compute RMS
        float rms = sqrtf(sum_sq / static_cast<float>(hidden_dim) + eps);

        // Normalize and scale
        for (int i = 0; i < hidden_dim; ++i) {
            output[offset + i] = (input[offset + i] / rms) * weight[i];
        }
    }
}

template <typename T>
void test_rmsnorm(
    int num_rows,
    int hidden_dim,
    KernelVersion version,
    bool verify = true
) {
    printf("\n");
    printf("===============================================================\n");
    printf("Test: num_rows=%d, hidden_dim=%d, version=%s\n",
           num_rows, hidden_dim,
           version == KernelVersion::VECTORIZED ? "vectorized" : "simple");
    printf("===============================================================\n\n");

    // Allocate memory
    size_t input_size = num_rows * hidden_dim;
    size_t weight_size = hidden_dim;
    size_t output_size = num_rows * hidden_dim;

    T *d_input, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(T)));

    // Prepare test data
    std::vector<float> h_input_float(input_size);
    std::vector<float> h_weight_float(weight_size);

    // Generate test data with different patterns for each row
    for (int row = 0; row < num_rows; ++row) {
        for (int i = 0; i < hidden_dim; ++i) {
            // Different pattern for each row
            int val = ((row * 13 + i) % 256) - 128;
            h_input_float[row * hidden_dim + i] = static_cast<float>(val) / 128.0f;
        }
    }

    // Weight: simple linear pattern
    for (int i = 0; i < weight_size; ++i) {
        h_weight_float[i] = 0.5f + static_cast<float>(i % 32) / 64.0f;
    }

    // Convert to target type
    std::vector<T> h_input(input_size);
    std::vector<T> h_weight(weight_size);

    for (size_t i = 0; i < input_size; ++i) {
        if constexpr (std::is_same_v<T, __half>) {
            h_input[i] = __float2half(h_input_float[i]);
        } else {
            h_input[i] = __float2bfloat16(h_input_float[i]);
        }
    }

    for (size_t i = 0; i < weight_size; ++i) {
        if constexpr (std::is_same_v<T, __half>) {
            h_weight[i] = __float2half(h_weight_float[i]);
        } else {
            h_weight[i] = __float2bfloat16(h_weight_float[i]);
        }
    }

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(T),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_size * sizeof(T),
                         cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        rmsnorm_launch(d_input, d_weight, d_output, num_rows, hidden_dim, version);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_iters = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; ++i) {
        rmsnorm_launch(d_input, d_weight, d_output, num_rows, hidden_dim, version);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_latency_ms = elapsed_ms / num_iters;

    // Compute statistics
    size_t total_bytes = (input_size + weight_size + output_size) * sizeof(T);
    float bandwidth_gb_per_sec = total_bytes / (avg_latency_ms / 1000.0f) / 1e9f;
    // Approx 5 ops per element: load input, load weight, square, add, sqrt, div, mul, store
    float ops = static_cast<float>(num_rows * hidden_dim * 8);
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

        // Compute reference on CPU
        std::vector<float> h_output_ref(output_size);
        rmsnorm_reference(h_input_float.data(), h_weight_float.data(),
                         h_output_ref.data(), num_rows, hidden_dim, 1e-6f);

        bool correct = true;
        float max_error = 0.0f;

        for (int row = 0; row < std::min(num_rows, 4); ++row) {
            for (int i = 0; i < std::min(hidden_dim, 8); ++i) {
                float expected = h_output_ref[row * hidden_dim + i];
                float actual;

                if constexpr (std::is_same_v<T, __half>) {
                    actual = __half2float(h_output[row * hidden_dim + i]);
                } else {
                    actual = __bfloat162float(h_output[row * hidden_dim + i]);
                }

                float error = fabs(expected - actual);
                max_error = fmax(max_error, error);

                if (error > 1e-2f) {
                    correct = false;
                }
            }
        }

        printf("\nVerification:\n");
        printf("  Result: %s\n", correct ? "PASSED" : "FAILED");
        printf("  Max error: %.6f\n", max_error);

        // Print first few outputs for first row
        printf("\nFirst 8 output values (row 0):\n");
        for (int i = 0; i < std::min(hidden_dim, 8); ++i) {
            float input_val = h_input_float[i];
            float weight_val = h_weight_float[i];
            float expected = h_output_ref[i];
            float actual;

            if constexpr (std::is_same_v<T, __half>) {
                actual = __half2float(h_output[i]);
            } else {
                actual = __bfloat162float(h_output[i]);
            }

            printf("  [%d] in=%.4f, w=%.4f, expected=%.4f, actual=%.4f, error=%.6f\n",
                   i, input_val, weight_val, expected, actual, fabs(expected - actual));
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    printf("\n");
    printf("===============================================================\n");
    printf("RMSNorm Standalone Implementation Test\n");
    printf("Supports: SM89 (Ada/RTX 40xx) and above\n");
    printf("===============================================================\n\n");

    print_gpu_info();

    // Get kernel version from command line
    KernelVersion version = KernelVersion::VECTORIZED;

    // Check if running in single-test mode (for benchmarking)
    if (argc > 1 && std::string(argv[1]) != "simple") {
        // Parse input file: <num_rows> <hidden_dim> <num_iters>
        FILE* f = fopen(argv[1], "r");
        if (!f) {
            fprintf(stderr, "Error: Cannot open input file %s\n", argv[1]);
            return 1;
        }

        int num_rows, hidden_dim, num_iters;
        if (fscanf(f, "%d %d %d", &num_rows, &hidden_dim, &num_iters) != 3) {
            fprintf(stderr, "Error: Invalid input file format\n");
            fclose(f);
            return 1;
        }
        fclose(f);

        // Run single test without verification
        test_rmsnorm<__half>(num_rows, hidden_dim, version, false);

        return 0;
    }

    // Parse simple flag
    if (argc > 1 && std::string(argv[1]) == "simple") {
        version = KernelVersion::SIMPLE;
    }

    // Test different configurations
    std::vector<std::pair<int, int>> configs = {
        {1, 128},     // Small
        {1, 512},     // Medium
        {1, 2048},    // Large (LLaMA hidden size)
        {1, 4096},    // XLarge (LLaMA-7B/13B hidden size)
        {4, 2048},    // Batch
        {16, 4096},   // Large batch
    };

    for (const auto& [num_rows, hidden_dim] : configs) {
        test_rmsnorm<__half>(num_rows, hidden_dim, version);
    }

    printf("\n");
    printf("===============================================================\n");
    printf("All tests completed!\n");
    printf("===============================================================\n");
    printf("\n");

    return 0;
}
