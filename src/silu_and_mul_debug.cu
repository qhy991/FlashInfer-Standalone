/*
 * Debug version to trace data flow
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

__device__ __forceinline__ float silu(const float& x) {
    return x / (1.0f + __expf(-x));
}

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
        float x = __half2float(input[offset + i]);
        float y = __half2float(input[offset + hidden_dim + i]);
        float result = silu(x) * y;
        output[token_idx * hidden_dim + i] = __float2half(result);
    }
}

int main() {
    const int hidden_dim = 128;
    const int num_tokens = 1;
    const size_t input_size = num_tokens * 2 * hidden_dim;
    const size_t output_size = num_tokens * hidden_dim;

    // Prepare test data
    std::vector<float> h_input_float(input_size);
    printf("Generating test data...\n");
    for (size_t i = 0; i < input_size; ++i) {
        int val = (static_cast<int>(i) % 256) - 128;
        h_input_float[i] = static_cast<float>(val) / 128.0f;
    }

    // Print first few values
    printf("First 16 input values:\n");
    for (int i = 0; i < 16; ++i) {
        printf("  h_input_float[%d] = %.6f\n", i, h_input_float[i]);
    }

    // Convert to FP16
    std::vector<__half> h_input(input_size);
    printf("\nConverting to FP16...\n");
    for (size_t i = 0; i < input_size; ++i) {
        h_input[i] = __float2half(h_input_float[i]);
    }

    // Verify conversion
    printf("First 16 FP16 values (converted back):\n");
    for (int i = 0; i < 16; ++i) {
        printf("  h_input[%d] -> %.6f\n", i, __half2float(h_input[i]));
    }

    // Allocate GPU memory
    __half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(__half)));

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(__half),
                         cudaMemcpyHostToDevice));

    // Run kernel
    dim3 grid(num_tokens);
    dim3 block(256);
    silu_and_mul_simple_kernel<__half><<<grid, block>>>(d_input, d_output, hidden_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    std::vector<__half> h_output(output_size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(__half),
                         cudaMemcpyDeviceToHost));

    // Verify output
    printf("\nFirst 16 output values:\n");
    for (int i = 0; i < 16; ++i) {
        float x = h_input_float[i];
        float y = h_input_float[hidden_dim + i];
        float expected = (x / (1.0f + expf(-x))) * y;
        float actual = __half2float(h_output[i]);
        printf("  [%d] x=%.4f, y=%.4f, expected=%.4f, actual=%.4f, error=%.6f\n",
               i, x, y, expected, actual, fabs(expected - actual));
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
