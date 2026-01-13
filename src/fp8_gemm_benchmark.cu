/**
 * Standalone FP8 GEMM with Performance Benchmarking
 * Based on FlashInfer's cuBLASLt implementation
 *
 * Supported architectures:
 * - SM89: Ada Lovelace (RTX 40xx series)
 * - SM90: Hopper (H100, H200)
 *
 * This version includes detailed performance timing for comparison
 */

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory>
#include <vector>
#include <chrono>

//==============================================================================
// Error checking
//==============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//==============================================================================
// RAII wrappers for cuBLASLt descriptors
//==============================================================================

template <typename T, cublasStatus_t (*Destroy)(T*)>
struct CuBlasLtDeleter {
    void operator()(T* x) const {
        if (x != nullptr) {
            Destroy(x);
        }
    }
};

template <typename T, cublasStatus_t (*Destroy)(T*)>
class CuBlasLtDescriptor {
public:
    T* descriptor() const { return descriptor_.get(); }
    T* descriptor() { return descriptor_.get(); }

protected:
    std::unique_ptr<T, CuBlasLtDeleter<T, Destroy>> descriptor_;
};

class CuBlasLtMatmulDescriptor
    : public CuBlasLtDescriptor<cublasLtMatmulDescOpaque_t, cublasLtMatmulDescDestroy> {
public:
    CuBlasLtMatmulDescriptor(cublasComputeType_t compute_type, cudaDataType_t scale_type) {
        cublasLtMatmulDesc_t desc = nullptr;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, compute_type, scale_type));
        descriptor_.reset(desc);
    }

    template <typename ValT>
    cublasStatus_t setAttribute(cublasLtMatmulDescAttributes_t attr, ValT value) {
        return cublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(ValT));
    }
};

class CuBlasLtMatrixLayout
    : public CuBlasLtDescriptor<cublasLtMatrixLayoutOpaque_t, cublasLtMatrixLayoutDestroy> {
public:
    CuBlasLtMatrixLayout(cudaDataType_t type, uint64_t rows, uint64_t cols, int64_t ld, bool transpose = false) {
        cublasLtMatrixLayout_t desc = nullptr;
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&desc, type,
                                                  transpose ? cols : rows,
                                                  transpose ? rows : cols,
                                                  ld));
        descriptor_.reset(desc);
    }

    template <typename ValT>
    cublasStatus_t setAttribute(cublasLtMatrixLayoutAttribute_t attr, ValT value) {
        return cublasLtMatrixLayoutSetAttribute(descriptor(), attr, &value, sizeof(ValT));
    }
};

class CuBlasLtMatmulPreference
    : public CuBlasLtDescriptor<cublasLtMatmulPreferenceOpaque_t, cublasLtMatmulPreferenceDestroy> {
public:
    CuBlasLtMatmulPreference() {
        cublasLtMatmulPreference_t pref = nullptr;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        descriptor_.reset(pref);
    }

    template <typename ValT>
    cublasStatus_t setAttribute(cublasLtMatmulPreferenceAttributes_t attr, ValT value) {
        return cublasLtMatmulPreferenceSetAttribute(descriptor(), attr, &value, sizeof(ValT));
    }
};

//==============================================================================
// Helper functions
//==============================================================================

template <typename T>
cudaDataType_t getCudaDataType() {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        return CUDA_R_8F_E4M3;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
        return CUDA_R_8F_E5M2;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return CUDA_R_16BF;
    } else if constexpr (std::is_same_v<T, half>) {
        return CUDA_R_16F;
    } else {
        return CUDA_R_32F;
    }
}

//==============================================================================
// FP8 GEMM using cuBLASLt (FlashInfer-style)
//==============================================================================

template <typename FP8Type, typename OutputType>
cublasStatus_t fp8_gemm_cublaslt(
    const FP8Type* d_A,
    const FP8Type* d_B,
    OutputType* d_C,
    int batch_size, int m, int n, int k,
    const float* d_scale_a,
    const float* d_scale_b,
    cublasLtHandle_t lt_handle,
    void* d_workspace,
    size_t workspace_size,
    cudaStream_t stream) {

    const void* A_scale_ptr = static_cast<const void*>(d_scale_a);
    const void* B_scale_ptr = static_cast<const void*>(d_scale_b);

    CuBlasLtMatmulDescriptor matmul_desc(CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);

    int8_t fast_accum = 1;
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum);

    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, A_scale_ptr);
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, B_scale_ptr);

    cudaDataType_t a_type = getCudaDataType<FP8Type>();
    cudaDataType_t b_type = getCudaDataType<FP8Type>();
    cudaDataType_t c_type = getCudaDataType<OutputType>();

    CuBlasLtMatrixLayout a_layout(a_type, m, k, k, true);
    CuBlasLtMatrixLayout b_layout(b_type, k, n, k);
    CuBlasLtMatrixLayout c_layout(c_type, m, n, m);

    if (batch_size > 1) {
        int64_t stride_a = m * k;
        int64_t stride_b = k * n;
        int64_t stride_c = m * n;

        a_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
        a_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a);

        b_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
        b_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_b);

        c_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
        c_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_c);
    }

    CuBlasLtMatmulPreference preference;
    preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                            static_cast<uint64_t>(workspace_size));

    cublasLtMatmulHeuristicResult_t heuristic_result = {};
    int returned_results = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle,
        matmul_desc.descriptor(),
        a_layout.descriptor(),
        b_layout.descriptor(),
        c_layout.descriptor(),
        c_layout.descriptor(),
        preference.descriptor(),
        1,
        &heuristic_result,
        &returned_results);

    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    if (returned_results == 0) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    status = cublasLtMatmul(
        lt_handle,
        matmul_desc.descriptor(),
        &alpha,
        d_A,
        a_layout.descriptor(),
        d_B,
        b_layout.descriptor(),
        &beta,
        nullptr,
        c_layout.descriptor(),
        d_C,
        c_layout.descriptor(),
        &heuristic_result.algo,
        d_workspace,
        workspace_size,
        stream);

    return status;
}

//==============================================================================
// Benchmark function
//==============================================================================

template <typename FP8Type, typename OutputType>
void benchmark_fp8_gemm(
    cublasLtHandle_t lt_handle,
    int batch_size, int m, int n, int k,
    int num_iters = 100) {

    // Allocate and initialize data
    std::vector<FP8Type> h_A(batch_size * m * k);
    std::vector<FP8Type> h_B(batch_size * k * n);
    std::vector<OutputType> h_C(batch_size * m * n);

    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A[i] = FP8Type(float(i % 127) / 127.0f);
    }
    for (size_t i = 0; i < h_B.size(); ++i) {
        h_B[i] = FP8Type(float(i % 127) / 127.0f);
    }

    float scale_a = 1.0f, scale_b = 1.0f;

    FP8Type *d_A, *d_B;
    OutputType *d_C;
    float *d_scale_a, *d_scale_b;

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(FP8Type) * h_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(FP8Type) * h_B.size()));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(OutputType) * h_C.size()));
    CUDA_CHECK(cudaMalloc(&d_scale_a, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_b, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(FP8Type) * h_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(FP8Type) * h_B.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_a, &scale_a, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_b, &scale_b, sizeof(float), cudaMemcpyHostToDevice));

    size_t workspace_size = 32 * 1024 * 1024;
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        fp8_gemm_cublaslt(d_A, d_B, d_C, batch_size, m, n, k,
                         d_scale_a, d_scale_b,
                         lt_handle, d_workspace, workspace_size, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; ++i) {
        fp8_gemm_cublaslt(d_A, d_B, d_C, batch_size, m, n, k,
                         d_scale_a, d_scale_b,
                         lt_handle, d_workspace, workspace_size, 0);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_latency_ms = elapsed.count() / num_iters;

    // Calculate performance
    double total_ops = 2.0 * batch_size * m * n * k * num_iters;
    double elapsed_sec = elapsed.count() / 1000.0;
    double gflops = total_ops / elapsed_sec / 1e9;

    printf("  性能: %.1f GFLOPS\n", gflops);
    printf("  延迟: %.3f ms/iter\n", avg_latency_ms);

    // Copy back and show sample output
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeof(OutputType) * h_C.size(), cudaMemcpyDeviceToHost));
    printf("  输出样本 (前5个): [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           float(h_C[0]), float(h_C[1]), float(h_C[2]), float(h_C[3]), float(h_C[4]));

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_scale_a);
    cudaFree(d_scale_b);
    cudaFree(d_workspace);
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    printf("===============================================================\n");
    printf("FP8 GEMM 性能测试 (Standalone + cuBLASLt)\n");
    printf("支持: SM89 (Ada/RTX 40xx) 和 SM90 (Hopper/H100/H200)\n");
    printf("===============================================================\n\n");

    // Check GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    // Create handle
    cublasLtHandle_t lt_handle;
    CUBLAS_CHECK(cublasLtCreate(&lt_handle));

    // Test configurations
    struct Config {
        int batch, m, n, k;
        const char* name;
    } configs[] = {
        {1, 128, 64, 128, "Small (batch=1)"},
        {2, 128, 64, 128, "Small (batch=2)"},
        {1, 256, 128, 256, "Medium"},
        {1, 512, 256, 512, "Large"},
        {2, 512, 256, 512, "Large (batch=2)"},
    };

    for (const auto& cfg : configs) {
        printf("===============================================================\n");
        printf("[%s]\n", cfg.name);
        printf("配置: batch=%d, m=%d, n=%d, k=%d\n", cfg.batch, cfg.m, cfg.n, cfg.k);
        printf("计算量: %.2f GFLOPS\n", 2.0 * cfg.batch * cfg.m * cfg.n * cfg.k / 1e9);
        printf("===============================================================\n");

        benchmark_fp8_gemm<__nv_fp8_e4m3, half>(lt_handle, cfg.batch, cfg.m, cfg.n, cfg.k);
        printf("\n");
    }

    cublasLtDestroy(lt_handle);

    printf("===============================================================\n");
    printf("性能测试完成\n");
    printf("===============================================================\n");

    return 0;
}
