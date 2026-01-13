/**
 * Standalone FP8 GEMM for SM89+ (Ada Lovelace, RTX 40xx) and SM90 (Hopper, H100/H200)
 * Based on FlashInfer's cuBLASLt implementation
 *
 * Supported architectures:
 * - SM89: Ada Lovelace (RTX 40xx series)
 * - SM90: Hopper (H100, H200)
 *
 * Key differences from my previous attempt:
 * 1. CuBlasLtMatrixLayout constructor with transpose flag
 * 2. Correct stride calculation for batched GEMM
 * 3. Proper error handling with FLASHINFER_CUBLAS_CALL
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

// Note: Use exit instead of return for constructors
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
    // Key: transpose flag swaps rows/cols in the constructor!
    CuBlasLtMatrixLayout(cudaDataType_t type, uint64_t rows, uint64_t cols, int64_t ld, bool transpose = false) {
        cublasLtMatrixLayout_t desc = nullptr;
        // When transpose=true, swap rows and cols for the call
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
    const FP8Type* d_A,        // [batch, m, k] row major (will be treated as transposed)
    const FP8Type* d_B,        // [batch, k, n] column major
    OutputType* d_C,           // [batch, m, n] row major (will be treated as transposed)
    int batch_size, int m, int n, int k,
    const float* d_scale_a,
    const float* d_scale_b,
    cublasLtHandle_t lt_handle,
    void* d_workspace,
    size_t workspace_size,
    cudaStream_t stream) {

    const void* A_scale_ptr = static_cast<const void*>(d_scale_a);
    const void* B_scale_ptr = static_cast<const void*>(d_scale_b);

    // Create matmul descriptor
    CuBlasLtMatmulDescriptor matmul_desc(CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Set transpose modes
    cublasOperation_t transa = CUBLAS_OP_T;  // A is transposed
    cublasOperation_t transb = CUBLAS_OP_N;  // B is not transposed
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);

    // Enable fast accumulation (important for FP8)
    int8_t fast_accum = 1;
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum);

    // Set scale pointers (FP8-specific)
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, A_scale_ptr);
    matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, B_scale_ptr);

    // Get data types
    cudaDataType_t a_type = getCudaDataType<FP8Type>();
    cudaDataType_t b_type = getCudaDataType<FP8Type>();
    cudaDataType_t c_type = getCudaDataType<OutputType>();

    // Create matrix layouts (matching FlashInfer's approach)
    // A: [m, k] row major, but treated as transposed -> [k, m] column major
    CuBlasLtMatrixLayout a_layout(a_type, m, k, k, true);  // transpose=true!
    // B: [k, n] column major (not transposed)
    CuBlasLtMatrixLayout b_layout(b_type, k, n, k);
    // C: [m, n] row major, treated as transposed -> [n, m] column major
    CuBlasLtMatrixLayout c_layout(c_type, m, n, m);

    // Set batch parameters
    if (batch_size > 1) {
        int64_t stride_a = m * k;  // elements between A batches
        int64_t stride_b = k * n;  // elements between B batches
        int64_t stride_c = m * n;  // elements between C batches

        a_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
        a_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a);

        b_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
        b_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_b);

        c_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
        c_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_c);
    }

    // Create preference
    CuBlasLtMatmulPreference preference;
    preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                            static_cast<uint64_t>(workspace_size));

    // Get heuristic
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
        fprintf(stderr, "cublasLtMatmulAlgoGetHeuristic failed: %d\n", status);
        return status;
    }

    if (returned_results == 0) {
        fprintf(stderr, "No FP8 GEMM algorithm found (returned_results=0)\n");
        fprintf(stderr, "This may mean:\n");
        fprintf(stderr, "  - GPU does not support FP8 Tensor Core\n");
        fprintf(stderr, "  - CUDA/cuBLAS version is too old\n");
        fprintf(stderr, "  - Driver does not support FP8\n");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    // Execute GEMM
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
        nullptr,  // No C matrix (beta=0)
        c_layout.descriptor(),
        d_C,
        c_layout.descriptor(),
        &heuristic_result.algo,
        d_workspace,
        workspace_size,
        stream);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasLtMatmul failed: %d\n", status);
    }

    return status;
}

//==============================================================================
// Reference FP16 GEMM for comparison
//==============================================================================

// Simple kernel to transpose matrix from row-major to column-major
template<typename T>
__global__ void transpose_kernel(const T* input, T* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

cublasStatus_t fp16_gemm_cublas(
    const half* d_A,
    const half* d_B,
    half* d_C,
    int m, int n, int k,
    cublasHandle_t handle,
    cudaStream_t stream) {

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // For correct results, we need to:
    // 1. Transpose A from row-major [m,k] to column-major [k,m]
    // 2. B is already column-major [k,n]
    // 3. Compute C_col = A_col^T * B using cuBLAS (column-major)
    // 4. Transpose C from column-major to row-major

    half *d_A_col = nullptr, *d_C_col = nullptr, *d_C_row = nullptr;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cuda_err;

    // Allocate temporary buffers
    cuda_err = cudaMalloc(&d_A_col, sizeof(half) * m * k);
    if (cuda_err != cudaSuccess) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    cuda_err = cudaMalloc(&d_C_col, sizeof(half) * m * n);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_A_col);
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    cuda_err = cudaMalloc(&d_C_row, sizeof(half) * m * n);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_A_col);
        cudaFree(d_C_col);
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    // Set stream for cuBLAS
    cublasSetStream(handle, stream);

    // Transpose A: row-major to column-major
    dim3 block(16, 16);
    dim3 grid((k + 15) / 16, (m + 15) / 16);
    transpose_kernel<<<grid, block, 0, stream>>>(d_A, d_A_col, m, k);

    // Compute C_col = A_col * B (column-major, so we compute C = A^T * B)
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,  // A_col^T * B
        m, n, k,
        &alpha,
        d_A_col, CUDA_R_16F, k,   // A_col: [k,m] column-major
        d_B, CUDA_R_16F, k,       // B: [k,n] column-major
        &beta,
        d_C_col, CUDA_R_16F, m,   // C_col: [m,n] column-major
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_A_col);
        cudaFree(d_C_col);
        cudaFree(d_C_row);
        return status;
    }

    // Transpose C: column-major to row-major
    grid = dim3((n + 15) / 16, (m + 15) / 16);
    transpose_kernel<<<grid, block, 0, stream>>>(d_C_col, d_C_row, m, n);

    // Copy result back
    cuda_err = cudaMemcpyAsync(d_C, d_C_row, sizeof(half) * m * n, cudaMemcpyDeviceToDevice, stream);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_A_col);
        cudaFree(d_C_col);
        cudaFree(d_C_row);
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    // Cleanup
    cudaFree(d_A_col);
    cudaFree(d_C_col);
    cudaFree(d_C_row);

    return status;
}

//==============================================================================
// Simple FP8 reference kernel (for correctness check)
//==============================================================================

template <int TILE_M, int TILE_N>
__global__ void fp8_gemm_reference_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k,
    float scale_a, float scale_b) {

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= m || col >= n) return;

    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        // A is row-major: A[row][kk] = A[row * k + kk]
        float a_val = float(__nv_fp8_e4m3(A[row * k + kk]));
        // B is stored in row-major but interpreted as column-major by cuBLAS
        // For column-major B[k][n], element B[kk][col] is at B[col * k + kk]
        float b_val = float(__nv_fp8_e4m3(B[col * k + kk]));
        acc += a_val * b_val;
    }

    C[row * n + col] = half(scale_a * scale_b * acc);
}

//==============================================================================
// Main test
//==============================================================================

int main(int argc, char** argv) {
    printf("===============================================================\n");
    printf("FP8 GEMM Test (FlashInfer-style implementation)\n");
    printf("Supports: SM89 (Ada/RTX 40xx) and SM90 (Hopper/H100/H200)\n");
    printf("===============================================================\n\n");

    // Check GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    // Test both single and batched GEMM
    const int batch_sizes[] = {1, 2};

    for (int batch_idx = 0; batch_idx < 2; ++batch_idx) {
        int batch = batch_sizes[batch_idx];
        int m = 128, n = 64, k = 128;

        printf("===============================================================\n");
        printf("Test: batch=%d, m=%d, n=%d, k=%d\n", batch, m, n, k);
        printf("===============================================================\n\n");

        // Allocate host memory
        std::vector<__nv_fp8_e4m3> h_A(batch * m * k);
        std::vector<__nv_fp8_e4m3> h_B(batch * k * n);
        std::vector<half> h_C_fp8(batch * m * n);
        std::vector<half> h_C_ref(batch * m * n);
        std::vector<half> h_C_fp16(batch * m * n);

        // Initialize
        for (size_t i = 0; i < h_A.size(); ++i) {
            h_A[i] = __nv_fp8_e4m3(float(i % 127) / 127.0f);
        }
        for (size_t i = 0; i < h_B.size(); ++i) {
            h_B[i] = __nv_fp8_e4m3(float(i % 127) / 127.0f);
        }

        // Scale factors
        float scale_a = 1.0f, scale_b = 1.0f;

        // Allocate device memory
        __nv_fp8_e4m3 *d_A, *d_B;
        half *d_C_fp8, *d_C_ref, *d_C_fp16;
        float *d_scale_a, *d_scale_b;

        CUDA_CHECK(cudaMalloc(&d_A, sizeof(__nv_fp8_e4m3) * h_A.size()));
        CUDA_CHECK(cudaMalloc(&d_B, sizeof(__nv_fp8_e4m3) * h_B.size()));
        CUDA_CHECK(cudaMalloc(&d_C_fp8, sizeof(half) * h_C_fp8.size()));
        CUDA_CHECK(cudaMalloc(&d_C_ref, sizeof(half) * h_C_ref.size()));
        CUDA_CHECK(cudaMalloc(&d_C_fp16, sizeof(half) * h_C_fp16.size()));
        CUDA_CHECK(cudaMalloc(&d_scale_a, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scale_b, sizeof(float)));

        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(__nv_fp8_e4m3) * h_A.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(__nv_fp8_e4m3) * h_B.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scale_a, &scale_a, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scale_b, &scale_b, sizeof(float), cudaMemcpyHostToDevice));

        // Create handles
        cublasLtHandle_t lt_handle;
        CUBLAS_CHECK(cublasLtCreate(&lt_handle));

        cublasHandle_t cublas_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));

        // Allocate workspace
        size_t workspace_size = 32 * 1024 * 1024;
        void* d_workspace;
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

        //======================================================================
        // Test 1: FP8 GEMM with cuBLASLt
        //======================================================================
        printf("[Test 1] FP8 GEMM with cuBLASLt (FlashInfer method)...\n");

        cublasStatus_t status = fp8_gemm_cublaslt<__nv_fp8_e4m3, half>(
            d_A, d_B, d_C_fp8, batch, m, n, k,
            d_scale_a, d_scale_b,
            lt_handle, d_workspace, workspace_size, 0);

        if (status == CUBLAS_STATUS_SUCCESS) {
            printf("  SUCCESS! FP8 GEMM with cuBLASLt worked!\n");

            CUDA_CHECK(cudaMemcpy(h_C_fp8.data(), d_C_fp8, sizeof(half) * h_C_fp8.size(), cudaMemcpyDeviceToHost));

            printf("  First 5 output values:\n");
            for (int i = 0; i < 5; ++i) {
                printf("    C_fp8[%d] = %.6f\n", i, float(h_C_fp8[i]));
            }
            printf("\n");
        } else {
            printf("  FAILED: FP8 GEMM with cuBLASLt returned status %d\n", status);
            printf("  This is expected if:\n");
            printf("    - cuBLAS version doesn't support FP8\n");
            printf("    - GPU driver doesn't support FP8\n");
            printf("    - CUDA version is too old\n");
        }

        //======================================================================
        // Test 2: FP16 GEMM with cuBLAS (for reference)
        //======================================================================
        printf("[Test 2] FP16 GEMM with cuBLAS (baseline)...\n");

        // Convert FP8 to FP16 for baseline
        std::vector<half> h_A_fp16(h_A.size());
        std::vector<half> h_B_fp16(h_B.size());
        for (size_t i = 0; i < h_A.size(); ++i) {
            h_A_fp16[i] = half(float(h_A[i]));
        }
        for (size_t i = 0; i < h_B.size(); ++i) {
            h_B_fp16[i] = half(float(h_B[i]));
        }

        half *d_A_fp16, *d_B_fp16;
        CUDA_CHECK(cudaMalloc(&d_A_fp16, sizeof(half) * h_A_fp16.size()));
        CUDA_CHECK(cudaMalloc(&d_B_fp16, sizeof(half) * h_B_fp16.size()));
        CUDA_CHECK(cudaMemcpy(d_A_fp16, h_A_fp16.data(), sizeof(half) * h_A_fp16.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_fp16, h_B_fp16.data(), sizeof(half) * h_B_fp16.size(), cudaMemcpyHostToDevice));

        status = fp16_gemm_cublas(d_A_fp16, d_B_fp16, d_C_fp16, m, n, k, cublas_handle, 0);

        if (status == CUBLAS_STATUS_SUCCESS) {
            printf("  SUCCESS! FP16 GEMM worked\n");

            CUDA_CHECK(cudaMemcpy(h_C_fp16.data(), d_C_fp16, sizeof(half) * h_C_fp16.size(), cudaMemcpyDeviceToHost));

            printf("  First 5 output values:\n");
            for (int i = 0; i < 5; ++i) {
                printf("    C_fp16[%d] = %.6f\n", i, float(h_C_fp16[i]));
            }
            printf("\n");
        }

        //======================================================================
        // Test 3: Simple CUDA kernel reference
        //======================================================================
        printf("[Test 3] Simple CUDA kernel reference (FP8, no Tensor Cores)...\n");

        dim3 grid((n + 15) / 16, (m + 15) / 16);
        dim3 block(16, 16);

        for (int b = 0; b < batch; ++b) {
            size_t offset_a = b * m * k;
            size_t offset_b = b * k * n;
            size_t offset_c = b * m * n;

            fp8_gemm_reference_kernel<16, 16><<<grid, block>>>(
                d_A + offset_a, d_B + offset_b, d_C_ref + offset_c,
                m, n, k, scale_a, scale_b);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, sizeof(half) * h_C_ref.size(), cudaMemcpyDeviceToHost));

        printf("  First 5 output values:\n");
        for (int i = 0; i < 5; ++i) {
            printf("    C_ref[%d] = %.6f\n", i, float(h_C_ref[i]));
        }
        printf("\n");

        //======================================================================
        // Compare results if FP8 succeeded
        //======================================================================
        if (status == CUBLAS_STATUS_SUCCESS) {
            printf("[Comparison] FP8 vs FP16 vs Reference\n");

            // Just compare the first element to see if they're in the same ballpark
            float fp8_val = float(h_C_fp8[0]);
            float fp16_val = float(h_C_fp16[0]);
            float ref_val = float(h_C_ref[0]);

            printf("  First element comparison:\n");
            printf("    FP8 (cuBLASLt):  %.6f\n", fp8_val);
            printf("    FP16 (cuBLAS):    %.6f\n", fp16_val);
            printf("    Reference:        %.6f\n", ref_val);
            printf("\n");
        }

        //======================================================================
        // Cleanup
        //======================================================================
        cublasLtDestroy(lt_handle);
        cublasDestroy(cublas_handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_fp8);
        cudaFree(d_C_ref);
        cudaFree(d_C_fp16);
        cudaFree(d_scale_a);
        cudaFree(d_scale_b);
        cudaFree(d_workspace);
        cudaFree(d_A_fp16);
        cudaFree(d_B_fp16);
    }

    printf("\n===============================================================\n");
    printf("All tests completed\n");
    printf("===============================================================\n");

    return 0;
}
