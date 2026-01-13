# FlashInfer FP8 GEMM 调用链分析

## 概述

本文档详细分析了 FlashInver 中 FP8 GEMM 的完整调用链，重点关注 SM89 (Ada Lovelace/RTX 4060) 架构。

## 架构差异

### SM89 vs SM100+ 特性对比

| 特性 | SM89 (Ada) | SM100+ (Blackwell) |
|------|-------------|---------------------|
| Backend | cuBLAS/cuDNN | CUTLASS 3.x |
| 内存加载 | ld.global | **TMA** (Tensor Memory Accelerator) |
| 集群形状 | 固定 | 可配置 (1x1x1, 2x1x1, ...) |
| Epilogue | 内置 | Sm90EVT 自定义 |
| 矩阵乘法 API | cuBLASLt | CUTLASS 3.x collective builder |

---

## SM89 FP8 GEMM 调用链

### Python API 层

**文件**: `flashinfer/gemm/gemm_base.py`

```python
# 行 2859: Python API 入口点
def bmm_fp8(
    A: torch.Tensor,          # [batch, m, k] FP16/BF16 (内部转换为 FP8)
    B: torch.Tensor,          # [batch, k, n] FP16/BF16 (内部转换为 FP8)
    A_scale: torch.Tensor,    # [1] 或 [batch] FP32
    B_scale: torch.Tensor,    # [1] 或 [batch] FP32
    out: torch.Tensor,        # [batch, m, n] FP16/BF16 输出
    ...,
) -> torch.Tensor:
```

### Backend 选择层

**文件**: `flashinfer/gemm/gemm_base.py`

```python
# 行 2819-2846: Backend 启发式函数
def _heuristic_func_bmm_fp8(
    workspace: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    ...
) -> Dict[str, Any]:
    # SM89 架构检测
    major, minor = torch.cuda.get_device_capability()
    if major == 8 and minor == 9:
        return {
            "backends": ["cublas", "cudnn"],  # 不包含 "cutlass_sm10x"
            ...
        }
```

**关键点**:
- SM89 返回 `backends = ["cublas", "cudnn"]`
- SM100+ 返回 `backends = ["cutlass_sm10x", "cutlass_sm12x"]`

### C++ 绑定层

**文件**: `flashinfer/csrc/flashinfer_gemm_binding.cu`

```cpp
// TVM-FFI 绑定
TVM_DLL_EXPORT TypedPackedFunc<
    void(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, DataType, void*)>
bmm_fp8() {
    return TypedPackedFunc<
        void(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, DataType, void*)>(
        [](Tensor A, Tensor B, Tensor A_scale, Tensor B_scale, Tensor out,
           Tensor workspace, DataType dtype, void* stream) {
            // 调用内部函数
            bmm_fp8_internal(A, B, A_scale, B_scale, out, workspace, dtype,
                           static_cast<cudaStream_t>(stream));
        });
}
```

### 核心实现层 (SM89)

**文件**: `flashinfer/include/flashinfer/gemm/bmm_fp8.cuh`

```cpp
template <typename AT, typename BT, typename DT>
cublasStatus_t bmm_fp8_internal_cublaslt(
    const void* A,           // [batch, m, k] FP8
    const void* B,           // [batch, k, n] FP8
    const void* A_scale,     // [batch] FP32
    const void* B_scale,     // [batch] FP32
    void* C,                 // [batch, m, n] FP16/BF16
    int batch_size,
    int m, int n, int k,
    cudaDataType_t a_type,
    cudaDataType_t b_type,
    cudaDataType_t c_type,
    cublasLtHandle_t handle,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {

    // 1. 创建矩阵乘法描述符
    auto matmul_desp = CuBlasLtMatmulDescriptor(CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // 2. 设置转置标志
    matmul_desp.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);  // A 转置
    matmul_desp.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);  // B 不转置

    // 3. 启用快速累加 (FP8 关键配置)
    int8_t fast_accum = 1;
    matmul_desp.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum);

    // 4. 设置 scale 指针
    matmul_desp.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, A_scale_ptr);
    matmul_desp.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, B_scale_ptr);

    // 5. 创建矩阵布局 (关键: transpose=true)
    auto a_desp = CuBlasLtMatrixLayout(a_type, m, k, k, true);   // A 转置
    auto b_desp = CuBlasLtMatrixLayout(b_type, k, n, k);          // B 不转置
    auto d_desp = CuBlasLtMatrixLayout(c_type, m, n, m);          // C

    // 6. 算法启发式搜索
    cublasLtMatmulAlgo_t algo;
    cublasLtMatmulAlgoGetHeuristic(..., &algo);

    // 7. 执行 GEMM
    cublasLtMatmul(
        handle,
        matmul_desp.descriptor(),
        &alpha,
        A, a_desp.descriptor(),
        B, b_desp.descriptor(),
        &beta,
        nullptr, c_desp.descriptor(),  // C 指针 (可为空)
        C, d_desp.descriptor(),        // D 指针 (输出)
        &algo,
        workspace, workspace_size,
        stream);

    return CUBLAS_STATUS_SUCCESS;
}
```

---

## cuBLASLt API 详细说明

### 1. 矩阵布局配置

```cpp
// CuBlasLtMatrixLayout 构造函数
CuBlasLtMatrixLayout(
    cudaDataType_t type,  // 数据类型 (CUDA_R_8F_E4M3, CUDA_R_16F, ...)
    int64_t rows,         // 行数
    int64_t cols,         // 列数
    int64_t ld,           // 主导维度 (leading dimension)
    bool transpose = false  // 是否转置 (关键参数!)
);
```

**SM89 FP8 配置**:
- A 矩阵: `CuBlasLtMatrixLayout(CUDA_R_8F_E4M3, m, k, k, **true**)`  ← 转置!
- B 矩阵: `CuBlasLtMatrixLayout(CUDA_R_8F_E4M3, k, n, k, false)`
- C 矩阵: `CuBlasLtMatrixLayout(CUDA_R_16F, m, n, m, false)`

### 2. FP8 特殊属性

```cpp
// 快速累加模式 (FP8 必需)
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, 1);

// Scale 指针 (FP8 必需)
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, scale_a);
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, scale_b);
```

### 3. 计算流程

```
输入: FP8 (e4m3/e5m2)
  ↓
Tensor Core 计算: FP8 × FP8 → FP32 累加
  ↓
Epilogue: scale_a × scale_b × accumulator
  ↓
输出: FP16/BF16
```

---

## SM100+ CUTLASS 调用链 (参考)

**文件**: `flashinfer/jit/gemm/core.py`

```python
# 行 143-190: CUTLASS JIT 模块生成
def gen_gemm_sm100_module_cutlass_fp8(...) -> str:
    return """
    #include <cutlass/arch/sm90.h>  // SM100+ 头文件
    #include <cutlass/gemm/grouped_runner.h>
    #include <cutlass/gemm/collective/fp8_accumulation.hpp>

    using TileShape = TileShape_MMA<128, 128, 128>;
    using ClusterShape = ClusterShape_MMA<1, 1, 1>;  // 可配置
    using CollectiveEpilogue = Sm90EVT<...>;          // 自定义 epilogue

    CUTLASS 3.x TMA 加载
        ↓
    Tensor Core (SM100)
        ↓
    自定义 Epilogue
    """
```

---

## 调用链总结

### SM89 完整调用链

```
Python: flashinfer.gemm.bmm_fp8(A, B, scale_a, scale_b)
    │
    ├─ 检测 GPU 架构: torch.cuda.get_device_capability() → (8, 9)
    │
    ├─ Backend 选择: ["cublas", "cudnn"]  (非 "cutlass_sm10x")
    │
    ▼
Python-C++ 绑定 (TVM-FFI)
    │
    ▼
C++: bmm_fp8_internal<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_half>(...)
    │
    ├─ FP16 → FP8 转换 (如果需要)
    │
    ▼
C++: bmm_fp8_internal_cublaslt(...)
    │
    ├─ cublasLtMatmulDescCreate(CUBLAS_COMPUTE_32F, CUDA_R_32F)
    ├─ setAttribute(TRANSA, CUBLAS_OP_T)
    ├─ setAttribute(FAST_ACCUM, 1)  ← FP8 关键
    ├─ setAttribute(A_SCALE_POINTER, scale_a)
    ├─ setAttribute(B_SCALE_POINTER, scale_b)
    │
    ▼
cuBLASLt: cublasLtMatmul(...)
    │
    ▼
CUDA Driver: Tensor Core (SM89 FP8)
    ├─ FP8 × FP8 → FP32 累加
    ├─ scale_a × scale_b × accumulator
    └─ 输出 FP16/BF16
```

### 关键文件位置

| 组件 | 文件路径 |
|------|----------|
| Python API | `flashinfer/gemm/gemm_base.py:2859` |
| Backend 选择 | `flashinfer/gemm/gemm_base.py:2819` |
| C++ 绑定 | `flashinfer/csrc/flashinfer_gemm_binding.cu` |
| SM89 实现 | `flashinfer/include/flashinfer/gemm/bmm_fp8.cuh` |
| SM100+ JIT | `flashinfer/jit/gemm/core.py:143` |

---

## 性能特点

### SM89 (Ada Lovelace)
- **理论峰值**: ~150 TFLOPS (FP8)
- **实际性能**: ~40-60 TFLOPS (取决于 GPU 型号和时钟频率)
- **API**: cuBLASLt (成熟、稳定)
- **优势**: 兼容性好，不需要 JIT 编译

### SM100+ (Blackwell)
- **理论峰值**: ~2000+ TFLOPS (FP8)
- **优化特性**: TMA、可配置集群、自定义 Epilogue
- **API**: CUTLASS 3.x (需要 JIT 编译)
- **优势**: 更高理论性能，更灵活的优化选项

---

## 调试技巧

### 1. 检查 Backend 选择

```python
import flashinfer
from flashinfer.gemm import bmm_fp8

# 检查你的 GPU 使用哪个 backend
major, minor = torch.cuda.get_device_capability()
print(f"Compute Capability: {major}.{minor}")
# SM89 → cuBLAS, SM100+ → CUTLASS
```

### 2. 启用详细日志

```bash
export FLASHINFER_VERBOSE=1
```

### 3. 检查 cuBLASLt 错误

```cpp
cublasStatus_t status = cublasLtMatmul(...);
if (status != CUBLAS_STATUS_SUCCESS) {
    const char* error_str;
    cublasGetErrorString(status, &error_str);
    printf("cuBLASLt Error: %s\n", error_str);
}
```

---

## 参考资源

- [cuBLASLt API 文档](https://docs.nvidia.com/cuda/cublas/index.html)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [CUDA FP8 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fp8-floating-point)
