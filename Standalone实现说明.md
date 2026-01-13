# Standalone FP8 GEMM 实现说明

## 概述

这是一个基于 FlashInfer 实现的 standalone FP8 GEMM 库，专门为 SM89 (Ada Lovelace/RTX 40xx) 架构优化。

## 设计目标

1. **完全独立**: 不依赖 FlashInfer，仅使用 cuBLASLt API
2. **易于理解**: 代码结构清晰，注释详细
3. **可扩展**: 支持批处理、不同数据类型
4. **生产就绪**: 完整的错误处理和资源管理

## 核心组件

### 1. RAII 封装 (cuBLASLt 描述符)

```cpp
template <typename T, cublasStatus_t (*Destroy)(T*)>
class CuBlasLtDescriptor {
    // 自动管理描述符生命周期
    // 防止资源泄漏
};

// 具体实现:
class CuBlasLtMatmulDescriptor : public CuBlasLtDescriptor<...>;
class CuBlasLtMatrixLayout : public CuBlasLtDescriptor<...>;
class CuBlasLtMatmulPreference : public CuBlasLtDescriptor<...>;
```

**优势**:
- 自动资源管理（类似 unique_ptr）
- 异常安全
- 代码简洁

### 2. FP8 GEMM 核心函数

```cpp
template <typename FP8Type, typename OutputType>
cublasStatus_t fp8_gemm_cublaslt(
    const FP8Type* d_A,        // [batch, m, k] row major
    const FP8Type* d_B,        // [batch, k, n] column major
    OutputType* d_C,           // [batch, m, n] row major
    int batch_size, int m, int n, int k,
    const float* d_scale_a,
    const float* d_scale_b,
    cublasLtHandle_t lt_handle,
    void* d_workspace,
    size_t workspace_size,
    cudaStream_t stream
);
```

**参数说明**:
- `FP8Type`: `__nv_fp8_e4m3` 或 `__nv_fp8_e5m2`
- `OutputType`: `half` (FP16) 或 `__nv_bfloat16`
- `batch_size`: 批大小 (1 或更大)
- `m, n, k`: 矩阵维度
- `d_scale_a, d_scale_b`: FP8 缩放因子

### 3. 类型推断

```cpp
template <typename T>
cudaDataType_t getCudaDataType() {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        return CUDA_R_8F_E4M3;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
        return CUDA_R_8F_E5M2;
    } else if constexpr (std::is_same_v<T, half>) {
        return CUDA_R_16F;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return CUDA_R_16BF;
    } else {
        return CUDA_R_32F;
    }
}
```

**优势**: 编译时类型检查，避免运行时错误

## 实现细节

### 关键点 1: 矩阵布局和转置

```cpp
// FlashInfer 的关键实现
CuBlasLtMatrixLayout a_layout(a_type, m, k, k, true);  // transpose=true!
```

**为什么需要 transpose=true?**

cuBLASLt 使用 **column-major** 布局（Fortran 风格），而我们的输入是 **row-major**（C 风格）。

```
Row-major (C): A[i][j]
Column-major (Fortran): A[j][i]

当 transpose=true 时:
  输入: [m, k] row-major
  解释: [k, m] column-major (即转置后的矩阵)
```

### 关键点 2: Fast Accumulation

```cpp
int8_t fast_accum = 1;
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum);
```

**为什么需要 fast_accum?**

FP8 Tensor Core 计算流程:
```
FP8 × FP8 → FP32 累加
              ↓
         [Fast Accum]  ← 使用 FP32 累加器
              ↓
    scale_a × scale_b × accumulator
              ↓
         输出 FP16/BF16
```

Fast accumulation 确保累加使用 FP32 而非 FP8，避免精度损失。

### 关键点 3: Scale 指针

```cpp
const void* A_scale_ptr = static_cast<const void*>(d_scale_a);
const void* B_scale_ptr = static_cast<const void*>(d_scale_b);

matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, A_scale_ptr);
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, B_scale_ptr);
```

**Scale 的作用:**

FP8 数据范围有限 (e4m3: [-448, 448])，需要 scale 来调整数值范围。

```
原始 FP8: A_fp8, B_fp8
实际数值: A_fp8 × scale_a, B_fp8 × scale_b
输出: (A_fp8 × scale_a) × (B_fp8 × scale_b) = A_fp8 × B_fp8 × scale_a × scale_b
```

### 关键点 4: 批处理支持

```cpp
if (batch_size > 1) {
    int64_t stride_a = m * k;  // A 批次间的元素偏移
    int64_t stride_b = k * n;  // B 批次间的元素偏移
    int64_t stride_c = m * n;  // C 批次间的元素偏移

    a_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
    a_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a);
    // ... 类似设置 B 和 C
}
```

**内存布局**:
```
batch_size = 2:
  A: [batch=0][m, k] [batch=1][m, k]
  B: [batch=0][k, n] [batch=1][k, n]
  C: [batch=0][m, n] [batch=1][m, n]
```

## 与 FlashInfer 的对比

| 特性 | FlashInfer | Standalone 实现 |
|------|------------|----------------|
| 依赖 | TVM、Python | 仅 cuBLASLt |
| 编译 | JIT (运行时) | AOT (编译时) |
| Backend 选择 | 自动 | 手动指定 |
| 错误处理 | 异常 | 状态码 |
| 调试 | 困难 | 容易 |

## 性能特性

### 1. 算法选择

```cpp
cublasLtMatmulHeuristicResult_t heuristic_result = {};
cublasLtMatmulAlgoGetHeuristic(
    lt_handle,
    matmul_desc.descriptor(),
    a_layout.descriptor(),
    b_layout.descriptor(),
    c_layout.descriptor(),
    c_layout.descriptor(),
    preference.descriptor(),
    1,  // 请求数量
    &heuristic_result,
    &returned_results);
```

**Heuristic 搜索**:
- 根据问题大小自动选择最优算法
- 考虑 Tensor Core 利用率、内存带宽等因素

### 2. Workspace 管理

```cpp
size_t workspace_size = 32 * 1024 * 1024;  // 32 MB
void* d_workspace;
cudaMalloc(&d_workspace, workspace_size);

preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                        static_cast<uint64_t>(workspace_size));
```

**Workspace 用途**:
- 临时缓冲区
- 算法内部使用
- 大小取决于算法选择

## 测试和验证

### 1. 正确性验证

```cpp
// FP8 GEMM
fp8_gemm_cublaslt<__nv_fp8_e4m3, half>(d_A, d_B, d_C_fp8, ...);

// FP16 GEMM (参考)
fp16_gemm_cublas(d_A_fp16, d_B_fp16, d_C_fp16, ...);

// 简单 kernel 参考 (正确性检查)
fp8_gemm_reference_kernel<16, 16><<<grid, block>>>(...);
```

### 2. 预期误差

- FP8 vs FP16: < 1% 误差（正常）
- FP8 vs 参考: < 5% 误差（可接受）

## 使用示例

### 基本用法

```cpp
// 1. 准备数据
__nv_fp8_e4m3 *d_A, *d_B;
half *d_C;
float *d_scale_a, *d_scale_b;

// 2. 创建 handle
cublasLtHandle_t lt_handle;
cublasLtCreate(&lt_handle);

// 3. 分配 workspace
void* d_workspace;
cudaMalloc(&d_workspace, workspace_size);

// 4. 执行 GEMM
fp8_gemm_cublaslt<__nv_fp8_e4m3, half>(
    d_A, d_B, d_C,
    batch_size, m, n, k,
    d_scale_a, d_scale_b,
    lt_handle, d_workspace, workspace_size,
    stream);

// 5. 清理
cublasLtDestroy(lt_handle);
cudaFree(d_workspace);
```

### 批处理

```cpp
// batch_size = 4
fp8_gemm_cublaslt<__nv_fp8_e4m3, half>(
    d_A, d_B, d_C,
    4,  // batch_size
    m, n, k,
    d_scale_a, d_scale_b,
    lt_handle, d_workspace, workspace_size,
    stream);
```

## 扩展性

### 支持新数据类型

```cpp
// 添加 BF16 输出支持
fp8_gemm_cublaslt<__nv_fp8_e4m3, __nv_bfloat16>(
    d_A, d_B, d_C_bf16, ...);
```

### 支持不同 FP8 格式

```cpp
// 使用 e5m2 格式
fp8_gemm_cublaslt<__nv_fp8_e5m2, half>(
    d_A_e5m2, d_B_e5m2, d_C, ...);
```

## 性能优化建议

1. **批处理**: 尽可能使用 batch_size > 1
2. **内存对齐**: 使用 128 字节对齐的内存
3. **Stream 复用**: 在同一个 stream 上执行多个操作
4. **Workspace 调优**: 根据问题大小调整 workspace

## 限制

1. **GPU 架构**: 仅支持 SM89+ (Ada Lovelace 及更高)
2. **CUDA 版本**: 需要 CUDA 11.4+
3. **cuBLAS 版本**: 需要支持 FP8 的 cuBLAS

## 故障排除

### 问题: 编译错误

```
error: 'cublasLtMatmulDescCreate' was not declared
```

**解决方案**: 确保 CUDA 版本 >= 11.4

### 问题: 运行时错误

```
No FP8 GEMM algorithm found (returned_results=0)
```

**解决方案**:
1. 检查 GPU 架构 (需要 SM89+)
2. 更新 CUDA 驱动和工具包
3. 检查 cuBLAS 版本

### 问题: 结果不正确

```
FP8 和 FP16 结果差异过大
```

**解决方案**:
1. 检查 scale 值是否正确
2. 验证矩阵布局 (transpose 标志)
3. 使用参考 kernel 验证

## 参考

- [cuBLASLt 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublaslt-matmul)
- [FlashInfer 实现](https://github.com/flashinfer-ai/flashinfer)
- [CUDA FP8 指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fp8-floating-point)
