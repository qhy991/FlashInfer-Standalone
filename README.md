# FlashInfer-Standalone

# FlashInfer FP8 GEMM 分析与实现

## 概述

本文件夹包含 FlashInver FP8 GEMM 的完整分析、standalone 实现和测试脚本，支持 SM89 (Ada Lovelace/RTX 40xx) 和 SM90 (Hopper/H100/H200) 架构。

## 文件夹结构

```
flashinfer_fp8_analysis/
├── FlashInfer调用链分析.md    # FlashInver FP8 GEMM 完整调用链文档
├── Standalone实现说明.md      # Standalone 实现详细说明
├── README.md                  # 本文件
├── src/
│   ├── fp8_gemm_sm89_standalone.cu  # 基础测试版本
│   └── fp8_gemm_benchmark.cu        # 性能测试版本
└── scripts/
    ├── build_windows.bat      # Windows 编译脚本
    ├── build_linux.sh         # Linux 编译脚本
    ├── test.bat               # Windows 测试脚本
    ├── test.sh                # Linux 测试脚本
    ├── benchmark_compare.py   # 性能对比脚本
    └── run_benchmark.sh       # 一键运行性能对比
```

## 快速开始

### 系统要求

- **GPU**: 
  - SM89: NVIDIA RTX 40 系列 (Ada Lovelace)
  - SM90: NVIDIA H100/H200 (Hopper)
- **CUDA**: 11.4+ (推荐 12.x)
- **cuBLAS**: 支持 FP8 Tensor Core
- **编译器**: 支持 C++17 的编译器

### 编译

#### Windows
```batch
cd C:\Users\infin\Documents\QHY\flashinfer_fp8_analysis
scripts\build_windows.bat
```

#### Linux
```bash
cd /path/to/flashinfer_fp8_analysis
chmod +x scripts/build_linux.sh

# 默认使用 SM90 (H200/H100)
./scripts/build_linux.sh

# 或指定架构
./scripts/build_linux.sh sm_90  # H200/H100
./scripts/build_linux.sh sm_89  # RTX 40xx
```

### 运行测试

#### 基础测试
**Windows**
```batch
scripts\test.bat
```

**Linux**
```bash
chmod +x scripts/test.sh
./scripts/test.sh
```

### 性能对比 (FlashInfer vs Standalone)

#### 安装 FlashInfer（可选）
```bash
pip install flashinfer-python
```

#### 运行性能对比

**方法1: 一键运行（推荐）**
```bash
cd flashinfer_fp8_analysis
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh
```

**方法2: 手动运行**
```bash
# 1. 编译
chmod +x scripts/build_linux.sh
./scripts/build_linux.sh

# 2. 运行性能对比
python3 scripts/benchmark_compare.py
```

#### 预期输出示例
```
======================================================================
性能对比测试环境
======================================================================
GPU: NVIDIA GeForce RTX 4060 (或 H200)
Compute Capability: 8.9 (或 9.0)
Total Memory: 8.0 GB
FlashInfer: 可用
======================================================================

======================================================================
测试: Small (batch=1)
配置: batch=1, m=128, n=64, k=128
计算量: 2.05 GFLOPS
======================================================================

[FlashInfer]
  性能: 45.2 GFLOPS
  延迟: 0.045 ms
  输出: (1, 128, 64), dtype=torch.float16

[Standalone]
  性能: 44.8 GFLOPS
  延迟: 0.046 ms

======================================================================
性能总结
======================================================================

配置                        FlashInfer           Standalone           差异
---------------------------------------------------------------------------
128x64x128 (batch=1)        45.2 GFLOPS          44.8 GFLOPS          -0.9% ✓
256x128x256 (batch=1)       52.1 GFLOPS          51.5 GFLOPS          -1.2% ✓
512x256x512 (batch=1)       58.3 GFLOPS          57.9 GFLOPS          -0.7% ✓

======================================================================

【性能分析】
1. Standalone 实现和 FlashInfer 使用相同的 cuBLASLt API
2. 两者都使用 SM89/SM90 Tensor Core 进行 FP8 计算
3. 理论上性能应该一致（误差 < 5%）

【建议】
- Standalone: 适合学习、调试、自定义修改
- FlashInfer:  适合生产环境（功能更丰富）
```

## 测试结果

### 预期输出

```
===============================================================
FP8 GEMM Test (FlashInfer-style implementation)
Supports: SM89 (Ada/RTX 40xx) and SM90 (Hopper/H100/H200)
===============================================================

GPU: NVIDIA GeForce RTX 4060 Laptop GPU (或 H200)
Compute Capability: 8.9 (或 9.0)

===============================================================
Test: batch=1, m=128, n=64, k=128
===============================================================

[Test 1] FP8 GEMM with cuBLASLt (FlashInfer method)...
  SUCCESS! FP8 GEMM with cuBLASLt worked!
  First 5 output values:
    C_fp8[0] = 41.687500
    C_fp8[1] = 41.187500
    C_fp8[2] = 40.656250
    C_fp8[3] = 40.156250
    C_fp8[4] = 39.718750

[Test 2] FP16 GEMM with cuBLAS (baseline)...
  SUCCESS! FP16 GEMM worked
  First 5 output values:
    C_fp16[0] = 41.718750
    C_fp16[1] = 41.187500
    C_fp16[2] = 40.656250
    C_fp16[3] = 40.156250
    C_fp16[4] = 39.718750

[Comparison] FP8 vs FP16 vs Reference
  First element comparison:
    FP8 (cuBLASLt):  41.687500
    FP16 (cuBLAS):    41.718750
    Reference:        41.703125
    差异:              0.031250 (非常小！)
```

## 核心概念

### 架构支持

| 架构 | GPU 型号 | Compute Capability | Backend |
|------|----------|-------------------|---------|
| SM89 | RTX 40xx 系列 | 8.9 | cuBLAS/cuDNN |
| SM90 | H100, H200 | 9.0 | cuBLAS/cuDNN |
| SM100+ | Blackwell | 10.0+ | CUTLASS 3.x |

**注意**: SM89 和 SM90 都使用 cuBLAS/cuDNN backend，代码实现相同。SM100+ 使用不同的 CUTLASS 3.x backend。

### FP8 GEMM 计算流程

```
输入: FP8 (e4m3/e5m2)
  ↓
Tensor Core: FP8 × FP8 → FP32 累加
  ↓
Epilogue: scale_a × scale_b × accumulator
  ↓
输出: FP16/BF16
```

## 关键实现细节

### 1. 矩阵布局配置

```cpp
// 关键: transpose=true 参数
CuBlasLtMatrixLayout a_layout(a_type, m, k, k, true);  // A 转置
CuBlasLtMatrixLayout b_layout(b_type, k, n, k);          // B 不转置
CuBlasLtMatrixLayout c_layout(c_type, m, n, m);          // C
```

### 2. FP8 特殊属性

```cpp
// 快速累加 (FP8 必需)
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, 1);

// Scale 指针 (FP8 必需)
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, scale_a);
matmul_desc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, scale_b);
```

### 3. 批处理支持

```cpp
if (batch_size > 1) {
    int64_t stride_a = m * k;
    int64_t stride_b = k * n;
    int64_t stride_c = m * n;

    a_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
    a_layout.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a);
    // ... 类似地设置 B 和 C
}
```

## 性能分析

### 理论性能

- **SM89 理论峰值**: ~150 TFLOPS (FP8)
  - RTX 4090: ~330 TFLOPS (理论), ~100-150 TFLOPS (实际)
  - RTX 4060: ~100 TFLOPS (理论), ~40-60 TFLOPS (实际)
- **SM90 理论峰值**: ~200+ TFLOPS (FP8)
  - H200: ~400+ TFLOPS (理论), ~150-200 TFLOPS (实际)
  - H100: ~300+ TFLOPS (理论), ~100-150 TFLOPS (实际)

### 性能对比

由于 FlashInfer 和 standalone 实现都使用相同的 cuBLASLt API 和底层 Tensor Core，**性能应该是一致的**。

## 常见问题

### Q1: 编译失败，提示找不到 cuBLASLt

**A**: 确保 CUDA 版本 >= 11.4，并正确安装了 cuBLAS。

### Q2: 运行时错误 "No FP8 GEMM algorithm found"

**A**: 可能的原因：
- GPU 不支持 FP8 Tensor Core (需要 SM89+，即 RTX 40xx 或 H100/H200)
- CUDA/cuBLAS 版本过旧（推荐 CUDA 12.0+）
- 驱动程序不支持 FP8

### Q3: FP8 和 FP16 结果不一致

**A**: FP8 (e4m3) 只有 4 位尾数，精度损失是正常的。允许误差在 1-5% 以内。

### Q4: FlashInfer Windows JIT 编译失败

**A**: 这是 FlashInfer 在 Windows 上的已知问题，不影响 cuBLASLt backend 的 FP8 功能。

## 参考

- [cuBLASLt API 文档](https://docs.nvidia.com/cuda/cublas/index.html)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [CUDA FP8 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fp8-floating-point)

## 许可

本 standalone 实现基于 FlashInver 的实现，仅供学习和研究使用。
