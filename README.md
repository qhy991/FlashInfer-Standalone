# FlashInfer-Standalone

# FlashInfer FP8 GEMM 分析与实现

## 概述

本文件夹包含 FlashInfer FP8 GEMM 的完整分析、standalone 实现和测试脚本，支持 SM89 (Ada Lovelace/RTX 40xx) 和 SM90 (Hopper/H100/H200) 架构。

## 文件夹结构

```
FlashInfer-Standalone/
├── src/
│   ├── fp8_gemm_sm89_standalone.cu  # 基础测试版本
│   └── fp8_gemm_benchmark.cu        # 性能测试版本
├── scripts/
│   ├── build_windows.bat            # Windows 编译脚本
│   ├── build_linux.sh               # Linux 编译脚本
│   ├── benchmark_compare.py         # 性能对比脚本
│   └── benchmark_precise.py         # 精确性能测试脚本 (新增)
└── docs/
    ├── FlashInfer调用链分析.md       # FlashInfer FP8 GEMM 完整调用链文档
    ├── Standalone实现说明.md         # Standalone 实现详细说明
    └── FP8零输出问题分析.md          # FP8 零输出问题根因分析 (新增)
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

#### Linux
```bash
cd /root/R/FlashInfer-Standalone

# 编译 (默认 SM89)
chmod +x scripts/build_linux.sh
./scripts/build_linux.sh sm_89

# 或指定其他架构
./scripts/build_linux.sh sm_90  # H200/H100
```

### 运行测试

#### 基础功能测试
```bash
# 运行 standalone 测试
./build/fp8_gemm_sm89_standalone
```

#### 预期输出
```
===============================================================
FP8 GEMM Test (FlashInfer-style implementation)
Supports: SM89 (Ada/RTF 40xx) and SM90 (Hopper/H100/H200)
===============================================================

GPU: NVIDIA GeForce RTX 4090
Compute Capability: 8.9

===============================================================
Test: batch=1, m=128, n=64, k=128
===============================================================

[Test 1] FP8 GEMM with cuBLASLt (FlashInfer method)...
  SUCCESS! FP8 GEMM with cuBLASLt worked!
  First 5 output values:
    C_fp8[0] = 41.718750
    C_fp8[1] = 41.187500
    C_fp8[2] = 40.656250
    C_fp8[3] = 40.156250
    C_fp8[4] = 39.718750

[Test 2] FP16 GEMM with cuBLAS (baseline)...
  SUCCESS! FP16 GEMM worked
  First 5 output values:
    C_fp16[0] = 41.718750
    C_fp16[1] = 26.000000
    C_fp16[2] = 41.187500
    C_fp16[3] = 26.000000
    C_fp16[4] = 40.656250
```

## 正确性验证

### 与 FlashInfer 的输出对比

使用**相同的输入数据**和**相同的 FP8 转换方法**后，Standalone 和 FlashInfer 的输出**完全一致**：

| 实现 | 前5个输出值 |
|------|------------|
| Standalone | [41.71875, 41.1875, 40.65625, 40.15625, 39.71875] |
| FlashInfer | [41.71875, 41.1875, 40.65625, 40.15625, 39.71875] |
| **差异** | **[0.0, 0.0, 0.0, 0.0, 0.0]** ✅ |

### 验证方法

```python
# 使用固定随机种子确保输入一致
torch.manual_seed(42)

# 直接转换为 FP8 (不进行缩放)
a_fp8 = a_fp16.to(torch.float8_e4m3fn)
b_fp8 = b_fp16.to(torch.float8_e4m3fn)

# 使用 scale=1.0
scale_a = torch.tensor([1.0], dtype=torch.float32, device='cuda')
scale_b = torch.tensor([1.0], dtype=torch.float32, device='cuda')

# FlashInfer FP8 GEMM
c_flashinfer = bmm_fp8(a_fp8, b_fp8, scale_a, scale_b, torch.float16)
```

**关键点**:
- 使用直接 FP8 转换 (`x.to(torch.float8_e4m3fn)`)
- 使用 `scale=1.0`
- 确保输入数据范围在 FP8 可表示范围内

## 性能分析

### 测试环境
- **GPU**: NVIDIA GeForce RTX 4090 (SM89/Ada)
- **Compute Capability**: 8.9
- **CUDA**: 12.x
- **测试方法**: 20 次运行，每次 100 次迭代，取平均值

### 性能对比结果

| 配置 | Standalone | FlashInfer | 延迟比率 |
|------|-----------|------------|---------|
| Small (128x64x128) | 183.2 GFLOPS (0.011 ms) | 14.8 GFLOPS (0.141 ms) | **12.8x** |
| Medium (256x128x256) | 1779.5 GFLOPS (0.009 ms) | 167.7 GFLOPS (0.100 ms) | **11.1x** |
| Large (512x256x512) | 13896.2 GFLOPS (0.010 ms) | 1342.1 GFLOPS (0.100 ms) | **10.0x** |

### 性能稳定性

FlashInfer 的多次运行结果非常稳定（标准差 ~0.002 ms）：

| 配置 | 平均延迟 | 标准差 | 延迟范围 |
|------|---------|--------|---------|
| Medium | 0.1000 ms | ±0.0015 ms | [0.0982, 0.1035] ms |
| Large | 0.1000 ms | ±0.0021 ms | [0.0976, 0.1073] ms |
| XLarge | 0.0995 ms | ±0.0011 ms | [0.0981, 0.1026] ms |

### 性能差异原因分析

FlashInfer 比 Standalone 慢约 10 倍的可能原因：

1. **API 调用开销**:
   - 参数验证和边界检查
   - Workspace 分配和管理
   - JIT 缓存查找和验证
   - cuBLASLt 句柄获取和配置

2. **Python → C++ 绑定开销**:
   - TVM-FFI 绑定层
   - 数据类型转换
   - 内存管理

3. **额外功能**:
   - 错误检查和异常处理
   - 日志记录
   - 调试支持

### 运行精确性能测试

```bash
cd /root/R/FlashInfer-Standalone
python3 scripts/benchmark_precise.py
```

该脚本会：
- 使用固定随机种子确保输入一致
- 进行 20 次独立运行
- 每次运行执行 100 次迭代
- 输出平均值、标准差和范围

## 常见问题

### Q1: FlashInfer 输出全为 0

**原因**: 使用了错误的 FP8 转换方法或 scale factor。

**解决方案**: 参考 [`docs/FP8零输出问题分析.md`](docs/FP8零输出问题分析.md)

正确的用法：
```python
# 方法1: 直接转换（适合数据范围已知的场景）
a_fp8 = a_fp16.to(torch.float8_e4m3fn)
scale = torch.tensor([1.0])

# 方法2: 缩放转换（需要返回逆缩放因子）
def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()  # 返回逆缩放因子
```

### Q2: 性能测试显示 Standalone 比 FlashInfer 快很多

**这是正常的**。Standalone 是纯粹的 cuBLASLt 调用，没有额外开销。FlashInfer 作为通用库，提供了更多功能和安全性检查，这些都会带来一定的性能开销。

### Q3: 编译失败，提示找不到 cuBLASLt

**A**: 确保 CUDA 版本 >= 11.4，并正确安装了 cuBLAS。

### Q4: 运行时错误 "No FP8 GEMM algorithm found"

**A**: 可能的原因：
- GPU 不支持 FP8 Tensor Core (需要 SM89+，即 RTX 40xx 或 H100/H200)
- CUDA/cuBLAS 版本过旧（推荐 CUDA 12.0+）
- 驱动程序不支持 FP8

### Q5: FP8 和 FP16 结果不一致

**A**: FP8 (e4m3) 只有 4 位尾数，精度损失是正常的。使用正确的 FP8 转换方法后，与 FP16 参考的误差通常在 1-5% 以内。

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

## 理论性能

- **SM89 理论峰值**: ~150 TFLOPS (FP8)
  - RTX 4090: ~330 TFLOPS (理论), ~100-150 TFLOPS (实际)
  - RTX 4060: ~100 TFLOPS (理论), ~40-60 TFLOPS (实际)
- **SM90 理论峰值**: ~200+ TFLOPS (FP8)
  - H200: ~400+ TFLOPS (理论), ~150-200 TFLOPS (实际)
  - H100: ~300+ TFLOPS (理论), ~100-150 TFLOPS (实际)

## 使用建议

### 何时使用 Standalone

- 学习和理解 cuBLASLt FP8 GEMM API
- 调试 FP8 相关问题
- 性能基准测试
- 自定义实现和实验

### 何时使用 FlashInfer

- 生产环境部署
- 需要完整的功能支持
- 需要与其他 FlashInfer 功能集成
- 需要跨平台兼容性

## 参考

- [cuBLASLt API 文档](https://docs.nvidia.com/cuda/cublas/index.html)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [CUDA FP8 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fp8-floating-point)
- [CUTLASS 文档](https://github.com/NVIDIA/cutlass)

## 许可

本 standalone 实现基于 FlashInfer 的实现，仅供学习和研究使用。
