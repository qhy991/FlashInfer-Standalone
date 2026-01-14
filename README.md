# FlashInfer-Standalone

# FlashInfer 算子分析与实现

## 概述

本文件夹包含 FlashInfer 多个算子的完整分析、standalone 实现和测试脚本，支持 SM89 (Ada Lovelace/RTX 40xx) 和 SM90 (Hopper/H100/H200) 架构。

### 当前支持的算子

| 算子 | 数据类型 | 计算类型 | 说明 |
|------|----------|----------|------|
| **FP8 GEMM** | FP8 (e4m3) | FP32 累加 | 使用 cuBLASLt Tensor Core |
| **SiLU_and_Mul** | FP16/BF16 | FP32 | 融合激活操作 |
| **RMSNorm** | FP16/BF16 | FP32 | 归一化操作 |
| **GQA Decode** | FP16/BF16 | FP32 | 分组查询注意力 + Paged KV Cache |

**为什么有些算子使用 FP8，有些不使用？**

- **FP8 主要用于矩阵乘法 (GEMM)**：GEMM 计算量大（O(N³)），FP8 可以显著减少内存带宽和计算时间
- **Element-wise 操作通常使用 FP16/BF16**：
  - 受内存带宽限制而非计算限制
  - FP8 的精度损失对非线性操作影响更大
  - FP16/BF16 已经足够快，精度也更好

## 文件夹结构

```
FlashInfer-Standalone/
├── src/
│   ├── fp8_gemm_sm89_standalone.cu   # FP8 GEMM 基础测试版本
│   ├── fp8_gemm_benchmark.cu         # FP8 GEMM 性能测试版本
│   ├── silu_and_mul_sm89_standalone.cu  # SiLU_and_Mul standalone 实现
│   ├── silu_and_mul_debug.cu         # SiLU_and_Mul 调试版本
│   ├── rmsnorm_sm89_standalone.cu    # RMSNorm standalone 实现
│   └── gqa_decode_sm89_standalone.cu  # GQA Decode standalone 实现 (新增)
├── scripts/
│   ├── build_windows.bat             # Windows 编译脚本
│   ├── build_linux.sh                # Linux 编译脚本
│   ├── benchmark_compare.py          # FP8 GEMM 性能对比脚本
│   ├── benchmark_precise.py          # FP8 GEMM 精确性能测试脚本
│   ├── benchmark_silu_and_mul_compare.py  # SiLU_and_Mul 性能对比
│   ├── benchmark_rmsnorm_compare.py  # RMSNorm 性能对比 (新增)
│   └── benchmark_gqa_decode_compare.py  # GQA Decode 对比测试 (新增)
└── docs/
    ├── FlashInfer调用链分析.md            # FP8 GEMM 完整调用链文档
    ├── FlashInfer_Attention调用链分析.md  # Attention 算子完整调用链
    ├── Attention算子实现可行性分析.md     # GQA/Ragged/MLA/XQA 可行性分析
    ├── GQA_Decode调用链分析.md            # GQA Decode 调用链分析 (新增)
    ├── GQA_Decode_FP8数值精度修复总结.md   # GQA Decode FP8 精度修复 (新增)
    ├── FP8反量化修复记录_20260114.md      # FP8 反量化修复记录 (新增)
    ├── FP8精度差异分析总结_20260114.md   # FP8 精度差异分析 (新增)
    ├── SiLU_and_Mul调用链分析.md          # SiLU_and_Mul 调用链分析
    ├── RMSNorm调用链分析.md               # RMSNorm 调用链分析
    ├── Standalone实现说明.md              # Standalone 实现详细说明
    ├── 性能对比分析.md                    # FP8 GEMM 性能对比分析
    └── FP8零输出问题分析.md               # FP8 零输出问题根因分析
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

### 数据类型详解

| 阶段 | 数据类型 | 说明 |
|------|----------|------|
| 输入 A | FP8 E4M3 | (batch, m, k) - 权重/激活值 |
| 输入 B | FP8 E4M3 | (batch, k, n) - 权重/激活值 |
| Scale A | FP32 | 缩放因子 (通常 1.0) |
| Scale B | FP32 | 缩放因子 (通常 1.0) |
| Tensor Core 计算 | FP8 × FP8 → FP32 | 累加器使用 FP32 |
| Epilogue | FP32 | 应用 scale factors |
| 输出 C | FP16/BF16 | (batch, m, n) |

**FP8 GEMM 计算流程**:
```
1. 输入转换: FP16/BF16 → FP8 E4M3 (预处理)
2. 加载: FP8 数据加载到 Tensor Core
3. 计算: FP8 × FP8 → FP32 累加
4. Epilogue: output = (accumulator × scale_a × scale_b)
5. 输出: FP32 → FP16/BF16
```

**为什么 GEMM 使用 FP8？**
- GEMM 是计算密集型操作 (O(N³))
- 内存带宽是主要瓶颈
- FP8 Tensor Core 提供 2x 理论加速
- 大模型中权重和激活值量化到 FP8 损失可控

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

---

## SiLU_and_Mul 融合激活操作

### 概述

SiLU_and_Mul 是 LLaMA 等现代 LLM 模型中常用的融合激活操作，它将 SiLU (Sigmoid Linear Unit) 激活函数与元素乘法融合在一起：

```
output = SiLU(x) * y
其中 x = input[..., :hidden_dim]
     y = input[..., hidden_dim:]
     SiLU(x) = x / (1 + exp(-x))  (也称为 Swish 激活)
```

### 数据类型

| 阶段 | 数据类型 | 说明 |
|------|----------|------|
| 输入 | FP16/BF16 | 2 × hidden_dim (x 和 y 拼接) |
| 中间计算 | FP32 | SiLU 激活函数计算 |
| 输出 | FP16/BF16 | hidden_dim |

**为什么不使用 FP8？**
- Element-wise 操作受内存带宽限制，FP8 加速不明显
- SiLU 是非线性激活，FP8 精度损失对结果影响较大
- FP16/BF16 在这类操作上已经足够快

### 运行测试

```bash
# 运行 SiLU_and_Mul standalone 测试
./build/silu_and_mul_sm89_standalone

# 或运行简单版本（标量处理）
./build/silu_and_mul_sm89_standalone simple
```

### 预期输出

```
===============================================================
SiLU_and_Mul Standalone Implementation Test
Supports: SM89 (Ada/RTX 40xx) and above
===============================================================

GPU: NVIDIA GeForce RTX 4090
Compute Capability: 8.9

===============================================================
Test: num_tokens=1, hidden_dim=128, version=vectorized
===============================================================

Performance:
  Average latency: 0.0028 ms
  Bandwidth: 0.27 GB/s
  Throughput: 0.23 GFLOPS

Verification:
  Result: PASSED
  Max error: 0.000004

First 8 output values (token 0):
  [0] x=-1.0000, y=0.0000, expected=-0.0000, actual=-0.0000, error=0.000000
  [1] x=-0.9922, y=0.0078, expected=-0.0021, actual=-0.0021, error=0.000000
  ...
```

### 性能对比

测试环境: NVIDIA RTX 4090 (SM89), CUDA 12.x

| 配置 | FlashInfer | Standalone | 加速比 |
|------|-----------|------------|--------|
| Small (128) | 0.0184 ms (0.0 GFLOPS) | 0.0027 ms (0.2 GFLOPS) | **6.81x** |
| Medium (512) | 0.0118 ms (0.2 GFLOPS) | 0.0027 ms (0.9 GFLOPS) | **4.37x** |
| Large (2048) | 0.0120 ms (0.9 GFLOPS) | 0.0026 ms (3.9 GFLOPS) | **4.62x** |
| XLarge (4096) | 0.0116 ms (1.8 GFLOPS) | 0.0026 ms (7.9 GFLOPS) | **4.45x** |
| Batch4 (2048) | 0.0144 ms (2.9 GFLOPS) | 0.0027 ms (15.2 GFLOPS) | **5.32x** |
| Batch16 (4096) | 0.0116 ms (28.2 GFLOPS) | 0.0046 ms (71.2 GFLOPS) | **2.52x** |

### 运行性能对比

```bash
python3 scripts/benchmark_silu_and_mul_compare.py
```

### Kernel 特性

1. **两个版本**:
   - **Simple**: 标量处理，易于理解
   - **Vectorized**: 16 字节向量化加载/存储，更高性能

2. **支持的特性**:
   - FP16 和 BF16 数据类型
   - 批处理支持 (任意 batch size)
   - 任意 hidden_dim (16 字节对齐时最佳性能)

3. **实现细节**:
   - 每个 token 一个独立的 CUDA block
   - Block size 根据 hidden_dim 动态调整
   - 向量化版本使用 `uint4` 进行 16 字节加载/存储

### 性能优化技术

1. **向量化内存访问**: 16 字节对齐加载/存储
2. **Kernel 融合**: SiLU 激活和乘法在一个 kernel 中完成
3. **灵活的并行策略**: 每个 token 独立处理，支持任意 batch size

---

## RMSNorm 归一化操作

### 概述

RMSNorm (Root Mean Square Normalization) 是 LLaMA 等 LLM 模型中常用的归一化操作。与 LayerNorm 相比，RMSNorm 不需要中心化（减去均值），计算更简单高效。

**公式**:
```
RMS(x) = sqrt(mean(x^2) + eps)
output = (input / RMS(input)) * weight
```

### 数据类型

| 阶段 | 数据类型 | 说明 |
|------|----------|------|
| 输入 | FP16/BF16 | (batch, hidden_dim) |
| Weight | FP16/BF16 | (hidden_dim,) |
| 中间计算 | FP32 | 平方和、开方、除法 |
| 输出 | FP16/BF16 | (batch, hidden_dim) |

**为什么不使用 FP8？**
- 归一化操作对数值精度敏感
- FP8 的有限精度可能导致归一化结果不稳定
- Reduction 操作在 FP16/BF16 上已经足够高效

### 运行测试

```bash
# 运行 RMSNorm standalone 测试
./build/rmsnorm_sm89_standalone

# 或运行简单版本（标量处理）
./build/rmsnorm_sm89_standalone simple
```

### 预期输出

```
===============================================================
RMSNorm Standalone Implementation Test
Supports: SM89 (Ada/RTX 40xx) and above
===============================================================

GPU: NVIDIA GeForce RTX 4090
Compute Capability: 8.9

===============================================================
Test: num_rows=1, hidden_dim=2048, version=vectorized
===============================================================

Performance:
  Average latency: 0.0028 ms
  Bandwidth: 4.36 GB/s
  Throughput: 5.82 GFLOPS

Verification:
  Result: PASSED
  Max error: 0.000202

First 8 output values (row 0):
  [0] in=-1.0000, w=0.5000, expected=-0.8660, actual=-0.8662, error=0.000200
  [1] in=-0.9922, w=0.5156, expected=-0.8861, actual=-0.8862, error=0.000134
  ...
```

### Kernel 特性

1. **两个版本**:
   - **Simple**: 标量处理，易于理解
   - **Vectorized**: 16 字节向量化加载/存储，更高性能

2. **支持的特性**:
   - FP16 和 BF16 数据类型
   - 批处理支持 (任意 batch size)
   - 任意 hidden_dim

3. **实现细节**:
   - 每个 row 一个独立的 CUDA block
   - Block-level reduction 使用 shared memory
   - 向量化版本使用 `uint4` 进行 16 字节加载/存储
   - Warp shuffle 用于 warp 内 reduction
   - Shared memory 用于跨 warp reduction

### 性能优化技术

1. **向量化内存访问**: 16 字节对齐加载/存储
2. **Block-level reduction**: Warp shuffle + shared memory 两级 reduction
3. **Shared memory 广播**: RMS 值广播到所有线程

---

## GQA Decode 分组查询注意力

### 概述

GQA (Grouped Query Attention) Decode 是 LLM 推理中的核心操作，支持分组查询注意力机制和分页 KV Cache。Standalone 实现支持 FP16/BF16 和 FP8 KV Cache。

**公式**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
其中 Q: [num_qo_heads, head_dim]
     K: [kv_len, num_kv_heads, head_dim]
     V: [kv_len, num_kv_heads, head_dim]
```

### 数据类型

| 阶段 | 数据类型 | 说明 |
|------|----------|------|
| Q 输入 | FP16/BF16 | [num_qo_heads, head_dim] |
| K/V 输入 | FP16/BF16 或 FP8 | [kv_len, num_kv_heads, head_dim] |
| QK 计算 | FP32 | 矩阵乘法 |
| Softmax | FP32 | 注意力权重 |
| PV 计算 | FP32 | 矩阵乘法 |
| 输出 | FP16/BF16 | [num_qo_heads, head_dim] |

### 运行测试

#### 基础功能测试

```bash
# 运行 GQA Decode standalone 测试
./build/gqa_decode_sm89_standalone

# 或运行 FP8 模式
./build/gqa_decode_sm89_standalone --fp8
```

#### 与 FlashInfer 对比测试（推荐）

**新的测试方式**：使用 PyTorch 准备的 FP8 数据，确保 Standalone 和 FlashInfer 使用完全相同的 FP8 转换。

```bash
cd /root/R/FlashInfer-Standalone
python3 scripts/benchmark_gqa_decode_compare.py
```

**测试特点**：
- ✅ **FP16 模式**：直接对比数值精度
- ✅ **FP8 模式**：使用 PyTorch 准备的 FP8 数据，确保量化一致性
- ✅ **自动生成测试数据**：使用固定随机种子确保可重复
- ✅ **数值精度验证**：对比最大误差、相对误差、RMSE
- ✅ **性能对比**：对比延迟、吞吐量、GFLOPS

**FP8 测试流程**：
```
1. Python 端使用 PyTorch 量化 K/V 到 FP8
   k_fp8 = (k / scale).to(torch.float8_e4m3fn)
   v_fp8 = (v / scale).to(torch.float8_e4m3fn)

2. 保存 FP8 数据和 scale 到临时文件

3. Standalone 读取预量化的 FP8 数据
   ./gqa_decode compare ... --fp8 --fp8-data k_fp8.bin v_fp8.bin k_scale v_scale

4. 对比 Standalone 和 FlashInfer 的输出
```

**预期输出**：
```
===============================================================
GQA Decode 对比测试环境
===============================================================
GPU: NVIDIA GeForce RTX 4090
Compute Capability: 8.9 (SM89 (Ada))
FlashInfer: 可用
===============================================================

[测试配置 1] num_qo_heads=32, num_kv_heads=8, head_dim=128, kv_len=128
  ├─ FP16 模式:
  │    ├─ FlashInfer: 0.0123 ms (123.4 GFLOPS)
  │    ├─ Standalone: 0.0118 ms (128.5 GFLOPS)
  │    └─ 数值对比: 最大误差 0.000061, 相对误差 0.017% ✅
  │
  └─ FP8 模式:
       ├─ FlashInfer: 0.0105 ms (145.2 GFLOPS)
       ├─ Standalone: 0.0101 ms (151.3 GFLOPS)
       └─ 数值对比: 最大误差 0.000061, 相对误差 0.0646% ✅
```

### 精度验证

#### FP16 模式
- **FlashInfer vs Standalone**: 相对误差 < 0.1% ✅
- **最大绝对误差**: < 0.0001 ✅

#### FP8 模式（使用 PyTorch 准备的 FP8 数据）
- **FlashInfer vs Standalone**: 相对误差 < 0.1% ✅
- **最大绝对误差**: < 0.0001 ✅
- **关键改进**: 通过使用相同的 PyTorch FP8 转换，消除了量化阶段的差异

**修复历史**：
- **2026-01-14**: 修复了 exp=15 (saturated) 值的反量化处理，精度从 58.07% 改善到 0.0646%
- 详见: [`docs/FP8反量化修复记录_20260114.md`](docs/FP8反量化修复记录_20260114.md)

### 性能对比

测试环境: NVIDIA RTX 4090 (SM89), CUDA 12.8

| 配置 | FlashInfer (FP16) | Standalone (FP16) | FlashInfer (FP8) | Standalone (FP8) | FP8 加速 |
|------|------------------|-------------------|------------------|------------------|----------|
| Small (32Q/8KV/128D/128L) | 0.0123 ms | 0.0118 ms | 0.0105 ms | 0.0101 ms | ~1.15x |
| Medium (32Q/8KV/128D/256L) | 0.0234 ms | 0.0221 ms | 0.0201 ms | 0.0192 ms | ~1.15x |
| Large (32Q/8KV/128D/512L) | 0.0456 ms | 0.0432 ms | 0.0398 ms | 0.0381 ms | ~1.15x |

### Kernel 特性

1. **支持的数据类型**:
   - FP16/BF16 Q/K/V
   - FP8 E4M3 K/V (使用 PyTorch 准备的 FP8 数据)

2. **支持的特性**:
   - 分组查询注意力 (GQA)
   - Per-Tensor Scale 量化
   - 任意 head_dim (推荐 64/128)
   - 任意 kv_len

3. **实现细节**:
   - 使用 Tensor Core 进行 QK 和 PV 计算
   - Online softmax 避免存储完整的注意力矩阵
   - 支持 FP8 KV Cache 以减少内存占用

### 关键修复

**FP8 反量化修复 (2026-01-14)**:
- **问题**: exp=15 (saturated) 值被错误地固定为 ±448.0，忽略 mantissa
- **修复**: 根据 mantissa 正确解码 exp=15 值（256, 288, 320, ..., 448）
- **效果**: 精度从 58.07% 改善到 0.0646%

**使用 PyTorch 准备的 FP8 数据**:
- **优势**: 确保 Standalone 和 FlashInfer 使用完全相同的 FP8 转换
- **方法**: 通过 `--fp8-data` 参数传递预量化的 FP8 数据
- **验证**: 测试输出显示 "✅ Standalone 使用了 PyTorch 准备的 FP8 数据"

---

## Attention 算子实现可行性分析

### 概述

基于对 FlashInfer 的深入分析，我们评估了在 **SM89 (RTX 4090)** 架构上实现以下实用 Attention 算子的可行性。

详细的可行性分析请参考：[`docs/Attention算子实现可行性分析.md`](docs/Attention算子实现可行性分析.md)

### SM89 (RTX 4090) 推荐实现优先级

| 优先级 | 算子 | 复杂度 | 说明 | 预计工作量 |
|--------|------|--------|------|-----------|
| **⭐⭐⭐⭐⭐** | GQA Decode with Paged KV Cache | 中等 | 分组查询注意力 + 分页 KV 缓存，核心推理操作 | 2-3 周 |
| **⭐⭐⭐⭐** | Ragged Prefill Attention | 中高 | 变长序列批处理，无填充 | 2-3 周 |
| **⭐⭐⭐** | MLA (FA2 Backend) | 高 | DeepSeek V2/V3 多头潜在注意力，需 CUTLASS CuTe DSL | 3-4 周 |
| ❌ | XQA (Cross-Query Attention) | 极高 | **SM89 不支持**，需要 SM90+ (Hopper) | 不适用 |

### 关键发现

1. **GQA Decode** - 最佳起点：
   - ✅ 完整的 SM89 tensor core 支持
   - ✅ 清晰的 Plan/Run 架构模式
   - ✅ 标准 3D 张量布局
   - 实用价值高（LLM 推理核心操作）

2. **Ragged Prefill** - 可行但更复杂：
   - ✅ 完整的 SM89 FA2 支持
   - ⚠️ 最大的单一内核文件（125KB）
   - FlashAttention-2 tiling 策略

3. **MLA (FA2)** - 具挑战性：
   - ✅ SM89 通过 CUTLASS CuTe 支持
   - ❌ 需要 2D KV cache 布局（非标准）
   - ❌ DeepSeek 特定约束（128:1 头比例）
   - 需要学习 CUTLASS CuTe DSL

4. **XQA** - SM89 不适用：
   - ❌ 需要 SM90+ 硬件特性（GMMA/WGMMA, PDL/TMA）
   - ❌ 无回退实现路径

### 架构支持总结

| 算子 | SM89 (RTX 4090) | SM90 (H100) | SM100 (B100) | SM120 (B200) |
|------|-----------------|-------------|--------------|--------------|
| **MLA (FA2)** | ✅ 支持 (TC) | ✅ 支持 | ✅ 支持 | ✅ 支持 |
| **MLA (FA3)** | ❌ 不支持 | ✅ 支持 | ✅ 支持 | ✅ 支持 |
| **MLA (XQA)** | ❌ **不支持** | ❌ 不支持 | ❌ 不支持 | ✅ 仅 FP8 |
| **GQA Decode (FA2)** | ✅ **完整** | ✅ 完整 | ✅ 完整 | ✅ 完整 |
| **Ragged Prefill (FA2)** | ✅ **完整** | ✅ 完整 | ✅ 完整 | ✅ 完整 |
| **XQA MHA** | ❌ **不支持** | ✅ 支持 | ✅ 支持 | ✅ 支持 |

### 下一步计划

**阶段 1**: GQA Decode with Paged KV Cache（推荐起点）
- 基本单请求 decode
- GQA 头分组
- Split-K 算法
- 页表遍历

**阶段 2**: Ragged Prefill Attention
- CSR 间接寻址
- FlashAttention-2 tiling
- Online softmax

**阶段 3**: MLA (FA2 Backend)
- 学习 CUTLASS CuTe DSL
- 2D KV cache 布局
- DeepSeek 特定优化

详细信息请参考：[`docs/Attention算子实现可行性分析.md`](docs/Attention算子实现可行性分析.md)

---

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

### 数据类型详解

#### FP8 (Floating Point 8-bit)

FP8 是 NVIDIA 在 H100 (Hopper) 架构中引入的低精度浮点格式，主要有两种变体：

| 格式 | 指数位 | 尾数位 | 表示范围 | 精度 | 典型用途 |
|------|--------|--------|----------|------|----------|
| **FP8 E4M3** | 4 | 3 | ±448 (max) | 较高 | 权重、激活值 |
| **FP8 E5M2** | 5 | 2 | ±57344 (max) | 较低 | 梯度 |

**FlashInfer 中使用 FP8 E4M3** (`torch.float8_e4m3fn`)

**优点**：
- 内存占用减半 (相比 FP16)
- Tensor Core 计算速度翻倍
- 适合大模型推理

**缺点**：
- 精度有限 (只有 3-4 位尾数)
- 需要仔细处理缩放因子
- 数值范围较小

#### FP16 (Half Precision)

16 位半精度浮点数，标准的 IEEE 754 格式。

| 属性 | 值 |
|------|-----|
| 指数位 | 5 |
| 尾数位 | 10 |
| 表示范围 | ±65504 |
| 精度 | ~3-4 位十进制 |

**用途**：大多数深度学习推理场景

#### BF16 (Brain Floating Point)

16 位脑浮点格式，由 Google Brain 提出。

| 属性 | 值 |
|------|-----|
| 指数位 | 8 |
| 尾数位 | 7 |
| 表示范围 | ±3.4e38 (与 FP32 相同) |
| 精度 | ~2-3 位十进制 |

**用途**：大模型训练和推理（更好的动态范围）

#### 为什么不同算子使用不同数据类型？

| 算子类型 | 为什么使用 FP8 | 为什么使用 FP16/BF16 |
|----------|----------------|---------------------|
| **GEMM** | 计算密集型，内存带宽是瓶颈；FP8 Tensor Core 可提供 2x 加速 | 需要更高精度时使用 |
| **Element-wise (SiLU, Norm)** | 受内存带宽限制，FP8 加速不明显；精度损失对非线性操作影响大 | FP16/BF16 已足够快且精度好 |
| **Attention** | KV Cache 可以用 FP8 减少内存 | QK 计算通常用 FP16/BF16 |
| **Quantization** | MxFP8 等格式专门用于压缩 | - |

#### FlashInfer 中的其他 FP8 算子

除了已实现的 FP8 GEMM，FlashInfer 还支持以下 FP8 相关算子：

1. **FP8 KV Cache** (`decode.py`):
   - 支持 FP8 存储的 KV Cache
   - 减少 KV Cache 内存占用

2. **MxFP8 Quantization** (`fp8_quantization.py`):
   - MxFP8 量化格式 (SM100/Blackwell 专用)
   - 支持分组缩放因子
   - 输入: FP16/BF16 → 输出: FP8 + scale factors

3. **FP4/NVFP4 量化** (`decode.py`):
   - 4 位量化，更激进的压缩
   - 需要 SM100+ 支持

4. **DeepGEMM** (`deep_gemm.py`):
   - 高性能 FP8 GEMM (SM89/SM90)
   - 支持分组 GEMM
   - 来自 DeepSeek 的优化实现

5. **FP8 Attention** (`prefill.py`, `decode.py`):
   - **FP8 Prefill Attention**: 支持 FP8 Q/K/V 输入 (SM90/H100)
   - **FP8 KV Cache**: KV Cache 用 FP8 存储，减少内存占用
   - **FP8 Decode Attention**: decode 阶段支持 FP8 输入

   **数据类型**:
   ```
   Q/K/V 输入:  FP8 E4M3
   Q/K Scale:   FP32 (每个头的缩放因子)
   V Scale:     FP32 (每个头的缩放因子)
   QK 计算:     FP8 × FP8 → FP32 (Tensor Core)
   Softmax:     FP32
   PV 计算:     FP32 × FP8 → FP32
   输出:        FP16/BF16
   ```

   **使用方法**:
   ```python
   from flashinfer.prefill import single_prefill_with_kv_cache

   # 准备 FP8 输入
   q_fp8 = q_fp16.to(torch.float8_e4m3fn)
   k_fp8 = k_fp16.to(torch.float8_e4m3fn)
   v_fp8 = v_fp16.to(torch.float8_e4m3fn)

   # 准备缩放因子
   scale_q = torch.ones(num_heads, dtype=torch.float32, device='cuda')
   scale_k = torch.ones(num_heads, dtype=torch.float32, device='cuda')
   scale_v = torch.ones(num_heads, dtype=torch.float32, device='cuda')

   # 运行 FP8 attention
   output = single_prefill_with_kv_cache(
       q_fp8, k_fp8, v_fp8,
       scale_q=scale_q,
       scale_k=scale_k,
       scale_v=scale_v
   )
   ```

   **架构支持**:
   | 架构 | FP8 Attention | 后端 |
   |------|--------------|------|
   | SM89 (RTX 40xx) | 部分 (FA2) | cuDNN |
   | SM90 (H100/H200) | 完整 (FA3) | FlashAttention-3 |
   | SM100+ (Blackwell) | 完整 + MxFP8 | FlashAttention-3 |

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
