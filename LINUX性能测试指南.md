# Linux 环境性能对比测试指南

## 快速开始

### 1. 准备环境

```bash
# 确保安装了必要的依赖
pip install torch flashinfer-python

# 检查 CUDA
nvcc --version  # 应该 >= 11.4
nvidia-smi      # 检查 GPU
```

### 2. 编译

```bash
cd flashinfer_fp8_analysis
chmod +x scripts/build_linux.sh

# 默认使用 SM90 (H200/H100)
./scripts/build_linux.sh

# 或指定架构
./scripts/build_linux.sh sm_90  # H200/H100 (默认)
./scripts/build_linux.sh sm_89  # RTX 40xx
```

### 3. 运行性能对比

#### 一键运行（推荐）
```bash
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh
```

#### 手动运行
```bash
# 只运行 benchmark 可执行文件
./build/fp8_gemm_benchmark

# 运行完整的性能对比（需要 FlashInfer）
python3 scripts/benchmark_compare.py
```

## 输出说明

### 性能指标
- **GFLOPS**: 每秒十亿次浮点运算（越高越好）
- **延迟**: 每次迭代的平均时间（ms）

### 对比结果
- **✓ (绿色)**: 差异 < 5%（正常）
- **~ (黄色)**: 差异 5-10%（可接受）
- **! (红色)**: 差异 > 10%（需要检查）

## 预期性能

### SM89 (Ada Lovelace / RTX 40xx)

| GPU | 理论峰值 | 实际性能 (FP8) |
|-----|----------|----------------|
| RTX 4090 | ~330 TFLOPS | ~100-150 TFLOPS |
| RTX 4080 | ~230 TFLOPS | ~70-100 TFLOPS |
| RTX 4070 | ~160 TFLOPS | ~50-70 TFLOPS |
| RTX 4060 | ~100 TFLOPS | ~30-50 TFLOPS |

### SM90 (Hopper / H100/H200)

| GPU | 理论峰值 | 实际性能 (FP8) |
|-----|----------|----------------|
| H200 | ~400+ TFLOPS | ~150-200 TFLOPS |
| H100 | ~300+ TFLOPS | ~100-150 TFLOPS |

**注意**: H200/H100 使用 SM90 架构，编译时默认使用 `sm_90`。如需在 RTX 40xx 上运行，请使用 `./scripts/build_linux.sh sm_89`。

## 常见问题

### Q1: 编译失败
```bash
# 检查 CUDA 版本（需要 >= 11.4，推荐 12.0+）
nvcc --version

# 确保 CUDA_PATH 正确设置
echo $CUDA_PATH

# 检查 GPU 架构
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# 如果 GPU 是 RTX 40xx (SM89)，需要指定架构
./scripts/build_linux.sh sm_89
```

### Q2: FlashInfer 未安装
```bash
# 安装 FlashInfer
pip install flashinfer-python

# 如果安装失败，可以跳过 FlashInfer 测试
python3 scripts/benchmark_compare.py  # 会自动跳过
```

### Q3: 性能异常低
```bash
# 检查 GPU 时钟频率
nvidia-smi -q -d CLOCK

# 确保没有其他进程占用 GPU
nvidia-smi
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `build/fp8_gemm_sm89_standalone` | 基础测试版本 |
| `build/fp8_gemm_benchmark` | 性能测试版本 |
| `scripts/benchmark_compare.py` | 性能对比脚本 |
| `scripts/run_benchmark.sh` | 一键运行脚本 |

## 下一步

1. 阅读 `FlashInfer调用链分析.md` 了解实现原理
2. 阅读 `Standalone实现说明.md` 了解代码细节
3. 根据需要修改 `src/fp8_gemm_benchmark.cu` 进行自定义测试
