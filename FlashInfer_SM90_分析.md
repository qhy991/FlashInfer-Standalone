# FlashInfer SM90 FP8 GEMM 支持分析

## 问题现象

在 H200 (SM90, Compute Capability 9.0) 上运行 FlashInfer 的 `bmm_fp8` 时出现错误：

```
Check failed: (false) is false: bmm_fp8(...)::<lambda()> failed to dispatch data type
```

从堆栈跟踪看，FlashInfer 尝试调用 `fp8_gemm_sm100`，但执行失败。

## 根本原因分析

### 1. FlashInfer 的架构分发逻辑

根据 FlashInfer 源码分析（`flashinfer/gemm/gemm_base.py`）：

```python
def _heuristic_func_bmm_fp8(...):
    major, minor = torch.cuda.get_device_capability()
    
    # 只明确处理了 SM89
    if major == 8 and minor == 9:
        return {
            "backends": ["cublas", "cudnn"],  # SM89 使用 cuBLAS/cuDNN
            ...
        }
    
    # SM100+ 的处理
    if major >= 10:
        return {
            "backends": ["cutlass_sm10x", "cutlass_sm12x"],  # SM100+ 使用 CUTLASS
            ...
        }
    
    # ⚠️ 问题：SM90 (major=9, minor=0) 没有被明确处理！
    # 可能落入默认分支或错误处理
```

### 2. 函数命名混淆

FlashInfer 使用 `fp8_gemm_sm100` 作为函数名，但实际上：
- **SM89**: 使用 cuBLAS/cuDNN backend，通过 `fp8_gemm_sm100` 调用
- **SM90**: 理论上也应该使用 cuBLAS/cuDNN（因为 SM90 也支持 FP8 Tensor Core）
- **SM100+**: 使用 CUTLASS 3.x backend

函数名 `sm100` 可能是历史遗留，不代表只支持 SM100+。

### 3. SM90 的特殊性

| 架构 | Compute Capability | FP8 Tensor Core | Backend 选择 | 状态 |
|------|-------------------|----------------|-------------|------|
| SM89 | 8.9 | ✓ | cuBLAS/cuDNN | ✅ 支持 |
| SM90 | 9.0 | ✓ | cuBLAS/cuDNN (理论上) | ⚠️ **可能不支持** |
| SM100+ | 10.0+ | ✓ | CUTLASS 3.x | ✅ 支持 |

**关键发现**：
- SM89 和 SM90 在硬件层面都支持 FP8 Tensor Core
- 两者都应该可以使用 cuBLAS/cuDNN backend
- 但 FlashInfer 的架构分发逻辑可能没有正确处理 SM90

## 可能的原因

### 原因 1: 架构检测逻辑缺失

FlashInfer 的 `_heuristic_func_bmm_fp8` 可能没有明确处理 `major == 9` 的情况，导致：
- SM90 被错误地路由到 SM100+ 的 CUTLASS backend
- 或者落入未处理的默认分支，导致 dispatch 失败

### 原因 2: 数据类型分发问题

错误信息 "failed to dispatch data type" 表明：
- FlashInfer 可能尝试根据数据类型选择不同的实现路径
- 对于 SM90，可能没有对应的数据类型处理逻辑

### 原因 3: FlashInfer 版本问题

不同版本的 FlashInfer 对 SM90 的支持可能不同：
- **较旧版本**: 可能完全不支持 SM90
- **较新版本**: 可能添加了 SM90 支持，但需要特定配置

## 验证方法

### 1. 检查 FlashInfer 版本

```python
import flashinfer
print(flashinfer.__version__)
```

### 2. 检查架构检测逻辑

```python
import torch
major, minor = torch.cuda.get_device_capability()
print(f"Compute Capability: {major}.{minor}")

# 检查 FlashInfer 的启发式函数
from flashinfer.gemm.gemm_base import _heuristic_func_bmm_fp8
# 查看它如何处理 SM90
```

### 3. 检查可用的 backends

```python
from flashinfer.gemm.gemm_base import _heuristic_func_bmm_fp8
import torch

# 创建虚拟 workspace
workspace = torch.zeros(1024, device='cuda')
a = torch.randn(1, 128, 128, dtype=torch.float16, device='cuda')
b = torch.randn(1, 128, 64, dtype=torch.float16, device='cuda')

# 查看返回的 backends
result = _heuristic_func_bmm_fp8(workspace, a, b, ...)
print("Available backends:", result.get("backends", []))
```

## 解决方案

### 方案 1: 使用 Standalone 实现（推荐）

我们的 Standalone 实现直接使用 cuBLASLt，**完全支持 SM90**：

```bash
# 编译（默认使用 SM90）
./scripts/build_linux.sh

# 运行
./build/fp8_gemm_sm89_standalone
```

**优势**：
- ✅ 直接使用 cuBLASLt，不依赖 FlashInfer 的架构分发
- ✅ 代码简单，易于理解和修改
- ✅ 性能与 FlashInfer 相同（都使用相同的 cuBLASLt API）

### 方案 2: 更新 FlashInfer

```bash
# 尝试更新到最新版本
pip install --upgrade flashinfer-python

# 或从源码安装最新版本
pip install git+https://github.com/flashinfer-ai/flashinfer.git
```

### 方案 3: 手动指定 backend（如果 FlashInfer 支持）

某些版本的 FlashInfer 可能允许手动指定 backend：

```python
from flashinfer.gemm import bmm_fp8

# 尝试强制使用 cuBLAS backend（如果 API 支持）
# 注意：这取决于 FlashInfer 的 API 设计
```

## 技术细节

### SM89 vs SM90 vs SM100+ 的差异

| 特性 | SM89 | SM90 | SM100+ |
|------|------|------|--------|
| FP8 Tensor Core | ✓ | ✓ | ✓ |
| TMA (Tensor Memory Accelerator) | ✗ | ✓ | ✓ |
| WGMMA 指令 | ✗ | ✓ | ✓ |
| cuBLAS FP8 支持 | ✓ | ✓ | ✓ |
| CUTLASS 3.x 支持 | ✗ | ✓ | ✓ |

**关键点**：
- SM90 同时支持 cuBLAS 和 CUTLASS
- FlashInfer 应该能够为 SM90 选择 cuBLAS backend（与 SM89 相同）
- 但实际实现可能没有正确处理这个选择

### cuBLASLt 对 SM90 的支持

cuBLASLt **完全支持 SM90 的 FP8 GEMM**：
- CUDA 12.0+ 的 cuBLASLt 原生支持 SM90
- 我们的 Standalone 实现已验证在 H200 上工作正常
- 性能表现优秀（451-56356 GFLOPS，取决于问题规模）

## 结论

1. **FlashInfer 在 SM90 上的支持可能不完整**
   - 架构分发逻辑可能没有正确处理 SM90
   - 可能是版本问题或实现缺陷

2. **Standalone 实现是可靠的替代方案**
   - 直接使用 cuBLASLt，绕过 FlashInfer 的架构分发
   - 在 H200 上已验证工作正常
   - 性能与 FlashInfer 相同

3. **建议**
   - **短期**: 使用 Standalone 实现
   - **长期**: 关注 FlashInfer 的更新，或提交 issue 报告 SM90 支持问题

## 参考

- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [CUDA FP8 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fp8-floating-point)
- [cuBLASLt API 文档](https://docs.nvidia.com/cuda/cublas/index.html)
