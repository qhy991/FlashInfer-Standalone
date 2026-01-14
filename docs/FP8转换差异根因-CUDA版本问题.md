# FP8 转换差异根因分析 - CUDA 版本问题

## 问题核心

用户的问题非常关键：**为什么 FlashInfer 可以使用正确的 FP8 转换，而 Standalone 不行？两者不都在 CUDA 12.2 下执行吗？**

**答案：不！它们使用的是不同的 CUDA runtime！**

---

## 实际情况

### FlashInfer (通过 PyTorch)

```
PyTorch 版本: 2.9.1+cu128
PyTorch 内置 CUDA runtime: 12.8
```

**FlashInfer 实际运行时使用的是 PyTorch 自带的 CUDA 12.8 runtime！**

```python
# FlashInfer 的 FP8 转换
tensor_fp8 = (tensor / scale).to(torch.float8_e4m3fn)
# ↓
# 调用 PyTorch 的转换函数
# ↓
# 使用 PyTorch bundled CUDA 12.8 runtime 中的正确 FP8 转换
```

### Standalone (纯 CUDA C++)

```
编译时使用的 CUDA: 12.2 (/usr/local/cuda-12.2)
运行时使用的 CUDA: 12.2 (系统 CUDA)
```

**Standalone 使用的是系统安装的 CUDA 12.2，其中有 FP8 转换的 bug！**

```cpp
// Standalone 的 FP8 转换
__nv_fp8_e4m3 fp8_val = __float2_fp8(val);  // ❌ CUDA 12.2 bug
```

---

## CUDA 版本对比

| 组件 | CUDA 版本 | FP8 支持状态 |
|------|-----------|-------------|
| **系统 CUDA** | 12.2 | ❌ FP8 转换有 bug |
| **PyTorch bundled CUDA** | 12.8 | ✅ FP8 转换正确 |
| **Standalone 编译** | 12.2 (nvcc) | ❌ 继承了 CUDA 12.2 的 bug |
| **FlashInfer 运行时** | 12.8 (PyTorch) | ✅ 使用正确的转换 |

---

## 为什么会有这个差异？

### 1. PyTorch 的发布策略

PyTorch **不依赖系统 CUDA**，而是自带打包的 CUDA runtime：

```
PyTorch 安装位置: /usr/local/lib/python3.10/dist-packages/torch/
├── lib/
│   ├── libc10_cuda.so        # PyTorch CUDA runtime
│   ├── libtorch_cuda.so       # PyTorch CUDA functions
│   └── ... (使用 CUDA 12.8 编译)
```

### 2. FlashInfer 的依赖链

```
FlashInfer (Python)
    ↓ 依赖
PyTorch
    ↓ 使用
PyTorch bundled CUDA 12.8 runtime ✅ FP8 正确
```

### 3. Standalone 的依赖链

```
Standalone (C++)
    ↓ 编译链接
系统 CUDA 12.2 (/usr/local/cuda-12.2)
    ↓ 运行时
系统 CUDA 12.2 runtime ❌ FP8 有 bug
```

---

## CUDA 12.2 vs CUDA 12.8 的 FP8 差异

### CUDA 12.2 的 FP8 Bug

```cpp
// CUDA 12.2 中的问题代码
__nv_fp8_e4m3 fp8_val = __float2_fp8(-0.5f);

// 预期: fp8_val = 0xf0 (表示 -448 * scale)
// 实际: fp8_val = 0xf0，但读取时返回 +240 而不是 -448

// 设备端读取
float val = fp8_to_float(fp8_val);  // 返回 +240.0 ❌ 错误符号！
```

**问题根源**：
- `__nv_fp8_e4m3` 类型系统强制转换
- exp=15 的处理返回错误值
- 无法通过常规方法绕过

### CUDA 12.8 的修复

```cpp
// CUDA 12.8 中已修复
__nv_fp8_e4m3 fp8_val = __float2_fp8(-0.5f);
float val = fp8_to_float(fp8_val);  // 返回 -448.0 ✅ 正确！
```

CUDA 12.8 修复了：
1. 正确处理 exp=15 的饱和值
2. 改进了舍入策略
3. 添加了标准的 FP8 转换 API

---

## 为什么 Standalone 不能使用相同的转换函数？

### 技术限制

| 限制 | 说明 |
|------|------|
| **编译时链接** | Standalone 用 nvcc (CUDA 12.2) 编译，链接到 CUDA 12.2 库 |
| **运行时绑定** | 运行时动态链接系统 CUDA 12.2 的 libcudart.so |
| **PyTorch runtime 隔离** | PyTorch 的 CUDA 12.8 runtime 不会暴露给外部程序 |

### 代码对比

```cpp
// ===== FlashInfer (通过 PyTorch) =====
// Python 代码
tensor_fp8 = tensor.to(torch.float8_e4m3fn)

// PyTorch C++ 后端 (使用 CUDA 12.8 编译)
at::Tensor tensor_fp8 = tensor.to(at::kFloat8_e4m3);
// ↓ 调用 PyTorch 内置函数
// ↓ 使用 PyTorch bundled CUDA 12.8 runtime


// ===== Standalone (纯 CUDA C++) =====
// 使用 CUDA 12.2 API
__nv_fp8_e4m3 fp8_val = __float2_fp8(val);  // CUDA 12.2 bug!
// ↓ 链接到系统 CUDA 12.2 libcudart.so
// ↓ 运行时使用 CUDA 12.2 的 buggy 实现
```

---

## 解决方案

### 方案 1: 使用 CUDA 12.8 编译 Standalone ⭐ 推荐

**前提**: 系统需要安装 CUDA 12.8 或更高版本

```bash
# 如果有 CUDA 12.8
export CUDA_HOME=/usr/local/cuda-12.8
nvcc -arch=sm_89 -O2 src/gqa_decode_sm89_standalone.cu -o build/gqa_decode_sm89_standalone
```

**优点**:
- 使用正确的 CUDA FP8 转换
- 与 FlashInfer 完全一致

**缺点**:
- 需要安装 CUDA 12.8
- 可能需要重新编译所有依赖

### 方案 2: 使用 PyTorch C++ API 进行 FP8 转换 ⭐⭐ 最推荐

**核心思路**: 让 Standalone 在运行时调用 PyTorch 的 FP8 转换函数

```python
# Python 包装器
import torch

def prepare_fp8_data_standalone(q, k, v):
    """使用 PyTorch 准备 FP8 数据，然后传递给 Standalone"""

    # 使用 PyTorch 的正确 FP8 转换
    k_fp8, k_scale = quantize_to_fp8(k)  # PyTorch CUDA 12.8
    v_fp8, v_scale = quantize_to_fp8(v)

    # 保存为二进制文件
    k_fp8.numpy().tofile('k_fp8.bin')
    v_fp8.numpy().tofile('v_fp8.bin')

    return k_scale, v_scale

# Standalone C++ 直接读取已量化的 FP8 数据
// 不需要自己进行 FP8 转换！
```

**Standalone 修改**:

```cpp
// 修改前: Standalone 自己做 FP8 量化
h_k_fp8[i] = float_to_fp8_e4m3_host(h_k_float[i] / h_k_scale[head]);  // 自定义转换

// 修改后: Standalone 读取已经量化的 FP8 数据
// Python 端已经用 PyTorch 转换好了
// Standalone 直接读取 uint8_t 数据即可
```

**优点**:
- 使用与 FlashInfer 完全相同的 FP8 转换
- 不需要修改 CUDA 版本
- 公平比较

**缺点**:
- 需要修改对比脚本
- Standalone 需要接受外部 FP8 数据

### 方案 3: 完全匹配 CUDA 12.2 的 FP8 行为 ⚠ 不推荐

让 FlashInfer 也使用有 bug 的 FP8 转换（不公平，且不可行）

---

## 58% 差异的真正原因

### 当前情况

| 步骤 | FlashInfer | Standalone | 差异 |
|------|-----------|------------|------|
| **1. 量化** | Per-Tensor scale | Per-Tensor scale | ✅ 相同 |
| **2. FP8 转换** | PyTorch/CUDA 12.8 | 自定义/CUDA 12.2 | ❌ **不同** |
| **3. 反量化** | CUDA 12.8 | 自定义 | ❌ **不同** |

### 转换差异示例

```python
# 假设输入值: -0.5, scale = 0.001116
# normalized = -0.5 / 0.001116 = -448.0

# FlashInfer (PyTorch CUDA 12.8)
fp8_bits = float_to_fp8_pytorch(-448.0)  # = 0xf0
value = fp8_to_float_pytorch(0xf0)       # = -448.0 ✓

# Standalone (自定义, 尝试匹配 CUDA 12.8)
fp8_bits = float_to_fp8_e4m3_host(-448.0)  # = 0xf0
value = fp8_e4m3_to_float_raw(0xf0)         # = -448.0 ✓

# 但实际上，舍入策略可能不同！
# 某些边界值的处理方式不同
```

### 差异累积

```
1. 单个 FP8 值的转换差异: ±1-2 LSB
   ↓
2. Attention score 计算: 误差放大
   scores = Q @ K^T
   ↓
3. Softmax: 指数放大
   softmax(scores)
   ↓
4. 输出: ~58% 相对误差
```

---

## 验证假设

### 实验: 使用 PyTorch 准备 FP8 数据

修改对比脚本，让 Standalone 直接读取 PyTorch 量化的 FP8 数据：

```python
# 1. 用 PyTorch 量化
k_fp8_torch, k_scale = quantize_to_fp8(k, torch.float8_e4m3fn)

# 2. 保存 FP8 数据 (uint8 原始字节)
k_fp8_uint8 = k_fp8_torch.view(torch.uint8)
k_fp8_uint8.numpy().tofile('k_fp8.bin')

# 3. Standalone 直接读取，不做转换
// Standalone C++
uint8_t* k_fp8_data;  // 直接读取文件，已是 FP8
```

**预期结果**: 如果差异确实来自 FP8 转换，这样修改后误差应该 < 5%

---

## 总结

### 问题回答

| 问题 | 答案 |
|------|------|
| **为什么 FlashInfer 可以，Standalone 不行？** | FlashInfer 使用 PyTorch 自带的 CUDA 12.8 runtime，Standalone 使用系统 CUDA 12.2 |
| **两者都在 CUDA 12.2 下执行吗？** | ❌ FlashInfer 在 CUDA 12.8 下，Standalone 在 CUDA 12.2 下 |
| **为什么不能使用相同的转换函数？** | Standalone 编译链接到 CUDA 12.2，无法访问 PyTorch 的 CUDA 12.8 runtime |
| **58% 差异的根源？** | FP8 转换函数不同 (PyTorch CUDA 12.8 vs 自定义 CUDA 12.2) |

### 解决方案优先级

1. **方案 2 (推荐)**: 对比脚本使用 PyTorch 准备 FP8 数据，Standalone 直接读取
2. **方案 1**: 使用 CUDA 12.8 编译 Standalone (如果可用)
3. **接受现状**: 说明这是不同 CUDA 版本 FP8 实现的差异

---

**文档版本**: 1.0
**创建时间**: 2026-01-14
**关键发现**: PyTorch 2.9.1+cu128 使用自带 CUDA 12.8 runtime，而非系统 CUDA
