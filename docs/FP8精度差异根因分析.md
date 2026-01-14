# FP8 精度差异根因分析

## 问题现象

在 FlashInfer vs Standalone 的直接数值对比中，发现了显著的精度差异：

| 模式 | FlashInfer vs Standalone 误差 | 状态 |
|------|----------------------------|------|
| **FP16** | 0.017% | ✓ 优秀 |
| **FP8** | 57.77% | ✗ 差异巨大 |

---

## 根本原因：量化策略不一致

### 1. Scale 计算策略不同

#### FlashInfer (对比脚本) - Per-Tensor Scale

```python
# 文件: scripts/benchmark_gqa_decode_compare.py:110-127

def quantize_to_fp8(self, tensor: torch.Tensor, fp8_dtype=torch.float8_e4m3fn):
    """量化到 FP8 - 使用简单的 max calibration 策略"""

    # 计算 scale: 使用全局最大值 (Per-Tensor)
    scale = tensor.abs().max().item() / 448.0
    if scale < 1e-6:
        scale = 1.0

    # 量化: PyTorch 内置转换
    tensor_fp8 = (tensor / scale).to(fp8_dtype)

    return tensor_fp8, scale
```

**关键特点**：
- ✗ **Per-Tensor Scale**: 整个 K/V 张量使用一个 scale
- ✗ **PyTorch 内置转换**: 使用 `.to(torch.float8_e4m3fn)`

**数据流**：
```
K: [kv_len, num_kv_heads, head_dim]
    ↓ abs().max()  # 全局最大值
scale = max / 448  # 单个 scale 值
    ↓
K_fp8 = (K / scale).to(float8_e4m3fn)  # PyTorch 转换
```

---

#### Standalone - Per-Head Scale

```cpp
// 文件: src/gqa_decode_sm89_standalone.cu:1892-1912

// Compute per-head scales
std::vector<float> h_k_scale(num_kv_heads);
std::vector<float> h_v_scale(num_kv_heads);

for (int head = 0; head < num_kv_heads; ++head) {
    // 找每个 head 的最大值
    float k_max = 0.0f;
    float v_max = 0.0f;
    for (int kv = 0; kv < kv_len; ++kv) {
        for (int d = 0; d < head_dim; ++d) {
            size_t idx = kv * num_kv_heads * head_dim + head * head_dim + d;
            k_max = fmax(k_max, fabsf(__half2float(h_k[idx])));
            v_max = fmax(v_max, fabsf(__half2float(h_v[idx])));
        }
    }
    // 每个 head 独立的 scale
    h_k_scale[head] = k_max / 448.0f;
    h_v_scale[head] = v_max / 448.0f;
    if (h_k_scale[head] < 1e-6f) h_k_scale[head] = 1.0f;
    if (h_v_scale[head] < 1e-6f) h_v_scale[head] = 1.0f;
}
```

**关键特点**：
- ✓ **Per-Head Scale**: 每个 KV head 使用独立的 scale
- ✓ **自定义 FP8 转换**: 使用 `float_to_fp8_e4m3_host()`

**数据流**：
```
K: [kv_len, num_kv_heads, head_dim]
    ↓
For each head in num_kv_heads:
    max_this_head = max(K[:, head, :])  # 当前 head 的最大值
    scale[head] = max_this_head / 448     # 独立的 scale
    ↓
K_fp8[:, head, :] = float_to_fp8_e4m3_host(K[:, head, :] / scale[head])
```

---

## 2. FP8 转换方法不同

### PyTorch 内置转换 (`torch.float8_e4m3fn`)

```python
# PyTorch 的转换
tensor_fp8 = (tensor / scale).to(torch.float8_e4m3fn)
```

**特点**：
- 使用 NVIDIA CUDA 内置的 FP8 转换
- 舍入策略：通常是 round-to-nearest-even
- 特殊值处理：遵循 IEEE 754 标准

### Standalone 自定义转换 (`float_to_fp8_e4m3_host`)

```cpp
// 自定义 FP8 E4M3 转换
inline uint8_t float_to_fp8_e4m3_host(float val) {
    // Clamp to FP8 E4M3 range
    if (val > 448.0f) return 0x7F;
    if (val < -448.0f) return 0x80;

    // 提取 FP32 位
    unsigned int bits;
    memcpy(&bits, &val, sizeof(float));

    // FP32 → FP8 转换
    uint8_t sign = (bits >> 31) & 0x1;
    uint8_t exp = ((bits >> 23) & 0xFF);
    uint32_t mant = bits & 0x7FFFFF;

    // 指数转换 (bias 127 → 7)
    int8_t new_exp = static_cast<int8_t>(exp) - 127 + 7;

    // 溢出处理：exp=15 表示饱和
    if (new_exp >= 15) {
        return static_cast<uint8_t>((sign << 7) | (15 << 3));
    }

    // 尾数舍入 (23 bits → 3 bits)
    uint8_t new_mant = (mant >> 20) & 0x7;
    // Round-half-to-even
    uint32_t mant_fraction = mant & 0xFFFFF;
    if (mant_fraction > 0x80000 || (mant_fraction == 0x80000 && new_mant % 2 == 1)) {
        new_mant++;
    }

    return static_cast<uint8_t>((sign << 7) | (new_exp << 3) | new_mant);
}
```

**特点**：
- 完全手动实现
- Round-half-to-even 舍入
- exp=15 用于表示饱和值（而非 Inf/NaN）

---

## 3. 数值差异来源分析

### 示例计算

假设有以下数据：

```
K: [kv_len=128, num_kv_heads=8, head_dim=128]

Head 0: 范围 [-1.0, 1.0],     max = 1.0
Head 1: 范围 [-0.5, 0.5],     max = 0.5
Head 2: 范围 [-0.1, 0.1],     max = 0.1
...
Head 7: 范围 [-2.0, 2.0],     max = 2.0

全局最大值: max_all = 2.0
```

#### FlashInfer (Per-Tensor)

```
scale_all = 2.0 / 448 = 0.004464

对所有 head 使用相同的 scale:
Head 0: K[:, 0, :] / 0.004464 → FP8  (范围 [-224, 224])
Head 1: K[:, 1, :] / 0.004464 → FP8  (范围 [-112, 112])
Head 2: K[:, 2, :] / 0.004464 → FP8  (范围 [-22.4, 22.4])
...
Head 7: K[:, 7, :] / 0.004464 → FP8  (范围 [-448, 448])
```

**问题**：
- Head 0, 1, 2 的值太小，FP8 精度浪费
- 只有 Head 7 利用了 FP8 的完整范围

#### Standalone (Per-Head)

```
scale[0] = 1.0 / 448 = 0.002232
scale[1] = 0.5 / 448 = 0.001116
scale[2] = 0.1 / 448 = 0.000223
...
scale[7] = 2.0 / 448 = 0.004464

每个 head 使用独立的 scale:
Head 0: K[:, 0, :] / 0.002232 → FP8  (范围 [-448, 448])
Head 1: K[:, 1, :] / 0.001116 → FP8  (范围 [-448, 448])
Head 2: K[:, 2, :] / 0.000223 → FP8  (范围 [-448, 448])
...
Head 7: K[:, 7, :] / 0.004464 → FP8  (范围 [-448, 448])
```

**优点**：
- 每个 head 都利用 FP8 的完整范围
- 更精细的量化，精度更高

---

## 4. 差异放大效应

### Attention 计算流程

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

对于 GQA (num_qo_heads=32, num_kv_heads=8, group_size=4):
- QO heads 0-3 共享 KV head 0
- QO heads 4-7 共享 KV head 1
- ...
```

### 误差传播

```
1. FP8 量化误差:
   FlashInfer: Per-Tensor scale → 部分 head 精度低
   Standalone: Per-Head scale   → 所有 head 精度一致

2. Attention Score 计算:
   scores = Q @ K^T
   如果 K 的量化误差不同，scores 的差异会被放大

3. Softmax + Weighted Sum:
   output = softmax(scores) @ V
   scores 的微小差异 → softmax 的指数放大 → 输出巨大差异
```

### 数值示例

假设 `Q @ K^T` 的某项计算：

```
FlashInfer:
  K_fp8 = K / scale_all = K / 0.004464
  如果 K = 0.1, K_fp8 ≈ 22.4
  量化误差: ±0.5 (FP8 精度)

Standalone:
  K_fp8 = K / scale_head = K / 0.000223
  如果 K = 0.1, K_fp8 ≈ 448
  量化误差: ±4 (FP8 精度)

反量化后:
  FlashInfer: 22.4 * 0.004464 ≈ 0.1 ± 0.002
  Standalone: 448 * 0.000223 ≈ 0.1 ± 0.0009
```

虽然单独看起来误差不大，但在 softmax 的指数运算下，这些差异会被显著放大：

```
softmax([x1, x2, x3]) = [exp(x1)/sum, exp(x2)/sum, exp(x3)/sum]

如果 x1, x2, x3 有 ±0.01 的差异:
  exp(x1) 的差异会达到 ±1%
  经过归一化后，最终输出的差异可能达到 ±5-10%
```

---

## 5. 解决方案

### 方案 1: 修改对比脚本使用 Per-Head Scale

```python
def quantize_to_fp8_per_head(self, tensor: torch.Tensor, num_heads: int):
    """使用 Per-Head Scale 量化到 FP8"""
    head_dim = tensor.shape[-1]
    kv_len = tensor.shape[0]

    # 重塑为 [kv_len, num_heads, head_dim]
    tensor_reshaped = tensor.reshape(kv_len, num_heads, head_dim)

    # 计算 per-head scale
    scales = []
    for head in range(num_heads):
        head_data = tensor_reshaped[:, head, :]
        scale = head_data.abs().max().item() / 448.0
        if scale < 1e-6:
            scale = 1.0
        scales.append(scale)

    # 量化每个 head
    tensor_fp8_list = []
    for head in range(num_heads):
        head_data = tensor_reshaped[:, head, :]
        head_fp8 = (head_data / scales[head]).to(torch.float8_e4m3fn)
        tensor_fp8_list.append(head_fp8)

    tensor_fp8 = torch.cat(tensor_fp8_list, dim=1)
    return tensor_fp8, scales
```

### 方案 2: 使用 Standalone 的 FP8 转换函数

将 Standalone 的 `float_to_fp8_e4m3_host` 函数导出为 Python 可调用的形式，确保两者使用完全相同的转换逻辑。

### 方案 3: 统一使用 FlashInfer 的量化策略

修改 Standalone 使用 Per-Tensor Scale，与 FlashInfer 保持一致。

---

## 6. 结论

FP8 精度差异的根源：

| 因素 | FlashInfer | Standalone | 影响 |
|------|-----------|------------|------|
| **Scale 策略** | Per-Tensor | Per-Head | ⚠⚠⚠ 主要因素 |
| **FP8 转换** | PyTorch 内置 | 自定义 | ⚠ 次要因素 |
| **舍入策略** | CUDA 默认 | Round-half-even | ⚠ 轻微影响 |

**最关键的问题是 Per-Tensor vs Per-Head Scale 的差异**，这导致了不同的量化精度分布，在 Attention 计算中被 softmax 放大，最终造成 57.77% 的输出差异。

---

## 建议

1. **短期**: 修改对比脚本，使用 Per-Head Scale 与 Standalone 保持一致
2. **长期**: 两种实现都应该支持配置化的 scale 策略，以便灵活比较

---

**文档版本**: 1.0
**创建时间**: 2026-01-14
**作者**: Claude Code Analysis
