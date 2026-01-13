# RMSNorm 调用链分析

## 概述

RMSNorm (Root Mean Square Normalization) 是 LLaMA 等 LLM 模型中常用的归一化操作。与 LayerNorm 相比，RMSNorm 不需要中心化（减去均值），计算更简单高效。

**公式**:
```
RMS(x) = sqrt(mean(x^2))
output = (input / RMS(input)) * weight
```

## FlashInfer 调用链

### 1. Python API 层 (`flashinfer/norm.py`)

```python
@functools.cache
def rmsnorm(in_norm, gamma):
    """
    RMS Normalization

    Args:
        in_norm: (batch_size, hidden_dim) 或 (batch_size, num_heads, head_dim)
        gamma: (hidden_dim,) 或 (num_heads, head_dim)

    Returns:
        output: 与 in_norm 相同形状
    """
    # 获取输入参数
    batch_size, num_heads, head_dim = ...
    eps = 1e-6  # 默认 epsilon 值

    # 生成 JIT 模块
    # 根据输入维度和 hidden_dim 是否 16 字节对齐选择不同的 kernel
    module = _norm_ops.rmsnorm(...)

    # 调度 kernel
    module.run(...)

    return out_norm
```

**关键参数**:
- `in_norm`: 输入张量，支持 2D (batch, hidden) 或 3D (batch, heads, hidden)
- `gamma`: 缩放参数 (weight)
- `eps`: 防止除零的小常数 (默认 1e-6)
- `zero_centered`: 是否使用 zero-centered 变体 (可选)

### 2. JIT 编译层 (`flashinfer/jit/norm.py`)

```python
def gen_norm_module(
    dtype_in: str,
    dtype_out: str,
    hidden_dim: int,
    ...  # 其他参数
):
    """生成 RMSNorm 的 JIT 模块"""

    # 1. 计算唯一标识符 (URI)
    uri = get_norm_uri(dtype_in, dtype_out, hidden_dim, ...)

    # 2. 创建生成目录
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri

    # 3. 渲染 Jinja 模板生成类型特化的配置
    template = jinja2.Template("norm_customize_config.jinja")
    config_content = template.render(
        dtype_in=dtype_map[dtype_in],
        dtype_out=dtype_map[dtype_out],
        hidden_dim=hidden_dim,
        eps=eps,
        ...
    )
    write_if_different(gen_directory / "norm_config.inc", config_content)

    # 4. 复制源文件到生成目录
    sources = ["norm.cu", "norm_jit_binding.cu"]
    for fname in sources:
        shutil.copy(jit_env.FLASHINFER_CSRC_DIR / fname, gen_directory / fname)

    # 5. 返回 JitSpec
    return gen_jit_spec(uri, sources, ...)
```

### 3. C++ Kernel 层 (`include/flashinfer/norm.cuh`)

```cpp
template <typename T, int VecSize>
__global__ void RMSNormKernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    float eps,
    int num_elements,
    int hidden_dim
) {
    // 1. 计算 global offset
    int64_t row_idx = blockIdx.x;
    int64_t offset = row_idx * hidden_dim;

    // 2. 向量化加载
    using VecT = GetVecType<T, VecSize>;
    VecT vec_input[VecSizePerLoad];

    // 3. 计算平方和 (warp-level reduction)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim / VecSize; i += blockDim.x) {
        // VecSize 个元素加载
        VecT vec_data = reinterpret_cast<const VecT*>(input + offset)[i];

        // 展开循环处理每个元素
        #pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float val = cast_to_float(vec_data[j]);
            sum_sq += val * val;
        }
    }

    // 4. Warp shuffle reduction
    sum_sq = warp_reduce_sum(sum_sq);

    // 5. 计算 RMS
    float rms = sqrt(sum_sq / hidden_dim + eps);

    // 6. 归一化并缩放
    for (int i = threadIdx.x; i < hidden_dim / VecSize; i += blockDim.x) {
        VecT vec_data = reinterpret_cast<const VecT*>(input + offset)[i];
        VecT vec_weight = reinterpret_cast<const VecT*>(weight)[i];

        #pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float val = cast_to_float(vec_data[j]);
            float w = cast_to_float(vec_weight[j]);
            float out = (val / rms) * w;
            vec_data[j] = cast_to_storage(out);
        }

        reinterpret_cast<VecT*>(output + offset)[i] = vec_data;
    }
}
```

**关键技术**:
1. **向量化加载**: 使用 `float4`/`uint4` 进行 16 字节对齐加载
2. **Warp-level reduction**: 使用 `__shfl_xor_sync` 进行 warp 内求和
3. **两遍处理**: 第一遍计算平方和，第二遍应用归一化
4. **Shared memory**: 如果需要跨 warp 同步，使用 shared memory

### 4. TVM-FFI 绑定层 (`csrc/norm_jit_binding.cu`)

```cpp
// 导出 RMSNorm 函数
TVM_FFI_DLL_EXPORT_TYPED_FUNC(RMSNorm, dtype_in, dtype_out, ...) {
    // 1. 解析输入参数
    DLTensor* in_norm = ...
    DLTensor* gamma = ...
    DLTensor* out_norm = ...

    // 2. 获取张量信息
    int batch_size = in_norm->shape[0];
    int hidden_dim = in_norm->shape[1];

    // 3. 获取数据指针
    T* input = static_cast<T*>(in_norm->data);
    T* weight = static_cast<T*>(gamma->data);
    T* output = static_cast<T*>(out_norm->data);

    // 4. 选择 kernel 版本
    if (hidden_dim % 16 == 0) {
        // 使用向量化版本
        constexpr int VecSize = 8;  // FP16/BF16: 8 elements = 16 bytes
        RMSNormKernel<T, VecSize><<<grid, block, 0, stream>>>(
            input, weight, output, eps, batch_size * hidden_dim, hidden_dim
        );
    } else {
        // 使用标量版本
        RMSNormKernel<T, 1><<<grid, block, 0, stream>>>(...);
    }

    return 0;
}
```

## Kernel 特性

### 1. 输入格式支持

| 形状 | 说明 | 示例 |
|------|------|------|
| 2D | (batch, hidden) | LLaMA MLP 层输出 |
| 3D | (batch, heads, hidden) | 注意力输出 |

### 2. 数据类型支持

| 类型 | 存储类型 | 计算类型 |
|------|----------|----------|
| FP16 | __half | float |
| BF16 | __nv_bfloat16 | float |

### 3. 向量化策略

| hidden_dim | VecSize | 说明 |
|------------|---------|------|
| 任意 | 1 | 标量处理 (最慢) |
| 8 字节对齐 | 4 | 2 个 FP16 = 8 字节 |
| 16 字节对齐 | 8 | 4 个 FP16 = 16 字节 (最快) |

## 性能优化技术

### 1. Warp-Level Reduction

```cpp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
```

### 2. 向量化内存访问

```cpp
// 16 字节对齐加载
using VecT = get_vec_type_t<T, VecSize>;
VecT data = *reinterpret_cast<const VecT*>(ptr);
```

### 3. 计算融合

```cpp
// 合并操作以减少寄存器压力
float norm_val = val / sqrt(sum_sq / hidden_dim + eps);
float out = norm_val * weight;
```

## 与 LayerNorm 的区别

| 特性 | RMSNorm | LayerNorm |
|------|---------|-----------|
| 中心化 | 否 (不减均值) | 是 (减去均值) |
| 计算量 | 更少 | 更多 |
| 性能 | 更快 | 稍慢 |
| 效果 | 相近 | 略好 |

**RMSNorm 公式**:
```
output = (input / sqrt(mean(input^2) + eps)) * weight
```

**LayerNorm 公式**:
```
mean = mean(input)
var = var(input) = mean((input - mean)^2)
output = ((input - mean) / sqrt(var + eps)) * weight + bias
```

## Standalone 实现要点

### 1. 简单版本 (标量处理)
- 每个线程处理一个元素
- 使用 warp shuffle 进行 reduction
- 易于理解和调试

### 2. 向量化版本
- 16 字节对齐加载/存储
- FP16: 8 元素 = 16 字节
- 减少内存事务数量

### 3. Grid/Block 配置
- Grid: `batch_size` 个 block
- Block: 根据hidden_dim 动态调整，最小 warp size (32)
