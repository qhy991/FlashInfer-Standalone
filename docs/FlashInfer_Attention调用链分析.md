# FlashInfer Attention 算子完整调用链分析

## 目录结构总览

```
FlashInfer/
├── flashinfer/                      # Python API 层
│   ├── prefill.py                   # Prefill attention Python API
│   ├── decode.py                    # Decode attention Python API
│   └── jit/
│       └── attention/
│           ├── modules.py           # JIT 模块生成器
│           └── fmha_v2/             # FA3 (Hopper) backend
│
├── csrc/                            # C++ 绑定层 (TVM-FFI)
│   ├── single_prefill.cu            # Single prefill launcher
│   ├── batch_prefill.cu             # Batch prefill launcher
│   ├── single_decode.cu             # Single decode launcher
│   ├── batch_decode.cu              # Batch decode launcher
│   ├── *_prefill_sm90.cu            # FP8/FA3 kernel launchers
│   └── *_jit_binding.cu             # TVM-FFI exports
│
└── include/flashinfer/attention/    # CUDA Kernels
    ├── prefill.cuh                  # Prefill kernels (FA2)
    ├── decode.cuh                   # Decode kernels
    ├── variants.cuh                 # Attention variants (FA2)
    ├── cascade.cuh                  # State merging
    ├── scheduler.cuh                # Batch scheduling
    └── hopper/                      # FA3 (FlashAttention-3) backend
        ├── prefill_sm90.cuh         # Hopper prefill kernels
        ├── variants.cuh             # Hopper variants
        └── quantization/            # FP8 kernels
```

---

## 完整调用链

### 1. Python API 层

#### 1.1 主要入口函数

**Prefill 入口:** `flashinfer/prefill.py`

```python
@flashinfer_api
def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
    causal: bool = False,
    backend: str = "auto",           # Backend selection
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

**Decode 入口:** `flashinfer/decode.py`

```python
@flashinfer_api
def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    window_left: int = -1,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

#### 1.2 Backend 选择机制

**Backend 选择函数:** `flashinfer/utils.py:455-497`

```python
def determine_attention_backend(
    device: torch.device,
    pos_encoding_mode: int,
    use_fp16_qk_reductions: bool,
    use_custom_mask: bool,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
) -> str:
    """
    Determine the appropriate attention backend.

    Returns: "fa3" (FlashAttention-3 for Hopper SM90) or "fa2" (FlashAttention-2)
    """
    if is_sm90a_supported(device) and is_fa3_backend_supported(...):
        return "fa3"
    else:
        return "fa2"
```

**FA3 支持检查:**

```python
def is_fa3_backend_supported(
    pos_encoding_mode, use_fp16_qk_reductions, use_custom_mask,
    dtype_q, dtype_kv
) -> bool:
    # FA3 不支持的特性
    if use_custom_mask: return False
    if pos_encoding_mode != PosEncodingMode.NONE.value: return False
    if use_fp16_qk_reductions: return False

    # FA3 不支持 FP8 输入（但有专门的 FP8 kernel）
    if dtype_q in [torch.float8_e4m3fn, torch.float8_e5m2]: return False
    if dtype_kv in [torch.float8_e4m3fn, torch.float8_e5m2]: return False

    return True
```

**GPU 架构检测:**

```python
def is_sm90a_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 9 and version_at_least(torch.version.cuda, "12.3")

def is_sm100a_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 10 and version_at_least(torch.version.cuda, "12.8")
```

#### 1.3 FP8 特殊处理

在 `single_prefill_with_kv_cache` 中 (line 1211-1223):

```python
if is_float8(q):
    # FP8 特殊处理
    assert window_left == -1  # FP8 不支持 sliding window
    assert q.dtype == k.dtype == v.dtype
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]

    # 设置默认缩放因子
    if scale_q is None:
        scale_q = torch.ones(q.shape[1], dtype=torch.float32, device=q.device)
    if scale_k is None:
        scale_k = torch.ones(q.shape[1], dtype=torch.float32, device=q.device)
    if scale_v is None:
        scale_v = torch.ones(q.shape[1], dtype=torch.float32, device=q.device)
```

---

### 2. JIT 模块生成层

#### 2.1 模块获取函数

```python
@functools.cache
def get_single_prefill_module(backend, *args):
    uri = get_single_prefill_uri(backend, *args)
    module = gen_single_prefill_module(backend, *args).build_and_load()
    run_func = module.run

    # 注册为 torch custom op
    @register_custom_op(f"flashinfer::{uri}_run", ...)
    def run_single_prefill(...):
        if backend == "fa3":
            if not is_float8(q):
                run_func(q, k, v, ...)
            else:
                # FP8 路径：传递 scale 参数
                run_func(q, k, v, scale_q_tensor, scale_k_tensor, ...)
        else:
            # FA2 路径
            run_func(q, k, v, ...)

    return SimpleNamespace(run=run_single_prefill)
```

#### 2.2 JIT 模块生成

```python
def gen_single_prefill_module(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> JitSpec:
    # 生成唯一 URI
    uri = get_single_prefill_uri(...)

    # 检查是否需要 FP8 tensor cores
    fp8_enabled = dtype_q in [torch.float8_e4m3fn, torch.float8_e5m2]

    if backend == "fa2":
        # FA2 backend
        variant_name = f"DefaultAttention<...>"
        variant_decl = "#include<flashinfer/attention/variants.cuh>"
    else:
        if not fp8_enabled:
            # FA3 标准路径
            variant_name = f"DefaultAttention<{use_logits_soft_cap}>"
            variant_decl = "#include<flashinfer/attention/hopper/variants.cuh>"
        else:
            # FA3 FP8 路径
            variant_name = "DefaultFP8Attention"
            variant_decl = "#include<flashinfer/attention/hopper/variants.cuh>"

    return gen_customize_single_prefill_module(...)
```

---

### 3. TVM-FFI 绑定层

#### 3.1 Kernel Launcher

**Single Prefill Launcher:** `csrc/single_prefill.cu`

```cpp
void single_prefill_with_kv_cache(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView tmp, ffi::TensorView o,
    Optional<ffi::TensorView> maybe_lse,
    int64_t mask_mode_code, int64_t layout, int64_t window_left
    ADDITIONAL_FUNC_PARAMS
) {
    // 提取张量元数据
    unsigned int head_dim_qk = q.size(2);
    const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

    // 分发到专用 kernel
    DISPATCH_context(
        DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE,
        HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
        USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP,
        USE_FP16_QK_REDUCTION, AttentionVariant, Params,
        [&] {
            Params params;
            params.q = static_cast<DTypeQ*>(q.data_ptr());
            params.k = static_cast<DTypeKV*>(k.data_ptr());
            params.v = static_cast<DTypeKV*>(v.data_ptr());
            params.o = static_cast<DTypeO*>(o.data_ptr());

            ADDITIONAL_PARAMS_SETTER

            cudaError_t status = flashinfer::SinglePrefillWithKVCacheDispatched<
                HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
                USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant
            >(params, static_cast<DTypeO*>(tmp.data_ptr()), stream);

            TVM_FFI_ICHECK(status == cudaSuccess);
            return true;
        }
    );
}
```

**FP8 Prefill Launcher:** `csrc/single_prefill_fp8_sm90.cu`

```cpp
void single_prefill_with_kv_cache_sm90(
    ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
    ffi::TensorView tmp, ffi::TensorView o,
    Optional<ffi::TensorView> maybe_lse,
    int64_t mask_mode_code, int64_t layout, int64_t window_left
    ADDITIONAL_FUNC_PARAMS  // 包含 scale_q, scale_k, scale_v
) {
    const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

    DISPATCH_context(
        DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE,
        HEAD_DIM_QK, HEAD_DIM_VO, USE_SLIDING_WINDOW,
        USE_LOGITS_SOFT_CAP, AttentionVariant, Params,
        [&] {
            Params params;
            params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
            params.k_ptr = static_cast<DTypeKV*>(k.data_ptr());
            params.v_ptr = static_cast<DTypeKV*>(v.data_ptr());
            params.o_ptr = static_cast<DTypeO*>(o.data_ptr());

            // FP8 特定参数
            params.scale_q = scale_q;
            params.scale_k = scale_k;
            params.scale_v = scale_v;

            ADDITIONAL_PARAMS_SETTER

            static_assert(HEAD_DIM_QK == HEAD_DIM_VO);
            static_assert(std::is_same_v<DTypeQ, DTypeKV>);

            cudaError_t status =
                flashinfer::SingleFP8PrefillWithKVCacheDispatched<
                    HEAD_DIM_QK, MASK_MODE, USE_SLIDING_WINDOW, AttentionVariant
                >(params, stream);

            TVM_FFI_ICHECK(status == cudaSuccess);
            return true;
        }
    );
}
```

---

### 4. CUDA Kernel 层

#### 4.1 Attention Variants (策略模式)

**FA2 Variants:** `include/flashinfer/attention/variants.cuh`

```cpp
template <
    bool use_custom_mask,
    bool use_sliding_window,
    bool use_logits_soft_cap,
    bool use_alibi
>
struct DefaultAttention : AttentionVariantBase {
    static constexpr bool use_softmax = true;

    uint8_t* custom_mask_ptr;
    uint32_t qo_len, kv_len;
    uint32_t window_left;
    float sm_scale_log2;
    float soft_cap_pre_tanh_scale;

    template <typename Params>
    __device__ __host__ DefaultAttention(
        const Params& params,
        uint32_t batch_idx,
        uint8_t* smem_ptr
    ) {
        qo_len = params.get_qo_len(batch_idx);
        kv_len = params.get_kv_len(batch_idx);

        if constexpr (use_logits_soft_cap) {
            soft_cap_pre_tanh_scale =
                params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
            sm_scale_log2 = math::log2e * params.logits_soft_cap;
        } else {
            sm_scale_log2 = params.sm_scale * math::log2e;
        }
    }

    REGISTER_LOGITS_TRANSFORM(params, logits, ...) {
        if constexpr (use_alibi) {
            logits = logits * params.sm_scale +
                     params.maybe_alibi_slopes[qo_head_idx] *
                     float(int(kv_idx) - int(qo_idx));
        }
        if constexpr (use_logits_soft_cap) {
            logits = float(math::tanh(logits * soft_cap_pre_tanh_scale));
        }
        return logits;
    }
};
```

**FA3 Variants:** `include/flashinfer/attention/hopper/variants.cuh`

```cpp
struct StandardAttention {
    float sm_scale_log2;
    float scale_pv;  // v_scale for non-FP8

    template <typename MainloopParams, typename BlockCoord>
    __device__ StandardAttention(
        const MainloopParams& params,
        const BlockCoord& block_coord
    ) {
        sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
        scale_pv = get_v_scale(params.additional_params, kv_head_idx);
    }

    template <int NUM_ROWS_PER_THREAD>
    __device__ auto GetAttentionUpdater() {
        return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/true>(
            sm_scale_log2
        );
    }
};

struct StandardFP8Attention {
    float p_scale, scale_pv, sm_scale_with_qk_log2;

    template <typename MainloopParams, typename BlockCoord>
    __device__ StandardFP8Attention(...) {
        // 448 for e4m3; 57344 for e5m2
        p_scale = std::numeric_limits<typename MainloopParams::DTypeKV>::max();
        float v_scale = get_v_scale(params.additional_params, kv_head_idx);
        scale_pv = v_scale / p_scale;
        float q_scale = get_q_scale(params.additional_params, qo_head_idx);
        float k_scale = get_k_scale(params.additional_params, kv_head_idx);
        sm_scale_with_qk_log2 =
            q_scale * k_scale * params.additional_params.sm_scale * math::log2e;
    }

    // P-量化：在存储到共享内存前
    template <typename Tensor0>
    __device__ __forceinline__ void PQuantize(Tensor0& tSrS) {
#pragma unroll
        for (int i = 0; i < size(tSrS); ++i) {
            tSrS(i) *= p_scale;  // 缩放到 FP8 范围
        }
    }
};
```

#### 4.2 Prefill Kernel (FA2)

**Kernel 定义:** `include/flashinfer/attention/prefill.cuh`

```cpp
template <
    typename KTraits,
    bool CAUSAL,
    typename AttentionVariant,
    typename Params
>
__global__ __launch_bounds__(KTraits::NUM_THREADS)
void SinglePrefillWithKVCacheKernel(Params params, typename Params::DTypeO* tmp) {
    using DTypeQ = typename KTraits::DTypeQ;
    using DTypeKV = typename KTraits::DTypeKV;
    using DTypeO = typename KTraits::DTypeO;

    constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;

    extern __shared__ char smem_buffer[];
    auto& smem = *reinterpret_cast<typename KTraits::SharedStorage*>(smem_buffer);

    AttentionVariant variant(params, 0, nullptr);

    // 计算 QK attention
    // ... [QK computation with tiling]

    // Softmax 和 V 累加
    // ... [Online softmax algorithm]

    // 写入输出
    // ... [Write to global memory]
}
```

#### 4.3 Prefill Kernel (FA3/Hopper)

**Kernel 定义:** `include/flashinfer/attention/hopper/prefill_sm90.cuh`

```cpp
template <
    typename CollectiveMainloop,
    typename CollectiveEpilogue,
    typename Ktraits,
    bool LEFT_SLIDING_WINDOW,
    bool CAUSAL,
    typename TileScheduler
>
__global__ __launch_bounds__(Ktraits::NUM_WARPS*cutlass::NumThreadsPerWarpGroup, 1)
void PrefillWithKVCacheKernel(
    CUTE_GRID_CONSTANT typename CollectiveMainloop::Params const mainloop_params,
    CUTE_GRID_CONSTANT typename CollectiveEpilogue::Params const epilogue_params,
    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
) {
    static constexpr int NUM_MMA_THREADS = KTraits::NUM_MMA_THREADS;
    static constexpr bool use_tma_load_kv = CollectiveMainloop::USE_TMA_LOAD_KV;

    extern __shared__ char shared_memory[];
    auto& shared_storage = *reinterpret_cast<typename KTraits::SharedStorage*>(shared_memory);

    // 初始化 pipelines
    MainloopPipeline pipeline_k = ...
    MainloopPipeline pipeline_v = ...

    // Producer warpgroup 使用 TMA 加载 K/V
    if (warp_group_idx == 0) {
        collective_mainloop.Load(...);
    }

    // Consumer warpgroups 使用 Tensor Cores 计算 attention
    if (warp_group_idx > 0) {
        // 使用 Tensor Cores 计算 QK^T
        // 应用 Online Softmax
        // 使用 Tensor Cores 计算 O = softmax(QK^T) * V
    }
}
```

**FP8 Kernel:** `include/flashinfer/attention/hopper/quantization/prefill_sm90.cuh`

```cpp
template <...>
__global__ void FP8PrefillWithKVCacheKernel(...) {
    // FP8 特定路径：
    // 1. 作为 FP8 加载 Q/K/V
    // 2. 在 GEMM 前反量化 (Q_scale * K_scale)
    // 3. 使用 FP32/BF16 累加计算 QK^T
    // 4. 使用 V_scale 计算 O = softmax(QK^T) * V
    // 5. 可选：将输出重新量化为 FP8
}
```

#### 4.4 Decode Kernel

**Kernel 定义:** `include/flashinfer/attention/decode.cuh`

```cpp
template <
    PosEncodingMode pos_encoding_mode,
    uint32_t vec_size,
    uint32_t bdx,
    uint32_t bdy,
    typename DTypeQ,
    typename DTypeKV,
    typename DTypeO,
    typename AttentionVariant,
    typename Params
>
__global__ __launch_bounds__(bdx * bdy)
void SingleDecodeWithKVCacheKernel(Params params) {
    // Decode 使用基于状态的算法：
    // 1. 初始化状态 (m, d, o)
    // 2. 分块迭代 KV cache
    // 3. 对于每个块：
    //    a. 加载 K 块，计算 QK
    //    b. 更新 m = max(m, QK)
    //    c. 更新 d = d * exp(m_prev - m) + exp(QK - m)
    //    d. 更新 o = o * exp(m_prev - m) + exp(QK - m) * V
    // 4. 归一化：o = o / d
}
```

---

### 5. 数据流和类型转换

#### 5.1 FP8 数据流

```
FP8 Q/K/V tensors (float8_e4m3fn)
    ↓
提取缩放因子 (per-head 或 per-tensor)
scale_q, scale_k, scale_v
    ↓
Kernel Launcher (single_prefill_fp8_sm90.cu)
    ↓
FP8 Kernel (hopper/quantization/prefill_sm90.cuh)
    ↓
内部计算：
    Q_dequant = Q_fp8 * scale_q
    K_dequant = K_fp8 * scale_k
    QK = Q_dequant @ K_dequant.T  // in FP32
    O = softmax(QK) @ (V_fp8 * scale_v)
    ↓
输出 (FP16/BF16 或 FP8)
```

**FA3 FP8 Variant 特殊处理:**
```cpp
struct StandardFP8Attention {
    float p_scale;  // 448 for e4m3, 57344 for e5m2
    float scale_pv;  // v_scale / p_scale
    float sm_scale_with_qk_log2;

    // P-量化：在存储到共享内存前
    template <typename Tensor0>
    __device__ void PQuantize(Tensor0& tSrS) {
        for (int i = 0; i < size(tSrS); ++i) {
            tSrS(i) *= p_scale;  // 缩放到 FP8 范围
        }
    }
};
```

---

## Backend 对比总结

| 特性 | FA2 (FlashAttention-2) | FA3 (FlashAttention-3/Hopper) |
|-----|------------------------|------------------------------|
| **支持架构** | SM75+ (Turing, Ampere, Hopper, Blackwell) | SM90+ (Hopper only) |
| **输入类型** | FP16, BF16 | FP16, BF16 |
| **FP8 输入** | ❌ 不支持 | ✅ 支持 (单独的 FP8 kernel) |
| **RoPE** | ✅ 支持 (LLaMA style) | ❌ 不支持 |
| **ALiBi** | ✅ 支持 | ❌ 不支持 |
| **Custom Mask** | ✅ 支持 | ❌ 不支持 |
| **Sliding Window** | ✅ 支持 | ✅ 支持 |
| **Logits Soft Cap** | ✅ 支持 | ✅ 支持 |
| **FP16 QK Reduction** | ✅ 可选 | ❌ 不支持 |
| **Tensor Cores** | ✅ 使用 | ✅ 使用 (更高效) |
| **TMA** | ❌ 不使用 | ✅ 使用 |
| **Warp Specialization** | ❌ 不使用 | ✅ 使用 |
| **Kernel 文件** | `prefill.cuh` | `hopper/prefill_sm90.cuh` |
| **Variant** | `variants.cuh` | `hopper/variants.cuh` |

---

## 关键文件路径总结

### Python API
- **Prefill:** `flashinfer/prefill.py`
- **Decode:** `flashinfer/decode.py`
- **Utils:** `flashinfer/utils.py`

### JIT 模块生成
- **Modules:** `flashinfer/jit/attention/modules.py`
- **FA3 Generator:** `flashinfer/jit/attention/fmha_v2/generate_kernels.py`

### C++ 绑定层
- **Single Prefill:** `csrc/single_prefill.cu`
- **Batch Prefill:** `csrc/batch_prefill.cu`
- **Single Decode:** `csrc/single_decode.cu`
- **Batch Decode:** `csrc/batch_decode.cu`
- **FP8 Prefill:** `csrc/single_prefill_fp8_sm90.cu`
- **FFI Utils:** `csrc/tvm_ffi_utils.h`

### CUDA Kernels
- **FA2 Prefill:** `include/flashinfer/attention/prefill.cuh`
- **FA2 Decode:** `include/flashinfer/attention/decode.cuh`
- **FA2 Variants:** `include/flashinfer/attention/variants.cuh`
- **FA3 Prefill:** `include/flashinfer/attention/hopper/prefill_sm90.cuh`
- **FA3 Variants:** `include/flashinfer/attention/hopper/variants.cuh`
- **FA3 FP8:** `include/flashinfer/attention/hopper/quantization/prefill_sm90.cuh`
- **Cascade:** `include/flashinfer/attention/cascade.cuh`
- **Scheduler:** `include/flashinfer/attention/scheduler.cuh`

### Jinja 模板
- **Single Prefill Config:** `csrc/single_prefill_customize_config.jinja`
- **Batch Prefill Config:** `csrc/batch_prefill_customize_config.jinja`
- **Single Decode Config:** `csrc/single_decode_customize_config.jinja`
- **Batch Decode Config:** `csrc/batch_decode_customize_config.jinja`
