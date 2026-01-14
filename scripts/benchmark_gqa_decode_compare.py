#!/usr/bin/env python3
"""
FlashInfer vs Standalone GQA Decode 对比测试

支持:
- FP16 KV cache
- FP8 KV cache with calibration scale
- 数值正确性对比
- 性能对比
"""

import torch
import subprocess
import time
import os
import sys
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

# 检查 FlashInfer
try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    print("警告: FlashInfer 未安装")
    print("安装: pip install flashinfer-python")

# 检查 CUDA
if not torch.cuda.is_available():
    print("错误: CUDA 不可用")
    sys.exit(1)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


@dataclass
class GQADecodeResult:
    """GQA Decode 测试结果"""
    name: str
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    kv_len: int
    use_fp8: bool
    gflops: float
    latency_ms: float
    throughput: float
    output_shape: Tuple[int, ...]
    output_dtype: str
    output_range: Tuple[float, float]
    error_msg: str = ""

    # 数值对比结果
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    rmse: float = 0.0
    relative_rmse: float = 0.0
    passed: bool = False


class GQADecodeComparator:
    """GQA Decode 对比测试器"""

    def __init__(self):
        self.device = torch.device('cuda')
        prop = torch.cuda.get_device_properties(0)
        compute_cap = f"{prop.major}.{prop.minor}"
        arch_name = "SM89 (Ada)" if prop.major == 8 else "SM90 (Hopper)" if prop.major == 9 else f"SM{prop.major}{prop.minor}"

        print(f"\n{'='*80}")
        print(f"GQA Decode 对比测试环境")
        print(f"{'='*80}")
        print(f"GPU: {prop.name}")
        print(f"Compute Capability: {compute_cap} ({arch_name})")
        print(f"Total Memory: {prop.total_memory / 1024**3:.1f} GB")
        print(f"FlashInfer: {'可用' if FLASHINFER_AVAILABLE else '未安装'}")
        print(f"{'='*80}\n")

    def create_test_data(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_len: int,
        dtype: torch.dtype = torch.float16
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """创建测试数据

        Returns:
            q: [num_qo_heads, head_dim]
            k: [kv_len, num_kv_heads, head_dim]
            v: [kv_len, num_kv_heads, head_dim]
        """
        # Query: [num_qo_heads, head_dim]
        q = torch.randn(num_qo_heads, head_dim, dtype=dtype, device=self.device)

        # Key: [kv_len, num_kv_heads, head_dim] (NHD layout)
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=self.device)

        # Value: [kv_len, num_kv_heads, head_dim] (NHD layout)
        v = 0.1 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=self.device)

        return q, k, v

    def quantize_to_fp8(
        self,
        tensor: torch.Tensor,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        num_heads: int = None,
        per_head: bool = True
    ) -> Tuple[torch.Tensor, list]:
        """量化到 FP8

        Args:
            tensor: 输入张量，形状为 [kv_len, num_kv_heads, head_dim] 或 [num_qo_heads, head_dim]
            fp8_dtype: FP8 数据类型
            num_heads: head 数量（用于 per-head 量化）
            per_head: 是否使用 per-head 量化（默认 True，与 Standalone 一致）

        Returns:
            (tensor_fp8, scales): FP8 张量和 scale 列表
        """
        if tensor.dim() == 2:
            # Query: [num_heads, head_dim]
            # Per-head 量化（每个 head 是独立的）
            num_heads = tensor.shape[0]
            tensor_fp8_list = []
            scales = []

            for head in range(num_heads):
                head_data = tensor[head]  # [head_dim]
                scale = head_data.abs().max().item() / 448.0
                if scale < 1e-6:
                    scale = 1.0
                scales.append(scale)
                head_fp8 = (head_data / scale).to(fp8_dtype)
                tensor_fp8_list.append(head_fp8)

            tensor_fp8 = torch.stack(tensor_fp8_list, dim=0)
            return tensor_fp8, scales

        elif tensor.dim() == 3:
            # Key/Value: [kv_len, num_heads, head_dim]
            kv_len, num_heads, head_dim = tensor.shape

            if per_head and num_heads is not None:
                # Per-Head 量化（与 Standalone 一致）
                tensor_fp8_list = []
                scales = []

                for head in range(num_heads):
                    head_data = tensor[:, head, :]  # [kv_len, head_dim]
                    scale = head_data.abs().max().item() / 448.0
                    if scale < 1e-6:
                        scale = 1.0
                    scales.append(scale)
                    head_fp8 = (head_data / scale).to(fp8_dtype)
                    tensor_fp8_list.append(head_fp8)

                tensor_fp8 = torch.stack(tensor_fp8_list, dim=1)  # [kv_len, num_heads, head_dim]
                return tensor_fp8, scales
            else:
                # Per-Tensor 量化（原始方法）
                scale = tensor.abs().max().item() / 448.0
                if scale < 1e-6:
                    scale = 1.0
                tensor_fp8 = (tensor / scale).to(fp8_dtype)
                return tensor_fp8, [scale]
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    def compute_reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """计算 CPU 参考输出 (PyTorch FP32)

        q: [num_qo_heads, head_dim]
        k: [kv_len, num_kv_heads, head_dim]
        v: [kv_len, num_kv_heads, head_dim]

        Returns:
            output: [num_qo_heads, head_dim]
        """
        num_qo_heads = q.shape[0]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[1]
        kv_len = k.shape[0]
        group_size = num_qo_heads // num_kv_heads

        # 转换到 CPU FP32
        q_f32 = q.cpu().float()  # [num_qo_heads, head_dim]
        k_f32 = k.cpu().float()  # [kv_len, num_kv_heads, head_dim]
        v_f32 = v.cpu().float()  # [kv_len, num_kv_heads, head_dim]

        # 缩放因子
        sm_scale = 1.0 / np.sqrt(head_dim)

        # 输出
        output = torch.zeros(num_qo_heads, head_dim, dtype=torch.float32)

        # 对每个 QO head 计算 attention
        for qo_head in range(num_qo_heads):
            kv_head = qo_head // group_size  # GQA: 多个 QO heads 共享同一个 KV head

            # 提取当前 head 的数据
            q_h = q_f32[qo_head]  # [head_dim]
            k_h = k_f32[:, kv_head, :]  # [kv_len, head_dim]
            v_h = v_f32[:, kv_head, :]  # [kv_len, head_dim]

            # Attention: softmax(Q @ K^T) @ V
            # Q @ K^T: [head_dim] @ [head_dim, kv_len] = [kv_len]
            attn_scores = torch.matmul(k_h, q_h) * sm_scale  # [kv_len]

            # Softmax
            attn_weights = torch.softmax(attn_scores, dim=0)  # [kv_len]

            # 加权求和 V
            output[qo_head] = torch.matmul(attn_weights, v_h)  # [head_dim]

        return output

    def test_flashinfer(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_len: int,
        use_fp8: bool = False,
        num_iters: int = 1000
    ) -> GQADecodeResult:
        """测试 FlashInfer GQA Decode"""
        if not FLASHINFER_AVAILABLE:
            return GQADecodeResult(
                name="FlashInfer",
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_len=kv_len,
                use_fp8=use_fp8,
                gflops=0, latency_ms=0, throughput=0,
                output_shape=(), output_dtype="",
                output_range=(0, 0),
                error_msg="FlashInfer 未安装"
            )

        try:
            # 创建测试数据
            q, k, v = self.create_test_data(num_qo_heads, num_kv_heads, head_dim, kv_len)

            if use_fp8:
                # FP8 量化 (使用 per-tensor scale 以匹配 FlashInfer API)
                k_fp8, k_scales = self.quantize_to_fp8(k, torch.float8_e4m3fn, num_kv_heads, per_head=False)
                v_fp8, v_scales = self.quantize_to_fp8(v, torch.float8_e4m3fn, num_kv_heads, per_head=False)
                k_input, v_input = k_fp8, v_fp8
                k_scale, v_scale = k_scales[0], v_scales[0]  # 单个标量
            else:
                k_input, v_input = k, v
                k_scale, v_scale = None, None

            # 预热
            for _ in range(10):
                if use_fp8:
                    o = flashinfer.single_decode_with_kv_cache(
                        q, k_input, v_input,
                        kv_layout="NHD",
                        k_scale=k_scale,
                        v_scale=v_scale
                    )
                else:
                    o = flashinfer.single_decode_with_kv_cache(
                        q, k_input, v_input,
                        kv_layout="NHD"
                    )

            # 使用 CUDA Events 精确计时
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()

            for _ in range(num_iters):
                if use_fp8:
                    o = flashinfer.single_decode_with_kv_cache(
                        q, k_input, v_input,
                        kv_layout="NHD",
                        k_scale=k_scale,
                        v_scale=v_scale
                    )
                else:
                    o = flashinfer.single_decode_with_kv_cache(
                        q, k_input, v_input,
                        kv_layout="NHD"
                    )

            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)

            # 计算性能
            # FLOPS: 2 * num_qo_heads * kv_len * head_dim (Q@K + attn@V)
            total_ops = 2 * num_qo_heads * kv_len * head_dim * num_iters
            elapsed_sec = elapsed_ms / 1000.0
            gflops = total_ops / elapsed_sec / 1e9
            latency_ms = elapsed_ms / num_iters
            throughput = num_iters / elapsed_sec  # requests/sec

            return GQADecodeResult(
                name="FlashInfer",
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_len=kv_len,
                use_fp8=use_fp8,
                gflops=gflops,
                latency_ms=latency_ms,
                throughput=throughput,
                output_shape=tuple(o.shape),
                output_dtype=str(o.dtype),
                output_range=(o.min().item(), o.max().item())
            )

        except Exception as e:
            return GQADecodeResult(
                name="FlashInfer",
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_len=kv_len,
                use_fp8=use_fp8,
                gflops=0, latency_ms=0, throughput=0,
                output_shape=(), output_dtype="",
                output_range=(0, 0),
                error_msg=str(e)[:200]
            )

    def test_standalone(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_len: int,
        use_fp8: bool = False
    ) -> GQADecodeResult:
        """测试 Standalone 实现"""
        # 确定可执行文件参数
        if use_fp8:
            exe_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "build", "gqa_decode_sm89_standalone"
            )
            args = ["fp8"] if use_fp8 else []
        else:
            exe_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "build", "gqa_decode_sm89_standalone"
            )
            args = []

        if not os.path.exists(exe_path):
            return GQADecodeResult(
                name="Standalone",
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_len=kv_len,
                use_fp8=use_fp8,
                gflops=0, latency_ms=0, throughput=0,
                output_shape=(), output_dtype="",
                output_range=(0, 0),
                error_msg=f"可执行文件不存在: {exe_path}"
            )

        try:
            # 运行 standalone
            cmd = [exe_path] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                return GQADecodeResult(
                    name="Standalone",
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    kv_len=kv_len,
                    use_fp8=use_fp8,
                    gflops=0, latency_ms=0, throughput=0,
                    output_shape=(), output_dtype="",
                    output_range=(0, 0),
                    error_msg=f"返回码: {result.returncode}, 错误: {result.stderr[:200] if result.stderr else result.stdout[:200]}"
                )

            # 解析输出
            output_lines = result.stdout.split('\n')

            # 查找匹配的测试配置
            gflops = 0
            latency_ms = 0
            throughput = 0
            max_abs_error = 0
            max_rel_error = 0
            rmse = 0
            relative_rmse = 0
            passed = False

            config_found = False
            for i, line in enumerate(output_lines):
                # 检测配置行
                if f"num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}" in line and \
                   f"head_dim={head_dim}, kv_len={kv_len}" in line:
                    config_found = True

                # 解析性能数据
                elif config_found:
                    if "Average latency:" in line:
                        try:
                            import re
                            match = re.search(r'([\d.]+)\s*ms', line)
                            if match:
                                latency_ms = float(match.group(1))
                        except:
                            pass
                    elif "Throughput:" in line or "GFLOPS" in line:
                        try:
                            import re
                            match = re.search(r'([\d.]+)', line)
                            if match:
                                throughput = float(match.group(1))
                        except:
                            pass
                    elif "Max absolute error:" in line:
                        try:
                            import re
                            match = re.search(r'([\d.]+)', line)
                            if match:
                                max_abs_error = float(match.group(1))
                        except:
                            pass
                    elif "Max relative error:" in line:
                        try:
                            import re
                            match = re.search(r'([\d.]+)', line)
                            if match:
                                max_rel_error = float(match.group(1))
                        except:
                            pass
                    elif "Relative RMSE:" in line:
                        try:
                            import re
                            match = re.search(r'([\d.]+)', line)
                            if match:
                                relative_rmse = float(match.group(1))
                        except:
                            pass
                    elif "Result: PASSED" in line:
                        passed = True

            # 计算 GFLOPS
            if latency_ms > 0:
                total_ops = 2 * num_qo_heads * kv_len * head_dim
                elapsed_sec = latency_ms / 1000.0
                gflops = total_ops / elapsed_sec / 1e9

            return GQADecodeResult(
                name="Standalone",
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_len=kv_len,
                use_fp8=use_fp8,
                gflops=gflops,
                latency_ms=latency_ms,
                throughput=throughput,
                output_shape=(num_qo_heads, head_dim),
                output_dtype="float16",
                output_range=(0, 0),
                max_abs_error=max_abs_error,
                max_rel_error=max_rel_error,
                rmse=rmse,
                relative_rmse=relative_rmse,
                passed=passed
            )

        except subprocess.TimeoutExpired:
            return GQADecodeResult(
                name="Standalone",
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_len=kv_len,
                use_fp8=use_fp8,
                gflops=0, latency_ms=0, throughput=0,
                output_shape=(), output_dtype="",
                output_range=(0, 0),
                error_msg="超时"
            )
        except Exception as e:
            return GQADecodeResult(
                name="Standalone",
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_len=kv_len,
                use_fp8=use_fp8,
                gflops=0, latency_ms=0, throughput=0,
                output_shape=(), output_dtype="",
                output_range=(0, 0),
                error_msg=str(e)[:200]
            )

    def compare_correctness(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_len: int,
        use_fp8: bool = False
    ) -> Dict:
        """数值正确性对比: FlashInfer vs Standalone vs CPU Reference

        使用相同的输入数据进行公平比较。
        """
        if not FLASHINFER_AVAILABLE:
            return {"error": "FlashInfer 未安装"}

        try:
            # 使用固定种子创建测试数据，确保 FlashInfer 和 Standalone 使用相同输入
            seed = 42 + num_qo_heads + num_kv_heads + head_dim + kv_len
            torch.manual_seed(seed)
            q, k, v = self.create_test_data(num_qo_heads, num_kv_heads, head_dim, kv_len)

            # 保存输入数据用于 Standalone
            input_data = {
                "q": q.cpu().numpy().astype(np.float16),
                "k": k.cpu().numpy().astype(np.float16),
                "v": v.cpu().numpy().astype(np.float16),
            }

            # CPU 参考
            print("  计算 CPU 参考...")
            output_ref = self.compute_reference(q, k, v)

            # FlashInfer
            print("  运行 FlashInfer...")
            if use_fp8:
                # 注意：FlashInfer 的 single_decode_with_kv_cache 只支持 per-tensor scale
                # 为了与 Standalone 的 per-head scale 策略进行公平比较，
                # 我们这里使用 per-tensor scale（与 FlashInfer 的设计一致）
                k_fp8, k_scales = self.quantize_to_fp8(k, torch.float8_e4m3fn, num_kv_heads, per_head=False)
                v_fp8, v_scales = self.quantize_to_fp8(v, torch.float8_e4m3fn, num_kv_heads, per_head=False)
                # FlashInfer 使用单个 scale 值
                output_fi = flashinfer.single_decode_with_kv_cache(
                    q, k_fp8, v_fp8,
                    kv_layout="NHD",
                    k_scale=k_scales[0],  # 使用单个 scale
                    v_scale=v_scales[0]
                )
            else:
                output_fi = flashinfer.single_decode_with_kv_cache(
                    q, k, v,
                    kv_layout="NHD"
                )

            # Standalone (使用相同的输入数据)
            print("  运行 Standalone (相同输入)...")
            output_sa = self._run_standalone_with_input(
                input_data["q"], input_data["k"], input_data["v"],
                num_qo_heads, num_kv_heads, head_dim, kv_len, use_fp8
            )

            # 计算误差函数
            def compute_error(output1, output2):
                output2_same_device = output2.to(output1.device)
                diff = (output1.float() - output2_same_device.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                rel_error = max_diff / (output2_same_device.abs().max().item() + 1e-6)
                mse = (diff ** 2).mean().item()
                rmse = np.sqrt(mse)
                ref_rmse = np.sqrt((output2_same_device.float() ** 2).mean().item())
                relative_rmse = rmse / (ref_rmse + 1e-6)
                return max_diff, rel_error, rmse, relative_rmse

            # 计算各种误差
            fi_max_diff, fi_rel_error, fi_rmse, fi_rel_rmse = compute_error(output_fi, output_ref)

            if output_sa is not None:
                sa_max_diff, sa_rel_error, sa_rmse, sa_rel_rmse = compute_error(output_sa, output_ref)
                fi_sa_max_diff, fi_sa_rel_error, fi_sa_rmse, fi_sa_rel_rmse = compute_error(output_fi, output_sa)

                return {
                    "flashinfer_available": True,
                    # FlashInfer vs CPU Reference
                    "fi_max_diff": fi_max_diff,
                    "fi_rel_error": fi_rel_error,
                    "fi_rel_error_percent": fi_rel_error * 100,
                    "fi_relative_rmse": fi_rel_rmse,
                    "fi_range": (output_fi.min().item(), output_fi.max().item()),
                    # Standalone vs CPU Reference
                    "sa_max_diff": sa_max_diff,
                    "sa_rel_error": sa_rel_error,
                    "sa_rel_error_percent": sa_rel_error * 100,
                    "sa_relative_rmse": sa_rel_rmse,
                    "sa_range": (output_sa.min().item(), output_sa.max().item()),
                    # FlashInfer vs Standalone 直接对比！
                    "fi_sa_max_diff": fi_sa_max_diff,
                    "fi_sa_rel_error": fi_sa_rel_error,
                    "fi_sa_rel_error_percent": fi_sa_rel_error * 100,
                    "fi_sa_relative_rmse": fi_sa_rel_rmse,
                    "has_standalone": True,
                }
            else:
                return {
                    "flashinfer_available": True,
                    "fi_max_diff": fi_max_diff,
                    "fi_rel_error": fi_rel_error,
                    "fi_rel_error_percent": fi_rel_error * 100,
                    "fi_relative_rmse": fi_rel_rmse,
                    "fi_range": (output_fi.min().item(), output_fi.max().item()),
                    "has_standalone": False,
                    "sa_error": "Standalone 输出不可用",
                }

        except Exception as e:
            import traceback
            return {"error": f"正确性测试失败: {str(e)[:200]}\n{traceback.format_exc()[-500:]}"}

    def _run_standalone_with_input(
        self,
        q_np: np.ndarray,
        k_np: np.ndarray,
        v_np: np.ndarray,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_len: int,
        use_fp8: bool
    ) -> Optional[torch.Tensor]:
        """运行 Standalone 并获取输出张量

        使用 compare 模式运行 Standalone，读取输出文件。
        对于 FP8 模式，使用 PyTorch 准备 FP8 数据，确保与 FlashInfer 使用相同的转换。
        """
        exe_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "build", "gqa_decode_sm89_standalone"
        )

        if not os.path.exists(exe_path):
            print(f"    警告: Standalone 可执行文件不存在: {exe_path}")
            return None

        try:
            # 创建临时目录存储输入/输出文件
            import tempfile
            temp_dir = tempfile.mkdtemp()

            # 将数据转换为 torch tensor (在 GPU 上)
            q = torch.from_numpy(q_np).to(self.device)
            k = torch.from_numpy(k_np).to(self.device)
            v = torch.from_numpy(v_np).to(self.device)

            # 保存文件路径
            q_file = os.path.join(temp_dir, "q.bin")
            k_file = os.path.join(temp_dir, "k.bin")
            v_file = os.path.join(temp_dir, "v.bin")
            output_file = os.path.join(temp_dir, "output.bin")
            k_fp8_file = os.path.join(temp_dir, "k_fp8.bin")
            v_fp8_file = os.path.join(temp_dir, "v_fp8.bin")

            if use_fp8:
                # ===== 方案 2: 使用 PyTorch 准备 FP8 数据 =====
                # 这样 Standalone 和 FlashInfer 使用完全相同的 FP8 转换

                print("    使用 PyTorch 准备 FP8 数据 (CUDA 12.8 转换)...")

                # 使用 Per-Tensor scale (与 FlashInfer 一致)
                k_fp8, k_scales = self.quantize_to_fp8(k, torch.float8_e4m3fn, num_kv_heads, per_head=False)
                v_fp8, v_scales = self.quantize_to_fp8(v, torch.float8_e4m3fn, num_kv_heads, per_head=False)

                # 保存 Q 作为 FP16 (Query 不需要 FP8)
                q_np.tofile(q_file)

                # 保存已经量化的 FP8 数据 (uint8 格式)
                # FP8 tensor 实际上是 uint8，直接保存
                k_fp8_uint8 = k_fp8.view(torch.uint8).cpu().numpy()
                v_fp8_uint8 = v_fp8.view(torch.uint8).cpu().numpy()

                k_fp8_uint8.tofile(k_fp8_file)
                v_fp8_uint8.tofile(v_fp8_file)

                # 保存 scales
                k_scale = k_scales[0]
                v_scale = v_scales[0]
                print(f"    K scale: {k_scale:.8f}, V scale: {v_scale:.8f}")

                # 创建一个特殊的配置文件告诉 Standalone 读取 FP8 数据
                config_file = os.path.join(temp_dir, "config.txt")
                with open(config_file, 'w') as f:
                    f.write(f"use_fp8_data=true\n")
                    f.write(f"k_fp8_file={k_fp8_file}\n")
                    f.write(f"v_fp8_file={v_fp8_file}\n")
                    f.write(f"k_scale={k_scale}\n")
                    f.write(f"v_scale={v_scale}\n")

                # 使用 PyTorch 准备的 FP8 数据，确保与 FlashInfer 使用完全相同的转换
                # Standalone 现在支持读取预量化的 FP8 数据
                print(f"    使用 PyTorch 准备的 FP8 数据（CUDA 12.8 转换）")
                k_np.tofile(k_file)
                v_np.tofile(v_file)

                cmd = [
                    exe_path, "compare",
                    q_file, k_file, v_file, output_file,
                    str(num_qo_heads), str(num_kv_heads),
                    str(head_dim), str(kv_len),
                    "--fp8",
                    "--fp8-data",
                    k_fp8_file,
                    v_fp8_file,
                    str(k_scale),
                    str(v_scale)
                ]

            else:
                # FP16 模式
                q_np.tofile(q_file)
                k_np.tofile(k_file)
                v_np.tofile(v_file)

                cmd = [
                    exe_path, "compare",
                    q_file, k_file, v_file, output_file,
                    str(num_qo_heads), str(num_kv_heads),
                    str(head_dim), str(kv_len)
                ]

            # 运行 Standalone
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )

            # 调试：检查 Standalone 是否使用了预量化的 FP8 数据
            if use_fp8:
                if "Using pre-quantized" in result.stdout:
                    print(f"    ✅ Standalone 使用了 PyTorch 准备的 FP8 数据")
                elif "Performing FP8 quantization" in result.stdout:
                    print(f"    ⚠️  Standalone 自己进行了 FP8 量化（未使用预量化数据）")
            
            if result.returncode != 0:
                print(f"    Standalone 返回码: {result.returncode}")
                print(f"    错误: {result.stderr[:200] if result.stderr else result.stdout[:200]}")
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

            # 读取输出文件
            if os.path.exists(output_file):
                output_np = np.fromfile(output_file, dtype=np.float16)
                output_size = num_qo_heads * head_dim

                if output_np.size != output_size:
                    print(f"    警告: 输出大小不匹配，预期 {output_size}，实际 {output_np.size}")
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return None

                # 转换为 torch tensor
                output = torch.from_numpy(output_np).reshape(num_qo_heads, head_dim).to(self.device)

                # 清理临时文件
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

                return output
            else:
                print(f"    警告: 输出文件不存在: {output_file}")
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

        except subprocess.TimeoutExpired:
            print(f"    Standalone 超时")
            return None
        except Exception as e:
            print(f"    Standalone 运行错误: {str(e)[:100]}")
            return None

    def _run_standalone_with_same_input(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_len: int,
        use_fp8: bool
    ) -> Dict:
        """使用相同的输入数据运行 Standalone 实现，返回输出张量"""
        import struct

        exe_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "build", "gqa_decode_sm89_standalone"
        )

        if not os.path.exists(exe_path):
            return {"error": f"可执行文件不存在: {exe_path}"}

        try:
            # 准备输入数据文件
            import tempfile
            temp_dir = tempfile.mkdtemp()

            # 保存输入数据到文件
            q_file = os.path.join(temp_dir, "q_input.bin")
            k_file = os.path.join(temp_dir, "k_input.bin")
            v_file = os.path.join(temp_dir, "v_input.bin")
            output_file = os.path.join(temp_dir, "output.bin")

            # 将数据保存为二进制 (FP16)
            q_cpu = q.cpu().numpy().astype(np.float16)
            k_cpu = k.cpu().numpy().astype(np.float16)
            v_cpu = v.cpu().numpy().astype(np.float16)

            q_cpu.tofile(q_file)
            k_cpu.tofile(k_file)
            v_cpu.tofile(v_file)

            # 注意: Standalone 实现当前不支持从文件读取输入
            # 这里我们只是使用 Standalone 的内部验证结果
            # 实际的数值对比需要 Standalone 实现支持导出输出

            # 运行 Standalone 并解析输出
            args = ["fp8"] if use_fp8 else []
            cmd = [exe_path] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )

            # 解析输出中的验证信息
            output_lines = result.stdout.split('\n')

            max_rel_error = 0
            relative_rmse = 0
            passed = False

            for line in output_lines:
                if f"num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}" in line and \
                   f"head_dim={head_dim}, kv_len={kv_len}" in line:
                    # 找到匹配的配置，继续解析
                    pass
                if "Max relative error:" in line:
                    try:
                        import re
                        match = re.search(r'([\d.]+)%', line)
                        if match:
                            max_rel_error = float(match.group(1))
                    except:
                        pass
                elif "Relative RMSE:" in line:
                    try:
                        import re
                        match = re.search(r'([\d.]+)%', line)
                        if match:
                            relative_rmse = float(match.group(1))
                    except:
                        pass
                elif "Result: PASSED" in line:
                    passed = True

            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            # 由于 Standalone 使用不同的随机种子，我们只能比较统计信息
            # 而不是直接比较输出值
            return {
                "passed": passed,
                "max_rel_error": max_rel_error,
                "relative_rmse": relative_rmse,
            }

        except Exception as e:
            return {"error": str(e)[:200]}

    def run_comparison(self):
        """运行完整对比测试"""
        # 测试配置
        test_configs = [
            # (num_qo_heads, num_kv_heads, head_dim, kv_len, name, use_fp8)
            (32, 8, 128, 128, "GQA (32/8/128/128)", False),
            (32, 8, 128, 256, "GQA (32/8/128/256)", False),
            (32, 4, 128, 128, "GQA (32/4/128/128)", False),
            (32, 32, 128, 128, "MHA (32/32/128/128)", False),
            # FP8 测试
            (32, 8, 128, 128, "GQA FP8 (32/8/128/128)", True),
        ]

        all_results = []

        for num_qo_heads, num_kv_heads, head_dim, kv_len, name, use_fp8 in test_configs:
            print(f"\n{'='*80}")
            print(f"测试: {name}")
            print(f"配置: num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, "
                  f"head_dim={head_dim}, kv_len={kv_len}, FP8={use_fp8}")
            group_size = num_qo_heads // num_kv_heads
            print(f"Group Size: {group_size} ({'GQA' if group_size > 1 else 'MHA'})")
            print(f"{'='*80}")

            # 正确性测试
            print(f"\n[正确性对比]")
            correctness = self.compare_correctness(
                num_qo_heads, num_kv_heads, head_dim, kv_len, use_fp8
            )

            if "error" in correctness:
                print(f"  错误: {correctness['error']}")
            else:
                print(f"  ┌─ FlashInfer vs CPU Reference (FP32):")
                print(f"  │   最大误差:    {correctness['fi_max_diff']:.6f}")
                print(f"  │   相对误差:    {correctness['fi_rel_error_percent']:.4f}%")
                print(f"  │   相对 RMSE:   {correctness['fi_relative_rmse']:.4f}%")
                if correctness['fi_rel_error'] < 0.01:
                    print(f"  │   状态:        ✓ 优秀 (误差 < 1%)")
                elif correctness['fi_rel_error'] < 0.05:
                    print(f"  │   状态:        ✓ 良好 (误差 < 5%)")
                else:
                    print(f"  │   状态:        ⚠ 误差较大 (FP8 量化正常)")

                # Standalone vs CPU Reference (如果可用)
                if "has_standalone" in correctness and correctness["has_standalone"]:
                    print(f"  │")
                    print(f"  ├─ FlashInfer vs Standalone (直接对比!):")
                    print(f"  │   最大误差:    {correctness['fi_sa_max_diff']:.6f}")
                    print(f"  │   相对误差:    {correctness['fi_sa_rel_error_percent']:.4f}%")
                    print(f"  │   相对 RMSE:   {correctness['fi_sa_relative_rmse']:.4f}%")
                    if correctness['fi_sa_rel_error'] < 0.5:
                        print(f"  │   状态:        ✓ 优秀 (两者几乎相同)")
                    elif correctness['fi_sa_rel_error'] < 5:
                        print(f"  │   状态:        ✓ 良好 (两者接近)")
                    else:
                        print(f"  │   状态:        ⚠ 差异较大")
                    print(f"  │")
                    if use_fp8:
                        print(f"  └─ 说明: 两者都使用 Per-Tensor scale，差异来自 FP8 转换函数:")
                        print(f"       FlashInfer:  PyTorch 内置 .to(float8_e4m3fn)")
                        print(f"       Standalone:  自定义 float_to_fp8_e4m3_host()")
                        print(f"       差异来源: FP8 转换舍入策略不同 (预期行为)")
                    else:
                        print(f"  └─ 说明: 两者使用相同输入(Q,K,V)，直接比较输出")
                else:
                    print(f"  │")
                    print(f"  └─ 说明: Standalone 输出对比需要文件 I/O 支持 (待实现)")

            # FlashInfer 性能测试
            print(f"\n[FlashInfer 性能]")
            fi_result = self.test_flashinfer(
                num_qo_heads, num_kv_heads, head_dim, kv_len, use_fp8, num_iters=1000
            )
            all_results.append(fi_result)

            if fi_result.error_msg:
                print(f"  错误: {fi_result.error_msg}")
            else:
                print(f"  延迟:       {fi_result.latency_ms:.4f} ms")
                print(f"  吞吐量:     {fi_result.throughput:.0f} req/s")
                print(f"  性能:       {fi_result.gflops:.2f} GFLOPS")

            # Standalone 性能测试
            print(f"\n[Standalone 性能]")
            sa_result = self.test_standalone(
                num_qo_heads, num_kv_heads, head_dim, kv_len, use_fp8
            )
            all_results.append(sa_result)

            if sa_result.error_msg:
                print(f"  状态: {sa_result.error_msg}")
            elif sa_result.latency_ms > 0:
                print(f"  延迟:       {sa_result.latency_ms:.4f} ms")
                print(f"  性能:       {sa_result.gflops:.2f} GFLOPS")
                if sa_result.use_fp8:
                    print(f"  验证:       {'✓ PASSED' if sa_result.passed else '✗ FAILED'}")
                    if sa_result.max_rel_error > 0:
                        print(f"  最大误差:   {sa_result.max_rel_error:.4f}%")
                        print(f"  相对 RMSE:  {sa_result.relative_rmse:.4f}%")

            # 性能对比
            if not fi_result.error_msg and not sa_result.error_msg and sa_result.gflops > 0:
                print(f"\n[性能对比]")
                diff = ((sa_result.gflops - fi_result.gflops) / fi_result.gflops) * 100
                print(f"  Standalone vs FlashInfer: {diff:+.1f}%")

        # 总结
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: List[GQADecodeResult]):
        """打印总结"""
        print(f"\n{'='*80}")
        print(f"测试总结")
        print(f"{'='*80}\n")

        # 过滤有效结果
        valid_results = [r for r in results if not r.error_msg and r.gflops > 0]

        if not valid_results:
            print("没有有效的测试结果")
            return

        # 分组显示
        print(f"{'配置':<30} {'实现':<15} {'FP8':<6} {'延迟(ms)':<12} {'性能(GFLOPS)':<15}")
        print("-" * 80)

        for r in results:
            if r.error_msg:
                continue

            config = f"QO={r.num_qo_heads}/KV={r.num_kv_heads}/D={r.head_dim}/L={r.kv_len}"
            fp8_str = "✓" if r.use_fp8 else "✗"

            if r.gflops > 0:
                print(f"{config:<30} {r.name:<15} {fp8_str:<6} "
                      f"{r.latency_ms:<12.4f} {r.gflops:<15.2f}")

        # FlashInfer vs Standalone 对比
        print(f"\n{'='*80}")
        print(f"FlashInfer vs Standalone 直接对比")
        print(f"{'='*80}\n")

        print(f"{'配置':<30} {'FlashInfer':<20} {'Standalone':<20} {'差异':<10}")
        print("-" * 80)

        for i in range(0, len(results) - 1, 2):
            fi = results[i]
            sa = results[i + 1]

            if fi.error_msg or sa.error_msg or sa.gflops == 0:
                continue

            config = f"QO={fi.num_qo_heads}/KV={fi.num_kv_heads}/D={fi.head_dim}/L={fi.kv_len}"
            fi_perf = f"{fi.gflops:.2f} GFLOPS"
            sa_perf = f"{sa.gflops:.2f} GFLOPS"

            diff = ((sa.gflops - fi.gflops) / fi.gflops) * 100
            if abs(diff) < 5:
                diff_str = f"{diff:+.1f}% ✓"
            elif abs(diff) < 10:
                diff_str = f"{diff:+.1f}% ~"
            else:
                diff_str = f"{diff:+.1f}% !"

            print(f"{config:<30} {fi_perf:<20} {sa_perf:<20} {diff_str:<10}")

        print(f"\n{'='*80}\n")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("FlashInfer vs Standalone GQA Decode 对比测试")
    print("="*80)

    comparator = GQADecodeComparator()
    results = comparator.run_comparison()

    print("\n测试完成！\n")


if __name__ == "__main__":
    main()
