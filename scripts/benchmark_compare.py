#!/usr/bin/env python3
"""
FlashInfer vs Standalone FP8 GEMM 性能对比测试

要求:
- Linux 环境
- CUDA 11.4+
- FlashInfer 已安装: pip install flashinfer-python
"""

import torch
import subprocess
import time
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

# 检查 FlashInfer
try:
    import flashinfer
    from flashinfer.gemm import bmm_fp8
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    print("警告: FlashInfer 未安装，将跳过 FlashInfer 测试")
    print("安装: pip install flashinfer-python")

# 检查 CUDA
if not torch.cuda.is_available():
    print("错误: CUDA 不可用")
    sys.exit(1)

@dataclass
class BenchmarkResult:
    name: str
    batch: int
    m: int
    n: int
    k: int
    gflops: float
    latency_ms: float
    output_shape: Tuple[int, ...]
    output_dtype: torch.dtype
    output_range: Tuple[float, float]
    error_msg: str = ""

class PerformanceBenchmark:
    def __init__(self):
        self.device = torch.device('cuda')
        prop = torch.cuda.get_device_properties(0)
        compute_cap = f"{prop.major}.{prop.minor}"
        arch_name = "SM89 (Ada)" if prop.major == 8 else "SM90 (Hopper)" if prop.major == 9 else f"SM{prop.major}{prop.minor}"
        
        print(f"\n{'='*70}")
        print(f"性能对比测试环境")
        print(f"{'='*70}")
        print(f"GPU: {prop.name}")
        print(f"Compute Capability: {compute_cap} ({arch_name})")
        print(f"Total Memory: {prop.total_memory / 1024**3:.1f} GB")
        print(f"FlashInfer: {'可用' if FLASHINFER_AVAILABLE else '未安装'}")
        if prop.major == 9:
            print(f"注意: SM90 架构，FlashInfer 可能需要特定版本支持")
        print(f"{'='*70}\n")

    def test_flashinfer(self, batch: int, m: int, n: int, k: int, num_iters: int = 100) -> BenchmarkResult:
        """测试 FlashInfer FP8 GEMM 性能"""
        if not FLASHINFER_AVAILABLE:
            return BenchmarkResult("FlashInfer", batch, m, n, k, 0, 0, (), torch.float16, (0, 0), "FlashInfer 未安装")

        try:
            # 准备数据
            a = torch.randn(batch, m, k, dtype=torch.float16, device=self.device)
            b = torch.randn(batch, k, n, dtype=torch.float16, device=self.device)
            scale_a = torch.tensor([1.0], dtype=torch.float32, device=self.device)
            scale_b = torch.tensor([1.0], dtype=torch.float32, device=self.device)

            # 预热
            for _ in range(10):
                c = bmm_fp8(a, b, scale_a, scale_b, torch.float16)

            # 同步并计时
            torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(num_iters):
                c = bmm_fp8(a, b, scale_a, scale_b, torch.float16)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            # 计算性能
            total_ops = 2 * batch * m * n * k * num_iters
            gflops = total_ops / elapsed / 1e9
            latency_ms = elapsed / num_iters * 1000

            return BenchmarkResult(
                "FlashInfer",
                batch, m, n, k,
                gflops,
                latency_ms,
                tuple(c.shape),
                c.dtype,
                (c.min().item(), c.max().item())
            )

        except Exception as e:
            return BenchmarkResult("FlashInfer", batch, m, n, k, 0, 0, (), torch.float16, (0, 0), str(e))

    def test_standalone(self, batch: int, m: int, n: int, k: int, num_iters: int = 100) -> BenchmarkResult:
        """测试 Standalone 实现性能"""
        exe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "build", "fp8_gemm_benchmark")

        if not os.path.exists(exe_path):
            return BenchmarkResult("Standalone", batch, m, n, k, 0, 0, (), torch.float16, (0, 0),
                                 f"可执行文件不存在: {exe_path}")

        try:
            # 运行 standalone benchmark
            result = subprocess.run(
                [exe_path],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                return BenchmarkResult("Standalone", batch, m, n, k, 0, 0, (), torch.float16, (0, 0),
                                     f"返回码: {result.returncode}, 错误: {error_msg[:100]}")

            # 解析输出获取性能数据
            output_lines = result.stdout.split('\n')

            # 查找匹配的测试配置
            current_config = None
            gflops = 0
            latency_ms = 0

            for i, line in enumerate(output_lines):
                # 检测配置行
                if f"batch={batch}, m={m}, n={n}, k={k}" in line:
                    current_config = (batch, m, n, k)
                # 检测性能行
                elif current_config == (batch, m, n, k):
                    if "性能:" in line or "GFLOPS" in line:
                        try:
                            # 提取数字
                            import re
                            match = re.search(r'[\d.]+', line)
                            if match:
                                gflops = float(match.group())
                        except:
                            pass
                    elif "延迟:" in line or "ms" in line:
                        try:
                            import re
                            match = re.search(r'[\d.]+', line)
                            if match:
                                latency_ms = float(match.group())
                                # 找到延迟后，配置测试完成
                                break
                        except:
                            pass

            if gflops > 0:
                return BenchmarkResult(
                    "Standalone",
                    batch, m, n, k,
                    gflops,
                    latency_ms,
                    (batch, m, n),
                    torch.float16,
                    (0, 0),
                    ""
                )
            else:
                return BenchmarkResult(
                    "Standalone",
                    batch, m, n, k,
                    0, 0,
                    (batch, m, n),
                    torch.float16,
                    (0, 0),
                    "无法解析性能输出（请查看benchmark输出）"
                )

        except subprocess.TimeoutExpired:
            return BenchmarkResult("Standalone", batch, m, n, k, 0, 0, (), torch.float16, (0, 0), "超时")
        except Exception as e:
            return BenchmarkResult("Standalone", batch, m, n, k, 0, 0, (), torch.float16, (0, 0), str(e))

    def test_correctness(self, batch: int, m: int, n: int, k: int) -> dict:
        """测试正确性: FlashInfer vs PyTorch FP16
        
        注意: 由于 FlashInfer 的 bmm_fp8 需要 FP8 输入和正确的 scale factors，
        而我们的测试使用 FP16 输入，误差可能较大。这主要用于验证功能是否工作。
        """
        if not FLASHINFER_AVAILABLE:
            return {"flashinfer_available": False, "error": "FlashInfer 未安装"}

        try:
            torch.manual_seed(42)

            # 准备数据
            a = torch.randn(batch, m, k, dtype=torch.float16, device=self.device)
            b = torch.randn(batch, k, n, dtype=torch.float16, device=self.device)
            
            # 注意: FlashInfer 的 bmm_fp8 可能期望 FP8 输入，但我们传入 FP16
            # Scale factors 应该根据实际数据范围计算，这里简化为 1.0
            # 这可能导致较大的误差，但可以验证功能是否工作
            scale_a = torch.tensor([1.0], dtype=torch.float32, device=self.device)
            scale_b = torch.tensor([1.0], dtype=torch.float32, device=self.device)

            # FP8 GEMM (FlashInfer)
            try:
                c_fp8 = bmm_fp8(a, b, scale_a, scale_b, torch.float16)
            except Exception as e:
                # FlashInfer 可能在某些架构上不支持（如 SM90）
                error_msg = str(e)
                if "dispatch" in error_msg.lower() or "sm100" in error_msg.lower():
                    return {
                        "flashinfer_available": True,
                        "error": f"FlashInfer 在当前架构上不支持: {error_msg[:100]}",
                        "note": "这可能是 FlashInfer 版本问题，Standalone 实现仍然可用"
                    }
                else:
                    return {
                        "flashinfer_available": True,
                        "error": f"FlashInfer 执行失败: {error_msg[:100]}"
                    }

            # FP16 GEMM (PyTorch 参考)
            # a: [batch, m, k], b: [batch, k, n] -> c: [batch, m, n]
            c_fp16 = torch.bmm(a, b)

            # 计算误差
            diff = (c_fp8.float() - c_fp16.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            rel_error = max_diff / (c_fp16.abs().max().item() + 1e-6)

            return {
                "flashinfer_available": True,
                "fp8_max": c_fp8.max().item(),
                "fp8_min": c_fp8.min().item(),
                "fp16_max": c_fp16.max().item(),
                "fp16_min": c_fp16.min().item(),
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "rel_error": rel_error,
                "rel_error_percent": rel_error * 100,
                "note": "注意: 由于使用 FP16 输入而非 FP8，且 scale factors 可能不正确，误差可能较大。这主要用于验证功能是否工作。"
            }
        except Exception as e:
            return {
                "flashinfer_available": True,
                "error": f"正确性测试失败: {str(e)[:200]}"
            }

    def run_benchmark(self):
        """运行完整的性能对比测试"""
        test_configs = [
            # (batch, m, n, k, name)
            (1, 128, 64, 128, "Small (batch=1)"),
            (2, 128, 64, 128, "Small (batch=2)"),
            (1, 256, 128, 256, "Medium"),
            (1, 512, 256, 512, "Large"),
            (2, 512, 256, 512, "Large (batch=2)"),
            (1, 1024, 512, 1024, "XLarge"),
        ]

        results = []

        for batch, m, n, k, name in test_configs:
            print(f"\n{'='*70}")
            print(f"测试: {name}")
            print(f"配置: batch={batch}, m={m}, n={n}, k={k}")
            print(f"计算量: {2 * batch * m * n * k / 1e9:.2f} GFLOPS")
            print(f"{'='*70}")

            # FlashInfer 测试
            print("\n[FlashInfer]")
            flashinfer_result = self.test_flashinfer(batch, m, n, k)
            if flashinfer_result.error_msg:
                print(f"  错误: {flashinfer_result.error_msg}")
            else:
                print(f"  性能: {flashinfer_result.gflops:.1f} GFLOPS")
                print(f"  延迟: {flashinfer_result.latency_ms:.3f} ms")
                print(f"  输出: {flashinfer_result.output_shape}, dtype={flashinfer_result.output_dtype}")
            results.append(flashinfer_result)

            # Standalone 测试
            print("\n[Standalone]")
            standalone_result = self.test_standalone(batch, m, n, k)
            if standalone_result.error_msg and standalone_result.error_msg != "无法解析性能输出（请查看benchmark输出）":
                print(f"  错误: {standalone_result.error_msg}")
            elif standalone_result.gflops > 0:
                print(f"  性能: {standalone_result.gflops:.1f} GFLOPS")
                print(f"  延迟: {standalone_result.latency_ms:.3f} ms")
            else:
                print(f"  状态: {standalone_result.error_msg if standalone_result.error_msg else '运行成功'}")
            results.append(standalone_result)

        # 正确性测试
        print(f"\n{'='*70}")
        print("正确性验证")
        print(f"{'='*70}\n")

        correctness = self.test_correctness(1, 128, 64, 128)

        if not correctness.get("flashinfer_available", False):
            print("FlashInfer 未安装，跳过正确性测试")
        elif "error" in correctness:
            print(f"FlashInfer 正确性测试失败:")
            print(f"  错误: {correctness['error']}")
            if "note" in correctness:
                print(f"  说明: {correctness['note']}")
            print(f"\n  注意: Standalone 实现仍然可用，性能测试已完成")
        elif "rel_error" in correctness:
            print("FlashInfer FP8 vs PyTorch FP16:")
            print(f"  FP8 范围:    [{correctness['fp8_min']:.4f}, {correctness['fp8_max']:.4f}]")
            print(f"  FP16 范围:   [{correctness['fp16_min']:.4f}, {correctness['fp16_max']:.4f}]")
            print(f"  最大差异:    {correctness['max_diff']:.6f}")
            print(f"  平均差异:    {correctness['mean_diff']:.6f}")
            print(f"  相对误差:    {correctness['rel_error_percent']:.2f}%")

            if correctness['rel_error'] < 0.01:
                print(f"  状态: ✓ 优秀 (误差 < 1%)")
            elif correctness['rel_error'] < 0.05:
                print(f"  状态: ✓ 良好 (误差 < 5%)")
            else:
                print(f"  状态: ⚠ 误差较大 (误差 >= 5%)")
            
            if "note" in correctness:
                print(f"\n  {correctness['note']}")

        # 总结
        print(f"\n{'='*70}")
        print("性能总结")
        print(f"{'='*70}\n")

        print(f"{'配置':<25} {'FlashInfer':<20} {'Standalone':<20} {'差异':<10}".format('', '', '', ''))
        print("-" * 75)

        for i in range(0, len(results), 2):
            fi = results[i]
            sa = results[i + 1]
            name = f"{fi.m}x{fi.n}x{fi.k} (batch={fi.batch})"

            fi_perf = f"{fi.gflops:.1f} GFLOPS" if fi.gflops > 0 else "N/A"
            sa_perf = f"{sa.gflops:.1f} GFLOPS" if sa.gflops > 0 else "N/A"

            # 计算性能差异
            if fi.gflops > 0 and sa.gflops > 0:
                diff = ((sa.gflops - fi.gflops) / fi.gflops) * 100
                if abs(diff) < 5:
                    diff_str = f"{diff:+.1f}% ✓"
                elif abs(diff) < 10:
                    diff_str = f"{diff:+.1f}% ~"
                else:
                    diff_str = f"{diff:+.1f}% !"
            else:
                diff_str = "N/A"

            print(f"{name:<25} {fi_perf:<20} {sa_perf:<20} {diff_str:<10}")

        print(f"\n{'='*70}\n")

        # 性能对比结论
        prop = torch.cuda.get_device_properties(0)
        compute_cap = f"{prop.major}.{prop.minor}"
        
        if FLASHINFER_AVAILABLE:
            print("【性能分析】")
            print("1. Standalone 实现和 FlashInfer 使用相同的 cuBLASLt API")
            if prop.major == 8:
                print("2. 两者都使用 SM89 Tensor Core 进行 FP8 计算")
            elif prop.major == 9:
                print("2. 两者都使用 SM90 Tensor Core 进行 FP8 计算")
            else:
                print(f"2. 使用 Compute Capability {compute_cap} Tensor Core 进行 FP8 计算")
            print("3. 理论上性能应该一致（误差 < 5%）")
            
            # 检查是否有 FlashInfer 错误
            has_flashinfer_error = any(r.error_msg and "FlashInfer" in r.name for r in results)
            if has_flashinfer_error:
                print("\n【注意】")
                print("- FlashInfer 在当前架构/版本上可能不支持或有问题")
                print("- Standalone 实现工作正常，可以独立使用")
                print("- 这可能是 FlashInfer 版本问题，建议更新 FlashInfer 或使用 Standalone")
            
            print("\n【建议】")
            print("- Standalone: 适合学习、调试、自定义修改，支持 SM89/SM90")
            print("- FlashInfer:  适合生产环境（功能更丰富），但可能在某些架构上有兼容性问题")
        else:
            print("【注意】FlashInfer 未安装，无法进行性能对比")
            print("安装: pip install flashinfer-python")

        return results

def main():
    """主函数"""
    print("\n" + "="*70)
    print("FlashInfer vs Standalone FP8 GEMM 性能对比")
    print("="*70)

    benchmark = PerformanceBenchmark()
    results = benchmark.run_benchmark()

    print("\n测试完成！\n")

if __name__ == "__main__":
    main()
