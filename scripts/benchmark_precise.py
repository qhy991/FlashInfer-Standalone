#!/usr/bin/env python3
"""
精确的性能对比测试 - 多次执行取平均值
"""

import torch
import subprocess
import numpy as np
from dataclasses import dataclass
from typing import List

# 检查 FlashInfer
try:
    import flashinfer
    from flashinfer.gemm import bmm_fp8
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

if not torch.cuda.is_available():
    print("错误: CUDA 不可用")
    exit(1)

@dataclass
class TestConfig:
    batch: int
    m: int
    n: int
    k: int
    name: str

@dataclass
class TestResult:
    name: str
    config: TestConfig
    gflops: float
    latency_ms: float
    latency_std_ms: float
    min_latency_ms: float
    max_latency_ms: float

class PreciseBenchmark:
    def __init__(self):
        self.device = torch.device('cuda')
        prop = torch.cuda.get_device_properties(0)

        print(f"\n{'='*70}")
        print(f"精确性能对比测试 (多次执行)")
        print(f"{'='*70}")
        print(f"GPU: {prop.name}")
        print(f"Compute Capability: {prop.major}.{prop.minor}")
        print(f"FlashInfer: {'可用' if FLASHINFER_AVAILABLE else '未安装'}")
        print(f"{'='*70}\n")

    def to_float8(self, x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn):
        """直接转换为 FP8，使用 scale=1.0"""
        x_fp8 = x.to(dtype)
        scale = torch.tensor([1.0], dtype=torch.float32, device=x.device)
        return x_fp8, scale

    def test_flashinfer_precise(self, config: TestConfig, num_runs: int = 20, num_iters_per_run: int = 100) -> TestResult:
        """精确测试 FlashInfer 性能"""
        if not FLASHINFER_AVAILABLE:
            return TestResult("FlashInfer", config, 0, 0, 0, 0, 0)

        try:
            # 使用固定随机种子，确保每次运行使用相同的数据
            torch.manual_seed(42)

            # 准备数据
            a = torch.randn(config.batch, config.m, config.k, dtype=torch.float16, device=self.device)
            b = torch.randn(config.batch, config.k, config.n, dtype=torch.float16, device=self.device)
            a_fp8, scale_a = self.to_float8(a)
            b_fp8, scale_b = self.to_float8(b)

            latencies = []

            # 首先进行一次完整的预热运行（触发 JIT 编译）
            print(f"  首次预热（触发 JIT 编译）...", end="", flush=True)
            for _ in range(30):
                c = bmm_fp8(a_fp8, b_fp8, scale_a, scale_b, torch.float16)
            torch.cuda.synchronize()
            print(" 完成")

            # 多次运行测量
            for run in range(num_runs):
                # 每次运行前的额外预热
                for _ in range(5):
                    c = bmm_fp8(a_fp8, b_fp8, scale_a, scale_b, torch.float16)

                # 使用 CUDA Events 计时
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(num_iters_per_run):
                    c = bmm_fp8(a_fp8, b_fp8, scale_a, scale_b, torch.float16)
                end_event.record()

                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                avg_latency_ms = elapsed_ms / num_iters_per_run
                latencies.append(avg_latency_ms)

                if run < 3:  # 打印前几次运行的时间
                    print(f"  运行 {run+1}: {avg_latency_ms:.4f} ms", end="\r", flush=True)

            print()  # 换行

            # 统计
            latencies = np.array(latencies)
            mean_latency = latencies.mean()
            std_latency = latencies.std()
            min_latency = latencies.min()
            max_latency = latencies.max()

            # 计算 GFLOPS
            total_ops = 2 * config.batch * config.m * config.n * config.k
            gflops = total_ops / (mean_latency / 1000.0) / 1e9

            return TestResult("FlashInfer", config, gflops, mean_latency, std_latency, min_latency, max_latency)

        except Exception as e:
            print(f"  FlashInfer 错误: {e}")
            return TestResult("FlashInfer", config, 0, 0, 0, 0, 0)

    def run_comparison(self):
        """运行完整对比"""
        configs = [
            TestConfig(1, 128, 64, 128, "Small (batch=1)"),
            TestConfig(1, 256, 128, 256, "Medium"),
            TestConfig(1, 512, 256, 512, "Large"),
            TestConfig(1, 1024, 512, 1024, "XLarge"),
        ]

        results = []

        for config in configs:
            print(f"\n{'='*70}")
            print(f"测试: {config.name}")
            print(f"配置: batch={config.batch}, m={config.m}, n={config.n}, k={config.k}")
            print(f"计算量: {2 * config.batch * config.m * config.n * config.k / 1e9:.2f} GFLOPS")
            print(f"{'='*70}")

            # FlashInfer 测试
            if FLASHINFER_AVAILABLE:
                print("\n[FlashInfer] 运行 20 次，每次 100 次迭代...")
                fi_result = self.test_flashinfer_precise(config, num_runs=20, num_iters_per_run=100)
                print(f"  平均延迟: {fi_result.latency_ms:.4f} ± {fi_result.latency_std_ms:.4f} ms")
                print(f"  延迟范围: [{fi_result.min_latency_ms:.4f}, {fi_result.max_latency_ms:.4f}] ms")
                print(f"  平均性能: {fi_result.gflops:.1f} GFLOPS")
                results.append(fi_result)

        # 总结
        self.print_summary(results)

    def print_summary(self, results: List[TestResult]):
        """打印总结"""
        print(f"\n{'='*70}")
        print("性能总结 (20次运行平均值 ± 标准差, 每次100次迭代)")
        print(f"{'='*70}\n")

        print(f"{'配置':<20} {'延迟(ms)':<30} {'性能(GFLOPS)':<15}")
        print("-" * 70)

        for result in results:
            if result.gflops > 0:
                latency_str = f"{result.latency_ms:.4f} ± {result.latency_std_ms:.4f}"
                perf_str = f"{result.gflops:.1f}"
                print(f"{result.config.name:<20} {latency_str:<30} {perf_str:<15}")

        print(f"\n{'='*70}\n")

def main():
    benchmark = PreciseBenchmark()
    benchmark.run_comparison()

if __name__ == "__main__":
    main()
