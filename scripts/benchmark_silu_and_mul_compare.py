#!/usr/bin/env python3
"""
SiLU_and_Mul 性能对比脚本 - FlashInfer vs Standalone

测试不同配置下的性能对比:
- Small (hidden_dim=128)
- Medium (hidden_dim=512)
- Large (hidden_dim=2048)
- XLarge (hidden_dim=4096) - LLaMA-7B hidden size
- XXLarge (hidden_dim=8192) - LLaMA-70B hidden size
"""

import torch
import subprocess
import numpy as np
from dataclasses import dataclass
from typing import List
import time
import os

# 检查 FlashInfer
try:
    from flashinfer import silu_and_mul
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    print("Warning: FlashInfer not available, only testing standalone")

if not torch.cuda.is_available():
    print("Error: CUDA 不可用")
    exit(1)

@dataclass
class TestConfig:
    batch: int
    hidden: int
    name: str

@dataclass
class TestResult:
    name: str
    config: TestConfig
    flashinfer_latency_ms: float
    standalone_latency_ms: float
    flashinfer_throughput_gflops: float
    standalone_throughput_gflops: float
    speedup: float

class SiluAndMulBenchmark:
    def __init__(self):
        self.device = torch.device('cuda')
        prop = torch.cuda.get_device_properties(0)

        print(f"\n{'='*70}")
        print(f"SiLU_and_Mul 性能对比测试")
        print(f"{'='*70}")
        print(f"GPU: {prop.name}")
        print(f"Compute Capability: {prop.major}.{prop.minor}")
        print(f"FlashInfer: {'可用' if FLASHINFER_AVAILABLE else '未安装'}")
        print(f"{'='*70}\n")

        # 编译 standalone
        self.compile_standalone()

    def compile_standalone(self):
        """编译 standalone 实现"""
        print("编译 standalone 实现...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        build_script = os.path.join(script_dir, "../scripts/build_linux.sh")

        # 检查是否已经编译
        standalone_path = os.path.join(script_dir, "../build/silu_and_mul_sm89_standalone")
        if not os.path.exists(standalone_path):
            # 需要编译
            result = subprocess.run(
                ["bash", build_script, "sm_89"],
                capture_output=True,
                text=True,
                cwd=os.path.join(script_dir, "..")
            )
            if result.returncode != 0:
                print(f"编译失败: {result.stderr}")
                exit(1)

        self.standalone_path = standalone_path
        print("Standalone 编译完成\n")

    def test_flashinfer(self, config: TestConfig, num_iters: int = 100) -> float:
        """测试 FlashInfer 性能"""
        if not FLASHINFER_AVAILABLE:
            return 0.0

        # 准备数据
        input_data = torch.randn(config.batch, 2 * config.hidden,
                                  dtype=torch.float16, device=self.device)

        # 预热（触发 JIT 编译）
        for _ in range(30):
            output = silu_and_mul(input_data)
        torch.cuda.synchronize()

        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            output = silu_and_mul(input_data)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = elapsed_ms / num_iters

        # 计算 GFLOPS
        ops = config.batch * config.hidden * 5  # SiLU(2 ops) + mul(1 op) + load/store
        gflops = ops / (avg_latency_ms / 1000.0) / 1e9

        return avg_latency_ms, gflops

    def test_standalone(self, config: TestConfig, num_iters: int = 100) -> float:
        """测试 Standalone 性能"""
        # 运行 standalone 程序
        # 注意: standalone 已经内置了计时和多次运行
        # 这里我们直接运行它并解析输出

        # 创建临时输入文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            input_file = f.name
            # 写入配置信息
            f.write(f"{config.batch} {config.hidden} {num_iters}\n")

        try:
            result = subprocess.run(
                [self.standalone_path, "benchmark", input_file],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                print(f"Standalone 运行错误: {result.stderr}")
                return 0.0, 0.0

            # 解析输出
            # 查找类似 "Average latency: 0.0027 ms" 的行
            for line in result.stdout.split('\n'):
                if 'Average latency' in line:
                    # 提取延迟值
                    parts = line.split(':')
                    if len(parts) >= 2:
                        latency_str = parts[1].strip().replace('ms', '').strip()
                        avg_latency_ms = float(latency_str)

                        # 计算 GFLOPS
                        ops = config.batch * config.hidden * 5
                        gflops = ops / (avg_latency_ms / 1000.0) / 1e9

                        return avg_latency_ms, gflops
        finally:
            os.unlink(input_file)

        return 0.0, 0.0

    def run_comparison(self):
        """运行完整对比"""
        configs = [
            TestConfig(1, 128, "Small (128)"),
            TestConfig(1, 512, "Medium (512)"),
            TestConfig(1, 2048, "Large (2048)"),
            TestConfig(1, 4096, "XLarge (4096) - LLaMA-7B"),
            TestConfig(4, 2048, "Batch4 (2048)"),
            TestConfig(16, 4096, "Batch16 (4096)"),
        ]

        results = []

        for config in configs:
            print(f"\n{'='*70}")
            print(f"测试: {config.name}")
            print(f"配置: batch={config.batch}, hidden_dim={config.hidden}")
            print(f"计算量: {config.batch * config.hidden * 5 / 1e6:.2f} MOPS")
            print(f"{'='*70}")

            # FlashInfer 测试
            flashinfer_latency = flashinfer_gflops = 0.0
            if FLASHINFER_AVAILABLE:
                print("\n[FlashInfer] 运行 100 次...")
                flashinfer_latency, flashinfer_gflops = self.test_flashinfer(config)
                print(f"  延迟: {flashinfer_latency:.4f} ms")
                print(f"  吞吐量: {flashinfer_gflops:.1f} GFLOPS")
            else:
                print("\n[FlashInfer] 跳过 (未安装)")

            # Standalone 测试
            print("\n[Standalone] 运行...")
            standalone_latency, standalone_gflops = self.test_standalone(config)
            if standalone_latency > 0:
                print(f"  延迟: {standalone_latency:.4f} ms")
                print(f"  吞吐量: {standalone_gflops:.1f} GFLOPS")
            else:
                print("  错误: 无法获取结果")

            # 计算加速比
            if FLASHINFER_AVAILABLE and standalone_latency > 0:
                speedup = flashinfer_latency / standalone_latency
                print(f"\n加速比: Standalone 是 FlashInfer 的 {speedup:.2f}x")

            results.append(TestResult(
                config.name,
                config,
                flashinfer_latency,
                standalone_latency,
                flashinfer_gflops,
                standalone_gflops,
                flashinfer_latency / standalone_latency if standalone_latency > 0 else 0
            ))

        # 打印总结
        self.print_summary(results)

    def print_summary(self, results: List[TestResult]):
        """打印总结表格"""
        print(f"\n{'='*70}")
        print("性能总结 (延迟越低越好, 吞吐量越高越好)")
        print(f"{'='*70}\n")

        print(f"{'配置':<25} {'FlashInfer':<20} {'Standalone':<20} {'加速比':<10}")
        print(f"{'':25} {'延迟(GFLOPS)':<20} {'延迟(GFLOPS)':<20} {'':10}")
        print("-" * 75)

        for result in results:
            fi_str = f"{result.flashinfer_latency_ms:.4f}ms ({result.flashinfer_throughput_gflops:.1f})" if result.flashinfer_latency_ms > 0 else "N/A"
            sa_str = f"{result.standalone_latency_ms:.4f}ms ({result.standalone_throughput_gflops:.1f})" if result.standalone_latency_ms > 0 else "N/A"
            speedup_str = f"{result.speedup:.2f}x" if result.speedup > 0 else "N/A"

            print(f"{result.config.name:<25} {fi_str:<20} {sa_str:<20} {speedup_str:<10}")

        print(f"\n{'='*70}\n")

def main():
    benchmark = SiluAndMulBenchmark()
    benchmark.run_comparison()

if __name__ == "__main__":
    main()
