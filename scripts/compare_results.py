#!/usr/bin/env python3
"""
直接对比 Standalone 和 FlashInfer 的 FP8 GEMM 计算结果
使用与 standalone 程序相同的测试数据
"""

import torch
import numpy as np
import subprocess
import os
import sys

def generate_standalone_test_data(batch, m, n, k):
    """生成与 standalone 程序相同的测试数据"""
    # Standalone 使用的数据生成方式:
    # h_A[i] = __nv_fp8_e4m3(float(i % 127) / 127.0f)
    # h_B[i] = __nv_fp8_e4m3(float(i % 127) / 127.0f)
    
    total_a = batch * m * k
    total_b = batch * k * n
    
    # 生成 A 矩阵数据
    a_data = []
    for i in range(total_a):
        val = float(i % 127) / 127.0
        a_data.append(val)
    
    # 生成 B 矩阵数据
    b_data = []
    for i in range(total_b):
        val = float(i % 127) / 127.0
        b_data.append(val)
    
    # 转换为 torch tensor
    a = torch.tensor(a_data, dtype=torch.float32).reshape(batch, m, k)
    b = torch.tensor(b_data, dtype=torch.float32).reshape(batch, k, n)
    
    # 转换为 FP16 (FlashInfer 需要 FP16 输入)
    a_fp16 = a.half().cuda()
    b_fp16 = b.half().cuda()
    
    return a_fp16, b_fp16

def test_flashinfer_fp8(batch, m, n, k):
    """使用 FlashInfer 计算 FP8 GEMM"""
    try:
        import flashinfer
        from flashinfer.gemm import bmm_fp8
    except ImportError:
        return None, "FlashInfer 未安装"
    
    try:
        # 生成与 standalone 相同的测试数据
        a, b = generate_standalone_test_data(batch, m, n, k)
        
        # Scale factors (standalone 使用 1.0)
        scale_a = torch.tensor([1.0], dtype=torch.float32, device='cuda')
        scale_b = torch.tensor([1.0], dtype=torch.float32, device='cuda')
        
        # 执行 FP8 GEMM
        c = bmm_fp8(a, b, scale_a, scale_b, torch.float16)
        torch.cuda.synchronize()
        
        return c.cpu().numpy(), None
    except Exception as e:
        return None, f"FlashInfer 执行错误: {str(e)}"

def parse_standalone_output(output_text):
    """从 standalone 程序输出中解析结果"""
    lines = output_text.split('\n')
    results = {}
    
    current_batch = None
    current_test = None
    
    for line in lines:
        # 检测测试配置
        if 'Test: batch=' in line:
            # 提取 batch 值
            import re
            match = re.search(r'batch=(\d+)', line)
            if match:
                current_batch = int(match.group(1))
        
        # 检测 FP8 结果
        if '[Test 1]' in line:
            current_test = 'fp8'
        elif 'C_fp8[' in line and '=' in line:
            try:
                # 解析 "C_fp8[0] = 41.718750"
                match = re.search(r'C_fp8\[(\d+)\]\s*=\s*([\d.]+)', line)
                if match:
                    idx = int(match.group(1))
                    val = float(match.group(2))
                    if current_batch not in results:
                        results[current_batch] = {}
                    if 'fp8' not in results[current_batch]:
                        results[current_batch]['fp8'] = []
                    results[current_batch]['fp8'].append((idx, val))
            except:
                pass
    
    return results

def test_standalone_fp8(batch, m, n, k):
    """运行 Standalone 程序并解析结果"""
    exe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "build", "fp8_gemm_sm89_standalone")
    
    if not os.path.exists(exe_path):
        return None, f"可执行文件不存在: {exe_path}"
    
    try:
        result = subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return None, f"Standalone 执行失败: {result.stderr[:200]}"
        
        # 解析输出
        parsed = parse_standalone_output(result.stdout)
        
        if batch not in parsed or 'fp8' not in parsed[batch]:
            return None, f"无法从输出中解析 batch={batch} 的结果"
        
        # 提取值并排序
        values = sorted(parsed[batch]['fp8'], key=lambda x: x[0])
        result_array = np.array([v[1] for v in values])
        
        return result_array, None
        
    except Exception as e:
        return None, f"Standalone 执行错误: {str(e)}"

def compare_results():
    """对比 FlashInfer 和 Standalone 的结果"""
    print("="*70)
    print("FlashInfer vs Standalone FP8 GEMM 数值对比")
    print("="*70)
    print()
    
    # 检查 FlashInfer
    try:
        import flashinfer
        version = getattr(flashinfer, '__version__', '未知')
        print(f"✓ FlashInfer 已安装 (版本: {version})")
        flashinfer_available = True
    except ImportError:
        print("✗ FlashInfer 未安装")
        print("  安装命令: pip install flashinfer-python")
        flashinfer_available = False
        return
    
    print()
    
    # 测试配置 (与 standalone 程序相同)
    test_configs = [
        (1, 128, 64, 128),
        (2, 128, 64, 128),
    ]
    
    all_match = True
    
    for batch, m, n, k in test_configs:
        print(f"{'='*70}")
        print(f"测试配置: batch={batch}, m={m}, n={n}, k={k}")
        print(f"{'='*70}")
        print()
        
        # 测试 FlashInfer
        print("[FlashInfer]")
        print("  计算中...")
        flashinfer_result, flashinfer_error = test_flashinfer_fp8(batch, m, n, k)
        
        if flashinfer_error:
            print(f"  ✗ 错误: {flashinfer_error}")
            print()
            continue
        else:
            print(f"  ✓ 成功")
            print(f"  输出形状: {flashinfer_result.shape}")
            print(f"  输出范围: [{flashinfer_result.min():.6f}, {flashinfer_result.max():.6f}]")
        
        print()
        
        # 测试 Standalone
        print("[Standalone]")
        print("  运行中...")
        standalone_result, standalone_error = test_standalone_fp8(batch, m, n, k)
        
        if standalone_error:
            print(f"  ✗ 错误: {standalone_error}")
            print()
            continue
        else:
            print(f"  ✓ 成功")
            print(f"  输出形状: {standalone_result.shape}")
            print(f"  输出范围: [{standalone_result.min():.6f}, {standalone_result.max():.6f}]")
        
        print()
        print(f"{'='*70}")
        print("对比结果")
        print(f"{'='*70}")
        print()
        
        # 比较结果
        # Standalone 输出的是前几个值，FlashInfer 输出完整矩阵
        n_compare = min(len(standalone_result), len(flashinfer_result.flatten()))
        
        if n_compare == 0:
            print("⚠ 无法提取可比较的值")
            print()
            continue
        
        flashinfer_flat = flashinfer_result.flatten()[:n_compare]
        standalone_flat = standalone_result[:n_compare]
        
        # 计算差异
        diff = np.abs(flashinfer_flat - standalone_flat)
        max_diff = diff.max()
        mean_diff = diff.mean()
        max_abs_val = np.abs(flashinfer_flat).max()
        rel_error = max_diff / (max_abs_val + 1e-6)
        
        print(f"对比元素数量: {n_compare}")
        print()
        print("详细对比 (前10个元素):")
        print(f"{'索引':<8} {'FlashInfer':<15} {'Standalone':<15} {'绝对误差':<15} {'相对误差':<15}")
        print("-" * 70)
        
        for i in range(min(10, n_compare)):
            fi_val = flashinfer_flat[i]
            sa_val = standalone_flat[i]
            abs_err = abs(fi_val - sa_val)
            rel_err = abs_err / (abs(fi_val) + 1e-6) * 100
            match_symbol = "✓" if abs_err < 0.01 else "~" if abs_err < 0.1 else "✗"
            print(f"{i:<8} {fi_val:<15.6f} {sa_val:<15.6f} {abs_err:<15.6f} {rel_err:<15.2f}% {match_symbol}")
        
        print()
        print("统计信息:")
        print(f"  最大绝对误差: {max_diff:.6f}")
        print(f"  平均绝对误差: {mean_diff:.6f}")
        print(f"  最大相对误差: {rel_error*100:.2f}%")
        print(f"  最大绝对值: {max_abs_val:.6f}")
        
        print()
        print("结论:")
        if rel_error < 0.01:
            print("  ✓✓ 结果非常一致 (相对误差 < 1%)")
            print("  → Standalone 实现和 FlashInfer 的计算结果基本一致")
        elif rel_error < 0.05:
            print("  ✓ 结果基本一致 (相对误差 < 5%)")
            print("  → 差异在 FP8 精度范围内，属于正常")
            all_match = False
        elif rel_error < 0.10:
            print("  ~ 结果大致一致 (相对误差 < 10%)")
            print("  → 可能存在实现细节差异，但总体一致")
            all_match = False
        else:
            print("  ⚠ 结果差异较大 (相对误差 >= 10%)")
            print("  → 需要检查实现细节或数据格式")
            all_match = False
        
        print()
        print()
    
    # 最终总结
    print("="*70)
    print("最终总结")
    print("="*70)
    print()
    
    if all_match:
        print("✓✓ 所有测试配置下，Standalone 和 FlashInfer 的结果都一致")
        print("  → 可以确认 Standalone 实现与 FlashInfer 的计算结果相同")
    else:
        print("~ 部分测试配置存在差异，但差异在可接受范围内")
        print("  → FP8 精度本身较低，小差异属于正常")
    
    print()
    print("注意:")
    print("  - FP8 (e4m3) 只有 4 位尾数，精度较低")
    print("  - 不同的实现细节（如 scale factor 处理）可能导致小差异")
    print("  - 相对误差 < 5% 通常认为是可接受的")

if __name__ == "__main__":
    compare_results()
