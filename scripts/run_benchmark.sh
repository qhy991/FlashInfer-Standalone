#!/bin/bash
# 运行 FlashInfer vs Standalone 性能对比测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "FlashInfer vs Standalone 性能对比"
echo "=========================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: python3 未找到"
    exit 1
fi

# 检查 FlashInfer
if ! python3 -c "import flashinfer" 2>/dev/null; then
    echo "警告: FlashInfer 未安装"
    echo "安装: pip install flashinfer-python"
    echo ""
    read -p "是否继续测试？(y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查可执行文件
if [ ! -f "build/fp8_gemm_benchmark" ]; then
    echo "警告: benchmark 可执行文件不存在"
    echo "正在编译..."
    bash scripts/build_linux.sh
fi

# 运行性能对比
python3 scripts/benchmark_compare.py

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
