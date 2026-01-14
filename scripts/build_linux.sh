#!/bin/bash
# Build script for FlashInfer-Standalone (Linux)
# Requires: CUDA Toolkit 11.4+ with cuBLASLt support
# Supports: SM89 (Ada/RTX 40xx), SM90 (Hopper/H100/H200)

set -e  # Exit on error

echo "========================================"
echo "Building FlashInfer-Standalone"
echo "========================================"
echo ""

# Source CUDA 12.8 environment if available
if [ -f /etc/profile.d/cuda-12.8.sh ]; then
    echo "Loading CUDA 12.8 environment..."
    source /etc/profile.d/cuda-12.8.sh
fi

# Set CUDA 12.8 runtime for linking
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH includes: /usr/local/cuda-12.8/targets/x86_64-linux/lib"
echo ""

# Check if nvcc exists (use system CUDA 12.2 for compilation)
if ! command -v nvcc &> /dev/null; then
    # Try to use CUDA 12.2 nvcc
    export PATH="/usr/local/cuda-12.2/bin:$PATH"
fi

if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found in PATH"
    echo "Please install CUDA Toolkit and add to PATH"
    exit 1
fi

# Display CUDA version
echo "Compiler: $(which nvcc)"
nvcc --version | head -3
echo ""

# Use specified architecture or default to SM89 for RTX 40xx
if [ -z "$1" ]; then
    # Default to SM89 for RTX 40xx
    CUDA_ARCH="sm_89"
    echo "Using default architecture: $CUDA_ARCH (RTX 40xx/SM89)"
    echo "  (To use SM90 for H100/H200, run: $0 sm_90)"
else
    # Use specified architecture
    CUDA_ARCH="$1"
    echo "Using specified architecture: $CUDA_ARCH"
fi

CXX_FLAGS="-O2 -std=c++17"
NVCC_FLAGS="-arch=$CUDA_ARCH $CXX_FLAGS"

echo "Compiler flags: $NVCC_FLAGS"
echo "Linking with CUDA 12.8 runtime for FP8 support"
echo ""

# Create build directory
mkdir -p build

# Build standalone test version
echo "Building standalone test version..."
nvcc $NVCC_FLAGS \
    -I/usr/local/cuda-12.8/include \
    -I/usr/local/cuda-12.8/targets/x86_64-linux/include \
    -L/usr/local/cuda-12.8/targets/x86_64-linux/lib \
    -L/usr/local/cuda-12.8/lib64 \
    src/fp8_gemm_sm89_standalone.cu \
    -lcublasLt -lcublas -lcudart \
    -o build/fp8_gemm_sm89_standalone

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Build benchmark version
echo "Building benchmark version..."
nvcc $NVCC_FLAGS \
    -I/usr/local/cuda-12.8/include \
    -I/usr/local/cuda-12.8/targets/x86_64-linux/include \
    -L/usr/local/cuda-12.8/targets/x86_64-linux/lib \
    -L/usr/local/cuda-12.8/lib64 \
    src/fp8_gemm_benchmark.cu \
    -lcublasLt -lcublas -lcudart \
    -o build/fp8_gemm_benchmark

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Build SiLU_and_Mul standalone version
echo "Building SiLU_and_Mul standalone version..."
nvcc $NVCC_FLAGS \
    -I/usr/local/cuda-12.8/include \
    -I/usr/local/cuda-12.8/targets/x86_64-linux/include \
    -L/usr/local/cuda-12.8/targets/x86_64-linux/lib \
    -L/usr/local/cuda-12.8/lib64 \
    src/silu_and_mul_sm89_standalone.cu \
    -lcudart \
    -o build/silu_and_mul_sm89_standalone

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Build RMSNorm standalone version
echo "Building RMSNorm standalone version..."
nvcc $NVCC_FLAGS \
    -I/usr/local/cuda-12.8/include \
    -I/usr/local/cuda-12.8/targets/x86_64-linux/include \
    -L/usr/local/cuda-12.8/targets/x86_64-linux/lib \
    -L/usr/local/cuda-12.8/lib64 \
    src/rmsnorm_sm89_standalone.cu \
    -lcudart \
    -o build/rmsnorm_sm89_standalone

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Build GQA Decode standalone version
echo "Building GQA Decode standalone version (with CUDA 12.8 FP8 support)..."
nvcc $NVCC_FLAGS \
    -I/usr/local/cuda-12.8/include \
    -I/usr/local/cuda-12.8/targets/x86_64-linux/include \
    -L/usr/local/cuda-12.8/targets/x86_64-linux/lib \
    -L/usr/local/cuda-12.8/lib64 \
    src/gqa_decode_sm89_standalone.cu \
    -lcudart \
    -o build/gqa_decode_sm89_standalone

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Build successful!"
echo "========================================"
echo ""
echo "Build configuration:"
echo "  Compiler: CUDA 12.2 (nvcc)"
echo "  Runtime:  CUDA 12.8 (linked libraries)"
echo "  Architecture: $CUDA_ARCH"
echo ""
echo "This configuration uses CUDA 12.8 runtime at execution time,"
echo "which includes the fixed FP8 conversion functions!"
echo ""
echo "Outputs:"
echo "  - build/fp8_gemm_sm89_standalone (test version)"
echo "  - build/fp8_gemm_benchmark (benchmark version)"
echo "  - build/silu_and_mul_sm89_standalone (SiLU_and_Mul)"
echo "  - build/rmsnorm_sm89_standalone (RMSNorm)"
echo "  - build/gqa_decode_sm89_standalone (GQA Decode)"
echo ""
echo "To run:"
echo "  ./build/fp8_gemm_sm89_standalone        # Basic FP8 GEMM test"
echo "  ./build/fp8_gemm_benchmark               # FP8 GEMM benchmark"
echo "  ./build/silu_and_mul_sm89_standalone     # SiLU_and_Mul test"
echo "  ./build/rmsnorm_sm89_standalone          # RMSNorm test"
echo "  ./build/gqa_decode_sm89_standalone      # GQA Decode test"
echo "  ./build/gqa_decode_sm89_standalone paged # GQA Decode with paged KV cache"
echo ""
