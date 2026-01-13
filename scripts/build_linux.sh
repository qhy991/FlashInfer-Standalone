#!/bin/bash
# Build script for FlashInfer-Standalone (Linux)
# Requires: CUDA Toolkit 11.4+ with cuBLASLt support
# Supports: SM89 (Ada/RTX 40xx), SM90 (Hopper/H100/H200)

set -e  # Exit on error

echo "========================================"
echo "Building FlashInfer-Standalone"
echo "========================================"
echo ""

# Check if nvcc exists
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found in PATH"
    echo "Please install CUDA Toolkit and add to PATH"
    exit 1
fi

# Display CUDA version
nvcc --version
echo ""

# Use specified architecture or default to SM90 (H200)
if [ -z "$1" ]; then
    # Default to SM90 for H200/H100
    CUDA_ARCH="sm_90"
    echo "Using default architecture: $CUDA_ARCH (H200/H100/SM90)"
    echo "  (To use SM89 for RTX 40xx, run: $0 sm_89)"
else
    # Use specified architecture
    CUDA_ARCH="$1"
    echo "Using specified architecture: $CUDA_ARCH"
fi

CXX_FLAGS="-O2 -std=c++17"
NVCC_FLAGS="-arch=$CUDA_ARCH $CXX_FLAGS"

echo "Compiler flags: $NVCC_FLAGS"
echo ""

# Create build directory
mkdir -p build

# Build standalone test version
echo "Building standalone test version..."
nvcc $NVCC_FLAGS \
    src/fp8_gemm_sm89_standalone.cu \
    -lcublasLt -lcublas \
    -o build/fp8_gemm_sm89_standalone

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Build benchmark version
echo "Building benchmark version..."
nvcc $NVCC_FLAGS \
    src/fp8_gemm_benchmark.cu \
    -lcublasLt -lcublas \
    -o build/fp8_gemm_benchmark

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Build SiLU_and_Mul standalone version
echo "Building SiLU_and_Mul standalone version..."
nvcc $NVCC_FLAGS \
    src/silu_and_mul_sm89_standalone.cu \
    -o build/silu_and_mul_sm89_standalone

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Build RMSNorm standalone version
echo "Building RMSNorm standalone version..."
nvcc $NVCC_FLAGS \
    src/rmsnorm_sm89_standalone.cu \
    -o build/rmsnorm_sm89_standalone

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Build GQA Decode standalone version
echo "Building GQA Decode standalone version..."
nvcc $NVCC_FLAGS \
    src/gqa_decode_sm89_standalone.cu \
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
