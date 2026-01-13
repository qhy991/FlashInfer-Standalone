#!/bin/bash
# Build script for FP8 GEMM standalone implementation (Linux)
# Requires: CUDA Toolkit 11.4+ with cuBLASLt support
# Supports: SM89 (Ada/RTX 40xx), SM90 (Hopper/H100/H200)

set -e  # Exit on error

echo "========================================"
echo "Building FP8 GEMM Standalone"
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

CXX_FLAGS="-O3 -std=c++17"
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

echo ""
echo "========================================"
echo "Build successful!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  - build/fp8_gemm_sm89_standalone (test version)"
echo "  - build/fp8_gemm_benchmark (benchmark version)"
echo ""
echo "To run:"
echo "  ./build/fp8_gemm_sm89_standalone   # Basic test"
echo "  ./build/fp8_gemm_benchmark          # Performance benchmark"
echo ""
