#!/bin/bash
# Test script for FP8 GEMM standalone implementation

echo "========================================"
echo "Testing SM89 FP8 GEMM Standalone"
echo "========================================"
echo ""

# Check if executable exists
if [ ! -f "build/fp8_gemm_sm89_standalone" ]; then
    echo "Error: Executable not found"
    echo "Please run build script first:"
    echo "  ./scripts/build_linux.sh"
    exit 1
fi

# Run the test
echo "Running FP8 GEMM test..."
echo ""
./build/fp8_gemm_sm89_standalone

if [ $? -ne 0 ]; then
    echo ""
    echo "Test failed with error code $?"
    exit 1
fi

echo ""
echo "========================================"
echo "Test completed successfully!"
echo "========================================"
