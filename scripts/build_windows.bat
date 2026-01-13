@echo off
REM Build script for FP8 GEMM standalone implementation (Windows)
REM Requires: CUDA Toolkit 11.4+ with cuBLASLt support

echo ========================================
echo Building SM89 FP8 GEMM Standalone
echo ========================================
echo.

REM Check if nvcc exists
where nvcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: nvcc not found in PATH
    echo Please install CUDA Toolkit and add to PATH
    exit /b 1
)

REM Display CUDA version
nvcc --version
echo.

REM Set compiler flags
set CUDA_ARCH=sm_89
set CXX_FLAGS=-O3 -std=c++17
set NVCC_FLAGS=-arch=%CUDA_ARCH% %CXX_FLAGS%

echo Compiler flags: %NVCC_FLAGS%
echo.

REM Build standalone implementation
echo Building standalone implementation...
nvcc %NVCC_FLAGS% ^
    src\fp8_gemm_sm89_standalone.cu ^
    -lcublasLt -lcublas ^
    -o build\fp8_gemm_sm89_standalone.exe

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo ========================================
echo Build successful!
echo ========================================
echo.
echo Output: build\fp8_gemm_sm89_standalone.exe
echo.
echo To run:
echo   build\fp8_gemm_sm89_standalone.exe
echo.
