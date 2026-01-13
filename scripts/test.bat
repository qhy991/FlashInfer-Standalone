@echo off
REM Test script for FP8 GEMM standalone implementation

echo ========================================
echo Testing SM89 FP8 GEMM Standalone
echo ========================================
echo.

REM Check if executable exists
if not exist "build\fp8_gemm_sm89_standalone.exe" (
    echo Error: Executable not found
    echo Please run build script first:
    echo   scripts\build_windows.bat
    exit /b 1
)

REM Run the test
echo Running FP8 GEMM test...
echo.
build\fp8_gemm_sm89_standalone.exe

if %ERRORLEVEL% neq 0 (
    echo.
    echo Test failed with error code %ERRORLEVEL%
    exit /b 1
)

echo.
echo ========================================
echo Test completed successfully!
echo ========================================
