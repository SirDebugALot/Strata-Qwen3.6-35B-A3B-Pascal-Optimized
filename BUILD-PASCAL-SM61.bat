@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "LLAMA_DIR=%~dp0third_party\llama.cpp"
set "BUILD_DIR=%~dp0build-sm61"
set "LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d"

where git >nul 2>nul || (echo ERROR: Git is required on PATH.& exit /b 1)
where cmake >nul 2>nul || (echo ERROR: CMake is required on PATH.& exit /b 1)
where ninja >nul 2>nul || (echo ERROR: Ninja is required on PATH.& exit /b 1)
where nvcc >nul 2>nul || (echo ERROR: CUDA Toolkit 12.x with nvcc is required on PATH.& exit /b 1)
for /f "delims=" %%I in ('where nvcc') do if not defined NVCC set "NVCC=%%I"
for /f "delims=" %%I in ('where ninja') do if not defined NINJA set "NINJA=%%I"
for %%I in ("%NVCC%\..\..") do set "CUDA_ROOT=%%~fI"

where cl >nul 2>nul
if errorlevel 1 (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if not exist "%VSWHERE%" (
    echo ERROR: Visual Studio 2022 Build Tools with Desktop C++ is required.
    exit /b 1
  )
  for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%I"
  if not defined VSROOT (
    echo ERROR: Visual Studio C++ tools were not found.
    exit /b 1
  )
  call "%VSROOT%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
  if errorlevel 1 exit /b 1
)

if not exist "%LLAMA_DIR%\.git" (
  echo Fetching pinned llama.cpp source...
  git clone https://github.com/ggml-org/llama.cpp.git "%LLAMA_DIR%"
  if errorlevel 1 exit /b 1
)

git -C "%LLAMA_DIR%" fetch --depth 1 origin "%LLAMA_COMMIT%"
if errorlevel 1 exit /b 1
git -C "%LLAMA_DIR%" checkout --detach "%LLAMA_COMMIT%"
if errorlevel 1 exit /b 1

cmake -S "%~dp0engines\qwen35moe" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TRY_COMPILE_CONFIGURATION=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=61 ^
  -DCMAKE_CUDA_COMPILER="%NVCC%" ^
  -DCUDAToolkit_ROOT="%CUDA_ROOT%" ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DSQ_ENABLE_PREFILL_MMQ=ON ^
  -DSQ_LLAMA_DIR="%LLAMA_DIR%"
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --target strata-qwen35moe -j
if errorlevel 1 exit /b 1

echo.
echo Built: %BUILD_DIR%\strata-qwen35moe.exe
exit /b 0
