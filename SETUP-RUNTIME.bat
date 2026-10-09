@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where py >nul 2>nul || (
  echo ERROR: Python 3.10 or newer is required.
  echo Download it from https://www.python.org/downloads/windows/
  exit /b 1
)

py -3 -m venv .venv
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" -m pip install -r requirements-runtime.txt
if errorlevel 1 exit /b 1

echo.
echo Runtime environment is ready.
echo Start with: START-OPTIMIZED-PASCAL.bat "D:\path\to\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
