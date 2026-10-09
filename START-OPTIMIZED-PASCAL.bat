@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="" (
  echo Usage: %~nx0 "D:\path\to\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" [options]
  echo Example: %~nx0 "D:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" --port 8005
  exit /b 2
)

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo ERROR: .venv is missing.
  echo Run: py -3 -m venv .venv
  echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0tools\run_qwen36_pascal_optimized.py" %*
set "STRATA_EXIT=%errorlevel%"
if not "%STRATA_EXIT%"=="0" pause
exit /b %STRATA_EXIT%
