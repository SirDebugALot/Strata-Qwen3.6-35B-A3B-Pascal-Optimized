param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$BuildDir = "build-sm61",
    [string]$OutputDir = "dist",
    [string]$CudaBin = "",
    [string]$VcRedistDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$build = if ([IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $root $BuildDir }
$output = if ([IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $root $OutputDir }
$safeVersion = $Version -replace '[^A-Za-z0-9._-]', '-'
$packageName = "Strata-Qwen3.6-35B-A3B-Pascal-Optimized-$safeVersion-windows-sm61"
$stage = Join-Path $output $packageName
$engine = Join-Path $build "strata-qwen35moe.exe"

if (-not (Test-Path -LiteralPath $engine -PathType Leaf)) {
    throw "Engine not found: $engine"
}
if (-not $CudaBin) {
    if (-not $env:CUDA_PATH) {
        throw "CUDA_PATH is not set; pass -CudaBin explicitly"
    }
    $CudaBin = Join-Path $env:CUDA_PATH "bin"
}

New-Item -ItemType Directory -Force -Path $output | Out-Null
if (Test-Path -LiteralPath $stage) {
    Remove-Item -LiteralPath $stage -Recurse -Force
}
$releaseBin = Join-Path $stage "build-sm61"
New-Item -ItemType Directory -Force -Path $releaseBin | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "tools") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "data") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "licenses") | Out-Null

Copy-Item -LiteralPath $engine -Destination $releaseBin
foreach ($dll in @("cublas64_12.dll", "cublasLt64_12.dll")) {
    $source = Join-Path $CudaBin $dll
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required CUDA runtime DLL not found: $source"
    }
    Copy-Item -LiteralPath $source -Destination $releaseBin
}

$vcDlls = @("msvcp140.dll", "msvcp140_atomic_wait.dll", "vcruntime140.dll", "vcruntime140_1.dll")
$vsRoots = @(
    (Join-Path $env:ProgramFiles "Microsoft Visual Studio"),
    (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio")
) | Where-Object { Test-Path -LiteralPath $_ }
foreach ($dll in $vcDlls) {
    $candidate = $null
    if ($VcRedistDir) {
        $explicit = Join-Path $VcRedistDir $dll
        if (Test-Path -LiteralPath $explicit -PathType Leaf) {
            $candidate = Get-Item -LiteralPath $explicit
        }
    } else {
        foreach ($vsRoot in $vsRoots) {
            $candidate = Get-ChildItem -LiteralPath $vsRoot -Filter $dll -File -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -match 'VC\\Redist\\MSVC.*\\x64\\Microsoft\.VC14.*\.CRT' } |
                Sort-Object FullName -Descending |
                Select-Object -First 1
            if ($candidate) { break }
        }
    }
    if (-not $candidate) {
        throw "Required Visual C++ runtime DLL not found: $dll"
    }
    Copy-Item -LiteralPath $candidate.FullName -Destination $releaseBin
}

foreach ($file in @(
    "START-OPTIMIZED-PASCAL.bat",
    "SETUP-RUNTIME.bat",
    "requirements-runtime.txt",
    "README.md",
    "BINARY-RELEASE.md",
    "SOURCE-PROVENANCE.md",
    "LICENSE"
)) {
    Copy-Item -LiteralPath (Join-Path $root $file) -Destination $stage
}
Copy-Item -LiteralPath (Join-Path $root "tools\run_qwen36_pascal_optimized.py") -Destination (Join-Path $stage "tools")
Copy-Item -LiteralPath (Join-Path $root "tools\strata_tokenizer.py") -Destination (Join-Path $stage "tools")
$releaseServe = Join-Path $stage "serve"
New-Item -ItemType Directory -Force -Path $releaseServe | Out-Null
foreach ($file in @("__init__.py", "frontend.py", "mcp.py", "server.py", "telemetry.py", "winjob.py")) {
    Copy-Item -LiteralPath (Join-Path $root "serve\$file") -Destination $releaseServe
}
Copy-Item -LiteralPath (Join-Path $root "serve\web") -Destination $releaseServe -Recurse
Copy-Item -LiteralPath (Join-Path $root "licenses\NVIDIA-CUDA-LICENSE.txt") -Destination (Join-Path $stage "licenses")
Copy-Item -LiteralPath (Join-Path $root "licenses\THIRD-PARTY-NOTICES.md") -Destination (Join-Path $stage "licenses")

$profile = Join-Path $root "data\expert-profile-qwen36.bin"
if (Test-Path -LiteralPath $profile -PathType Leaf) {
    Copy-Item -LiteralPath $profile -Destination (Join-Path $stage "data")
}

$buildInfo = [ordered]@{
    version = $Version
    git_commit = (git -C $root rev-parse HEAD).Trim()
    target = "windows-x64-cuda12.9-sm61"
    cuda_toolkit = "12.9"
    model_included = $false
    tested_model = "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
}
$buildInfo | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage "BUILD-INFO.json") -Encoding utf8

$zip = Join-Path $output "$packageName.zip"
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}
Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal
$digest = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
$hashFile = "$zip.sha256"
"$digest  $([IO.Path]::GetFileName($zip))" | Set-Content -LiteralPath $hashFile -Encoding ascii

Write-Host "Created $zip"
Write-Host "Created $hashFile"
