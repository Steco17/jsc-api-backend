[CmdletBinding()]
param(
    [string]$BasePython = "",

    [switch]$Rebuild,

    [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    throw "uv was not found. Install uv from https://docs.astral.sh/uv/ and rerun this script."
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if ($Rebuild -or -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    if (-not $BasePython) {
        $pathPython = Get-Command python -ErrorAction SilentlyContinue
        $candidates = @()
        if ($pathPython -and $pathPython.Source -notlike "*\WindowsApps\*") {
            $candidates += $pathPython.Source
        }
        $candidates += @(
            "$env:USERPROFILE\anaconda3\python.exe",
            "$env:USERPROFILE\miniconda3\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
            "C:\Program Files\Python312\python.exe",
            "C:\Program Files\Python311\python.exe"
        )
        $BasePython = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    }
    if (-not $BasePython) {
        throw "Python 3.11 or 3.12 was not found. Pass its path with -BasePython."
    }

    $version = & $BasePython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0 -or $version -notin @("3.11", "3.12")) {
        throw "Local setup requires Python 3.11 or 3.12, but '$BasePython' reports '$version'."
    }

    $venvPath = [IO.Path]::GetFullPath((Join-Path $root ".venv"))
    if (-not $venvPath.StartsWith([IO.Path]::GetFullPath($root))) {
        throw "Refusing to rebuild an environment outside the repository."
    }
    & $uv.Source venv --clear --python $BasePython $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create the local virtual environment."
    }
}

$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"

# Use an official CUDA wheel instead of allowing a CPU-only Torch build.
& $uv.Source pip install `
    --python $venvPython `
    --index-url "https://download.pytorch.org/whl/cu128" `
    "torch==2.7.0+cu128"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to install CUDA-enabled PyTorch."
}

& $uv.Source pip install `
    --python $venvPython `
    -r requirements-dev.txt `
    -r requirements-train.txt
if ($LASTEXITCODE -ne 0) {
    throw "Unable to install the pinned project dependencies."
}

& $venvPython -c @'
import torch

if not torch.cuda.is_available():
    raise RuntimeError('PyTorch cannot access the NVIDIA GPU')

gpu = torch.cuda.get_device_properties(0)
print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')
print(f'GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GiB)')
'@
if ($LASTEXITCODE -ne 0) {
    throw "The CUDA preflight check failed."
}

if (-not $SkipTests) {
    $testTemp = Join-Path $root ".pytest-local-setup"
    & $venvPython -m pytest -q -W error --basetemp $testTemp
    $testExit = $LASTEXITCODE
    $resolvedRoot = [IO.Path]::GetFullPath($root).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $resolvedTemp = [IO.Path]::GetFullPath($testTemp)
    $safeTestTemp = (
        $resolvedTemp.StartsWith($resolvedRoot + [IO.Path]::DirectorySeparatorChar) -and
        (Split-Path -Leaf $resolvedTemp) -eq ".pytest-local-setup"
    )
    if ($safeTestTemp -and (Test-Path $resolvedTemp)) {
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
    }
    if ($testExit -ne 0) {
        throw "The local test suite failed."
    }
}

Write-Host "Local training environment is ready."
Write-Host "Next: .\scripts\train_local.ps1 -Mode smoke"
