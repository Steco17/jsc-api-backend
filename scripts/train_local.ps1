[CmdletBinding()]
param(
    [ValidateSet("smoke", "pilot", "language", "all")]
    [string]$Mode = "smoke",

    [string[]]$TargetLanguage = @(),

    [ValidateRange(1, 1000000)]
    [int]$SamplesPerDirection = 500,

    [string]$Python = "",

    [string]$OutDir = "",

    [switch]$KeepSmoke,

    [switch]$SkipValidation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Python) {
    $projectPython = Join-Path $root ".venv\Scripts\python.exe"
    $profilePython = "$env:USERPROFILE\.venvs\jsc-translator\Scripts\python.exe"
    if (Test-Path -LiteralPath $projectPython -PathType Leaf) {
        $Python = $projectPython
    }
    else {
        $Python = $profilePython
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python was not found at '$Python'. Run scripts\setup_local.ps1 first."
}

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    throw "nvidia-smi was not found. Local training requires an NVIDIA CUDA GPU."
}

$memoryText = & $nvidiaSmi.Source --query-gpu=memory.total --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or -not $memoryText) {
    throw "Unable to query GPU memory with nvidia-smi."
}
$memoryMiB = [int]($memoryText | Select-Object -First 1)
Write-Host "Detected NVIDIA GPU with $memoryMiB MiB VRAM."

if ($memoryMiB -lt 3900) {
    throw "At least 4 GB of GPU memory is required by the local profile."
}
if ($Mode -in @("pilot", "language") -and $TargetLanguage.Count -eq 0) {
    throw "Mode '$Mode' requires -TargetLanguage, for example ewo_Latn."
}
if ($Mode -notin @("pilot", "language") -and $TargetLanguage.Count -gt 0) {
    throw "-TargetLanguage is only valid with -Mode pilot or -Mode language."
}

if (-not $OutDir) {
    if ($Mode -eq "smoke") {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $OutDir = "model_smoke\local-$stamp"
    }
    elseif ($Mode -eq "language") {
        $label = $TargetLanguage -join "_"
        $OutDir = "model_out\local-$label"
    }
    elseif ($Mode -eq "pilot") {
        $label = $TargetLanguage -join "_"
        $OutDir = "model_out\pilot-$label"
    }
    else {
        $OutDir = "model_out\local-all"
    }
}

$trainFile = "data\prepared\train.jsonl"
$devFile = "data\prepared\dev.jsonl"
if ($Mode -ne "smoke" -and -not $SkipValidation) {
    & $Python scripts\validate_prepared.py data\prepared
    if ($LASTEXITCODE -ne 0) {
        throw "Prepared-data validation failed."
    }
}

if ($Mode -in @("smoke", "pilot")) {
    $subsetLabel = if ($Mode -eq "smoke") { "smoke" } else { "pilot-" + ($TargetLanguage -join "_") }
    $subsetDir = ".local-training\$subsetLabel"
    $trainSamples = if ($Mode -eq "smoke") { 2 } else { $SamplesPerDirection }
    $devSamples = if ($Mode -eq "smoke") { 1 } else { [Math]::Max(1, [Math]::Min(25, [Math]::Ceiling($SamplesPerDirection / 10))) }
    $sampleArgs = @(
        "scripts\sample_prepared.py",
        "data\prepared",
        "--out-dir", $subsetDir,
        "--train-per-direction", $trainSamples,
        "--dev-per-direction", $devSamples,
        "--strategy", $(if ($Mode -eq "smoke") { "first" } else { "reservoir" })
    )
    if ($TargetLanguage.Count -gt 0) {
        $sampleArgs += "--targets"
        $sampleArgs += $TargetLanguage
    }
    & $Python @sampleArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create the direction-balanced local training subset."
    }
    $trainFile = Join-Path $subsetDir "train.jsonl"
    $devFile = Join-Path $subsetDir "dev.jsonl"
}

$trainingArgs = @(
    "scripts\finetune.py",
    "--train", $trainFile,
    "--dev", $devFile,
    "--out", $OutDir,
    "--epochs", "1",
    "--batch", "1",
    "--eval-batch", "1",
    "--grad-accum", "32",
    "--eval-steps", "500",
    "--max-length", "128",
    "--empty-cache-steps", "25",
    "--dataloader-workers", "0",
    "--precision", "auto"
)

if ($Mode -eq "smoke") {
    $trainingArgs += @(
        "--max-steps", "2",
        "--grad-accum", "1",
        "--eval-steps", "1",
        "--resume", "never"
    )
}
elseif ($Mode -eq "language") {
    $trainingArgs += "--target-langs"
    $trainingArgs += $TargetLanguage
}
else {
    if ($Mode -eq "all") {
        Write-Warning "The all-language epoch contains 3.38 million rows and can take months on 4 GB VRAM."
    }
}

$env:HF_HOME = Join-Path $root ".hf-cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Write-Host "Training output: $OutDir"
& $Python @trainingArgs
if ($LASTEXITCODE -ne 0) {
    throw "Local training failed with exit code $LASTEXITCODE."
}

if ($Mode -eq "smoke" -and -not $KeepSmoke) {
    $smokeRoot = [IO.Path]::GetFullPath((Join-Path $root "model_smoke"))
    $smokeTarget = [IO.Path]::GetFullPath((Join-Path $root $OutDir))
    if (-not $smokeTarget.StartsWith($smokeRoot + [IO.Path]::DirectorySeparatorChar)) {
        throw "Refusing to clean an unexpected smoke output path: $smokeTarget"
    }
    Remove-Item -LiteralPath $smokeTarget -Recurse -Force
    Write-Host "Smoke artifacts removed. Use -KeepSmoke to retain them."
}
