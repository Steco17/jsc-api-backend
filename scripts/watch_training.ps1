[CmdletBinding()]
param(
    [string]$RunName = "local-ewo_Latn",

    [int]$TrainingProcessId = 0,

    [ValidateRange(1, 300)]
    [int]$RefreshSeconds = 5,

    [ValidateRange(5, 200)]
    [int]$LogLines = 40,

    [switch]$Once
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root "model_out\$RunName"
$stdoutLog = Join-Path $root "model_out\$RunName.stdout.log"
$stderrLog = Join-Path $root "model_out\$RunName.stderr.log"

try {
    $Host.UI.RawUI.WindowTitle = "JSC local training - $RunName"
}
catch {
    # Some non-interactive hosts do not expose a window title.
}

do {
    Clear-Host
    Write-Host "JSC local training monitor" -ForegroundColor Cyan
    Write-Host "Run: $RunName"
    Write-Host "Updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ""

    if ($TrainingProcessId -gt 0) {
        $process = Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Trainer: RUNNING (PID $TrainingProcessId)" -ForegroundColor Green
        }
        else {
            Write-Host "Trainer: NOT RUNNING (PID $TrainingProcessId)" -ForegroundColor Yellow
        }
    }

    if (Test-Path -LiteralPath $stderrLog) {
        $progress = Get-Content -LiteralPath $stderrLog -Tail 300 |
            Where-Object { $_ -match '\d+%\|.*\|\s*\d+/\d+' } |
            Select-Object -Last 1
        if ($progress) {
            Write-Host "Progress: $progress"
        }
        else {
            Write-Host "Progress: waiting for the first training step"
        }
    }
    else {
        Write-Host "Progress log: waiting for $stderrLog"
    }

    $checkpoints = @(
        Get-ChildItem -LiteralPath $runDir -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue |
            Sort-Object { [int]($_.Name -replace '^checkpoint-', '') }
    )
    if ($checkpoints.Count -gt 0) {
        Write-Host "Latest checkpoint: $($checkpoints[-1].Name)" -ForegroundColor Green
    }
    else {
        Write-Host "Latest checkpoint: none yet"
    }

    $runtimeMetrics = Join-Path $runDir "runtime_metrics.json"
    $mergedModel = Join-Path $runDir "merged\model.safetensors"
    if ((Test-Path -LiteralPath $runtimeMetrics) -and (Test-Path -LiteralPath $mergedModel)) {
        Write-Host "Result: COMPLETE, merged model is ready" -ForegroundColor Green
    }
    else {
        Write-Host "Result: training in progress"
    }

    Write-Host ""
    Write-Host "GPU" -ForegroundColor Cyan
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        & $nvidiaSmi.Source `
            --query-gpu=name,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw `
            --format=csv,noheader
    }
    else {
        Write-Host "nvidia-smi was not found" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "Recent training log ($LogLines lines)" -ForegroundColor Cyan
    if (Test-Path -LiteralPath $stderrLog) {
        Get-Content -LiteralPath $stderrLog -Tail $LogLines
    }
    else {
        Write-Host "The training log has not been created yet."
    }

    Write-Host ""
    Write-Host "Trainer output" -ForegroundColor Cyan
    if (Test-Path -LiteralPath $stdoutLog) {
        Get-Content -LiteralPath $stdoutLog -Tail 20
    }
    else {
        Write-Host "The trainer output log has not been created yet."
    }

    Write-Host ""
    Write-Host "Close this window to stop monitoring. Training will continue."

    if (-not $Once) {
        Start-Sleep -Seconds $RefreshSeconds
    }
} while (-not $Once)
