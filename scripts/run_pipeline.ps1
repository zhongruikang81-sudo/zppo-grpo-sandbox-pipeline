# ZPPO Automated Training & Evaluation Pipeline
# This script finishes ZPPO training (up to step 1000), runs the 500-question test benchmark,
# and compiles the final master analysis report.

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
# Override with the PYTHON environment variable if you need a specific interpreter.
$pythonPath = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "STAGE 1: Resuming ZPPO Training to Step 1000..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# Run the training loop with auto-recovery to completion
& powershell.exe -File (Join-Path $PSScriptRoot "run_zppo_recovery.ps1")

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "STAGE 2: Training finished! Running 500-Question Benchmark..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# Run the 500-question comparative benchmark (auto-compiles the report into results/)
& $pythonPath (Join-Path $repoRoot "evaluation\run_bench500.py")

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "PIPELINE COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "Benchmark report: results\bench500_overfitting_report.md" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
