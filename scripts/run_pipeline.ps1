# ZPPO Automated Training & Evaluation Pipeline
# This script finishes ZPPO training (up to step 1000), runs the 500-question test benchmark,
# and compiles the final master analysis report.

$ErrorActionPreference = "Stop"

$workspaceDir = "E:\math workspace"
$scratchDir = "C:\Users\rick john\.gemini\antigravity\brain\8ceedcb6-148d-478c-b186-c0bb494fe889\scratch"
$pythonPath = "E:\AI_Workspace\.venv\Scripts\python.exe"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "STAGE 1: Resuming ZPPO Training to Step 1000..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# Run the training loop with auto-recovery to completion
& powershell.exe -File "$workspaceDir\run_zppo_recovery.ps1"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "STAGE 2: Training finished! Running 500-Question Benchmark..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# Run the 500-question three-way benchmark
& $pythonPath "$scratchDir\bench500_three_way.py"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "STAGE 3: Benchmark finished! Compiling Master Report..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# Run the report compilation script
& $pythonPath "$scratchDir\zppo_generate_final_analysis.py"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "PIPELINE COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "Master report: C:\Users\rick john\.gemini\antigravity\brain\8ceedcb6-148d-478c-b186-c0bb494fe889\final_walkthrough.md" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
