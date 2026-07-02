# ZPPO Auto Recovery Runner Script
# This script executes train_zppo.py in a loop. If the process crashes (e.g. due to CUDA OOM),
# it will wait for 5 seconds and automatically restart training.
# Since train_zppo.py is configured to load the latest saved checkpoint, it will seamlessly resume.

$scriptPath = "E:\math workspace\train_zppo.py"
$pythonPath = "E:\AI_Workspace\.venv\Scripts\python.exe"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Starting ZPPO Training with Auto-OOM Recovery Loop" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

while ($true) {
    Write-Host "Launching ZPPO training process..." -ForegroundColor Cyan
    
    # Run the python script in unbuffered mode for real-time logging
    & $pythonPath -u $scriptPath
    
    $exitCode = $LASTEXITCODE
    Write-Host "Process exited with code: $exitCode" -ForegroundColor Yellow
    
    if ($exitCode -eq 0) {
        Write-Host "Training completed successfully. Exiting recovery loop." -ForegroundColor Green
        break
    } else {
        Write-Host "Training process crashed or encountered OOM. Cleaning up GPU cache and restarting in 5 seconds..." -ForegroundColor Red
        # Clear garbage collection and wait
        [System.GC]::Collect()
        Start-Sleep -Seconds 5
    }
}
