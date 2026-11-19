<#
.SYNOPSIS
    Starts the MBCT server and Ollama backend daemon.
#>

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting MBCT (Model-Based Contrastive Traits)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check/Start Ollama
$ollamaProc = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($ollamaProc) {
    Write-Host "[OK] Ollama is already running (PID: $($ollamaProc[0].Id))." -ForegroundColor Green
} else {
    Write-Host "[*] Launching Ollama daemon in background..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

# 2. Check/Start FastAPI Server
Set-Location $PSScriptRoot
Write-Host "[*] Starting MBCT FastAPI server on http://127.0.0.1:8088..." -ForegroundColor Green
Write-Host "[*] Press Ctrl+C at any time to stop the server." -ForegroundColor DarkGray
Write-Host ""

python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8088
