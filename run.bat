@echo off
title MBCT Server Launcher
echo ========================================================
echo   Starting MBCT (Model-Based Contrastive Traits)
echo ========================================================
echo.

REM 1. Start Ollama in background if not already running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] Ollama is already running.
) else (
    echo [*] Starting Ollama serve in background...
    start "" ollama serve
    timeout /t 3 /nobreak >nul
)

REM 2. Start MBCT FastAPI Backend
echo [*] Starting MBCT FastAPI Backend on http://127.0.0.1:8088...
cd /d "%~dp0"
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8088
pause
