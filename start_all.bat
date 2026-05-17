@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PY=D:\Anaconda3\python.exe"
set "HF_HOME=%ROOT%hf_cache"
set "HF_HUB_DISABLE_TELEMETRY=1"
if not exist "%PY%" (
  echo [ERROR] Python not found: %PY%
  exit /b 1
)

for %%P in (7860 8000) do (
  for /f "tokens=5" %%I in ('netstat -ano ^| findstr ":%%P" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%I >nul 2>nul
  )
)

timeout /t 1 >nul

start "SpatialVLA_Backend" /MIN "%PY%" api.py --host 127.0.0.1 --port 8000
start "SpatialVLA_Frontend" /MIN "%PY%" front.py --host 127.0.0.1 --port 7860

echo Started.
echo Frontend: http://127.0.0.1:7860
echo Backend : http://127.0.0.1:8000/v1
endlocal
