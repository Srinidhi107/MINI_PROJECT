@echo off
if "%~1"=="" (
    .\.venv\Scripts\python.exe scripts\predict.py
) else (
    .\.venv\Scripts\python.exe scripts\predict.py --code %1
)
pause
