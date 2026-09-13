@echo off
set "PROJECT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%setup_voice.ps1"
if errorlevel 1 (
    echo.
    echo Voice setup failed. See the error above.
    pause
    exit /b 1
)
pause

