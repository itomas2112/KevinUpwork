@echo off
title Trading App - First Time Setup
echo ============================================
echo   Trading Analysis Platform - Installation
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Download Python from: https://www.python.org/downloads/
    echo IMPORTANT: Check "Add Python to PATH" during install!
    echo.
    pause
    exit /b 1
)
echo [OK] Python found:
python --version
echo.

:: Check Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH.
    echo.
    echo Download Git from: https://git-scm.com/download/win
    echo Use default settings during install.
    echo.
    pause
    exit /b 1
)
echo [OK] Git found:
git --version
echo.

:: Create virtual environment
echo Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

:: Install dependencies
echo Installing dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet
echo [OK] Dependencies installed.
echo.

:: Create desktop shortcut
echo Creating desktop shortcut...
set SCRIPT_DIR=%~dp0
set SHORTCUT_NAME=Trading App.bat

:: Create a launcher bat on the Desktop
echo @echo off > "%USERPROFILE%\Desktop\%SHORTCUT_NAME%"
echo cd /d "%SCRIPT_DIR%" >> "%USERPROFILE%\Desktop\%SHORTCUT_NAME%"
echo call start_app.bat >> "%USERPROFILE%\Desktop\%SHORTCUT_NAME%"

echo [OK] Desktop shortcut created: "%SHORTCUT_NAME%"
echo.

echo ============================================
echo   Installation complete!
echo.
echo   Your client can now double-click
echo   "Trading App" on the Desktop to launch.
echo ============================================
echo.
pause
