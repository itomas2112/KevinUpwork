@echo off
title Trading Analysis Platform
echo ============================================
echo   Trading Analysis Platform
echo ============================================
echo.

:: Navigate to project directory (in case launched from shortcut)
cd /d "%~dp0"

:: Pull latest changes from git
echo Checking for updates...
git pull 2>nul
if %errorlevel% equ 0 (
    echo [OK] Up to date.
) else (
    echo [SKIP] Could not check for updates (no internet or git issue). Continuing...
)
echo.

:: Activate virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

:: Install any new dependencies (fast if nothing changed)
pip install -r requirements.txt --quiet 2>nul
echo.

:: Show CPU info
echo Starting app...
python -c "import os; print(f'CPU cores available: {os.cpu_count()}')"
echo.

:: Launch Streamlit
echo Launching Streamlit...
echo (If the app crashes, the error will be shown above)
echo.
streamlit run app.py
echo.
echo ============================================
echo   App has stopped.
echo ============================================
pause
