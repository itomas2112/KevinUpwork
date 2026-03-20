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
    echo [SKIP] Could not check for updates. Continuing...
)
echo.

:: Check virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call .venv\Scripts\activate.bat
echo [OK] Virtual environment activated.
echo.

:: Install any new dependencies (fast if nothing changed)
pip install -r requirements.txt --quiet 2>nul
echo.

:: Show CPU info
python -c "import os; print(f'CPU cores available: {os.cpu_count()}')"
echo.

:: Launch Streamlit
echo Launching Streamlit...
echo.
streamlit run app.py

:: If we get here, streamlit exited
echo.
echo ============================================
echo   App has stopped. See errors above.
echo ============================================
pause
