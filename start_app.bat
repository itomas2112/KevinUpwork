@echo off
title Trading Analysis Platform
setlocal EnableDelayedExpansion

set LOG=startup_log.txt

:: Navigate to project directory (in case launched from shortcut)
cd /d "%~dp0"

echo ============================================
echo   Trading Analysis Platform
echo ============================================
echo.
echo Folder:   %CD%
echo Log file: %CD%\%LOG%
echo.

:: Start fresh log
> "%LOG%" echo ============================================
>> "%LOG%" echo   Startup log: %DATE% %TIME%
>> "%LOG%" echo   Folder: %CD%
>> "%LOG%" echo ============================================
>> "%LOG%" echo.

:: --- Git pull ---
echo ============================================
echo   Checking for updates
echo ============================================
echo.

:: Capture current commit (may be empty if not a git repo)
set BEFORE=unknown
for /f "delims=" %%h in ('git rev-parse --short HEAD 2^>nul') do set BEFORE=%%h
echo Current commit: !BEFORE!
echo Current commit: !BEFORE!>> "%LOG%"
echo.
echo.>> "%LOG%"

echo --- git status ---
echo --- git status --->> "%LOG%"
powershell -NoProfile -Command "git status 2>&1 | Tee-Object -FilePath '%LOG%' -Append"
echo.
echo.>> "%LOG%"

:: --- Protect local saved_strategies.json from being blocked by pull ---
:: If the user has locally modified saved_strategies.json (tracked file),
:: back it up and reset the tracked copy so git pull can proceed. The backup
:: is restored after pull, preserving his strategies.
set STRAT_BACKED_UP=0
if exist "saved_strategies.json" (
    echo Backing up local saved_strategies.json...
    echo Backing up local saved_strategies.json>> "%LOG%"
    copy /Y "saved_strategies.json" "saved_strategies.json.bak" >nul
    git ls-files --error-unmatch saved_strategies.json >nul 2>&1
    if !errorlevel! equ 0 (
        :: File is tracked by git - discard local modifications so pull works
        git checkout -- saved_strategies.json >> "%LOG%" 2>&1
        echo [OK] Local saved_strategies.json saved to .bak and reset for pull.
        echo [OK] saved_strategies.json backed up + reset>> "%LOG%"
    ) else (
        echo [OK] saved_strategies.json is untracked - backup kept just in case.
        echo [OK] saved_strategies.json untracked - backup kept>> "%LOG%"
    )
    set STRAT_BACKED_UP=1
    echo.
)

echo --- git pull ---
echo --- git pull --->> "%LOG%"
powershell -NoProfile -Command "$ErrorActionPreference='Continue'; git pull 2>&1 | Tee-Object -FilePath '%LOG%' -Append; exit $LASTEXITCODE"
set PULL_ERR=!errorlevel!
echo.
echo.>> "%LOG%"

:: --- Restore local saved_strategies.json ---
if !STRAT_BACKED_UP! equ 1 (
    if exist "saved_strategies.json.bak" (
        move /Y "saved_strategies.json.bak" "saved_strategies.json" >nul
        echo [OK] Restored local saved_strategies.json from backup.
        echo [OK] Restored saved_strategies.json from backup>> "%LOG%"
        echo.
    )
)

set AFTER=unknown
for /f "delims=" %%h in ('git rev-parse --short HEAD 2^>nul') do set AFTER=%%h
echo After pull:     !AFTER!
echo After pull: !AFTER!>> "%LOG%"
echo.

if !PULL_ERR! neq 0 (
    echo.
    echo ============================================
    echo   [ERROR] git pull FAILED  ^(exit code !PULL_ERR!^)
    echo ============================================
    echo.
    echo The real error message from git is shown ABOVE this banner.
    echo A full log was also saved to:
    echo   %CD%\%LOG%
    echo.
    echo The app was NOT launched. Send the log file or a
    echo screenshot of this window so the problem can be fixed.
    echo.
    pause
    exit /b 1
)

if "!BEFORE!"=="!AFTER!" (
    echo [OK] Already up to date at !AFTER!.
) else (
    echo [OK] Updated !BEFORE! -^> !AFTER!.
)
echo.

:: --- Virtual environment ---
if not exist ".venv\Scripts\activate.bat" (
    echo ============================================
    echo   [ERROR] Virtual environment not found
    echo ============================================
    echo Expected: %CD%\.venv\Scripts\activate.bat
    echo Run install.bat first.
    echo [ERROR] Virtual environment not found>> "%LOG%"
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo [OK] Virtual environment activated.
echo.

:: --- Install dependencies ---
echo ============================================
echo   Installing/updating dependencies
echo ============================================
echo --- pip install --->> "%LOG%"
powershell -NoProfile -Command "pip install -r requirements.txt 2>&1 | Tee-Object -FilePath '%LOG%' -Append; exit $LASTEXITCODE"
set PIP_ERR=!errorlevel!
echo.
if !PIP_ERR! neq 0 (
    echo [WARN] pip install reported errors ^(code !PIP_ERR!^).
    echo See %LOG% for details. Continuing anyway...
    echo.
)

:: Show CPU info
python -c "import os; print(f'CPU cores available: {os.cpu_count()}')"
echo.

:: --- Launch Streamlit ---
echo ============================================
echo   Launching Streamlit
echo ============================================
echo.
streamlit run app.py
set APP_ERR=!errorlevel!

:: If we get here, streamlit exited
echo.
echo ============================================
if !APP_ERR! neq 0 (
    echo   [ERROR] App exited with code !APP_ERR!
    echo   [ERROR] streamlit exit code !APP_ERR!>> "%LOG%"
) else (
    echo   App has stopped.
)
echo   See messages above or log: %CD%\%LOG%
echo ============================================
pause
