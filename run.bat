@echo off
REM One-command launcher for the Nagpur cross-modal debris dashboard (Windows).
REM
REM   run.bat            set up a virtualenv, install deps, launch the dashboard
REM   run.bat demo       run the batch benchmark instead of the dashboard
REM   run.bat test       run the test suite
REM
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: python not found. Install Python 3.9+ from https://python.org
  echo Make sure to tick "Add Python to PATH" during installation, then retry.
  exit /b 1
)

if not exist ".venv" (
  echo Creating virtual environment ^(.venv^) ...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

echo Installing dependencies ^(first run only, ~1-2 min^) ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

set MODE=%1
if "%MODE%"=="demo" (
  python run_demo.py
) else if "%MODE%"=="test" (
  python tests\test_pipeline.py
) else (
  echo.
  echo ======================================================================
  echo  Launching the dashboard. Open the "Local URL" it prints below
  echo  ^(usually http://localhost:8501^) in your browser.
  echo  Press Ctrl+C here to stop.
  echo ======================================================================
  echo.
  streamlit run app.py
)
endlocal
