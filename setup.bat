@echo off
rem One-time setup: create the virtualenv and install dependencies.
cd /d "%~dp0"

if not exist .venv (
    echo Creating .venv ...
    python -m venv .venv
)

echo Installing dependencies ...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo Done. Launch the app with run.bat
