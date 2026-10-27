@echo off
rem Launch the AModel Streamlit UI.
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run src\ui\app.py
