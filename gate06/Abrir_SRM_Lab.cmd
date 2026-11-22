@echo off
setlocal
set "PY=%SRM_LAB_PYTHON%"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "C:\Windows\System32\work\g5-cpu\Scripts\python.exe" set "PY=C:\Windows\System32\work\g5-cpu\Scripts\python.exe"
if not defined PY set "PY=python"
"%PY%" -B "%~dp0app.py" %*
if errorlevel 1 pause
