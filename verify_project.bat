@echo off
setlocal
cd /d "%~dp0"
python run_automated_tests.py
exit /b %errorlevel%
