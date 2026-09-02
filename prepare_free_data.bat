@echo off
setlocal
cd /d "%~dp0"
python prepare_free_data.py %*
exit /b %errorlevel%
