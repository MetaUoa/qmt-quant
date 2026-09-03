@echo off
setlocal
cd /d "%~dp0"
python run_free_v4_pipeline.py --stage baseline %*
exit /b %errorlevel%
