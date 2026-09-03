@echo off
setlocal
cd /d "%~dp0"
python run_free_v4_pipeline.py --stage full-data %*
exit /b %errorlevel%
