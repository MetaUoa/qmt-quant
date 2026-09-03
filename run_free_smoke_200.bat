@echo off
setlocal
cd /d "%~dp0"
python run_free_v4_pipeline.py --stage smoke --smoke-stocks 200 %*
exit /b %errorlevel%
