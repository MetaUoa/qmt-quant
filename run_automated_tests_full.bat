@echo off
setlocal
cd /d "%~dp0"
python run_automated_tests.py --qmt-smoke --reference-dir data\reference
exit /b %errorlevel%
