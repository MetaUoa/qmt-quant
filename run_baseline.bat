@echo off
setlocal
cd /d "%~dp0"
python run_backtest.py --start 20180101 --end 20251231 --reference-dir data\reference --strict-reference --output output\v2_5_baseline
exit /b %errorlevel%
