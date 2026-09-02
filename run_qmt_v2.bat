@echo off
setlocal
cd /d "%~dp0"
python check_qmt_env.py --reference-dir data\reference
if errorlevel 1 (
  echo.
  echo QMT environment or local market data check failed.
  pause
  exit /b 1
)
python run_backtest.py --start 20180101 --end 20251231 --reference-dir data\reference --strict-reference --output output\v2_baseline
pause
