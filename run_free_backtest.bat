@echo off
setlocal
cd /d "%~dp0"
set QMT_QUANT_CACHE_ONLY=1
python run_backtest.py --start 20180101 --end 20251231 --reference-dir data\reference --bar-cache-dir data\qmt_bars --strict-reference --output output\free_baseline %*
exit /b %errorlevel%
