@echo off
setlocal
cd /d "%~dp0"
echo ERROR: run_acceptance.bat no longer supplies implicit V3/V4 evidence paths.
echo Use run_acceptance.py with explicit --backtest --walk-forward --folds --stress --strategy-sha256 arguments.
exit /b 2
