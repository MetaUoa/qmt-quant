@echo off
setlocal
cd /d "%~dp0"
python run_acceptance.py --backtest output\v3_research\best_full_result\metrics.json --walk-forward output\walk_forward\walk_forward_metrics.json --folds output\walk_forward\walk_forward_folds.csv --stress output\v4_5_stress\stress_summary.json --require-grade A
exit /b %errorlevel%
