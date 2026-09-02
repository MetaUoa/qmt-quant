@echo off
setlocal
cd /d "%~dp0"
python run_stress_tests.py --start 20180101 --end 20251231 --reference-dir data\reference --strategy-config output\v3_research\best_config.json --strict-reference --output output\v4_5_stress
exit /b %errorlevel%
