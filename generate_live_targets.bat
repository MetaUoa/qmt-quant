@echo off
setlocal
cd /d "%~dp0"
python generate_live_targets.py --download --reference-dir data\reference --strategy-config output\v3_research\best_config.json --output output\live_targets
exit /b %errorlevel%
