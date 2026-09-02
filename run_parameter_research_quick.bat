@echo off
setlocal
cd /d "%~dp0"
python run_parameter_research.py --start 20180101 --end 20251231 --reference-dir data\reference --profile quick --strict-reference --output output\v3_research
exit /b %errorlevel%
