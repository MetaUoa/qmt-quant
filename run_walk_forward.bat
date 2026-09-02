@echo off
setlocal
cd /d "%~dp0"
python run_walk_forward.py --start 20180101 --end 20251231 --reference-dir data\reference --strict-reference --output output\walk_forward
pause
