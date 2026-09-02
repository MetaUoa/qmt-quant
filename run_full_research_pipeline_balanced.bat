@echo off
setlocal
cd /d "%~dp0"
python run_full_research_pipeline.py --prepare-reference --download --profile balanced --require-grade C
exit /b %errorlevel%
