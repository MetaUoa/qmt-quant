@echo off
setlocal
cd /d "%~dp0"
set QMT_QUANT_CACHE_ONLY=1
python run_full_research_pipeline.py --data-source baostock --prepare-reference --profile balanced --require-grade A %*
exit /b %errorlevel%
