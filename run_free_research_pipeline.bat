@echo off
setlocal
cd /d "%~dp0"
set QMT_QUANT_CACHE_ONLY=1
python run_full_research_pipeline.py --data-source baostock --prepare-reference --profile quick --require-grade C %*
exit /b %errorlevel%
