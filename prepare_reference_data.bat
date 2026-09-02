@echo off
setlocal
cd /d "%~dp0"
if "%TUSHARE_TOKEN%"=="" (
  echo TUSHARE_TOKEN is not set.
  echo Example for this terminal only:
  echo   set TUSHARE_TOKEN=your_token_here
  exit /b 1
)
python prepare_reference_data.py --start 20180101 --end 20251231 --output data\reference
