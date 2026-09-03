# V4.0 Free Data Validation & Baseline

`qmt-quant` can build and validate a 2018-2025 SSE/SZSE A-share research database without a Tushare token and without a local QMT history database.

## Data sources

- **BaoStock**: primary free source for security basics, IPO/out dates, trading calendar, daily OHLCV/amount/pre-close, adjusted/raw bars, historical `isST` and `tradestatus`.
- **AKShare**: independent optional cross-check of BaoStock unadjusted closes. It is never used to select strategy parameters.
- **QMT**: optional final verification source. Free-data pipelines set `QMT_QUANT_CACHE_ONLY=1`, so missing free caches never silently fall back to `xtquant`.

## Install

```bat
pip install -r requirements.txt
```

No BaoStock/AKShare token is required.

## Recommended staged workflow

### V3.8.1 — 200-stock real-data smoke

```bat
run_free_smoke_200.bat
```

Smoke data is isolated under `data/smoke/` and cannot overwrite the full research database. The stage downloads real BaoStock data, optionally cross-checks AKShare, runs the full data audit, executes a strict PIT backtest and validates the backtest output.

### V3.9 — full 2018-2025 A-share database

```bat
run_free_full_data.bat
```

The downloader is resumable. Existing per-symbol Parquet files are reused unless `--refresh` is supplied. Formal gates default to:

- symbol coverage >= 98%;
- raw symbol coverage >= 98%;
- session coverage >= 97%;
- raw session coverage >= 97%;
- benchmark present;
- historical PIT/ST/suspension/limit references present.

### V4.0 — first strict baseline

```bat
run_free_baseline.bat
```

This stage first re-validates the full database, then runs the existing strategy without parameter optimization. Outputs include the regular `metrics.json`, `yearly_returns.csv`, trades/equity files and a new `baseline_summary.json` with CAGR, multiple, drawdown, Sharpe, Calmar, positive/negative years and whether 150x was actually reached.

### One command for all three stages

```bat
run_free_v4_pipeline.bat
```

Equivalent command:

```bat
python run_free_v4_pipeline.py --stage all
```

Use `--refresh` only when you intentionally want to re-download existing cached symbols.

## AKShare policy

AKShare comparison is enabled by default in the staged pipeline. Because a second public endpoint can temporarily be unavailable, it is diagnostic by default. To make it a hard gate:

```bat
run_free_v4_pipeline.bat --require-akshare
```

Defaults for a hard gate are at least 5 comparable symbols and >= 80% pass ratio. Change them with `--min-akshare-compared` and `--min-akshare-pass-ratio`.

## Machine-readable validation

Every preparation writes `free_data_manifest.json`. `validate_free_data.py` converts it into `free_data_validation.json` and checks adjusted/raw cache coverage. `run_data_audit.py` independently checks per-symbol and per-session historical coverage, so the manifest cannot certify itself.

The orchestration result is written to:

```text
output/free_v4_pipeline/pipeline_summary.json
```

Any required gate failure returns a non-zero process status.

## GitHub live integration smoke

`.github/workflows/free-data-smoke.yml` is a manual live test. From GitHub Actions choose **free-data-live-smoke**, select 20/50/200 symbols and run it. It downloads real BaoStock data on a clean Windows runner and uploads only reports, not the market-data cache.

The ordinary `automated-tests` workflow remains offline and deterministic; a third-party outage therefore cannot make every code push red.

## Historical price-limit model

BaoStock provides raw `preclose`, historical ST status and suspension state but not a ready-made daily up/down-limit table. The project derives limit references from unadjusted pre-close and historical board rules used by the strategy:

- SSE/SZSE main board: 10%;
- main-board ST: 5%;
- STAR Market: 20%;
- ChiNext: 10% before 2020-08-24 and 20% from 2020-08-24.

Prices use CNY 0.01 half-up rounding. The strategy requires a minimum listing age of 120 trading sessions, so IPO no-limit opening days are outside its eligible buy universe.

## What V4.0 does not claim

A 150x result is never hard-coded. V4.0 exists to obtain the first credible baseline from real historical data. Only after that baseline should parameter research, Walk-forward and stress optimization decide whether the strategy is genuinely approaching the 150x Grade-A target.
