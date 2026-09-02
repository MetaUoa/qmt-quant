# V3.8 Free Data Edition

`qmt-quant` can run the complete 2018-2025 research pipeline without a Tushare token and without a local QMT history database.

## Data sources

### BaoStock — primary free source

The downloader uses BaoStock for:

- SSE/SZSE A-share security basics, including IPO/out dates;
- exchange trading calendar;
- daily OHLCV / amount / pre-close;
- front-adjusted signal bars;
- unadjusted execution/price-limit reference bars;
- historical `isST`;
- historical `tradestatus` suspension state.

BaoStock codes are converted to the project's canonical `000001.SZ` / `600000.SH` format.

### AKShare — independent cross-check

AKShare is optional and is not used to select strategy parameters. It samples BaoStock unadjusted close data against `stock_zh_a_hist(adjust="")` and writes a cross-check report.

### QMT — optional final verification

The original QMT loader remains supported. Free-data mode sets `QMT_QUANT_CACHE_ONLY=1`; in that mode missing cache files are reported through coverage gates and the code never silently falls back to `xtquant`.

## Install

```bat
pip install -r requirements.txt
```

No token is required for BaoStock or AKShare.

## One-command free research

Baseline acceptance target:

```bat
run_free_research_pipeline.bat
```

Strict 150x Grade-A gate:

```bat
run_free_research_pipeline_150x.bat
```

Equivalent Python command:

```bat
python run_full_research_pipeline.py ^
  --data-source baostock ^
  --prepare-reference ^
  --start 20180101 ^
  --end 20251231 ^
  --profile quick ^
  --require-grade C
```

The first run downloads the historical database and writes it into the same Parquet layout consumed by the existing QMT research stack.

After the first successful download, parameter search, Walk-forward and stress-test phases use local Parquet files only.

## Resume and refresh

The downloader is resumable. Existing per-symbol cache files are reused.

Force a full refresh:

```bat
python prepare_free_data.py --refresh
```

Small smoke/dev download:

```bat
python prepare_free_data.py --max-stocks 50
```

Do **not** use `--max-stocks` for final acceptance.

## AKShare cross-check

```bat
python prepare_free_data.py --verify-akshare --verify-sample 20
```

Output: `data/reference/akshare_crosscheck.csv`.

The cross-check compares unadjusted closes so differences in adjustment-factor conventions do not create false mismatches.

## Historical price-limit model

BaoStock exposes raw `preclose`, historical ST status and suspension state, but not a ready-made daily up/down-limit table. V3.8 derives executable price-limit references from unadjusted pre-close and the historical board regime used by this strategy:

- SSE/SZSE main board: 10%;
- main-board ST: 5%;
- STAR Market: 20%;
- ChiNext: 10% before 2020-08-24, 20% from 2020-08-24.

Prices are rounded to CNY 0.01 using half-up rounding.

The strategy already requires a minimum listing age of 120 trading sessions, so IPO no-limit opening days are outside the eligible buy universe. Final production candidates should still be replayed against QMT/broker data before live deployment.

## Safety and reproducibility

Free-data mode is fail-closed:

```text
BaoStock download
  -> canonical Parquet cache
  -> PIT/ST/suspension/limit references
  -> data audit
  -> strict baseline
  -> V3 parameter research
  -> V4 Walk-forward
  -> V4.5 stress tests
  -> V5 acceptance
```

If a required symbol is absent, `QMT_QUANT_CACHE_ONLY=1` prevents `xtquant` fallback. Coverage checks decide whether the run is acceptable.

A 150x result is never hard-coded. Grade A still requires the existing return, drawdown, Sharpe, OOS and stress-test gates.
