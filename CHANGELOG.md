# Changelog

## Unreleased - V5-C pre-2026 research hardening

### Research/data lineage

- Froze the authoritative pre-2026 V5-C lineage in `research_lineage/v5_c_pre2026.json`.
- Preserved exactly 20 deterministic historical/PIT exposure shards and `max-parallel: 5`.
- Kept BaoStock pinned to 0.9.3 with the existing reconnect/self-healing and bounded socket-timeout rebinding contract.
- Archived frozen 2017-2025 data, recovered shard13, authoritative PIT exposures, pinned industry recovery, and C1/C7 reports to GitHub Release `v5-c-pre2026-frozen-lineage-v1` without reacquiring data.
- C1 and C7 nested Basic Alpha Gates remain failed; 2026 holdout performance therefore remains blinded.

### Research integrity

- Kept V5-C stock-selection-only with always-on risk state and purged nested boundaries.
- Preserved fail-closed PIT universe, ST, suspension, price-limit and raw-reference guards.
- Preserved 0.98 symbol and 0.97 session coverage thresholds.
- Added logical market-data audits for adjusted OHLC, raw reference prices, volume/amount and daily price-limit reference consistency.
- Made A-share T+1 explicit in the backtest execution model and stopped treating unknown strict-mode suspension state as automatically tradable.
- Documented that daily bars do not model intraday price-limit touch ordering.

### Research-to-production safety

- Removed the implicit V3 live-target configuration default and missing-config fallback.
- Added a strict production-candidate schema and exact SHA256 verification; V5 candidates fail closed until a matching production scoring adapter exists.
- Live targets carry `signal_date`, strategy source and strategy SHA256.
- Live execution rejects stale targets, requires acceptance bound to the same strategy SHA256, retries MiniQMT connection with fresh trader instances, journals each order durably and reconciles submitted/partial orders.
- Uncertain order submission stops the remaining batch and requires manual reconciliation; no automatic cancellation was introduced.
- Acceptance evidence is being migrated away from implicit V3/walk-forward/stress paths to explicit, SHA-bound inputs.

### Reproducibility/tooling

- Replaced floating dependency ranges with a single exact-pin Python 3.10/3.11/3.12-compatible requirements set.
- Kept `baostock==0.9.3` in the repository dependency contract.
- All Python CI jobs install the same requirements file and run `pip check`.
- Removed the stale fixed test-count claim from the release narrative; current GitHub Actions is authoritative.

## V3.7 - legacy full research-to-execution suite

Historical milestone retained for context. Its V3/V4/V5/V6/V7 labels are no longer the authoritative production-state narrative.

- Added V2.2 session-level historical data audit for adjusted and unadjusted QMT bars.
- Added market-breadth regime filter and configurable momentum/volatility factor weights.
- Added V3 multi-objective parameter research with annual-instability, turnover-concentration and neighborhood-stability penalties.
- Kept 2023-2025 holdout metrics out of parameter selection.
- Upgraded Walk-forward selection to risk/stability-aware scoring.
- Added V4.5 true-path stress tests: cost shocks, T+2 execution, probabilistic fills, parameter perturbation and universe deletion.
- Added bootstrap Monte Carlo diagnostics.
- Added V5 A/B/C/REJECT acceptance grading.
- Added guarded MiniQMT execution adapter and pre-trade checks.

## V2.1 - automated verification

- Added local and GitHub Actions automated test runners with JUnit and coverage reports.
- Added an optional real QMT 2018-2025 end-to-end smoke backtest plus artifact validator.
- Added a causality regression test: mutating future prices must not change past equity or trades.
- Fixed strict handling of empty ST/limit tables.
- Strict mode requires PIT reference data and unadjusted QMT limit-reference bars.

## V2

- Replaced current-sector historical backtest universe with point-in-time SSE/SZSE membership.
- Added historical ST filtering, suspension checks and daily price-limit execution guards.
- Added adjusted/unadjusted price-scale separation and Parquet market-data mirror.
- Added walk-forward validation.
