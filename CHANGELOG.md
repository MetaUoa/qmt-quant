# Changelog

## V3.7 - full research-to-execution suite

- Added V2.2 session-level historical data audit for adjusted and unadjusted QMT bars.
- Added market-breadth regime filter and configurable momentum/volatility factor weights.
- Added V3 multi-objective parameter research with annual-instability, turnover-concentration and neighborhood-stability penalties.
- Kept 2023-2025 holdout metrics out of parameter selection.
- Upgraded Walk-forward selection to risk/stability-aware scoring.
- Added V4.5 true-path stress tests: cost shocks, T+2 execution, probabilistic fills, parameter perturbation and universe deletion.
- Added bootstrap Monte Carlo diagnostics.
- Added V5 A/B/C/REJECT acceptance grading; A requires the 150x target plus OOS and risk gates.
- Added live target generation from the same causal signal engine.
- Added guarded MiniQMT execution adapter using account/position queries, full-tick prices and synchronous stock orders.
- Live mode is opt-in with two explicit flags and an acceptance-grade gate.
- Added pre-trade concentration/order-size/order-count risk checks.
- Added one-command V2.2 -> V5 research pipeline.
- Expanded offline automated tests to cover research, stress, acceptance, signals and live order planning.

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
