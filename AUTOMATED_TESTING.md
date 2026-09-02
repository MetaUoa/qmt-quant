# Automated testing

## Offline gate

```bat
run_automated_tests.bat
```

Runs:

1. `compileall`
2. `pytest`
3. JUnit output
4. coverage JSON
5. minimum core coverage gate of 60%

The suite covers causal signal/execution ordering, future-data mutation, strict PIT references, ST/limit/suspension execution behavior, cash/lot constraints, metrics/reporting, QMT cache behavior, Walk-forward scoring, V3 research scoring, V4.5 stress testing, Monte Carlo, V5 acceptance grades, live signal generation, live order planning and pre-trade risk.

Current container verification for this package: **31 passed, 1 skipped**, core package coverage **65.76%**. The skipped test is the real Parquet cache roundtrip when `pyarrow` is unavailable in the test interpreter; `requirements.txt` includes `pyarrow`.

## Real QMT smoke

```bat
run_automated_tests_full.bat
```

Adds:

- `xtquant` import/environment validation;
- a small real 2018-2025 strict QMT history backtest;
- output artifact validation;
- PIT/raw-limit/coverage/lot/causality checks.

This step must be run in the user's QMT/MiniQMT Python environment because the historical local market database is not bundled with the project.

## Full research acceptance

```bat
run_full_research_pipeline.bat
```

This is heavier than the smoke suite. It prepares references, audits historical data, runs the strict full-market baseline, V3 research, V4 Walk-forward, V4.5 stress tests and V5 acceptance grading.
