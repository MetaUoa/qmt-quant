from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a strict QMT Quant backtest output directory")
    parser.add_argument("output_dir")
    parser.add_argument("--min-symbol-coverage", type=float, default=0.95)
    return parser.parse_args()


def validate_output(output_dir: str | Path, min_symbol_coverage: float = 0.95) -> dict:
    root = Path(output_dir)
    required = ["metrics.json", "data_quality.json", "equity.csv", "trades.csv", "universe_coverage.csv"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise AssertionError(f"Missing backtest artifacts: {', '.join(missing)}")

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    quality = json.loads((root / "data_quality.json").read_text(encoding="utf-8"))
    equity = pd.read_csv(root / "equity.csv")
    trades = pd.read_csv(root / "trades.csv")
    coverage = pd.read_csv(root / "universe_coverage.csv")

    assert metrics.get("point_in_time_universe") is True, "PIT historical universe is not enabled"
    assert metrics.get("strict_reference") is True, "strict_reference is not enabled"
    assert metrics.get("raw_limit_reference") is True, "unadjusted raw limit reference is not enabled"
    assert int(metrics.get("missing_st_dates", -1)) == 0, "historical ST snapshots are incomplete"
    assert int(metrics.get("missing_limit_dates", -1)) == 0, "daily price-limit snapshots are incomplete"
    assert int(metrics.get("missing_limit_rows", -1)) == 0, "per-symbol price-limit rows are incomplete"

    ratio = float(metrics.get("symbol_coverage_ratio", quality.get("symbol_coverage_ratio", 0.0)))
    assert ratio >= float(min_symbol_coverage), f"symbol coverage {ratio:.2%} is below {min_symbol_coverage:.2%}"
    raw_ratio = quality.get("raw_limit_reference_coverage_ratio")
    if raw_ratio is not None:
        assert float(raw_ratio) >= float(min_symbol_coverage), (
            f"raw limit-reference coverage {float(raw_ratio):.2%} is below {min_symbol_coverage:.2%}"
        )

    assert not equity.empty, "equity.csv is empty"
    assert "equity" in equity.columns, "equity.csv is missing equity column"
    eq = pd.to_numeric(equity["equity"], errors="coerce")
    assert eq.notna().all(), "equity contains NaN/non-numeric values"
    assert (eq > 0).all(), "equity contains non-positive values"

    if not trades.empty:
        for col in ("date", "signal_date", "shares", "side"):
            assert col in trades.columns, f"trades.csv is missing {col}"
        trade_date = pd.to_datetime(trades["date"], errors="coerce")
        signal_date = pd.to_datetime(trades["signal_date"], errors="coerce")
        assert trade_date.notna().all() and signal_date.notna().all(), "trade dates contain invalid values"
        assert (signal_date < trade_date).all(), "found a trade whose signal is not earlier than execution"
        shares = pd.to_numeric(trades["shares"], errors="coerce")
        assert shares.notna().all() and (shares > 0).all(), "trade shares must be positive numeric values"
        assert (shares % 100 == 0).all(), "A-share trade quantity is not a 100-share board-lot multiple"

    if not coverage.empty and "loaded" in coverage.columns:
        loaded = coverage["loaded"].astype(str).str.lower().isin(["true", "1"])
        observed_ratio = float(loaded.mean()) if len(loaded) else 0.0
        assert observed_ratio >= float(min_symbol_coverage), (
            f"coverage CSV ratio {observed_ratio:.2%} is below {min_symbol_coverage:.2%}"
        )

    return {
        "passed": True,
        "symbol_coverage_ratio": ratio,
        "trade_count": int(len(trades)),
        "ending_equity": float(eq.iloc[-1]),
        "multiple": metrics.get("multiple"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
    }


def main() -> int:
    args = parse_args()
    result = validate_output(args.output_dir, args.min_symbol_coverage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
