from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from qmt_quant.config import StrategyConfig
from qmt_quant.qmt_data import download_daily_history, load_daily_bars, load_limit_reference_bars
from qmt_quant.reference_data import ReferenceData
from qmt_quant.signals import latest_target_codes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate current QMT target weights without placing orders")
    p.add_argument("--as-of", default=date.today().strftime("%Y%m%d"))
    p.add_argument("--lookback-days", type=int, default=550)
    p.add_argument("--benchmark", default="000905.SH")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--bar-cache-dir", default="data/qmt_bars_live")
    p.add_argument("--strategy-config", default="output/v3_research/best_config.json")
    p.add_argument("--output", default="output/live_targets")
    p.add_argument("--download", action="store_true")
    return p.parse_args()


def load_strategy(path: str) -> StrategyConfig:
    p = Path(path)
    if not p.exists():
        return StrategyConfig()
    payload = json.loads(p.read_text(encoding="utf-8"))
    allowed = StrategyConfig.__dataclass_fields__.keys()
    return StrategyConfig(**{k: v for k, v in payload.items() if k in allowed})


def main() -> int:
    args = parse_args()
    asof = pd.Timestamp(args.as_of)
    start = (asof - timedelta(days=args.lookback_days)).strftime("%Y%m%d")
    end = asof.strftime("%Y%m%d")
    ref = ReferenceData.from_dir(args.reference_dir)
    strategy = load_strategy(args.strategy_config)
    universe = ref.codes_ever_active(end, end)
    codes = list(dict.fromkeys(universe + [args.benchmark]))
    if args.download:
        download_daily_history(codes, start, end)
    bars = load_daily_bars(
        codes,
        start,
        end,
        dividend_type="front",
        cache_dir=Path(args.bar_cache_dir) / f"front_{start}_{end}",
    )
    raw = load_limit_reference_bars(
        universe,
        start,
        end,
        cache_dir=Path(args.bar_cache_dir) / f"none_{start}_{end}",
    )
    signal_ts, selected, diagnostics = latest_target_codes(
        bars,
        args.benchmark,
        strategy,
        reference=ref,
        signal_date=asof,
        raw_bars=raw,
        strict_st=True,
    )
    weight = 1.0 / len(selected) if selected else 0.0
    frame = pd.DataFrame({"code": selected, "target_weight": [weight] * len(selected)})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "target_weights.csv", index=False, encoding="utf-8-sig")
    diagnostics["strategy_config"] = args.strategy_config
    diagnostics["target_weight_sum"] = float(frame["target_weight"].sum()) if len(frame) else 0.0
    (out / "signal_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(frame.to_string(index=False) if len(frame) else "No equity targets: risk-off/candidate-empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
