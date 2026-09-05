from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from qmt_quant.production_candidate import (
    load_legacy_strategy_config,
    load_production_candidate_bundle,
    strategy_source_manifest,
)
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
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--strategy-config", help="Explicit legacy StrategyConfig JSON; never defaults to V3")
    source.add_argument("--candidate-manifest", help="Holdout-passed V5 production candidate bundle")
    p.add_argument("--output", default="output/live_targets")
    p.add_argument("--download", action="store_true")
    return p.parse_args()


def _load_strategy_source(args: argparse.Namespace):
    if args.candidate_manifest:
        source = load_production_candidate_bundle(args.candidate_manifest)
        # A FrozenCandidate contains neutralization/factor weights that cannot be mapped to
        # StrategyConfig. Never substitute the legacy V3 scorer for a verified V5 candidate.
        raise RuntimeError(
            "Verified V5 production candidate has no production scoring adapter; "
            "refusing to generate legacy/V3 targets under a V5 identity"
        )
    return load_legacy_strategy_config(args.strategy_config)


def main() -> int:
    args = parse_args()
    asof = pd.Timestamp(args.as_of).normalize()
    start = (asof - timedelta(days=args.lookback_days)).strftime("%Y%m%d")
    end = asof.strftime("%Y%m%d")
    ref = ReferenceData.from_dir(args.reference_dir)
    source = _load_strategy_source(args)
    if source.strategy is None:
        raise RuntimeError("selected strategy source does not provide a legacy StrategyConfig")
    strategy = source.strategy
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
    frame = pd.DataFrame(
        {
            "signal_date": [str(pd.Timestamp(signal_ts).date())] * len(selected),
            "strategy_source": [source.kind] * len(selected),
            "strategy_sha256": [source.sha256] * len(selected),
            "code": selected,
            "target_weight": [weight] * len(selected),
        },
        columns=["signal_date", "strategy_source", "strategy_sha256", "code", "target_weight"],
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "target_weights.csv", index=False, encoding="utf-8-sig")
    diagnostics["requested_as_of"] = str(asof.date())
    diagnostics["strategy_source"] = strategy_source_manifest(source)
    diagnostics["target_weight_sum"] = float(frame["target_weight"].sum()) if len(frame) else 0.0
    (out / "signal_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(frame.to_string(index=False) if len(frame) else "No equity targets: risk-off/candidate-empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
