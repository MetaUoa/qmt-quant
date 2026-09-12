from __future__ import annotations

import argparse
import json

from qmt_quant.config import CostConfig
from qmt_quant.cost_manifest import write_cost_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build immutable raw-execution research cost manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.00025)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--limit-tolerance", type=float, default=0.001)
    parser.add_argument("--fill-probability", type=float, default=1.0)
    parser.add_argument("--fill-seed", type=int, default=20260902)
    parser.add_argument("--commission-excludes-regulatory-fees", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cost = CostConfig(
        initial_cash=float(args.initial_cash),
        commission_rate=float(args.commission_rate),
        min_commission=float(args.min_commission),
        slippage_bps=float(args.slippage_bps),
        lot_size=int(args.lot_size),
        limit_tolerance=float(args.limit_tolerance),
        fill_probability=float(args.fill_probability),
        fill_seed=int(args.fill_seed),
    )
    payload = write_cost_manifest(
        args.output,
        cost,
        commission_includes_exchange_and_management=not bool(
            args.commission_excludes_regulatory_fees
        ),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
