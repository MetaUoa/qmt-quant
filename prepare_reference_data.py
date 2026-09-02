from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qmt_quant.backtest import rebalance_schedule
from qmt_quant.config import StrategyConfig
from qmt_quant.tushare_sync import (
    sync_limit_dates,
    sync_stock_basic,
    sync_stock_st_dates,
    sync_suspension_range,
    sync_trade_calendar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time A-share reference data with Tushare Pro")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--output", default="data/reference")
    parser.add_argument("--token", default=None, help="Prefer environment variable TUSHARE_TOKEN instead of command line")
    parser.add_argument("--rebalance-grid", default="3,5,10", help="Comma-separated rebalance intervals whose reference dates must be cached")
    parser.add_argument("--execution-delay-grid", default="1,2", help="Comma-separated signal-to-execution delays used by research/stress tests")
    parser.add_argument("--min-listing-sessions", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--skip-st", action="store_true")
    parser.add_argument("--skip-limits", action="store_true")
    parser.add_argument("--skip-suspensions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Syncing SSE/SZSE stock_basic (listed/delisted/suspended-listing states)...")
    basic = sync_stock_basic(out, args.token)
    print(f"  stock_basic rows: {len(basic)}")

    print("[2/5] Syncing SSE trade calendar...")
    cal = sync_trade_calendar(out, args.start, args.end, args.token)
    open_days = pd.to_datetime(
        cal.loc[pd.to_numeric(cal["is_open"], errors="coerce").fillna(0).astype(int).eq(1), "cal_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    ).dropna()
    calendar = pd.DatetimeIndex(open_days).sort_values()
    rebalance_grid = sorted({int(x.strip()) for x in args.rebalance_grid.split(",") if x.strip()})
    if not rebalance_grid:
        raise ValueError("--rebalance-grid must contain at least one positive integer")
    delay_grid = sorted({int(x.strip()) for x in args.execution_delay_grid.split(",") if x.strip()})
    if not delay_grid or any(x <= 0 for x in delay_grid):
        raise ValueError("--execution-delay-grid must contain positive integers")
    schedules = []
    for rebalance_days in rebalance_grid:
        if rebalance_days <= 0:
            raise ValueError("rebalance intervals must be positive")
        for execution_delay in delay_grid:
            cfg = StrategyConfig(
                rebalance_days=rebalance_days,
                execution_delay_sessions=execution_delay,
                min_listing_sessions=args.min_listing_sessions,
            )
            schedules.extend(rebalance_schedule(calendar, cfg))
    signal_dates = sorted({signal for signal, _ in schedules})
    execution_dates = sorted({execution for _, execution in schedules})
    print(
        f"  open sessions: {len(calendar)}, rebalance grid: {rebalance_grid}, delay grid: {delay_grid}, "
        f"unique signal dates: {len(signal_dates)}, unique execution dates: {len(execution_dates)}"
    )

    if args.skip_st:
        print("[3/5] ST sync skipped by request. Strict V2 backtest will not run without these snapshots.")
    else:
        print("[3/5] Syncing historical ST snapshots for signal dates (Tushare stock_st)...")
        st = sync_stock_st_dates(out, signal_dates, args.token, sleep_seconds=args.sleep)
        print(f"  ST rows cached: {len(st)}")

    if args.skip_limits:
        print("[4/5] Daily price-limit sync skipped by request. Backtest will use one-price-board fallback unless strict mode is enabled.")
    else:
        print("[4/5] Syncing daily price-limit snapshots for execution dates (Tushare stk_limit)...")
        limits = sync_limit_dates(out, execution_dates, args.token, sleep_seconds=args.sleep)
        print(f"  price-limit rows cached: {len(limits)}")

    if args.skip_suspensions:
        print("[5/5] Daily suspension sync skipped by request.")
    else:
        print("[5/5] Syncing daily suspension rows (Tushare suspend_d, month chunks)...")
        susp = sync_suspension_range(out, args.start, args.end, args.token, sleep_seconds=args.sleep)
        print(f"  suspension rows cached: {len(susp)}")

    print(f"Reference data ready: {out.resolve()}")


if __name__ == "__main__":
    main()
