from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from prepare_free_data_shard import (
    _bind_baostock_socket_timeout,
    _configure_socket_timeout,
    _reconnect_baostock,
)
from qmt_quant.free_data import _result_frame, baostock_session, fetch_trade_calendar
from qmt_quant.pit_exposures import monthly_first_open_dates, normalize_industry_snapshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download monthly PIT BaoStock industry snapshots")
    p.add_argument("--start", default="20170101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--output", required=True)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--socket-timeout", type=float, default=45.0)
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    return p.parse_args()


def select_snapshot_shard(
    dates: pd.DatetimeIndex,
    shard_index: int,
    shard_count: int,
) -> pd.DatetimeIndex:
    """Return one deterministic modulo shard without changing date order."""
    count = int(shard_count)
    index = int(shard_index)
    if count <= 0:
        raise ValueError("shard_count must be positive")
    if index < 0 or index >= count:
        raise ValueError(f"shard_index must be in [0, {count})")
    return pd.DatetimeIndex(dates)[index::count]


def fetch_trade_calendar_with_retry(
    api,
    start,
    end,
    *,
    attempts: int = 4,
    sleep_seconds: float = 0.5,
    socket_timeout_seconds: float = 45.0,
) -> pd.DataFrame:
    """Fetch the calendar with bounded reconnect/rebind protection."""
    if int(attempts) <= 0:
        raise ValueError("attempts must be positive")
    last: Exception | None = None
    for attempt in range(int(attempts)):
        try:
            return fetch_trade_calendar(api, start, end)
        except Exception as exc:
            last = exc
            if attempt >= int(attempts) - 1:
                break
            _reconnect_baostock(
                api,
                socket_timeout_seconds=float(socket_timeout_seconds),
            )
            time.sleep(max(0.0, float(sleep_seconds)) * (2**attempt))
    raise RuntimeError(f"trade-calendar bootstrap failed after retries: {last}") from last


def _query_snapshot(
    api,
    date: pd.Timestamp,
    *,
    attempts: int,
    socket_timeout: float,
    sleep: float,
) -> pd.DataFrame:
    last: Exception | None = None
    for attempt in range(int(attempts)):
        try:
            result = api.query_stock_industry(date=date.strftime("%Y-%m-%d"))
            return _result_frame(result)
        except Exception as exc:
            last = exc
            if attempt >= attempts - 1:
                break
            _reconnect_baostock(api, socket_timeout_seconds=socket_timeout)
            time.sleep(max(0.5, sleep) * (2**attempt))
    raise RuntimeError(f"industry snapshot {date.date()} failed after retries: {last}") from last


def main() -> int:
    args = parse_args()
    _configure_socket_timeout(args.socket_timeout)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    with baostock_session() as bs:
        _bind_baostock_socket_timeout(args.socket_timeout)
        calendar = fetch_trade_calendar_with_retry(
            bs,
            args.start,
            args.end,
            attempts=args.attempts,
            sleep_seconds=max(args.sleep, 0.5),
            socket_timeout_seconds=args.socket_timeout,
        )
        all_dates = monthly_first_open_dates(calendar)
        dates = select_snapshot_shard(all_dates, args.shard_index, args.shard_count)
        for number, date in enumerate(dates, 1):
            try:
                raw = _query_snapshot(
                    bs,
                    date,
                    attempts=args.attempts,
                    socket_timeout=args.socket_timeout,
                    sleep=args.sleep,
                )
                frames.append(normalize_industry_snapshot(raw, asof_date=date))
            except Exception as exc:
                errors.append({"date": str(date.date()), "error": str(exc)})
            if number == 1 or number % 12 == 0 or number == len(dates):
                print(
                    f"[PIT industry shard {args.shard_index}/{args.shard_count}] "
                    f"{number}/{len(dates)} snapshots errors={len(errors)}"
                )
            if args.sleep > 0 and number < len(dates):
                time.sleep(args.sleep)

    snapshots = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    snapshots.to_parquet(root / "industry_snapshots.parquet", index=False)
    written = int(snapshots["asof_date"].nunique()) if not snapshots.empty else 0
    manifest = {
        "source": "baostock-query_stock_industry",
        "start": args.start,
        "end": args.end,
        "snapshot_frequency": "monthly_first_open_session",
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "snapshot_candidates_total": int(len(all_dates)),
        "snapshots_expected": int(len(dates)),
        "snapshots_written": written,
        "rows": int(len(snapshots)),
        "errors": errors,
        "strict_ready": len(errors) == 0 and written == int(len(dates)),
    }
    (root / "industry_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["strict_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
