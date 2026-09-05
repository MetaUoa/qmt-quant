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
    active_in_range,
    fetch_history_with_retry,
    select_stock_shard,
)
from qmt_quant.free_data import (
    _result_frame,
    baostock_session,
    fetch_stock_basic,
    ts_code_to_baostock,
)
from qmt_quant.pit_exposures import turnover_implied_float_market_cap


EXPOSURE_FIELDS = "date,code,close,volume,turn,tradestatus"


def _fetch_exposure_history(api, code, start, end, *, adjusted=False, include_meta=False):
    del adjusted, include_meta
    result = api.query_history_k_data_plus(
        ts_code_to_baostock(code),
        EXPOSURE_FIELDS,
        start_date=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end).strftime("%Y-%m-%d"),
        frequency="d",
        adjustflag="3",
    )
    return _result_frame(result)


def fetch_active_stock_basic_with_retry(
    api,
    start,
    end,
    *,
    attempts: int = 4,
    sleep_seconds: float = 0.5,
    socket_timeout_seconds: float = 45.0,
) -> pd.DataFrame:
    """Fetch PIT shard membership with the same bounded reconnect policy as bars.

    BaoStock can reset or time out while streaming query_stock_basic before the
    per-symbol history loop begins.  Every retry reconnects and rebinds the active
    context socket timeout; exhaustion remains fail-closed.
    """
    if int(attempts) <= 0:
        raise ValueError("attempts must be positive")
    last: Exception | None = None
    for attempt in range(int(attempts)):
        try:
            return active_in_range(fetch_stock_basic(api), start, end)
        except Exception as exc:
            last = exc
            if attempt >= int(attempts) - 1:
                break
            _reconnect_baostock(
                api,
                socket_timeout_seconds=float(socket_timeout_seconds),
            )
            time.sleep(max(0.0, float(sleep_seconds)) * (2**attempt))
    raise RuntimeError(f"stock-basic bootstrap failed after retries: {last}") from last


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download one deterministic PIT exposure shard")
    p.add_argument("--start", default="20170101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--output", required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-count", type=int, default=20)
    p.add_argument("--sleep", type=float, default=0.02)
    p.add_argument("--socket-timeout", type=float, default=45.0)
    p.add_argument("--attempts", type=int, default=4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _configure_socket_timeout(args.socket_timeout)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    written = 0

    with baostock_session() as bs:
        _bind_baostock_socket_timeout(args.socket_timeout)
        basic = fetch_active_stock_basic_with_retry(
            bs,
            args.start,
            args.end,
            attempts=args.attempts,
            sleep_seconds=max(args.sleep, 0.5),
            socket_timeout_seconds=args.socket_timeout,
        )
        shard = select_stock_shard(basic, args.shard_index, args.shard_count)
        codes = list(shard["ts_code"])
        shard[["ts_code", "list_date", "delist_date"]].to_parquet(
            root / "stock_basic_shard.parquet", index=False
        )
        for number, code in enumerate(codes, 1):
            try:
                history = fetch_history_with_retry(
                    bs,
                    code,
                    args.start,
                    args.end,
                    adjusted=False,
                    include_meta=False,
                    attempts=args.attempts,
                    sleep_seconds=args.sleep,
                    socket_timeout_seconds=args.socket_timeout,
                    fetcher=_fetch_exposure_history,
                )
                exposure = turnover_implied_float_market_cap(history)
                exposure.insert(1, "ts_code", code)
                exposure.to_parquet(root / f"{code}.parquet", index=False)
                written += 1
            except Exception as exc:
                errors.append({"code": code, "error": str(exc)})
            if number == 1 or number % 50 == 0 or number == len(codes):
                print(
                    f"[PIT exposure] {number}/{len(codes)} "
                    f"written={written} errors={len(errors)}"
                )
            if args.sleep > 0 and number < len(codes):
                time.sleep(args.sleep)

    manifest = {
        "source": "baostock-turnover-implied-float-cap",
        "start": args.start,
        "end": args.end,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "symbols_expected": int(len(codes)),
        "symbols_written": int(written),
        "errors": errors,
        "strict_ready": len(errors) == 0 and written == len(codes),
    }
    (root / "pit_exposure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["strict_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
