from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import socket
import sys
import time
from typing import Callable

import pandas as pd

from qmt_quant.adjustment_provenance import (
    front_adjustment_provenance,
    raw_reference_provenance,
)
from qmt_quant.free_data import (
    _write_qmt_cache,
    baostock_session,
    build_reference_tables,
    fetch_history,
    fetch_stock_basic,
    fetch_trade_calendar,
)


def select_stock_shard(
    stock_basic: pd.DataFrame,
    shard_index: int,
    shard_count: int,
) -> pd.DataFrame:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    ordered = stock_basic.sort_values("ts_code").reset_index(drop=True)
    mask = (pd.Series(range(len(ordered)), index=ordered.index) % shard_count).eq(shard_index)
    return ordered.loc[mask].reset_index(drop=True)


def active_in_range(stock_basic: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    list_ts = pd.to_datetime(stock_basic["list_date"], format="%Y%m%d", errors="coerce")
    delist_ts = pd.to_datetime(stock_basic["delist_date"], format="%Y%m%d", errors="coerce")
    overlap = list_ts.le(end_ts) & (delist_ts.isna() | delist_ts.ge(start_ts))
    return stock_basic.loc[overlap].sort_values("ts_code").reset_index(drop=True)


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (AttributeError, ValueError):
                pass


def _configure_socket_timeout(timeout_seconds: float) -> None:
    if timeout_seconds <= 0:
        raise ValueError("socket_timeout must be > 0")
    socket.setdefaulttimeout(timeout_seconds)


def _bind_baostock_socket_timeout(timeout_seconds: float) -> None:
    """Bound the active BaoStock 0.9.3 socket after every successful login.

    BaoStock 0.9.3 stores the live TCP socket at
    ``baostock.common.context.default_socket``. Login/reconnect replaces that
    object, so the timeout must be rebound after each successful login rather
    than assuming a stable private socket or the pre-0.9.3 socketpool layout.
    """
    if timeout_seconds <= 0:
        raise ValueError("socket_timeout must be > 0")

    context = importlib.import_module("baostock.common.context")
    active_socket = getattr(context, "default_socket", None)
    settimeout = getattr(active_socket, "settimeout", None)
    if not callable(settimeout):
        raise RuntimeError("BaoStock default_socket is unavailable for timeout binding")
    settimeout(timeout_seconds)


def _reconnect_baostock(api, *, socket_timeout_seconds: float) -> None:
    try:
        api.logout()
    except Exception:
        pass
    _configure_socket_timeout(socket_timeout_seconds)
    login = api.login()
    if str(getattr(login, "error_code", "0")) != "0":
        raise RuntimeError(
            f"BaoStock reconnect failed: {getattr(login, 'error_code', '')} "
            f"{getattr(login, 'error_msg', '')}".strip()
        )
    _bind_baostock_socket_timeout(socket_timeout_seconds)


def fetch_history_with_retry(
    api,
    code: str,
    start: str,
    end: str,
    *,
    adjusted: bool,
    include_meta: bool,
    attempts: int = 4,
    sleep_seconds: float = 0.5,
    socket_timeout_seconds: float = 45.0,
    fetcher: Callable | None = None,
) -> pd.DataFrame:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    fetch = fetcher or fetch_history
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fetch(
                api,
                code,
                start,
                end,
                adjusted=adjusted,
                include_meta=include_meta,
            )
        except Exception as exc:
            last = exc
            if attempt >= attempts - 1:
                break
            _reconnect_baostock(api, socket_timeout_seconds=socket_timeout_seconds)
            time.sleep(max(0.5, sleep_seconds) * (2 ** attempt))
    raise RuntimeError(f"{code} history fetch failed after retries: {last}") from last


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download one deterministic BaoStock A-share shard")
    p.add_argument("--start", default="20170101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--benchmark", default="000905.SH")
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--bar-cache-dir", required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-count", type=int, default=20)
    p.add_argument("--sleep", type=float, default=0.02)
    p.add_argument("--socket-timeout", type=float, default=45.0)
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def main() -> int:
    _configure_utf8_console()
    args = parse_args()
    _configure_socket_timeout(args.socket_timeout)
    ref_root = Path(args.reference_dir)
    cache_root = Path(args.bar_cache_dir)
    front_root = cache_root / f"front_{args.start}_{args.end}"
    raw_root = cache_root / f"none_limits_{args.start}_{args.end}"
    ref_root.mkdir(parents=True, exist_ok=True)
    front_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    errors: list[dict[str, str]] = []
    raw_meta: dict[str, pd.DataFrame] = {}

    with baostock_session() as bs:
        _bind_baostock_socket_timeout(args.socket_timeout)
        all_basic = active_in_range(fetch_stock_basic(bs), args.start, args.end)
        total_symbols = len(all_basic)
        basic = select_stock_shard(all_basic, args.shard_index, args.shard_count)
        calendar = fetch_trade_calendar(bs, args.start, args.end)
        basic.to_parquet(ref_root / "stock_basic.parquet", index=False)
        calendar.to_parquet(ref_root / "trade_calendar.parquet", index=False)

        adjusted_loaded = 0
        raw_loaded = 0
        codes = list(basic["ts_code"])
        for number, code in enumerate(codes, 1):
            front_path = front_root / f"{code}.parquet"
            raw_path = raw_root / f"{code}.parquet"
            if args.refresh or not front_path.exists():
                try:
                    frame = fetch_history_with_retry(
                        bs,
                        code,
                        args.start,
                        args.end,
                        adjusted=True,
                        include_meta=False,
                        sleep_seconds=args.sleep,
                        socket_timeout_seconds=args.socket_timeout,
                    )
                    if not frame.empty:
                        _write_qmt_cache(frame, front_path)
                except Exception as exc:
                    errors.append({"code": code, "kind": "front", "error": str(exc)})
            if front_path.exists():
                adjusted_loaded += 1

            if args.refresh or not raw_path.exists():
                try:
                    raw = fetch_history_with_retry(
                        bs,
                        code,
                        args.start,
                        args.end,
                        adjusted=False,
                        include_meta=True,
                        sleep_seconds=args.sleep,
                        socket_timeout_seconds=args.socket_timeout,
                    )
                    if not raw.empty:
                        _write_qmt_cache(raw, raw_path)
                        raw[["date", "isST", "tradestatus"]].to_parquet(
                            raw_root / f"{code}.meta.parquet", index=False
                        )
                except Exception as exc:
                    errors.append({"code": code, "kind": "raw", "error": str(exc)})
            if raw_path.exists():
                raw_loaded += 1
                raw_cache = pd.read_parquet(raw_path)
                meta_path = raw_root / f"{code}.meta.parquet"
                if meta_path.exists():
                    raw_cache = raw_cache.merge(pd.read_parquet(meta_path), on="date", how="left")
                raw_meta[code] = raw_cache

            if number == 1 or number % 25 == 0 or number == len(codes):
                print(
                    f"[shard {args.shard_index}/{args.shard_count}] {number}/{len(codes)} "
                    f"front={adjusted_loaded} raw={raw_loaded} errors={len(errors)}"
                )
            if args.sleep > 0 and number < len(codes):
                time.sleep(args.sleep)

        benchmark_path = front_root / f"{args.benchmark}.parquet"
        if args.refresh or not benchmark_path.exists():
            benchmark = fetch_history_with_retry(
                bs,
                args.benchmark,
                args.start,
                args.end,
                adjusted=True,
                include_meta=False,
                sleep_seconds=args.sleep,
                socket_timeout_seconds=args.socket_timeout,
            )
            if benchmark.empty:
                raise RuntimeError(f"BaoStock returned no benchmark data for {args.benchmark}")
            _write_qmt_cache(benchmark, benchmark_path)

    st, limits, susp = build_reference_tables(basic, calendar, raw_meta)
    st.to_parquet(ref_root / "stock_st.parquet", index=False)
    limits.to_parquet(ref_root / "stk_limit.parquet", index=False)
    susp.to_parquet(ref_root / "suspend_d.parquet", index=False)

    manifest = {
        "source": "baostock",
        "start": args.start,
        "end": args.end,
        "benchmark": args.benchmark,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "candidate_symbols_total": total_symbols,
        "symbols": len(basic),
        "adjusted_symbols_cached": adjusted_loaded,
        "raw_symbols_cached": raw_loaded,
        "adjustment_provenance": {
            "adjusted": front_adjustment_provenance(
                provider="baostock", requested_end=args.end
            ),
            "raw": raw_reference_provenance(
                provider="baostock", requested_end=args.end
            ),
        },
        "strict_ready": bool(
            adjusted_loaded == len(basic) and raw_loaded == len(basic) and not errors
        ),
        "errors": errors,
    }
    (ref_root / "free_data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["strict_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
