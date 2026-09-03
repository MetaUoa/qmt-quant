from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge deterministic BaoStock shard artifacts")
    p.add_argument("--input", required=True, help="Root containing downloaded shard artifacts")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--bar-cache-dir", default="data/qmt_bars")
    return p.parse_args()


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_shards(root: Path) -> list[tuple[dict, Path]]:
    found: list[tuple[dict, Path]] = []
    for manifest_path in root.rglob("free_data_manifest.json"):
        ref_dir = manifest_path.parent
        shard_root = ref_dir.parent
        if not (shard_root / "qmt_bars").exists():
            continue
        found.append((_read_manifest(manifest_path), shard_root))
    found.sort(key=lambda item: int(item[0].get("shard_index", -1)))
    return found


def _copy_cache_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*.parquet"):
        target = dst / path.name
        if not target.exists():
            shutil.copy2(path, target)


def _add_sentinels(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    columns: list[str],
    empty_row: dict,
) -> pd.DataFrame:
    actual = frame.loc[frame["ts_code"].astype(str).ne("__NONE__")].copy()
    actual = actual.drop_duplicates(["trade_date", "ts_code"], keep="last")
    present = set(actual["trade_date"].astype(str)) if not actual.empty else set()
    open_dates = calendar.loc[calendar["is_open"].eq(1), "cal_date"].astype(str)
    rows = []
    for day in open_dates:
        if day not in present:
            row = dict(empty_row)
            row["trade_date"] = day
            rows.append(row)
    if rows:
        actual = pd.concat([actual, pd.DataFrame(rows, columns=columns)], ignore_index=True)
    return actual[columns].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def merge_shards(
    input_root: Path,
    reference_dir: Path,
    bar_cache_dir: Path,
) -> dict:
    shards = discover_shards(input_root)
    if not shards:
        raise RuntimeError("No shard manifests found")

    manifests = [item[0] for item in shards]
    starts = {str(m["start"]) for m in manifests}
    ends = {str(m["end"]) for m in manifests}
    benchmarks = {str(m["benchmark"]) for m in manifests}
    counts = {int(m["shard_count"]) for m in manifests}
    if len(starts) != 1 or len(ends) != 1 or len(benchmarks) != 1 or len(counts) != 1:
        raise RuntimeError("Shard manifests disagree on start/end/benchmark/shard_count")
    start, end, benchmark, shard_count = (
        next(iter(starts)),
        next(iter(ends)),
        next(iter(benchmarks)),
        next(iter(counts)),
    )
    shard_indexes = {int(m["shard_index"]) for m in manifests}
    if shard_indexes != set(range(shard_count)):
        raise RuntimeError(
            f"Missing shard indexes: {sorted(set(range(shard_count)) - shard_indexes)}"
        )

    basic_parts = []
    calendar_parts = []
    st_parts = []
    limit_parts = []
    suspension_parts = []
    errors = []

    front_out = bar_cache_dir / f"front_{start}_{end}"
    raw_out = bar_cache_dir / f"none_limits_{start}_{end}"
    for manifest, shard_root in shards:
        ref = shard_root / "reference"
        basic_parts.append(pd.read_parquet(ref / "stock_basic.parquet"))
        calendar_parts.append(pd.read_parquet(ref / "trade_calendar.parquet"))
        st_parts.append(pd.read_parquet(ref / "stock_st.parquet"))
        limit_parts.append(pd.read_parquet(ref / "stk_limit.parquet"))
        suspension_parts.append(pd.read_parquet(ref / "suspend_d.parquet"))
        errors.extend(manifest.get("errors", []))
        _copy_cache_tree(shard_root / "qmt_bars" / f"front_{start}_{end}", front_out)
        _copy_cache_tree(shard_root / "qmt_bars" / f"none_limits_{start}_{end}", raw_out)

    basic = (
        pd.concat(basic_parts, ignore_index=True)
        .drop_duplicates("ts_code", keep="last")
        .sort_values("ts_code")
        .reset_index(drop=True)
    )
    calendar = calendar_parts[0].sort_values("cal_date").reset_index(drop=True)
    key_cols = ["cal_date", "is_open"]
    for other in calendar_parts[1:]:
        other = other.sort_values("cal_date").reset_index(drop=True)
        if not calendar[key_cols].equals(other[key_cols]):
            raise RuntimeError("Shard trade calendars differ")

    st = _add_sentinels(
        pd.concat(st_parts, ignore_index=True),
        calendar,
        columns=["trade_date", "ts_code", "name", "type", "type_name"],
        empty_row={"ts_code": "__NONE__", "name": "", "type": "", "type_name": ""},
    )
    limits = _add_sentinels(
        pd.concat(limit_parts, ignore_index=True),
        calendar,
        columns=["trade_date", "ts_code", "pre_close", "up_limit", "down_limit"],
        empty_row={
            "ts_code": "__NONE__",
            "pre_close": float("nan"),
            "up_limit": float("nan"),
            "down_limit": float("nan"),
        },
    )
    susp = (
        pd.concat(suspension_parts, ignore_index=True)
        .drop_duplicates(["ts_code", "trade_date", "suspend_type"], keep="last")
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )

    reference_dir.mkdir(parents=True, exist_ok=True)
    basic.to_parquet(reference_dir / "stock_basic.parquet", index=False)
    calendar.to_parquet(reference_dir / "trade_calendar.parquet", index=False)
    st.to_parquet(reference_dir / "stock_st.parquet", index=False)
    limits.to_parquet(reference_dir / "stk_limit.parquet", index=False)
    susp.to_parquet(reference_dir / "suspend_d.parquet", index=False)

    front_codes = {p.stem for p in front_out.glob("*.parquet") if p.stem != benchmark}
    raw_codes = {
        p.name[:-8]
        for p in raw_out.glob("*.parquet")
        if not p.name.endswith(".meta.parquet")
    }
    expected = set(basic["ts_code"].astype(str))
    expected_total_values = {
        int(m.get("candidate_symbols_total", len(expected))) for m in manifests
    }
    expected_total = max(expected_total_values) if expected_total_values else len(expected)
    strict_ready = bool(
        len(shards) == shard_count
        and len(expected) == expected_total
        and expected.issubset(front_codes)
        and expected.issubset(raw_codes)
        and not errors
        and (front_out / f"{benchmark}.parquet").exists()
    )
    manifest = {
        "source": "baostock-sharded",
        "start": start,
        "end": end,
        "benchmark": benchmark,
        "shard_count": shard_count,
        "merged_shards": len(shards),
        "candidate_symbols_total": expected_total,
        "symbols": len(expected),
        "adjusted_symbols_cached": len(expected & front_codes),
        "raw_symbols_cached": len(expected & raw_codes),
        "strict_ready": strict_ready,
        "errors": errors,
    }
    (reference_dir / "free_data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = merge_shards(
        Path(args.input), Path(args.reference_dir), Path(args.bar_cache_dir)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["strict_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
