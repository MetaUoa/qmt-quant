from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge deterministic PIT industry shard artifacts")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--upstream-exposure-run-id", type=int, required=True)
    return p.parse_args()


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_shards(root: Path) -> list[tuple[dict, Path]]:
    found: list[tuple[dict, Path]] = []
    for manifest_path in root.rglob("industry_manifest.json"):
        frame_path = manifest_path.parent / "industry_snapshots.parquet"
        if frame_path.exists():
            found.append((_load_manifest(manifest_path), manifest_path.parent))
    found.sort(key=lambda item: int(item[0].get("shard_index", -1)))
    return found


def merge_industry_shards(
    input_root: Path,
    output_root: Path,
    *,
    upstream_exposure_run_id: int,
) -> dict:
    if int(upstream_exposure_run_id) <= 0:
        raise ValueError("upstream_exposure_run_id must be positive")
    shards = discover_shards(input_root)
    if not shards:
        raise RuntimeError("No PIT industry shard manifests found")

    manifests = [manifest for manifest, _ in shards]
    counts = {int(m.get("shard_count", -1)) for m in manifests}
    starts = {str(m.get("start")) for m in manifests}
    ends = {str(m.get("end")) for m in manifests}
    totals = {int(m.get("snapshot_candidates_total", -1)) for m in manifests}
    if len(counts) != 1 or len(starts) != 1 or len(ends) != 1 or len(totals) != 1:
        raise RuntimeError("PIT industry shard manifests disagree on contract")

    shard_count = next(iter(counts))
    total_expected = next(iter(totals))
    indexes = {int(m.get("shard_index", -1)) for m in manifests}
    if indexes != set(range(shard_count)):
        missing = sorted(set(range(shard_count)).difference(indexes))
        raise RuntimeError(f"Missing PIT industry shard indexes: {missing}")
    if len(shards) != shard_count:
        raise RuntimeError(f"Expected {shard_count} PIT industry shards, found {len(shards)}")

    errors: list[dict] = []
    expected_sum = 0
    frames: list[pd.DataFrame] = []
    for manifest, root in shards:
        if not bool(manifest.get("strict_ready")):
            raise RuntimeError(f"Non-strict PIT industry shard {manifest.get('shard_index')}")
        shard_errors = list(manifest.get("errors") or [])
        errors.extend(shard_errors)
        expected_sum += int(manifest.get("snapshots_expected", 0))
        frame = pd.read_parquet(root / "industry_snapshots.parquet")
        if frame.empty:
            raise RuntimeError(f"Empty PIT industry shard {manifest.get('shard_index')}")
        frames.append(frame)

    if errors:
        raise RuntimeError(f"PIT industry shards contain {len(errors)} errors")
    if expected_sum != total_expected:
        raise RuntimeError(
            f"Shard snapshot expectations sum to {expected_sum}, expected {total_expected}"
        )

    snapshots = pd.concat(frames, ignore_index=True)
    required = {"asof_date", "ts_code", "industry"}
    missing_columns = required.difference(snapshots.columns)
    if missing_columns:
        raise RuntimeError(f"Merged PIT industry data missing columns: {sorted(missing_columns)}")
    snapshots["asof_date"] = pd.to_datetime(snapshots["asof_date"], errors="raise").dt.normalize()
    duplicate_mask = snapshots.duplicated(["asof_date", "ts_code"], keep=False)
    if bool(duplicate_mask.any()):
        raise RuntimeError("Duplicate asof_date/ts_code rows across PIT industry shards")
    snapshots = snapshots.sort_values(["asof_date", "ts_code"]).reset_index(drop=True)
    written = int(snapshots["asof_date"].nunique())
    if written != total_expected:
        raise RuntimeError(
            f"Merged PIT industry snapshot count {written} does not equal expected {total_expected}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(output_root / "industry_snapshots.parquet", index=False)
    manifest = {
        "source": "baostock-query_stock_industry-sharded",
        "upstream_exposure_run_id": int(upstream_exposure_run_id),
        "start": next(iter(starts)),
        "end": next(iter(ends)),
        "snapshot_frequency": "monthly_first_open_session",
        "shard_count": shard_count,
        "merged_shards": len(shards),
        "snapshots_expected": total_expected,
        "snapshots_written": written,
        "rows": int(len(snapshots)),
        "errors": [],
        "strict_ready": True,
    }
    (output_root / "industry_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = merge_industry_shards(
        Path(args.input),
        Path(args.output),
        upstream_exposure_run_id=args.upstream_exposure_run_id,
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["strict_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
