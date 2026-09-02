from __future__ import annotations

import argparse
import json
from pathlib import Path

from qmt_quant.free_data import prepare_baostock_cache, verify_with_akshare
from qmt_quant.reference_data import ReferenceData


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare free 2018-2025 A-share research data with BaoStock."
    )
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--benchmark", default="000905.SH")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--bar-cache-dir", default="data/qmt_bars")
    p.add_argument("--max-stocks", type=int, default=0, help="Smoke/dev only; 0 means all PIT symbols")
    p.add_argument("--sleep", type=float, default=0.02, help="Delay between symbols")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--verify-akshare", action="store_true")
    p.add_argument("--verify-sample", type=int, default=20)
    p.add_argument("--verify-tolerance", type=float, default=0.01)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_baostock_cache(
        args.reference_dir,
        args.bar_cache_dir,
        args.start,
        args.end,
        benchmark=args.benchmark,
        max_stocks=args.max_stocks,
        sleep_seconds=args.sleep,
        refresh=args.refresh,
    )

    ref = ReferenceData.from_dir(args.reference_dir)
    manifest["reference_audit"] = ref.audit().__dict__
    manifest_path = Path(args.reference_dir) / "free_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))

    if args.verify_akshare:
        raw_root = Path(args.bar_cache_dir) / f"none_limits_{args.start}_{args.end}"
        codes = ref.codes_ever_active(args.start, args.end)
        report = verify_with_akshare(
            codes,
            args.start,
            args.end,
            raw_root,
            sample_size=args.verify_sample,
            tolerance=args.verify_tolerance,
        )
        verify_out = Path(args.reference_dir) / "akshare_crosscheck.csv"
        report.to_csv(verify_out, index=False, encoding="utf-8-sig")
        passed = int(report["status"].eq("pass").sum()) if not report.empty else 0
        print(f"AKShare cross-check: {passed}/{len(report)} pass -> {verify_out}")

    if not manifest.get("strict_ready", False):
        print(
            "WARNING: some symbols are missing from one or both BaoStock caches. "
            "The strict data audit will decide whether coverage is sufficient."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
