from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qmt_quant.data_validation import (
    assess_akshare_crosscheck,
    assess_free_data_manifest,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate BaoStock free-data manifest and optional AKShare cross-check")
    p.add_argument("manifest")
    p.add_argument("--akshare-report", default="")
    p.add_argument("--min-symbol-coverage", type=float, default=0.98)
    p.add_argument("--min-akshare-pass-ratio", type=float, default=0.80)
    p.add_argument("--min-akshare-compared", type=int, default=5)
    p.add_argument("--require-akshare", action="store_true")
    p.add_argument("--output", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest_check = assess_free_data_manifest(
        read_json(manifest_path),
        min_symbol_coverage=args.min_symbol_coverage,
    )
    crosscheck = None
    if args.akshare_report:
        report_path = Path(args.akshare_report)
        if report_path.exists():
            crosscheck = assess_akshare_crosscheck(
                pd.read_csv(report_path),
                min_pass_ratio=args.min_akshare_pass_ratio,
                min_compared=args.min_akshare_compared,
            )
        else:
            crosscheck = assess_akshare_crosscheck(
                None,
                min_pass_ratio=args.min_akshare_pass_ratio,
                min_compared=args.min_akshare_compared,
            )

    failures = list(manifest_check["failures"])
    if args.require_akshare and (crosscheck is None or not crosscheck["ready"]):
        failures.append("AKShare cross-check did not meet the required comparison/pass thresholds")

    payload = {
        "passed": not failures,
        "manifest": manifest_check,
        "akshare_crosscheck": crosscheck,
        "akshare_required": bool(args.require_akshare),
        "failures": failures,
    }
    output = Path(args.output) if args.output else manifest_path.with_name("free_data_validation.json")
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Validation report: {output.resolve()}")
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
