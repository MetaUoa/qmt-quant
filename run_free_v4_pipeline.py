from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from qmt_quant.data_validation import build_baseline_summary, read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V3.8.1 smoke -> V3.9 full free-data audit -> V4.0 strict baseline"
    )
    p.add_argument("--stage", choices=["all", "smoke", "full-data", "baseline"], default="all")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--benchmark", default="000905.SH")
    p.add_argument("--smoke-stocks", type=int, default=200)
    p.add_argument("--smoke-reference-dir", default="data/smoke/reference")
    p.add_argument("--smoke-bar-cache-dir", default="data/smoke/qmt_bars")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--bar-cache-dir", default="data/qmt_bars")
    p.add_argument("--min-symbol-coverage", type=float, default=0.98)
    p.add_argument("--min-session-coverage", type=float, default=0.97)
    p.add_argument("--akshare-sample", type=int, default=20)
    p.add_argument("--min-akshare-pass-ratio", type=float, default=0.80)
    p.add_argument("--min-akshare-compared", type=int, default=5)
    p.add_argument("--skip-akshare", action="store_true")
    p.add_argument("--require-akshare", action="store_true")
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def _run(name: str, cmd: list[str], root: Path, env: dict[str, str], steps: list[dict]) -> bool:
    print(f"\n========== {name} ==========")
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=root, env=env)
    row = {
        "name": name,
        "command": cmd,
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
    }
    steps.append(row)
    return bool(row["passed"])


def _prepare_and_validate(
    *,
    label: str,
    reference_dir: str,
    bar_cache_dir: str,
    max_stocks: int,
    args: argparse.Namespace,
    root: Path,
    env: dict[str, str],
    steps: list[dict],
) -> bool:
    py = sys.executable
    prepare = [
        py,
        "prepare_free_data.py",
        "--start",
        args.start,
        "--end",
        args.end,
        "--benchmark",
        args.benchmark,
        "--reference-dir",
        reference_dir,
        "--bar-cache-dir",
        bar_cache_dir,
    ]
    if max_stocks > 0:
        prepare.extend(["--max-stocks", str(max_stocks)])
    if args.refresh:
        prepare.append("--refresh")
    if not args.skip_akshare:
        prepare.extend(["--verify-akshare", "--verify-sample", str(args.akshare_sample)])
    if not _run(f"{label}_prepare", prepare, root, env, steps):
        return False

    manifest = str(Path(reference_dir) / "free_data_manifest.json")
    crosscheck = Path(reference_dir) / "akshare_crosscheck.csv"
    validate = [
        py,
        "validate_free_data.py",
        manifest,
        "--min-symbol-coverage",
        str(args.min_symbol_coverage),
        "--min-akshare-pass-ratio",
        str(args.min_akshare_pass_ratio),
        "--min-akshare-compared",
        str(args.min_akshare_compared),
    ]
    if not args.skip_akshare:
        validate.extend(["--akshare-report", str(crosscheck)])
    if args.require_akshare:
        validate.append("--require-akshare")
    return _run(f"{label}_manifest_validation", validate, root, env, steps)


def _audit(
    *,
    label: str,
    reference_dir: str,
    bar_cache_dir: str,
    output: str,
    args: argparse.Namespace,
    root: Path,
    env: dict[str, str],
    steps: list[dict],
) -> bool:
    cmd = [
        sys.executable,
        "run_data_audit.py",
        "--start",
        args.start,
        "--end",
        args.end,
        "--benchmark",
        args.benchmark,
        "--reference-dir",
        reference_dir,
        "--bar-cache-dir",
        bar_cache_dir,
        "--output",
        output,
        "--min-symbol-coverage",
        str(args.min_symbol_coverage),
        "--min-session-coverage",
        str(args.min_session_coverage),
    ]
    return _run(f"{label}_data_audit", cmd, root, env, steps)


def _baseline(
    *,
    label: str,
    reference_dir: str,
    bar_cache_dir: str,
    output: str,
    args: argparse.Namespace,
    root: Path,
    env: dict[str, str],
    steps: list[dict],
) -> bool:
    py = sys.executable
    cmd = [
        py,
        "run_backtest.py",
        "--start",
        args.start,
        "--end",
        args.end,
        "--benchmark",
        args.benchmark,
        "--reference-dir",
        reference_dir,
        "--bar-cache-dir",
        bar_cache_dir,
        "--strict-reference",
        "--min-symbol-coverage",
        str(args.min_symbol_coverage),
        "--output",
        output,
    ]
    if not _run(f"{label}_strict_baseline", cmd, root, env, steps):
        return False
    validate = [
        py,
        "validate_backtest_output.py",
        output,
        "--min-symbol-coverage",
        str(args.min_symbol_coverage),
    ]
    return _run(f"{label}_baseline_validation", validate, root, env, steps)


def _write_v4_baseline_summary(root: Path, output: str) -> Path:
    out = root / output
    metrics = read_json(out / "metrics.json")
    yearly_path = out / "yearly_returns.csv"
    yearly = pd.read_csv(yearly_path) if yearly_path.exists() else pd.DataFrame()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "baostock",
        "period": [metrics.get("start"), metrics.get("end")],
        "baseline": build_baseline_summary(metrics, yearly),
        "metrics_file": str(out / "metrics.json"),
        "yearly_returns_file": str(yearly_path),
    }
    target = out / "baseline_summary.json"
    write_json(target, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return target


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["QMT_QUANT_CACHE_ONLY"] = "1"
    steps: list[dict] = []

    def fail() -> int:
        return _finish(root, args, steps, False)

    if args.stage in {"all", "smoke"}:
        if not _prepare_and_validate(
            label="v3_8_1_smoke",
            reference_dir=args.smoke_reference_dir,
            bar_cache_dir=args.smoke_bar_cache_dir,
            max_stocks=args.smoke_stocks,
            args=args,
            root=root,
            env=env,
            steps=steps,
        ):
            return fail()
        if not _audit(
            label="v3_8_1_smoke",
            reference_dir=args.smoke_reference_dir,
            bar_cache_dir=args.smoke_bar_cache_dir,
            output="output/v3_8_1_smoke/data_audit",
            args=args,
            root=root,
            env=env,
            steps=steps,
        ):
            return fail()
        if not _baseline(
            label="v3_8_1_smoke",
            reference_dir=args.smoke_reference_dir,
            bar_cache_dir=args.smoke_bar_cache_dir,
            output="output/v3_8_1_smoke/baseline",
            args=args,
            root=root,
            env=env,
            steps=steps,
        ):
            return fail()

    if args.stage in {"all", "full-data"}:
        if not _prepare_and_validate(
            label="v3_9_full",
            reference_dir=args.reference_dir,
            bar_cache_dir=args.bar_cache_dir,
            max_stocks=0,
            args=args,
            root=root,
            env=env,
            steps=steps,
        ):
            return fail()
        if not _audit(
            label="v3_9_full",
            reference_dir=args.reference_dir,
            bar_cache_dir=args.bar_cache_dir,
            output="output/v3_9_data_audit",
            args=args,
            root=root,
            env=env,
            steps=steps,
        ):
            return fail()

    if args.stage in {"all", "baseline"}:
        manifest_path = Path(args.reference_dir) / "free_data_manifest.json"
        if not manifest_path.exists():
            print(f"Missing full-data manifest: {manifest_path}. Run --stage full-data first.")
            return fail()
        validation = [
            sys.executable,
            "validate_free_data.py",
            str(manifest_path),
            "--min-symbol-coverage",
            str(args.min_symbol_coverage),
        ]
        crosscheck = Path(args.reference_dir) / "akshare_crosscheck.csv"
        if crosscheck.exists() and not args.skip_akshare:
            validation.extend(["--akshare-report", str(crosscheck)])
        if args.require_akshare:
            validation.append("--require-akshare")
        if not _run("v4_0_full_data_preflight", validation, root, env, steps):
            return fail()
        if not _audit(
            label="v4_0_preflight",
            reference_dir=args.reference_dir,
            bar_cache_dir=args.bar_cache_dir,
            output="output/v3_9_data_audit",
            args=args,
            root=root,
            env=env,
            steps=steps,
        ):
            return fail()
        if not _baseline(
            label="v4_0",
            reference_dir=args.reference_dir,
            bar_cache_dir=args.bar_cache_dir,
            output="output/v4_0_baseline",
            args=args,
            root=root,
            env=env,
            steps=steps,
        ):
            return fail()
        _write_v4_baseline_summary(root, "output/v4_0_baseline")

    return _finish(root, args, steps, True)


def _finish(root: Path, args: argparse.Namespace, steps: list[dict], passed: bool) -> int:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "data_source": "baostock",
        "cache_only": True,
        "passed": bool(passed and all(bool(x.get("passed")) for x in steps)),
        "steps": steps,
    }
    target = root / "output" / "free_v4_pipeline" / "pipeline_summary.json"
    write_json(target, payload)
    print("\n========== FREE DATA V4 PIPELINE ==========")
    print("PASS" if payload["passed"] else "FAIL")
    print(target)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
