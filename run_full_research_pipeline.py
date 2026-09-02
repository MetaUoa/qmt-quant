from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-command V2.2 -> V5 research/acceptance pipeline")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--benchmark", default="000905.SH")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--bar-cache-dir", default="data/qmt_bars")
    p.add_argument("--data-source", choices=["qmt", "baostock"], default="qmt")
    p.add_argument("--profile", choices=["quick", "balanced", "deep"], default="quick")
    p.add_argument(
        "--prepare-reference",
        action="store_true",
        help="QMT: prepare Tushare PIT reference. BaoStock: download free reference + bars.",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="QMT: download QMT history. BaoStock: refresh free caches when preparing.",
    )
    p.add_argument("--max-stocks", type=int, default=0, help="BaoStock smoke/dev only")
    p.add_argument("--verify-akshare", action="store_true", help="BaoStock: sample cross-check with AKShare")
    p.add_argument("--require-grade", choices=["A", "B", "C"], default="C")
    p.add_argument("--continue-on-grade-fail", action="store_true")
    return p.parse_args()


def run(
    name: str,
    cmd: list[str],
    root: Path,
    steps: list[dict],
    *,
    env: dict[str, str] | None = None,
) -> bool:
    print(f"\n========== {name} ==========")
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=root, env=env)
    item = {
        "name": name,
        "command": cmd,
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
    }
    steps.append(item)
    return item["passed"]


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    steps: list[dict] = []
    py = sys.executable
    env = os.environ.copy()

    if args.data_source == "baostock":
        env["QMT_QUANT_CACHE_ONLY"] = "1"
        if args.prepare_reference:
            cmd = [
                py,
                "prepare_free_data.py",
                "--start",
                args.start,
                "--end",
                args.end,
                "--benchmark",
                args.benchmark,
                "--reference-dir",
                args.reference_dir,
                "--bar-cache-dir",
                args.bar_cache_dir,
            ]
            if args.max_stocks > 0:
                cmd.extend(["--max-stocks", str(args.max_stocks)])
            if args.download:
                cmd.append("--refresh")
            if args.verify_akshare:
                cmd.append("--verify-akshare")
            if not run("prepare_free_data", cmd, root, steps, env=env):
                return finish(root, steps, args.data_source)
    elif args.prepare_reference:
        if not run(
            "prepare_reference",
            [
                py,
                "prepare_reference_data.py",
                "--start",
                args.start,
                "--end",
                args.end,
                "--output",
                args.reference_dir,
            ],
            root,
            steps,
            env=env,
        ):
            return finish(root, steps, args.data_source)

    common = [
        "--start",
        args.start,
        "--end",
        args.end,
        "--benchmark",
        args.benchmark,
        "--reference-dir",
        args.reference_dir,
        "--bar-cache-dir",
        args.bar_cache_dir,
    ]

    audit_cmd = [py, "run_data_audit.py", *common]
    if args.data_source == "qmt" and args.download:
        audit_cmd.append("--download")
    if not run("v2_2_data_audit", audit_cmd, root, steps, env=env):
        return finish(root, steps, args.data_source)

    baseline_cmd = [
        py,
        "run_backtest.py",
        *common,
        "--strict-reference",
        "--output",
        "output/v2_5_baseline",
    ]
    if args.data_source == "qmt" and args.download:
        baseline_cmd.append("--download")
    if not run("v2_5_baseline", baseline_cmd, root, steps, env=env):
        return finish(root, steps, args.data_source)
    if not run(
        "v2_5_baseline_validation",
        [
            py,
            "validate_backtest_output.py",
            "output/v2_5_baseline",
            "--min-symbol-coverage",
            "0.98",
        ],
        root,
        steps,
        env=env,
    ):
        return finish(root, steps, args.data_source)

    if not run(
        "v3_parameter_research",
        [
            py,
            "run_parameter_research.py",
            *common,
            "--profile",
            args.profile,
            "--strict-reference",
        ],
        root,
        steps,
        env=env,
    ):
        return finish(root, steps, args.data_source)

    if not run(
        "v4_walk_forward",
        [
            py,
            "run_walk_forward.py",
            *common,
            "--strict-reference",
        ],
        root,
        steps,
        env=env,
    ):
        return finish(root, steps, args.data_source)

    if not run(
        "v4_5_stress",
        [
            py,
            "run_stress_tests.py",
            *common,
            "--strategy-config",
            "output/v3_research/best_config.json",
            "--strict-reference",
        ],
        root,
        steps,
        env=env,
    ):
        return finish(root, steps, args.data_source)

    acceptance_cmd = [
        py,
        "run_acceptance.py",
        "--backtest",
        "output/v3_research/best_full_result/metrics.json",
        "--walk-forward",
        "output/walk_forward/walk_forward_metrics.json",
        "--folds",
        "output/walk_forward/walk_forward_folds.csv",
        "--stress",
        "output/v4_5_stress/stress_summary.json",
        "--require-grade",
        args.require_grade,
    ]
    accepted = run("v5_acceptance", acceptance_cmd, root, steps, env=env)
    if not accepted and args.continue_on_grade_fail:
        steps[-1]["passed"] = True
        steps[-1]["note"] = "Grade below requested threshold; pipeline continued by request."
    return finish(root, steps, args.data_source)


def finish(root: Path, steps: list[dict], data_source: str = "qmt") -> int:
    passed = all(bool(x.get("passed")) for x in steps)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "passed": passed,
        "steps": steps,
    }
    out = root / "output" / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n========== PIPELINE RESULT ==========")
    print("PASS" if passed else "FAIL")
    print(out / "pipeline_summary.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
