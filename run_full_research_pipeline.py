from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-command V2.2 -> V5 QMT research/acceptance pipeline")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--profile", choices=["quick", "balanced", "deep"], default="quick")
    p.add_argument("--prepare-reference", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--require-grade", choices=["A", "B", "C"], default="C")
    p.add_argument("--continue-on-grade-fail", action="store_true")
    return p.parse_args()


def run(name: str, cmd: list[str], root: Path, steps: list[dict]) -> bool:
    print(f"\n========== {name} ==========")
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=root)
    item = {"name": name, "command": cmd, "returncode": int(proc.returncode), "passed": proc.returncode == 0}
    steps.append(item)
    return item["passed"]


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    steps: list[dict] = []
    py = sys.executable

    if args.prepare_reference:
        if not run(
            "prepare_reference",
            [py, "prepare_reference_data.py", "--start", args.start, "--end", args.end, "--output", args.reference_dir],
            root,
            steps,
        ):
            return finish(root, steps)

    audit_cmd = [
        py,
        "run_data_audit.py",
        "--start",
        args.start,
        "--end",
        args.end,
        "--reference-dir",
        args.reference_dir,
    ]
    if args.download:
        audit_cmd.append("--download")
    if not run("v2_2_data_audit", audit_cmd, root, steps):
        return finish(root, steps)

    baseline_cmd = [
        py,
        "run_backtest.py",
        "--start",
        args.start,
        "--end",
        args.end,
        "--reference-dir",
        args.reference_dir,
        "--strict-reference",
        "--output",
        "output/v2_5_baseline",
    ]
    if args.download:
        baseline_cmd.append("--download")
    if not run("v2_5_baseline", baseline_cmd, root, steps):
        return finish(root, steps)
    if not run(
        "v2_5_baseline_validation",
        [py, "validate_backtest_output.py", "output/v2_5_baseline", "--min-symbol-coverage", "0.98"],
        root,
        steps,
    ):
        return finish(root, steps)

    if not run(
        "v3_parameter_research",
        [
            py,
            "run_parameter_research.py",
            "--start",
            args.start,
            "--end",
            args.end,
            "--reference-dir",
            args.reference_dir,
            "--profile",
            args.profile,
            "--strict-reference",
        ],
        root,
        steps,
    ):
        return finish(root, steps)

    if not run(
        "v4_walk_forward",
        [
            py,
            "run_walk_forward.py",
            "--start",
            args.start,
            "--end",
            args.end,
            "--reference-dir",
            args.reference_dir,
            "--strict-reference",
        ],
        root,
        steps,
    ):
        return finish(root, steps)

    if not run(
        "v4_5_stress",
        [
            py,
            "run_stress_tests.py",
            "--start",
            args.start,
            "--end",
            args.end,
            "--reference-dir",
            args.reference_dir,
            "--strategy-config",
            "output/v3_research/best_config.json",
            "--strict-reference",
        ],
        root,
        steps,
    ):
        return finish(root, steps)

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
    accepted = run("v5_acceptance", acceptance_cmd, root, steps)
    if not accepted and args.continue_on_grade_fail:
        steps[-1]["passed"] = True
        steps[-1]["note"] = "Grade below requested threshold; pipeline continued by request."
    return finish(root, steps)


def finish(root: Path, steps: list[dict]) -> int:
    passed = all(bool(x.get("passed")) for x in steps)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "steps": steps,
    }
    out = root / "output" / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== PIPELINE RESULT ==========")
    print("PASS" if passed else "FAIL")
    print(out / "pipeline_summary.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
