from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_step(name: str, command: list[str], cwd: Path) -> dict:
    print(f"\n=== {name} ===")
    print("$", " ".join(command))
    proc = subprocess.run(command, cwd=cwd, text=True)
    return {"name": name, "command": command, "returncode": int(proc.returncode), "passed": proc.returncode == 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated verification for QMT Quant V2")
    parser.add_argument("--qmt-smoke", action="store_true", help="Also test local xtquant/QMT + reference-data availability")
    parser.add_argument("--reference-dir", default="data/reference")
    parser.add_argument("--qmt-smoke-symbols", type=int, default=20, help="Historical symbols used by the real QMT end-to-end smoke")
    parser.add_argument("--qmt-smoke-min-coverage", type=float, default=0.95)
    parser.add_argument("--no-coverage", action="store_true", help="Skip pytest coverage reporting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    report_dir = root / "output" / "test_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict] = []
    steps.append(run_step("compileall", [sys.executable, "-m", "compileall", "-q", "."], root))
    if not steps[-1]["passed"]:
        return _finish(report_dir, steps)

    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"--junitxml={report_dir / 'junit.xml'}",
    ]
    if not args.no_coverage:
        pytest_cmd += [
            "--cov=qmt_quant",
            "--cov-report=term-missing",
            f"--cov-report=json:{report_dir / 'coverage.json'}",
            "--cov-fail-under=60",
        ]
    steps.append(run_step("offline_pytest", pytest_cmd, root))

    if args.qmt_smoke and steps[-1]["passed"]:
        steps.append(
            run_step(
                "qmt_environment_smoke",
                [sys.executable, "check_qmt_env.py", "--reference-dir", args.reference_dir],
                root,
            )
        )
        if steps[-1]["passed"]:
            smoke_out = report_dir / "qmt_real_smoke"
            steps.append(
                run_step(
                    "qmt_real_backtest_smoke",
                    [
                        sys.executable,
                        "run_backtest.py",
                        "--start",
                        "20180101",
                        "--end",
                        "20251231",
                        "--reference-dir",
                        args.reference_dir,
                        "--max-stocks",
                        str(args.qmt_smoke_symbols),
                        "--download",
                        "--strict-reference",
                        "--min-symbol-coverage",
                        str(args.qmt_smoke_min_coverage),
                        "--output",
                        str(smoke_out),
                    ],
                    root,
                )
            )
            if steps[-1]["passed"]:
                steps.append(
                    run_step(
                        "qmt_backtest_output_validation",
                        [
                            sys.executable,
                            "validate_backtest_output.py",
                            str(smoke_out),
                            "--min-symbol-coverage",
                            str(args.qmt_smoke_min_coverage),
                        ],
                        root,
                    )
                )

    return _finish(report_dir, steps)


def _finish(report_dir: Path, steps: list[dict]) -> int:
    passed = all(step["passed"] for step in steps)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "steps": steps,
    }
    with (report_dir / "test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print("\n=== AUTOMATED TEST RESULT ===")
    print("PASS" if passed else "FAIL")
    print(f"Report: {report_dir / 'test_summary.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
