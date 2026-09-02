from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qmt_quant.acceptance import grade_strategy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V5 final strategy acceptance grading")
    p.add_argument("--backtest", default="output/v3_research/best_full_result/metrics.json")
    p.add_argument("--walk-forward", default="output/walk_forward/walk_forward_metrics.json")
    p.add_argument("--folds", default="output/walk_forward/walk_forward_folds.csv")
    p.add_argument("--stress", default="output/v4_5_stress/stress_summary.json")
    p.add_argument("--output", default="output/v5_acceptance")
    p.add_argument("--require-grade", choices=["A", "B", "C"], default="C")
    return p.parse_args()


def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    backtest = _load_json(args.backtest)
    oos = _load_json(args.walk_forward)
    stress = _load_json(args.stress)
    folds = pd.read_csv(args.folds) if Path(args.folds).exists() else pd.DataFrame()
    report = grade_strategy(backtest, oos, folds, stress)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "acceptance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {"gate": k, "passed": v, "grade": "A"} for k, v in report["grade_a_checks"].items()
        ]
        + [{"gate": k, "passed": v, "grade": "B"} for k, v in report["grade_b_checks"].items()]
        + [{"gate": k, "passed": v, "grade": "C"} for k, v in report["grade_c_checks"].items()]
    ).to_csv(out / "acceptance_gates.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    rank = {"REJECT": 0, "C": 1, "B": 2, "A": 3}
    required = rank[args.require_grade]
    observed = rank.get(report["grade"], 0)
    return 0 if observed >= required else 2


if __name__ == "__main__":
    raise SystemExit(main())
