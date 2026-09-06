from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from qmt_quant.acceptance import grade_strategy


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explicit-lineage strategy acceptance grading")
    p.add_argument("--backtest", required=True)
    p.add_argument("--walk-forward", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--stress", required=True)
    p.add_argument("--strategy-sha256", required=True)
    p.add_argument("--output", default="output/v5_acceptance")
    p.add_argument("--require-grade", choices=["A", "B", "C"], default="C")
    return p.parse_args()


def _load_json(path: str) -> dict:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {source}")
    return payload


def _require_strategy_sha(value: str) -> str:
    sha = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(sha):
        raise ValueError("--strategy-sha256 must be an exact lowercase 64-hex SHA256")
    return sha


def main() -> int:
    args = parse_args()
    strategy_sha256 = _require_strategy_sha(args.strategy_sha256)
    backtest = _load_json(args.backtest)
    oos = _load_json(args.walk_forward)
    stress = _load_json(args.stress)
    folds_path = Path(args.folds)
    if not folds_path.exists():
        raise FileNotFoundError(folds_path)
    folds = pd.read_csv(folds_path)
    report = grade_strategy(backtest, oos, folds, stress)
    report["strategy_sha256"] = strategy_sha256
    report["evidence"] = {
        "backtest": str(Path(args.backtest)),
        "walk_forward": str(Path(args.walk_forward)),
        "folds": str(folds_path),
        "stress": str(Path(args.stress)),
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "acceptance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"gate": k, "passed": v, "grade": "A"}
            for k, v in report["grade_a_checks"].items()
        ]
        + [
            {"gate": k, "passed": v, "grade": "B"}
            for k, v in report["grade_b_checks"].items()
        ]
        + [
            {"gate": k, "passed": v, "grade": "C"}
            for k, v in report["grade_c_checks"].items()
        ]
    ).to_csv(out / "acceptance_gates.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    rank = {"REJECT": 0, "C": 1, "B": 2, "A": 3}
    required = rank[args.require_grade]
    observed = rank.get(report["grade"], 0)
    return 0 if observed >= required else 2


if __name__ == "__main__":
    raise SystemExit(main())
