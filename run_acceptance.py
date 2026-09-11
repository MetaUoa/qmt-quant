from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qmt_quant.acceptance import grade_strategy
from qmt_quant.acceptance_lineage import (
    ACCEPTANCE_SCHEMA,
    build_acceptance_lineage,
    require_sha256,
    sha256_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Content-addressed strategy acceptance grading")
    p.add_argument("--backtest", required=True)
    p.add_argument("--walk-forward", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--stress", required=True)
    p.add_argument("--strategy-sha256", required=True)
    p.add_argument("--strategy-source", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data-lineage", required=True)
    p.add_argument("--engine-manifest", required=True)
    p.add_argument("--dependency-lock", required=True)
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


def _sha256_path(path: str | Path) -> str:
    return sha256_path(path)


def _require_strategy_sha(value: str) -> str:
    return require_sha256(value, name="--strategy-sha256")


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

    evidence_paths = {
        "backtest": str(Path(args.backtest)),
        "walk_forward": str(Path(args.walk_forward)),
        "folds": str(folds_path),
        "stress": str(Path(args.stress)),
    }
    lineage_paths = {
        "strategy_source": str(Path(args.strategy_source)),
        "config": str(Path(args.config)),
        "data_lineage": str(Path(args.data_lineage)),
        "engine_manifest": str(Path(args.engine_manifest)),
        "dependency_lock": str(Path(args.dependency_lock)),
    }
    lineage = build_acceptance_lineage(
        strategy_sha256=strategy_sha256,
        evidence_paths=evidence_paths,
        artifact_paths=lineage_paths,
    )

    report = grade_strategy(backtest, oos, folds, stress)
    report["schema"] = ACCEPTANCE_SCHEMA
    report["strategy_sha256"] = strategy_sha256
    report["evidence"] = evidence_paths
    report["lineage_artifacts"] = lineage_paths
    report["lineage"] = lineage
    report["evidence_sha256"] = lineage["evidence_sha256"]

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
