from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from qmt_quant.config import DataConfig
from qmt_quant.qmt_data import get_sector_universe, load_daily_bars
from qmt_quant.reference_data import ReferenceData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="data/reference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = DataConfig(reference_dir=args.reference_dir)
    report = {
        "python": sys.version,
        "period": [cfg.start, cfg.end],
        "sector": cfg.sector,
        "benchmark": cfg.benchmark,
        "reference_dir": str(Path(cfg.reference_dir).resolve()),
    }
    try:
        from xtquant import xtdata  # noqa: F401

        report["xtquant_import"] = "ok"
    except Exception as exc:
        report["xtquant_import"] = f"failed: {exc}"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    try:
        universe = get_sector_universe(cfg.sector)
        report["current_qmt_universe_size"] = len(universe)
    except Exception as exc:
        report["universe_error"] = str(exc)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3

    try:
        reference = ReferenceData.from_dir(cfg.reference_dir)
        report["reference"] = reference.audit().__dict__
        report["historical_candidate_symbols"] = len(reference.codes_ever_active(cfg.start, cfg.end))
    except Exception as exc:
        report["reference_error"] = str(exc)
        report["reference"] = None

    sample = universe[:20]
    codes = list(dict.fromkeys(sample + [cfg.benchmark]))
    try:
        bars = load_daily_bars(codes, cfg.start, cfg.end, cfg.dividend_type, batch_size=50)
    except Exception as exc:
        report["load_error"] = str(exc)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 4

    coverage = {}
    for code in codes:
        frame = bars.get(code)
        if frame is None or frame.empty:
            coverage[code] = {"rows": 0, "start": None, "end": None}
            continue
        valid_close = pd.to_numeric(frame["close"], errors="coerce").dropna()
        coverage[code] = {
            "rows": int(len(valid_close)),
            "start": str(valid_close.index.min().date()) if len(valid_close) else None,
            "end": str(valid_close.index.max().date()) if len(valid_close) else None,
        }
    report["sample_coverage"] = coverage
    report["loaded_symbols"] = len(bars)
    report["benchmark_ok"] = bool(cfg.benchmark in bars and not bars[cfg.benchmark].empty)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["benchmark_ok"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
