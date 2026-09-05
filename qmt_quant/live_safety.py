from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Mapping
from zoneinfo import ZoneInfo
from datetime import datetime

import pandas as pd


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidatedTargets:
    frame: pd.DataFrame
    diagnostics: dict
    signal_date: date
    strategy_sha256: str


def china_market_date() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def validate_target_bundle(
    targets_path: str | Path,
    diagnostics_path: str | Path,
    *,
    require_current_session: bool,
) -> ValidatedTargets:
    targets_file = Path(targets_path)
    diagnostics_file = Path(diagnostics_path)
    if not targets_file.exists():
        raise FileNotFoundError(targets_file)
    if not diagnostics_file.exists():
        raise FileNotFoundError(diagnostics_file)

    frame = pd.read_csv(targets_file)
    required = {"code", "target_weight"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"target file missing columns: {', '.join(missing)}")
    codes = frame["code"].dropna().astype(str)
    if codes.duplicated().any():
        raise ValueError("target file contains duplicate codes")
    weights = pd.to_numeric(frame["target_weight"], errors="coerce")
    if weights.isna().any() or (weights < 0.0).any() or (weights > 1.0).any():
        raise ValueError("target weights must be finite values in [0, 1]")
    if len(frame) and float(weights.sum()) > 1.000001:
        raise ValueError("target weights exceed 100% gross exposure")

    diagnostics = json.loads(diagnostics_file.read_text(encoding="utf-8"))
    if not isinstance(diagnostics, Mapping):
        raise ValueError("signal diagnostics must be a JSON object")
    raw_signal_date = diagnostics.get("signal_date")
    if not raw_signal_date:
        raise RuntimeError("signal diagnostics missing signal_date")
    signal_ts = pd.Timestamp(raw_signal_date).normalize()
    if pd.isna(signal_ts):
        raise RuntimeError("signal diagnostics contain invalid signal_date")
    signal_date = signal_ts.date()
    market_date = china_market_date()
    if signal_date > market_date:
        raise RuntimeError(f"future signal_date {signal_date} is invalid for market date {market_date}")
    if require_current_session and signal_date != market_date:
        raise RuntimeError(
            f"stale live targets: signal_date={signal_date} market_date={market_date}"
        )

    selected_count = int(diagnostics.get("selected_count", len(frame)))
    if selected_count != len(frame):
        raise RuntimeError(
            f"target count mismatch: diagnostics selected_count={selected_count}, csv rows={len(frame)}"
        )

    source = diagnostics.get("strategy_source")
    if require_current_session:
        if not isinstance(source, Mapping):
            raise RuntimeError("live targets require fingerprinted strategy_source metadata")
        strategy_sha256 = str(source.get("sha256", ""))
        if not _SHA256_RE.fullmatch(strategy_sha256):
            raise RuntimeError("live targets require a valid strategy SHA256 fingerprint")
    else:
        strategy_sha256 = str(source.get("sha256", "")) if isinstance(source, Mapping) else ""

    return ValidatedTargets(
        frame=frame,
        diagnostics=dict(diagnostics),
        signal_date=signal_date,
        strategy_sha256=strategy_sha256,
    )


def validate_acceptance_for_strategy(path: str | Path, minimum: str, strategy_sha256: str) -> dict:
    source = Path(path)
    if not source.exists():
        raise RuntimeError(f"Acceptance report missing: {source}")
    report = json.loads(source.read_text(encoding="utf-8"))
    rank = {"REJECT": 0, "C": 1, "B": 2, "A": 3}
    grade = str(report.get("grade", "REJECT"))
    if rank.get(grade, 0) < rank[minimum]:
        raise RuntimeError(f"Strategy grade {grade} is below live minimum {minimum}")
    observed_sha = str(report.get("strategy_sha256", ""))
    if not strategy_sha256 or observed_sha != strategy_sha256:
        raise RuntimeError("acceptance report is not bound to the exact target strategy SHA256")
    return report
