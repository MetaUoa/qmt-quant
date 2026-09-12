from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from .acceptance_lineage import ACCEPTANCE_SCHEMA, validate_acceptance_lineage


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidatedTargets:
    frame: pd.DataFrame
    diagnostics: dict
    signal_date: date
    expected_execution_session: date
    expires_after_session: date
    strategy_sha256: str
    target_file_sha256: str


def china_market_date() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def next_trading_session(calendar: pd.DatetimeIndex, signal_date: date | str | pd.Timestamp) -> date:
    sessions = pd.DatetimeIndex(calendar).normalize().drop_duplicates().sort_values()
    signal = pd.Timestamp(signal_date).normalize()
    index = int(sessions.searchsorted(signal, side="right"))
    if index >= len(sessions):
        raise RuntimeError(f"no trading session exists after signal_date={signal.date()}")
    return pd.Timestamp(sessions[index]).date()


def _required_date(value: object, *, name: str) -> date:
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"signal diagnostics missing {name}")
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise RuntimeError(f"signal diagnostics contain invalid {name}")
    return pd.Timestamp(timestamp).normalize().date()


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

    raw_target_bytes = targets_file.read_bytes()
    target_file_sha256 = hashlib.sha256(raw_target_bytes).hexdigest()
    frame = pd.read_csv(targets_file)
    required = {"code", "target_weight"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"target file missing columns: {', '.join(missing)}")
    if frame["code"].isna().any() or frame["code"].astype(str).str.strip().eq("").any():
        raise ValueError("target file contains missing or blank codes")
    codes = frame["code"].astype(str)
    if codes.duplicated().any():
        raise ValueError("target file contains duplicate codes")
    weights = pd.to_numeric(frame["target_weight"], errors="coerce")
    if weights.isna().any() or not weights.astype(float).map(math.isfinite).all():
        raise ValueError("target weights must be finite values in [0, 1]")
    numeric_weights = weights.astype(float)
    if (numeric_weights < 0.0).any() or (numeric_weights > 1.0).any():
        raise ValueError("target weights must be finite values in [0, 1]")
    if len(frame) and float(numeric_weights.sum()) > 1.000001:
        raise ValueError("target weights exceed 100% gross exposure")

    diagnostics = json.loads(diagnostics_file.read_text(encoding="utf-8"))
    if not isinstance(diagnostics, Mapping):
        raise ValueError("signal diagnostics must be a JSON object")
    signal_date = _required_date(diagnostics.get("signal_date"), name="signal_date")
    expected_execution_session = _required_date(
        diagnostics.get("expected_execution_session", diagnostics.get("signal_date")),
        name="expected_execution_session",
    )
    expires_after_session = _required_date(
        diagnostics.get(
            "expires_after_session",
            diagnostics.get("expected_execution_session", diagnostics.get("signal_date")),
        ),
        name="expires_after_session",
    )
    if expected_execution_session <= signal_date:
        raise RuntimeError("expected_execution_session must be after signal_date")
    if expires_after_session < expected_execution_session:
        raise RuntimeError("expires_after_session must not precede expected_execution_session")

    market_date = china_market_date()
    if signal_date > market_date:
        raise RuntimeError(f"future signal_date {signal_date} is invalid for market date {market_date}")
    if require_current_session and not (
        expected_execution_session <= market_date <= expires_after_session
    ):
        raise RuntimeError(
            "live targets are outside their execution window: "
            f"signal_date={signal_date} expected_execution_session={expected_execution_session} "
            f"expires_after_session={expires_after_session} market_date={market_date}"
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
        for column in (
            "signal_date",
            "expected_execution_session",
            "expires_after_session",
            "strategy_sha256",
        ):
            if column not in frame.columns:
                raise RuntimeError(f"live target CSV requires {column} column")
        csv_shas = sorted(set(frame["strategy_sha256"].dropna().astype(str)))
        if csv_shas != [strategy_sha256]:
            raise RuntimeError("target CSV strategy SHA256 does not match signal diagnostics")
        date_columns = {
            "signal_date": signal_date,
            "expected_execution_session": expected_execution_session,
            "expires_after_session": expires_after_session,
        }
        for column, expected_date in date_columns.items():
            csv_dates = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
            if csv_dates.isna().any() or not csv_dates.eq(pd.Timestamp(expected_date)).all():
                raise RuntimeError(f"target CSV {column} does not match signal diagnostics")
    else:
        strategy_sha256 = str(source.get("sha256", "")) if isinstance(source, Mapping) else ""

    return ValidatedTargets(
        frame=frame,
        diagnostics=dict(diagnostics),
        signal_date=signal_date,
        expected_execution_session=expected_execution_session,
        expires_after_session=expires_after_session,
        strategy_sha256=strategy_sha256,
        target_file_sha256=target_file_sha256,
    )


def validate_acceptance_for_strategy(path: str | Path, minimum: str, strategy_sha256: str) -> dict:
    source = Path(path)
    if not source.exists():
        raise RuntimeError(f"Acceptance report missing: {source}")
    report = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise RuntimeError("acceptance report must be a JSON object")
    if str(report.get("schema", "")) != ACCEPTANCE_SCHEMA:
        raise RuntimeError(f"live acceptance requires schema {ACCEPTANCE_SCHEMA}")

    lineage = report.get("lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError("live acceptance requires content-addressed lineage binding")
    validated_lineage = validate_acceptance_lineage(
        lineage,
        strategy_sha256=strategy_sha256,
    )
    top_level_evidence = report.get("evidence_sha256")
    top_level_artifacts = report.get("artifact_sha256")
    if not isinstance(top_level_evidence, Mapping) or not isinstance(top_level_artifacts, Mapping):
        raise RuntimeError("live acceptance requires evidence and artifact SHA256 metadata")
    if {str(k): str(v) for k, v in top_level_evidence.items()} != validated_lineage["evidence_sha256"]:
        raise RuntimeError("acceptance evidence SHA256 does not match lineage binding")
    if {str(k): str(v) for k, v in top_level_artifacts.items()} != validated_lineage["artifact_sha256"]:
        raise RuntimeError("acceptance artifact SHA256 does not match lineage binding")

    top_level_run_id = str(report.get("run_id", ""))
    if top_level_run_id != validated_lineage["run_id"]:
        raise RuntimeError("acceptance run_id does not match immutable lineage run manifest")
    top_level_manifest = report.get("run_manifest")
    if not isinstance(top_level_manifest, Mapping):
        raise RuntimeError("live acceptance requires top-level immutable run_manifest")
    if dict(top_level_manifest) != validated_lineage["run_manifest"]:
        raise RuntimeError("acceptance top-level run_manifest does not match lineage binding")

    rank = {"REJECT": 0, "C": 1, "B": 2, "A": 3}
    grade = str(report.get("grade", "REJECT"))
    if rank.get(grade, 0) < rank[minimum]:
        raise RuntimeError(f"Strategy grade {grade} is below live minimum {minimum}")
    observed_sha = str(report.get("strategy_sha256", ""))
    if not strategy_sha256 or observed_sha != strategy_sha256:
        raise RuntimeError("acceptance report is not bound to the exact target strategy SHA256")
    return dict(report)
