from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Mapping, Sequence


class FreshnessError(RuntimeError):
    """Raised when a live broker or quote observation is stale or has an unknown clock."""


@dataclass(frozen=True)
class FreshnessPolicy:
    quote_max_age_seconds: float = 5.0
    query_max_duration_seconds: float = 5.0
    clock_skew_tolerance_seconds: float = 2.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise FreshnessError("timestamp must include timezone information")
    return value.astimezone(timezone.utc)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise FreshnessError(f"{name} cannot be boolean")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise FreshnessError(f"{name} is empty")
        try:
            number = float(text)
        except ValueError as exc:
            raise FreshnessError(f"{name} is not numeric") from exc
    else:
        raise FreshnessError(f"{name} is not numeric")
    if not math.isfinite(number):
        raise FreshnessError(f"{name} must be finite")
    return number


def _tick_timestamp(tick: Mapping[str, object]) -> datetime:
    raw = tick.get("time")
    if raw not in (None, ""):
        number = _finite_float(raw, name="tick time")
        if number <= 0:
            raise FreshnessError("tick time must be a positive finite epoch timestamp")
        seconds = number / 1000.0 if number >= 100_000_000_000 else number
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise FreshnessError("tick time is outside the supported timestamp range") from exc

    text = str(tick.get("stime") or tick.get("timetag") or "").strip()
    if not text:
        raise FreshnessError("tick does not expose time/stime/timetag")
    for fmt in ("%Y%m%d %H:%M:%S.%f", "%Y%m%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            china_tz = timezone(timedelta(hours=8))
            return parsed.replace(tzinfo=china_tz).astimezone(timezone.utc)
        except ValueError:
            continue
    raise FreshnessError("tick stime/timetag has an unsupported format")


def validate_tick_freshness(
    ticks: Mapping[str, Mapping[str, object]],
    codes: Sequence[str],
    *,
    policy: FreshnessPolicy,
    now: datetime | None = None,
) -> dict[str, object]:
    policy.validate()
    observed_now = _aware_utc(now or utc_now())
    rows: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    for code in list(dict.fromkeys(str(item) for item in codes)):
        tick = ticks.get(code)
        if tick is None:
            violation: dict[str, object] = {"code": code, "reason": "missing_tick"}
            violations.append(violation)
            rows.append(violation)
            continue
        try:
            timestamp = _tick_timestamp(tick)
            age_seconds = (observed_now - timestamp).total_seconds()
            row: dict[str, object] = {
                "code": code,
                "tick_timestamp_utc": timestamp.isoformat(),
                "age_seconds": float(age_seconds),
            }
            reasons: list[str] = []
            if age_seconds < -float(policy.clock_skew_tolerance_seconds):
                reasons.append("tick_from_future")
            if age_seconds > float(policy.quote_max_age_seconds):
                reasons.append("stale_tick")
            if reasons:
                row["reasons"] = reasons
                violations.append(dict(row))
            rows.append(row)
        except FreshnessError as exc:
            violation = {
                "code": code,
                "reason": "unknown_tick_clock",
                "error": str(exc),
            }
            violations.append(violation)
            rows.append(violation)
    return {
        "passed": not violations,
        "checked_at_utc": observed_now.isoformat(),
        "policy": asdict(policy),
        "quotes": rows,
        "violations": violations,
    }


def validate_query_window(
    *,
    source: str,
    started_at: datetime,
    completed_at: datetime,
    policy: FreshnessPolicy,
    now: datetime | None = None,
) -> dict[str, object]:
    policy.validate()
    started = _aware_utc(started_at)
    completed = _aware_utc(completed_at)
    observed_now = _aware_utc(now or utc_now())
    duration = (completed - started).total_seconds()
    age = (observed_now - completed).total_seconds()
    violations: list[str] = []
    if duration < 0:
        violations.append("clock_moved_backwards_during_query")
    if duration > float(policy.query_max_duration_seconds):
        violations.append("query_too_slow")
    if age < -float(policy.clock_skew_tolerance_seconds):
        violations.append("query_completed_in_future")
    if age > float(policy.query_max_duration_seconds):
        violations.append("query_result_too_old")
    return {
        "source": str(source),
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "checked_at_utc": observed_now.isoformat(),
        "duration_seconds": float(duration),
        "age_seconds": float(age),
        "passed": not violations,
        "violations": violations,
        "policy": asdict(policy),
    }


def require_fresh(report: Mapping[str, object], *, label: str) -> None:
    if report.get("passed") is not True:
        raise FreshnessError(f"{label} freshness gate failed: {report.get('violations')}")
