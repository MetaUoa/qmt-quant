from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class AlertRecord:
    severity: str
    code: str
    message: str
    details: dict

    def to_dict(self) -> dict:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            **asdict(self),
        }


class JsonlAlertSink:
    """Credential-free local alert channel.

    This deliberately does not send network messages or read Secrets.  An external
    operator/monitor can tail the JSONL file and forward alerts through a separately
    controlled channel later.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def emit(self, alert: AlertRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def runtime_health_alert(checks: dict) -> AlertRecord | None:
    if bool(checks.get("passed")):
        return None
    failed = sorted(
        key
        for key, value in checks.items()
        if key.endswith("_ok") or key.endswith("_passed") or key.endswith("_present")
        if value is False
    )
    return AlertRecord(
        severity="ERROR",
        code="runtime_health_failed",
        message="QMT runtime health check failed; keep live execution blocked until resolved",
        details={"failed_checks": failed},
    )
