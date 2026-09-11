from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


def execution_batch_id(
    *,
    signal_date: str,
    strategy_sha256: str,
    target_weights: Mapping[str, float],
) -> str:
    payload = {
        "signal_date": str(signal_date),
        "strategy_sha256": str(strategy_sha256),
        "target_weights": {
            str(code): float(target_weights[code]) for code in sorted(target_weights)
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reserve_execution_batch(
    root: str | Path,
    *,
    batch_id: str,
    metadata: Mapping[str, object],
) -> Path:
    """Atomically reserve one live batch.

    The marker is deliberately persistent after success or failure. Re-running the
    exact strategy/signal/target bundle therefore requires explicit operator review
    rather than silently submitting the same batch twice.
    """
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{batch_id}.json"
    payload = {
        "batch_id": batch_id,
        "status": "RESERVED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **dict(metadata),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(path), flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"execution batch {batch_id} already exists; reconcile it before any retry"
        ) from exc
    try:
        raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_parent(path)
    return path


def update_execution_batch(
    path: str | Path,
    *,
    status: str,
    details: Mapping[str, object] | None = None,
) -> None:
    target = Path(path)
    if not target.exists():
        raise RuntimeError(f"execution batch marker is missing: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("execution batch marker is invalid")
    payload["status"] = str(status)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if details:
        payload["details"] = dict(details)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    _fsync_parent(target)
