from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping


@dataclass(frozen=True)
class RuntimeRiskPolicy:
    max_intraday_drawdown: float = 0.05
    max_position_loss: float = 0.15
    blacklist_codes: frozenset[str] = field(default_factory=frozenset)
    kill_switch: bool = False


def evaluate_runtime_risk(
    *,
    start_of_day_equity: float,
    current_equity: float,
    target_codes: list[str] | tuple[str, ...],
    position_returns: Mapping[str, float] | None = None,
    policy: RuntimeRiskPolicy | None = None,
) -> dict:
    """Evaluate dynamic trading risk without creating any trading side effect.

    The result is intentionally fail-closed. Callers may refuse new orders when
    ``passed`` is false, but this function never submits, cancels, or liquidates.
    """
    cfg = policy or RuntimeRiskPolicy()
    violations: list[str] = []
    start = float(start_of_day_equity)
    current = float(current_equity)
    max_drawdown = float(cfg.max_intraday_drawdown)
    max_position_loss = abs(float(cfg.max_position_loss))

    if (
        not math.isfinite(start)
        or not math.isfinite(current)
        or start <= 0.0
        or current < 0.0
        or not math.isfinite(max_drawdown)
        or max_drawdown < 0.0
        or max_drawdown > 1.0
        or not math.isfinite(max_position_loss)
        or max_position_loss > 1.0
    ):
        violations.append("invalid_equity_state")
        drawdown = 1.0
    else:
        drawdown = max(0.0, 1.0 - current / start)
        if drawdown > max_drawdown:
            violations.append(f"intraday_drawdown_circuit_breaker:{drawdown:.6f}")

    if bool(cfg.kill_switch):
        violations.append("manual_kill_switch")

    blacklist = {str(code) for code in cfg.blacklist_codes}
    blacklisted_targets = sorted({str(code) for code in target_codes}.intersection(blacklist))
    for code in blacklisted_targets:
        violations.append(f"blacklisted_target:{code}")

    losses: dict[str, float] = {}
    invalid_position_returns: list[str] = []
    for code, value in (position_returns or {}).items():
        ret = float(value)
        if not math.isfinite(ret):
            invalid_position_returns.append(str(code))
            violations.append(f"invalid_position_return:{code}")
            continue
        if ret <= -max_position_loss:
            losses[str(code)] = ret
            violations.append(f"position_stop_loss:{code}:{ret:.6f}")

    return {
        "passed": not violations,
        "violations": violations,
        "intraday_drawdown": float(drawdown),
        "blacklisted_targets": blacklisted_targets,
        "stop_loss_positions": losses,
        "invalid_position_returns": invalid_position_returns,
        "policy": {
            **asdict(cfg),
            "blacklist_codes": sorted(blacklist),
        },
        "side_effects": "none",
    }
