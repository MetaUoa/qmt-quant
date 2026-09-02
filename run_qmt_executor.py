from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qmt_quant.live_trader import QmtBroker, build_equal_weight_plan, serialize_plan
from risk.pretrade import validate_pretrade


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6/V7 MiniQMT rebalance executor; dry-run by default")
    p.add_argument("--userdata", required=True, help="MiniQMT userdata_mini directory")
    p.add_argument("--account", required=True)
    p.add_argument("--account-type", default="STOCK")
    p.add_argument("--session-id", type=int, default=26090201)
    p.add_argument("--targets", default="output/live_targets/target_weights.csv")
    p.add_argument("--acceptance", default="output/v5_acceptance/acceptance_report.json")
    p.add_argument("--min-live-grade", choices=["A", "B", "C"], default="C")
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--output", default="output/live_execution")
    p.add_argument("--enable-live", action="store_true")
    p.add_argument("--confirm-live", default="", help="Live mode requires exact value LIVE")
    p.add_argument("--ignore-acceptance", action="store_true")
    return p.parse_args()


def check_acceptance(path: str, minimum: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Acceptance report missing: {p}")
    report = json.loads(p.read_text(encoding="utf-8"))
    rank = {"REJECT": 0, "C": 1, "B": 2, "A": 3}
    grade = str(report.get("grade", "REJECT"))
    if rank.get(grade, 0) < rank[minimum]:
        raise RuntimeError(f"Strategy grade {grade} is below live minimum {minimum}")
    return report


def main() -> int:
    args = parse_args()
    target_path = Path(args.targets)
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    targets = pd.read_csv(target_path)
    if "code" not in targets.columns:
        raise ValueError("Target file must contain code column")
    target_codes = [str(x) for x in targets["code"].dropna().tolist()]

    if args.enable_live:
        if args.confirm_live != "LIVE":
            raise RuntimeError("Live execution requires --confirm-live LIVE")
        if not args.ignore_acceptance:
            check_acceptance(args.acceptance, args.min_live_grade)

    broker = QmtBroker(args.userdata, args.account, args.session_id, args.account_type)
    broker.connect()
    total_asset, cash, positions = broker.snapshot()
    all_codes = list(dict.fromkeys(target_codes + list(positions)))
    ticks = broker.full_tick(all_codes)
    executable = broker.executable_prices(ticks)
    prices = {code: float(v["last"] or v["buy"] or v["sell"]) for code, v in executable.items()}
    plan = build_equal_weight_plan(
        target_codes,
        prices,
        positions,
        total_asset=total_asset,
        exposure=args.exposure,
        lot_size=100,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "total_asset": total_asset,
        "cash": cash,
        "position_count": len(positions),
        "target_count": len(target_codes),
        "dry_run": not args.enable_live,
    }
    (out / "pretrade_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(serialize_plan(plan)).to_csv(out / "order_plan.csv", index=False, encoding="utf-8-sig")
    risk_report = validate_pretrade(plan, total_asset=total_asset, target_count=len(target_codes))
    (out / "pretrade_risk.json").write_text(json.dumps(risk_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    print(json.dumps({"pretrade_risk": risk_report}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(serialize_plan(plan)).to_string(index=False) if plan else "No orders required")

    if not args.enable_live:
        print("DRY RUN: no orders were sent")
        return 0
    if not risk_report["passed"]:
        raise RuntimeError(f"Pre-trade risk gate failed: {risk_report['violations']}")

    results = broker.submit_plan(plan)
    pd.DataFrame(results).to_csv(out / "submitted_orders.csv", index=False, encoding="utf-8-sig")
    (out / "submitted_orders.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [x for x in results if x.get("status") == "FAILED"]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
