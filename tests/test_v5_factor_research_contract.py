from pathlib import Path

import numpy as np
import pandas as pd

from qmt_quant.factors import V5FactorConfig, iter_v5_raw_factors
from qmt_quant.workflow_contract import env_value, load_workflow, normalized_run, step, structured_text


def test_streamed_factor_engine_exposes_expected_research_factors():
    idx = pd.bdate_range("2020-01-01", periods=180)
    x = np.arange(len(idx), dtype=float)
    close = pd.DataFrame(
        {"AAA.SZ": 10.0 + x * 0.05, "BBB.SH": 12.0 + x * 0.02},
        index=idx,
    )
    amount = pd.DataFrame(
        {"AAA.SZ": 50_000_000.0 + x, "BBB.SH": 40_000_000.0 + x},
        index=idx,
    )
    benchmark = pd.Series(100.0 + x * 0.01, index=idx)

    names = [name for name, _ in iter_v5_raw_factors(close, amount, benchmark, V5FactorConfig())]
    assert names == [
        "momentum_20_5",
        "momentum_60_5",
        "relative_strength_60_5",
        "residual_relative_strength_60_5",
        "momentum_120_5",
        "trend_quality",
        "trend_persistence",
        "low_volatility",
        "low_downside_risk",
        "liquidity",
        "liquidity_stability",
        "short_reversal",
    ]


def test_v5_workflow_reuses_frozen_shards_and_keeps_strict_guards():
    workflow = load_workflow(Path(".github/workflows/v5-factor-research.yml"))
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "SOURCE_RUN_ID") == "33811845110"
    assert env_value(workflow, "RECOVERY_RUN_ID") == "33887254974"
    assert env_value(workflow, "QMT_QUANT_CACHE_ONLY") == "1"
    for name in (
        "Require exactly one complete set of 20 shard artifacts",
        "Remove stale shard 13 from source run",
        "Download recovered shard 13",
    ):
        assert step(workflow, "factor-research", name)
    audit = normalized_run(
        workflow, "factor-research", "Revalidate full historical data before V5 research"
    )
    assert "--min-symbol-coverage 0.98" in audit
    assert "--min-session-coverage 0.97" in audit
    semantic = structured_text(workflow)
    assert "prepare_free_data_shard.py" not in semantic
    assert "prepare_free_data.py" not in semantic
