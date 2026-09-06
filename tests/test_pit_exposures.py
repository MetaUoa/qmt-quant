from pathlib import Path

import numpy as np
import pandas as pd

from qmt_quant.pit_exposures import (
    asof_industry_panel,
    turnover_implied_float_market_cap,
)
from qmt_quant.workflow_contract import (
    env_value,
    load_workflow,
    matrix_values,
    max_parallel,
    normalized_run,
    structured_text,
)


def test_turnover_implied_float_market_cap_uses_same_day_fields():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "close": [10.0, 11.0],
            "volume": [1_000_000.0, 0.0],
            "turn": [2.0, 0.0],
        }
    )
    out = turnover_implied_float_market_cap(frame)
    assert np.isclose(out.loc[0, "float_shares_implied"], 50_000_000.0)
    assert np.isclose(out.loc[0, "float_market_cap"], 500_000_000.0)
    assert pd.isna(out.loc[1, "float_market_cap"])
    assert out.loc[0, "exposure_source"] == "baostock_turnover_implied_float_cap"


def test_industry_panel_only_forward_fills_from_past_snapshots():
    snapshots = pd.DataFrame(
        {
            "asof_date": ["2024-01-02", "2024-02-01"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "industry": ["Bank", "Finance"],
        }
    )
    dates = pd.DatetimeIndex(["2024-01-15", "2024-02-05"])
    panel = asof_industry_panel(snapshots, dates, ["000001.SZ"])
    assert panel.loc[pd.Timestamp("2024-01-15"), "000001.SZ"] == "Bank"
    assert panel.loc[pd.Timestamp("2024-02-05"), "000001.SZ"] == "Finance"


def test_pit_exposure_workflow_preserves_sharding_and_baostock_pin():
    workflow = load_workflow(Path(".github/workflows/v5-pit-exposures.yml"))
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert max_parallel(workflow, "exposure-shard") == 5
    assert matrix_values(workflow, "exposure-shard", "shard") == [str(i) for i in range(20)]
    install = normalized_run(workflow, "exposure-shard", "Install exposure dependencies")
    exposure = normalized_run(
        workflow, "exposure-shard", "Download deterministic PIT float-cap sidecar"
    )
    industry = normalized_run(workflow, "industry", "Download monthly PIT industry snapshots")
    assert "baostock==0.9.3" in install
    assert "prepare_pit_exposure_shard.py" in exposure
    assert "prepare_pit_industry.py" in industry
    assert "prepare_free_data.py" not in structured_text(workflow)
