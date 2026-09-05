import pandas as pd

from qmt_quant.factor_selection import duplicate_observation_groups, learn_factor_orientations


def _observations():
    rows = []
    for factor, sign in [("good", 1.0), ("bad", -1.0), ("dup", 1.0)]:
        for horizon in (5, 20):
            for i, date in enumerate(pd.date_range("2020-01-01", periods=30, freq="7D")):
                value = sign * (0.04 + 0.001 * i)
                rows.append(
                    {
                        "factor": factor,
                        "horizon": horizon,
                        "date": date,
                        "rank_ic": value,
                        "top_bottom_spread": value / 10.0,
                    }
                )
    return pd.DataFrame(rows)


def test_learns_positive_and_inverse_orientations_from_training_only():
    observations = _observations()
    learned = learn_factor_orientations(observations, end="2020-12-31")
    mapping = learned.set_index("factor")["orientation"].to_dict()
    assert mapping["good"] == 1
    assert mapping["bad"] == -1


def test_duplicate_observation_groups_detects_identical_research_series():
    observations = _observations()
    groups = duplicate_observation_groups(observations)
    assert any(set(group) == {"good", "dup"} for group in groups)


def test_future_rows_do_not_change_training_orientation():
    observations = _observations()
    future = observations.loc[observations["factor"] == "good"].copy()
    future["date"] = pd.Timestamp("2025-01-01")
    future["rank_ic"] = -1.0
    combined = pd.concat([observations, future], ignore_index=True)
    learned = learn_factor_orientations(combined, end="2020-12-31")
    assert learned.set_index("factor").loc["good", "orientation"] == 1
