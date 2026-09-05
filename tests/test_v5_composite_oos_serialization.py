from __future__ import annotations

import json

import numpy as np

from run_v5_composite_oos import _native_int_list


def test_native_int_list_makes_numpy_horizons_json_serializable():
    horizons = _native_int_list(np.array([5, 20, 60], dtype=np.int64))
    assert horizons == [5, 20, 60]
    assert all(type(value) is int for value in horizons)
    assert json.dumps({"factor_horizons": horizons}) == '{"factor_horizons": [5, 20, 60]}'
