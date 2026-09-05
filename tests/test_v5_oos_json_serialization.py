import io
import json

import numpy as np

from run_v5_composite_oos_entry import _safe_json_dump


def test_safe_json_dump_serializes_numpy_scalars():
    stream = io.StringIO()
    _safe_json_dump(
        {
            "factor_horizons": [np.int64(5), np.int64(20)],
            "score": np.float64(0.125),
        },
        stream,
        ensure_ascii=False,
    )

    assert json.loads(stream.getvalue()) == {
        "factor_horizons": [5, 20],
        "score": 0.125,
    }
