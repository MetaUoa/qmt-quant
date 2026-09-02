import pandas as pd

from qmt_quant.reference_data import ReferenceData


def test_point_in_time_membership_st_and_limits():
    calendar = pd.bdate_range("2018-01-01", "2020-12-31")
    basic = pd.DataFrame(
        [
            {"ts_code": "AAA.SZ", "exchange": "SZSE", "list_date": "20180101", "delist_date": None},
            {"ts_code": "BBB.SH", "exchange": "SSE", "list_date": "20190102", "delist_date": "20191231"},
            {"ts_code": "BJE.BJ", "exchange": "BSE", "list_date": "20180101", "delist_date": None},
        ]
    )
    st = pd.DataFrame([{"trade_date": "20190603", "ts_code": "BBB.SH", "name": "ST BBB"}])
    limits = pd.DataFrame(
        [
            {
                "trade_date": "20190604",
                "ts_code": "AAA.SZ",
                "pre_close": 10.0,
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
        ]
    )
    ref = ReferenceData(basic, calendar, st, limits)

    assert "BJE.BJ" not in ref.codes_ever_active("20180101", "20201231")
    assert ref.is_member("AAA.SZ", "20190603", min_listing_sessions=120)
    assert not ref.is_member("BBB.SH", "20181231", min_listing_sessions=0)
    assert not ref.is_member("BBB.SH", "20200102", min_listing_sessions=0)
    assert ref.is_st("BBB.SH", "20190603")
    assert ref.limit_blocked("AAA.SZ", "20190604", 1.10, "BUY")
    assert ref.limit_blocked("AAA.SZ", "20190604", 0.90, "SELL")
    assert not ref.limit_blocked("AAA.SZ", "20190604", 1.05, "BUY")
