from pathlib import Path


def test_backtest_routes_target_and_cash_sizing_through_pure_helpers() -> None:
    text = Path('qmt_quant/backtest.py').read_text(encoding='utf-8')
    assert 'equal_weight_target_shares(' in text
    assert 'affordable_buy_quantity(' in text
    assert 'target_value = (' not in text
    assert 'while qty >= lot:' not in text
    assert 'deterministic_fill(cost, ts, code, "SELL")' in text
    assert 'deterministic_fill(cost, ts, code, "BUY")' in text
    assert text.index('for code in list(positions):') < text.index('for code in selected:')
