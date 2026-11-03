"""测试分板块涨跌停规则与整手。"""

from __future__ import annotations

import pandas as pd

from quant.backtest.rules import limit_pct_for, limit_pct_panel, round_lot


def test_board_limits():
    assert limit_pct_for("600000.SH") == 0.10
    assert limit_pct_for("000001.SZ") == 0.10
    assert limit_pct_for("300750.SZ") == 0.20   # 创业板
    assert limit_pct_for("688981.SH") == 0.20   # 科创板
    assert limit_pct_for("830799.BJ") == 0.30   # 北交所


def test_st_limit_overrides_board():
    assert limit_pct_for("600000.SH", is_st=True) == 0.05
    assert limit_pct_for("300750.SZ", is_st=True) == 0.05


def test_limit_pct_panel_daily_st_change():
    import numpy as np

    dates = pd.bdate_range("2024-01-01", periods=3)
    is_st = pd.DataFrame({"600000.SH": [False, True, False]}, index=dates)
    panel = limit_pct_panel(["600000.SH"], is_st)
    assert panel["600000.SH"].tolist() == [0.10, 0.05, 0.10]


def test_round_lot():
    assert round_lot(13333.3) == 13300
    assert round_lot(99.9) == 0
    s = pd.Series([250.0, 1050.0])
    assert round_lot(s).tolist() == [200.0, 1000.0]
