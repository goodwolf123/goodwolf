"""测试交易日历与数据对齐。"""

from __future__ import annotations

import pandas as pd

from quant.data.calendar import get_calendar
from quant.data.demo import generate_demo_data
from quant.data.schema import to_panels, validate


def test_demo_calendar_is_bdays():
    cal = get_calendar("2024-01-01", "2024-01-31", source="demo")
    assert len(cal) == 23  # 2024-01 工作日数
    assert all(d.dayofweek < 5 for d in cal)


def test_calendar_fallback_for_unknown_source():
    cal = get_calendar("2024-01-01", "2024-01-10", source="akshare", cache_dir=None)
    # 无 akshare/网络时回退工作日近似
    assert len(cal) > 0


def test_market_dates_within_calendar():
    df, meta = generate_demo_data(["600000.SH", "300750.SZ"], "2024-01-01", "2024-06-30")
    df = validate(df)
    cal = get_calendar("2024-01-01", "2024-06-30", source="demo")
    data_dates = pd.DatetimeIndex(df["date"].unique())
    assert data_dates.isin(cal).all(), "数据日期必须落在交易日历内"


def test_meta_fields_complete():
    df, meta = generate_demo_data(["600000.SH", "300750.SZ", "688981.SH"], "2024-01-01", "2024-06-30")
    for col in ("symbol", "board", "list_date", "delist_date", "industry"):
        assert col in meta.columns
    boards = dict(zip(meta["symbol"], meta["board"]))
    assert boards["600000.SH"] == "MAIN"
    assert boards["300750.SZ"] == "CYB"
    assert boards["688981.SH"] == "KCB"


def test_demo_adjust_factor_split():
    """拆股标的的 adj_factor 应出现阶跃，hfq 复权后价格连续。"""
    from quant.data.demo import SPLIT_SYMBOL

    df, meta = generate_demo_data([SPLIT_SYMBOL], "2022-01-01", "2024-12-31")
    df = df[df["symbol"] == SPLIT_SYMBOL]
    factor = df.set_index("date")["adj_factor"]
    assert factor.nunique() == 2, "拆股应产生两个因子水平"

    panels = to_panels(df)
    ratio = factor.sort_index() / factor.sort_index().iloc[0]
    hfq_close = panels["close"][SPLIT_SYMBOL] * ratio.values
    ret = hfq_close.pct_change().abs()
    assert (ret.dropna() < 0.35).all(), "hfq 复权后不应出现拆股造成的跳空"
