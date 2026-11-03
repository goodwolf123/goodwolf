"""测试数据字段规范与绩效指标计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.analysis.metrics import compute_metrics, max_drawdown
from quant.data.schema import REQUIRED_COLUMNS, to_panels, validate


def good_df() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=5)
    rows = []
    for sym in ("A", "B"):
        for d in dates:
            rows.append({"date": d, "symbol": sym, "open": 10.0, "high": 10.5,
                         "low": 9.5, "close": 10.0, "volume": 1e6})
    return pd.DataFrame(rows)


def test_validate_accepts_good_data():
    df = validate(good_df())
    assert list(df.columns)[:7] == REQUIRED_COLUMNS or set(REQUIRED_COLUMNS) <= set(df.columns)
    assert df["adj_factor"].eq(1.0).all()


def test_validate_rejects_missing_column():
    df = good_df().drop(columns=["close"])
    with pytest.raises(ValueError, match="缺少必需列"):
        validate(df)


def test_validate_rejects_duplicates():
    df = pd.concat([good_df(), good_df().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="重复"):
        validate(df)


def test_panels_alignment():
    panels = to_panels(validate(good_df()))
    close = panels["close"]
    assert close.shape == (5, 2)
    assert list(close.columns) == ["A", "B"]
    # 各面板 index/columns 对齐
    for f in ("open", "high", "low", "volume"):
        assert panels[f].index.equals(close.index)
        assert panels[f].columns.equals(close.columns)


def test_max_drawdown():
    nav = pd.Series([1.0, 1.2, 0.9, 1.1, 1.0])
    mdd, dd = max_drawdown(nav)
    assert mdd == pytest.approx(0.9 / 1.2 - 1.0)
    assert dd.min() == pytest.approx(mdd)


def test_metrics_basic():
    nav = pd.Series(np.linspace(1.0, 1.1, 253))
    rets = nav.pct_change().fillna(0.0)
    m = compute_metrics(nav, rets, turnover_total=2.0)
    assert m["total_return"] == pytest.approx(0.1, abs=1e-9)
    assert m["annual_return"] > 0
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert m["turnover_annual"] == pytest.approx(2.0, rel=0.01)
