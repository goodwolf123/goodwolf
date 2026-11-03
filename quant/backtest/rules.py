"""交易规则：分板块涨跌停幅度、整手等。"""

from __future__ import annotations

import pandas as pd

from ..data.schema import board_of

# 板块默认涨跌停幅度
BOARD_LIMIT_PCT = {"MAIN": 0.10, "CYB": 0.20, "KCB": 0.20, "BJ": 0.30}
ST_LIMIT_PCT = 0.05
LOT_SIZE = 100  # A股买入整手


def limit_pct_for(symbol: str, is_st: bool = False) -> float:
    """单标的涨跌停幅度：ST 5%，创业板/科创板 20%，北交所 30%，主板 10%。"""
    if is_st:
        return ST_LIMIT_PCT
    return BOARD_LIMIT_PCT[board_of(symbol)]


def limit_pct_panel(symbols: list[str], is_st: pd.DataFrame) -> pd.DataFrame:
    """生成 (date x symbol) 的涨跌停幅度面板，随 ST 状态逐日变化。"""
    base = pd.Series({s: BOARD_LIMIT_PCT[board_of(s)] for s in symbols})
    panel = pd.DataFrame(
        {s: base[s] for s in symbols}, index=is_st.index
    )
    panel[is_st.fillna(False).astype(bool)] = ST_LIMIT_PCT
    return panel


def round_lot(shares: pd.Series | float, lot: int = LOT_SIZE) -> pd.Series | float:
    """买入股数向下取整到手。"""
    return (shares // lot) * lot
