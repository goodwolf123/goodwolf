"""股票池（universe）模块：按上市天数、ST、成交额、流通市值等条件过滤。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data.loader import MarketData


@dataclass
class UniverseFilter:
    min_list_days: int = 0          # 最少上市天数（剔除次新股）
    exclude_st: bool = False        # 剔除 ST
    min_amount: float = 0.0         # 近 lookback 日日均成交额下限（元）
    min_float_mktcap: float = 0.0   # 流通市值下限（元）
    amount_lookback: int = 20

    @classmethod
    def from_dict(cls, d: dict) -> "UniverseFilter":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def filter_universe(
    market: MarketData, filt: UniverseFilter, as_of: pd.Timestamp | None = None
) -> list[str]:
    """按过滤条件返回可用股票池。as_of 默认为数据最后一天（静态池）。

    说明：demo/骨架层面做静态过滤；更精细的"逐日动态池"可在策略/因子层
    结合 flags 面板实现（清洗层的 drop_untradable 即按日处理）。
    """
    as_of = as_of or market.dates[-1]
    keep: list[str] = []

    list_dates = market.meta.set_index("symbol")["list_date"]
    amount = market.panels["amount"].loc[:as_of].tail(filt.amount_lookback).mean()
    mktcap = market.panels["float_mktcap"].loc[:as_of].ffill().iloc[-1]
    is_st = market.flags["is_st"].loc[:as_of].iloc[-1]

    for sym in market.symbols:
        if filt.min_list_days > 0:
            ld = list_dates.get(sym)
            if pd.notna(ld):
                if (as_of - ld).days < filt.min_list_days:
                    continue
            # list_date 未知（真实源未接入）时不剔除，避免误伤
        if filt.exclude_st and bool(is_st.get(sym, False)):
            continue
        if filt.min_amount > 0 and float(amount.get(sym, 0.0)) < filt.min_amount:
            continue
        if filt.min_float_mktcap > 0:
            mc = float(mktcap.get(sym, float("nan")))
            if not (mc >= filt.min_float_mktcap):  # NaN 不剔除（字段缺失时不生效）
                continue
        keep.append(sym)
    return keep
