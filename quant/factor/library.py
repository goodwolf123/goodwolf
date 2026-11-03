"""示例因子：动量 / 波动率 / 换手率。仅用于验证因子检验框架，不构成投资建议。"""

from __future__ import annotations

import pandas as pd

from .base import Factor, register_factor


@register_factor("momentum")
class MomentumFactor(Factor):
    """过去 lookback 日收益（跳过最近 skip 日）。"""

    def compute(self, market) -> pd.DataFrame:
        lookback = int(self.params.get("lookback", 20))
        skip = int(self.params.get("skip", 1))
        close = market.panels["close"].ffill()
        mom = close.shift(skip) / close.shift(lookback + skip) - 1.0
        return mom


@register_factor("volatility")
class VolatilityFactor(Factor):
    """过去 lookback 日日收益标准差（低波异象方向）。"""

    def compute(self, market) -> pd.DataFrame:
        lookback = int(self.params.get("lookback", 20))
        ret = market.panels["close"].ffill().pct_change()
        return ret.rolling(lookback).std()


@register_factor("turnover")
class TurnoverFactor(Factor):
    """过去 lookback 日平均换手率 ≈ 成交额 / 流通市值。"""

    def compute(self, market) -> pd.DataFrame:
        lookback = int(self.params.get("lookback", 20))
        amount = market.panels["amount"]
        mktcap = market.panels["float_mktcap"]
        daily_to = amount / mktcap.where(mktcap > 0)
        return daily_to.rolling(lookback).mean()
