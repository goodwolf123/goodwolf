"""示例策略 2：双均线（快线上穿慢线持有）。仅用于验证框架链路。"""

from __future__ import annotations

import pandas as pd

from .base import Strategy, register


@register("ma_cross")
class MACrossStrategy(Strategy):
    """收盘价快线 > 慢线则持有该标的，等权分配。"""

    def target_weights(
        self,
        history: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        current_weights: pd.Series,
    ) -> pd.Series:
        fast = int(self.params.get("fast", 5))
        slow = int(self.params.get("slow", 20))

        close = history["close"]
        if len(close) < slow + 1:
            return pd.Series(dtype=float)

        ma_fast = close.rolling(fast).mean().iloc[-1]
        ma_slow = close.rolling(slow).mean().iloc[-1]
        signal = (ma_fast > ma_slow).dropna()
        picks = signal[signal].index
        if len(picks) == 0:
            return pd.Series(dtype=float)

        w = 1.0 / len(picks)
        return pd.Series(w, index=picks)
