"""示例策略 1：简单截面动量。仅用于验证框架链路，不构成投资建议。"""

from __future__ import annotations

import pandas as pd

from .base import Strategy, register


@register("momentum")
class MomentumStrategy(Strategy):
    """买入过去 lookback 日收益最高的 top_n 只，等权。"""

    def target_weights(
        self,
        history: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        current_weights: pd.Series,
    ) -> pd.Series:
        lookback = int(self.params.get("lookback", 20))
        top_n = int(self.params.get("top_n", 5))

        close = history["close"]
        if len(close) <= lookback:
            return pd.Series(dtype=float)

        mom = close.iloc[-1] / close.iloc[-lookback - 1] - 1.0
        mom = mom.dropna()
        mom = mom[mom > 0]  # 只做多正动量
        if mom.empty:
            return pd.Series(dtype=float)

        picks = mom.nlargest(top_n)
        w = 1.0 / len(picks)
        return pd.Series(w, index=picks.index)
