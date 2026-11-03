"""单因子检验核心：IC/RankIC、统计量、分层回测、多空组合。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """h 日前瞻收益：close[t+h] / close[t] - 1。尾部 h 行为 NaN。"""
    return close.shift(-horizon) / close - 1.0


def ic_series(factor: pd.DataFrame, fwd_ret: pd.DataFrame, method: str = "rank") -> pd.Series:
    """逐日截面 IC。method: rank(Spearman) | pearson。

    Spearman 通过先逐日截面排名再做 Pearson 实现，避免 scipy 依赖。
    """
    f, r = factor.align(fwd_ret, join="inner")
    if method == "rank":
        f = f.rank(axis=1)
        r = r.rank(axis=1)
    return f.corrwith(r, axis=1).dropna()


@dataclass
class ICStats:
    ic: pd.Series
    mean: float
    std: float
    icir: float          # mean / std
    t_stat: float
    win_rate: float      # IC > 0 占比

    def as_dict(self) -> dict[str, float]:
        return {
            "ic_mean": self.mean, "ic_std": self.std, "icir": self.icir,
            "t_stat": self.t_stat, "ic_win_rate": self.win_rate,
            "ic_count": int(len(self.ic)),
        }


def ic_stats(ic: pd.Series) -> ICStats:
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    n = len(ic)
    return ICStats(
        ic=ic,
        mean=mean,
        std=std,
        icir=mean / std if std > 0 else 0.0,
        t_stat=mean / (std / np.sqrt(n)) if std > 0 and n > 0 else 0.0,
        win_rate=float((ic > 0).mean()) if n else 0.0,
    )


def factor_autocorr(factor: pd.DataFrame, lag: int) -> float:
    """因子自相关系数：隔 lag 日的截面 rank 相关的均值。"""
    a = factor.rank(axis=1)
    b = a.shift(lag)
    ac = a.corrwith(b, axis=1).dropna()
    return float(ac.mean()) if len(ac) else 0.0


@dataclass
class LayerResult:
    layer_nav: pd.DataFrame          # 各分层净值 (date x layer)
    layer_annual: pd.Series          # 各分层年化收益
    ls_nav: pd.Series                # 多空净值（顶 - 底）
    ls_annual: float
    ls_sharpe: float
    ls_max_drawdown: float
    top_turnover: float              # 顶层组合每期换手（成员变动比例）
    coverage: float                  # 因子覆盖率（非 NaN 占比）


def layer_backtest(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int,
    quantiles: int = 5,
) -> LayerResult:
    """分层回测：每隔 horizon 日按因子值分 quantiles 层等权持有。"""
    ret = close.ffill().pct_change().fillna(0.0)
    dates = factor.index
    rebal = dates[::horizon]

    # 每层每日收益
    layers: dict[int, list] = {q: [] for q in range(quantiles)}
    top_members_prev: set | None = None
    top_turnovers: list[float] = []

    for d in rebal:
        f = factor.loc[d].dropna()
        if len(f) < quantiles:
            continue
        try:
            q = pd.qcut(f.rank(method="first"), quantiles, labels=False)
        except ValueError:
            continue
        i = dates.get_loc(d)
        window = ret.iloc[i + 1: i + 1 + horizon]  # 次日起持有
        for layer in range(quantiles):
            members = q[q == layer].index
            layers[layer].append(window[members].mean(axis=1))
        top = set(q[q == quantiles - 1].index)
        if top_members_prev is not None and top:
            changed = len(top - top_members_prev) / len(top)
            top_turnovers.append(changed)
        top_members_prev = top

    layer_ret = {}
    for q, parts in layers.items():
        parts = [p for p in parts if len(p)]
        if parts:
            layer_ret[f"Q{q + 1}"] = pd.concat(parts).sort_index()
    layer_ret = pd.DataFrame(layer_ret)
    if layer_ret.empty:
        raise RuntimeError(
            f"分层回测无有效样本：检查因子覆盖度与 quantiles={quantiles} 是否超过股票池规模"
        )
    layer_nav = (1 + layer_ret).cumprod()

    years = max(len(layer_ret) / TRADING_DAYS, 1e-9)
    layer_annual = layer_nav.iloc[-1] ** (1 / years) - 1

    ls_ret = layer_ret[f"Q{quantiles}"] - layer_ret["Q1"]
    ls_nav = (1 + ls_ret).cumprod()
    ls_annual = float(ls_nav.iloc[-1] ** (1 / years) - 1)
    sd = float(ls_ret.std(ddof=1))
    ls_sharpe = float(ls_ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    dd = (ls_nav / ls_nav.cummax() - 1).min()

    return LayerResult(
        layer_nav=layer_nav,
        layer_annual=layer_annual,
        ls_nav=ls_nav,
        ls_annual=ls_annual,
        ls_sharpe=ls_sharpe,
        ls_max_drawdown=float(dd),
        top_turnover=float(np.mean(top_turnovers)) if top_turnovers else 0.0,
        coverage=float(factor.notna().mean().mean()),
    )
