"""清洗算子：每个算子为纯函数 (factor, ctx, **params) -> factor。

factor: DataFrame (index=date, columns=symbol)，如因子值或收益。
ctx: CleanContext，提供停牌/涨停/ST/上市天数/行业/市值等面板。
所有算子逐日截面处理，不使用未来数据。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from ..backtest.rules import limit_pct_panel


@dataclass
class CleanContext:
    """清洗算子共享的上下文面板（index=date, columns=symbol）。"""

    is_suspended: pd.DataFrame
    limit_locked: pd.DataFrame   # 一字板（开=高=低=收 且触及涨跌停）
    is_st: pd.DataFrame
    list_days: pd.DataFrame      # 截至当日的上市天数
    industry: pd.Series          # symbol -> industry
    log_mktcap: pd.DataFrame     # log 流通市值

    @classmethod
    def from_market(cls, market) -> "CleanContext":
        dates, symbols = market.dates, market.symbols
        close_raw = market.raw_panels["close"]
        open_raw = market.raw_panels["open"]
        prev = close_raw.shift(1)
        chg = open_raw / prev - 1.0
        lim = limit_pct_panel(symbols, market.flags["is_st"])
        # 一字板：开=收 且 开盘即触及涨跌停
        one_word = (open_raw == close_raw) & (chg.abs() >= lim - 1e-4)

        list_date = market.meta.set_index("symbol")["list_date"]
        list_days = pd.DataFrame(index=dates, columns=symbols, dtype=float)
        for sym in symbols:
            ld = list_date.get(sym)
            if pd.notna(ld):
                list_days[sym] = (dates - ld).days
            else:
                list_days[sym] = np.inf  # 未知上市日期：不视为次新

        mktcap = market.panels["float_mktcap"].reindex(index=dates, columns=symbols)
        industry = market.meta.set_index("symbol")["industry"].reindex(symbols).fillna("UNKNOWN")

        return cls(
            is_suspended=market.flags["is_suspended"].reindex(index=dates, columns=symbols).fillna(True),
            limit_locked=one_word.reindex(index=dates, columns=symbols).fillna(False),
            is_st=market.flags["is_st"].reindex(index=dates, columns=symbols).fillna(False),
            list_days=list_days,
            industry=industry,
            log_mktcap=np.log(mktcap.where(mktcap > 0)),
        )


# ---------------- 算子 ----------------

def drop_missing_rate(factor: pd.DataFrame, ctx: CleanContext, max_missing: float = 0.4) -> pd.DataFrame:
    """剔除缺失率过高的标的（整列）。"""
    rate = factor.isna().mean()
    bad = rate[rate > max_missing].index
    return factor.drop(columns=bad)


def drop_untradable(
    factor: pd.DataFrame,
    ctx: CleanContext,
    remove_suspended: bool = True,
    remove_limit_locked: bool = True,
    remove_st: bool = True,
    min_list_days: int = 60,
) -> pd.DataFrame:
    """逐日剔除停牌 / 一字板 / ST / 次新股（值置 NaN）。"""
    out = factor.copy()
    if remove_suspended:
        out = out.mask(ctx.is_suspended.reindex_like(out).fillna(True))
    if remove_limit_locked:
        out = out.mask(ctx.limit_locked.reindex_like(out).fillna(False))
    if remove_st:
        out = out.mask(ctx.is_st.reindex_like(out).fillna(False))
    if min_list_days > 0:
        out = out.mask(ctx.list_days.reindex_like(out) < min_list_days)
    return out


def fillna_cs(factor: pd.DataFrame, ctx: CleanContext, method: str = "cross_section_mean") -> pd.DataFrame:
    """缺失值处理：cross_section_mean（截面均值）| industry_mean | none。"""
    if method == "none":
        return factor
    if method == "cross_section_mean":
        return factor.T.fillna(factor.mean(axis=1)).T
    if method == "industry_mean":
        out = factor.copy()
        for ind in ctx.industry.unique():
            cols = ctx.industry[ctx.industry == ind].index.intersection(out.columns)
            if len(cols) > 0:
                out[cols] = out[cols].T.fillna(out[cols].mean(axis=1)).T
        return out
    raise ValueError(f"未知 fillna 方法: {method}")


def winsorize(factor: pd.DataFrame, ctx: CleanContext, method: str = "mad", n: float = 5.0) -> pd.DataFrame:
    """去极值：mad（中位数±n*MAD）| sigma（均值±n*σ），逐日截面。"""
    x = factor
    if method == "mad":
        med = x.median(axis=1)
        mad = (x.sub(med, axis=0)).abs().median(axis=1)
        lo, hi = med - n * mad, med + n * mad
    elif method == "sigma":
        mu, sd = x.mean(axis=1), x.std(axis=1)
        lo, hi = mu - n * sd, mu + n * sd
    else:
        raise ValueError(f"未知去极值方法: {method}")
    return x.clip(lo, hi, axis=0)


def standardize(factor: pd.DataFrame, ctx: CleanContext, method: str = "zscore") -> pd.DataFrame:
    """标准化：逐日截面 z-score。"""
    if method != "zscore":
        raise ValueError(f"未知标准化方法: {method}")
    mu, sd = factor.mean(axis=1), factor.std(axis=1)
    sd = sd.replace(0, np.nan)
    return factor.sub(mu, axis=0).div(sd, axis=0)


def neutralize(factor: pd.DataFrame, ctx: CleanContext, industry: bool = True, size: bool = True) -> pd.DataFrame:
    """行业 + 市值中性化：逐日对行业哑变量与 log 市值回归取残差。"""
    if not industry and not size:
        return factor
    out = factor.copy()
    industries = ctx.industry
    for dt in out.index:
        y = out.loc[dt]
        valid = y.notna()
        if industry:
            valid &= industries.reindex(y.index).notna()
        if size:
            valid &= ctx.log_mktcap.loc[dt].notna()
        if valid.sum() < 5:
            continue
        yv = y[valid].to_numpy(dtype=float)
        cols = []
        if industry:
            dummies = pd.get_dummies(industries[valid], dtype=float)
            dummies = dummies.iloc[:, 1:] if dummies.shape[1] > 1 else dummies.iloc[:, :0]
            cols.append(dummies.to_numpy())
        if size:
            cols.append(ctx.log_mktcap.loc[dt, valid].to_numpy().reshape(-1, 1))
        cols.append(np.ones((valid.sum(), 1)))
        X = np.hstack([c for c in cols if c.size > 0])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        out.loc[dt, valid] = yv - X @ beta
    return out


# 算子注册表
OPS = {
    "drop_missing_rate": drop_missing_rate,
    "drop_untradable": drop_untradable,
    "fillna": fillna_cs,
    "winsorize": winsorize,
    "standardize": standardize,
    "neutralize": neutralize,
}


def point_in_time(df: pd.DataFrame, as_of: pd.Timestamp,
                  announce_col: str = "announce_date") -> pd.DataFrame:
    """财务类字段的披露时点过滤：只保留 as_of 当日已公布（announce_date <= as_of）的记录。

    用于真实数据源的财务因子，防止使用尚未披露的数据（未来函数）。
    """
    if announce_col not in df.columns:
        raise ValueError(f"财务数据必须包含披露日期列 {announce_col}")
    return df[pd.to_datetime(df[announce_col]) <= as_of]
