"""绩效指标：年化收益、最大回撤、夏普、胜率、换手率、相对基准超额。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(nav: pd.Series) -> tuple[float, pd.Series]:
    """返回 (最大回撤, 回撤序列)。"""
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min()), dd


def compute_metrics(
    nav: pd.Series,
    daily_returns: pd.Series,
    turnover_total: float,
    benchmark_nav: pd.Series | None = None,
    risk_free: float = 0.0,
) -> dict[str, float]:
    n = len(nav)
    years = max(n / TRADING_DAYS, 1e-9)

    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0

    vol = float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    rf_daily = risk_free / TRADING_DAYS
    excess = daily_returns - rf_daily
    sd = daily_returns.std(ddof=1)
    sharpe = float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

    mdd, _ = max_drawdown(nav)
    calmar = ann_return / abs(mdd) if mdd < 0 else 0.0

    active = daily_returns[daily_returns != 0]
    win_rate = float((active > 0).mean()) if len(active) > 0 else 0.0

    metrics = {
        "total_return": total_return,
        "annual_return": float(ann_return),
        "annual_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": float(calmar),
        "win_rate_daily": win_rate,
        "turnover_annual": float(turnover_total / years),
        "n_days": int(n),
    }

    if benchmark_nav is not None and len(benchmark_nav) == n:
        b_total = float(benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1.0)
        b_ann = (1.0 + b_total) ** (1.0 / years) - 1.0
        excess_ret = daily_returns - benchmark_nav.pct_change().fillna(0.0)
        te = float(excess_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
        metrics["benchmark_annual_return"] = float(b_ann)
        metrics["excess_annual_return"] = float(ann_return - b_ann)
        metrics["info_ratio"] = float((ann_return - b_ann) / te) if te > 0 else 0.0

    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    pct = {
        "total_return", "annual_return", "annual_volatility", "max_drawdown",
        "win_rate_daily", "benchmark_annual_return", "excess_annual_return",
    }
    lines = []
    for k, v in metrics.items():
        lines.append(f"{k:>24s}: {v:.2%}" if k in pct else f"{k:>24s}: {v:.3f}")
    return "\n".join(lines)
