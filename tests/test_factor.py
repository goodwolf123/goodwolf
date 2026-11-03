"""测试因子检验核心：IC 计算、分层回测、统计量。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.analysis.factor_eval import (
    factor_autocorr,
    forward_returns,
    ic_series,
    ic_stats,
    layer_backtest,
)

DATES = pd.bdate_range("2024-01-01", periods=60)
SYMS = [f"S{i}" for i in range(10)]


def test_forward_returns():
    close = pd.DataFrame(np.arange(600, dtype=float).reshape(60, 10) + 100,
                         index=DATES, columns=SYMS)
    fwd = forward_returns(close, 5)
    assert fwd.iloc[-5:].isna().all().all(), "尾部 horizon 行应为 NaN"
    assert fwd.iloc[0, 0] == pytest.approx(close.iloc[5, 0] / close.iloc[0, 0] - 1)


def test_ic_perfect_predictor():
    """因子与未来收益完全线性相关时 IC ≈ ±1。"""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(60, 10))
    close = pd.DataFrame(np.exp(np.cumsum(base * 0.01, axis=0)) * 100,
                         index=DATES, columns=SYMS)
    fwd = forward_returns(close, 1)
    factor = fwd.copy()  # 用未来收益本身做因子（仅验证 IC 数学性质）
    ic = ic_series(factor, fwd, method="pearson")
    assert ic.abs().mean() == pytest.approx(1.0, abs=1e-9)


def test_ic_rank_monotone():
    """因子与未来收益单调但非线性时，RankIC=1，Pearson<1。"""
    rng = np.random.default_rng(1)
    rets = rng.normal(size=(60, 10))
    close = pd.DataFrame(np.exp(np.cumsum(rets * 0.01, axis=0)) * 100,
                         index=DATES, columns=SYMS)
    fwd = forward_returns(close, 1)
    factor = np.sign(fwd) * fwd.abs() ** 3  # 保序非线性
    ic_rank = ic_series(factor, fwd, method="rank").dropna()
    ic_pearson = ic_series(factor, fwd, method="pearson").dropna()
    assert (ic_rank - 1).abs().max() < 1e-9
    assert ic_pearson.mean() < 1.0


def test_ic_stats_fields():
    rng = np.random.default_rng(2)
    ic = pd.Series(rng.normal(0.03, 0.1, 100))
    s = ic_stats(ic)
    assert s.mean == pytest.approx(0.03, abs=0.02)
    assert s.icir == pytest.approx(s.mean / s.std)
    assert 0 <= s.win_rate <= 1
    d = s.as_dict()
    assert set(d) == {"ic_mean", "ic_std", "icir", "t_stat", "ic_win_rate", "ic_count"}


def test_factor_autocorr_constant_factor():
    """常数因子自相关应为 1。"""
    f = pd.DataFrame(np.tile(np.arange(10, dtype=float), (60, 1)),
                     index=DATES, columns=SYMS)
    assert factor_autocorr(f, 5) == pytest.approx(1.0)


def test_layer_backtest_monotone_factor():
    """因子预测次日收益时，顶层年化应高于底层，多空为正。"""
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(60, 10))
    # 次日收益 = 当日信号 + 噪声（信号有一期预测力）
    sig = pd.DataFrame(signal, index=DATES, columns=SYMS)
    ret = sig.shift(1).fillna(0.0) * 0.02 + pd.DataFrame(
        rng.normal(0, 0.005, size=(60, 10)), index=DATES, columns=SYMS
    )
    close = np.exp(np.log(1 + ret).cumsum()) * 100
    factor = sig
    res = layer_backtest(factor, close, horizon=5, quantiles=5)
    assert res.layer_annual["Q5"] > res.layer_annual["Q1"]
    assert res.ls_annual > 0
    assert 0 <= res.top_turnover <= 1
    assert res.coverage == pytest.approx(1.0)
    assert len(res.layer_nav) > 0
