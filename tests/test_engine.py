"""测试回测引擎关键逻辑：防未来函数、费用扣减、分板块涨跌停、停牌、整手。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestEngine
from quant.config import BacktestConfig
from quant.data.loader import MarketData
from quant.data.schema import make_meta
from quant.portfolio.constraints import PortfolioConstraints
from quant.strategy.base import Strategy


def make_market(close: np.ndarray, open_: np.ndarray | None = None,
                volume: np.ndarray | None = None, start="2024-01-01",
                symbols: list[str] | None = None,
                is_st: np.ndarray | None = None) -> MarketData:
    """构造小型 MarketData。close/open_/volume 形状 (n_days, n_symbols)。"""
    n_days, n_syms = close.shape
    dates = pd.bdate_range(start, periods=n_days)
    symbols = symbols or ["600000.SH"] * n_syms  # 默认主板
    open_ = close if open_ is None else open_
    volume = np.full_like(close, 1e6) if volume is None else volume
    is_st = np.zeros_like(close, dtype=bool) if is_st is None else is_st
    idx_kw = dict(index=dates, columns=symbols)
    panels = {
        "open": pd.DataFrame(open_, **idx_kw),
        "high": pd.DataFrame(open_, **idx_kw),
        "low": pd.DataFrame(open_, **idx_kw),
        "close": pd.DataFrame(close, **idx_kw),
        "volume": pd.DataFrame(volume, **idx_kw),
        "amount": pd.DataFrame(volume * close, **idx_kw),
        "float_mktcap": pd.DataFrame(close * 1e8, **idx_kw),
    }
    raw_panels = {"open": panels["open"].copy(), "close": panels["close"].copy()}
    flags = {
        "is_suspended": pd.DataFrame(volume <= 0, **idx_kw),
        "is_st": pd.DataFrame(is_st, **idx_kw),
    }
    benchmark = panels["close"].mean(axis=1)
    return MarketData(
        panels=panels, raw_panels=raw_panels, flags=flags,
        meta=make_meta(symbols), calendar=dates, benchmark=benchmark,
        dates=dates, symbols=symbols,
    )


class FullWeightStrategy(Strategy):
    """始终满仓第一只标的。"""

    def __init__(self):
        super().__init__({})

    def target_weights(self, history, date, current_weights):
        sym = history["close"].columns[0]
        return pd.Series({sym: 1.0})


class NoFutureSpy(Strategy):
    """断言 history 不含 date 之后的数据。"""

    def __init__(self):
        super().__init__({})
        self.checked = 0

    def target_weights(self, history, date, current_weights):
        assert history["close"].index.max() == date, "检测到未来数据泄漏!"
        self.checked += 1
        return pd.Series(dtype=float)


def bt_cfg(**kw) -> BacktestConfig:
    base = dict(commission=0.0, min_commission=0.0, stamp_tax=0.0,
                slippage=0.0, transfer_fee=0.0)
    base.update(kw)
    return BacktestConfig(**base)


def make_engine(market: MarketData, cfg: BacktestConfig) -> BacktestEngine:
    """测试引擎：放宽个股权重上限，便于验证满仓行为。"""
    cons = PortfolioConstraints(max_positions=100, max_weight_per_stock=1.0)
    return BacktestEngine(market, cfg, cons)


def test_no_future_data():
    close = np.full((10, 2), 10.0)
    market = make_market(close, symbols=["600000.SH", "000001.SZ"])
    spy = NoFutureSpy()
    make_engine(market, bt_cfg(rebalance="daily")).run(spy)
    assert spy.checked >= 9  # 除最后一日外每天都触发


def test_buy_and_hold_flat_price():
    """价格恒定时满仓买入，零费用下净值应保持 1。"""
    close = np.full((10, 1), 10.0)
    market = make_market(close)
    res = make_engine(market, bt_cfg(rebalance="daily")).run(FullWeightStrategy())
    assert res.nav.iloc[-1] == pytest.approx(1.0, abs=1e-9)
    assert len(res.trades) == 1  # 只在首日信号后的第二天买一次


def test_lot_rounding_100_shares():
    """买入按 100 股整手，碎股部分留现金。"""
    close = np.full((5, 1), 7.5)  # 1e6/7.5 = 133333 股 -> 133300 股整手
    market = make_market(close)
    res = make_engine(market, bt_cfg(rebalance="daily")).run(FullWeightStrategy())
    assert res.trades.iloc[0]["shares"] == 133300
    assert res.extra["final_cash"] > 0  # 碎股资金留在现金


def test_costs_reduce_nav():
    close = np.full((10, 1), 10.0)
    market = make_market(close)
    free = make_engine(market, bt_cfg(rebalance="daily")).run(FullWeightStrategy())
    costly = make_engine(
        market, bt_cfg(rebalance="daily", commission=0.001, min_commission=0.0,
                       slippage=0.001, transfer_fee=0.0)
    ).run(FullWeightStrategy())
    assert costly.nav.iloc[-1] < free.nav.iloc[-1]


def test_limit_up_blocks_buy_main_board():
    """主板 T+1 开盘涨停（+10%）当日不允许买入；次日未涨停可正常成交。"""
    close = np.array([[10.0], [11.0], [11.0], [11.0]])
    market = make_market(close, symbols=["600000.SH"])
    res = make_engine(market, bt_cfg(rebalance="daily")).run(FullWeightStrategy())
    assert (res.trades["date"] == market.dates[1]).sum() == 0, "开盘涨停日不应有成交"
    assert (res.trades["date"] == market.dates[2]).sum() == 1, "涨停结束后应正常买入"


def test_board_limit_cyb_20pct():
    """创业板（300）涨跌幅 20%：+15% 开盘不算涨停，仍可买入。"""
    close = np.array([[10.0], [11.5], [11.5], [11.5]])
    market = make_market(close, symbols=["300750.SZ"])
    res = make_engine(market, bt_cfg(rebalance="daily")).run(FullWeightStrategy())
    assert (res.trades["date"] == market.dates[1]).sum() == 1, "创业板 +15% 未涨停，应成交"


def test_st_limit_5pct():
    """ST 标的涨跌幅 5%：+5% 开盘即涨停，买入被拒。"""
    close = np.array([[10.0], [10.5], [10.5], [10.5]])
    is_st = np.ones((4, 1), dtype=bool)
    market = make_market(close, symbols=["000333.SZ"], is_st=is_st)
    res = make_engine(market, bt_cfg(rebalance="daily")).run(FullWeightStrategy())
    assert (res.trades["date"] == market.dates[1]).sum() == 0, "ST +5% 涨停日不应成交"


def test_suspension_blocks_trade():
    """停牌（is_suspended）日不允许成交。"""
    close = np.full((5, 1), 10.0)
    volume = np.full((5, 1), 1e6)
    volume[1] = 0.0  # 第二天停牌
    market = make_market(close, volume=volume)
    res = make_engine(market, bt_cfg(rebalance="daily")).run(FullWeightStrategy())
    assert (res.trades["date"] == market.dates[1]).sum() == 0
    assert len(res.trades) == 1


def test_benchmark_nav_present():
    close = np.full((5, 2), 10.0)
    market = make_market(close, symbols=["600000.SH", "000001.SZ"])
    res = make_engine(market, bt_cfg(rebalance="daily")).run(FullWeightStrategy())
    assert res.benchmark_nav is not None
    assert len(res.benchmark_nav) == len(res.nav)


def test_partial_trades_do_not_nan_positions():
    """只有部分标的成交时，其余标的持仓不得被 NaN 污染（Series 对齐回归测试）。"""
    close = np.full((10, 3), 10.0)
    market = make_market(close, symbols=["600000.SH", "000001.SZ", "601318.SH"])

    class SingleSymbol(Strategy):
        def __init__(self):
            super().__init__({})

        def target_weights(self, history, date, current_weights):
            return pd.Series({"600000.SH": 1.0})  # 只交易一只

    res = make_engine(market, bt_cfg(rebalance="daily")).run(SingleSymbol())
    assert res.nav.notna().all()
    assert res.positions.notna().all().all()
    # 平价格下净值保持 1
    assert res.nav.iloc[-1] == pytest.approx(1.0, abs=1e-9)
