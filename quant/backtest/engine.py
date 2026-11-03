"""回测引擎：真实交易日历驱动，T+1 开盘成交，整手，分板块涨跌停。

执行模型（防未来函数）：
- 调仓日 T 收盘后生成目标权重（策略只能看到 T 及之前的数据）；
- T+1 开盘成交，成交价 = T+1 复权开盘价（含滑点与费用），买入按 100 股整手；
- 涨跌停/停牌判定只使用 T+1 开盘时已知信息（开盘价、昨收、停牌公告）；
- 涨跌停幅度按板块区分：主板 10%，创业板/科创板 20%，北交所 30%，ST 5%。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..data.loader import MarketData
from ..portfolio.constraints import PortfolioConstraints, apply_constraints
from ..strategy.base import Strategy
from .rules import LOT_SIZE, limit_pct_panel

_EPS = 1e-4


@dataclass
class BacktestResult:
    nav: pd.Series                     # 逐日净值（首日=1）
    daily_returns: pd.Series           # 逐日收益
    positions: pd.DataFrame            # 逐日持仓权重 (date x symbol)
    trades: pd.DataFrame               # 成交记录
    turnover_total: float = 0.0        # 累计换手（双边成交额/组合价值 之和）
    benchmark_nav: pd.Series | None = None
    extra: dict = field(default_factory=dict)


def rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """按频率取调仓日：每个自然周/月的第一个交易日。"""
    if freq == "daily":
        return dates
    s = pd.Series(dates, index=dates)
    key = dates.to_period("W-SUN") if freq == "weekly" else dates.to_period("M")
    return pd.DatetimeIndex(s.groupby(key).first().values)


class BacktestEngine:
    def __init__(
        self,
        market: MarketData,
        bt_cfg: BacktestConfig,
        constraints: PortfolioConstraints | None = None,
    ):
        self.m = market
        self.cfg = bt_cfg
        self.cons = constraints or PortfolioConstraints()
        # 前向填充价格（停牌沿用昨收），避免 NaN 污染净值
        self.open = market.panels["open"].ffill()
        self.close = market.panels["close"].ffill()
        self.raw_open = market.raw_panels["open"].ffill()
        self.raw_close = market.raw_panels["close"].ffill()
        self.suspended = market.flags["is_suspended"].fillna(True)
        self.limit_pct = limit_pct_panel(market.symbols, market.flags["is_st"])

    # ---------- 交易可行性（仅用开盘时点已知信息） ----------
    def _tradable(self, i: int) -> tuple[pd.Series, pd.Series]:
        """返回 (can_buy, can_sell) 两个布尔 Series。"""
        symbols = self.m.symbols
        suspended = self.suspended.iloc[i]  # 停牌（事前公告，视为已知）
        can_buy = ~suspended
        can_sell = ~suspended
        if i > 0:
            chg = self.raw_open.iloc[i] / self.raw_close.iloc[i - 1] - 1.0
            lim = self.limit_pct.iloc[i]
            can_buy &= ~(chg >= lim - _EPS)    # 开盘涨停，无法买入
            can_sell &= ~(chg <= -lim + _EPS)  # 开盘跌停，无法卖出
        return can_buy.fillna(False), can_sell.fillna(False)

    # ---------- 费用 ----------
    def _cost(self, values: pd.Series, side: str) -> pd.Series:
        cfg = self.cfg
        commission = (values * cfg.commission).clip(lower=cfg.min_commission)
        # 无成交的标的不收最低佣金
        commission = commission.where(values > 0, 0.0)
        slip = values * cfg.slippage
        transfer = values * cfg.transfer_fee
        tax = values * cfg.stamp_tax if side == "sell" else 0.0
        return commission + slip + transfer + tax

    # ---------- 主循环 ----------
    def run(self, strategy: Strategy) -> BacktestResult:
        cfg = self.cfg
        dates = self.m.dates
        symbols = self.m.symbols
        rebal = set(rebalance_dates(dates, cfg.rebalance))

        strategy.on_init({"market": self.m, "config": cfg})

        cash = float(cfg.initial_cash)
        shares = pd.Series(0.0, index=symbols)
        pending: pd.Series | None = None

        nav_records, pos_records, trade_records = [], [], []
        turnover_total = 0.0

        for i, date in enumerate(dates):
            open_px = self.open.iloc[i]
            close_px = self.close.iloc[i]

            # ---- T+1 开盘执行上一调仓日的目标权重 ----
            if pending is not None and i > 0:
                cash, shares, trades, tv = self._execute(i, date, pending, cash, shares)
                trade_records.extend(trades)
                turnover_total += tv
                pending = None

            shares[shares.abs() < 1e-6] = 0.0

            # ---- 收盘估值 ----
            nav = cash + float((shares * close_px).sum())
            nav_records.append((date, nav))
            w = (shares * close_px) / nav if nav > 0 else shares * 0
            pos_records.append((date, w))

            # ---- 收盘后生成信号（只能看到 date 及之前的数据）----
            if date in rebal and i < len(dates) - 1:
                history = self.m.history(date)
                cur_weights = w[w > 1e-8]
                target = strategy.target_weights(history, date, cur_weights)
                pending = apply_constraints(target, self.cons)

        nav_s = pd.Series(dict(nav_records)).sort_index()
        rets = nav_s.pct_change().fillna(0.0)
        positions = pd.DataFrame(dict(pos_records)).T.sort_index().fillna(0.0)
        trades = pd.DataFrame(trade_records)

        bench = self.m.benchmark
        bench_nav = bench / bench.iloc[0] if bench is not None and len(bench) else None

        return BacktestResult(
            nav=nav_s / nav_s.iloc[0],
            daily_returns=rets,
            positions=positions,
            trades=trades,
            turnover_total=turnover_total,
            benchmark_nav=bench_nav,
            extra={"final_cash": cash},
        )

    # ---------- 交易执行 ----------
    def _execute(
        self, i: int, date, pending: pd.Series, cash: float, shares: pd.Series
    ) -> tuple[float, pd.Series, list, float]:
        cfg = self.cfg
        symbols = self.m.symbols
        open_px = self.open.iloc[i]
        can_buy, can_sell = self._tradable(i)

        port_value = cash + float((shares * open_px).sum())
        target_w = pending.reindex(symbols).fillna(0.0)
        target_val = target_w * port_value

        # 目标股数：买入方向整手向下取整；清仓允许碎股（目标 0 -> 0 股）
        with np.errstate(divide="ignore", invalid="ignore"):
            tgt_shares = (target_val / open_px / LOT_SIZE).apply(np.floor) * LOT_SIZE
        tgt_shares = tgt_shares.fillna(0.0).where(target_w > 0, 0.0)

        delta = tgt_shares - shares  # 股
        # 不可交易约束；无开盘价（未上市/数据缺失）的标的不交易
        delta[(delta > 0) & ~can_buy] = 0.0
        delta[(delta < 0) & ~can_sell] = 0.0
        delta[open_px.isna()] = 0.0

        trades: list[dict] = []
        # 先卖
        sell_shares = -delta[delta < 0]
        if len(sell_shares):
            sell_val = sell_shares * open_px[sell_shares.index]
            cost = self._cost(sell_val, side="sell")
            cash += float(sell_val.sum() - cost.sum())
            for sym, sh in sell_shares.items():
                trades.append(self._trade(date, sym, "sell", float(sell_val[sym]), float(open_px[sym]), sh))
            shares = shares.add(delta[delta < 0], fill_value=0.0)

        # 后买：现金不足时按比例缩减并重新整手
        buy_shares = delta[delta > 0].copy()
        if len(buy_shares):
            buy_val = buy_shares * open_px[buy_shares.index]
            est_cost = self._cost(buy_val, side="buy")
            total_need = float(buy_val.sum() + est_cost.sum())
            if total_need > cash and total_need > 0:
                scale = max(cash / total_need, 0.0)
                buy_shares = (buy_shares * scale // LOT_SIZE) * LOT_SIZE
                buy_val = buy_shares * open_px[buy_shares.index]
                buy_shares = buy_shares[buy_shares > 0]
                buy_val = buy_val[buy_shares.index]
            cost = self._cost(buy_val, side="buy")
            cash -= float(buy_val.sum() + cost.sum())
            for sym, sh in buy_shares.items():
                trades.append(self._trade(date, sym, "buy", float(buy_val[sym]), float(open_px[sym]), sh))
            shares = shares.add(buy_shares, fill_value=0.0)

        traded_value = 0.0
        if trades:
            traded_value = sum(t["value"] for t in trades)
        tv = traded_value / port_value if port_value > 0 else 0.0
        return cash, shares, trades, tv

    @staticmethod
    def _trade(date, symbol, side, value, price, shares) -> dict:
        return {
            "date": date,
            "symbol": symbol,
            "side": side,
            "value": round(value, 2),
            "price": round(price, 4),
            "shares": round(float(shares), 2),
        }
