"""命令行入口：python -m quant.run --config config.yaml"""

from __future__ import annotations

import argparse
import sys

from .analysis.metrics import compute_metrics, format_metrics
from .analysis.report import generate_report
from .backtest.engine import BacktestEngine
from .config import load_config
from .data.loader import DataLoader
from .portfolio.constraints import PortfolioConstraints
from .strategy import create_strategy
from .universe import UniverseFilter, filter_universe


def load_market(cfg):
    loader = DataLoader(cfg.data, cfg.root)
    try:
        market = loader.load()
    except Exception as e:
        if cfg.data.provider != "demo":
            print(f"[warn] 数据源 {cfg.data.provider} 不可用：{e}\n[warn] 回退到 demo 数据。")
            cfg.data.provider = "demo"
            market = loader.load()
        else:
            raise
    return market


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A股量化回测骨架")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # 1) 数据（含交易日历对齐与质量校验）
    market = load_market(cfg)
    print(f"[data] {len(market.symbols)} 只标的, {len(market.dates)} 个交易日 "
          f"({market.dates[0].date()} ~ {market.dates[-1].date()})")
    if market.quality:
        print(f"[quality]\n{market.quality.summary()}")

    # 2) 股票池过滤
    filt = UniverseFilter.from_dict(cfg.universe_filter)
    keep = filter_universe(market, filt)
    if len(keep) < len(market.symbols):
        dropped = sorted(set(market.symbols) - set(keep))
        print(f"[universe] {len(market.symbols)} -> {len(keep)}, 剔除: {dropped}")
        market = market.subset(keep)

    # 3) 策略
    strategy = create_strategy(cfg.strategy.name, cfg.strategy.params)
    print(f"[strategy] {cfg.strategy.name} params={cfg.strategy.params}")

    # 4) 回测
    cons = PortfolioConstraints(
        max_positions=cfg.portfolio.max_positions,
        max_weight_per_stock=cfg.portfolio.max_weight_per_stock,
        max_weight_per_industry=cfg.portfolio.max_weight_per_industry,
    )
    engine = BacktestEngine(market, cfg.backtest, cons)
    result = engine.run(strategy)

    # 5) 绩效与报告
    metrics = compute_metrics(
        result.nav, result.daily_returns, result.turnover_total,
        benchmark_nav=result.benchmark_nav,
    )
    print("\n===== 绩效指标 =====")
    print(format_metrics(metrics))

    report_path = generate_report(
        result, metrics, cfg.root / cfg.report.output,
        title=f"Backtest: {cfg.strategy.name}",
    )
    print(f"\n[report] {report_path}")
    print(f"[trades] {len(result.trades)} 笔成交")
    return 0


if __name__ == "__main__":
    sys.exit(main())
