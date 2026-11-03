"""配置层：单一入口加载 config.yaml。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    provider: str = "demo"
    cache_dir: str = "data/cache"
    demo_file: str = "data/demo.parquet"
    universe: list[str] = field(default_factory=list)
    demo_n: int = 8
    start: str = "2022-01-01"
    end: str = "2024-12-31"
    adjust: str = "hfq"  # hfq | qfq | none


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000
    rebalance: str = "weekly"  # daily | weekly | monthly
    commission: float = 0.0003
    min_commission: float = 5.0
    stamp_tax: float = 0.0005
    transfer_fee: float = 0.00002  # 过户费（双边）
    slippage: float = 0.001
    limit_pct: float = 0.10        # 主板默认幅度（板块/ST 规则自动覆盖）
    benchmark: str = "auto"        # auto=股票池等权合成；或指数代码（真实源）


@dataclass
class PortfolioConfig:
    max_positions: int = 10
    max_weight_per_stock: float = 0.25
    max_weight_per_industry: float = 1.0


@dataclass
class StrategyConfig:
    name: str = "momentum"
    params: dict = field(default_factory=dict)


@dataclass
class FactorTestConfig:
    horizon: int = 5          # 前瞻收益周期（交易日）
    quantiles: int = 5        # 分层数
    ic_method: str = "rank"   # rank | pearson
    output_dir: str = "output"
    params: dict = field(default_factory=dict)  # 因子参数


@dataclass
class ReportConfig:
    output: str = "output/report.html"


@dataclass
class Config:
    data: DataConfig
    backtest: BacktestConfig
    portfolio: PortfolioConfig
    strategy: StrategyConfig
    report: ReportConfig
    universe_filter: dict = field(default_factory=dict)
    cleaning: list = field(default_factory=list)
    factor_test: FactorTestConfig = field(default_factory=FactorTestConfig)
    root: Path = Path(".")


def load_config(path: str | Path) -> Config:
    """加载 YAML 配置为强类型 Config 对象。"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        data=DataConfig(**raw.get("data", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        portfolio=PortfolioConfig(**raw.get("portfolio", {})),
        strategy=StrategyConfig(**raw.get("strategy", {})),
        report=ReportConfig(**raw.get("report", {})),
        universe_filter=raw.get("universe_filter", {}) or {},
        cleaning=(raw.get("cleaning", {}) or {}).get("steps", []),
        factor_test=FactorTestConfig(**raw.get("factor_test", {})),
        root=path.resolve().parent,
    )
