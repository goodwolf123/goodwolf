"""单因子检验 CLI：python -m quant.factor_test --factor momentum --config config.yaml"""

from __future__ import annotations

import argparse
import sys

from .analysis.factor_eval import (
    factor_autocorr,
    forward_returns,
    ic_series,
    ic_stats,
    layer_backtest,
)
from .analysis.factor_report import generate_factor_report
from .clean import CleanContext, run_pipeline
from .config import load_config
from .data.loader import DataLoader
from .factor import available_factors, create_factor
from .universe import UniverseFilter, filter_universe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="单因子检验")
    parser.add_argument("--factor", required=True,
                        help=f"因子名，可用: {available_factors()}")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--params", default="", help='因子参数覆盖，如 "lookback=60,skip=5"')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    fcfg = cfg.factor_test

    # 1) 数据
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
    print(f"[data] {len(market.symbols)} 只标的, {len(market.dates)} 个交易日")
    if market.quality:
        print(f"[quality]\n{market.quality.summary()}")

    # 2) 股票池过滤
    filt = UniverseFilter.from_dict(cfg.universe_filter)
    keep = filter_universe(market, filt)
    market = market.subset(keep)
    print(f"[universe] 过滤后 {len(market.symbols)} 只")

    # 3) 因子计算
    params = dict(fcfg.params)
    if args.params:
        for kv in args.params.split(","):
            k, v = kv.split("=")
            params[k.strip()] = float(v) if "." in v else int(v)
    factor = create_factor(args.factor, params)
    raw = factor.compute(market).reindex(index=market.dates, columns=market.symbols)
    print(f"[factor] {args.factor} params={params}")

    # 4) 清洗管线
    ctx = CleanContext.from_market(market)
    cleaned, pipe_log = run_pipeline(raw, ctx, cfg.cleaning)
    if not pipe_log.empty:
        print("[cleaning]")
        print(pipe_log.to_string(index=False))

    # 5) IC
    fwd = forward_returns(market.panels["close"], fcfg.horizon)
    ic = ic_series(cleaned, fwd, method=fcfg.ic_method)
    stats = ic_stats(ic).as_dict()

    # 6) 因子特征：自相关、覆盖率
    stats["factor_autocorr"] = factor_autocorr(cleaned, fcfg.horizon)

    # 7) 分层回测
    layer = layer_backtest(cleaned, market.panels["close"], fcfg.horizon, fcfg.quantiles)
    stats.update(
        {
            "coverage": layer.coverage,
            "top_layer_turnover": layer.top_turnover,
            "ls_annual_return": layer.ls_annual,
            "ls_sharpe": layer.ls_sharpe,
            "ls_max_drawdown": layer.ls_max_drawdown,
        }
    )

    print("\n===== 因子检验结果 =====")
    for k, v in stats.items():
        print(f"{k:>22s}: {v:,.4f}")
    print("\n分层年化收益:")
    for k, v in layer.layer_annual.items():
        print(f"  {k}: {v:,.2%}")

    out = cfg.root / fcfg.output_dir / f"factor_{args.factor}.html"
    path = generate_factor_report(args.factor, stats, ic, layer, pipe_log, out)
    print(f"\n[report] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
