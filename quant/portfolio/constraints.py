"""组合与风控约束钩子：对策略给出的目标权重施加可配置约束。

约束按固定顺序应用，每个约束是独立函数，可自行扩展（如行业中性、换手率上限）。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PortfolioConstraints:
    max_positions: int = 10
    max_weight_per_stock: float = 0.25
    max_weight_per_industry: float = 1.0
    industry_map: dict[str, str] | None = None  # symbol -> industry，扩展位


def apply_constraints(weights: pd.Series, cons: PortfolioConstraints) -> pd.Series:
    """应用组合约束，返回修正后的目标权重。"""
    if weights.empty:
        return weights

    w = weights[weights > 0].copy()

    # 1) 最大持仓数：保留权重最大的前 N 只
    if len(w) > cons.max_positions:
        w = w.nlargest(cons.max_positions)

    # 2) 归一化到满仓（现金 = 1 - sum）
    total = w.sum()
    if total > 0:
        w = w / total

    # 3) 单一个股权重上限（超出部分简单截断，留现金）
    w = w.clip(upper=cons.max_weight_per_stock)

    # 4) 行业权重上限（提供了 industry_map 才生效，否则跳过）
    if cons.industry_map and cons.max_weight_per_industry < 1.0:
        ind = pd.Series({s: cons.industry_map.get(s, "UNKNOWN") for s in w.index})
        for industry, group in ind.groupby(ind):
            idx = group.index
            sub_total = w[idx].sum()
            if sub_total > cons.max_weight_per_industry:
                w[idx] = w[idx] * (cons.max_weight_per_industry / sub_total)

    return w.sort_index()
