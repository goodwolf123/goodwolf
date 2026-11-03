"""清洗管线：按 config 顺序执行算子，记录每步处理前后的样本数变化。"""

from __future__ import annotations

import pandas as pd

from .ops import OPS, CleanContext


def run_pipeline(
    factor: pd.DataFrame,
    ctx: CleanContext,
    steps: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """执行清洗管线。

    Args:
        factor: (date x symbol) 因子/待清洗数据
        ctx: CleanContext
        steps: [{"name": "winsorize", "params": {...}} 或 {"name": "winsorize", ...}]

    Returns:
        (清洗后 factor, 处理记录 DataFrame[step, symbols_before, symbols_after, cells_before, cells_after])
    """
    out = factor
    records = []
    for step in steps:
        step = dict(step)
        name = step.pop("name")
        enabled = step.pop("enabled", True)
        if not enabled:
            continue
        if name not in OPS:
            raise KeyError(f"未知清洗算子: {name}; 可用: {sorted(OPS)}")
        params = step.pop("params", {}) or {}
        params.update(step)  # 允许平铺写法

        before_cells = int(out.notna().sum().sum())
        before_syms = int((out.notna().sum() > 0).sum())
        out = OPS[name](out, ctx, **params)
        records.append(
            {
                "step": name,
                "symbols_before": before_syms,
                "symbols_after": int((out.notna().sum() > 0).sum()),
                "cells_before": before_cells,
                "cells_after": int(out.notna().sum().sum()),
            }
        )
    return out, pd.DataFrame(records)
