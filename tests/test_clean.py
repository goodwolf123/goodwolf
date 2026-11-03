"""测试清洗管线算子：去极值、标准化、中性化、剔除、缺失率、管线记录。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.clean import CleanContext, run_pipeline
from quant.clean.ops import (
    drop_missing_rate,
    drop_untradable,
    neutralize,
    standardize,
    winsorize,
)

DATES = pd.bdate_range("2024-01-01", periods=30)
SYMS = ["A", "B", "C", "D", "E"]


def make_ctx(st_syms=(), susp_date=None, list_days_val=1000) -> CleanContext:
    n, m = len(DATES), len(SYMS)
    is_st = pd.DataFrame(False, index=DATES, columns=SYMS)
    for s in st_syms:
        is_st[s] = True
    is_susp = pd.DataFrame(False, index=DATES, columns=SYMS)
    if susp_date is not None:
        is_susp.loc[susp_date, "A"] = True
    return CleanContext(
        is_suspended=is_susp,
        limit_locked=pd.DataFrame(False, index=DATES, columns=SYMS),
        is_st=is_st,
        list_days=pd.DataFrame(list_days_val, index=DATES, columns=SYMS, dtype=float),
        industry=pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Y", "E": "Y"}),
        log_mktcap=pd.DataFrame(
            np.tile(np.log(np.array([1e9, 2e9, 3e9, 4e9, 5e9])), (n, 1)),
            index=DATES, columns=SYMS,
        ),
    )


def rand_factor(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(DATES), len(SYMS))), index=DATES, columns=SYMS)


def test_winsorize_mad_clips_outliers():
    f = rand_factor()
    f.loc[DATES[0], "A"] = 100.0  # 极端值
    out = winsorize(f, make_ctx(), method="mad", n=5)
    assert out.loc[DATES[0], "A"] < 100.0
    # 中位数对极端值稳健：截断后行中位数不变
    assert out.loc[DATES[0]].median() == pytest.approx(f.loc[DATES[0]].median())


def test_winsorize_sigma():
    f = rand_factor()
    f.iloc[0, 0] = 5.0
    f.iloc[0, 1:] = 1.0  # 其余值聚拢，sigma 边界内缩
    out = winsorize(f, make_ctx(), method="sigma", n=1)
    row = f.iloc[0]
    bound = row.mean() + 1.0 * row.std()
    assert out.iloc[0, 0] == pytest.approx(bound)
    assert out.iloc[0, 1] == pytest.approx(1.0)  # 未越界值不变


def test_standardize_zscore():
    out = standardize(rand_factor(), make_ctx())
    assert out.mean(axis=1).abs().max() < 1e-10
    assert (out.std(axis=1) - 1).abs().max() < 1e-10


def test_neutralize_removes_size_effect():
    rng = np.random.default_rng(1)
    ctx = make_ctx()
    # 构造与 log 市值完全线性相关的因子
    f = ctx.log_mktcap * 2.0 + pd.DataFrame(
        rng.normal(0, 0.01, size=(len(DATES), len(SYMS))), index=DATES, columns=SYMS
    )
    out = neutralize(f, ctx, industry=False, size=True)
    corr = out.corrwith(ctx.log_mktcap, axis=1).abs().mean()
    assert corr < 0.05


def test_drop_untradable_st_and_suspension():
    ctx = make_ctx(st_syms=["C"], susp_date=DATES[5])
    f = rand_factor()
    out = drop_untradable(f, ctx, remove_st=True, remove_suspended=True,
                          remove_limit_locked=False, min_list_days=0)
    assert out["C"].isna().all(), "ST 应全程被剔除"
    assert pd.isna(out.loc[DATES[5], "A"]), "停牌日应剔除"
    assert out["B"].notna().all()


def test_drop_untradable_min_list_days():
    ctx = make_ctx(list_days_val=30)  # 全部次新
    f = rand_factor()
    out = drop_untradable(f, ctx, min_list_days=60,
                          remove_st=False, remove_suspended=False, remove_limit_locked=False)
    assert out.isna().all().all()


def test_drop_missing_rate():
    f = rand_factor()
    f.loc[: DATES[20], "A"] = np.nan  # A 缺失率 21/30 = 0.7
    out = drop_missing_rate(f, make_ctx(), max_missing=0.4)
    assert "A" not in out.columns
    assert "B" in out.columns


def test_pipeline_records_counts():
    f = rand_factor()
    f.loc[: DATES[20], "A"] = np.nan
    steps = [
        {"name": "drop_missing_rate", "params": {"max_missing": 0.4}},
        {"name": "winsorize", "params": {"method": "mad", "n": 5}},
        {"name": "standardize"},
        {"name": "disabled_step", "enabled": False} if False else
        {"name": "winsorize", "enabled": False, "params": {"method": "sigma"}},
    ]
    out, log = run_pipeline(f, make_ctx(), steps)
    assert list(log["step"]) == ["drop_missing_rate", "winsorize", "standardize"]
    assert log.loc[0, "symbols_after"] == 4  # A 被剔除
    assert "A" not in out.columns
    # 最终输出已标准化
    assert out.mean(axis=1).abs().max() < 1e-10


def test_pipeline_unknown_op_raises():
    with pytest.raises(KeyError):
        run_pipeline(rand_factor(), make_ctx(), [{"name": "not_exist"}])
