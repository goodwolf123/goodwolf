"""本地 HTML 报告：内嵌净值/回撤图表（base64 PNG）+ 指标与成交表格。"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..backtest.engine import BacktestResult
from .metrics import max_drawdown

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _nav_chart(nav: pd.Series, benchmark: pd.Series | None) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(nav.index, nav.values, lw=1.4, color="#c0392b", label="strategy")
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, lw=1.1, color="#7f8c8d",
                alpha=0.9, label="benchmark")
        ax.legend()
    ax.set_title("NAV")
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def _drawdown_chart(nav: pd.Series) -> str:
    _, dd = max_drawdown(nav)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dd.index, dd.values, 0, color="#2c3e50", alpha=0.6)
    ax.set_title("Drawdown")
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def _table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "<p><i>无记录</i></p>"
    return df.tail(max_rows).to_html(border=0, classes="tbl", float_format=lambda x: f"{x:,.4f}")


def generate_report(
    result: BacktestResult,
    metrics: dict[str, float],
    output: str | Path,
    title: str = "Quant Backtest Report",
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    metrics_rows = "".join(
        f"<tr><td>{k}</td><td>{v:,.4f}</td></tr>" for k, v in metrics.items()
    )
    extra_html = "".join(
        f"<h2>{name}</h2>{_table(df)}" for name, df in (extra_tables or {}).items()
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 32px; color: #222; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 28px; }}
img {{ max-width: 100%; border: 1px solid #eee; }}
table.tbl, table.metrics {{ border-collapse: collapse; font-size: 13px; }}
td, th {{ padding: 4px 10px; border-bottom: 1px solid #ddd; text-align: right; }}
td:first-child {{ text-align: left; }}
</style></head><body>
<h1>{title}</h1>
<h2>绩效指标</h2>
<table class="metrics">{metrics_rows}</table>
<h2>净值曲线</h2>
<img src="data:image/png;base64,{_nav_chart(result.nav, result.benchmark_nav)}">
<h2>回撤</h2>
<img src="data:image/png;base64,{_drawdown_chart(result.nav)}">
<h2>最近持仓权重</h2>
{_table(result.positions.loc[:, (result.positions != 0).any()].tail(5))}
<h2>成交记录（最近 50 条）</h2>
{_table(result.trades)}
{extra_html}
</body></html>"""

    output.write_text(html, encoding="utf-8")
    return output
