"""单因子检验 HTML 报告：IC 时序、分层柱状、多空净值。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .factor_eval import ICStats, LayerResult
from .report import fig_to_base64


def _ic_chart(ic: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = ["#c0392b" if v > 0 else "#2c3e50" for v in ic.values]
    ax.bar(ic.index, ic.values, color=colors, width=1.0, alpha=0.5, label="IC")
    if len(ic) >= 20:
        ax.plot(ic.index, ic.rolling(20).mean().values, color="#e67e22", lw=1.4, label="IC MA20")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_title("IC series")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def _layer_chart(layer_annual: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(layer_annual.index, layer_annual.values * 100, color="#2980b9")
    ax.set_ylabel("annual return (%)")
    ax.set_title("Quantile annual returns")
    ax.grid(alpha=0.3, axis="y")
    return fig_to_base64(fig)


def _ls_chart(ls_nav: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(ls_nav.index, ls_nav.values, color="#27ae60", lw=1.4)
    ax.set_title("Long-Short NAV (top - bottom)")
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def generate_factor_report(
    factor_name: str,
    stats: dict[str, float],
    ic: pd.Series,
    layer: LayerResult,
    pipeline_log: pd.DataFrame | None,
    output: str | Path,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    stat_rows = "".join(f"<tr><td>{k}</td><td>{v:,.4f}</td></tr>" for k, v in stats.items())
    layer_rows = "".join(
        f"<tr><td>{k}</td><td>{v:,.2%}</td></tr>" for k, v in layer.layer_annual.items()
    )
    pipe_html = (
        pipeline_log.to_html(border=0, classes="tbl", index=False)
        if pipeline_log is not None and not pipeline_log.empty
        else "<p><i>未配置清洗步骤</i></p>"
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Factor Test: {factor_name}</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 32px; color: #222; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 28px; }}
img {{ max-width: 100%; border: 1px solid #eee; }}
table.tbl, table.metrics {{ border-collapse: collapse; font-size: 13px; }}
td, th {{ padding: 4px 10px; border-bottom: 1px solid #ddd; text-align: right; }}
td:first-child {{ text-align: left; }}
</style></head><body>
<h1>单因子检验：{factor_name}</h1>
<h2>IC 统计与因子特征</h2>
<table class="metrics">{stat_rows}</table>
<h2>IC 时序</h2>
<img src="data:image/png;base64,{_ic_chart(ic)}">
<h2>分层年化收益</h2>
<img src="data:image/png;base64,{_layer_chart(layer.layer_annual)}">
<table class="metrics">{layer_rows}</table>
<h2>多空组合净值</h2>
<img src="data:image/png;base64,{_ls_chart(layer.ls_nav)}">
<h2>清洗管线记录</h2>
{pipe_html}
</body></html>"""

    output.write_text(html, encoding="utf-8")
    return output
