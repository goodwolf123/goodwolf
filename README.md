# A股量化交易系统骨架

本地单机运行的量化研究框架。策略层是可插拔接口：你只写策略/因子，框架负责数据、清洗、执行、约束与检验。

## 目录结构

```
quant/
  config.py             # 配置加载器（config.yaml -> Config 对象）
  data/
    schema.py           # 统一字段规范 + 板块识别 + 元数据规范
    calendar.py         # 交易日历（akshare 真实日历优先，失败回退工作日近似）
    base.py             # DataProvider 抽象接口（fetch / fetch_meta / fetch_benchmark）
    demo.py             # 离线 demo 源（ST/停牌/次新/拆股/行业/流通市值齐全）
    akshare_provider.py # 免费真实源（可选依赖）
    loader.py           # Parquet 缓存 + 按日期水位增量更新 + 复权 + 质量校验
    quality.py          # 数据质量校验（重复/缺失/日期缺口/价格异常）
  universe.py           # 股票池过滤（上市天数/ST/成交额/流通市值）
  clean/                # 清洗层（可配置函数式管线）
    ops.py              #   算子：缺失率剔除、不可交易剔除、填充、去极值、标准化、中性化
    pipeline.py         #   管线执行器（记录每步前后样本数）
  strategy/             # 策略接口 + momentum / ma_cross 示例（仅验证框架）
  portfolio/            # 组合约束（持仓数/个股权重/行业权重上限）
  backtest/
    rules.py            #   分板块涨跌停（主板10%/创业科创20%/北交所30%/ST 5%）、整手
    engine.py           #   回测引擎（T+1 开盘、100 股整手、费用、基准）
  analysis/
    metrics.py          #   年化/回撤/夏普/胜率/换手/超额/信息比率
    report.py           #   回测 HTML 报告
    factor_eval.py      #   IC/RankIC/ICIR/t 统计/自相关/分层回测/多空组合
    factor_report.py    #   因子检验 HTML 报告
  factor/               # 因子接口 + momentum / volatility / turnover 示例
  run.py                # 回测入口
  factor_test.py        # 单因子检验入口
tests/                  # 40 个测试
config.yaml             # 单一配置入口
```

## 环境与运行（独立 venv，不用 Anaconda）

```powershell
# 使用独立安装的 Python（3.10+），不要用 Anaconda 自带的 python
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 策略回测（离线 demo 数据可直接跑）
.\.venv\Scripts\python.exe -m quant.run --config config.yaml

# 单因子检验
.\.venv\Scripts\python.exe -m quant.factor_test --factor momentum --config config.yaml
.\.venv\Scripts\python.exe -m quant.factor_test --factor turnover --config config.yaml --params "lookback=10"

# 测试
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 清洗管线配置

`config.yaml` 的 `cleaning.steps` 按顺序执行，每步可 `enabled: false` 关闭，
参数在 `params` 中传递，执行时打印并记录每步前后样本数：

```yaml
cleaning:
  steps:
    - name: drop_missing_rate            # 剔除缺失率过高的标的（整列）
      params: {max_missing: 0.4}
    - name: drop_untradable              # 剔除停牌/一字板/ST/次新（逐日）
      params: {remove_suspended: true, remove_limit_locked: true, remove_st: true, min_list_days: 60}
    - name: fillna                       # 缺失值：cross_section_mean | industry_mean | none
      enabled: false
      params: {method: cross_section_mean}
    - name: winsorize                    # 去极值：mad | sigma
      params: {method: mad, n: 5}
    - name: standardize                  # z-score
      params: {method: zscore}
    - name: neutralize                   # 行业 + 市值中性化
      params: {industry: true, size: true}
```

财务类字段（接真实源时）必须带 `announce_date`，用 `quant.clean.ops.point_in_time(df, as_of)`
过滤，保证只用已披露数据。

## 如何新增因子

1. 新建 `quant/factor/my_factor.py`：

```python
import pandas as pd
from .base import Factor, register_factor

@register_factor("reversal")
class ReversalFactor(Factor):
    def compute(self, market):
        n = int(self.params.get("lookback", 10))
        close = market.panels["close"].ffill()
        return -(close / close.shift(n) - 1.0)   # 只用历史数据
```

2. 在 `quant/factor/__init__.py` 中 `from . import my_factor`。
3. 运行 `python -m quant.factor_test --factor reversal --config config.yaml`。
   因子参数写在 config 的 `factor_test.params` 或用 `--params "lookback=10"` 覆盖。

## 如何新增策略

同因子：`quant/strategy/` 下新建文件，继承 `Strategy`，实现
`target_weights(history, date, current_weights)`，`@register("name")` 注册，
在 `quant/strategy/__init__.py` import 一次，config 改 `strategy.name` 即可。
完整示例见 [strategy/low_vol 示例段落](quant/strategy/momentum.py)。

## 接入真实数据源

```powershell
.\.venv\Scripts\python.exe -m pip install akshare
```

`config.yaml` 中 `data.provider: akshare`，universe 填真实代码（如 `600519.SH`）。
行情/基准/交易日历走 akshare，缓存按标的存 `data/cache/*.parquet`，重复运行只增量拉取。
网络失败自动回退 demo 数据。

## 已知限制

- **未接入实盘/仿真交易**，仅历史回测。
- demo 数据为合成数据：收益过程带 AR(1) 短期动量，因子检验结果（如换手率因子 IC 很高）
  是数据构造的产物，仅验证框架链路，不代表真实市场。
- 交易日历：demo 模式用工作日近似；akshare 模式用真实交易所日历（带缓存）。
- akshare 源的已知近似：ST 标记为静态（按当前名称，无法还原历史戴帽/摘帽）；
  上市日期、流通市值未接入（相关 universe 过滤对 akshare 不生效，demo 完整支持）。
- 买入按 100 股整手；卖出允许碎股；未做空、未融资。
- 涨跌停按板块区分（主板 10%/创业科创 20%/北交所 30%/ST 5%），以开盘价相对昨收判定；
  未处理盘中开板后可成交的情形。
- 滑点为成交额比例模型，未做流动性/冲击成本建模。
- 因子分层回测为等权、零成本、固定 horizon 的简化实现。
- 示例策略与因子仅用于验证框架，不构成投资建议。
