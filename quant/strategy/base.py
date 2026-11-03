"""策略接口：用户只需实现 Strategy 协议并通过 @register 注册即可接入框架。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

_REGISTRY: dict[str, type["Strategy"]] = {}


def register(name: str):
    """注册策略：@register("my_strategy") 装饰在 Strategy 子类上即可。"""

    def deco(cls: type[Strategy]) -> type[Strategy]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def create_strategy(name: str, params: dict | None = None) -> "Strategy":
    if name not in _REGISTRY:
        raise KeyError(f"未注册的策略: {name}; 可用: {sorted(_REGISTRY)}")
    return _REGISTRY[name](params or {})


def available() -> list[str]:
    return sorted(_REGISTRY)


class Strategy(ABC):
    """策略抽象基类（可插拔接口）。

    框架约定（防未来函数）：
    - `target_weights` 收到的 `history` 只包含调仓日（含）之前的数据；
    - 返回的目标权重在**下一交易日开盘**执行；
    - 权重为 symbol -> weight 的 Series，现金部分 = 1 - sum(weights)。

    可选扩展：覆写 `rebalance_dates` 自定义调仓日。
    """

    name: str = "base"

    def __init__(self, params: dict):
        self.params = params

    def on_init(self, context: dict) -> None:
        """回测开始前调用一次，可做参数校验/预计算。可选覆写。"""

    @abstractmethod
    def target_weights(
        self,
        history: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        current_weights: pd.Series,
    ) -> pd.Series:
        """生成调仓日目标权重。

        Args:
            history: 截至 date（含）的历史面板 {"open","high","low","close","volume",...}，
                     每个面板是 index=date, columns=symbol 的 DataFrame。
            date: 当前调仓日。
            current_weights: 当前持仓权重（symbol -> weight），无持仓为空 Series。

        Returns:
            目标权重 Series（symbol -> weight），权重和应 <= 1。
        """
        raise NotImplementedError
