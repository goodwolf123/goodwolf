"""因子接口与注册表：用户新增因子只需实现 compute 并 @register_factor。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

_REGISTRY: dict[str, type["Factor"]] = {}


def register_factor(name: str):
    """注册因子：@register_factor("my_factor") 装饰在 Factor 子类上。"""

    def deco(cls: type["Factor"]) -> type["Factor"]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def create_factor(name: str, params: dict | None = None) -> "Factor":
    if name not in _REGISTRY:
        raise KeyError(f"未注册的因子: {name}; 可用: {sorted(_REGISTRY)}")
    return _REGISTRY[name](params or {})


def available_factors() -> list[str]:
    return sorted(_REGISTRY)


class Factor(ABC):
    """因子抽象基类。

    约定：
    - compute 返回 (date x symbol) 因子值面板；
    - 只能使用截至当日的数据（rolling/shift 等，禁止负向 shift）；
    - 不做截面预处理（去极值/标准化/中性化由清洗层统一处理）。
    """

    name: str = "base"

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def compute(self, market) -> pd.DataFrame:
        raise NotImplementedError
