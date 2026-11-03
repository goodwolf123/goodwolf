"""策略注册表。新增策略文件后在此 import 一次即可被发现。"""

from .base import Strategy, available, create_strategy, register

# 内置示例策略（import 触发 @register）
from . import ma_cross as _ma_cross  # noqa: F401
from . import momentum as _momentum  # noqa: F401

__all__ = ["Strategy", "register", "create_strategy", "available"]
