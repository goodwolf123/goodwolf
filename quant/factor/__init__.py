"""因子注册表。新增因子文件后在此 import 一次即可被发现。"""

from .base import Factor, available_factors, create_factor, register_factor

from . import library as _library  # noqa: F401  （import 触发 @register_factor）

__all__ = ["Factor", "register_factor", "create_factor", "available_factors"]
