"""
信号生成调度器 — 策略注册表与工厂函数

根据 config 中的 strategy 名称查找并实例化对应的策略类。
"""

from typing import Dict, Type

from .strategies.base import BaseSignalStrategy
from .strategies.uniform_random import UniformRandomStrategy
from .strategies.sinusoidal import SinusoidalStrategy
from .strategies.binary_sequence import BinarySequenceStrategy
from .strategies.discrete_set import DiscreteSetStrategy


# 策略注册表：名称 -> 类
_strategies: Dict[str, Type[BaseSignalStrategy]] = {
    "uniform_random": UniformRandomStrategy,
    "sinusoidal": SinusoidalStrategy,
    "binary_sequence": BinarySequenceStrategy,
    "discrete_set": DiscreteSetStrategy,
}


def get_strategy(name: str) -> BaseSignalStrategy:
    """
    根据名称获取信号生成策略实例。

    Args:
        name: 策略名称，对应 config/stage1_generate.yaml 中的 signal.strategy。

    Returns:
        BaseSignalStrategy 实例。

    Raises:
        ValueError: 策略名称未知时抛出。
    """
    cls = _strategies.get(name)
    if cls is None:
        available = ", ".join(_strategies.keys())
        raise ValueError(
            f"未知的信号策略: '{name}'。可用的策略: {available}"
        )
    return cls()


def get_available_strategies() -> list:
    """返回所有已注册策略的名称列表。"""
    return list(_strategies.keys())


def register_strategy(cls: Type[BaseSignalStrategy]) -> Type[BaseSignalStrategy]:
    """
    注册新的策略类（用于扩展，无需修改本模块）。

    Usage:
        @register_strategy
        class MyStrategy(BaseSignalStrategy):
            ...
    """
    instance = cls()
    _strategies[instance.name()] = cls
    return cls
