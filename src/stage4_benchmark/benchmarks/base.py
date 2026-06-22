"""
基准测试抽象基类

所有 benchmark 必须继承 BaseBenchmark，实现 name(), run(), plot()。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Type

import numpy as np


class BaseBenchmark(ABC):
    """储备计算基准测试的抽象基类。"""

    @abstractmethod
    def name(self) -> str:
        """
        返回 benchmark 的唯一标识符。

        Returns:
            名称，如 "memory_capacity", "nonlinearity"。
            该名称对应 config/stage4_benchmark.yaml 中 benchmarks.enabled 列表。
        """
        ...

    @abstractmethod
    def run(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        config: dict,
    ) -> dict:
        """
        执行基准测试计算。

        Args:
            features: 预处理后的特征矩阵 (N, D)。
            targets: 目标值数组 (N,)，如原始输入 u(t)。
            config: 该 benchmark 的配置节。

        Returns:
            包含指标名称和标量值的字典，如 {"total_mc": 15.3, "r2_curve": [...]}。
        """
        ...

    @abstractmethod
    def plot(self, results: dict, output_dir: Path) -> None:
        """
        生成并保存可视化图表。

        Args:
            results: run() 返回的结果字典。
            output_dir: 输出目录。
        """
        ...


# 注册表
_benchmark_registry: Dict[str, Type[BaseBenchmark]] = {}


def get_benchmark_class(name: str) -> Type[BaseBenchmark]:
    """根据名称获取 benchmark 类。"""
    for cls in _benchmark_registry.values():
        instance = cls()
        if instance.name() == name:
            return cls
    available = [cls().name() for cls in _benchmark_registry.values()]
    raise ValueError(f"未知的 benchmark: '{name}'。可用: {available}")


def get_available_benchmarks() -> list:
    """返回所有已注册 benchmark 的名称。"""
    return [cls().name() for cls in _benchmark_registry.values()]
