"""
信号生成策略抽象基类

所有信号生成策略必须继承 BaseSignalStrategy，实现 name() 和 generate()。
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseSignalStrategy(ABC):
    """信号生成策略的抽象基类。"""

    @abstractmethod
    def name(self) -> str:
        """
        返回策略的唯一标识符。

        Returns:
            策略名称，如 "uniform_random", "sinusoidal" 等。
            该名称与 config/stage1_generate.yaml 中的 signal.strategy 对应。
        """
        ...

    @abstractmethod
    def generate(
        self, n_steps: int, config: dict, rng: np.random.Generator
    ) -> np.ndarray:
        """
        生成电流序列的原始浮点值。

        Args:
            n_steps: 要生成的步数。
            config: 阶段1信号配置字典（signal 节）。
            rng: NumPy 随机数生成器（使用 global random_seed 初始化）。

        Returns:
            形状为 (n_steps,) 的 np.ndarray，包含原始（未量化）电流值。
        """
        ...
