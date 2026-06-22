"""
二进制序列信号策略（预留）

在 low 和 high 两个电流值之间切换，模拟二进制输入。
"""

import numpy as np

from .base import BaseSignalStrategy


class BinarySequenceStrategy(BaseSignalStrategy):
    """二进制序列信号生成策略。"""

    def name(self) -> str:
        return "binary_sequence"

    def generate(
        self, n_steps: int, config: dict, rng: np.random.Generator
    ) -> np.ndarray:
        """
        生成二进制开关电流序列。

        Args:
            n_steps: 步数。
            config: 信号配置，包含 binary_sequence.low, high, min_run_length。
            rng: 随机数生成器。

        Returns:
            形状为 (n_steps,) 的电流值数组。
        """
        cfg = config.get("binary_sequence", {})
        low = float(cfg.get("low", 1.0))
        high = float(cfg.get("high", 2.0))
        min_run = int(cfg.get("min_run_length", 3))

        sequence = []
        current_val = rng.choice([low, high])

        while len(sequence) < n_steps:
            run_length = max(min_run, rng.poisson(min_run))
            sequence.extend([current_val] * run_length)
            current_val = high if current_val == low else low

        return np.array(sequence[:n_steps], dtype=float)
