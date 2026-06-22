"""
正弦波信号策略（预留）

生成正弦波形式的电流序列。
"""

import numpy as np

from .base import BaseSignalStrategy


class SinusoidalStrategy(BaseSignalStrategy):
    """正弦波信号生成策略。"""

    def name(self) -> str:
        return "sinusoidal"

    def generate(
        self, n_steps: int, config: dict, rng: np.random.Generator
    ) -> np.ndarray:
        """
        生成正弦波电流序列。

        公式: i(t) = offset + amplitude * sin(frequency * t)

        Args:
            n_steps: 步数。
            config: 信号配置，必须包含 sinusoidal.amplitude, frequency, offset。
            rng: 随机数生成器（本策略不使用）。

        Returns:
            形状为 (n_steps,) 的电流值数组。
        """
        cfg = config.get("sinusoidal", {})
        amplitude = float(cfg.get("amplitude", 0.5))
        frequency = float(cfg.get("frequency", 0.1))
        offset = float(cfg.get("offset", 1.5))

        t = np.arange(n_steps)
        return offset + amplitude * np.sin(frequency * t)
