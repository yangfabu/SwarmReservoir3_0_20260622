"""
均匀随机信号策略

在 [low_limit, high_limit] 范围内生成均匀分布的随机电流序列。

迁移自: Benchmark/Data_generate.py -> generate_mc_dataset_v2()
"""

import numpy as np

from .base import BaseSignalStrategy


class UniformRandomStrategy(BaseSignalStrategy):
    """均匀随机信号生成策略。"""

    def name(self) -> str:
        return "uniform_random"

    def generate(
        self, n_steps: int, config: dict, rng: np.random.Generator
    ) -> np.ndarray:
        """
        生成均匀随机电流序列。

        公式: i(t) = low_limit + (high_limit - low_limit) * u(t)
        其中 u(t) ~ Uniform(0, 1)

        Args:
            n_steps: 步数。
            config: 信号配置，必须包含 uniform_random.low_limit 和 uniform_random.high_limit。
            rng: 随机数生成器。

        Returns:
            形状为 (n_steps,) 的电流值数组。
        """
        cfg = config.get("uniform_random", {})
        low_limit = float(cfg.get("low_limit", 1.0))
        high_limit = float(cfg.get("high_limit", 2.0))

        # 生成 [0, 1] 均匀分布
        u_raw = rng.random(n_steps)

        # 映射到 [low_limit, high_limit]
        bias = low_limit
        sigma_in = high_limit - low_limit
        i_theoretical = bias + sigma_in * u_raw

        return i_theoretical
