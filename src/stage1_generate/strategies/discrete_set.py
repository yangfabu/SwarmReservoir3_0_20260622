"""
离散集合随机信号策略

从一个预定义的离散电流值集合中，独立均匀随机采样生成电流序列。
每一步从 allowed_values 中随机选取一个值，各步之间相互独立。

适用场景: 硬件仅支持有限几个离散电流档位（如 1.0, 1.5, 2.0, 2.5, 3.0 A），
         MC test 时也只需覆盖这些档位。
"""

import numpy as np

from .base import BaseSignalStrategy


class DiscreteSetStrategy(BaseSignalStrategy):
    """离散集合随机信号生成策略。"""

    def name(self) -> str:
        return "discrete_set"

    def generate(
        self, n_steps: int, config: dict, rng: np.random.Generator
    ) -> np.ndarray:
        """
        从离散电流集合中独立均匀随机采样。

        每一步: i(t) = allowed_values[k]，其中 k ~ Uniform{0, 1, ..., N-1}

        Args:
            n_steps: 步数。
            config: 信号配置，必须包含 discrete_set.allowed_values（list[float]）。
            rng: 随机数生成器。

        Returns:
            形状为 (n_steps,) 的电流值数组。
        """
        cfg = config.get("discrete_set", {})
        allowed_values = cfg.get("allowed_values", [1.0, 1.5, 2.0, 2.5, 3.0])

        # 确保 allowed_values 是 list
        if not isinstance(allowed_values, (list, tuple, np.ndarray)):
            raise TypeError(
                f"discrete_set.allowed_values 必须是列表，收到: {type(allowed_values)}"
            )

        allowed_values = np.asarray(allowed_values, dtype=float)

        if len(allowed_values) == 0:
            raise ValueError("discrete_set.allowed_values 不能为空")

        # 从 allowed_values 中等概率独立采样 n_steps 次
        indices = rng.integers(0, len(allowed_values), size=n_steps)
        return allowed_values[indices]
