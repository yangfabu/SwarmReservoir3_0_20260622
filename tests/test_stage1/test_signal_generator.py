"""
阶段1 单元测试：信号生成策略
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from src.stage1_generate.strategies.uniform_random import UniformRandomStrategy
from src.stage1_generate.signal_generator import get_strategy, get_available_strategies
from src.stage1_generate.quantization import quantize_current


class TestUniformRandomStrategy:
    """测试 UniformRandomStrategy。"""

    def test_name(self):
        s = UniformRandomStrategy()
        assert s.name() == "uniform_random"

    def test_generate_range(self):
        s = UniformRandomStrategy()
        rng = np.random.default_rng(42)
        config = {"uniform_random": {"low_limit": 1.0, "high_limit": 2.0}}
        result = s.generate(100, config, rng)
        assert len(result) == 100
        assert result.min() >= 1.0
        assert result.max() <= 2.0

    def test_generate_reproducible(self):
        s = UniformRandomStrategy()
        config = {"uniform_random": {"low_limit": 1.0, "high_limit": 2.0}}
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1 = s.generate(50, config, rng1)
        r2 = s.generate(50, config, rng2)
        assert np.allclose(r1, r2)


class TestQuantization:
    """测试量化。"""

    def test_quantize_decimal_places(self):
        values = np.array([1.234, 5.678, 9.999])
        result = quantize_current(values, 2)
        assert np.allclose(result, [1.23, 5.68, 10.00])

    def test_quantize_zero(self):
        result = quantize_current(np.array([0.001, 0.009]), 2)
        assert np.allclose(result, [0.00, 0.01])


class TestStrategyRegistry:
    """测试策略注册表。"""

    def test_get_known_strategy(self):
        s = get_strategy("uniform_random")
        assert isinstance(s, UniformRandomStrategy)

    def test_get_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            get_strategy("non_existent")

    def test_available_strategies(self):
        names = get_available_strategies()
        assert "uniform_random" in names
        assert "sinusoidal" in names
