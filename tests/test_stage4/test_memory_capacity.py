"""
阶段4 单元测试：Memory Capacity benchmark
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from src.stage4_benchmark.benchmarks.memory_capacity import MemoryCapacityBenchmark


class TestMemoryCapacity:
    """测试 MemoryCapacityBenchmark。"""

    def test_name(self):
        assert MemoryCapacityBenchmark().name() == "memory_capacity"

    def test_perfect_memory(self):
        """
        构造一个完美可预测的系统：
        X 的第 k 列正好等于 u(t-k)，因此 MC_d 对 d <= n_features 应为 1.0。
        """
        n_steps = 200
        d_max = 5
        u_t = np.random.default_rng(42).random(n_steps)

        # 构造特征: X[t, d] = u[t-d]（完美延迟线）
        n_features = 10
        X = np.zeros((n_steps, n_features))
        for d in range(n_features):
            X[d:, d] = u_t[:n_steps - d]

        benchmark = MemoryCapacityBenchmark()
        config = {
            "memory_capacity": {
                "d_max": d_max,
                "train_ratio": 0.8,
                "regression": {"method": "ridge", "alpha": 0.1},
            }
        }
        results = benchmark.run(X, u_t, config)
        assert "total_mc" in results
        assert "r2_curve" in results
        assert results["d_max"] == d_max
        # 完美预测系统，MC 应接近 d_max
        assert results["total_mc"] > d_max * 0.5

    def test_full_fit_mode(self):
        """全量拟合模式。"""
        n_steps = 100
        u_t = np.random.default_rng(42).random(n_steps)
        X = u_t.reshape(-1, 1)  # 1维特征

        benchmark = MemoryCapacityBenchmark()
        config = {
            "memory_capacity": {
                "d_max": 3,
                "train_ratio": 0.0,
                "regression": {"method": "linear", "alpha": 0.0},
            }
        }
        results = benchmark.run(X, u_t, config)
        assert len(results["r2_curve"]) == 3

    def test_r2_curve_length(self):
        """验证 R² 曲线长度等于 d_max。"""
        u_t = np.random.default_rng(42).random(50)
        X = np.random.default_rng(43).random((50, 5))

        benchmark = MemoryCapacityBenchmark()
        config = {
            "memory_capacity": {
                "d_max": 10,
                "train_ratio": 0.8,
                "regression": {"method": "ridge", "alpha": 0.1},
            }
        }
        results = benchmark.run(X, u_t, config)
        assert len(results["r2_curve"]) == 10
        assert all(0 <= r2 <= 1.0 for r2 in results["r2_curve"])
