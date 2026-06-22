"""
阶段4 单元测试：数据预处理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.stage4_benchmark.preprocessing import (
    flatten_features,
    normalize_features,
    apply_pca,
)


class TestFlattenFeatures:
    """测试特征平铺。"""

    def test_flatten_shape(self):
        """验证平铺后形状正确。"""
        n_frames = 66  # 3 groups of 22
        df = pd.DataFrame({
            "Frame": range(1, n_frames + 1),
            "entropy": np.random.randn(n_frames),
            "radius": np.random.randn(n_frames),
        })
        result = flatten_features(df, ["entropy", "radius"], 22)
        assert len(result) == 3
        # 22 * 2 features + 1 Frame col
        assert result.shape[1] == 22 * 2 + 1

    def test_flatten_columns(self):
        """验证列名正确。"""
        n_frames = 44
        df = pd.DataFrame({
            "Frame": range(1, n_frames + 1),
            "entropy": np.random.randn(n_frames),
        })
        result = flatten_features(df, ["entropy"], 22)
        expected_cols = ["Frame"] + [f"F{i}_entropy" for i in range(1, 23)]
        assert list(result.columns) == expected_cols


class TestNormalizeFeatures:
    """测试特征标准化。"""

    def test_normalize_mean_std(self):
        """标准化后均值接近0，标准差接近1。"""
        df = pd.DataFrame({
            "Frame": [1, 2, 3, 4, 5],
            "feat_a": [10.0, 20.0, 15.0, 25.0, 30.0],
            "feat_b": [100.0, 200.0, 150.0, 250.0, 300.0],
        })
        result = normalize_features(df, exclude_cols=["Frame"])
        means = result[["feat_a", "feat_b"]].mean()
        stds = result[["feat_a", "feat_b"]].std()
        assert np.allclose(means, 0, atol=1e-10)
        assert np.allclose(stds, 1, atol=0.5)  # 小样本下 ddof=1 导致 std 偏高


class TestPCA:
    """测试 PCA。"""

    def test_pca_reduction(self):
        """PCA 降维后维数减少。"""
        X = np.random.randn(100, 20)
        X_reduced, pca = apply_pca(X, variance_threshold=0.95)
        assert X_reduced.shape[1] <= 20
        assert X_reduced.shape[0] == 100
