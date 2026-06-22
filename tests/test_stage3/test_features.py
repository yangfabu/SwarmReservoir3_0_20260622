"""
阶段3 单元测试：特征计算
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.stage3_extract.features.entropy import EntropyFeature
from src.stage3_extract.features.system_radius import SystemRadiusFeature
from src.stage3_extract.features.neighbor_spacing import NeighborSpacingFeature


def _make_particles(n: int, seed: int = 42) -> pd.DataFrame:
    """生成 n 个随机粒子。"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Center_X": rng.uniform(100, 400, n),
        "Center_Y": rng.uniform(100, 400, n),
        "Radius": np.full(n, 10.0),
    })


class TestEntropyFeature:
    """测试 EntropyFeature。"""

    def test_name(self):
        assert EntropyFeature().name() == "entropy"

    def test_compute_voronoi(self):
        feature = EntropyFeature()
        df = _make_particles(20)
        config = {"entropy": {"method": "voronoi", "particle_diameter": 20.0, "min_particles": 5}}
        result = feature.compute(df, 1, config)
        assert not np.isnan(result)
        assert result >= 0

    def test_compute_kdtree(self):
        feature = EntropyFeature()
        df = _make_particles(20)
        config = {"entropy": {"method": "kdtree", "num_neighbors": 6, "particle_diameter": 20.0, "min_particles": 5}}
        result = feature.compute(df, 1, config)
        assert not np.isnan(result)
        assert result >= 0

    def test_too_few_particles(self):
        feature = EntropyFeature()
        df = _make_particles(3)
        config = {"entropy": {"min_particles": 5}}
        result = feature.compute(df, 1, config)
        assert np.isnan(result)


class TestSystemRadiusFeature:
    """测试 SystemRadiusFeature。"""

    def test_name(self):
        assert SystemRadiusFeature().name() == "system_radius"

    def test_compute(self):
        feature = SystemRadiusFeature()
        df = _make_particles(20)
        config = {"system_radius": {"min_particles": 3}}
        result = feature.compute(df, 1, config)
        assert not np.isnan(result)
        assert result > 0


class TestNeighborSpacingFeature:
    """测试 NeighborSpacingFeature。"""

    def test_name(self):
        assert NeighborSpacingFeature().name() == "neighbor_spacing"

    def test_compute_voronoi(self):
        feature = NeighborSpacingFeature()
        df = _make_particles(20)
        config = {"neighbor_spacing": {"method": "voronoi", "min_particles": 4}}
        result = feature.compute(df, 1, config)
        assert not np.isnan(result)
        assert result > 0
