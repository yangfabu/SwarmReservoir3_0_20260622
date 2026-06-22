"""
邻居距离熵 (Neighbor Distance Entropy)

通过 Voronoi 剖分或 KDTree 搜索找到粒子邻居关系，计算归一化邻居距离的
香农熵 H_NDist = -sum(p(r) * log(p(r)))。

合并自: HNDist_Voronoi.py + HNDist_KDTree.py + swarm_reservoir_lib.py（三份重复实现）
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, KDTree

from .base import BaseFeature, register_feature


@register_feature
class EntropyFeature(BaseFeature):
    """
    邻居距离熵特征。

    支持两种计算方法:
      - voronoi: 基于 Voronoi 剖分的邻居关系（自然邻居，推荐）
      - kdtree: 基于 KDTree 的 k 近邻搜索
    """

    def name(self) -> str:
        return "entropy"

    def compute(
        self,
        particles_df: pd.DataFrame,
        frame_id: int,
        config: dict,
    ) -> float:
        """
        计算邻居距离熵。

        Args:
            particles_df: 单帧粒子数据。
            frame_id: 帧编号（未使用，保留以兼容接口）。
            config: 特征配置，必须包含 entropy.method 和相关参数。

        Returns:
            熵值，或 np.nan。
        """
        cfg = config.get("entropy", {})
        method = cfg.get("method", "voronoi")
        d = float(cfg.get("particle_diameter", 20.0))
        bins = cfg.get("bins", None)

        positions = particles_df[["Center_X", "Center_Y"]].values

        if not self.validate(particles_df, cfg):
            return np.nan

        # 根据方法获取邻居距离
        if method == "voronoi":
            distances = self._voronoi_distances(positions)
        elif method == "kdtree":
            num_neighbors = cfg.get("num_neighbors", 6)
            distances = self._kdtree_distances(positions, num_neighbors)
        else:
            raise ValueError(f"未知的 entropy 方法: '{method}'。可选: voronoi, kdtree")

        if distances is None or len(distances) == 0:
            return np.nan

        # 归一化：除以粒子直径
        normalized = np.array(distances) / d

        # 直方图估计概率分布
        if bins is not None:
            hist, _ = np.histogram(normalized, bins=bins, density=False)
        else:
            bins_auto = max(6, int(np.sqrt(len(normalized))))
            hist, _ = np.histogram(normalized, bins=bins_auto, density=False)

        # 概率
        total = hist.sum()
        if total == 0:
            return np.nan
        probs = hist[hist > 0] / total

        # 熵
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)

    @staticmethod
    def _voronoi_distances(positions: np.ndarray) -> Optional[np.ndarray]:
        """
        通过 Voronoi 剖分获取所有邻居对的距离。

        Args:
            positions: (N, 2) 粒子坐标数组。

        Returns:
            所有邻居距离的一维数组，或 None。
        """
        if len(positions) < 4:
            return None
        try:
            vor = Voronoi(positions)
            neighbor_pairs = vor.ridge_points
            distances = np.linalg.norm(
                positions[neighbor_pairs[:, 0]] - positions[neighbor_pairs[:, 1]],
                axis=1,
            )
            return distances
        except Exception:
            return None

    @staticmethod
    def _kdtree_distances(
        positions: np.ndarray, num_neighbors: int = 6
    ) -> Optional[np.ndarray]:
        """
        通过 KDTree 获取每个粒子到其 k 近邻的距离。

        Args:
            positions: (N, 2) 粒子坐标数组。
            num_neighbors: 每个粒子考虑的最近邻数量。

        Returns:
            所有邻居距离的一维数组，或 None。
        """
        if len(positions) <= num_neighbors:
            return None
        try:
            tree = KDTree(positions)
            all_distances = []
            for pos in positions:
                distances, _ = tree.query(pos, k=num_neighbors + 1)
                # 排除自身 (距离为 0)
                all_distances.extend(distances[1:])
            return np.array(all_distances)
        except Exception:
            return None


def compute_entropy_with_distribution(
    particles_df: pd.DataFrame,
    config: dict,
) -> Tuple[Optional[np.ndarray], float]:
    """
    计算熵并返回归一化距离分布（用于直方图可视化）。

    Args:
        particles_df: 单帧粒子数据。
        config: 配置。

    Returns:
        (normalized_distances, entropy) 或 (None, np.nan)。
    """
    feature = EntropyFeature()
    cfg = config.get("entropy", {})
    method = cfg.get("method", "voronoi")
    d = float(cfg.get("particle_diameter", 20.0))

    positions = particles_df[["Center_X", "Center_Y"]].values

    if method == "voronoi":
        distances = feature._voronoi_distances(positions)
    else:
        num_neighbors = cfg.get("num_neighbors", 6)
        distances = feature._kdtree_distances(positions, num_neighbors)

    if distances is None or len(distances) == 0:
        return None, np.nan

    normalized = np.array(distances) / d
    entropy = feature.compute(particles_df, 0, config)
    return normalized, entropy
