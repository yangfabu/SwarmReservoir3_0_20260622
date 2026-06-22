"""
邻居距离分布特征 (Neighbor Distance Distribution)

保存每帧邻居距离的完整直方图分布（entropy 只返回标量熵值）。
同时返回均值归一化邻居距离作为 features.csv 中的标量特征。

依赖: EntropyFeature 中的邻居距离计算方法。
"""

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from .base import BaseFeature, register_feature
from .entropy import EntropyFeature


@register_feature
class DistributionFeature(BaseFeature):
    """
    邻居距离分布特征。

    与 EntropyFeature 共享相同的邻居距离计算逻辑，
    但额外输出完整的直方图分布（保存到 distribution.csv）。
    features.csv 中的标量值 = 均值归一化邻居距离。
    """

    def name(self) -> str:
        return "distribution"

    def compute(
        self,
        particles_df: pd.DataFrame,
        frame_id: int,
        config: dict,
    ) -> float:
        """
        计算均值归一化邻居距离。

        Args:
            particles_df: 单帧粒子数据。
            frame_id: 帧编号。
            config: 特征配置，含 distribution.method / particle_diameter / bins / min_particles。

        Returns:
            均值归一化邻居距离，或 np.nan。
        """
        cfg = config.get("distribution", {})
        if not self.validate(particles_df, cfg):
            return np.nan

        distances = self._get_neighbor_distances(particles_df, cfg)
        if distances is None or len(distances) == 0:
            return np.nan

        d = float(cfg.get("particle_diameter", 20.0))
        normalized = np.array(distances) / d
        return float(np.mean(normalized))

    def compute_histogram(
        self,
        particles_df: pd.DataFrame,
        config: dict,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
        """
        计算邻居距离直方图。

        Args:
            particles_df: 单帧粒子数据。
            config: 特征配置。

        Returns:
            (bin_edges, hist_counts, bin_labels) 三元组。
            如果粒子数不足或计算出错，返回 (None, None, None)。
            bin_labels 如 ["1.0-1.5", "1.5-2.0", ...]，用于 CSV 表头。
        """
        cfg = config.get("distribution", {})
        if not self.validate(particles_df, cfg):
            return None, None, None

        distances = self._get_neighbor_distances(particles_df, cfg)
        if distances is None or len(distances) == 0:
            return None, None, None

        d = float(cfg.get("particle_diameter", 20.0))
        normalized = np.array(distances) / d
        bins = cfg.get("bins", None)

        if bins is None:
            n_bins = max(6, int(np.sqrt(len(normalized))))
            hist, bin_edges = np.histogram(normalized, bins=n_bins)
        else:
            hist, bin_edges = np.histogram(normalized, bins=bins)

        # 生成 bin 标签
        bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)]

        return bin_edges, hist, bin_labels

    @staticmethod
    def _get_neighbor_distances(
        particles_df: pd.DataFrame,
        config: dict,
    ) -> Optional[np.ndarray]:
        """
        获取所有邻居对的距离（复用 EntropyFeature 的静态方法）。

        Args:
            particles_df: 单帧粒子数据。
            config: 分布特征的配置。

        Returns:
            一维距离数组，或 None。
        """
        positions = particles_df[["Center_X", "Center_Y"]].values
        method = config.get("method", "voronoi")

        if method == "voronoi":
            return EntropyFeature._voronoi_distances(positions)
        elif method == "kdtree":
            num_neighbors = config.get("num_neighbors", 6)
            return EntropyFeature._kdtree_distances(positions, num_neighbors)
        else:
            raise ValueError(f"未知的 distribution 方法: '{method}'。可选: voronoi, kdtree")


def get_distribution_bin_columns(config: dict) -> List[str]:
    """
    根据配置中的 bins 参数生成 distribution.csv 的列名。

    Args:
        config: 阶段3配置字典。

    Returns:
        列名列表，如 ["bin_1.0-1.5", "bin_1.5-2.0", ...]，失败返回空列表。
    """
    cfg = config.get("distribution", {})
    bins = cfg.get("bins", None)
    if bins is None:
        # 默认 bin 数量（与 histogram 自动模式一致）
        return [f"bin_{i}" for i in range(10)]
    return [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
