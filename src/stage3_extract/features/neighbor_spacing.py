"""
归一化邻居间距

通过 Voronoi 剖分或 KDTree 获取邻居粒子对的平均距离，
除以粒子半径进行归一化。反映粒子聚集程度。

迁移自: Parameter_extract/Code/Contraction_interdistance.py -> analyze_frame_dynamics() -> norm_neighbor_dist
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, KDTree

from .base import BaseFeature, register_feature


@register_feature
class NeighborSpacingFeature(BaseFeature):
    """
    归一化邻居间距特征。

    支持两种方法:
      - voronoi: Voronoi 自然邻居
      - kdtree: 固定 k 近邻
    """

    def name(self) -> str:
        return "neighbor_spacing"

    def compute(
        self,
        particles_df: pd.DataFrame,
        frame_id: int,
        config: dict,
    ) -> float:
        """
        计算归一化邻居平均距离。

        Args:
            particles_df: 单帧粒子数据。
            frame_id: 帧编号。
            config: 配置，含 neighbor_spacing.method。

        Returns:
            归一化邻居平均距离，或 np.nan。
        """
        cfg = config.get("neighbor_spacing", {})
        if not self.validate(particles_df, cfg):
            return np.nan

        positions = particles_df[["Center_X", "Center_Y"]].values

        # 获取邻居距离
        method = cfg.get("method", "voronoi")
        distances = self._get_neighbor_distances(positions, method, cfg)

        if distances is None or len(distances) == 0:
            return np.nan

        mean_spacing = np.mean(distances)

        # 归一化：除以平均粒子半径
        mean_radius = float(particles_df["Radius"].mean())
        if mean_radius == 0:
            return np.nan

        return float(mean_spacing / mean_radius)

    @staticmethod
    def _get_neighbor_distances(
        positions: np.ndarray,
        method: str,
        config: dict,
    ) -> Optional[np.ndarray]:
        """
        获取所有邻居对的距离。

        Args:
            positions: (N, 2) 坐标。
            method: "voronoi" 或 "kdtree"。
            config: 配置。

        Returns:
            距离数组或 None。
        """
        if method == "voronoi":
            if len(positions) < 4:
                return None
            try:
                vor = Voronoi(positions)
                pairs = vor.ridge_points
                return np.linalg.norm(
                    positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1
                )
            except Exception:
                return None
        elif method == "kdtree":
            k = config.get("num_neighbors", 6)
            if len(positions) <= k:
                return None
            try:
                tree = KDTree(positions)
                all_dist = []
                for pos in positions:
                    dist, _ = tree.query(pos, k=k + 1)
                    all_dist.extend(dist[1:])
                return np.array(all_dist)
            except Exception:
                return None
        else:
            raise ValueError(f"未知方法: '{method}'。可选: voronoi, kdtree")
