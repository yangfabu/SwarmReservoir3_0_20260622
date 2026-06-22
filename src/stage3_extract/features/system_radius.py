"""
归一化系统半径

计算所有粒子到质心的平均距离，并除以粒子半径进行归一化。
用于衡量系统的收缩/扩张趋势。

迁移自: Parameter_extract/Code/Contraction_interdistance.py -> analyze_frame_dynamics() -> norm_system_radius
"""

import numpy as np
import pandas as pd

from .base import BaseFeature, register_feature


@register_feature
class SystemRadiusFeature(BaseFeature):
    """
    归一化系统半径特征。

    算法:
      1. 计算粒子群质心
      2. 计算每个粒子到质心的距离
      3. 取平均值并除以平均粒子半径进行归一化
    """

    def name(self) -> str:
        return "system_radius"

    def compute(
        self,
        particles_df: pd.DataFrame,
        frame_id: int,
        config: dict,
    ) -> float:
        """
        计算归一化系统半径。

        Args:
            particles_df: 单帧粒子数据，需含 Center_X, Center_Y, Radius。
            frame_id: 帧编号。
            config: 特征配置，system_radius.min_particles。

        Returns:
            归一化系统半径值，或 np.nan。
        """
        cfg = config.get("system_radius", {})
        if not self.validate(particles_df, cfg):
            return np.nan

        positions = particles_df[["Center_X", "Center_Y"]].values
        if len(positions) < 3:
            return np.nan

        # 质心
        centroid = np.mean(positions, axis=0)

        # 平均距离到质心
        dist_to_center = np.linalg.norm(positions - centroid, axis=1)
        mean_distance = np.mean(dist_to_center)

        # 归一化：除以平均粒子半径
        mean_radius = float(particles_df["Radius"].mean())
        if mean_radius == 0:
            return np.nan

        return float(mean_distance / mean_radius)
