"""
粒子速度追踪 (Particle Velocity Tracking)

通过匈牙利算法在相邻帧之间匹配粒子，基于文件名中的时间戳计算速度。

算法:
  1. 从文件名解析绝对时间戳 (YYYYMMDDHHMMSSmmm)
  2. 对相邻帧的所有粒子对构建欧氏距离代价矩阵
  3. 使用 Hungarian 算法 (linear_sum_assignment) 求全局最优一对一匹配
  4. 过滤大于 max_displacement 的匹配
  5. 计算每对匹配粒子的位移和速度

文件名格式支持:
  - 新格式: {prefix}_{frame}_{YYYYMMDDHHMMSSmmm}.jpg  (如 001_000000_20260622184622360.jpg)
  - 旧格式: {prefix}_{frame}_{ms}.jpg                   (如 001_000000_858.jpg, 无绝对时间)
"""

from datetime import datetime
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


class VelocityTracker:
    """
    跨帧粒子速度追踪器。

    使用 Hungarian 算法进行最优粒子匹配，
    支持从文件名解析绝对时间戳以计算精确时间差。
    """

    def __init__(
        self,
        max_displacement: float = 50.0,
        min_matched_pairs: int = 3,
        fallback_fps: Optional[float] = None,
    ):
        """
        Args:
            max_displacement: 粒子在相邻帧间的最大允许位移（像素）。
            min_matched_pairs: 最少匹配对数，低于此值则该帧对标记为无效。
            fallback_fps: 无法从文件名解析时间戳时使用的回退帧率。
        """
        self.max_displacement = max_displacement
        self.min_matched_pairs = min_matched_pairs
        self.fallback_fps = fallback_fps

    # ================================================================
    # 时间戳解析
    # ================================================================

    @staticmethod
    def parse_timestamp(filename: str) -> Optional[datetime]:
        """
        从文件名中提取绝对时间戳。

        支持格式:
          - {prefix}_{frame}_{YYYYMMDDHHMMSSmmm}.jpg  (17位数字时间戳)
          - {prefix}_{frame}_{ms}.jpg                  (1-3位毫秒，无绝对时间)

        Args:
            filename: 图像文件名（可含路径，自动取 stem）。

        Returns:
            datetime 对象，或 None（旧格式或无时间戳）。
        """
        from pathlib import Path

        stem = Path(filename).stem
        parts = stem.split("_")

        if len(parts) >= 3:
            # 尝试将最后一段解析为时间戳
            last_part = parts[-1]
            if last_part.isdigit() and len(last_part) == 17:
                # YYYYMMDDHHMMSSmmm (17 digits)
                try:
                    return datetime.strptime(last_part, "%Y%m%d%H%M%S%f")
                except ValueError:
                    pass

        return None

    @staticmethod
    def compute_dt(
        filename_prev: str,
        filename_curr: str,
        fallback_fps: Optional[float] = None,
    ) -> Optional[float]:
        """
        计算相邻两帧之间的时间差（秒）。

        Args:
            filename_prev: 前一帧文件名。
            filename_curr: 当前帧文件名。
            fallback_fps: 无法解析时间戳时的回退帧率（fps）。

        Returns:
            时间差（秒），或 None。
        """
        t_prev = VelocityTracker.parse_timestamp(filename_prev)
        t_curr = VelocityTracker.parse_timestamp(filename_curr)

        if t_prev is not None and t_curr is not None:
            dt = (t_curr - t_prev).total_seconds()
            if dt > 0:
                return dt

        # 回退: 使用帧率估算
        if fallback_fps and fallback_fps > 0:
            return 1.0 / fallback_fps

        return None

    # ================================================================
    # Hungarian 粒子匹配
    # ================================================================

    def match_frames(
        self,
        prev_particles: pd.DataFrame,
        curr_particles: pd.DataFrame,
        dt_seconds: float,
        prev_filename: str = "",
        curr_filename: str = "",
    ) -> List[dict]:
        """
        使用 Hungarian 算法匹配相邻帧的粒子。

        Args:
            prev_particles: 前一帧粒子 DataFrame (Center_X, Center_Y, Radius)。
            curr_particles: 当前帧粒子 DataFrame。
            dt_seconds: 帧间时间差（秒）。
            prev_filename: 前一帧文件名（用于输出调试）。
            curr_filename: 当前帧文件名。

        Returns:
            匹配结果列表，每项为 dict:
              {prev_id, curr_id, dx, dy, distance_px, velocity_px_per_s, dt_ms}
            如果粒子数不足或无法匹配，返回空列表。
        """
        n_prev = len(prev_particles)
        n_curr = len(curr_particles)

        if n_prev == 0 or n_curr == 0:
            return []

        # 构建代价矩阵 (n_prev × n_curr)
        prev_pos = prev_particles[["Center_X", "Center_Y"]].values.astype(np.float64)
        curr_pos = curr_particles[["Center_X", "Center_Y"]].values.astype(np.float64)

        # 欧氏距离矩阵
        cost_matrix = np.zeros((n_prev, n_curr))
        for i in range(n_prev):
            diff = curr_pos - prev_pos[i]
            cost_matrix[i, :] = np.sqrt(np.sum(diff ** 2, axis=1))

        # Hungarian 最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 过滤 + 构建结果
        results = []
        for r, c in zip(row_ind, col_ind):
            distance = cost_matrix[r, c]
            if distance > self.max_displacement:
                continue

            dx = curr_pos[c, 0] - prev_pos[r, 0]
            dy = curr_pos[c, 1] - prev_pos[r, 1]
            velocity = distance / dt_seconds if dt_seconds > 0 else np.nan

            results.append({
                "prev_particle_id": int(r) + 1,
                "curr_particle_id": int(c) + 1,
                "dx": round(float(dx), 2),
                "dy": round(float(dy), 2),
                "distance_px": round(float(distance), 2),
                "velocity_px_per_s": round(float(velocity), 2),
                "dt_ms": round(dt_seconds * 1000, 2),
            })

        return results

    # ================================================================
    # 批量追踪
    # ================================================================

    def track_all_frames(
        self,
        particles_cache: List[pd.DataFrame],
        image_filenames: List[str],
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """
        对所有相邻帧对执行粒子追踪。

        Args:
            particles_cache: 每帧的粒子 DataFrame 列表。
            image_filenames: 对应的文件名列表（同长度）。

        Returns:
            (velocity_df, per_frame_stats)：
            - velocity_df: 所有匹配对的详细数据 DataFrame。
            - per_frame_stats: 每帧（从第2帧起）的汇总统计列表，
               每项为 {frame, mean_velocity, median_velocity, n_matched, n_prev, n_curr}。
        """
        all_matches = []
        per_frame_stats = []

        for i in range(len(particles_cache) - 1):
            prev_df = particles_cache[i]
            curr_df = particles_cache[i + 1]
            frame_id = i + 2  # 速度属于第 i+2 帧（需要前一帧做参考）

            dt = self.compute_dt(
                image_filenames[i],
                image_filenames[i + 1],
                fallback_fps=self.fallback_fps,
            )

            if dt is None or dt <= 0:
                per_frame_stats.append({
                    "frame": frame_id,
                    "mean_velocity": np.nan,
                    "median_velocity": np.nan,
                    "n_matched": 0,
                    "n_prev": len(prev_df),
                    "n_curr": len(curr_df),
                })
                continue

            matches = self.match_frames(
                prev_df, curr_df, dt,
                prev_filename=image_filenames[i],
                curr_filename=image_filenames[i + 1],
            )

            for m in matches:
                m["frame"] = frame_id
                m["prev_filename"] = image_filenames[i]
                m["curr_filename"] = image_filenames[i + 1]
            all_matches.extend(matches)

            if len(matches) >= self.min_matched_pairs:
                velocities = [m["velocity_px_per_s"] for m in matches]
                per_frame_stats.append({
                    "frame": frame_id,
                    "mean_velocity": round(float(np.mean(velocities)), 2),
                    "median_velocity": round(float(np.median(velocities)), 2),
                    "n_matched": len(matches),
                    "n_prev": len(prev_df),
                    "n_curr": len(curr_df),
                })
            else:
                per_frame_stats.append({
                    "frame": frame_id,
                    "mean_velocity": np.nan,
                    "median_velocity": np.nan,
                    "n_matched": len(matches),
                    "n_prev": len(prev_df),
                    "n_curr": len(curr_df),
                })

        velocity_df = pd.DataFrame(all_matches) if all_matches else pd.DataFrame()
        return velocity_df, per_frame_stats


def velocity_feature_name() -> str:
    """速度标量特征的名称（用于 features.csv 列名）。"""
    return "mean_velocity"
