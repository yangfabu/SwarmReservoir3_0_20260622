"""
参数网格搜索对比

对多组参数组合运行圆检测，对比检测结果，帮助确定最优参数。

Usage:
    from src.stage3_extract.verification.grid_search import grid_search_params

    param_grid = {
        "binary_threshold": [120, 130, 140],
        "param2": [12, 16, 20],
    }
    results = grid_search_params(image_path, param_grid)
    print(results.sort_values("circle_count", ascending=False))
"""

from pathlib import Path
from typing import Dict, List, Any
from itertools import product

import numpy as np
import pandas as pd

from src.stage3_extract.preprocessing.mask import apply_circular_mask
from src.stage3_extract.detection.circle_detector import CircleDetector
from src.common.io_utils import load_image


def grid_search_params(
    image_path: Path,
    param_grid: Dict[str, List[Any]],
    base_config: dict,
) -> pd.DataFrame:
    """
    对圆检测参数进行网格搜索。

    Args:
        image_path: 测试图像路径。
        param_grid: 参数网格，如 {"binary_threshold": [120, 130, 140], "param2": [12, 16]}。
        base_config: 基础配置（提供 mask 参数和默认检测参数）。

    Returns:
        每行一个参数组合的结果 DataFrame。
    """
    img = load_image(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    mask_cfg = base_config.get("mask", {})
    masked = apply_circular_mask(
        img,
        mask_cfg.get("center_x", 902),
        mask_cfg.get("center_y", 1157),
        mask_cfg.get("radius", 450),
    )

    det_cfg = base_config.get("circle_detection", {})

    # 生成所有参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    results = []
    for combo in combinations:
        params = dict(det_cfg)  # 从默认值开始
        params.update(dict(zip(keys, combo)))

        detector = CircleDetector(
            binary_threshold=params.get("binary_threshold", 130),
            canny_weak=params.get("canny_weak", 135),
            canny_strong=params.get("canny_strong", 170),
            dp=params.get("dp", 1.5),
            min_dist=params.get("min_dist", 20),
            param1=params.get("param1", 80),
            param2=params.get("param2", 16),
            min_radius=params.get("min_radius", 6),
            max_radius=params.get("max_radius", 12),
        )

        circles = detector.detect(masked)
        mean_radius = float(np.mean(circles[:, 2])) if len(circles) > 0 else np.nan

        results.append({
            **dict(zip(keys, combo)),
            "circle_count": len(circles),
            "mean_radius": mean_radius,
        })

    df = pd.DataFrame(results)
    df = df.sort_values("circle_count", ascending=False)
    return df
