"""
硬件精度量化模块

将理论计算值截断/舍入到硬件支持的小数位数。
"""

import numpy as np


def quantize_current(values: np.ndarray, decimal_places: int = 2) -> np.ndarray:
    """
    将电流值量化为硬件支持的精度。

    Args:
        values: 原始浮点电流值数组。
        decimal_places: 保留的小数位数（如 2 = 精度 0.01A）。

    Returns:
        量化后的电流值数组。
    """
    return np.round(values, decimal_places)
