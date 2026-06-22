"""
圆形蒙版应用

将培养皿外的区域填充为白色，只保留圆形区域内的图像内容。

迁移自: Parameter_extract/Code/Mask.py -> apply_circular_mask()
改进: 移除 output_path 参数，改为纯函数；添加类型标注。
"""

import cv2
import numpy as np


def apply_circular_mask(
    image: np.ndarray,
    cx: int,
    cy: int,
    r: int,
) -> np.ndarray:
    """
    对图像应用圆形蒙版，将圆外区域填充为白色。

    Args:
        image: BGR 格式的输入图像 (H, W, 3)。
        cx: 圆心 X 坐标 (像素)。
        cy: 圆心 Y 坐标 (像素)。
        r: 圆的半径 (像素)。

    Returns:
        蒙版处理后的图像 (H, W, 3)。圆外区域为纯白色 (255, 255, 255)。
    """
    h, w = image.shape[:2]

    # 创建单通道黑色掩膜
    mask = np.zeros((h, w), dtype=np.uint8)

    # 绘制填充白色圆
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

    # 应用掩膜：圆外 → 白色
    result = image.copy()
    result[mask == 0] = (255, 255, 255)

    return result
