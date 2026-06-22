"""
粒子圆检测器

使用霍夫圆检测 (Hough Circle Transform) 从预处理后的图像中检测粒子。

迁移自: Parameter_extract/Code/Circle_Recognize.py -> detect_circles_gray()
改进: 移除原地绘制逻辑，检测和可视化分离；添加类型标注。
"""

from typing import Optional

import cv2
import numpy as np


class CircleDetector:
    """
    基于霍夫梯度法的粒子圆检测器。

    流程: BGR → Gray → Binary Threshold → Gaussian Blur → Canny Edge → HoughCircles

    所有参数可从 config/stage3_extract.yaml 的 circle_detection 节加载。
    """

    def __init__(
        self,
        binary_threshold: int = 130,
        canny_weak: int = 135,
        canny_strong: int = 170,
        dp: float = 1.5,
        min_dist: int = 20,
        param1: int = 80,
        param2: int = 16,
        min_radius: int = 6,
        max_radius: int = 12,
        gaussian_kernel: tuple = (5, 5),
        gaussian_sigma: float = 2.0,
    ):
        """
        Args:
            binary_threshold: 二值化阈值。
            canny_weak: Canny 低阈值。
            canny_strong: Canny 高阈值。
            dp: Hough 累加器分辨率倒数。
            min_dist: 最小圆心间距。
            param1: Hough param1（Canny 高阈值）。
            param2: Hough param2（累加器阈值）。
            min_radius: 最小圆半径（像素）。
            max_radius: 最大圆半径（像素）。
            gaussian_kernel: 高斯模糊核大小。
            gaussian_sigma: 高斯模糊 sigma。
        """
        self.binary_threshold = binary_threshold
        self.canny_weak = canny_weak
        self.canny_strong = canny_strong
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        检测图像中的粒子圆。

        Args:
            image: BGR 格式的输入图像。不会被修改。

        Returns:
            形状为 (N, 3) 的 np.ndarray，每行 [x, y, radius] (int)。
            如果未检测到任何圆，返回形状为 (0, 3) 的空数组。
        """
        # 复制输入图像以保持不可变性
        img = image.copy()

        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 二值化
        _, binary = cv2.threshold(
            gray, self.binary_threshold, 255, cv2.THRESH_BINARY
        )

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(binary, self.gaussian_kernel, self.gaussian_sigma)

        # Canny 边缘检测
        edges = cv2.Canny(blurred, self.canny_weak, self.canny_strong)

        # Hough 圆检测
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None:
            return np.empty((0, 3), dtype=int)

        circles = np.round(circles[0, :]).astype("int")
        return circles

    @staticmethod
    def draw_circles(
        image: np.ndarray,
        circles: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        draw_center: bool = True,
    ) -> np.ndarray:
        """
        在图像上绘制检测到的圆（可视化辅助）。

        Args:
            image: 输入图像（不会被修改）。
            circles: detect() 返回的圆数组。
            color: 绘制颜色 (B, G, R)。
            thickness: 圆轮廓线宽。
            draw_center: 是否在圆心画点。

        Returns:
            绘制后的图像副本。
        """
        result = image.copy()
        for x, y, r in circles:
            cv2.circle(result, (int(x), int(y)), int(r), color, thickness)
            if draw_center:
                cv2.circle(result, (int(x), int(y)), 3, color, -1)
        return result

    @classmethod
    def from_config(cls, config: dict) -> "CircleDetector":
        """从配置字典创建 CircleDetector 实例。"""
        cfg = config.get("circle_detection", {})
        return cls(
            binary_threshold=cfg.get("binary_threshold", 130),
            canny_weak=cfg.get("canny_weak", 135),
            canny_strong=cfg.get("canny_strong", 170),
            dp=cfg.get("dp", 1.5),
            min_dist=cfg.get("min_dist", 20),
            param1=cfg.get("param1", 80),
            param2=cfg.get("param2", 16),
            min_radius=cfg.get("min_radius", 6),
            max_radius=cfg.get("max_radius", 12),
        )
