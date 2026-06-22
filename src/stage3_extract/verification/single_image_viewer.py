"""
单张图像参数调优查看器

交互式地在一张代表性图像上逐步调整 mask、threshold、Hough 参数，
直观看到每个参数变化对检测结果的影响。

用于 ARCHITECTURE.md 中定义的「路径A: 参数调优」。
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.stage3_extract.preprocessing.mask import apply_circular_mask
from src.stage3_extract.detection.circle_detector import CircleDetector


def view_single_image(
    image_path: Path,
    config: dict,
    output_dir: Optional[Path] = None,
) -> None:
    """
    交互式单图像查看器。

    展示: 原图 | 蒙版图 | 二值化 | Canny 边缘 | 检测结果
    并在控制台输出检测统计信息。

    Args:
        image_path: 输入图像路径。
        config: 阶段3配置。
        output_dir: 可选，保存标注图像。
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    mask_cfg = config.get("mask", {})
    cx = mask_cfg.get("center_x", 902)
    cy = mask_cfg.get("center_y", 1157)
    r = mask_cfg.get("radius", 450)

    # 1. 蒙版处理
    masked = apply_circular_mask(img, cx, cy, r)

    # 2. 圆检测
    detector = CircleDetector.from_config(config)
    circles = detector.detect(masked)

    # 3. 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 原图
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # 蒙版图
    axes[0, 1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Masked (cx={cx}, cy={cy}, r={r})")
    axes[0, 1].axis("off")

    # 灰度 + 二值化
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, detector.binary_threshold, 255, cv2.THRESH_BINARY)
    axes[0, 2].imshow(binary, cmap="gray")
    axes[0, 2].set_title(f"Binary (th={detector.binary_threshold})")
    axes[0, 2].axis("off")

    # Canny 边缘
    blur = cv2.GaussianBlur(binary, (5, 5), 2)
    edges = cv2.Canny(blur, detector.canny_weak, detector.canny_strong)
    axes[1, 0].imshow(edges, cmap="gray")
    axes[1, 0].set_title(
        f"Canny (weak={detector.canny_weak}, strong={detector.canny_strong})"
    )
    axes[1, 0].axis("off")

    # 检测结果叠加
    result = CircleDetector.draw_circles(masked, circles)
    axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Detected: {len(circles)} circles")
    axes[1, 1].axis("off")

    # 参数信息
    axes[1, 2].axis("off")
    info = (
        f"Detection Parameters:\n\n"
        f"  binary_threshold: {detector.binary_threshold}\n"
        f"  canny_weak: {detector.canny_weak}\n"
        f"  canny_strong: {detector.canny_strong}\n"
        f"  dp: {detector.dp}\n"
        f"  min_dist: {detector.min_dist}\n"
        f"  param1: {detector.param1}\n"
        f"  param2: {detector.param2}\n"
        f"  min_radius: {detector.min_radius}\n"
        f"  max_radius: {detector.max_radius}\n\n"
        f"Result:\n"
        f"  Circles detected: {len(circles)}\n"
    )
    axes[1, 2].text(0.1, 0.9, info, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{image_path.stem}_verification.png", dpi=150)
        print(f"验证图已保存: {output_dir}")

    plt.show()

    # 控制台汇总
    print(f"\n--- 检测汇总 ---")
    print(f"检测到粒子数: {len(circles)}")
    if len(circles) > 0:
        print(f"平均半径: {circles[:, 2].mean():.1f} px")
        print(f"半径范围: [{circles[:, 2].min()}, {circles[:, 2].max()}] px")
