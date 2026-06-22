"""
培养皿圆心检测

自动使用霍夫圆检测定位培养皿，也支持手动点击获取坐标。

迁移自: Parameter_extract/Code/Find_Circle_Center.py
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def find_petri_dish_center(
    image_path: Path,
    output_path: Optional[Path] = None,
) -> Optional[Tuple[int, int, int]]:
    """
    自动检测培养皿的圆心和半径。

    使用霍夫圆检测找到图像中最大的完整圆（即培养皿）。

    Args:
        image_path: 输入图像路径。
        output_path: 可选的输出图像路径（绘制检测结果）。

    Returns:
        (center_x, center_y, radius) 或 None（未找到）。
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (11, 11), 2)

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=200,
        param1=100,
        param2=30,
        minRadius=300,
        maxRadius=1000,
    )

    if circles is None:
        print("未检测到培养皿圆形。")
        return None

    circles = np.uint16(np.around(circles))
    largest_circle = None
    max_r = 0

    for cx, cy, r in circles[0, :]:
        # 只接受完全在图像内的圆
        if (cx - r >= 0) and (cy - r >= 0) and (cx + r <= w) and (cy + r <= h):
            if r > max_r:
                max_r = r
                largest_circle = (int(cx), int(cy), int(r))

    if largest_circle and output_path:
        cx, cy, r = largest_circle
        result_img = img.copy()
        cv2.circle(result_img, (cx, cy), r, (0, 255, 0), 5)
        cv2.circle(result_img, (cx, cy), 15, (0, 0, 255), -1)
        cv2.putText(
            result_img, f"Center: ({cx}, {cy})",
            (cx - 100, cy - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4,
        )
        cv2.imwrite(str(output_path), result_img)
        print(f"结果已保存: {output_path}")

    if largest_circle:
        cx, cy, r = largest_circle
        print(f"培养皿检测完成: 圆心=({cx}, {cy}), 半径={r}")
    return largest_circle


def manual_center_selector(image_path: Path) -> Tuple[int, int]:
    """
    交互式手动点选培养皿圆心。

    点击图像任意位置获取像素坐标，按 ESC 退出。

    Args:
        image_path: 输入图像路径。

    Returns:
        (x, y) 最后一个点击的坐标。
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    last_click = (0, 0)

    def on_click(event, x, y, flags, param):
        nonlocal last_click
        if event == cv2.EVENT_LBUTTONDOWN:
            last_click = (x, y)
            print(f"点击坐标: x={x}, y={y}")
            img_copy = img.copy()
            cv2.circle(img_copy, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(
                img_copy, f"({x},{y})", (x + 20, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
            )
            cv2.imshow("Image", img_copy)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", on_click)
    print("--- 操作提示 ---")
    print("1. 点击鼠标左键获取坐标")
    print("2. 按 ESC 退出")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return last_click
