"""
阶段3 单元测试：图像预处理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from src.stage3_extract.preprocessing.mask import apply_circular_mask


class TestCircularMask:
    """测试圆形蒙版。"""

    def test_mask_dims_preserved(self):
        """蒙版后图像尺寸不变。"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = apply_circular_mask(img, 50, 50, 30)
        assert result.shape == (100, 100, 3)

    def test_outside_circle_is_white(self):
        """圆外区域应为白色。"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = apply_circular_mask(img, 50, 50, 20)
        # 角落应该在圆外
        assert np.all(result[0, 0] == [255, 255, 255])
        assert np.all(result[99, 99] == [255, 255, 255])

    def test_inside_circle_unchanged(self):
        """圆心处应保持原值。"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = apply_circular_mask(img, 50, 50, 30)
        # 圆心在圆内，应保持原图值
        assert np.all(result[50, 50] == [0, 0, 0])

    def test_input_not_mutated(self):
        """原图不应被修改（纯函数）。"""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        original = img.copy()
        apply_circular_mask(img, 25, 25, 15)
        assert np.array_equal(img, original)
