"""
特征提取抽象基类

所有特征计算类必须继承 BaseFeature，实现 name() 和 compute()。
管线通过特征名称自动发现和调用。
"""

from abc import ABC, abstractmethod
from typing import Dict, Type

import numpy as np
import pandas as pd


class BaseFeature(ABC):
    """粒子特征提取的抽象基类。"""

    @abstractmethod
    def name(self) -> str:
        """
        返回特征的唯一标识符。

        Returns:
            特征名称，如 "entropy", "system_radius"。
            该名称对应 config/stage3_extract.yaml 中 features.enabled 列表。
        """
        ...

    @abstractmethod
    def compute(
        self,
        particles_df: pd.DataFrame,
        frame_id: int,
        config: dict,
    ) -> float:
        """
        对单帧的粒子数据计算一个标量特征值。

        Args:
            particles_df: 单帧的粒子数据 DataFrame，
                          至少包含 ['Center_X', 'Center_Y', 'Radius'] 列。
            frame_id: 帧编号。
            config: 该特征的配置节。

        Returns:
            标量特征值。如果无法计算（粒子数不足等），返回 np.nan。
        """
        ...

    def validate(self, particles_df: pd.DataFrame, config: dict) -> bool:
        """
        检查当前帧是否满足特征计算的最低条件。

        Args:
            particles_df: 单帧粒子数据。
            config: 特征配置。

        Returns:
            True 表示可以计算。
        """
        min_particles = config.get("min_particles", 5)
        return len(particles_df) >= min_particles


# ================================================================
# 特征注册表
# ================================================================

_feature_registry: Dict[str, Type[BaseFeature]] = {}


def register_feature(cls: Type[BaseFeature]) -> Type[BaseFeature]:
    """
    装饰器：注册特征类。

    Usage:
        @register_feature
        class MyFeature(BaseFeature):
            def name(self) -> str:
                return "my_feature"
            ...
    """
    # 用 name() 注册需要实例，这里用类名推断
    return cls


def get_feature_class(name: str) -> Type[BaseFeature]:
    """根据特征名称查找类。"""
    for cls in _feature_registry.values():
        instance = cls()
        if instance.name() == name:
            return cls
    available = [cls().name() for cls in _feature_registry.values()]
    raise ValueError(
        f"未知的特征: '{name}'。可用特征: {available}"
    )


def get_available_features() -> list:
    """返回所有已注册特征的名称列表。"""
    return [cls().name() for cls in _feature_registry.values()]
