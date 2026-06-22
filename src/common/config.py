"""
集中配置加载器 (Singleton)

加载 config/ 目录下的所有 YAML 文件，提供统一的配置访问接口。
路径解析基于项目根目录（src/common/config.py 向上两级）。
"""

from pathlib import Path
from typing import Dict, Any, Optional

import yaml


class ConfigError(Exception):
    """配置文件相关错误。"""
    pass


class ConfigLoader:
    """
    单例配置加载器。

    在首次调用时加载全部 5 个 YAML 文件，合并为统一字典，
    后续调用返回缓存结果。使用 reload() 强制重新加载。
    """

    _instance: Optional["ConfigLoader"] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _project_root() -> Path:
        """返回项目根目录 (Swarm_reservoir3_0/)."""
        return Path(__file__).resolve().parent.parent.parent

    @staticmethod
    def _config_dir() -> Path:
        """返回 config/ 目录。"""
        return ConfigLoader._project_root() / "config"

    @classmethod
    def load_all(cls) -> Dict[str, Any]:
        """
        加载全部配置文件，合并为一个嵌套字典。

        Returns:
            {
                "global": {...},
                "stage1": {...},
                "stage2": {...},
                "stage3": {...},
                "stage4": {...},
            }
        """
        config_dir = cls._config_dir()
        config: Dict[str, Any] = {}

        stage_files = {
            "global": "global.yaml",
            "stage1": "stage1_generate.yaml",
            "stage2": "stage2_experiment.yaml",
            "stage3": "stage3_extract.yaml",
            "stage4": "stage4_benchmark.yaml",
        }

        for key, filename in stage_files.items():
            file_path = config_dir / filename
            if not file_path.exists():
                raise ConfigError(
                    f"配置文件不存在: {file_path}\n"
                    f"请确保 {filename} 存在于 {config_dir}"
                )
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    config[key] = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigError(f"解析 YAML 文件失败: {file_path}\n{e}")

        cls._config = config
        return config

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取完整配置（自动加载，缓存结果）。"""
        if cls._config is None:
            cls.load_all()
        assert cls._config is not None
        return cls._config

    @classmethod
    def get_stage_config(cls, stage: int) -> Dict[str, Any]:
        """
        获取指定阶段的配置，并自动合并 global 配置。

        Args:
            stage: 阶段编号 (1-4)。

        Returns:
            合并后的配置字典，包含 'global' 键和阶段特定键。
        """
        config = cls.get_config()
        stage_key = f"stage{stage}"
        if stage_key not in config:
            raise ConfigError(f"未知的阶段: {stage}，可用: stage1 ~ stage4")
        merged = {"global": config.get("global", {})}
        merged.update(config[stage_key])
        return merged

    @classmethod
    def resolve_path(cls, relative_path: str) -> Path:
        """
        将配置中的相对路径解析为绝对路径。

        Args:
            relative_path: 以项目根为基准的相对路径。

        Returns:
            绝对路径。
        """
        return cls._project_root() / relative_path

    @classmethod
    def reload(cls) -> Dict[str, Any]:
        """强制重新加载所有配置文件。"""
        cls._config = None
        return cls.load_all()


# 模块级便捷访问函数
def get_config() -> Dict[str, Any]:
    """获取完整配置。"""
    return ConfigLoader.get_config()


def get_stage_config(stage: int) -> Dict[str, Any]:
    """获取指定阶段配置（含 global）。"""
    return ConfigLoader.get_stage_config(stage)


def resolve_path(relative_path: str) -> Path:
    """将相对路径解析为绝对路径。"""
    return ConfigLoader.resolve_path(relative_path)
