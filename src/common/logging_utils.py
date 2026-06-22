"""
统一日志工具

提供毫秒级时间戳的统一日志格式，支持文件和控制台双输出。
日志格式来自 config/global.yaml 的 logging 节。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "DEBUG",
    log_to_console: bool = True,
) -> logging.Logger:
    """
    创建并配置一个 Logger 实例。

    Args:
        name: Logger 名称（通常使用 __name__ 或模块名）。
        log_file: 日志文件路径。为 None 则不输出到文件。
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)。
        log_to_console: 是否同时输出到控制台。

    Returns:
        配置完成的 logging.Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    # 避免重复添加 handler（幂等）
    if logger.handlers:
        return logger

    # 日志格式：毫秒级时间戳
    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件处理器
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取或创建 logger（简易版，仅控制台输出）。

    Args:
        name: Logger 名称。

    Returns:
        logging.Logger 实例。
    """
    return setup_logger(name)
