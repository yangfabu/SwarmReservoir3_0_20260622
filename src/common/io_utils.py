"""
文件 I/O 工具集

整合自旧项目 Parameter_extract/utils/ 下的各个 CSV 和图像处理工具。
提供 CSV 读写、图像批量 IO、文件查找等通用功能。
"""

from pathlib import Path
from typing import List, Optional, Tuple, Any
import re

import numpy as np
import pandas as pd
import cv2


# ==============================================================================
# CSV 操作
# ==============================================================================

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    安全读取 CSV 文件。

    Args:
        path: CSV 文件路径。
        **kwargs: 传递给 pd.read_csv 的额外参数。

    Returns:
        DataFrame。
    """
    defaults = {"encoding": "utf-8-sig"}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def write_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """
    安全写入 CSV 文件（自动创建父目录）。

    Args:
        df: 要写入的 DataFrame。
        path: 目标路径。
        **kwargs: 传递给 df.to_csv 的额外参数。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = {"index": False, "encoding": "utf-8-sig"}
    defaults.update(kwargs)
    df.to_csv(path, **defaults)


def combine_csv_files(
    file1: Path, file2: Path, output: Path, on: Optional[str] = None
) -> pd.DataFrame:
    """
    合并两个 CSV 文件。

    Args:
        file1: 第一个 CSV。
        file2: 第二个 CSV。
        output: 输出路径。
        on: 合并键列名。

    Returns:
        合并后的 DataFrame。
    """
    df1 = read_csv(file1)
    df2 = read_csv(file2)
    merged = pd.merge(df1, df2, on=on) if on else pd.concat([df1, df2], axis=1)
    write_csv(merged, output)
    return merged


def filter_csv_by_frame_range(
    input_csv: Path, output_csv: Path, min_frame: int, max_frame: int
) -> pd.DataFrame:
    """
    保留指定帧范围内的数据并重新编号。

    Args:
        input_csv: 输入的 CSV (需含 'Frame' 列)。
        output_csv: 输出路径。
        min_frame: 最小帧号。
        max_frame: 最大帧号。

    Returns:
        过滤后的 DataFrame。
    """
    df = read_csv(input_csv)
    df_filtered = df[(df["Frame"] >= min_frame) & (df["Frame"] <= max_frame)].copy()
    df_filtered["Frame"] = range(1, len(df_filtered) + 1)
    write_csv(df_filtered, output_csv)
    return df_filtered


def delete_columns_from_csv(input_csv: Path, cols: List[str], output: Path) -> None:
    """从 CSV 中删除指定列。"""
    df = read_csv(input_csv)
    df = df.drop(columns=[c for c in cols if c in df.columns])
    write_csv(df, output)


# ==============================================================================
# 图像 I/O
# ==============================================================================

def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    安全加载图像。

    Args:
        image_path: 图像文件路径。

    Returns:
        BGR 格式的 numpy 数组，失败则返回 None。
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[警告] 无法读取图像: {image_path}")
    return img


def save_image(image: np.ndarray, output_path: Path) -> bool:
    """
    安全保存图像（自动创建父目录）。

    Args:
        image: 要保存的图像 (numpy array)。
        output_path: 输出路径。

    Returns:
        成功返回 True。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), image)


# ==============================================================================
# 文件查找
# ==============================================================================

def find_image_files(
    directory: Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    pattern: Optional[str] = None,
) -> List[Path]:
    """
    在目录中查找图像文件，按自然排序返回。

    Args:
        directory: 搜索目录。
        extensions: 接受的扩展名。
        pattern: 可选的 glob 模式过滤（如 "*.jpg"）。

    Returns:
        排序后的 Path 列表。
    """
    if not directory.exists():
        return []

    files: List[Path] = []
    if pattern:
        files = sorted(directory.glob(pattern))
    else:
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        files = sorted(files)

    return files


def _natural_sort_key(path: Path) -> tuple:
    """
    从文件名提取所有数字作为排序键（元组），确保帧序列正确排序。

    支持格式:
      - 旧格式: 001_000000_858.jpg       → (1, 0, 858)
      - 新格式: 001_000000_20260622184622360.jpg → (1, 0, 20260622184622360)
    """
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        return (0,)
    return tuple(int(n) for n in numbers)


def parse_timestamp_from_filename(filename: str) -> Optional[Any]:
    """
    从图像文件名中提取绝对时间戳。

    支持格式:
      - 新格式: {prefix}_{frame}_{YYYYMMDDHHMMSSmmm}.jpg  (17位时间戳)
      - 旧格式: {prefix}_{frame}_{ms}.jpg                  (1-3位，无绝对时间)

    Args:
        filename: 文件名（可含路径）。

    Returns:
        datetime 对象，或 None。
    """
    from datetime import datetime

    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) >= 3:
        last_part = parts[-1]
        # 17 位数字 = YYYYMMDDHHMMSSmmm
        if last_part.isdigit() and len(last_part) == 17:
            try:
                return datetime.strptime(last_part, "%Y%m%d%H%M%S%f")
            except ValueError:
                pass

    return None


def find_image_files_naturally(
    directory: Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> List[Path]:
    """
    在目录中查找图像文件，按文件名中的数字自然排序。

    适用于帧序列文件，如: 001_000000_858.jpg, 001_000001_875.jpg。
    也支持新格式: 001_000000_20260622184622360.jpg。

    Args:
        directory: 搜索目录。
        extensions: 接受的扩展名。

    Returns:
        自然排序后的 Path 列表。
    """
    files = find_image_files(directory, extensions)
    files.sort(key=_natural_sort_key)
    return files
