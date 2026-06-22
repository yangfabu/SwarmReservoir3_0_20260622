"""
阶段4 数据预处理

将阶段3产出的 features.csv 转换为基准测试可用的格式。

流程:
  1. flatten: 将每周期多帧的特征平铺为一行 (N_frames × N_features 维)
  2. normalize: StandardScaler 标准化
  3. PCA: 降维（可选）

合并自: Benchmark/flatten_data.py + normalize_data.py + entropy.py + radius.py
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def flatten_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    frames_per_group: int,
) -> pd.DataFrame:
    """
    将特征 DataFrame 平铺：每组 frames_per_group 行 → 一行。

    例如 features.csv 每 22 帧为一个周期:
      原始: (N×22, 2)  → 平铺后: (N, 22×2) = (N, 44)

    Args:
        df: 包含 Frame 列和特征列的 DataFrame。
        feature_cols: 需要平铺的特征列名列表。
        frames_per_group: 每组行数（每周期帧数）。

    Returns:
        平铺后的 DataFrame，列名为 F1_entropy, F1_system_radius, F2_entropy, ...
    """
    data_values = df[feature_cols].values
    total_rows = data_values.shape[0]
    num_groups = total_rows // frames_per_group

    if total_rows % frames_per_group != 0:
        print(f"警告: 总行数 {total_rows} 不是 {frames_per_group} 的倍数，截断末尾。")
        data_values = data_values[:num_groups * frames_per_group]

    # reshape: (num_groups, frames_per_group * num_features)
    n_features = len(feature_cols)
    flattened = data_values.reshape(num_groups, frames_per_group * n_features)

    # 构造列名
    columns = []
    for i in range(1, frames_per_group + 1):
        for col in feature_cols:
            columns.append(f"F{i}_{col}")

    result = pd.DataFrame(flattened, columns=columns)
    result.insert(0, "Frame", range(1, num_groups + 1))
    return result


def normalize_features(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    使用 StandardScaler 标准化特征列。

    Args:
        df: 输入 DataFrame。
        exclude_cols: 不参与标准化的列名列表（如 "Frame"）。

    Returns:
        标准化后的 DataFrame（排除列保持不变）。
    """
    if exclude_cols is None:
        exclude_cols = ["Frame"]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    scaler = StandardScaler()
    df_result = df.copy()
    df_result[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 验证
    means = df_result[feature_cols].mean()
    stds = df_result[feature_cols].std()
    print(f"标准化: 均值范围 [{means.min():.4f}, {means.max():.4f}], "
          f"标准差范围 [{stds.min():.4f}, {stds.max():.4f}]")

    return df_result


def apply_pca(
    X: np.ndarray,
    variance_threshold: float = 0.95,
) -> Tuple[np.ndarray, PCA]:
    """
    应用 PCA 降维。

    Args:
        X: 输入特征矩阵 (N, D)。
        variance_threshold: 保留的方差比例。

    Returns:
        (X_reduced, pca_object)
    """
    pca = PCA(n_components=variance_threshold)
    X_reduced = pca.fit_transform(X)
    print(f"PCA: {X.shape[1]} 维 → {X_reduced.shape[1]} 维 "
          f"(保留 {variance_threshold * 100:.0f}% 方差)")
    return X_reduced, pca


def preprocess_pipeline(
    features_csv: Path,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, Optional[PCA]]:
    """
    完整预处理管线: load → flatten → normalize → PCA。

    Args:
        features_csv: 阶段3产出的 features.csv 路径。
        config: 阶段4配置。

    Returns:
        (X, u_t, pca_object) — X 是预处理后的特征矩阵，u_t 是目标值。
        pca_object 为 None 表示跳过 PCA。
    """
    preproc_cfg = config.get("preprocessing", {})

    # 1. 加载
    df = pd.read_csv(features_csv)
    feature_cols = preproc_cfg.get("features", ["entropy", "system_radius"])
    frames_per_group = preproc_cfg.get("frames_per_group", 22)

    # 确保特征列存在
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"features.csv 中缺少列: {missing}。可用列: {list(df.columns)}")

    # 2. Flatten
    df_flat = flatten_features(df, feature_cols, frames_per_group)
    print(f"平铺: {df.shape[0]} 行 → {df_flat.shape[0]} 行 × {df_flat.shape[1] - 1} 特征")

    # 3. Normalize
    if preproc_cfg.get("normalize", True):
        df_flat = normalize_features(df_flat, exclude_cols=["Frame"])

    # 4. PCA
    feature_matrix = df_flat.drop(columns=["Frame"]).values
    pca_obj = None
    if preproc_cfg.get("pca", {}).get("enabled", True):
        threshold = preproc_cfg.get("pca", {}).get("variance_threshold", 0.95)
        feature_matrix, pca_obj = apply_pca(feature_matrix, threshold)

    # 5. 目标值: 使用平铺后的 Frame 编号作为目标（可扩展为从 metadata 加载）
    u_t = df_flat["Frame"].values.astype(float)

    return feature_matrix, u_t, pca_obj
