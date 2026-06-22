"""
阶段3 主管线：图像特征提取

流程:
  1. 加载所有图像
  2. 对每张图像:
     a. 应用圆形蒙版
     b. 检测粒子圆
     c. 保存粒子数据到 particles.csv
     d. 对所有启用的特征调用 compute()
     e. 计算邻居距离分布直方图 → distribution.csv
  3. 跨帧粒子速度追踪 → velocity.csv
  4. 汇总特征写入 features.csv (含 mean_velocity)

支持三种运行模式:
  - 批量模式 (run_stage3): 全量处理
  - 验证模式 (run_verification): 单张图逐步调参 (路径A)
  - 测试模式 (run_test): 抽样10张验证逻辑 (路径B)

迁移自: Parameter_extract/Code/Circle_recognize_Main.py + 各个分散的单步脚本
"""

import csv
import sys
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config, resolve_path
from src.common.logging_utils import get_logger
from src.common.io_utils import (
    find_image_files_naturally, load_image, write_csv, read_csv,
)
from src.stage3_extract.preprocessing.mask import apply_circular_mask
from src.stage3_extract.detection.circle_detector import CircleDetector
from src.stage3_extract.features.base import BaseFeature, get_available_features
from src.stage3_extract.features.distribution import (
    DistributionFeature, get_distribution_bin_columns,
)
from src.stage3_extract.features.velocity import (
    VelocityTracker, velocity_feature_name,
)

# 延迟导入特征类（触发 @register_feature 装饰器注册）
from src.stage3_extract.features.entropy import EntropyFeature
from src.stage3_extract.features.system_radius import SystemRadiusFeature
from src.stage3_extract.features.neighbor_spacing import NeighborSpacingFeature


def _build_feature_instances(enabled_names: List[str], config: dict) -> List[BaseFeature]:
    """
    根据启用的特征名称列表创建特征实例。

    Args:
        enabled_names: 如 ["entropy", "system_radius"]。
        config: 阶段3特征配置。

    Returns:
        BaseFeature 实例列表。
    """
    feature_map: dict = {}
    for cls in BaseFeature.__subclasses__():
        instance = cls()
        feature_map[instance.name()] = instance

    instances = []
    for name in enabled_names:
        if name in feature_map:
            instances.append(feature_map[name])
        else:
            print(f"[警告] 未知特征 '{name}'，已跳过。可用: {list(feature_map.keys())}")
    return instances


# ================================================================
# 批量模式 (路径C)
# ================================================================

def run_stage3(
    config: Optional[dict] = None,
    image_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
) -> None:
    """
    执行阶段3：全量特征提取。

    Args:
        config: 阶段3配置（含 global）。为 None 则自动加载。
        image_dir: 图像目录。为 None 则使用 data/stage2_output/ 下最新实验目录。
        max_images: 最多处理的图像数（None = 全部）。

    产物:
        - data/stage3_output/particles.csv
        - data/stage3_output/features.csv
    """
    if config is None:
        config = get_stage_config(3)

    logger = get_logger("stage3")
    input_cfg = config.get("input", {})
    mask_cfg = config.get("mask", {})
    det_cfg = config.get("circle_detection", {})
    feat_cfg = config.get("features", {})
    output_cfg = config.get("output", {})

    # 1. 确定图像目录（优先级: CLI参数 > 配置文件 > 自动查找最新）
    if image_dir is None:
        cfg_img_dir = input_cfg.get("image_dir")
        if cfg_img_dir:
            image_dir = resolve_path(cfg_img_dir)
        else:
            image_dir = _find_latest_experiment_dir(config)

    if not image_dir or not image_dir.exists():
        logger.error(
            f"图像目录不存在: {image_dir}\n"
            f"请在 config/stage3_extract.yaml 的 input.image_dir 中指定正确的路径"
        )
        return

    images = find_image_files_naturally(image_dir)
    logger.info(f"找到 {len(images)} 张图像")

    if max_images is None:
        max_images = input_cfg.get("max_images")
    if max_images:
        images = images[:max_images]
        logger.info(f"限制处理前 {max_images} 张")

    # 2. 初始化检测器和特征
    detector = CircleDetector.from_config(config)
    cx, cy, cr = mask_cfg.get("center_x", 902), mask_cfg.get("center_y", 1157), mask_cfg.get("radius", 450)

    enabled_features = feat_cfg.get("enabled", ["entropy", "system_radius", "neighbor_spacing"])
    features = _build_feature_instances(enabled_features, feat_cfg)
    feature_names = [f.name() for f in features]
    logger.info(f"启用特征: {feature_names}")

    # 3. 输出文件
    particles_csv = resolve_path(output_cfg.get("particles_csv", "data/stage3_output/particles.csv"))
    features_csv = resolve_path(output_cfg.get("features_csv", "data/stage3_output/features.csv"))
    distribution_csv = resolve_path(output_cfg.get("distribution_csv", "data/stage3_output/distribution.csv"))
    velocity_csv = resolve_path(output_cfg.get("velocity_csv", "data/stage3_output/velocity.csv"))

    # 写入 particles.csv 表头
    particles_header = ["original_filename", "file_prefix", "frame", "id", "center_x", "center_y", "radius"]
    with open(particles_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(particles_header)

    # 初始化分布特征
    dist_feature = DistributionFeature()
    dist_bin_columns = get_distribution_bin_columns(config)
    all_dist_rows = []   # 每帧的直方图行

    # 初始化速度追踪的缓存
    particles_cache: List[pd.DataFrame] = []
    image_filenames: List[str] = []

    all_feature_rows = []
    anomaly_frames = []

    # 4. 逐帧处理
    for idx, img_path in enumerate(images):
        img = load_image(img_path)
        if img is None:
            continue

        # a. 蒙版
        masked = apply_circular_mask(img, cx, cy, cr)

        # b. 检测
        circles = detector.detect(masked)
        frame_id = idx + 1

        # 从文件名提取前缀
        prefix = _extract_prefix(img_path.name)

        # c. 写入 particles.csv
        _append_particles(particles_csv, img_path.name, prefix, frame_id, circles)

        # 缓存粒子数据和文件名（用于速度追踪）
        image_filenames.append(img_path.name)

        # d. 计算特征 + 分布
        if len(circles) == 0:
            anomaly_frames.append(frame_id)
            row = {"frame": frame_id}
            for f in features:
                row[f.name()] = np.nan
            all_feature_rows.append(row)
            particles_cache.append(pd.DataFrame(columns=["Center_X", "Center_Y", "Radius"]))
            all_dist_rows.append({"frame": frame_id})
            continue

        particles_df = pd.DataFrame({
            "Center_X": circles[:, 0],
            "Center_Y": circles[:, 1],
            "Radius": circles[:, 2],
        })
        particles_cache.append(particles_df)

        # 标量特征
        row = {"frame": frame_id}
        for feature in features:
            try:
                value = feature.compute(particles_df, frame_id, feat_cfg)
            except Exception as e:
                logger.warning(f"特征 {feature.name()} 帧 {frame_id} 计算失败: {e}")
                value = np.nan
            row[feature.name()] = value
        all_feature_rows.append(row)

        # 邻居距离分布直方图
        _, hist_counts, bin_labels = dist_feature.compute_histogram(particles_df, config)
        if hist_counts is not None and len(bin_labels) == len(hist_counts):
            dist_row = {"frame": frame_id}
            for bl, hc in zip(bin_labels, hist_counts):
                dist_row[bl] = int(hc)
            all_dist_rows.append(dist_row)
        else:
            all_dist_rows.append({"frame": frame_id})

        if (idx + 1) % 100 == 0:
            logger.info(f"进度: {idx + 1}/{len(images)} 帧")

    # 5. 保存 distribution.csv
    if all_dist_rows and dist_bin_columns:
        df_dist = pd.DataFrame(all_dist_rows)
        # 确保所有 bin 列都存在
        for col in dist_bin_columns:
            if col not in df_dist.columns:
                df_dist[col] = 0
        # 按 frame, bin_columns 顺序排列
        ordered_cols = ["frame"] + [c for c in dist_bin_columns if c in df_dist.columns]
        df_dist = df_dist[ordered_cols]
        write_csv(df_dist, distribution_csv)
        logger.info(f"分布已保存: {distribution_csv}")
    else:
        logger.warning("无分布数据可保存")

    # 6. 速度追踪（跨帧粒子匹配）
    vel_cfg = config.get("velocity", {})
    if len(particles_cache) >= 2:
        tracker = VelocityTracker(
            max_displacement=vel_cfg.get("max_displacement", 50.0),
            min_matched_pairs=vel_cfg.get("min_matched_pairs", 3),
            fallback_fps=vel_cfg.get("fps"),
        )

        velocity_df, frame_stats = tracker.track_all_frames(particles_cache, image_filenames)

        # 保存 velocity.csv
        if not velocity_df.empty:
            vel_cols = [
                "frame", "prev_particle_id", "curr_particle_id",
                "dx", "dy", "distance_px", "velocity_px_per_s", "dt_ms",
                "prev_filename", "curr_filename",
            ]
            velocity_df = velocity_df[vel_cols]
            write_csv(velocity_df, velocity_csv)
            logger.info(f"速度数据已保存: {velocity_csv} ({len(velocity_df)} 对匹配)")
        else:
            logger.warning("无有效的粒子匹配对，velocity.csv 未生成")

        # 将 mean_velocity 合并到 features.csv
        vel_name = velocity_feature_name()
        vel_map = {s["frame"]: s["mean_velocity"] for s in frame_stats}
        for row in all_feature_rows:
            fid = row["frame"]
            row[vel_name] = vel_map.get(fid, np.nan)
        logger.info(f"速度特征 '{vel_name}' 已合并到特征表")
    else:
        logger.warning(f"帧数不足 ({len(particles_cache)}), 跳过速度追踪")

    # 7. 保存 features.csv
    df_features = pd.DataFrame(all_feature_rows)
    write_csv(df_features, features_csv)
    logger.info(f"特征已保存: {features_csv}")

    # 8. 汇总
    total_particles = sum(1 for _ in pd.read_csv(particles_csv).iterrows())
    logger.info(f"阶段3 完成: {len(images)} 帧, {total_particles} 个粒子检测")
    if anomaly_frames:
        logger.warning(f"异常帧 (粒子数=0): {anomaly_frames}")
    logger.info(f"平均粒子数: {total_particles / len(images):.1f}")
    if all_dist_rows:
        logger.info(f"分布数据: {len(all_dist_rows)} 帧")


def _append_particles(csv_path: Path, filename: str, prefix: str, frame: int, circles: np.ndarray) -> None:
    """追加粒子行到 particles.csv。"""
    import csv
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for pid, (cx, cy, cr) in enumerate(circles, start=1):
            writer.writerow([filename, prefix, frame, pid, int(cx), int(cy), int(cr)])


def _extract_prefix(filename: str) -> str:
    """从文件名提取前缀（如 001_000000_858.jpg → 001）。"""
    parts = filename.replace(".jpg", "").replace(".bmp", "").split("_")
    return parts[0] if parts else "0"


def _find_latest_experiment_dir(config: dict) -> Optional[Path]:
    """在 data/stage2_output/ 下查找最新实验目录。"""
    stage2_root = resolve_path("data/stage2_output")
    if not stage2_root.exists():
        return None
    subdirs = sorted(
        [d for d in stage2_root.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return subdirs[0] if subdirs else None


# ================================================================
# 验证模式 (路径A)
# ================================================================

def run_verification(config: Optional[dict] = None, image_path: Optional[Path] = None) -> None:
    """
    单张图像参数调优查看器。

    Args:
        config: 阶段3配置。
        image_path: 测试图像路径。
    """
    if config is None:
        config = get_stage_config(3)

    from src.stage3_extract.verification.single_image_viewer import view_single_image

    if image_path is None:
        logger = get_logger("stage3")
        logger.error("请指定 --image 参数")
        return

    view_single_image(image_path, config)


# ================================================================
# 单步测试模式 (路径B)
# ================================================================

def run_test(
    config: Optional[dict] = None,
    image_dir: Optional[Path] = None,
    n_samples: int = 10,
) -> bool:
    """
    抽样 N 张图像快速验证管线逻辑。

    Args:
        config: 阶段3配置。
        image_dir: 图像目录。
        n_samples: 随机抽样的图像数。

    Returns:
        True 表示测试通过（无异常）。
    """
    if config is None:
        config = get_stage_config(3)

    logger = get_logger("stage3")
    input_cfg = config.get("input", {})

    if image_dir is None:
        cfg_img_dir = input_cfg.get("image_dir")
        if cfg_img_dir:
            image_dir = resolve_path(cfg_img_dir)
        else:
            image_dir = _find_latest_experiment_dir(config)

    if not image_dir or not image_dir.exists():
        logger.error(
            f"图像目录不存在: {image_dir}\n"
            f"请在 config/stage3_extract.yaml 的 input.image_dir 中指定正确的路径"
        )
        return False

    all_images = find_image_files_naturally(image_dir)
    sampled = random.sample(all_images, min(n_samples, len(all_images)))
    logger.info(f"随机抽样 {len(sampled)} 张图像 (共 {len(all_images)} 张)")

    mask_cfg = config.get("mask", {})
    cx, cy, cr = mask_cfg.get("center_x", 902), mask_cfg.get("center_y", 1157), mask_cfg.get("radius", 450)

    detector = CircleDetector.from_config(config)
    feat_cfg = config.get("features", {})
    enabled_names = feat_cfg.get("enabled", ["entropy"])
    features = _build_feature_instances(enabled_names, feat_cfg)

    # 分布和速度
    dist_feature = DistributionFeature()
    vel_cfg = config.get("velocity", {})

    results = []
    particles_cache = []
    filenames_cache = []
    for img_path in sampled:
        img = load_image(img_path)
        if img is None:
            continue
        masked = apply_circular_mask(img, cx, cy, cr)
        circles = detector.detect(masked)

        particles_df = pd.DataFrame({
            "Center_X": circles[:, 0],
            "Center_Y": circles[:, 1],
            "Radius": circles[:, 2],
        }) if len(circles) > 0 else pd.DataFrame()

        particles_cache.append(particles_df)
        filenames_cache.append(img_path.name)

        feat_values = {}
        for feature in features:
            if len(circles) > 0:
                feat_values[feature.name()] = feature.compute(particles_df, 0, feat_cfg)
            else:
                feat_values[feature.name()] = np.nan

        # 分布直方图
        _, hist_counts, _ = dist_feature.compute_histogram(particles_df, config)
        dist_summary = f"hist_bins={len(hist_counts)}" if hist_counts is not None else "N/A"

        results.append({
            "file": img_path.name,
            "circles": len(circles),
            **feat_values,
        })

    df = pd.DataFrame(results)

    # 检查
    has_anomaly = (df["circles"] == 0).any()
    has_nan = df[[f.name() for f in features]].isna().any().any()

    print("\n--- 测试结果 ---")
    print(df.to_string())
    print(f"\n异常帧 (粒子数=0): {(df['circles'] == 0).sum()} / {len(df)}")
    print(f"NaN 特征值: {'有' if has_nan else '无'}")
    print(f"粒子数范围: {df['circles'].min()} - {df['circles'].max()}")
    print(f"平均粒子数: {df['circles'].mean():.1f}")

    # 速度测试
    if len(particles_cache) >= 2:
        tracker = VelocityTracker(
            max_displacement=vel_cfg.get("max_displacement", 50.0),
            min_matched_pairs=vel_cfg.get("min_matched_pairs", 3),
            fallback_fps=vel_cfg.get("fps"),
        )
        vel_df, frame_stats = tracker.track_all_frames(particles_cache, filenames_cache)
        if not vel_df.empty:
            velocities = vel_df["velocity_px_per_s"].dropna()
            print(f"\n速度追踪: {len(vel_df)} 对匹配, "
                  f"平均速度: {velocities.mean():.1f} px/s, "
                  f"中位速度: {velocities.median():.1f} px/s")
        else:
            print("\n速度追踪: 无有效匹配对")
    else:
        print(f"\n速度追踪: 帧数不足 ({len(particles_cache)}), 跳过")

    if has_anomaly or has_nan:
        print("⚠ 测试发现异常！建议回到路径A（参数调优）。")
        return False

    print("✓ 测试通过！可以进入批量运行 (路径C)。")
    return True
