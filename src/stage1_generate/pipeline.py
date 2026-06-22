"""
阶段1 主管线：输入信号生成

流程:
  1. 加载阶段1配置
  2. 根据 config 选择信号生成策略
  3. 生成原始信号
  4. 硬件精度量化
  5. 导出 current_sequence.csv 和 mc_metadata_log.csv

迁移自: Benchmark/Data_generate.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# 确保项目根在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config, resolve_path
from src.common.logging_utils import get_logger
from src.common.io_utils import write_csv
from src.stage1_generate.signal_generator import get_strategy
from src.stage1_generate.quantization import quantize_current


def run_stage1(config: dict | None = None) -> None:
    """
    执行阶段1：生成电流序列。

    Args:
        config: 阶段1配置（含 global）。为 None 则自动加载。

    产物:
        - data/stage1_output/current_sequence.csv
        - data/stage1_output/mc_metadata_log.csv
    """
    if config is None:
        config = get_stage_config(1)

    logger = get_logger("stage1")
    global_cfg = config.get("global", {})
    signal_cfg = config.get("signal", {})
    output_cfg = config.get("output", {})

    # 1. 初始化随机数生成器
    seed = global_cfg.get("project", {}).get("random_seed", 42)
    rng = np.random.default_rng(seed)

    # 2. 选择策略
    strategy_name = signal_cfg.get("strategy", "uniform_random")
    strategy = get_strategy(strategy_name)
    logger.info(f"使用信号策略: {strategy_name}")

    # 3. 生成原始信号
    n_steps = signal_cfg.get("n_steps", 100)
    raw_signal = strategy.generate(n_steps, signal_cfg, rng)

    # 4. 量化
    decimal_places = signal_cfg.get("decimal_places", 2)
    applied_current = quantize_current(raw_signal, decimal_places)

    # 5. 导出 CSV
    # current_sequence.csv: step_index, target_u, applied_current_A
    current_list_path = resolve_path(
        output_cfg.get("current_list_file", "data/stage1_output/current_sequence.csv")
    )
    df_current = pd.DataFrame({
        "step_index": np.arange(n_steps),
        "target_u": raw_signal,
        "applied_current_A": applied_current,
    })

    write_csv(df_current, current_list_path)
    logger.info(f"电流序列已保存: {current_list_path}")

    # mc_metadata_log.csv: 同上格式（完整元数据）
    metadata_path = resolve_path(
        output_cfg.get("metadata_file", "data/stage1_output/mc_metadata_log.csv")
    )
    write_csv(df_current, metadata_path)
    logger.info(f"元数据日志已保存: {metadata_path}")

    # 汇总
    logger.info(
        f"阶段1 完成: {n_steps} 步, "
        f"范围 [{applied_current.min():.2f}, {applied_current.max():.2f}] A, "
        f"精度 {decimal_places} 位小数"
    )
