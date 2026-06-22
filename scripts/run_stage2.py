#!/usr/bin/env python3
"""
阶段2 运行入口：硬件实验与数据采集。

Usage:
    python scripts/run_stage2.py              # GUI 模式（在 UI 中手动控制相机）
    python scripts/run_stage2.py --headless   # 无头模式（自动打开相机）
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config
from src.stage2_experiment.pipeline import run_stage2


if __name__ == "__main__":
    headless = "--headless" in sys.argv
    config = get_stage_config(2)
    run_stage2(config, headless=headless)
