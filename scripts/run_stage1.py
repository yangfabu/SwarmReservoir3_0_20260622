#!/usr/bin/env python3
"""
阶段1 运行入口：生成输入电流序列。

Usage:
    python scripts/run_stage1.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config
from src.stage1_generate.pipeline import run_stage1


if __name__ == "__main__":
    config = get_stage_config(1)
    run_stage1(config)
