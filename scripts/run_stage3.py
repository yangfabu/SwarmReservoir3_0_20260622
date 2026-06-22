#!/usr/bin/env python3
"""
阶段3 运行入口：图像特征提取（批量模式 — 路径C）。

Usage:
    python scripts/run_stage3.py
    python scripts/run_stage3.py --input data/stage2_output/MCTest_20260622/
    python scripts/run_stage3.py --max 100        # 仅处理前100张
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config
from src.stage3_extract.pipeline import run_stage3


def _parse_args():
    """简易命令行参数解析。"""
    args = {"image_dir": None, "max_images": None}
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--input" and i + 1 < len(argv):
            args["image_dir"] = Path(argv[i + 1])
            i += 2
        elif argv[i] == "--max" and i + 1 < len(argv):
            args["max_images"] = int(argv[i + 1])
            i += 2
        else:
            i += 1
    return args


if __name__ == "__main__":
    args = _parse_args()
    config = get_stage_config(3)
    run_stage3(
        config=config,
        image_dir=args["image_dir"],
        max_images=args["max_images"],
    )
