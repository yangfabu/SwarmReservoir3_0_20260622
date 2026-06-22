#!/usr/bin/env python3
"""
阶段3 验证入口：单张图参数调优（路径A）。

Usage:
    python scripts/verify_stage3.py --image path/to/frame.jpg
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config
from src.stage3_extract.pipeline import run_verification


if __name__ == "__main__":
    image_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--image" and i + 1 < len(sys.argv):
            image_path = Path(sys.argv[i + 1])

    if image_path is None:
        print("错误：请指定 --image 参数")
        print("用法：python scripts/verify_stage3.py --image path/to/frame.jpg")
        sys.exit(1)

    config = get_stage_config(3)
    run_verification(config=config, image_path=image_path)
