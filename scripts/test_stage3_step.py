#!/usr/bin/env python3
"""
阶段3 单步测试入口：随机抽样 N 张验证管线逻辑（路径B）。

Usage:
    python scripts/test_stage3_step.py
    python scripts/test_stage3_step.py --samples 20
    python scripts/test_stage3_step.py --input data/stage2_output/MyExperiment/
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config
from src.stage3_extract.pipeline import run_test


if __name__ == "__main__":
    n_samples = 10
    image_dir = None

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--samples" and i + 1 < len(argv):
            n_samples = int(argv[i + 1])
            i += 2
        elif argv[i] == "--input" and i + 1 < len(argv):
            image_dir = Path(argv[i + 1])
            i += 2
        else:
            i += 1

    config = get_stage_config(3)
    success = run_test(
        config=config,
        image_dir=image_dir,
        n_samples=n_samples,
    )
    sys.exit(0 if success else 1)
