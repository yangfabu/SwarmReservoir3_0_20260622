#!/usr/bin/env python3
"""
阶段4 运行入口：储层计算基准测试。

Usage:
    python scripts/run_stage4.py
    python scripts/run_stage4.py --input data/stage3_output/features.csv
    python scripts/run_stage4.py --skip-preprocess   # 跳过预处理
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config
from src.stage4_benchmark.pipeline import run_stage4


if __name__ == "__main__":
    skip_pre = "--skip-preprocess" in sys.argv
    features_csv = None
    for i, arg in enumerate(sys.argv):
        if arg == "--input" and i + 1 < len(sys.argv):
            features_csv = Path(sys.argv[i + 1])

    config = get_stage_config(4)
    run_stage4(
        config=config,
        features_csv=features_csv,
        skip_preprocess=skip_pre,
    )
