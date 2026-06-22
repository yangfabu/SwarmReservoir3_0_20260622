#!/usr/bin/env python3
"""
一键运行全管线。

流程: 阶段1 → 阶段2 (需确认硬件) → 阶段3 → 阶段4

Usage:
    python scripts/run_all.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config


def confirm(prompt: str) -> bool:
    """请求用户确认。"""
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in ("y", "yes")


if __name__ == "__main__":
    print("=" * 80)
    print("SwarmReservoir v3.0 — 全管线运行")
    print("=" * 80)

    # 阶段1: 信号生成（无需硬件）
    print("\n[阶段 1/4] 输入信号生成...")
    from src.stage1_generate.pipeline import run_stage1
    run_stage1(get_stage_config(1))

    # 阶段2: 硬件实验
    print("\n[阶段 2/4] 硬件实验采集...")
    if confirm("是否已连接相机和电源？"):
        from src.stage2_experiment.pipeline import run_stage2
        run_stage2(get_stage_config(2), headless="--headless" in sys.argv)
    else:
        print("跳过阶段2（硬件未就绪）。")

    # 阶段3: 特征提取
    print("\n[阶段 3/4] 图像特征提取...")
    if confirm("是否先运行参数验证？"):
        print("请使用: python scripts/verify_stage3.py --image <路径>")
        if not confirm("继续批量处理？"):
            print("已停止。请调整参数后重新运行。")
            sys.exit(0)

    from src.stage3_extract.pipeline import run_stage3
    run_stage3(get_stage_config(3))

    # 阶段4: 基准测试
    print("\n[阶段 4/4] 储层基准测试...")
    from src.stage4_benchmark.pipeline import run_stage4
    run_stage4(get_stage_config(4))

    print("\n" + "=" * 80)
    print("全管线完成！")
    print("=" * 80)
