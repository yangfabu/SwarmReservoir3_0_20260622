"""
阶段4 主管线：储备计算基准测试

流程:
  1. 预处理阶段3产物 (flatten → normalize → PCA)
  2. 加载目标值
  3. 对每个启用的 benchmark: run() → plot()
  4. 汇总结果，生成报告

迁移自: Benchmark/MC.py + MC_PAC.py 的独立运行逻辑
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config, resolve_path
from src.common.logging_utils import get_logger
from src.stage4_benchmark.preprocessing import preprocess_pipeline
from src.stage4_benchmark.benchmarks.base import get_benchmark_class, get_available_benchmarks

# 触发 benchmark 注册
from src.stage4_benchmark.benchmarks.memory_capacity import MemoryCapacityBenchmark
from src.stage4_benchmark.evaluation import aggregate_results, generate_report


def run_stage4(
    config: Optional[dict] = None,
    features_csv: Optional[Path] = None,
    skip_preprocess: bool = False,
) -> None:
    """
    执行阶段4：储备计算基准测试。

    Args:
        config: 阶段4配置（含 global）。
        features_csv: 阶段3产出的 features.csv。为 None 则使用默认路径。
        skip_preprocess: 跳过预处理（假设输入已是预处理后的数据）。

    产物:
        - output/figures/mc_curve.png
        - output/reports/benchmark_summary.csv
    """
    if config is None:
        config = get_stage_config(4)

    logger = get_logger("stage4")
    bench_cfg = config.get("benchmarks", {})
    output_cfg = config.get("output", {})

    figures_dir = resolve_path(output_cfg.get("figures_dir", "output/figures"))
    reports_dir = resolve_path(output_cfg.get("reports_dir", "output/reports"))

    # 1. 预处理
    if skip_preprocess:
        logger.info("跳过预处理，加载已有数据...")
        raise NotImplementedError(
            "skip_preprocess 模式需要预先保存的 X.npy 和 u_t.npy"
        )

    if features_csv is None:
        features_csv = resolve_path("data/stage3_output/features.csv")

    if not features_csv.exists():
        logger.error(
            f"阶段3产物不存在: {features_csv}\n"
            f"请先运行: python scripts/run_stage3.py"
        )
        return

    logger.info(f"加载特征: {features_csv}")
    X, u_t, pca_obj = preprocess_pipeline(features_csv, config)

    # 2. 选择启用哪些 benchmark
    enabled_names = bench_cfg.get("enabled", ["memory_capacity"])
    logger.info(f"启用 benchmark: {enabled_names}")
    logger.info(f"特征矩阵: {X.shape[0]} 组 × {X.shape[1]} 维")

    # 3. 执行每个 benchmark
    all_results = []
    for name in enabled_names:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"执行: {name}")
        logger.info(f"{'=' * 60}")

        try:
            bench_cls = get_benchmark_class(name)
        except ValueError as e:
            logger.warning(f"跳过: {e}")
            continue

        bench = bench_cls()
        bench_config = config.get("benchmarks", {})

        try:
            results = bench.run(X, u_t, bench_config)
            logger.info(f"  结果: {bench.name()}")
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

            bench.plot(results, figures_dir)
            all_results.append({"name": name, "results": results})

        except Exception as e:
            logger.error(f"  {name} 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # 4. 汇总
    if all_results:
        summary_df = aggregate_results(all_results)
        generate_report(summary_df, reports_dir)
        logger.info(f"\n阶段4 完成: {len(all_results)} 个 benchmark 已评估")
    else:
        logger.warning("没有成功执行的 benchmark")
