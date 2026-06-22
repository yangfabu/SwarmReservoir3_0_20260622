"""
记忆容量 (Memory Capacity) Benchmark

评估储层对不同延迟 d 的输入 u(t-d) 的记忆能力。

公式: MC_d = R^2(u(t-d), y_d(t)) = cov(u, y)^2 / (var(u) * var(y))
总记忆容量: MC_total = sum(MC_d) for d = 1..d_max

支持:
  - 全量拟合 (train_ratio=0)
  - 训练/测试集划分 (train_ratio > 0)
  - 多种回归方法: Ridge, LinearRegression, Lasso

合并自: Benchmark/MC.py + MC_PAC.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression, Lasso

from .base import BaseBenchmark
from ..benchmarks.base import _benchmark_registry


# 注册
class MemoryCapacityBenchmark(BaseBenchmark):
    """记忆容量基准测试。"""

    def name(self) -> str:
        return "memory_capacity"

    def run(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        config: dict,
    ) -> dict:
        """
        计算记忆容量曲线。

        Args:
            features: (N, D) 特征矩阵。
            targets: (N,) 目标值数组 u(t)。
            config: memory_capacity 配置节。

        Returns:
            {"total_mc": float, "r2_curve": list, "d_max": int}
        """
        cfg = config.get("memory_capacity", {})
        d_max = cfg.get("d_max", 40)
        train_ratio = cfg.get("train_ratio", 0.8)
        reg_cfg = cfg.get("regression", {})
        reg_method = reg_cfg.get("method", "ridge")
        alpha = reg_cfg.get("alpha", 0.1)

        n_steps = len(targets)

        r2_results = []
        for d in range(1, d_max + 1):
            if train_ratio > 0:
                score = self._evaluate_with_split(
                    features, targets, d, train_ratio, reg_method, alpha
                )
            else:
                score = self._evaluate_full(
                    features, targets, d, reg_method, alpha
                )
            r2_results.append(max(0.0, score))

        total_mc = sum(r2_results)
        return {
            "total_mc": total_mc,
            "r2_curve": r2_results,
            "d_max": d_max,
        }

    @staticmethod
    def _evaluate_full(
        X: np.ndarray,
        u_t: np.ndarray,
        d: int,
        method: str,
        alpha: float,
    ) -> float:
        """全量数据拟合并评估。"""
        n_steps = len(u_t)
        if n_steps <= d:
            return 0.0

        X_current = X[d:]
        u_past = u_t[:n_steps - d]

        model = MemoryCapacityBenchmark._create_model(method, alpha)
        model.fit(X_current, u_past)
        y_pred = model.predict(X_current)

        return MemoryCapacityBenchmark._compute_r2_cov(u_past, y_pred)

    @staticmethod
    def _evaluate_with_split(
        X: np.ndarray,
        u_t: np.ndarray,
        d: int,
        train_ratio: float,
        method: str,
        alpha: float,
    ) -> float:
        """训练/测试集划分评估。"""
        n_steps = len(u_t)
        train_len = int(n_steps * train_ratio)
        if train_len <= d or train_len >= n_steps:
            return 0.0

        # 训练
        X_train = X[d:train_len]
        u_train = u_t[:train_len - d]

        model = MemoryCapacityBenchmark._create_model(method, alpha)
        model.fit(X_train, u_train)

        # 测试
        X_test = X[train_len:]
        u_test = u_t[train_len - d:n_steps - d]

        if len(X_test) == 0:
            return 0.0

        y_pred = model.predict(X_test)
        score = MemoryCapacityBenchmark._compute_r2_cov(u_test, y_pred)
        return score

    @staticmethod
    def _create_model(method: str, alpha: float):
        """根据配置创建回归模型。"""
        if method == "ridge":
            return Ridge(alpha=alpha)
        elif method == "linear":
            return LinearRegression()
        elif method == "lasso":
            return Lasso(alpha=alpha, max_iter=5000)
        else:
            raise ValueError(f"未知回归方法: '{method}'。可选: ridge, linear, lasso")

    @staticmethod
    def _compute_r2_cov(u_true: np.ndarray, u_pred: np.ndarray) -> float:
        """
        基于协方差公式计算 R^2。

        R^2 = cov(u, y)^2 / (var(u) * var(y))
        """
        cov_matrix = np.cov(u_true, u_pred)
        covariance = cov_matrix[0, 1]
        var_u = np.var(u_true)
        var_y = np.var(u_pred)
        return float((covariance ** 2) / (var_u * var_y + 1e-9))

    def plot(self, results: dict, output_dir: Path) -> None:
        """绘制记忆容量曲线并保存。"""
        r2_curve = results["r2_curve"]
        d_max = results["d_max"]
        total_mc = results["total_mc"]
        delays = range(1, d_max + 1)

        # 保存 CSV
        mc_df = pd.DataFrame({
            "Delay_d": delays,
            "R2_Score": r2_curve,
        })
        mc_df.to_csv(output_dir / "mc_curve.csv", index=False)

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.fill_between(delays, r2_curve, color="gray", alpha=0.2)
        plt.plot(delays, r2_curve, "o-", color="black", markersize=4,
                 label="$R^2$ score")

        plt.title(
            f"Physical Reservoir Memory Capacity\nTotal MC = {total_mc:.3f}",
            fontsize=14,
        )
        plt.xlabel("Delay $d$ (Time Steps)", fontsize=12)
        plt.ylabel("Capacity ($R^2$)", fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "mc_curve.png", dpi=300)
        plt.close()
        print(f"MC 曲线已保存: {output_dir / 'mc_curve.png'}")


# 手动注册（避免循环导入）
_benchmark_registry["memory_capacity"] = MemoryCapacityBenchmark
