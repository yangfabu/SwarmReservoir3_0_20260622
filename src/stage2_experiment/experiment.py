"""
实验核心逻辑

管理电流序列 → 图像采集循环的完整实验流程。
包含严格的时序控制、多线程图像保存和安全清理机制。

迁移自: Current_Input/SRCExperiment.py -> SRCExperiment 类
"""

import threading
import time
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.common.logging_utils import setup_logger
from src.stage2_experiment.hardware.base_power_supply import BasePowerSupply


class Experiment:
    """
    Swarm Reservoir Computing 实验管理器。

    按严格时序遍历电流序列，每个周期包含:
      1. 发送电流命令
      2. 保持电流 + 启动高速图像保存
      3. 断电 (CURR 0.0)
      4. 停止图像保存
      5. 进入下一周期
    """

    def __init__(
        self,
        camera_driver,
        power_supply: BasePowerSupply,
        output_dir: str = "./experiment_output",
        current_list_file: str = "current_list.txt",
        timing_config: Optional[dict] = None,
        frame_interval_ms: float = 16.67,
    ):
        """
        Args:
            camera_driver: CameraDriver 实例（相机操作封装）。
            power_supply: BasePowerSupply 实例（电源控制）。
            output_dir: 图像输出目录。
            current_list_file: 电流序列文件路径。
            timing_config: 时序配置，默认使用内置值。
            frame_interval_ms: 图像保存帧间隔（毫秒），默认 16.67ms (60 FPS)。
        """
        self.camera = camera_driver
        self.power_supply = power_supply
        self.output_dir = Path(output_dir)
        self.current_list_file = current_list_file
        self.timing = timing_config or {
            "phase_a": 0.0, "phase_b": 0.5,
            "phase_c": 0.7, "phase_d": 0.9, "phase_e": 1.0,
        }

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志
        log_file = self.output_dir / "experiment.log"
        self.logger = setup_logger("Experiment", log_file=log_file)

        # 实验状态
        self.is_running = False
        self.b_save_image = False
        self._stop_save_event = threading.Event()
        self._save_thread: Optional[threading.Thread] = None

        # 实验参数
        self.current_list: List[float] = []
        self.frame_count = 0
        self.current_prefix = ""
        self.frame_interval = frame_interval_ms / 1000.0

        # 线程安全
        self._exp_lock = threading.Lock()

        self.logger.info("=" * 80)
        self.logger.info("Experiment 初始化完成")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info("=" * 80)

    # ================================================================
    # 电流序列加载
    # ================================================================

    def load_current_list(self) -> bool:
        """
        从 CSV 文件加载电流值序列。

        支持的格式:
          - 单列: 每行一个电流值
          - 带表头: 从 'applied_current_A' 或 'target_u' 列读取

        Returns:
            加载成功返回 True。
        """
        try:
            path = Path(self.current_list_file)
            if not path.exists():
                self.logger.error(f"电流列表文件不存在: {path}")
                return False

            import pandas as pd
            df = pd.read_csv(path)

            # 尝试从列名读取
            if "applied_current_A" in df.columns:
                values = df["applied_current_A"].tolist()
            elif "target_u" in df.columns:
                values = df["target_u"].tolist()
            else:
                # 尝试第一列
                values = df.iloc[:, 0].tolist()

            self.current_list = [float(v) for v in values if not pd.isna(v)]

            if not self.current_list:
                self.logger.error("电流列表为空")
                return False

            self.logger.info(f"成功加载 {len(self.current_list)} 个电流值")
            return True

        except Exception as e:
            self.logger.error(f"加载电流列表失败: {e}")
            return False

    # ================================================================
    # 命令发送
    # ================================================================

    def _send_command(self, command: str) -> bool:
        """安全的命令发送，带日志。"""
        try:
            result = self.power_supply.send_command(command)
            if result:
                self.logger.info(f"  >> {command}")
            else:
                self.logger.warning(f"  !! 命令失败: {command}")
            return result
        except Exception as e:
            self.logger.error(f"  !! 发送异常: {command} - {e}")
            return False

    # ================================================================
    # 图像保存线程
    # ================================================================

    def _save_image_loop(self) -> None:
        """图像保存循环（独立线程，绝对时间基准防累积误差）。"""
        self.logger.info(f"[保存线程] 启动, 前缀: {self.current_prefix}")
        self.frame_count = 0
        self._stop_save_event.clear()

        try:
            next_frame_time = time.time()
            while not self._stop_save_event.is_set() and self.b_save_image:
                current_time = time.time()

                # 严重超时则重置基准
                if current_time - next_frame_time > 0.5:
                    next_frame_time = current_time

                timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
                filename = self.output_dir / (
                    f"{self.current_prefix}_{self.frame_count:06d}_{timestamp_str}.jpg"
                )

                try:
                    ret = self.camera.save_jpg(str(filename))
                    if ret == 0:
                        self.frame_count += 1
                        self.logger.debug(
                            f"  [帧 {self.frame_count}] {filename.name}"
                        )
                    elif ret is None:
                        self.logger.debug("  [跳帧] 缓存为空")
                    else:
                        self.logger.warning(f"  [保存失败] 返回码: {ret}")
                except Exception as e:
                    self.logger.error(f"  [保存异常] {e}")

                next_frame_time += self.frame_interval
                current_after = time.time()
                sleep_time = next_frame_time - current_after

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = current_after
                    self.logger.warning(
                        f"  [帧率警告] 超时 {-sleep_time * 1000:.1f}ms"
                    )

        except Exception as e:
            self.logger.error(f"[保存线程] 异常退出: {e}")
        finally:
            self.b_save_image = False
            self.logger.info(f"[保存线程] 已停止 (共 {self.frame_count} 帧)")

    def _start_image_saving(self, prefix: str) -> None:
        """启动图像保存线程。"""
        if self._save_thread and self._save_thread.is_alive():
            self.logger.warning("保存线程已在运行")
            return
        self.current_prefix = prefix
        self.b_save_image = True
        self._save_thread = threading.Thread(target=self._save_image_loop, daemon=False)
        self._save_thread.start()

    def _stop_image_saving(self, wait_timeout: float = 2.0) -> bool:
        """停止图像保存线程。"""
        if not self._save_thread or not self._save_thread.is_alive():
            return True
        self.b_save_image = False
        self._stop_save_event.set()
        try:
            self._save_thread.join(timeout=wait_timeout)
            return not self._save_thread.is_alive()
        except Exception as e:
            self.logger.error(f"停止保存线程异常: {e}")
            return False

    # ================================================================
    # 单周期执行
    # ================================================================

    def _run_single_cycle(self, index: int, current_value: float) -> bool:
        """
        运行单个实验周期。

        时序:
          t=0.0: 发送 CURR {i}
          t=0.5: 启动图像保存
          t=0.7: 发送 CURR 0.0
          t=0.9: 停止图像保存
          t=1.0: 进入下一周期
        """
        try:
            cycle_start = time.time()
            prefix = f"{index + 1:03d}"

            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"[周期 {index + 1}] 开始 | 电流: {current_value} A")
            self.logger.info("=" * 80)

            # 阶段 A: 发送电流值
            self.logger.info(f"[A] t={time.time() - cycle_start:.3f}s | 发送电流")
            self._send_command(f"CURR {current_value}")

            # 阶段 B: 保持 + 启动保存
            time.sleep(self.timing.get("phase_b", 0.5))
            self.logger.info(f"[B] t={time.time() - cycle_start:.3f}s | 启动保存")
            self._start_image_saving(prefix)

            time.sleep(self.timing.get("phase_c", 0.7) - time.time() + cycle_start)
            # 阶段 C: 断电
            self.logger.info(f"[C] t={time.time() - cycle_start:.3f}s | 关闭电源")
            self._send_command("CURR 0.0")

            # 阶段 D: 继续保存
            time.sleep(self.timing.get("phase_d", 0.9) - time.time() + cycle_start)
            # 阶段 E: 停止保存
            self.logger.info(f"[E] t={time.time() - cycle_start:.3f}s | 停止保存")
            self._stop_image_saving()

            # 阶段 F: 等待进入下一周期
            remaining = self.timing.get("phase_e", 1.0) - (time.time() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)

            elapsed = time.time() - cycle_start
            self.logger.info(
                f"[周期 {index + 1}] 完成 | 耗时 {elapsed:.3f}s | 保存 {self.frame_count} 帧"
            )
            return True

        except Exception as e:
            self.logger.error(f"[周期 {index + 1}] 异常: {e}")
            return False

    # ================================================================
    # 实验主循环
    # ================================================================

    def _experiment_thread(self) -> None:
        """实验主逻辑线程。"""
        results_data = []
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("实验开始".center(80))
            self.logger.info("=" * 80)

            # 初始化电源
            self.logger.info("[初始化] 设置电源...")
            self._send_command("SYST:REM")
            self._send_command("INST CH1")
            self._send_command("VOLT 15.0")
            self._send_command("CURR 0.0")
            self._send_command("OUTP 1")

            # 遍历电流序列
            for idx, current_val in enumerate(self.current_list):
                if not self.is_running:
                    self.logger.warning("实验被中止")
                    break

                self._run_single_cycle(idx, current_val)
                results_data.append({
                    "cycle_index": idx + 1,
                    "current_value": current_val,
                    "frames_saved": self.frame_count,
                })

            self._save_summary_csv(results_data)
            self.logger.info("\n实验完成\n")

        except Exception as e:
            self.logger.error(f"实验线程异常: {e}")
        finally:
            self._safety_cleanup()

    def _save_summary_csv(self, data: List[dict]) -> None:
        """保存实验汇总 CSV。"""
        csv_file = self.output_dir / "experiment_summary.csv"
        try:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["cycle_index", "current_value", "frames_saved"]
                )
                writer.writeheader()
                writer.writerows(data)
            self.logger.info(f"汇总已保存: {csv_file}")
        except Exception as e:
            self.logger.error(f"保存汇总 CSV 失败: {e}")

    def _safety_cleanup(self) -> None:
        """安全清理（必须执行）。"""
        self.logger.info("[清理] 执行安全清理...")
        try:
            self._stop_image_saving(wait_timeout=1.0)
            self._send_command("OUTP 0")
            self._send_command("SYST:LOC")
            self.logger.info("[清理] 完成")
        except Exception as e:
            self.logger.error(f"[清理] 出错: {e}")
        self.is_running = False

    # ================================================================
    # 公共接口
    # ================================================================

    def start(self, blocking: bool = True) -> bool:
        """启动实验。blocking=True 则阻塞直到完成。"""
        if self.is_running:
            self.logger.warning("实验已在运行")
            return False
        if not self.load_current_list():
            return False

        self.is_running = True
        if blocking:
            self._experiment_thread()
        else:
            threading.Thread(target=self._experiment_thread, daemon=True).start()
            self.logger.info("实验已在后台启动")
        return True

    def stop(self) -> None:
        """请求停止实验。"""
        self.logger.info("停止实验请求...")
        self.is_running = False
        self._stop_image_saving()

    def get_status(self) -> dict:
        """获取实验状态。"""
        return {
            "is_running": self.is_running,
            "total_cycles": len(self.current_list),
            "frame_count": self.frame_count,
            "output_dir": str(self.output_dir),
        }
