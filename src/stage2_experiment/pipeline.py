"""
阶段2 主管线：硬件实验与数据采集

流程:
  1. 加载阶段1产物: current_sequence.csv
  2. 初始化相机（GUI 或 headless 模式）和电源
  3. 创建 Experiment 实例并执行
  4. 安全清理硬件资源

迁移自: Current_Input/main.py + SRCExperiment.py
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional

# 确保项目根在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common.config import get_stage_config, resolve_path
from src.common.logging_utils import get_logger
from src.stage2_experiment.hardware.power_supply import ITECHPowerSupply
from src.stage2_experiment.experiment import Experiment


def run_stage2(
    config: Optional[dict] = None,
    headless: bool = False,
) -> None:
    """
    执行阶段2：硬件实验采集。

    Args:
        config: 阶段2配置（含 global）。为 None 则自动加载。
        headless: True 则使用无 GUI 模式（自动打开相机）。

    产物:
        - data/stage2_output/{experiment_name}/
          ├── {cycle}_{frame}_{timestamp}.jpg  (原始图像)
          ├── experiment_summary.csv
          └── experiment.log
    """
    if config is None:
        config = get_stage_config(2)

    logger = get_logger("stage2")
    global_cfg = config.get("global", {})
    exp_cfg = config.get("experiment", {})
    hw_cfg = global_cfg.get("hardware", {})
    input_cfg = config.get("input", {})

    # 1. 确定电流序列路径（优先使用配置文件中的路径）
    current_list_file = resolve_path(
        input_cfg.get("current_list_file", "data/stage1_output/current_sequence.csv")
    )
    if not current_list_file.exists():
        logger.error(
            f"电流序列文件不存在: {current_list_file}\n"
            f"请在 config/stage2_experiment.yaml 的 input.current_list_file 中指定正确的路径\n"
            f"如果尚未生成，请先运行: python scripts/run_stage1.py"
        )
        return

    # 2. 初始化电源
    ps_cfg = hw_cfg.get("power_supply", {})
    power_supply = ITECHPowerSupply(
        port=ps_cfg.get("port", "COM8"),
        baudrate=ps_cfg.get("baudrate", 9600),
    )
    if not power_supply.connect(power_supply.port, power_supply.baudrate):
        logger.error("电源初始化失败")
        return

    # 3. 初始化相机
    camera_driver = None
    try:
        if headless:
            from src.stage2_experiment.ui import CameraControlWindow
            ui = CameraControlWindow(Path(__file__).resolve().parent.parent.parent / "config")
            if not ui.auto_open_and_grab():
                logger.error("相机初始化失败（headless 模式）")
                power_supply.close()
                return
            camera_driver = ui.camera_driver
        else:
            # GUI 模式: 启动 PyQt5 窗口，用户在 UI 中操作相机并启动实验
            print("=" * 80)
            print("相机控制窗口已启动。")
            print("操作流程: 枚举设备 → 打开 → 设置参数 → 开始取流 → 启动实验")
            print("=" * 80)

            ui = CameraControlWindow(Path(__file__).resolve().parent.parent.parent / "config")

            # 定义实验启动回调：用户在 UI 中点击"启动实验"时触发
            def _start_experiment_from_ui(camera_drv):
                logger.info("用户通过 UI 启动了实验")
                exp_name = exp_cfg.get("name", "MCTest")
                output_dir = resolve_path(f"data/stage2_output/{exp_name}")

                experiment = Experiment(
                    camera_driver=camera_drv,
                    power_supply=power_supply,
                    output_dir=str(output_dir),
                    current_list_file=str(current_list_file),
                    timing_config=exp_cfg.get("timing"),
                    frame_interval_ms=exp_cfg.get("frame_interval_ms", 16.67),
                )

                try:
                    logger.info(f"实验开始: {exp_name}")
                    experiment.start(blocking=True)
                    logger.info("阶段2 完成")
                except KeyboardInterrupt:
                    logger.info("用户手动停止实验")
                    experiment.stop()
                except Exception as e:
                    logger.error(f"实验异常: {e}")

            ui.set_experiment_callback(_start_experiment_from_ui)

            # show() 会阻塞直到用户关闭窗口
            ui.show()

            # UI 关闭后清理
            power_supply.close()
            logger.info("阶段2 GUI 已关闭，资源已释放")
            return
    except ImportError as e:
        logger.error(f"相机模块导入失败: {e}")
        power_supply.close()
        return

    # 4. 创建实验管理器
    exp_name = exp_cfg.get("name", "MCTest")
    output_dir = resolve_path(f"data/stage2_output/{exp_name}")

    experiment = Experiment(
        camera_driver=camera_driver,
        power_supply=power_supply,
        output_dir=str(output_dir),
        current_list_file=str(current_list_file),
        timing_config=exp_cfg.get("timing"),
        frame_interval_ms=exp_cfg.get("frame_interval_ms", 16.67),
    )

    # 5. 运行实验
    try:
        logger.info(f"实验开始: {exp_name}")
        experiment.start(blocking=True)
        logger.info("阶段2 完成")
    except KeyboardInterrupt:
        logger.info("用户手动停止实验")
        experiment.stop()
    except Exception as e:
        logger.error(f"实验异常: {e}")
    finally:
        power_supply.close()
        if camera_driver:
            camera_driver.close_device()
        logger.info("阶段2 资源已释放")
