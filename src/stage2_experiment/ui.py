"""
PyQt5 相机控制界面

提供相机的手动控制、实时预览和参数调整功能。
支持 GUI 模式和自动(headless)模式的相机初始化。

迁移自: Current_Input/camera_initial.py + PyUICBasicDemo.py
所有模块级全局变量已重构为 CameraControlWindow 实例属性。
"""

import sys
import os
import json
import threading
import time
from pathlib import Path
from typing import Optional

# PyQt5 可能不在所有环境中可用
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QMessageBox,
    )
    from PyQt5.QtCore import QFileSystemWatcher

    _HAS_PYQT5 = True
except ImportError:
    _HAS_PYQT5 = False


class CameraControlWindow:
    """
    相机控制 GUI 窗口。

    封装所有相机初始化、枚举、参数设置和图像采集的 UI 逻辑。
    不再使用模块级全局变量，所有状态都在实例属性中。

    Usage:
        window = CameraControlWindow()
        window.show()           # 启动 GUI
        # 或
        window.auto_open_and_grab()  # 无头模式
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Args:
            config_dir: 相机配置文件目录，默认使用项目 config/ 目录。
        """
        if not _HAS_PYQT5:
            raise ImportError("PyQt5 未安装，无法启动相机 UI。")

        self.config_dir = config_dir or Path(__file__).resolve().parent.parent.parent / "config"
        self.config_path = self.config_dir / "camera_config.json"

        # === 相机状态（替代旧代码的全局变量）===
        self.device_list = None
        self.cam = None
        self.n_sel_cam_index = 0
        self.camera_driver = None
        self.is_open = False
        self.is_grabbing = False
        self.is_trigger_mode = False

        # 同步事件
        self.camera_ready_event = threading.Event()

        # 保存计数
        self.save_counter = 1

        # 捕获标志
        self.capture_flag = 0
        self._capture_callback = None
        self._stop_capture_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        # === 延迟导入以避免循环依赖 ===
        from src.stage2_experiment.hardware.camera import CameraDriver
        from src.stage2_experiment.hardware.lib.MvCameraControl_class import (
            MvCamera, MV_CC_DEVICE_INFO_LIST,
            MV_GIGE_DEVICE, MV_USB_DEVICE,
            MV_GENTL_CAMERALINK_DEVICE, MV_GENTL_CXP_DEVICE,
            MV_GENTL_XOF_DEVICE,
        )
        self._MvCamera = MvCamera
        self._MV_CC_DEVICE_INFO_LIST = MV_CC_DEVICE_INFO_LIST

        # UI 组件（延迟初始化）
        self.app: Optional[QApplication] = None
        self.main_window: Optional[QMainWindow] = None
        self.ui = None

    # ================================================================
    # 配置加载/保存
    # ================================================================

    def load_config(self) -> dict:
        """加载相机配置 JSON 文件。"""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"framerate": 90.0, "exposuretime": 10000.0, "gain": 0.0}

    def save_config(self, framerate: float, exposuretime: float, gain: float = 0.0) -> None:
        """保存相机配置。"""
        config = {"framerate": framerate, "exposuretime": exposuretime, "gain": gain}
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

    # ================================================================
    # 相机 SDk 初始化 & 枚举
    # ================================================================

    def init_sdk(self) -> None:
        """初始化海康相机 SDK。"""
        self._MvCamera.MV_CC_Initialize()

    def finalize_sdk(self) -> None:
        """释放 SDK。"""
        self._MvCamera.MV_CC_Finalize()

    def enumerate_devices(self) -> list:
        """枚举连接的相机设备。返回设备描述列表。"""
        device_list = self._MV_CC_DEVICE_INFO_LIST()
        n_layer_type = (
            MV_GIGE_DEVICE | MV_USB_DEVICE
            | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE
            | MV_GENTL_XOF_DEVICE
        )
        ret = self._MvCamera.MV_CC_EnumDevices(n_layer_type, device_list)
        self.device_list = device_list
        return device_list

    # ================================================================
    # 自动打开和抓取（无头模式）
    # ================================================================

    def auto_open_and_grab(self) -> bool:
        """
        自动打开第一个设备并开始连续抓取（无 GUI 模式）。

        Returns:
            成功返回 True。
        """
        from src.stage2_experiment.hardware.camera import CameraDriver

        self.init_sdk()
        self.cam = self._MvCamera()

        device_list = self._MV_CC_DEVICE_INFO_LIST()
        n_layer_type = (
            MV_GIGE_DEVICE | MV_USB_DEVICE
            | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE
            | MV_GENTL_XOF_DEVICE
        )
        ret = self._MvCamera.MV_CC_EnumDevices(n_layer_type, device_list)
        if ret != 0 or device_list.nDeviceNum == 0:
            print("[UI] 未找到相机设备")
            return False

        self.device_list = device_list
        self.n_sel_cam_index = 0

        self.camera_driver = CameraDriver(self.cam, device_list, 0)
        ret = self.camera_driver.open_device()
        if ret != 0:
            print("[UI] 打开设备失败")
            return False
        self.is_open = True

        config = self.load_config()
        self.camera_driver.set_parameter(
            config["framerate"], config["exposuretime"], config.get("gain", 0.0)
        )

        ret = self.camera_driver.start_grabbing(0)
        if ret != 0:
            print("[UI] 开始采集失败")
            return False
        self.is_grabbing = True

        self.camera_ready_event.set()
        print("[UI] 相机已就绪 (无头模式)")
        return True

    def auto_close(self) -> None:
        """自动关闭相机设备。"""
        if self.is_grabbing and self.camera_driver:
            self.camera_driver.stop_grabbing()
            self.is_grabbing = False
        if self.is_open and self.camera_driver:
            self.camera_driver.close_device()
            self.is_open = False
        self.camera_ready_event.clear()
        print("[UI] 相机已关闭")

    # ================================================================
    # 捕获线程（用于实验自动采集）
    # ================================================================

    def set_capture_flag(self, value: int) -> None:
        """设置捕获标志位，触发自动保存。"""
        if value != self.capture_flag:
            self.capture_flag = value
            if self._capture_callback:
                self._capture_callback(value)

    def set_capture_callback(self, callback) -> None:
        """设置捕获标志变化回调。"""
        self._capture_callback = callback

    def start_capture_thread(self, capture_interval: float) -> None:
        """启动自动捕获线程。"""
        if self._capture_thread and self._capture_thread.is_alive():
            return
        self._stop_capture_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_worker, args=(capture_interval,), daemon=True
        )
        self._capture_thread.start()

    def stop_capture_thread(self) -> None:
        """停止自动捕获线程。"""
        self._stop_capture_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

    def _capture_worker(self, capture_interval: float) -> None:
        """自动捕获工作线程（绝对时间基准）。"""
        self.camera_ready_event.wait()
        next_save_time = time.time() + capture_interval
        while not self._stop_capture_event.is_set():
            if self.capture_flag == 1:
                current_time = time.time()
                if current_time >= next_save_time:
                    filename = str(self.save_counter) + ".bmp"
                    if self.camera_driver:
                        self.camera_driver.save_bmp(filename=filename)
                    self.save_counter += 1
                    next_save_time += capture_interval
                    if time.time() > next_save_time:
                        next_save_time = time.time() + capture_interval
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.05)
