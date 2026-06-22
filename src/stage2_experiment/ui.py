"""
PyQt5 相机控制界面（海康 Hikvision 相机）

提供完整的相机手动控制、实时预览和参数调整功能。
支持 GUI 模式和 headless（自动）模式。

GUI 模式下用户操作流程:
  枚举设备 → 打开 → 设置参数 → 开始取流 → 启动实验

迁移自: Current_Input/camera_initial.py + PyUICBasicDemo.py
"""

import sys
import json
import threading
import time
from pathlib import Path
from typing import Optional

# PyQt5
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QComboBox, QDoubleSpinBox, QGroupBox,
        QTextEdit, QMessageBox, QGridLayout, QFrame,
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
    _HAS_PYQT5 = True
except ImportError:
    _HAS_PYQT5 = False


# ================================================================
# 信号发射器（跨线程通信）
# ================================================================

class _ExperimentSignals(QObject):
    """用于 UI 线程与实验线程通信的信号。"""
    experiment_requested = pyqtSignal()   # 用户点击"启动实验"
    log_message = pyqtSignal(str)          # 日志消息 → UI


# ================================================================
# 相机控制窗口
# ================================================================

class CameraControlWindow:
    """
    海康相机控制 GUI 窗口。

    Usage:
        window = CameraControlWindow(config_dir)
        window.show()                    # 启动 GUI（阻塞，直到窗口关闭）
        # 或
        window.auto_open_and_grab()      # headless 模式
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Args:
            config_dir: 相机配置文件目录。
        """
        if not _HAS_PYQT5:
            raise ImportError("PyQt5 未安装，无法启动相机 UI。请: pip install pyqt5")

        self.config_dir = config_dir or Path(__file__).resolve().parent.parent.parent / "config"
        self.config_path = self.config_dir / "camera_config.json"

        # === 相机状态 ===
        self.device_list = None
        self.cam = None
        self.n_sel_cam_index = 0
        self.camera_driver = None
        self.is_open = False
        self.is_grabbing = False
        self.is_trigger_mode = False

        # === 同步 ===
        self.camera_ready_event = threading.Event()

        # === 信号 ===
        self.signals = _ExperimentSignals()

        # === 实验回调 ===
        self._experiment_callback = None

        # === 延迟导入 ===
        from src.stage2_experiment.hardware.camera import CameraDriver
        from src.stage2_experiment.hardware.lib.MvCameraControl_class import (
            MvCamera, MV_CC_DEVICE_INFO_LIST,
            MV_GIGE_DEVICE, MV_USB_DEVICE,
            MV_GENTL_CAMERALINK_DEVICE, MV_GENTL_CXP_DEVICE,
            MV_GENTL_XOF_DEVICE,
        )
        self._MvCamera = MvCamera
        self._MV_CC_DEVICE_INFO_LIST = MV_CC_DEVICE_INFO_LIST
        self._MV_GIGE_DEVICE = MV_GIGE_DEVICE
        self._MV_USB_DEVICE = MV_USB_DEVICE
        self._MV_GENTL_CAMERALINK_DEVICE = MV_GENTL_CAMERALINK_DEVICE
        self._MV_GENTL_CXP_DEVICE = MV_GENTL_CXP_DEVICE
        self._MV_GENTL_XOF_DEVICE = MV_GENTL_XOF_DEVICE

        # === Qt 组件 ===
        self.app: Optional[QApplication] = None
        self.main_window: Optional[QMainWindow] = None

        # === UI 控件引用 ===
        self._cmb_devices: Optional[QComboBox] = None
        self._btn_enumerate: Optional[QPushButton] = None
        self._btn_open: Optional[QPushButton] = None
        self._btn_close: Optional[QPushButton] = None
        self._btn_start_grab: Optional[QPushButton] = None
        self._btn_stop_grab: Optional[QPushButton] = None
        self._btn_experiment: Optional[QPushButton] = None
        self._spn_framerate: Optional[QDoubleSpinBox] = None
        self._spn_exposure: Optional[QDoubleSpinBox] = None
        self._spn_gain: Optional[QDoubleSpinBox] = None
        self._txt_log: Optional[QTextEdit] = None
        self._lbl_status: Optional[QLabel] = None

    # ================================================================
    # 配置
    # ================================================================

    def load_config(self) -> dict:
        """加载相机配置文件。"""
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
        self._log(f"配置已保存: framerate={framerate}, exposure={exposuretime}, gain={gain}")

    # ================================================================
    # SDK
    # ================================================================

    def init_sdk(self) -> None:
        """初始化海康相机 SDK。"""
        self._MvCamera.MV_CC_Initialize()
        self._log("SDK 已初始化")

    def finalize_sdk(self) -> None:
        """释放 SDK。"""
        self._MvCamera.MV_CC_Finalize()
        self._log("SDK 已释放")

    def enumerate_devices(self) -> list:
        """枚举相机设备。"""
        device_list = self._MV_CC_DEVICE_INFO_LIST()
        n_layer_type = (
            self._MV_GIGE_DEVICE | self._MV_USB_DEVICE
            | self._MV_GENTL_CAMERALINK_DEVICE | self._MV_GENTL_CXP_DEVICE
            | self._MV_GENTL_XOF_DEVICE
        )
        ret = self._MvCamera.MV_CC_EnumDevices(n_layer_type, device_list)
        self.device_list = device_list
        return device_list

    # ================================================================
    # 相机操作
    # ================================================================

    def open_device(self, index: int = 0) -> bool:
        """打开指定索引的相机设备。"""
        if self.cam is None:
            self.cam = self._MvCamera()
        self.camera_driver = CameraDriver(self.cam, self.device_list, index)
        ret = self.camera_driver.open_device()
        if ret == 0:
            self.is_open = True
            self._log(f"设备 {index} 已打开")
            return True
        self._log(f"打开设备 {index} 失败 (返回码: {ret})")
        return False

    def close_device(self) -> None:
        """关闭相机设备。"""
        if self.is_grabbing:
            self.stop_grabbing()
        if self.camera_driver:
            self.camera_driver.close_device()
        self.is_open = False
        self.camera_ready_event.clear()
        self._log("设备已关闭")

    def start_grabbing(self) -> bool:
        """开始连续取流。"""
        if not self.camera_driver or not self.is_open:
            self._log("错误: 请先打开设备")
            return False
        ret = self.camera_driver.start_grabbing(0)
        if ret == 0:
            self.is_grabbing = True
            self.camera_ready_event.set()
            self._log("取流已开始")
            return True
        self._log(f"开始取流失败 (返回码: {ret})")
        return False

    def stop_grabbing(self) -> None:
        """停止取流。"""
        if self.camera_driver:
            self.camera_driver.stop_grabbing()
        self.is_grabbing = False
        self.camera_ready_event.clear()
        self._log("取流已停止")

    def set_parameters(self, framerate: float, exposure: float, gain: float) -> None:
        """设置相机参数。"""
        if self.camera_driver:
            self.camera_driver.set_parameter(framerate, exposure, gain)
            self._log(f"参数已设置: FPS={framerate}, Exp={exposure}us, Gain={gain}")

    # ================================================================
    # Headless 模式
    # ================================================================

    def auto_open_and_grab(self) -> bool:
        """
        无 GUI 模式：自动打开第一个设备并开始取流。

        Returns:
            成功返回 True。
        """
        self.init_sdk()
        device_list = self.enumerate_devices()
        if device_list.nDeviceNum == 0:
            self._log("未找到相机设备")
            return False
        self._log(f"找到 {device_list.nDeviceNum} 个设备")

        if not self.open_device(0):
            return False

        config = self.load_config()
        self.set_parameters(
            config.get("framerate", 90.0),
            config.get("exposuretime", 10000.0),
            config.get("gain", 0.0),
        )

        if not self.start_grabbing():
            return False

        self._log("相机已就绪 (headless 模式)")
        return True

    def auto_close(self) -> None:
        """关闭相机（headless 模式清理）。"""
        self.close_device()

    # ================================================================
    # 实验回调
    # ================================================================

    def set_experiment_callback(self, callback) -> None:
        """
        设置实验启动回调。当用户点击"启动实验"时调用。

        Args:
            callback: 可调用对象，签名为 callback(camera_driver)。
        """
        self._experiment_callback = callback

    # ================================================================
    # GUI 构建
    # ================================================================

    def show(self) -> None:
        """
        构建并显示 PyQt5 GUI 窗口。

        该调用会阻塞，直到用户关闭窗口。
        """
        if not _HAS_PYQT5:
            raise ImportError("PyQt5 未安装")

        self.app = QApplication.instance() or QApplication(sys.argv)

        # --- 主窗口 ---
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle("海康相机控制 — SwarmReservoir Stage 2")
        self.main_window.setMinimumSize(640, 520)

        central = QWidget()
        self.main_window.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        # --- 状态栏 ---
        self._lbl_status = QLabel("就绪 — 请先枚举设备")
        self._lbl_status.setStyleSheet(
            "QLabel { background: #333; color: #0f0; padding: 6px; "
            "font-family: Consolas; font-size: 13px; }"
        )
        layout.addWidget(self._lbl_status)

        # --- 设备选择 ---
        grp_dev = QGroupBox("设备选择")
        grp_dev_layout = QGridLayout(grp_dev)
        self._cmb_devices = QComboBox()
        self._cmb_devices.setMinimumWidth(350)
        grp_dev_layout.addWidget(QLabel("设备列表:"), 0, 0)
        grp_dev_layout.addWidget(self._cmb_devices, 0, 1)
        self._btn_enumerate = QPushButton("枚举设备")
        self._btn_enumerate.clicked.connect(self._on_enumerate)
        grp_dev_layout.addWidget(self._btn_enumerate, 0, 2)
        layout.addWidget(grp_dev)

        # --- 相机控制按钮 ---
        grp_ctrl = QGroupBox("相机控制")
        grp_ctrl_layout = QHBoxLayout(grp_ctrl)
        self._btn_open = QPushButton("打开")
        self._btn_open.clicked.connect(self._on_open)
        self._btn_open.setEnabled(False)
        grp_ctrl_layout.addWidget(self._btn_open)
        self._btn_close = QPushButton("关闭")
        self._btn_close.clicked.connect(self._on_close)
        self._btn_close.setEnabled(False)
        grp_ctrl_layout.addWidget(self._btn_close)
        self._btn_start_grab = QPushButton("开始取流")
        self._btn_start_grab.clicked.connect(self._on_start_grab)
        self._btn_start_grab.setEnabled(False)
        grp_ctrl_layout.addWidget(self._btn_start_grab)
        self._btn_stop_grab = QPushButton("停止取流")
        self._btn_stop_grab.clicked.connect(self._on_stop_grab)
        self._btn_stop_grab.setEnabled(False)
        grp_ctrl_layout.addWidget(self._btn_stop_grab)
        layout.addWidget(grp_ctrl)

        # --- 参数设置 ---
        grp_param = QGroupBox("相机参数")
        grp_param_layout = QGridLayout(grp_param)
        config = self.load_config()

        grp_param_layout.addWidget(QLabel("帧率 (FPS):"), 0, 0)
        self._spn_framerate = QDoubleSpinBox()
        self._spn_framerate.setRange(1.0, 200.0)
        self._spn_framerate.setValue(config.get("framerate", 90.0))
        grp_param_layout.addWidget(self._spn_framerate, 0, 1)

        grp_param_layout.addWidget(QLabel("曝光时间 (us):"), 0, 2)
        self._spn_exposure = QDoubleSpinBox()
        self._spn_exposure.setRange(10.0, 1000000.0)
        self._spn_exposure.setDecimals(0)
        self._spn_exposure.setValue(config.get("exposuretime", 10000.0))
        grp_param_layout.addWidget(self._spn_exposure, 0, 3)

        grp_param_layout.addWidget(QLabel("增益:"), 1, 0)
        self._spn_gain = QDoubleSpinBox()
        self._spn_gain.setRange(0.0, 100.0)
        self._spn_gain.setValue(config.get("gain", 0.0))
        grp_param_layout.addWidget(self._spn_gain, 1, 1)

        btn_apply = QPushButton("应用参数")
        btn_apply.clicked.connect(self._on_apply_params)
        grp_param_layout.addWidget(btn_apply, 1, 3)
        btn_save = QPushButton("保存配置")
        btn_save.clicked.connect(self._on_save_config)
        grp_param_layout.addWidget(btn_save, 1, 4)
        layout.addWidget(grp_param)

        # --- 实验控制 ---
        grp_exp = QGroupBox("实验控制")
        grp_exp_layout = QHBoxLayout(grp_exp)
        self._btn_experiment = QPushButton("▶ 启动实验")
        self._btn_experiment.setMinimumHeight(40)
        self._btn_experiment.setStyleSheet(
            "QPushButton { background: #28a745; color: white; font-size: 14px; "
            "font-weight: bold; border-radius: 4px; }"
            "QPushButton:disabled { background: #666; color: #999; }"
            "QPushButton:hover { background: #218838; }"
        )
        self._btn_experiment.clicked.connect(self._on_start_experiment)
        self._btn_experiment.setEnabled(False)
        grp_exp_layout.addWidget(self._btn_experiment)
        layout.addWidget(grp_exp)

        # --- 日志 ---
        grp_log = QGroupBox("日志")
        grp_log_layout = QVBoxLayout(grp_log)
        self._txt_log = QTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setMaximumHeight(150)
        self._txt_log.setStyleSheet(
            "QTextEdit { background: #1e1e1e; color: #ddd; font-family: Consolas; "
            "font-size: 11px; }"
        )
        grp_log_layout.addWidget(self._txt_log)
        layout.addWidget(grp_log)

        # --- 信号连接 ---
        self.signals.log_message.connect(self._append_log)

        # --- 初始化 SDK ---
        try:
            self.init_sdk()
        except Exception as e:
            self._log(f"SDK 初始化失败: {e}")

        # --- 显示 ---
        self.main_window.show()
        self._log("GUI 已启动 — 请枚举设备并打开相机")

        # 窗口关闭时清理
        self.main_window.destroyed.connect(self._on_window_closed)

        # 阻塞直到窗口关闭
        self.app.exec_()

    # ================================================================
    # UI 事件处理
    # ================================================================

    def _on_enumerate(self) -> None:
        """点击「枚举设备」。"""
        try:
            device_list = self.enumerate_devices()
            self._cmb_devices.clear()
            n = device_list.nDeviceNum
            if n == 0:
                self._cmb_devices.addItem("(无设备)")
                self._log("未找到相机设备")
                self._update_buttons(has_device=False)
                return
            # Hikvision SDK 设备名在设备信息结构中
            for i in range(n):
                self._cmb_devices.addItem(f"相机 {i}")
            self._cmb_devices.setCurrentIndex(0)
            self._log(f"找到 {n} 个设备")
            self._update_buttons(has_device=True)
        except Exception as e:
            self._log(f"枚举失败: {e}")

    def _on_open(self) -> None:
        """点击「打开」。"""
        idx = self._cmb_devices.currentIndex()
        self.open_device(idx)
        self._update_buttons()

    def _on_close(self) -> None:
        """点击「关闭」。"""
        self.close_device()
        self._update_buttons()

    def _on_start_grab(self) -> None:
        """点击「开始取流」。"""
        self.start_grabbing()
        self._update_buttons()

    def _on_stop_grab(self) -> None:
        """点击「停止取流」。"""
        self.stop_grabbing()
        self._update_buttons()

    def _on_apply_params(self) -> None:
        """点击「应用参数」。"""
        framerate = self._spn_framerate.value()
        exposure = self._spn_exposure.value()
        gain = self._spn_gain.value()
        self.set_parameters(framerate, exposure, gain)

    def _on_save_config(self) -> None:
        """点击「保存配置」。"""
        self.save_config(
            self._spn_framerate.value(),
            self._spn_exposure.value(),
            self._spn_gain.value(),
        )

    def _on_start_experiment(self) -> None:
        """点击「启动实验」。"""
        if not self.is_grabbing:
            self._log("警告: 请先开始取流再启动实验")
            return
        self._btn_experiment.setEnabled(False)
        self._btn_experiment.setText("实验运行中...")
        self._log("=" * 60)
        self._log("实验启动!")
        self._log("=" * 60)

        if self._experiment_callback:
            # 在后台线程运行实验，避免阻塞 UI
            threading.Thread(
                target=self._experiment_callback,
                args=(self.camera_driver,),
                daemon=True,
            ).start()
        else:
            self._log("警告: 未设置实验回调函数")

    def _on_window_closed(self) -> None:
        """窗口关闭时的清理。"""
        self._log("窗口关闭，清理资源...")
        self.close_device()
        try:
            self.finalize_sdk()
        except Exception:
            pass

    # ================================================================
    # 辅助
    # ================================================================

    def _update_buttons(self, has_device: Optional[bool] = None) -> None:
        """根据相机状态更新按钮启用/禁用。"""
        if has_device is not None:
            has_dev = has_device
        else:
            has_dev = self.device_list is not None and self.device_list.nDeviceNum > 0

        self._btn_open.setEnabled(has_dev and not self.is_open)
        self._btn_close.setEnabled(self.is_open)
        self._btn_start_grab.setEnabled(self.is_open and not self.is_grabbing)
        self._btn_stop_grab.setEnabled(self.is_grabbing)
        self._btn_experiment.setEnabled(self.is_grabbing)

        if self.is_grabbing:
            self._lbl_status.setText("● 取流中 — 可以启动实验")
            self._lbl_status.setStyleSheet(
                "QLabel { background: #1a3a1a; color: #0f0; padding: 6px; "
                "font-family: Consolas; font-size: 13px; }"
            )
        elif self.is_open:
            self._lbl_status.setText("设备已打开 — 请开始取流")
            self._lbl_status.setStyleSheet(
                "QLabel { background: #333; color: #ff0; padding: 6px; "
                "font-family: Consolas; font-size: 13px; }"
            )
        elif has_dev:
            self._lbl_status.setText("设备就绪 — 请打开设备")
            self._lbl_status.setStyleSheet(
                "QLabel { background: #333; color: #0f0; padding: 6px; "
                "font-family: Consolas; font-size: 13px; }"
            )
        else:
            self._lbl_status.setText("就绪 — 请先枚举设备")
            self._lbl_status.setStyleSheet(
                "QLabel { background: #333; color: #0f0; padding: 6px; "
                "font-family: Consolas; font-size: 13px; }"
            )

    def _log(self, message: str) -> None:
        """记录日志（线程安全）。"""
        if self._txt_log:
            # 从任意线程安全地追加日志
            self.signals.log_message.emit(message)
        else:
            print(f"[UI] {message}")

    def _append_log(self, message: str) -> None:
        """实际追加日志到 QTextEdit（必须在主线程调用）。"""
        if self._txt_log:
            timestamp = time.strftime("%H:%M:%S")
            self._txt_log.append(f"[{timestamp}] {message}")
