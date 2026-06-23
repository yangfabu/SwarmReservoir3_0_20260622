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
    experiment_requested = pyqtSignal()    # 用户点击"启动实验"
    experiment_finished = pyqtSignal()     # 实验线程结束（自然完成或异常退出）
    log_message = pyqtSignal(str)           # 日志消息 → UI


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
        self._active_experiment = None  # 正在运行的实验实例引用

        # === 延迟导入 ===
        from src.stage2_experiment.hardware.camera import CameraDriver
        from src.stage2_experiment.hardware.lib.MvCameraControl_class import (
            MvCamera, MV_CC_DEVICE_INFO_LIST,
            MV_GIGE_DEVICE, MV_USB_DEVICE,
            MV_GENTL_CAMERALINK_DEVICE, MV_GENTL_CXP_DEVICE,
            MV_GENTL_XOF_DEVICE,
        )
        self._CameraDriver = CameraDriver
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
        self._btn_stop_experiment: Optional[QPushButton] = None
        self._frame_display: Optional[QWidget] = None
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
        self.camera_driver = self._CameraDriver(self.cam, self.device_list, index)
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

    def start_grabbing(self, display_handle: int = 0) -> bool:
        """开始连续取流。display_handle 为预览窗口 HWND（0 表示不显示）。"""
        if not self.camera_driver or not self.is_open:
            self._log("错误: 请先打开设备")
            return False
        ret = self.camera_driver.start_grabbing(display_handle)
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
            callback: 可调用对象，签名为 callback(camera_driver, ui)。
        """
        self._experiment_callback = callback

    def set_experiment(self, experiment) -> None:
        """
        设置/清除当前活动实验的引用（由 pipeline 回调调用）。

        可从任意线程调用（PyQt5 信号是线程安全的）。

        Args:
            experiment: Experiment 实例或 None。
        """
        self._active_experiment = experiment
        if experiment is None:
            # 实验已完成，通知 UI 线程更新按钮状态
            self.signals.experiment_finished.emit()

    def _get_display_handle(self) -> int:
        """获取相机预览窗口的原生 HWND（Windows）。"""
        if self._frame_display is not None:
            try:
                return int(self._frame_display.winId())
            except Exception:
                return 0
        return 0

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

        # --- 强制使用 Fusion 样式（跨平台一致） ---
        self.app.setStyle("Fusion")

        # --- 全局样式表：确保所有控件清晰可见 ---
        # 关键：Fusion 的 QPalette.Button 和 QPalette.Window 颜色相同，
        # 导致按钮无边框时完全不可见。因此必须用 stylesheet 显式定义。
        self.app.setStyleSheet("""
            /* 主窗口背景 */
            QMainWindow {
                background: #f0f0f5;
            }

            /* 分组框：可见边框 + 深色标题 */
            QGroupBox {
                border: 1px solid #b0b0b0;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 14px;
                font-weight: bold;
                font-size: 13px;
                color: #222;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #222;
            }

            /* 按钮：青灰色背景 + 可见边框 */
            QPushButton {
                background: #dce4ec;
                border: 1px solid #8899aa;
                border-radius: 3px;
                padding: 4px 10px;
                color: #111;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #c4d4e4;
                border: 1px solid #667788;
            }
            QPushButton:pressed {
                background: #b0c4d8;
            }
            QPushButton:disabled {
                background: #e8e8e8;
                color: #999;
                border: 1px solid #ccc;
            }

            /* 下拉框：白底 + 可见边框 */
            QComboBox {
                background: #ffffff;
                border: 1px solid #8899aa;
                border-radius: 3px;
                padding: 3px 8px;
                color: #111;
                font-size: 12px;
            }
            QComboBox:hover {
                border: 1px solid #556677;
            }
            QComboBox:disabled {
                background: #f0f0f0;
                color: #999;
                border: 1px solid #ccc;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #b0b0b0;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                border: 1px solid #8899aa;
                color: #111;
                selection-background-color: #0078d7;
                selection-color: #ffffff;
            }

            /* 数字输入框：白底 + 可见边框 */
            QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #8899aa;
                border-radius: 3px;
                padding: 3px 6px;
                color: #111;
                font-size: 12px;
            }
            QDoubleSpinBox:hover {
                border: 1px solid #556677;
            }
            QDoubleSpinBox:disabled {
                background: #f0f0f0;
                color: #999;
                border: 1px solid #ccc;
            }

            /* 标签：深色文字 */
            QLabel {
                color: #222;
                font-size: 12px;
            }

            /* 只读文本框 */
            QTextEdit[readOnly="true"] {
                background: #f8f8f8;
                color: #333;
                border: 1px solid #bbb;
            }
        """)

        # --- 主窗口 ---
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle("海康相机控制 — SwarmReservoir Stage 2")
        self.main_window.setMinimumSize(900, 650)

        central = QWidget()
        central.setStyleSheet("background: #f0f0f5;")
        self.main_window.setCentralWidget(central)
        # 主布局：水平分割 — 左侧预览，右侧控制面板
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(8)

        # === 左侧：相机预览 ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # --- 状态栏 ---
        self._lbl_status = QLabel("就绪 — 请先枚举设备")
        self._lbl_status.setStyleSheet(
            "QLabel { background: #e8e8e8; color: #1a7a1a; padding: 6px; "
            "font-family: Consolas; font-size: 13px; font-weight: bold; }"
        )
        left_layout.addWidget(self._lbl_status)

        # --- 相机预览显示区域 ---
        #  相机 SDK 通过 MV_CC_DisplayOneFrame 直接绘制到 widget 的
        #  原生窗口 HDC。给该 widget 设置任何 stylesheet 都会导致 Qt
        #  接管 paintEvent 并用默认背景色覆盖相机画面 → 灰色无图像。
        #  因此：内层 QWidget 完全不设置 stylesheet，边框由外层 QFrame 提供。

        # 外层 QFrame 提供可见边框
        frame_wrapper = QFrame()
        frame_wrapper.setFrameStyle(QFrame.Box)
        frame_wrapper.setStyleSheet(
            "QFrame { border: 1px solid #aaa; background: transparent; }"
        )
        frame_wrapper_layout = QVBoxLayout(frame_wrapper)
        frame_wrapper_layout.setContentsMargins(0, 0, 0, 0)

        # 内层 QWidget：供相机 SDK 渲染，不设任何 stylesheet
        self._frame_display = QWidget()
        self._frame_display.setMinimumSize(640, 480)
        # 确保有稳定的原生窗口句柄供相机 SDK 渲染
        self._frame_display.setAttribute(Qt.WA_NativeWindow, True)
        self._frame_display.setAttribute(Qt.WA_PaintOnScreen, True)
        self._frame_display.setAutoFillBackground(False)
        frame_wrapper_layout.addWidget(self._frame_display)

        left_layout.addWidget(frame_wrapper, 1)  # stretch=1 填充剩余空间

        main_layout.addWidget(left_widget, 3)  # stretch=3

        # === 右侧：控制面板 ===
        right_widget = QWidget()
        right_widget.setMinimumWidth(280)
        right_layout = QVBoxLayout(right_widget)
        main_layout.addWidget(right_widget, 1)  # stretch=1 添加右侧面板到主布局
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # --- 设备选择 ---
        grp_dev = QGroupBox("设备选择")
        grp_dev_layout = QVBoxLayout(grp_dev)
        hbox_dev = QHBoxLayout()
        self._cmb_devices = QComboBox()
        self._cmb_devices.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        hbox_dev.addWidget(QLabel("设备:"))
        hbox_dev.addWidget(self._cmb_devices, 1)
        self._btn_enumerate = QPushButton("枚举设备")
        self._btn_enumerate.clicked.connect(self._on_enumerate)
        hbox_dev.addWidget(self._btn_enumerate)
        grp_dev_layout.addLayout(hbox_dev)
        right_layout.addWidget(grp_dev)

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
        right_layout.addWidget(grp_ctrl)

        # --- 参数设置 ---
        grp_param = QGroupBox("相机参数")
        grp_param_layout = QGridLayout(grp_param)
        config = self.load_config()

        grp_param_layout.addWidget(QLabel("帧率 (FPS):"), 0, 0)
        self._spn_framerate = QDoubleSpinBox()
        self._spn_framerate.setRange(1.0, 200.0)
        self._spn_framerate.setValue(config.get("framerate", 90.0))
        grp_param_layout.addWidget(self._spn_framerate, 0, 1)

        grp_param_layout.addWidget(QLabel("曝光 (us):"), 0, 2)
        self._spn_exposure = QDoubleSpinBox()
        self._spn_exposure.setRange(10.0, 1000000.0)
        self._spn_exposure.setDecimals(0)
        self._spn_exposure.setValue(config.get("exposuretime", 10000.0))
        grp_param_layout.addWidget(self._spn_exposure, 0, 3)

        grp_param_layout.addWidget(QLabel("增益 (dB):"), 1, 0)
        self._spn_gain = QDoubleSpinBox()
        self._spn_gain.setRange(0.0, 100.0)
        self._spn_gain.setValue(config.get("gain", 0.0))
        grp_param_layout.addWidget(self._spn_gain, 1, 1)

        grp_param_layout.addWidget(QLabel(""), 1, 2)
        btn_apply = QPushButton("应用")
        btn_apply.clicked.connect(self._on_apply_params)
        grp_param_layout.addWidget(btn_apply, 1, 3)
        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self._on_save_config)
        grp_param_layout.addWidget(btn_save, 1, 4)
        right_layout.addWidget(grp_param)

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
        grp_exp_layout.addWidget(self._btn_experiment, 1)
        self._btn_stop_experiment = QPushButton("■ 停止")
        self._btn_stop_experiment.setMinimumHeight(40)
        self._btn_stop_experiment.setStyleSheet(
            "QPushButton { background: #dc3545; color: white; font-size: 14px; "
            "font-weight: bold; border-radius: 4px; }"
            "QPushButton:disabled { background: #666; color: #999; }"
            "QPushButton:hover { background: #c82333; }"
        )
        self._btn_stop_experiment.clicked.connect(self._on_stop_experiment)
        self._btn_stop_experiment.setEnabled(False)
        grp_exp_layout.addWidget(self._btn_stop_experiment, 1)
        right_layout.addWidget(grp_exp)

        # --- 日志 ---
        grp_log = QGroupBox("日志")
        grp_log_layout = QVBoxLayout(grp_log)
        self._txt_log = QTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setMaximumHeight(120)
        self._txt_log.setStyleSheet(
            "QTextEdit { background: #f5f5f5; color: #333; font-family: Consolas; "
            "font-size: 11px; border: 1px solid #ccc; }"
        )
        grp_log_layout.addWidget(self._txt_log)
        right_layout.addWidget(grp_log)

        # --- 信号连接 ---
        self.signals.log_message.connect(self._append_log)
        self.signals.experiment_finished.connect(self._on_experiment_finished)

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
        self._stop_experiment_if_running()
        self.close_device()
        self._update_buttons()

    def _on_start_grab(self) -> None:
        """点击「开始取流」。"""
        display_hwnd = self._get_display_handle()
        self.start_grabbing(display_handle=display_hwnd)
        self._update_buttons()

    def _on_stop_grab(self) -> None:
        """点击「停止取流」。"""
        self._stop_experiment_if_running()
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
        self._btn_stop_experiment.setEnabled(True)
        self._log("=" * 60)
        self._log("实验启动!")
        self._log("=" * 60)

        if self._experiment_callback:
            # 在后台线程运行实验，避免阻塞 UI
            # 传递 self 以便回调可以设置实验引用
            threading.Thread(
                target=self._experiment_callback,
                args=(self.camera_driver, self),
                daemon=True,
            ).start()
        else:
            self._log("警告: 未设置实验回调函数")

    def _on_stop_experiment(self) -> None:
        """点击「停止实验」（只停止实验，不停相机）。"""
        self._stop_experiment_if_running()
        self._update_buttons()

    def _on_experiment_finished(self) -> None:
        """实验自然完成或异常退出时（由信号触发，确保主线程更新 UI）。"""
        self._log("实验已结束")
        self._btn_experiment.setText("▶ 启动实验")
        self._btn_experiment.setEnabled(self.is_grabbing)
        self._btn_stop_experiment.setEnabled(False)

    def _stop_experiment_if_running(self) -> None:
        """安全地停止正在运行的实验。"""
        if self._active_experiment is not None:
            self._log("正在停止实验...")
            try:
                self._active_experiment.stop()
            except Exception as e:
                self._log(f"停止实验时出错: {e}")
            self._active_experiment = None
            self._btn_experiment.setText("▶ 启动实验")
            self._btn_stop_experiment.setEnabled(False)

    def _on_window_closed(self) -> None:
        """窗口关闭时的清理。"""
        self._log("窗口关闭，清理资源...")
        self._stop_experiment_if_running()
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
        # 实验按钮：采集中 && 实验未在运行
        exp_running = self._active_experiment is not None
        self._btn_experiment.setEnabled(self.is_grabbing and not exp_running)
        if not exp_running and self._btn_experiment.text() != "▶ 启动实验":
            self._btn_experiment.setText("▶ 启动实验")
        self._btn_stop_experiment.setEnabled(exp_running)

        if self.is_grabbing:
            self._lbl_status.setText("● 取流中 — 可以启动实验")
            self._lbl_status.setStyleSheet(
                "QLabel { background: #d4edda; color: #155724; padding: 6px; "
                "font-family: Consolas; font-size: 13px; font-weight: bold; }"
            )
        elif self.is_open:
            self._lbl_status.setText("设备已打开 — 请开始取流")
            self._lbl_status.setStyleSheet(
                "QLabel { background: #fff3cd; color: #856404; padding: 6px; "
                "font-family: Consolas; font-size: 13px; font-weight: bold; }"
            )
        elif has_dev:
            self._lbl_status.setText("设备就绪 — 请打开设备")
            self._lbl_status.setStyleSheet(
                "QLabel { background: #e8e8e8; color: #1a7a1a; padding: 6px; "
                "font-family: Consolas; font-size: 13px; font-weight: bold; }"
            )
        else:
            self._lbl_status.setText("就绪 — 请先枚举设备")
            self._lbl_status.setStyleSheet(
                "QLabel { background: #e8e8e8; color: #1a7a1a; padding: 6px; "
                "font-family: Consolas; font-size: 13px; font-weight: bold; }"
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
