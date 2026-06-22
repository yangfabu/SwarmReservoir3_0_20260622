"""
海康工业相机驱动封装

基于海康 MvCamera SDK (v3.x) 的 Python 封装。
关键改进：用 threading.Event 优雅关闭线程，替代旧代码的暴力 kill 线程。

迁移自: Current_Input/CamOperation_class.py -> CameraOperation 类
"""

import threading
import time
import os
import platform
import queue
from ctypes import *
from typing import Optional

# ============================================================
# SDK 库路径设置
# ============================================================
_current_dir = os.path.dirname(os.path.abspath(__file__))
if platform.system() == "Windows":
    # 设置海康 SDK 的 common runenv 路径
    sdk_samples = os.path.join(
        os.getenv("MVCAM_COMMON_RUNENV", ""), "Samples", "Python", "MvImport"
    )
    if sdk_samples and os.path.exists(sdk_samples):
        import sys
        sys.path.insert(0, sdk_samples)

# 本地 lib/ 中的 SDK 头文件
from .lib.CameraParams_header import *
from .lib.MvCameraControl_class import *


# ============================================================
# 工具函数
# ============================================================

def _to_hex_str(num: int) -> str:
    """整数转十六进制字符串。"""
    cha_dict = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    if num < 0:
        num = num + 2 ** 32
    hex_str = ""
    while num >= 16:
        digit = num % 16
        hex_str = cha_dict.get(digit, str(digit)) + hex_str
        num //= 16
    hex_str = cha_dict.get(num, str(num)) + hex_str
    return hex_str


def _is_mono_data(pixel_type: int) -> bool:
    """判断是否为 Mono 格式。"""
    return pixel_type in (
        PixelType_Gvsp_Mono8, PixelType_Gvsp_Mono10,
        PixelType_Gvsp_Mono10_Packed, PixelType_Gvsp_Mono12,
        PixelType_Gvsp_Mono12_Packed,
    )


def _is_color_data(pixel_type: int) -> bool:
    """判断是否为彩色格式。"""
    color_types = (
        PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGR10, PixelType_Gvsp_BayerRG10,
        PixelType_Gvsp_BayerGB10, PixelType_Gvsp_BayerBG10,
        PixelType_Gvsp_BayerGR12, PixelType_Gvsp_BayerRG12,
        PixelType_Gvsp_BayerGB12, PixelType_Gvsp_BayerBG12,
        PixelType_Gvsp_BayerGR10_Packed, PixelType_Gvsp_BayerRG10_Packed,
        PixelType_Gvsp_BayerGB10_Packed, PixelType_Gvsp_BayerBG10_Packed,
        PixelType_Gvsp_BayerGR12_Packed, PixelType_Gvsp_BayerRG12_Packed,
        PixelType_Gvsp_BayerGB12_Packed, PixelType_Gvsp_BayerBG12_Packed,
        PixelType_Gvsp_BayerRBGG8,
        PixelType_Gvsp_BayerGR16, PixelType_Gvsp_BayerRG16,
        PixelType_Gvsp_BayerGB16, PixelType_Gvsp_BayerBG16,
        PixelType_Gvsp_YUV422_Packed, PixelType_Gvsp_YUV422_YUYV_Packed,
    )
    return pixel_type in color_types


# ============================================================
# CameraDriver 类
# ============================================================

class CameraError(Exception):
    """相机相关错误。"""
    pass


class CameraDriver:
    """
    海康工业相机的 Python 驱动封装。

    Usage:
        # 枚举设备
        device_list = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        cam = MvCamera()

        # 创建驱动
        driver = CameraDriver(cam, device_list, 0)
        driver.open_device()
        driver.start_grabbing(0)
        driver.save_jpg("test.jpg")
        driver.stop_grabbing()
        driver.close_device()
    """

    def __init__(
        self,
        obj_cam,
        st_device_list,
        n_connect_num: int = 0,
    ):
        """
        Args:
            obj_cam: MvCamera 实例。
            st_device_list: MV_CC_DEVICE_INFO_LIST 实例。
            n_connect_num: 设备索引号。
        """
        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num

        # 设备状态
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit = False

        # 帧信息
        self.st_frame_info = MV_FRAME_OUT_INFO_EX()
        self.buf_save_image = None
        self.buf_save_image_len = 0

        # 缓存锁
        self.buf_lock = threading.Lock()

        # === 关键改进：用 Event 替代暴力 kill 线程 ===
        self._exit_event = threading.Event()
        self._work_thread: Optional[threading.Thread] = None

        # 相机参数缓存
        self.frame_rate: float = 0.0
        self.exposure_time: float = 0.0
        self.gain: float = 0.0

        # 后台保存队列
        self.image_queue = queue.Queue(maxsize=300)
        self._save_thread = threading.Thread(
            target=self._background_save_worker, daemon=True
        )
        self._save_thread.start()

    # ================================================================
    # 设备生命周期
    # ================================================================

    def open_device(self) -> int:
        """打开相机设备。返回 SDK 状态码。"""
        if self.b_open_device:
            return MV_E_CALLORDER
        if self.n_connect_num < 0:
            return MV_E_CALLORDER

        n_conn = int(self.n_connect_num)
        st_device_list = cast(
            self.st_device_list.pDeviceInfo[int(n_conn)],
            POINTER(MV_CC_DEVICE_INFO),
        ).contents

        self.obj_cam = MvCamera()
        ret = self.obj_cam.MV_CC_CreateHandle(st_device_list)
        if ret != 0:
            self.obj_cam.MV_CC_DestroyHandle()
            return ret

        ret = self.obj_cam.MV_CC_OpenDevice()
        if ret != 0:
            return ret
        print("[相机] 设备已打开")
        self.b_open_device = True
        self._exit_event.clear()

        # GigE: 探测最佳包大小
        if st_device_list.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            n_packet_size = self.obj_cam.MV_CC_GetOptimalPacketSize()
            if int(n_packet_size) > 0:
                self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", n_packet_size)

        # 设置触发模式为 OFF（连续采集）
        self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        return MV_OK

    def start_grabbing(self, win_handle: int = 0) -> int:
        """开始采集。win_handle: 显示窗口句柄（0 = 不显示）。"""
        if not self.b_open_device or self.b_start_grabbing:
            return MV_E_CALLORDER

        self._exit_event.clear()
        self.b_exit = False

        ret = self.obj_cam.MV_CC_StartGrabbing()
        if ret != 0:
            return ret
        self.b_start_grabbing = True
        print("[相机] 开始采集")

        self._work_thread = threading.Thread(
            target=self._work_loop, args=(win_handle,), daemon=True
        )
        self._work_thread.start()
        return MV_OK

    def stop_grabbing(self) -> int:
        """停止采集（优雅关闭）。"""
        if not self.b_start_grabbing:
            return MV_E_CALLORDER

        # === 关键改进：设置退出标志，等待线程自行结束 ===
        self._exit_event.set()
        self.b_exit = True

        if self._work_thread and self._work_thread.is_alive():
            self._work_thread.join(timeout=3.0)

        ret = self.obj_cam.MV_CC_StopGrabbing()
        if ret != 0:
            return ret
        print("[相机] 停止采集")
        self.b_start_grabbing = False
        return MV_OK

    def close_device(self) -> int:
        """关闭相机设备。"""
        if not self.b_open_device:
            return MV_E_CALLORDER

        self._exit_event.set()
        self.b_exit = True

        if self._work_thread and self._work_thread.is_alive():
            self._work_thread.join(timeout=3.0)

        self.obj_cam.MV_CC_CloseDevice()
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        print("[相机] 设备已关闭")

        # 清理后台保存队列
        try:
            self.image_queue.put_nowait(None)
        except queue.Full:
            pass
        return MV_OK

    # ================================================================
    # 参数读写
    # ================================================================

    def get_parameter(self) -> int:
        """读取相机当前参数 (帧率、曝光、增益)。"""
        if not self.b_open_device:
            return MV_E_CALLORDER

        st_float = MVCC_FLOATVALUE()
        memset(byref(st_float), 0, sizeof(MVCC_FLOATVALUE))

        self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", st_float)
        self.frame_rate = st_float.fCurValue

        self.obj_cam.MV_CC_GetFloatValue("ExposureTime", st_float)
        self.exposure_time = st_float.fCurValue

        self.obj_cam.MV_CC_GetFloatValue("Gain", st_float)
        self.gain = st_float.fCurValue

        return MV_OK

    def set_parameter(self, frame_rate: float, exposure_time: float, gain: float) -> int:
        """设置相机参数。"""
        if not self.b_open_device:
            return MV_E_CALLORDER

        # 关自动曝光
        self.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        time.sleep(0.05)

        ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
        if ret != 0:
            return ret

        ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain))
        if ret != 0:
            return ret

        ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frame_rate))
        if ret != 0:
            return ret

        return MV_OK

    # ================================================================
    # 触发控制
    # ================================================================

    def set_trigger_mode(self, is_trigger: bool) -> int:
        """设置触发模式。"""
        if not self.b_open_device:
            return MV_E_CALLORDER
        mode = 1 if is_trigger else 0
        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", mode)
        if ret == 0 and is_trigger:
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSource", 7)
        return ret

    def trigger_once(self) -> int:
        """软触发一次。"""
        if self.b_open_device:
            return self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
        return MV_E_CALLORDER

    # ================================================================
    # 图像保存
    # ================================================================

    def _background_save_worker(self) -> None:
        """后台存图线程：从队列获取帧数据并写入磁盘。"""
        while True:
            try:
                task = self.image_queue.get()
                if task is None:  # 退出信号
                    break

                local_buf, pixel_type, n_width, n_height, data_len, file_path, img_type = task

                c_file_path = file_path.encode("ascii")

                st_param = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
                st_param.enPixelType = pixel_type
                st_param.nWidth = n_width
                st_param.nHeight = n_height
                st_param.nDataLen = data_len
                st_param.pData = cast(local_buf, POINTER(c_ubyte))
                st_param.enImageType = img_type
                if img_type == MV_Image_Jpeg:
                    st_param.nQuality = 80
                st_param.pcImagePath = create_string_buffer(c_file_path)
                st_param.iMethodValue = 1

                ret = self.obj_cam.MV_CC_SaveImageToFileEx(st_param)
                if ret != 0:
                    print(f"[相机] 后台保存失败: 0x{ret:X}")

            except Exception as e:
                print(f"[相机] 后台保存线程异常: {e}")
            finally:
                try:
                    self.image_queue.task_done()
                except ValueError:
                    pass

    def save_jpg(self, filename: Optional[str] = None) -> Optional[int]:
        """保存 JPG 图像（异步，立即返回）。"""
        return self._save_image(filename, MV_Image_Jpeg)

    def save_bmp(self, filename: Optional[str] = None) -> Optional[int]:
        """保存 BMP 图像（异步，立即返回）。"""
        return self._save_image(filename, MV_Image_Bmp)

    def _save_image(self, filename: Optional[str], img_type: int) -> Optional[int]:
        """
        从当前帧缓存保存图像到磁盘。
        使用深拷贝 + 队列实现非阻塞保存。
        Returns:
            0: 成功入队
            None: 缓存为空
            -1: 队列已满（丢帧）
        """
        self.buf_lock.acquire()
        if self.buf_save_image is None:
            self.buf_lock.release()
            return None

        # 深拷贝当前帧数据
        local_data_len = self.st_frame_info.nFrameLen
        local_buf = (c_ubyte * local_data_len)()
        memmove(byref(local_buf), self.buf_save_image, local_data_len)

        pixel_type = self.st_frame_info.enPixelType
        n_width = self.st_frame_info.nWidth
        n_height = self.st_frame_info.nHeight
        self.buf_lock.release()

        if filename is None:
            file_path = str(self.st_frame_info.nFrameNum) + (
                ".jpg" if img_type == MV_Image_Jpeg else ".bmp"
            )
        else:
            file_path = filename

        try:
            task = (local_buf, pixel_type, n_width, n_height, local_data_len, file_path, img_type)
            self.image_queue.put_nowait(task)
            return 0
        except queue.Full:
            print("[相机] 警告: 保存队列已满，帧已丢弃!")
            return -1

    # ================================================================
    # 取图循环（工作线程）
    # ================================================================

    def _work_loop(self, win_handle: int) -> None:
        """主取图循环。使用 _exit_event 实现优雅退出。"""
        st_out_frame = MV_FRAME_OUT()
        memset(byref(st_out_frame), 0, sizeof(st_out_frame))

        while not self._exit_event.is_set():
            ret = self.obj_cam.MV_CC_GetImageBuffer(st_out_frame, 500)
            if ret != 0:
                continue

            # 拷贝图像到共享缓存
            self.buf_lock.acquire()
            try:
                if self.buf_save_image_len < st_out_frame.stFrameInfo.nFrameLen:
                    self.buf_save_image = (c_ubyte * st_out_frame.stFrameInfo.nFrameLen)()
                    self.buf_save_image_len = st_out_frame.stFrameInfo.nFrameLen

                memmove(
                    byref(self.st_frame_info),
                    byref(st_out_frame.stFrameInfo),
                    sizeof(MV_FRAME_OUT_INFO_EX),
                )
                memmove(
                    byref(self.buf_save_image),
                    st_out_frame.pBufAddr,
                    self.st_frame_info.nFrameLen,
                )
            finally:
                self.buf_lock.release()

            # 释放 SDK 缓存
            self.obj_cam.MV_CC_FreeImageBuffer(st_out_frame)

            # 如果需要显示
            if win_handle != 0:
                st_display = MV_DISPLAY_FRAME_INFO()
                memset(byref(st_display), 0, sizeof(st_display))
                st_display.hWnd = int(win_handle)
                st_display.nWidth = self.st_frame_info.nWidth
                st_display.nHeight = self.st_frame_info.nHeight
                st_display.enPixelType = self.st_frame_info.enPixelType
                st_display.pData = self.buf_save_image
                st_display.nDataLen = self.st_frame_info.nFrameLen
                self.obj_cam.MV_CC_DisplayOneFrame(st_display)

        # 退出时清理缓存
        if self.buf_save_image is not None:
            del self.buf_save_image
            self.buf_save_image = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.b_start_grabbing:
            self.stop_grabbing()
        if self.b_open_device:
            self.close_device()
        return False
