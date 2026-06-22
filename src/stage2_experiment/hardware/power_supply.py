"""
ITECH 程控电源串口驱动

基于 SCPI 协议的串口通信实现，支持 ITECH 品牌电源。
通过 pyserial 发送 SCPI 命令。

迁移自: Current_Input/Serial.py -> SimpleSerial 类
"""

import threading
import time
from typing import Callable, Optional

import serial

from .base_power_supply import (
    BasePowerSupply,
    PowerSupplyConnectionError,
    PowerSupplyError,
    register_power_supply,
)


@register_power_supply
class ITECHPowerSupply(BasePowerSupply):
    """
    ITECH 品牌程控电源的 SCPI 串口驱动。

    Usage:
        psu = ITECHPowerSupply()
        psu.connect("COM8", 9600)
        psu.set_voltage(15.0)
        psu.enable_output()
        psu.set_current(1.5)
        psu.disable_output()
        psu.close()

        # 或使用上下文管理器
        with ITECHPowerSupply() as psu:
            psu.connect("COM8", 9600)
            ...
    """

    def __init__(
        self,
        port: str = "COM8",
        baudrate: int = 9600,
        timeout: float = 1.0,
    ):
        """
        Args:
            port: 串口号 (如 "COM8")。
            baudrate: 波特率 (默认 9600)。
            timeout: 串口读取超时 (秒)。
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._receiving = False
        self._thread: Optional[threading.Thread] = None
        self._receive_callback: Optional[Callable[[str], None]] = None

    def set_receive_callback(self, callback: Callable[[str], None]) -> None:
        """设置接收数据的回调函数。"""
        self._receive_callback = callback

    def connect(self, port: str, baudrate: int) -> bool:
        """打开串口连接。"""
        self.port = port
        self.baudrate = baudrate
        try:
            self._ser = serial.Serial(
                self.port, self.baudrate, timeout=self.timeout
            )
            print(f"[电源] 串口已打开: {self.port} @ {self.baudrate}")
            return True
        except serial.SerialException as e:
            raise PowerSupplyConnectionError(
                f"无法打开串口 {self.port}: {e}"
            )

    def start_receiving(self) -> bool:
        """开始后台接收数据。"""
        if not self._ser or not self._ser.is_open:
            return False
        self._receiving = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        return True

    def _receive_loop(self) -> None:
        """后台接收线程。"""
        while self._receiving and self._ser and self._ser.is_open:
            try:
                if self._ser.in_waiting > 0:
                    data = (
                        self._ser.readline()
                        .decode("utf-8", errors="ignore")
                        .strip()
                    )
                    if data and self._receive_callback:
                        self._receive_callback(data)
                time.sleep(0.01)
            except (serial.SerialException, OSError):
                if self._receiving:
                    break

    def send_command(self, command: str) -> bool:
        """
        发送 SCPI 命令字符串。

        Args:
            command: SCPI 命令，自动追加换行符。

        Returns:
            发送成功返回 True。
        """
        if not self._ser or not self._ser.is_open:
            raise PowerSupplyConnectionError("串口未打开，无法发送命令")

        try:
            if not command.endswith("\n"):
                command += "\n"
            self._ser.write(command.encode("utf-8"))
            return True
        except serial.SerialException as e:
            raise PowerSupplyError(f"发送命令失败 '{command.strip()}': {e}")

    def set_current(self, value: float) -> bool:
        """设置输出电流 (A)。"""
        return self.send_command(f"CURR {value}")

    def set_voltage(self, value: float) -> bool:
        """设置输出电压 (V)。"""
        return self.send_command(f"VOLT {value}")

    def enable_output(self) -> bool:
        """打开电源输出。"""
        return self.send_command("OUTP 1")

    def disable_output(self) -> bool:
        """关闭电源输出。"""
        return self.send_command("OUTP 0")

    def enable_remote(self) -> bool:
        """切换到远程控制模式。"""
        return self.send_command("SYST:REM")

    def enable_local(self) -> bool:
        """恢复到本地控制模式。"""
        return self.send_command("SYST:LOC")

    def select_channel(self, channel: int = 1) -> bool:
        """选择输出通道。"""
        return self.send_command(f"INST CH{channel}")

    def close(self) -> None:
        """断开连接并清理资源。"""
        self._receiving = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[电源] 串口已关闭")
