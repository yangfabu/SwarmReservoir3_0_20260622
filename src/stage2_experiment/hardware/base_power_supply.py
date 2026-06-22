"""
电源抽象基类

所有程控电源驱动必须继承此基类，实现核心通信和命令方法。
支持通过品牌名称注册和工厂创建。
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type


class PowerSupplyError(Exception):
    """电源相关错误。"""
    pass


class PowerSupplyConnectionError(PowerSupplyError):
    """电源连接错误。"""
    pass


class BasePowerSupply(ABC):
    """程控电源的抽象基类。"""

    @abstractmethod
    def connect(self, port: str, baudrate: int) -> bool:
        """打开串口/网络连接。"""
        ...

    @abstractmethod
    def send_command(self, command: str) -> bool:
        """发送 SCPI 命令字符串。"""
        ...

    @abstractmethod
    def set_current(self, value: float) -> bool:
        """设置输出电流值 (A)。"""
        ...

    @abstractmethod
    def set_voltage(self, value: float) -> bool:
        """设置输出电压值 (V)。"""
        ...

    @abstractmethod
    def enable_output(self) -> bool:
        """打开电源输出。"""
        ...

    @abstractmethod
    def disable_output(self) -> bool:
        """关闭电源输出。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """断开连接并释放资源。"""
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# 电源注册表
_power_supply_registry: Dict[str, Type[BasePowerSupply]] = {}


def register_power_supply(cls: Type[BasePowerSupply]) -> Type[BasePowerSupply]:
    """装饰器：注册电源驱动类。"""
    _power_supply_registry[cls.__name__] = cls
    return cls


def create_power_supply(model: str, **kwargs) -> BasePowerSupply:
    """根据型号名称创建电源实例。"""
    for cls in _power_supply_registry.values():
        if cls.__name__.lower().startswith(model.lower()):
            return cls(**kwargs)
    available = ", ".join(_power_supply_registry.keys())
    raise PowerSupplyError(
        f"未知电源型号: '{model}'。可用驱动: {available}"
    )
