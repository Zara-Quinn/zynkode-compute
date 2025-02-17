"""Runtime initialization and HIP device management."""

import os
from typing import Optional
from .device import Device
from .memory import MemoryPool


class Runtime:
    """Central runtime for Zynkode Compute.
    
    Manages device selection, memory pools, and provides
    a context manager for safe GPU resource cleanup.
    """
    
    _instance: Optional["Runtime"] = None
    
    def __init__(self, device_id: int = 0, pool_size: str = "4GB"):
        self.device = Device(device_id)
        self.pool = MemoryPool(self.device, pool_size)
        Runtime._instance = self
    
    @classmethod
    def get_instance(cls) -> "Runtime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def cleanup(self):
        for buf in list(self.pool.allocated.values()):
            self.pool.free(buf)
    
    def info(self) -> dict:
        return {
            "device": repr(self.device),
            "pool": self.pool.stats(),
            "rocm_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", "all"),
        }
