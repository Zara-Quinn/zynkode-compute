"""Memory management for GPU buffers."""

from typing import Dict
from dataclasses import dataclass, field


@dataclass
class Buffer:
    """Represents a GPU memory buffer."""
    ptr: int
    size: int
    device_id: int
    unified: bool = True
    _freed: bool = field(default=False, repr=False)
    
    def free(self):
        if not self._freed:
            self._freed = True
    
    def __del__(self):
        self.free()


class MemoryPool:
    """Pooled GPU memory allocator.
    
    Pre-allocates a large chunk of GPU memory and sub-allocates
    from it to avoid repeated hipMalloc/hipFree calls.
    """
    
    def __init__(self, device, size: str = "4GB"):
        self.device = device
        self.total_bytes = self._parse_size(size)
        self.allocated: Dict[int, Buffer] = {}
        self._offset = 0
    
    def _parse_size(self, size_str: str) -> int:
        units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
        size_str = size_str.upper().strip()
        for unit, mult in units.items():
            if size_str.endswith(unit):
                return int(size_str[:-len(unit)]) * mult
        return int(size_str)
    
    def allocate(self, size: int, unified: bool = True) -> Buffer:
        if self._offset + size > self.total_bytes:
            raise MemoryError(f"Pool exhausted: {size} bytes requested")
        buf = Buffer(ptr=self._offset, size=size, device_id=self.device.device_id, unified=unified)
        self.allocated[buf.ptr] = buf
        self._offset += size
        return buf
    
    def free(self, buffer: Buffer):
        if buffer.ptr in self.allocated:
            del self.allocated[buffer.ptr]
            buffer.free()
    
    @property
    def used_bytes(self) -> int:
        return sum(b.size for b in self.allocated.values())
    
    @property
    def free_bytes(self) -> int:
        return self.total_bytes - self.used_bytes
    
    def stats(self) -> dict:
        return {
            "total": self.total_bytes, "used": self.used_bytes,
            "free": self.free_bytes, "buffers": len(self.allocated),
            "utilization": self.used_bytes / self.total_bytes if self.total_bytes else 0,
        }
