"""Tests for memory pool."""
import pytest
from zynkode_compute.device import Device
from zynkode_compute.memory import MemoryPool

class TestMemoryPool:
    def test_pool_creation(self):
        dev = Device(0)
        pool = MemoryPool(dev, size="1GB")
        assert pool.total_bytes == 1024**3
    
    def test_allocate_and_free(self):
        dev = Device(0)
        pool = MemoryPool(dev, size="1GB")
        buf = pool.allocate(1024 * 1024)
        assert pool.used_bytes == 1024 * 1024
        pool.free(buf)
        assert pool.used_bytes == 0
    
    def test_pool_exhaustion(self):
        dev = Device(0)
        pool = MemoryPool(dev, size="1MB")
        with pytest.raises(MemoryError):
            pool.allocate(2 * 1024 * 1024)
