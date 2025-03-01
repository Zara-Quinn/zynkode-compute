"""Tests for device abstraction layer."""
import pytest
from zynkode_compute.device import Device, GPUInfo

class TestDevice:
    def test_device_creation(self):
        dev = Device(0)
        assert dev.device_id == 0
        assert dev.info is not None
    
    def test_gpu_info_fields(self):
        info = GPUInfo(device_id=0, name="Test", vram_total_mb=80000,
                       vram_free_mb=79000, compute_units=100, clock_mhz=1700, arch="CDNA3")
        assert info.arch == "CDNA3"
    
    def test_count_returns_int(self):
        assert isinstance(Device.count(), int)
