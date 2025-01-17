"""Device abstraction layer for AMD GPU hardware."""

import subprocess
from dataclasses import dataclass
from typing import List


@dataclass
class GPUInfo:
    """Hardware information for a single GPU device."""
    device_id: int
    name: str
    vram_total_mb: int
    vram_free_mb: int
    compute_units: int
    clock_mhz: int
    arch: str  # CDNA1, CDNA2, CDNA3


class Device:
    """Represents a single AMD GPU device.
    
    Wraps ROCm device queries and provides memory/compute
    information needed by the scheduler and memory manager.
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._info = None
        self._initialize()
    
    def _initialize(self):
        """Query device properties from ROCm runtime."""
        self._info = GPUInfo(
            device_id=self.device_id,
            name="AMD GPU",
            vram_total_mb=0, vram_free_mb=0,
            compute_units=0, clock_mhz=0, arch="unknown"
        )
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--json", "-d", str(self.device_id)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                self._parse_device_info(data)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    
    def _parse_device_info(self, data: dict):
        arch_map = {
            "MI210": "CDNA2", "MI250": "CDNA2", "MI250X": "CDNA2",
            "MI300A": "CDNA3", "MI300X": "CDNA3", "MI325X": "CDNA3",
        }
        name = data.get("card_series", "AMD GPU")
        arch = "CDNA"
        for key, val in arch_map.items():
            if key in name:
                arch = val
                break
        self._info = GPUInfo(
            device_id=self.device_id, name=name,
            vram_total_mb=data.get("vram_total_mb", 0),
            vram_free_mb=data.get("vram_free_mb", 0),
            compute_units=data.get("compute_units", 0),
            clock_mhz=data.get("clock_mhz", 0), arch=arch,
        )
    
    @property
    def info(self) -> GPUInfo:
        return self._info
    
    @staticmethod
    def count() -> int:
        """Return the number of AMD GPUs available."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
                return max(1, len(lines) - 1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return 0
    
    @staticmethod
    def list_all() -> List["Device"]:
        """Return Device objects for all available GPUs."""
        return [Device(i) for i in range(Device.count())]
    
    def __repr__(self):
        return f"Device(id={self.device_id}, name='{self.info.name}')"
