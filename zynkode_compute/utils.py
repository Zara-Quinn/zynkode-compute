"""Utility functions for Zynkode Compute."""

import time
import functools
from typing import Callable


def timed(func: Callable) -> Callable:
    """Decorator that prints execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed*1000:.2f}ms")
        return result
    return wrapper


def format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def check_rocm() -> dict:
    """Check ROCm installation status."""
    import subprocess
    try:
        result = subprocess.run(["rocm-smi", "--version"], capture_output=True, text=True, timeout=10)
        return {"available": True, "version": result.stdout.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False, "version": "not found"}
