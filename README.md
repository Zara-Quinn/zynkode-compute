# Zynkode Compute

A high-performance GPU compute orchestration framework built on AMD ROCm/HIP.

## Overview

Zynkode Compute provides a unified abstraction layer over ROCm and HIP, enabling developers to write GPU-accelerated workloads that run natively on AMD Instinct MI200/MI300 series accelerators.

## Architecture

```
┌─────────────────────────────────────┐
│         Application Layer           │
├─────────────────────────────────────┤
│      Zynkode Compute API            │
│  (Python bindings + C++ core)       │
├─────────────────────────────────────┤
│    Memory Manager │ Task Scheduler  │
├─────────────────────────────────────┤
│         ROCm / HIP Runtime          │
├─────────────────────────────────────┤
│    AMD Instinct MI200/MI300 GPU     │
└─────────────────────────────────────┘
```

## Why AMD?

AMD's ROCm ecosystem has matured significantly. With CDNA architecture powering the MI300X (192GB HBM3), we see a clear path for cost-effective, high-throughput compute. Zynkode Compute is built ground-up for this hardware.

## Features

- Zero-copy memory management between CPU and GPU
- Async task graph execution with dependency resolution
- Multi-GPU coordination across PCIe and Infinity Fabric
- Built-in profiling hooks for compute/memory/thermal metrics
- Python API with C++ backend for maximum throughput

## Quick Start

```python
from zynkode_compute import Device, MemoryPool, TaskGraph

device = Device(0)
pool = MemoryPool(device, size="8GB")

graph = TaskGraph()
graph.add_task("matmul", func=matmul_kernel, inputs=[A, B], output=C)
graph.add_task("bias_add", func=bias_kernel, inputs=[C, bias], output=D, deps=["matmul"])
graph.execute()
```

## Requirements

- AMD ROCm 6.0+
- AMD Instinct MI200/MI300 series GPU
- Python 3.10+

## License

MIT

# Improvement: Update README with multi-GPU usage examples
