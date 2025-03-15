# Architecture

## Design Philosophy

Zynkode Compute is built around three core principles:

1. **Hardware-Native Performance**: Target AMD CDNA architecture directly through HIP
2. **Developer Ergonomics**: Python API with C++ backend for maximum throughput
3. **Production Reliability**: Error handling, timeout guards, and resource cleanup

## Component Overview

### Device Layer (`device.py`)
Queries AMD GPU hardware through ROCm system calls.

### Memory Manager (`memory.py`)
Pooled allocator that pre-allocates GPU memory to avoid hipMalloc/hipFree overhead.

### Task Scheduler (`scheduler.py`)
DAG-based scheduler that resolves dependencies and runs independent tasks in parallel.

### Runtime (`runtime.py`)
Singleton managing device connections and memory pools with context manager cleanup.

## Data Flow

```
User Code -> TaskGraph.execute() -> Topological Sort -> Stream Assignment
           -> Memory Allocation -> Kernel Launch -> Sync & Cleanup
```
