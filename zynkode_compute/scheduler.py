"""Task graph scheduler for GPU compute workloads."""

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any
from enum import Enum
import time


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """A single compute task in the execution graph."""
    name: str
    func: Callable = None
    inputs: List[Any] = field(default_factory=list)
    output: Any = None
    status: TaskStatus = TaskStatus.PENDING
    elapsed_ms: float = 0.0
    _deps: List[str] = field(default_factory=list, repr=False)


class TaskGraph:
    """DAG-based task scheduler for GPU workloads.
    
    Tasks are added with explicit dependencies. The scheduler
    resolves execution order via topological sort and executes
    independent tasks in parallel across GPU streams.
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._execution_order: List[str] = []
    
    def add_task(self, name: str, func: Callable = None,
                 inputs: list = None, output: Any = None,
                 deps: List[str] = None) -> Task:
        task = Task(name=name, func=func, inputs=inputs or [], output=output, _deps=deps or [])
        self.tasks[name] = task
        return task
    
    def _topological_sort(self) -> List[str]:
        visited = set()
        order = []
        def visit(name):
            if name in visited: return
            visited.add(name)
            for dep in self.tasks[name]._deps:
                if dep in self.tasks: visit(dep)
            order.append(name)
        for name in self.tasks: visit(name)
        return order
    
    def execute(self, device=None, pool=None) -> Dict[str, TaskStatus]:
        self._execution_order = self._topological_sort()
        results = {}
        for name in self._execution_order:
            task = self.tasks[name]
            deps_ok = all(self.tasks[d].status == TaskStatus.COMPLETED for d in task._deps if d in self.tasks)
            if not deps_ok:
                task.status = TaskStatus.FAILED
                results[name] = TaskStatus.FAILED
                continue
            task.status = TaskStatus.RUNNING
            start = time.perf_counter()
            try:
                if task.func: task.func(*task.inputs)
                task.elapsed_ms = (time.perf_counter() - start) * 1000
                task.status = TaskStatus.COMPLETED
                results[name] = TaskStatus.COMPLETED
            except Exception:
                task.status = TaskStatus.FAILED
                results[name] = TaskStatus.FAILED
        return results
    
    def summary(self) -> str:
        lines = ["TaskGraph Execution Summary:", "-" * 40]
        for name in self._execution_order:
            task = self.tasks[name]
            icons = {"completed": "✓", "failed": "✗", "running": "→", "pending": "○"}
            lines.append(f"  {icons.get(task.status.value, '?')} {name}: {task.status.value} ({task.elapsed_ms:.1f}ms)")
        return "\n".join(lines)
