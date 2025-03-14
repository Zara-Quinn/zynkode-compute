"""Tests for task scheduler."""
import pytest
from zynkode_compute.scheduler import TaskGraph, TaskStatus

class TestTaskGraph:
    def test_add_task(self):
        graph = TaskGraph()
        task = graph.add_task("test", func=lambda: None)
        assert task.name == "test"
    
    def test_execution_order(self):
        graph = TaskGraph()
        graph.add_task("a", func=lambda: None)
        graph.add_task("b", func=lambda: None, deps=["a"])
        order = graph._topological_sort()
        assert order.index("a") < order.index("b")
    
    def test_execute_simple(self):
        graph = TaskGraph()
        graph.add_task("noop", func=lambda: None)
        results = graph.execute()
        assert results["noop"] == TaskStatus.COMPLETED
