"""
Performance profiling module for PSTM.

Provides profiling utilities and performance analysis.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
import psutil

from pstm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimingResult:
    """Result from a timing measurement."""

    name: str
    elapsed_s: float
    n_iterations: int = 1

    @property
    def per_iteration_s(self) -> float:
        """Time per iteration."""
        return self.elapsed_s / self.n_iterations

    @property
    def per_iteration_ms(self) -> float:
        """Time per iteration in milliseconds."""
        return self.per_iteration_s * 1000


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of total system memory

    @classmethod
    def capture(cls) -> "MemorySnapshot":
        """Capture current memory usage."""
        process = psutil.Process()
        mem_info = process.memory_info()
        return cls(
            timestamp=time.time(),
            rss_mb=mem_info.rss / (1024 ** 2),
            vms_mb=mem_info.vms / (1024 ** 2),
            percent=process.memory_percent(),
        )


@dataclass
class ProfileResult:
    """Complete profiling result."""

    name: str
    timing: TimingResult
    memory_start: MemorySnapshot
    memory_end: MemorySnapshot
    profile_stats: str | None = None

    @property
    def memory_delta_mb(self) -> float:
        """Change in memory usage."""
        return self.memory_end.rss_mb - self.memory_start.rss_mb


class Timer:
    """
    High-precision timer for performance measurement.

    Usage:
        timer = Timer()
        timer.start()
        # ... do work ...
        timer.stop()
        print(f"Elapsed: {timer.elapsed_s:.3f}s")

    Or as context manager:
        with Timer() as t:
            # ... do work ...
        print(f"Elapsed: {t.elapsed_s:.3f}s")
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._start_time: float = 0
        self._end_time: float = 0
        self._running = False

    def start(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._running = True
        return self

    def stop(self) -> "Timer":
        """Stop the timer."""
        self._end_time = time.perf_counter()
        self._running = False
        return self

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        if self._running:
            return time.perf_counter() - self._start_time
        return self._end_time - self._start_time

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_s * 1000

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()


class Profiler:
    """
    Performance profiler for PSTM operations.

    Collects timing, memory, and cProfile statistics.
    """

    def __init__(self, name: str = "PSTM Profiler"):
        self.name = name
        self.results: list[ProfileResult] = []
        self._timings: dict[str, list[float]] = {}

    @contextmanager
    def profile(
        self,
        name: str,
        include_cprofile: bool = False,
    ) -> Generator[None, None, None]:
        """
        Profile a code block.

        Args:
            name: Name for this profiling section
            include_cprofile: Include detailed cProfile statistics

        Yields:
            None

        Example:
            profiler = Profiler()
            with profiler.profile("migration"):
                run_migration()
        """
        mem_start = MemorySnapshot.capture()
        timer = Timer(name)

        # Optional cProfile
        cp = None
        if include_cprofile:
            cp = cProfile.Profile()
            cp.enable()

        timer.start()
        try:
            yield
        finally:
            timer.stop()

            if cp:
                cp.disable()

            mem_end = MemorySnapshot.capture()

            # Get cProfile stats as string
            profile_stats = None
            if cp:
                s = io.StringIO()
                ps = pstats.Stats(cp, stream=s).sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 functions
                profile_stats = s.getvalue()

            result = ProfileResult(
                name=name,
                timing=TimingResult(name=name, elapsed_s=timer.elapsed_s),
                memory_start=mem_start,
                memory_end=mem_end,
                profile_stats=profile_stats,
            )
            self.results.append(result)

            # Track for averaging
            if name not in self._timings:
                self._timings[name] = []
            self._timings[name].append(timer.elapsed_s)

    def time_function(
        self,
        func: Callable,
        *args,
        n_iterations: int = 1,
        warmup: int = 0,
        **kwargs,
    ) -> TimingResult:
        """
        Time a function call.

        Args:
            func: Function to time
            *args: Positional arguments
            n_iterations: Number of iterations
            warmup: Warmup iterations (not timed)
            **kwargs: Keyword arguments

        Returns:
            TimingResult
        """
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)

        # Timed iterations
        timer = Timer()
        timer.start()
        for _ in range(n_iterations):
            func(*args, **kwargs)
        timer.stop()

        return TimingResult(
            name=func.__name__,
            elapsed_s=timer.elapsed_s,
            n_iterations=n_iterations,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get profiling summary."""
        summary = {
            "name": self.name,
            "n_sections": len(self.results),
            "sections": {},
        }

        for name, times in self._timings.items():
            summary["sections"][name] = {
                "n_calls": len(times),
                "total_s": sum(times),
                "mean_s": np.mean(times),
                "std_s": np.std(times) if len(times) > 1 else 0,
                "min_s": min(times),
                "max_s": max(times),
            }

        return summary

    def print_summary(self) -> None:
        """Print profiling summary."""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title=f"Profile Summary: {self.name}")
        table.add_column("Section", style="cyan")
        table.add_column("Calls", justify="right")
        table.add_column("Total (s)", justify="right")
        table.add_column("Mean (ms)", justify="right")
        table.add_column("Std (ms)", justify="right")

        for name, times in self._timings.items():
            table.add_row(
                name,
                str(len(times)),
                f"{sum(times):.3f}",
                f"{np.mean(times) * 1000:.2f}",
                f"{np.std(times) * 1000:.2f}" if len(times) > 1 else "-",
            )

        console.print(table)

    def save_report(self, path: Path | str) -> None:
        """Save profiling report to file."""
        import json

        path = Path(path)
        summary = self.get_summary()

        # Add detailed results
        summary["detailed_results"] = [
            {
                "name": r.name,
                "elapsed_s": r.timing.elapsed_s,
                "memory_start_mb": r.memory_start.rss_mb,
                "memory_end_mb": r.memory_end.rss_mb,
                "memory_delta_mb": r.memory_delta_mb,
            }
            for r in self.results
        ]

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved profile report to {path}")


class MemoryTracker:
    """
    Track memory usage over time.

    Useful for identifying memory leaks or peak usage.
    """

    def __init__(self, sample_interval_s: float = 1.0):
        self.sample_interval_s = sample_interval_s
        self.samples: list[MemorySnapshot] = []
        self._running = False
        self._thread = None

    def start(self) -> None:
        """Start tracking."""
        import threading

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop tracking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        while self._running:
            self.samples.append(MemorySnapshot.capture())
            time.sleep(self.sample_interval_s)

    def sample_now(self) -> MemorySnapshot:
        """Take a sample immediately."""
        snapshot = MemorySnapshot.capture()
        self.samples.append(snapshot)
        return snapshot

    @property
    def peak_rss_mb(self) -> float:
        """Peak RSS memory usage."""
        if not self.samples:
            return 0.0
        return max(s.rss_mb for s in self.samples)

    @property
    def mean_rss_mb(self) -> float:
        """Mean RSS memory usage."""
        if not self.samples:
            return 0.0
        return np.mean([s.rss_mb for s in self.samples])

    def get_timeline(self) -> tuple[list[float], list[float]]:
        """Get memory usage timeline."""
        if not self.samples:
            return [], []
        t0 = self.samples[0].timestamp
        times = [s.timestamp - t0 for s in self.samples]
        values = [s.rss_mb for s in self.samples]
        return times, values


def benchmark_function(
    func: Callable,
    *args,
    n_iterations: int = 10,
    warmup: int = 2,
    **kwargs,
) -> dict[str, Any]:
    """
    Comprehensive benchmark of a function.

    Args:
        func: Function to benchmark
        *args: Positional arguments
        n_iterations: Number of benchmark iterations
        warmup: Warmup iterations
        **kwargs: Keyword arguments

    Returns:
        Benchmark results dictionary
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Memory before
    mem_before = MemorySnapshot.capture()

    # Timed iterations
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    # Memory after
    mem_after = MemorySnapshot.capture()

    return {
        "function": func.__name__,
        "n_iterations": n_iterations,
        "times_s": times,
        "mean_s": np.mean(times),
        "std_s": np.std(times),
        "min_s": min(times),
        "max_s": max(times),
        "memory_before_mb": mem_before.rss_mb,
        "memory_after_mb": mem_after.rss_mb,
        "memory_delta_mb": mem_after.rss_mb - mem_before.rss_mb,
    }


def get_system_info() -> dict[str, Any]:
    """Get system information for profiling context."""
    import platform

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        "memory_available_gb": psutil.virtual_memory().available / (1024 ** 3),
    }
