"""Tests for profiling utilities."""

import time
import pytest

from pstm.utils.profiling import (
    Timer,
    Profiler,
    MemorySnapshot,
    MemoryTracker,
    TimingResult,
    ProfileResult,
    benchmark_function,
    get_system_info,
)


class TestTimer:
    def test_basic_timer(self):
        timer = Timer()
        timer.start()
        time.sleep(0.1)
        timer.stop()
        assert timer.elapsed_s >= 0.1

    def test_timer_ms(self):
        timer = Timer()
        timer.start()
        time.sleep(0.05)
        timer.stop()
        assert timer.elapsed_ms >= 50

    def test_context_manager(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed_s >= 0.05

    def test_timer_with_name(self):
        timer = Timer("test_timer")
        assert timer.name == "test_timer"


class TestTimingResult:
    def test_timing_result(self):
        result = TimingResult(
            name="test",
            elapsed_s=1.0,
            n_iterations=10,
        )
        assert result.per_iteration_s == 0.1
        assert result.per_iteration_ms == 100


class TestMemorySnapshot:
    def test_capture_snapshot(self):
        snapshot = MemorySnapshot.capture()
        assert snapshot.rss_mb > 0
        assert snapshot.vms_mb > 0
        assert 0 <= snapshot.percent <= 100


class TestProfiler:
    def test_create_profiler(self):
        profiler = Profiler("test")
        assert profiler.name == "test"

    def test_profile_block(self):
        profiler = Profiler()
        with profiler.profile("test_section"):
            time.sleep(0.05)

        assert len(profiler.results) == 1
        assert profiler.results[0].timing.elapsed_s >= 0.05

    def test_time_function(self):
        profiler = Profiler()

        def simple_func():
            return sum(range(1000))

        result = profiler.time_function(simple_func, n_iterations=10)
        assert result.n_iterations == 10
        assert result.elapsed_s > 0

    def test_get_summary(self):
        profiler = Profiler("test")
        with profiler.profile("section1"):
            time.sleep(0.01)
        with profiler.profile("section1"):
            time.sleep(0.01)

        summary = profiler.get_summary()
        assert summary["n_sections"] == 2
        assert "section1" in summary["sections"]


class TestMemoryTracker:
    def test_sample_now(self):
        tracker = MemoryTracker()
        snapshot = tracker.sample_now()
        assert snapshot.rss_mb > 0

    def test_peak_memory(self):
        tracker = MemoryTracker()
        tracker.sample_now()
        tracker.sample_now()
        assert tracker.peak_rss_mb > 0

    def test_timeline(self):
        tracker = MemoryTracker()
        tracker.sample_now()
        tracker.sample_now()
        times, values = tracker.get_timeline()
        assert len(times) == 2
        assert len(values) == 2


class TestBenchmarkFunction:
    def test_benchmark(self):
        def add(a, b):
            return a + b

        result = benchmark_function(add, 1, 2, n_iterations=10)
        assert result["n_iterations"] == 10
        assert result["mean_s"] > 0
        assert result["function"] == "add"


class TestGetSystemInfo:
    def test_system_info(self):
        info = get_system_info()
        assert "platform" in info
        assert "cpu_count" in info
        assert "memory_total_gb" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
