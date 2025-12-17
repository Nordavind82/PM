#!/usr/bin/env python3
"""
Test kernel performance and UI responsiveness at different data scales.

Tests both Numba and MLX kernels (if available) with varying trace counts
to identify data-size dependent issues.
"""

import sys
import time
import numpy as np
from dataclasses import dataclass

# Test configurations
TRACE_COUNTS = [1000, 5000, 10000, 20000, 50000, 100000, 300000]
OUTPUT_GRID_SIZE = (50, 50, 100)  # (nx, ny, nt) - moderate size for testing


@dataclass
class TestResult:
    n_traces: int
    backend: str
    kernel_time: float
    heartbeat_count: int
    expected_heartbeats: int
    max_gap: float
    passed: bool
    error: str = ""


def create_synthetic_data(n_traces: int, n_samples: int = 1000):
    """Create synthetic trace data for testing."""
    rng = np.random.default_rng(42)

    # Random geometry in 10km x 10km area
    return {
        'amplitudes': rng.standard_normal((n_traces, n_samples)).astype(np.float32),
        'source_x': rng.uniform(0, 10000, n_traces).astype(np.float64),
        'source_y': rng.uniform(0, 10000, n_traces).astype(np.float64),
        'receiver_x': rng.uniform(0, 10000, n_traces).astype(np.float64),
        'receiver_y': rng.uniform(0, 10000, n_traces).astype(np.float64),
        'midpoint_x': rng.uniform(2000, 8000, n_traces).astype(np.float64),
        'midpoint_y': rng.uniform(2000, 8000, n_traces).astype(np.float64),
        'sample_rate_ms': 2.0,
        'start_time_ms': 0.0,
    }


def test_with_heartbeat(backend: str, n_traces: int, timeout_sec: int = 300) -> TestResult:
    """
    Test kernel execution with heartbeat monitoring.

    Returns TestResult with timing and heartbeat info.
    """
    from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel
    from PyQt6.QtCore import QThread, pyqtSignal, QTimer

    # Import kernel components
    from pstm.kernels.factory import create_kernel
    from pstm.kernels.base import (
        KernelConfig, TraceBlock, OutputTile, VelocitySlice, create_trace_block
    )

    nx, ny, nt = OUTPUT_GRID_SIZE

    class KernelWorker(QThread):
        finished = pyqtSignal(float, str)  # elapsed, error

        def __init__(self, backend, n_traces):
            super().__init__()
            self.backend = backend
            self.n_traces = n_traces
            self.elapsed = 0
            self.error = ""

        def run(self):
            try:
                print(f"    [WORKER] Creating {self.backend} kernel...", file=sys.stderr, flush=True)
                kernel = create_kernel(self.backend)

                # Config
                config = KernelConfig(
                    max_aperture_m=3000.0,
                    min_aperture_m=100.0,
                    max_dip_degrees=45.0,
                    taper_fraction=0.1,
                    apply_spreading=True,
                    apply_obliquity=True,
                )
                kernel.initialize(config)

                # Create data
                print(f"    [WORKER] Creating {self.n_traces:,} synthetic traces...", file=sys.stderr, flush=True)
                data = create_synthetic_data(self.n_traces)

                traces = create_trace_block(
                    amplitudes=data['amplitudes'],
                    source_x=data['source_x'],
                    source_y=data['source_y'],
                    receiver_x=data['receiver_x'],
                    receiver_y=data['receiver_y'],
                    sample_rate_ms=data['sample_rate_ms'],
                    start_time_ms=data['start_time_ms'],
                )

                # Output tile centered in survey
                output = OutputTile(
                    image=np.zeros((nx, ny, nt), dtype=np.float64),
                    fold=np.zeros((nx, ny), dtype=np.int32),
                    x_axis=np.linspace(3000, 7000, nx),
                    y_axis=np.linspace(3000, 7000, ny),
                    t_axis_ms=np.linspace(0, 4000, nt),
                )

                # Velocity
                velocity = VelocitySlice(
                    vrms=np.linspace(1500, 3500, nt),
                    is_1d=True,
                    t_axis_ms=output.t_axis_ms,
                )

                # Run kernel
                print(f"    [WORKER] Running kernel...", file=sys.stderr, flush=True)
                start = time.time()
                metrics = kernel.migrate_tile(traces, output, velocity, config)
                self.elapsed = time.time() - start

                kernel.cleanup()
                print(f"    [WORKER] Kernel done: {self.elapsed:.2f}s", file=sys.stderr, flush=True)

                self.finished.emit(self.elapsed, "")

            except Exception as e:
                import traceback
                self.error = str(e)
                print(f"    [WORKER] ERROR: {e}", file=sys.stderr, flush=True)
                traceback.print_exc()
                self.finished.emit(0, str(e))

    # Setup Qt app if needed
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # Create test dialog
    class TestDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Test {backend} - {n_traces:,} traces")
            self.setMinimumSize(300, 80)
            layout = QVBoxLayout(self)
            self.label = QLabel("Starting...")
            layout.addWidget(self.label)

            self.heartbeat_count = 0
            self.heartbeat_times = []
            self.start_time = 0
            self.result_elapsed = 0
            self.result_error = ""

        def start_test(self):
            self.start_time = time.time()

            # Heartbeat timer
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._on_heartbeat)
            self.timer.start(500)

            # Worker
            self.worker = KernelWorker(backend, n_traces)
            self.worker.finished.connect(self._on_done)
            self.worker.start()

        def _on_heartbeat(self):
            self.heartbeat_count += 1
            elapsed = time.time() - self.start_time
            self.heartbeat_times.append(elapsed)
            self.label.setText(f"HB #{self.heartbeat_count} @ {elapsed:.1f}s")

            # Log every 2 seconds
            if self.heartbeat_count % 4 == 0:
                print(f"    [HB] #{self.heartbeat_count} @ {elapsed:.1f}s", file=sys.stderr, flush=True)

            QApplication.processEvents()

        def _on_done(self, elapsed, error):
            self.timer.stop()
            self.result_elapsed = elapsed
            self.result_error = error
            self.accept()

    dialog = TestDialog()
    dialog.start_test()

    # Run with timeout
    from PyQt6.QtCore import QTimer as QT
    timeout_timer = QT()
    timeout_timer.setSingleShot(True)
    timeout_timer.timeout.connect(dialog.reject)
    timeout_timer.start(timeout_sec * 1000)

    result_code = dialog.exec()
    timeout_timer.stop()

    # Calculate results
    if result_code == 0:  # Rejected (timeout)
        return TestResult(
            n_traces=n_traces,
            backend=backend,
            kernel_time=0,
            heartbeat_count=dialog.heartbeat_count,
            expected_heartbeats=0,
            max_gap=0,
            passed=False,
            error="TIMEOUT"
        )

    elapsed = dialog.result_elapsed
    expected_hb = int(elapsed * 2) if elapsed > 0 else 0

    max_gap = 0
    if len(dialog.heartbeat_times) >= 2:
        gaps = [dialog.heartbeat_times[i] - dialog.heartbeat_times[i-1]
                for i in range(1, len(dialog.heartbeat_times))]
        max_gap = max(gaps) if gaps else 0

    # Pass if heartbeats were reasonably consistent (max gap < 2s)
    passed = max_gap < 2.0 and not dialog.result_error

    return TestResult(
        n_traces=n_traces,
        backend=backend,
        kernel_time=elapsed,
        heartbeat_count=dialog.heartbeat_count,
        expected_heartbeats=expected_hb,
        max_gap=max_gap,
        passed=passed,
        error=dialog.result_error
    )


def run_scaling_tests():
    """Run tests at different scales for available backends."""
    print("=" * 70)
    print("PSTM KERNEL SCALING TEST")
    print("=" * 70)
    print(f"Output grid: {OUTPUT_GRID_SIZE[0]}x{OUTPUT_GRID_SIZE[1]}x{OUTPUT_GRID_SIZE[2]}")
    print(f"Trace counts to test: {TRACE_COUNTS}")
    print()

    # Check available backends
    from pstm.kernels.factory import get_available_backends
    backends = [b.value for b in get_available_backends()]
    print(f"Available backends: {backends}")
    print()

    # Prefer testing order: numba_cpu first, then mlx if available
    test_backends = []
    if 'numba_cpu' in backends:
        test_backends.append('numba_cpu')
    if 'mlx_metal' in backends:
        test_backends.append('mlx_metal')
    if 'numpy' in backends and not test_backends:
        test_backends.append('numpy')

    print(f"Testing backends: {test_backends}")
    print()

    results = []

    for backend in test_backends:
        print(f"\n{'='*70}")
        print(f"BACKEND: {backend.upper()}")
        print(f"{'='*70}")

        for n_traces in TRACE_COUNTS:
            print(f"\n  Testing {n_traces:,} traces...")

            try:
                result = test_with_heartbeat(backend, n_traces, timeout_sec=180)
                results.append(result)

                status = "PASS" if result.passed else "FAIL"
                print(f"    Result: {status}")
                print(f"    Kernel time: {result.kernel_time:.2f}s")
                print(f"    Heartbeats: {result.heartbeat_count} (expected ~{result.expected_heartbeats})")
                print(f"    Max HB gap: {result.max_gap:.3f}s")
                if result.error:
                    print(f"    Error: {result.error}")

            except Exception as e:
                print(f"    EXCEPTION: {e}")
                results.append(TestResult(
                    n_traces=n_traces,
                    backend=backend,
                    kernel_time=0,
                    heartbeat_count=0,
                    expected_heartbeats=0,
                    max_gap=0,
                    passed=False,
                    error=str(e)
                ))

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Backend':<12} {'Traces':>10} {'Time':>8} {'HBs':>6} {'MaxGap':>8} {'Status':<8}")
    print("-" * 60)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        time_str = f"{r.kernel_time:.2f}s" if r.kernel_time > 0 else "N/A"
        gap_str = f"{r.max_gap:.3f}s" if r.max_gap > 0 else "N/A"
        print(f"{r.backend:<12} {r.n_traces:>10,} {time_str:>8} {r.heartbeat_count:>6} {gap_str:>8} {status:<8}")

    print()

    # Identify breakpoint
    for backend in test_backends:
        backend_results = [r for r in results if r.backend == backend]
        failed = [r for r in backend_results if not r.passed]
        if failed:
            first_fail = min(failed, key=lambda x: x.n_traces)
            print(f"{backend}: First failure at {first_fail.n_traces:,} traces (gap={first_fail.max_gap:.3f}s)")
        else:
            print(f"{backend}: All tests passed!")

    print()
    return results


if __name__ == "__main__":
    # Allow filtering trace counts from command line
    if len(sys.argv) > 1:
        TRACE_COUNTS = [int(x) for x in sys.argv[1].split(",")]

    results = run_scaling_tests()

    # Exit with error if any failed
    if any(not r.passed for r in results):
        sys.exit(1)
    sys.exit(0)
