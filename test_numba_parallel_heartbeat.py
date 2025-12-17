#!/usr/bin/env python3
"""
Test Numba parallel kernel behavior with Qt heartbeat.

This simulates the actual PSTM migration kernel structure to verify
if Numba's parallel=True releases the GIL properly.
"""

import sys
import time
import numpy as np
from numba import njit, prange

# Qt imports
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import QThread, pyqtSignal, QTimer


# ============================================================================
# Numba kernel that mimics migration structure
# ============================================================================

@njit(parallel=True, cache=False, fastmath=True)
def migrate_kernel_test(
    nx: int,
    ny: int,
    nt: int,
    n_traces: int,
    output: np.ndarray,
) -> float:
    """
    Test kernel that mimics migration loop structure.

    This has the same loop structure as the real migration kernel:
    - Parallel over output X (prange)
    - Loop over output Y
    - Loop over traces
    - Loop over output times
    """
    total = 0.0

    # Parallel outer loop (like migration)
    for ix in prange(nx):
        for iy in range(ny):
            local_sum = 0.0

            # Loop over traces (like migration)
            for it in range(n_traces):
                # Loop over output times (like migration)
                for iot in range(nt):
                    # Simulate DSR travel time computation
                    dist = np.sqrt(float(ix * ix + iy * iy + it))
                    t_travel = dist / 2000.0  # Fake velocity

                    # Simulate interpolation
                    amp = np.sin(t_travel * iot)

                    # Accumulate (like migration)
                    output[ix, iy, iot % 10] += amp * 0.001
                    local_sum += amp

            total += local_sum

    return total


# ============================================================================
# Worker thread
# ============================================================================

class NumbaWorker(QThread):
    """Worker that runs Numba parallel kernel."""

    progress = pyqtSignal(str)
    finished_work = pyqtSignal(bool, str, float)  # success, message, elapsed

    def __init__(self, nx, ny, nt, n_traces):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.n_traces = n_traces

    def run(self):
        """Execute Numba kernel."""
        print(f"\n[WORKER] Starting Numba parallel kernel", file=sys.stderr, flush=True)
        print(f"  Grid: {self.nx} x {self.ny} x {self.nt}", file=sys.stderr, flush=True)
        print(f"  Traces: {self.n_traces:,}", file=sys.stderr, flush=True)
        ops = self.nx * self.ny * self.n_traces * self.nt
        print(f"  Operations: {ops:,.0f}", file=sys.stderr, flush=True)

        self.progress.emit(f"Running {ops:,.0f} operations...")

        # Create output array
        output = np.zeros((self.nx, self.ny, 10), dtype=np.float64)

        # Warm up JIT (small run)
        print(f"[WORKER] JIT warm-up...", file=sys.stderr, flush=True)
        warmup_out = np.zeros((2, 2, 10), dtype=np.float64)
        migrate_kernel_test(2, 2, 10, 10, warmup_out)
        print(f"[WORKER] JIT warm-up complete", file=sys.stderr, flush=True)

        # Main run
        print(f"[WORKER] === KERNEL START ===", file=sys.stderr, flush=True)
        start = time.time()
        result = migrate_kernel_test(self.nx, self.ny, self.nt, self.n_traces, output)
        elapsed = time.time() - start
        print(f"[WORKER] === KERNEL COMPLETE in {elapsed:.1f}s ===", file=sys.stderr, flush=True)

        self.finished_work.emit(True, f"Kernel done: result={result:.2e}", elapsed)


# ============================================================================
# Test dialog
# ============================================================================

class TestDialog(QDialog):
    """Test dialog with heartbeat timer."""

    def __init__(self, nx, ny, nt, n_traces):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.n_traces = n_traces

        self.worker = None
        self.heartbeat_count = 0
        self.start_time = 0
        self.heartbeat_times = []  # Record when heartbeats fire

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Numba Parallel + Qt Heartbeat Test")
        self.setModal(True)
        self.setMinimumSize(500, 250)

        layout = QVBoxLayout(self)

        info = QLabel(f"Grid: {self.nx}x{self.ny}x{self.nt}, Traces: {self.n_traces:,}")
        layout.addWidget(info)

        ops = self.nx * self.ny * self.n_traces * self.nt
        ops_label = QLabel(f"Operations: {ops:,.0f}")
        layout.addWidget(ops_label)

        self.elapsed_label = QLabel("Elapsed: 0s")
        layout.addWidget(self.elapsed_label)

        self.heartbeat_label = QLabel("Heartbeat #0")
        layout.addWidget(self.heartbeat_label)

        self.status_label = QLabel("Starting...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self.progress_bar)

    def start_test(self):
        """Start the test."""
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"STARTING NUMBA PARALLEL HEARTBEAT TEST", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)

        self.start_time = time.time()
        self.heartbeat_times = []

        # Start heartbeat timer
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self._on_heartbeat)
        self.heartbeat_timer.start(500)  # Every 500ms

        # Create and start worker
        self.worker = NumbaWorker(self.nx, self.ny, self.nt, self.n_traces)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_work.connect(self._on_finished)
        self.worker.start()

        print(f"[DIALOG] Worker and heartbeat timer started", file=sys.stderr, flush=True)

    def _on_heartbeat(self):
        """Called every 500ms."""
        self.heartbeat_count += 1
        elapsed = time.time() - self.start_time
        self.heartbeat_times.append(elapsed)

        # Update UI
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
        self.heartbeat_label.setText(f"Heartbeat #{self.heartbeat_count}")

        # Print every 2 seconds (4 ticks at 500ms)
        if self.heartbeat_count % 4 == 0:
            print(f"[HEARTBEAT] #{self.heartbeat_count} at {elapsed:.1f}s", file=sys.stderr, flush=True)

        QApplication.processEvents()

    def _on_progress(self, message):
        """Handle progress from worker."""
        self.status_label.setText(message)

    def _on_finished(self, success, message, kernel_elapsed):
        """Handle completion."""
        print(f"\n[DIALOG] Worker finished: {message}", file=sys.stderr, flush=True)
        self.heartbeat_timer.stop()

        total_elapsed = time.time() - self.start_time
        self.status_label.setText(f"DONE: {message}")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

        # Analyze heartbeat timing
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"HEARTBEAT ANALYSIS", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        print(f"Total heartbeats: {self.heartbeat_count}", file=sys.stderr, flush=True)
        print(f"Kernel time: {kernel_elapsed:.1f}s", file=sys.stderr, flush=True)
        print(f"Total time: {total_elapsed:.1f}s", file=sys.stderr, flush=True)

        expected_heartbeats = int(total_elapsed * 2)  # 2 per second at 500ms
        print(f"Expected heartbeats: ~{expected_heartbeats}", file=sys.stderr, flush=True)

        if len(self.heartbeat_times) >= 2:
            gaps = [self.heartbeat_times[i] - self.heartbeat_times[i-1]
                    for i in range(1, len(self.heartbeat_times))]
            max_gap = max(gaps)
            avg_gap = sum(gaps) / len(gaps)
            print(f"Heartbeat gaps: avg={avg_gap:.3f}s, max={max_gap:.3f}s", file=sys.stderr, flush=True)

            if max_gap > 2.0:
                print(f"WARNING: Max gap > 2s indicates GIL blocking!", file=sys.stderr, flush=True)
            else:
                print(f"OK: Heartbeats ran consistently during kernel", file=sys.stderr, flush=True)

        print(f"{'='*60}\n", file=sys.stderr, flush=True)

        # Close after 3 seconds
        QTimer.singleShot(3000, self.accept)


def main():
    print("Numba Parallel Kernel + Qt Heartbeat Test")
    print("=" * 50)
    print()
    print("This test runs a Numba parallel kernel similar to PSTM migration")
    print("while monitoring if Qt heartbeat timer fires correctly.")
    print()

    # Parameters that produce ~30 second runtime
    # Adjust based on your CPU
    nx = 100  # Output X points
    ny = 100  # Output Y points
    nt = 100  # Output times (inner loop)
    n_traces = 5000  # Number of traces

    ops = nx * ny * nt * n_traces
    print(f"Test parameters:")
    print(f"  Grid: {nx} x {ny} x {nt}")
    print(f"  Traces: {n_traces:,}")
    print(f"  Operations: {ops:,.0f}")
    print()

    # Allow command line override
    if len(sys.argv) > 1:
        n_traces = int(sys.argv[1])

    app = QApplication(sys.argv)

    dialog = TestDialog(nx, ny, nt, n_traces)
    dialog.start_test()
    dialog.exec()

    # Final analysis
    if dialog.heartbeat_count >= (dialog.heartbeat_times[-1] if dialog.heartbeat_times else 1) * 1.5:
        print("TEST PASSED: Heartbeat ran during Numba parallel execution")
        return 0
    else:
        print("TEST FAILED: Heartbeat was blocked during Numba execution")
        return 1


if __name__ == "__main__":
    sys.exit(main())
