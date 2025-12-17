#!/usr/bin/env python3
"""
Test MLX kernel behavior with Qt heartbeat.

This tests that the MLX kernel's Python for loops now yield the GIL
periodically to allow Qt event loop processing.
"""

import sys
import time
import numpy as np

# Check MLX availability
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available - skipping test")
    sys.exit(0)

from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import QThread, pyqtSignal, QTimer


class MLXWorker(QThread):
    """Worker that simulates MLX kernel behavior."""

    progress = pyqtSignal(str)
    finished_work = pyqtSignal(bool, str, float)

    def __init__(self, n_traces=10000, nt=100, nx=50, ny=50):
        super().__init__()
        self.n_traces = n_traces
        self.nt = nt
        self.nx = nx
        self.ny = ny

    def run(self):
        """Simulate MLX kernel with periodic GIL release."""
        import time as _time

        print(f"\n[WORKER] Starting MLX simulation", file=sys.stderr, flush=True)
        print(f"  Traces: {self.n_traces}, Grid: {self.nx}x{self.ny}x{self.nt}", file=sys.stderr, flush=True)

        self.progress.emit(f"Processing {self.n_traces} traces...")

        # Create output arrays
        image = mx.zeros((self.nx, self.ny, self.nt), dtype=mx.float32)
        fold = mx.zeros((self.nx, self.ny), dtype=mx.int32)

        # Track GIL yield timing
        last_yield_time = _time.time()
        yield_interval = 0.1  # 100ms
        yield_count = 0

        start = _time.time()

        # Simulate the trace loop (like MLX _migrate_chunk)
        for i in range(self.n_traces):
            # Periodically release GIL
            now = _time.time()
            if now - last_yield_time > yield_interval:
                mx.eval(image, fold)  # Force evaluation
                _time.sleep(0)  # Yield GIL
                last_yield_time = now
                yield_count += 1

            # Simulate per-trace work
            contribution = mx.sin(mx.array([float(i % 100)]))
            image = image + contribution[0] * 0.00001

        mx.eval(image, fold)
        elapsed = _time.time() - start

        print(f"\n[WORKER] Complete: {elapsed:.1f}s, yielded GIL {yield_count} times", file=sys.stderr, flush=True)
        self.finished_work.emit(True, f"Done in {elapsed:.1f}s", elapsed)


class TestDialog(QDialog):
    """Test dialog with heartbeat."""

    def __init__(self, n_traces):
        super().__init__()
        self.n_traces = n_traces
        self.worker = None
        self.heartbeat_count = 0
        self.start_time = 0
        self.heartbeat_times = []

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("MLX GIL Yield Test")
        self.setModal(True)
        self.setMinimumSize(400, 200)

        layout = QVBoxLayout(self)

        info = QLabel(f"MLX kernel test with {self.n_traces:,} trace iterations")
        layout.addWidget(info)

        self.elapsed_label = QLabel("Elapsed: 0s")
        layout.addWidget(self.elapsed_label)

        self.heartbeat_label = QLabel("Heartbeat #0")
        layout.addWidget(self.heartbeat_label)

        self.status_label = QLabel("Starting...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)

    def start_test(self):
        """Start test."""
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print("MLX GIL YIELD TEST", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)

        self.start_time = time.time()
        self.heartbeat_times = []

        # Start heartbeat
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self._on_heartbeat)
        self.heartbeat_timer.start(500)

        # Start worker
        self.worker = MLXWorker(self.n_traces)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_work.connect(self._on_finished)
        self.worker.start()

        print("[DIALOG] Worker and heartbeat started", file=sys.stderr, flush=True)

    def _on_heartbeat(self):
        """Heartbeat handler."""
        self.heartbeat_count += 1
        elapsed = time.time() - self.start_time
        self.heartbeat_times.append(elapsed)

        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
        self.heartbeat_label.setText(f"Heartbeat #{self.heartbeat_count}")

        if self.heartbeat_count % 4 == 0:
            print(f"[HEARTBEAT] #{self.heartbeat_count} at {elapsed:.1f}s", file=sys.stderr, flush=True)

        QApplication.processEvents()

    def _on_progress(self, message):
        self.status_label.setText(message)

    def _on_finished(self, success, message, elapsed):
        print(f"\n[DIALOG] Worker finished", file=sys.stderr, flush=True)
        self.heartbeat_timer.stop()

        self.status_label.setText(f"DONE: {message}")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

        # Analyze
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print("RESULTS", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        print(f"Total heartbeats: {self.heartbeat_count}", file=sys.stderr, flush=True)
        print(f"Worker time: {elapsed:.1f}s", file=sys.stderr, flush=True)

        expected = int(elapsed * 2)
        print(f"Expected heartbeats: ~{expected}", file=sys.stderr, flush=True)

        if len(self.heartbeat_times) >= 2:
            gaps = [self.heartbeat_times[i] - self.heartbeat_times[i-1]
                    for i in range(1, len(self.heartbeat_times))]
            max_gap = max(gaps)
            print(f"Max heartbeat gap: {max_gap:.3f}s", file=sys.stderr, flush=True)

            if max_gap > 1.0:
                print("WARNING: Large gap detected - GIL may still be blocking", file=sys.stderr, flush=True)
            else:
                print("OK: Heartbeats ran consistently", file=sys.stderr, flush=True)

        print(f"{'='*60}\n", file=sys.stderr, flush=True)

        QTimer.singleShot(2000, self.accept)


def main():
    print("MLX Kernel GIL Yield Test")
    print("=" * 40)
    print()

    n_traces = 50000  # Enough iterations to test yielding

    if len(sys.argv) > 1:
        n_traces = int(sys.argv[1])

    print(f"Testing with {n_traces:,} trace iterations")
    print()

    app = QApplication(sys.argv)

    dialog = TestDialog(n_traces)
    dialog.start_test()
    dialog.exec()

    return 0


if __name__ == "__main__":
    sys.exit(main())
