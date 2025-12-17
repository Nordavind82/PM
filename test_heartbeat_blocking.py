#!/usr/bin/env python3
"""
Diagnostic test to verify Qt heartbeat timer during blocking operations.

This simulates the exact architecture:
1. Modal dialog with heartbeat timer (main thread)
2. QThread worker doing blocking work (worker thread)
3. Verify heartbeat continues during blocking

If heartbeat prints show every 2s, Qt architecture is fine.
If they stop during "kernel simulation", there's a GIL/blocking issue.
"""

import sys
import time
import logging

# Minimal logging for clarity
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("test")

from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import QThread, pyqtSignal, QTimer


class BlockingWorker(QThread):
    """Worker that simulates blocking kernel execution."""

    progress = pyqtSignal(str)  # message
    finished_work = pyqtSignal(bool, str)  # success, message

    def __init__(self, duration_sec=30, blocking_type="python_sleep"):
        super().__init__()
        self.duration_sec = duration_sec
        self.blocking_type = blocking_type

    def run(self):
        """Execute blocking work."""
        print(f"\n[WORKER] Starting {self.blocking_type} for {self.duration_sec}s", file=sys.stderr, flush=True)

        if self.blocking_type == "python_sleep":
            # Simple Python sleep - should NOT block main thread
            self.progress.emit("Starting Python sleep...")
            for i in range(self.duration_sec):
                time.sleep(1)
                self.progress.emit(f"Sleep {i+1}/{self.duration_sec}s")

        elif self.blocking_type == "numpy_compute":
            # NumPy compute - may partially release GIL
            import numpy as np
            self.progress.emit("Starting NumPy compute...")
            # Create large arrays and do computation
            size = 10000
            for i in range(self.duration_sec):
                a = np.random.randn(size, size // 10)
                b = np.random.randn(size // 10, size)
                c = np.dot(a, b)  # Large matrix multiplication
                del a, b, c
                self.progress.emit(f"NumPy iteration {i+1}/{self.duration_sec}")

        elif self.blocking_type == "numba_simulate":
            # Simulate Numba-like blocking behavior
            import numpy as np
            try:
                from numba import njit

                @njit
                def heavy_loop(n):
                    """CPU-intensive JIT-compiled loop."""
                    result = 0.0
                    for i in range(n):
                        for j in range(1000):
                            result += (i * j) % 1000
                    return result

                self.progress.emit("Starting Numba JIT compile + execute...")
                print(f"[WORKER] Numba: JIT compiling...", file=sys.stderr, flush=True)

                # This triggers JIT compilation (holds GIL)
                start = time.time()
                result = heavy_loop(100)  # Small warm-up
                print(f"[WORKER] JIT compile done in {time.time()-start:.1f}s", file=sys.stderr, flush=True)

                # This runs compiled code (releases GIL)
                print(f"[WORKER] Numba: Executing heavy loop...", file=sys.stderr, flush=True)
                start = time.time()
                result = heavy_loop(self.duration_sec * 100000)  # ~30s of work
                print(f"[WORKER] Heavy loop done in {time.time()-start:.1f}s, result={result}", file=sys.stderr, flush=True)

            except ImportError:
                self.progress.emit("Numba not available, using fallback")
                time.sleep(self.duration_sec)

        elif self.blocking_type == "pure_python":
            # Pure Python - HOLDS GIL the entire time
            self.progress.emit("Starting pure Python CPU loop...")
            print(f"[WORKER] Pure Python: This WILL block main thread!", file=sys.stderr, flush=True)
            start = time.time()
            result = 0
            target_end = start + self.duration_sec
            while time.time() < target_end:
                for _ in range(100000):
                    result += 1
            print(f"[WORKER] Pure Python done, result={result}", file=sys.stderr, flush=True)

        print(f"\n[WORKER] Work complete!", file=sys.stderr, flush=True)
        self.finished_work.emit(True, "Work completed!")


class DiagnosticDialog(QDialog):
    """Diagnostic dialog that mimics MigrationProgressDialog."""

    def __init__(self, blocking_type="python_sleep", duration=30):
        super().__init__()
        self.blocking_type = blocking_type
        self.duration = duration
        self.worker = None
        self.heartbeat_count = 0
        self.start_time = 0

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle(f"Heartbeat Test - {self.blocking_type}")
        self.setModal(True)
        self.setMinimumSize(400, 200)

        layout = QVBoxLayout(self)

        self.elapsed_label = QLabel("Elapsed: 0s")
        layout.addWidget(self.elapsed_label)

        self.heartbeat_label = QLabel("Heartbeat #0")
        layout.addWidget(self.heartbeat_label)

        self.status_label = QLabel("Starting...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.duration)
        layout.addWidget(self.progress_bar)

    def start_test(self):
        """Start the blocking test."""
        self.start_time = time.time()

        # Start heartbeat timer - runs in main thread
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self._on_heartbeat)
        self.heartbeat_timer.start(500)  # Every 500ms

        # Create and start worker
        self.worker = BlockingWorker(self.duration, self.blocking_type)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_work.connect(self._on_finished)
        self.worker.start()

        print(f"\n[DIALOG] Started worker and heartbeat timer", file=sys.stderr, flush=True)

    def _on_heartbeat(self):
        """Called every 500ms by timer."""
        self.heartbeat_count += 1
        elapsed = time.time() - self.start_time

        # Always update UI
        self.elapsed_label.setText(f"Elapsed: {int(elapsed)}s")
        self.heartbeat_label.setText(f"Heartbeat #{self.heartbeat_count}")
        self.progress_bar.setValue(min(int(elapsed), self.duration))

        # Print to stderr every 2 seconds (4 ticks)
        if self.heartbeat_count % 4 == 0:
            print(f"[HEARTBEAT] #{self.heartbeat_count} at {int(elapsed)}s", file=sys.stderr, flush=True)

        # Force repaint
        QApplication.processEvents()

    def _on_progress(self, message):
        """Handle progress from worker."""
        print(f"[PROGRESS] {message}", file=sys.stderr, flush=True)
        self.status_label.setText(message)

    def _on_finished(self, success, message):
        """Handle completion."""
        print(f"\n[DIALOG] Worker finished: {message}", file=sys.stderr, flush=True)
        self.heartbeat_timer.stop()
        self.status_label.setText(f"DONE: {message}")

        # Close after 2 seconds
        QTimer.singleShot(2000, self.accept)


def main():
    print("=" * 60)
    print("Qt Heartbeat Blocking Diagnostic")
    print("=" * 60)
    print()
    print("This test verifies if the Qt event loop processes")
    print("timer events during different blocking operations.")
    print()
    print("If heartbeat messages appear every 2s, the architecture is OK.")
    print("If they stop, there's a GIL/threading issue.")
    print()
    print("Available blocking types:")
    print("  1. python_sleep  - Should NOT block (sleeps release GIL)")
    print("  2. numpy_compute - Partial GIL release during NumPy ops")
    print("  3. numba_simulate - Tests Numba JIT behavior")
    print("  4. pure_python   - WILL block (holds GIL entire time)")
    print()

    # Default to Numba simulation since that's what we're debugging
    blocking_type = "numba_simulate"
    duration = 30

    # Allow command line override
    if len(sys.argv) > 1:
        blocking_type = sys.argv[1]
    if len(sys.argv) > 2:
        duration = int(sys.argv[2])

    print(f"Running test: {blocking_type} for {duration}s")
    print("=" * 60)
    print()

    app = QApplication(sys.argv)

    dialog = DiagnosticDialog(blocking_type, duration)
    dialog.start_test()

    result = dialog.exec()

    print()
    print("=" * 60)
    print(f"Test complete. Heartbeat count: {dialog.heartbeat_count}")
    expected = duration * 2  # 2 heartbeats per second at 500ms interval
    print(f"Expected ~{expected} heartbeats for {duration}s test")
    if dialog.heartbeat_count >= expected * 0.9:
        print("PASS: Heartbeat timer worked correctly!")
    else:
        print("FAIL: Heartbeat timer was blocked!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    sys.exit(main())
