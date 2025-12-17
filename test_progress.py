#!/usr/bin/env python3
"""
Test script to verify migration progress reporting architecture.

This creates a mock migration that simulates progress updates
without actually running the heavy kernel computation.
"""

import sys
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("test_progress")

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import time


class MockWorker(QThread):
    """Mock worker that simulates migration progress."""

    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    finished_signal = pyqtSignal(bool, str)  # success, message

    def __init__(self, n_tiles=5, delay_per_tile=2.0):
        super().__init__()
        self.n_tiles = n_tiles
        self.delay_per_tile = delay_per_tile
        self._stop = False

    def run(self):
        logger.info(f"MockWorker: Starting with {self.n_tiles} tiles")

        for i in range(self.n_tiles):
            if self._stop:
                break

            # Report starting tile
            msg = f"Processing tile {i+1}/{self.n_tiles}"
            logger.info(f"MockWorker: {msg}")
            self.progress_updated.emit(i, self.n_tiles, msg)

            # Simulate processing time
            time.sleep(self.delay_per_tile)

            # Report completed tile
            msg = f"Completed tile {i+1}/{self.n_tiles}"
            logger.info(f"MockWorker: {msg}")
            self.progress_updated.emit(i + 1, self.n_tiles, msg)

        logger.info("MockWorker: Finished")
        self.finished_signal.emit(True, "Migration complete!")

    def stop(self):
        self._stop = True


class TestWindow(QMainWindow):
    """Test window to verify progress updates."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Progress Test")
        self.setMinimumSize(400, 200)

        self.worker = None
        self.start_time = 0

        # UI
        central = QWidget()
        layout = QVBoxLayout(central)

        from PyQt6.QtWidgets import QLabel, QProgressBar

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.elapsed_label = QLabel("Elapsed: 0s")
        layout.addWidget(self.elapsed_label)

        self.start_btn = QPushButton("Start Mock Migration")
        self.start_btn.clicked.connect(self.start_migration)
        layout.addWidget(self.start_btn)

        self.setCentralWidget(central)

        # Heartbeat timer
        self.heartbeat = QTimer()
        self.heartbeat.timeout.connect(self.on_heartbeat)

    def start_migration(self):
        logger.info("TestWindow: Starting mock migration")
        self.start_time = time.time()
        self.start_btn.setEnabled(False)

        # Start heartbeat
        self.heartbeat.start(500)

        # Create and start worker
        self.worker = MockWorker(n_tiles=5, delay_per_tile=1.0)
        self.worker.progress_updated.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_heartbeat(self):
        elapsed = time.time() - self.start_time
        self.elapsed_label.setText(f"Elapsed: {int(elapsed)}s")
        logger.debug(f"Heartbeat: {int(elapsed)}s")
        QApplication.processEvents()

    def on_progress(self, current, total, message):
        logger.info(f"TestWindow.on_progress: {current}/{total} - {message}")

        if total > 0:
            percent = int(100 * current / total)
            self.progress_bar.setValue(percent)

        self.status_label.setText(message)
        QApplication.processEvents()

    def on_finished(self, success, message):
        logger.info(f"TestWindow.on_finished: success={success}, message={message}")
        self.heartbeat.stop()
        self.status_label.setText(message)
        self.progress_bar.setValue(100)
        self.start_btn.setEnabled(True)


def main():
    logger.info("=" * 50)
    logger.info("Starting Progress Test")
    logger.info("=" * 50)

    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()

    logger.info("Test window shown, entering event loop")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
