#!/usr/bin/env python3
"""
Test script to verify Qt signal/slot mechanism in threads.
Non-GUI test that can run in terminal.
"""

import sys
import time
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("test")

from PyQt6.QtCore import QCoreApplication, QThread, pyqtSignal, QTimer, QObject


class Worker(QThread):
    """Worker thread that emits progress signals."""

    progress = pyqtSignal(int, int, str)
    finished_work = pyqtSignal()

    def __init__(self, n_steps=5, delay=0.5):
        super().__init__()
        self.n_steps = n_steps
        self.delay = delay

    def run(self):
        logger.info(f"Worker.run() started in thread {QThread.currentThread()}")

        for i in range(self.n_steps):
            msg = f"Step {i+1}/{self.n_steps}"
            logger.info(f"Worker: Emitting progress: {msg}")
            self.progress.emit(i + 1, self.n_steps, msg)

            logger.info(f"Worker: Sleeping {self.delay}s...")
            time.sleep(self.delay)

        logger.info("Worker: Done, emitting finished_work")
        self.finished_work.emit()


class Receiver(QObject):
    """Receives signals from worker."""

    def __init__(self):
        super().__init__()
        self.received_count = 0

    def on_progress(self, current, total, message):
        self.received_count += 1
        logger.info(f"Receiver.on_progress #{self.received_count}: {current}/{total} - {message}")

    def on_finished(self):
        logger.info("Receiver.on_finished called")
        QCoreApplication.quit()


def main():
    logger.info("=" * 60)
    logger.info("Testing Qt Signal/Slot in Threads")
    logger.info("=" * 60)

    app = QCoreApplication(sys.argv)

    logger.info(f"Main thread: {QThread.currentThread()}")

    # Create receiver
    receiver = Receiver()

    # Create worker
    worker = Worker(n_steps=5, delay=0.3)

    # Connect signals
    worker.progress.connect(receiver.on_progress)
    worker.finished_work.connect(receiver.on_finished)

    logger.info("Starting worker thread...")
    worker.start()

    # Also test a timer in main thread
    tick_count = [0]
    def on_tick():
        tick_count[0] += 1
        logger.info(f"Timer tick #{tick_count[0]}")

    timer = QTimer()
    timer.timeout.connect(on_tick)
    timer.start(200)  # Every 200ms

    logger.info("Entering event loop...")
    result = app.exec()

    logger.info(f"Event loop exited with result={result}")
    logger.info(f"Total progress signals received: {receiver.received_count}")
    logger.info(f"Total timer ticks: {tick_count[0]}")

    return result


if __name__ == "__main__":
    sys.exit(main())
