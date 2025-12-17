"""
Migration Progress Dialog

Modal dialog that shows migration progress and blocks the main window.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGridLayout, QFrame, QMessageBox,
    QApplication,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from pstm.gui.migration_worker import MigrationWorker, MigrationProgress

if TYPE_CHECKING:
    from pstm.config.models import MigrationConfig

# Setup debug logging
logger = logging.getLogger("pstm.migration.debug")


class MigrationProgressDialog(QDialog):
    """
    Modal dialog showing migration progress.

    Blocks the main application while migration runs.
    """

    def __init__(self, config: "MigrationConfig", parent=None):
        super().__init__(parent)
        self.config = config
        self._worker: MigrationWorker | None = None
        self._is_paused = False
        self._start_time = 0.0
        self._last_progress_msg = "Initializing..."

        # Heartbeat timer
        self._heartbeat_timer: QTimer | None = None
        self._heartbeat_count = 0

        # Track last tile traces to avoid duplicate logging
        self._last_tile_traces = 0

        # Result
        self._success = False
        self._output_path = ""
        self._error_message = ""

        self._setup_ui()
        self._setup_style()

    def _setup_ui(self) -> None:
        """Setup the dialog UI."""
        self.setWindowTitle("Migration Progress")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.resize(700, 600)

        # Prevent closing with X button during migration
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Migration in Progress")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Stats grid
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setSpacing(10)

        self._stats_labels = {}
        stats = [
            ("Phase:", "phase"),
            ("Elapsed:", "elapsed"),
            ("Tiles:", "tiles"),
            ("ETA:", "eta"),
            ("Output Bins:", "output_bins"),       # Output bins completed / total
            ("Current Tile:", "current_tile"),    # Current tile info
            ("Input Traces:", "input_traces"),    # Input traces in current tile
            ("Rate:", "rate"),
        ]

        for i, (text, key) in enumerate(stats):
            row, col = divmod(i, 2)
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #888888; font-weight: bold;")
            stats_layout.addWidget(lbl, row, col * 2)

            val = QLabel("--")
            val.setStyleSheet("color: #ffffff;")
            val.setMinimumWidth(150)
            self._stats_labels[key] = val
            stats_layout.addWidget(val, row, col * 2 + 1)

        layout.addWidget(stats_frame)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setMinimumHeight(30)
        layout.addWidget(self._progress_bar)

        # Status label
        self._status_label = QLabel("Starting migration...")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #4fc3f7; font-size: 14px;")
        layout.addWidget(self._status_label)

        # Log output
        log_label = QLabel("Log Output:")
        log_label.setStyleSheet("color: #888888; margin-top: 10px;")
        layout.addWidget(log_label)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(200)
        self._log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                color: #cccccc;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self._log_text)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.setMinimumWidth(120)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        btn_layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.setMinimumWidth(120)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
            }
            QPushButton:hover {
                background-color: #ef5350;
            }
        """)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        btn_layout.addWidget(self._stop_btn)

        btn_layout.addStretch()

        self._close_btn = QPushButton("Close")
        self._close_btn.setMinimumWidth(120)
        self._close_btn.setEnabled(False)  # Enabled when done
        self._close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self._close_btn)

        layout.addLayout(btn_layout)

    def _setup_style(self) -> None:
        """Setup dialog style."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #ffffff;
            }
            QFrame {
                background-color: #363636;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:disabled {
                background-color: #3a3a3a;
                color: #666666;
            }
            QProgressBar {
                border: 2px solid #4a4a4a;
                border-radius: 5px;
                text-align: center;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 3px;
            }
        """)

    def start_migration(self) -> None:
        """Start the migration worker."""
        logger.info("=" * 60)
        logger.info("MIGRATION DIALOG: Starting migration")
        logger.info("=" * 60)

        self._log("Starting migration...")
        self._start_time = time.time()

        # Start heartbeat timer
        self._heartbeat_timer = QTimer(self)
        self._heartbeat_timer.timeout.connect(self._on_heartbeat)
        self._heartbeat_timer.start(500)  # Update every 500ms

        # Create worker
        self._worker = MigrationWorker(self.config, resume=False, parent=self)

        # Connect signals
        self._worker.progress_updated.connect(self._on_progress_updated)
        self._worker.phase_changed.connect(self._on_phase_changed)
        self._worker.finished_success.connect(self._on_success)
        self._worker.finished_error.connect(self._on_error)
        self._worker.log_message.connect(self._on_log_message)
        self._worker.finished.connect(self._on_worker_finished)

        logger.info("MIGRATION DIALOG: Worker created, starting thread...")
        self._worker.start()
        logger.info("MIGRATION DIALOG: Worker thread started")

    def _on_heartbeat(self) -> None:
        """Update elapsed time display."""
        self._heartbeat_count += 1
        elapsed = time.time() - self._start_time
        elapsed_str = self._format_time(elapsed)

        # Log every 20 ticks (~10 seconds at 500ms interval) AND print to stderr
        if self._heartbeat_count % 20 == 0:
            import sys
            msg_preview = self._last_progress_msg[:40] if len(self._last_progress_msg) > 40 else self._last_progress_msg
            logger.info(f"HEARTBEAT #{self._heartbeat_count}: elapsed={elapsed_str}, msg='{msg_preview}...'")
            print(f"[HEARTBEAT] {elapsed_str} - {msg_preview}", file=sys.stderr, flush=True)

        self._stats_labels["elapsed"].setText(elapsed_str)

        # Update status with elapsed time - make it clear processing is happening
        if "kernel" in self._last_progress_msg.lower() or "processing" in self._last_progress_msg.lower():
            # Show animated dots to indicate activity
            dots = "." * ((self._heartbeat_count % 4) + 1)
            status_text = f"{self._last_progress_msg} [{elapsed_str}]{dots}"
        else:
            status_text = f"{self._last_progress_msg} [{elapsed_str}]"

        self._status_label.setText(status_text)

        # Force UI repaint to ensure updates are visible
        self.repaint()
        QApplication.processEvents()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _log(self, message: str, level: str = "INFO") -> None:
        """Add message to log output."""
        timestamp = time.strftime("%H:%M:%S")
        self._log_text.append(f"[{timestamp}] {level}: {message}")

        # Auto-scroll
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_progress_updated(self, progress: MigrationProgress) -> None:
        """Handle progress update from worker."""
        logger.info("=" * 50)
        logger.info("DIALOG._on_progress_updated called")
        logger.info(f"  phase={progress.phase}")
        logger.info(f"  current={progress.current}, total={progress.total}")
        logger.info(f"  traces_processed={progress.traces_processed}")
        logger.info(f"  traces_in_tile={progress.traces_in_tile}")
        logger.info(f"  traces_per_second={progress.traces_per_second}")
        logger.info(f"  tiles_per_second={progress.tiles_per_second}")
        logger.info(f"  eta_seconds={progress.eta_seconds}")
        logger.info(f"  current_tile_info='{progress.current_tile_info}'")
        logger.info(f"  message='{progress.message}'")
        logger.info("=" * 50)

        # Update progress bar
        if progress.total > 0:
            # If we're processing (current < total and have traces), show partial progress
            if progress.current < progress.total and progress.traces_in_tile > 0:
                # Show that we're working: e.g., 0/1 tiles but processing
                # Use a pulsing effect by showing current tile as partial
                percent = int(100 * progress.current / progress.total)
                # Add a small amount to show activity (won't exceed 99%)
                if percent == 0:
                    percent = 5  # Show some progress to indicate activity
                self._progress_bar.setValue(percent)
                logger.info(f"DIALOG: Progress bar set to {percent}% (processing tile)")
            else:
                percent = int(100 * progress.current / progress.total)
                self._progress_bar.setValue(percent)
                logger.info(f"DIALOG: Progress bar set to {percent}%")
            self._progress_bar.setMaximum(100)
        else:
            logger.info("DIALOG: progress.total is 0, not updating progress bar")

        # Update stats - show tile being processed vs completed
        if progress.total > 0:
            if progress.traces_in_tile > 0:
                # Currently processing a tile
                tile_text = f"{progress.current}/{progress.total} (processing)"
                self._stats_labels["tiles"].setText(tile_text)
                logger.info(f"DIALOG: Set tiles label to '{tile_text}'")
            else:
                tile_text = f"{progress.current}/{progress.total}"
                self._stats_labels["tiles"].setText(tile_text)
                logger.info(f"DIALOG: Set tiles label to '{tile_text}'")

        # Update OUTPUT BINS progress (this is what matters for completion)
        if progress.total_output_bins > 0:
            if progress.output_bins_in_tile > 0:
                # Currently processing: show "X/Y (processing Z)"
                bins_text = f"{progress.output_bins_completed:,}/{progress.total_output_bins:,} (+{progress.output_bins_in_tile:,})"
            else:
                bins_text = f"{progress.output_bins_completed:,}/{progress.total_output_bins:,}"
            self._stats_labels["output_bins"].setText(bins_text)
            logger.info(f"DIALOG: Set output_bins label to '{bins_text}'")
        else:
            logger.info("DIALOG: No output bin info available yet")

        # Update input traces display (for current tile being processed)
        if progress.traces_in_tile > 0:
            traces_text = f"{progress.traces_in_tile:,} (processing)"
            self._stats_labels["input_traces"].setText(traces_text)
            logger.info(f"DIALOG: Set input_traces label to '{traces_text}'")
            # Log to output (only once per tile)
            if not hasattr(self, '_last_tile_traces') or self._last_tile_traces != progress.traces_in_tile:
                self._log(f"Processing tile with {progress.traces_in_tile:,} input traces...")
                self._last_tile_traces = progress.traces_in_tile
            # Update status prominently
            self._last_progress_msg = f"Migrating {progress.output_bins_in_tile:,} output bins..."
            logger.info(f"DIALOG: Updated last_progress_msg to '{self._last_progress_msg}'")
        else:
            if progress.output_bins_completed > 0:
                self._stats_labels["input_traces"].setText("--")
            logger.info("DIALOG: traces_in_tile is 0")

        # Update rate - use traces_per_second or estimate from elapsed time
        if progress.traces_per_second > 0:
            if progress.traces_per_second >= 1000:
                rate_text = f"{progress.traces_per_second/1000:.1f}k/s"
            else:
                rate_text = f"{progress.traces_per_second:.0f}/s"
            self._stats_labels["rate"].setText(rate_text)
            logger.info(f"DIALOG: Set rate label to '{rate_text}'")
        elif progress.tiles_per_second > 0:
            rate_text = f"{progress.tiles_per_second:.2f} tiles/s"
            self._stats_labels["rate"].setText(rate_text)
            logger.info(f"DIALOG: Set rate label to '{rate_text}'")
        elif progress.traces_in_tile > 0 and progress.elapsed_seconds > 0:
            # Show "calculating..." when we're actively processing but no rate yet
            self._stats_labels["rate"].setText("calculating...")
            logger.info("DIALOG: Set rate label to 'calculating...' (processing)")
        else:
            logger.info("DIALOG: No rate info available")

        # Update ETA
        if progress.eta_seconds is not None:
            eta_text = self._format_time(progress.eta_seconds)
            self._stats_labels["eta"].setText(eta_text)
            logger.info(f"DIALOG: Set eta label to '{eta_text}'")
        elif progress.traces_in_tile > 0:
            # Show "calculating..." when processing
            self._stats_labels["eta"].setText("calculating...")
            logger.info("DIALOG: Set eta label to 'calculating...' (processing)")
        else:
            logger.info("DIALOG: eta_seconds is None")

        # Update current tile info - show the actual tile info, not just indices
        if progress.current_tile_info:
            # Show full tile info like "Tile (0,0) - 643,204 traces"
            self._stats_labels["current_tile"].setText(progress.current_tile_info)
            if not self._last_progress_msg.startswith("Processing"):
                self._last_progress_msg = progress.current_tile_info
            logger.info(f"DIALOG: Set current_tile to '{progress.current_tile_info}'")
        elif progress.message:
            # Fallback to message if no tile info
            self._stats_labels["current_tile"].setText(progress.message[:50])
            if not self._last_progress_msg.startswith("Processing"):
                self._last_progress_msg = progress.message
            logger.info(f"DIALOG: Set current_tile from message")

        # Force UI update - process pending events to ensure labels are repainted
        QApplication.processEvents()
        logger.info("DIALOG: Called processEvents() to force UI update")

    def _on_phase_changed(self, phase: str, description: str) -> None:
        """Handle phase change."""
        logger.info(f"DIALOG: Phase changed to {phase}: {description}")
        self._stats_labels["phase"].setText(phase.upper())
        self._last_progress_msg = description
        self._log(f"Phase: {description}")
        QApplication.processEvents()

    def _on_log_message(self, level: str, message: str) -> None:
        """Handle log message from worker."""
        self._log(message, level.upper())

    def _on_success(self, output_path: str) -> None:
        """Handle successful completion."""
        logger.info(f"DIALOG: Migration completed successfully: {output_path}")
        self._success = True
        self._output_path = output_path

        self._progress_bar.setValue(100)
        self._status_label.setText("Migration completed successfully!")
        self._status_label.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: bold;")
        self._log(f"SUCCESS: Output saved to {output_path}")

        self._finish_migration()

    def _on_error(self, error_msg: str) -> None:
        """Handle migration error."""
        logger.error(f"DIALOG: Migration failed: {error_msg}")
        self._success = False
        self._error_message = error_msg

        self._status_label.setText("Migration failed!")
        self._status_label.setStyleSheet("color: #f44336; font-size: 14px; font-weight: bold;")
        self._log(f"ERROR: {error_msg}", "ERROR")

        self._finish_migration()

    def _on_worker_finished(self) -> None:
        """Handle worker thread completion."""
        logger.info("DIALOG: Worker thread finished")
        self._finish_migration()

    def _finish_migration(self) -> None:
        """Clean up after migration finishes."""
        # Stop heartbeat
        if self._heartbeat_timer:
            self._heartbeat_timer.stop()
            self._heartbeat_timer = None

        # Update UI
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

        # Final elapsed time
        elapsed = time.time() - self._start_time
        self._stats_labels["elapsed"].setText(self._format_time(elapsed))

        self._worker = None

    def _on_pause_clicked(self) -> None:
        """Handle pause/resume button click."""
        if not self._worker:
            return

        if self._is_paused:
            self._worker.request_resume()
            self._is_paused = False
            self._pause_btn.setText("⏸ Pause")
            self._log("Resuming migration...")
        else:
            self._worker.request_pause()
            self._is_paused = True
            self._pause_btn.setText("▶ Resume")
            self._log("Pausing migration...")

    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        if not self._worker:
            return

        reply = QMessageBox.question(
            self, "Stop Migration",
            "Are you sure you want to stop the migration?\n"
            "Progress will be saved to checkpoint.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._log("Stopping migration...")
            self._stop_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            self._worker.request_stop()

    def closeEvent(self, event) -> None:
        """Handle close event."""
        if self._worker and self._worker.isRunning():
            # Don't allow closing while running
            event.ignore()
            QMessageBox.warning(
                self, "Migration Running",
                "Cannot close while migration is running.\n"
                "Use the Stop button to stop the migration first."
            )
        else:
            if self._heartbeat_timer:
                self._heartbeat_timer.stop()
            event.accept()

    @property
    def was_successful(self) -> bool:
        """Return whether migration was successful."""
        return self._success

    @property
    def output_path(self) -> str:
        """Return output path if successful."""
        return self._output_path

    @property
    def error_message(self) -> str:
        """Return error message if failed."""
        return self._error_message
