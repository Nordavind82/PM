"""
Migration Worker Thread

QThread-based worker for running migration in the background without freezing the UI.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from pstm.config.models import MigrationConfig

# Configure logging for migration worker
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class MigrationProgress:
    """Progress information for UI updates."""
    phase: str
    current: int  # Current tile number
    total: int  # Total tiles
    message: str
    elapsed_seconds: float = 0.0
    tiles_per_second: float = 0.0
    eta_seconds: float | None = None
    # Input trace info (for rate calculation)
    traces_processed: int = 0  # Input traces processed so far
    traces_in_tile: int = 0  # Input traces in current tile
    traces_per_second: float = 0.0  # Processing rate
    total_traces: int = 0  # Total input traces in dataset
    # OUTPUT progress (what matters for completion tracking)
    output_bins_completed: int = 0  # Output bins (pillars) fully migrated
    total_output_bins: int = 0  # Total output bins = nx * ny
    output_bins_in_tile: int = 0  # Output bins in current tile
    # Tile info (for display)
    current_tile_info: str = ""  # e.g., "Tile (3,5) - 1,024 bins"


class MigrationWorker(QThread):
    """
    Background worker thread for migration execution.

    Wraps MigrationExecutor and emits Qt signals for thread-safe UI updates.
    Supports pause, resume, and stop operations.

    Signals:
        progress_updated: Emitted on each progress callback (throttled)
        phase_changed: Emitted when execution phase changes
        finished_success: Emitted when migration completes successfully
        finished_error: Emitted when migration fails with error message
        log_message: Emitted for log messages
    """

    # Signals for thread-safe communication with UI
    progress_updated = pyqtSignal(object)  # MigrationProgress
    phase_changed = pyqtSignal(str, str)  # phase_name, description
    finished_success = pyqtSignal(str)  # output_path
    finished_error = pyqtSignal(str)  # error_message
    log_message = pyqtSignal(str, str)  # level, message

    # Throttle settings
    MIN_UPDATE_INTERVAL = 0.1  # Minimum seconds between UI updates (10 Hz max)

    def __init__(self, config: MigrationConfig, resume: bool = False, parent=None):
        super().__init__(parent)
        self.config = config
        self.resume = resume

        # Executor instance (created in run())
        self._executor = None

        # Control flags
        self._stop_requested = False
        self._pause_requested = False

        # Progress tracking
        self._start_time = 0.0
        self._last_update_time = 0.0
        self._completed_tiles = 0
        self._current_phase = ""

    def run(self):
        """Execute migration in background thread."""
        logger.info("=" * 60)
        logger.info("MigrationWorker.run() started")
        logger.info("=" * 60)

        try:
            from pstm.pipeline.executor import MigrationExecutor, ExecutionPhase
            logger.debug("Successfully imported MigrationExecutor")
        except Exception as e:
            error_msg = f"Failed to import MigrationExecutor: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.log_message.emit("error", error_msg)
            self.finished_error.emit(error_msg)
            return

        self._start_time = time.time()
        self._last_update_time = 0.0
        self._completed_tiles = 0

        try:
            logger.info("Initializing migration executor...")
            logger.debug(f"Config type: {type(self.config)}")
            logger.debug(f"Config: {self.config}")
            self.log_message.emit("info", "Initializing migration executor...")

            # Create executor with our progress callback
            logger.debug("Creating MigrationExecutor instance...")
            self._executor = MigrationExecutor(
                self.config,
                progress_callback=self._on_progress
            )
            logger.info("MigrationExecutor created successfully")

            # Run migration
            logger.info("Starting migration execution...")
            self.log_message.emit("info", "Starting migration...")
            success = self._executor.run(resume=self.resume)
            logger.info(f"Migration run() returned: success={success}")

            if self._stop_requested:
                logger.info("Migration stopped by user request")
                self.log_message.emit("info", "Migration stopped by user")
                self.finished_error.emit("Migration stopped by user. Checkpoint saved.")
            elif success:
                output_path = str(self.config.output.output_dir)
                logger.info(f"Migration completed successfully: {output_path}")
                self.log_message.emit("info", f"Migration completed successfully: {output_path}")
                self.finished_success.emit(output_path)
            else:
                logger.error("Migration failed (returned False)")
                self.log_message.emit("error", "Migration failed")
                self.finished_error.emit("Migration failed. Check logs for details.")

        except Exception as e:
            error_msg = f"Migration error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.log_message.emit("error", error_msg)
            self.finished_error.emit(f"{error_msg}\n\nTraceback:\n{traceback.format_exc()}")

    def _on_progress(self, info):
        """
        Progress callback from executor.

        Throttled to avoid overwhelming the UI with updates.

        Args:
            info: ProgressInfo dataclass from executor
        """
        from pstm.pipeline.executor import ExecutionPhase

        logger.debug(f"WORKER._on_progress called: phase={info.phase}, tile={info.current_tile}/{info.total_tiles}, "
                    f"traces_in_tile={info.traces_in_tile}, total_traces={info.traces_processed}")

        # Check for stop/pause requests
        if self._stop_requested and self._executor:
            self._executor._stop_requested = True
            return

        if self._pause_requested and self._executor:
            self._executor._pause_requested = True

        # Track phase changes
        phase_name = info.phase.value if hasattr(info.phase, 'value') else str(info.phase)
        if phase_name != self._current_phase:
            logger.info(f"WORKER: Phase changed from '{self._current_phase}' to '{phase_name}'")
            self._current_phase = phase_name
            self._emit_phase_change(info.phase)

        # Check if this is an important update that should bypass throttling
        # Important updates: new tile started, new trace info available
        is_important = (
            info.traces_in_tile > 0 or  # Has trace info
            info.current_tile > 0 or  # Processing a tile
            info.traces_processed > 0  # Has processed traces
        )

        # Throttle progress updates (but not important ones)
        now = time.time()
        time_since_last = now - self._last_update_time
        if time_since_last < self.MIN_UPDATE_INTERVAL and not is_important:
            logger.debug(f"WORKER: Throttled (only {time_since_last:.3f}s since last update)")
            return

        # Always update for important updates, even if recently updated
        if is_important:
            logger.info(f"WORKER: Important update - bypassing throttle (traces_in_tile={info.traces_in_tile})")

        self._last_update_time = now

        # Calculate metrics
        elapsed = now - self._start_time
        tiles_per_second = info.current_tile / elapsed if elapsed > 0 and info.current_tile > 0 else 0

        logger.info(f"WORKER: Calculating metrics:")
        logger.info(f"  elapsed={elapsed:.2f}s, tiles_per_second={tiles_per_second:.4f}")

        # Calculate ETA
        eta = None
        if tiles_per_second > 0 and info.total_tiles > info.current_tile:
            remaining = info.total_tiles - info.current_tile
            eta = remaining / tiles_per_second
            logger.info(f"  ETA calculated: remaining={remaining} tiles, eta={eta:.2f}s")
        else:
            logger.info(f"  ETA not calculated: tiles_per_second={tiles_per_second}, total={info.total_tiles}, current={info.current_tile}")

        # Calculate traces per second
        traces_per_sec = info.traces_processed / elapsed if elapsed > 0 and info.traces_processed > 0 else 0
        logger.info(f"  traces_per_sec={traces_per_sec:.2f} (traces_processed={info.traces_processed})")

        # Build tile info string - show output bins being processed
        tile_info = ""
        if info.output_bins_in_tile > 0:
            tile_info = f"Tile ({info.tile_x},{info.tile_y}) - {info.output_bins_in_tile:,} output bins"
        elif info.traces_in_tile > 0:
            tile_info = f"Tile ({info.tile_x},{info.tile_y}) - {info.traces_in_tile:,} input traces"
        logger.info(f"  tile_info='{tile_info}'")

        # Emit progress signal
        progress = MigrationProgress(
            phase=phase_name,
            current=info.current_tile,
            total=info.total_tiles,
            message=info.message,
            elapsed_seconds=elapsed,
            tiles_per_second=tiles_per_second,
            eta_seconds=eta,
            traces_processed=info.traces_processed,
            traces_in_tile=info.traces_in_tile,
            traces_per_second=traces_per_sec,
            total_traces=info.total_traces,
            output_bins_completed=info.output_bins_completed,
            total_output_bins=info.total_output_bins,
            output_bins_in_tile=info.output_bins_in_tile,
            current_tile_info=tile_info,
        )

        logger.info("WORKER: Emitting progress_updated signal with:")
        logger.info(f"  current={progress.current}, total={progress.total}")
        logger.info(f"  traces_processed={progress.traces_processed}, traces_in_tile={progress.traces_in_tile}")
        logger.info(f"  traces_per_second={progress.traces_per_second}, tiles_per_second={progress.tiles_per_second}")
        logger.info(f"  eta_seconds={progress.eta_seconds}")
        logger.info(f"  current_tile_info='{progress.current_tile_info}'")
        self.progress_updated.emit(progress)
        logger.info("WORKER: Signal emitted")

    def _emit_phase_change(self, phase):
        """Emit phase change signal with description."""
        from pstm.pipeline.executor import ExecutionPhase

        descriptions = {
            ExecutionPhase.INIT: "Initializing data readers and compute kernel...",
            ExecutionPhase.PLANNING: "Planning tile execution order...",
            ExecutionPhase.MIGRATION: "Processing tiles...",
            ExecutionPhase.FINALIZATION: "Normalizing and writing output...",
            ExecutionPhase.COMPLETE: "Migration complete!",
            ExecutionPhase.FAILED: "Migration failed",
        }

        phase_name = phase.value if hasattr(phase, 'value') else str(phase)
        description = descriptions.get(phase, phase_name)
        self.phase_changed.emit(phase_name, description)

    def request_pause(self):
        """Request migration to pause (will pause at next tile boundary)."""
        self._pause_requested = True
        if self._executor:
            self._executor._pause_requested = True
        self.log_message.emit("info", "Pause requested - will pause after current tile")

    def request_resume(self):
        """Request migration to resume from pause."""
        self._pause_requested = False
        if self._executor:
            self._executor._pause_requested = False
        self.log_message.emit("info", "Resuming migration...")

    def request_stop(self):
        """Request migration to stop (will save checkpoint)."""
        self._stop_requested = True
        if self._executor:
            self._executor._stop_requested = True
        self.log_message.emit("info", "Stop requested - saving checkpoint...")

    def is_paused(self) -> bool:
        """Check if migration is paused."""
        return self._pause_requested

    def is_stopping(self) -> bool:
        """Check if migration is stopping."""
        return self._stop_requested
