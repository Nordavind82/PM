"""
Progress reporting system for PSTM.

Provides event-based progress tracking and display.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue
from threading import Thread
from typing import Any, Callable

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from pstm.utils.logging import console as default_console


class EventType(Enum):
    """Types of progress events."""

    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    STEP_START = "step_start"
    STEP_PROGRESS = "step_progress"
    STEP_COMPLETE = "step_complete"
    TILE_START = "tile_start"
    TILE_COMPLETE = "tile_complete"
    METRICS = "metrics"
    WARNING = "warning"
    ERROR = "error"
    LOG = "log"


@dataclass
class ProgressEvent:
    """Progress event data."""

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    phase: str = ""
    step: str = ""
    tile_id: int = -1
    current: int = 0
    total: int = 0
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def progress_fraction(self) -> float:
        """Progress as fraction."""
        if self.total == 0:
            return 0.0
        return self.current / self.total

    @property
    def progress_percent(self) -> float:
        """Progress as percentage."""
        return self.progress_fraction * 100


class ProgressDispatcher:
    """
    Thread-safe event dispatcher for progress updates.

    Subscribers receive events asynchronously via a queue.
    """

    def __init__(self):
        self._subscribers: list[Callable[[ProgressEvent], None]] = []
        self._queue: Queue[ProgressEvent] = Queue()
        self._running = False
        self._thread: Thread | None = None

    def subscribe(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Add a subscriber callback."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Remove a subscriber callback."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, event: ProgressEvent) -> None:
        """Emit an event to all subscribers."""
        self._queue.put(event)

    def start(self) -> None:
        """Start the dispatcher thread."""
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._dispatch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the dispatcher thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _dispatch_loop(self) -> None:
        """Main dispatch loop."""
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                for subscriber in self._subscribers:
                    try:
                        subscriber(event)
                    except Exception:
                        pass  # Don't let subscriber errors crash dispatcher
            except Exception:
                pass  # Queue timeout


class ETACalculator:
    """
    Calculates estimated time remaining based on recent performance.

    Uses a rolling window of tile completion times for robustness.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._times: list[float] = []
        self._start_time: float = 0.0
        self._last_tile_time: float = 0.0

    def start(self) -> None:
        """Mark start of processing."""
        self._start_time = time.time()
        self._last_tile_time = self._start_time

    def record_tile(self) -> None:
        """Record completion of a tile."""
        now = time.time()
        tile_time = now - self._last_tile_time
        self._last_tile_time = now

        self._times.append(tile_time)
        if len(self._times) > self.window_size:
            self._times.pop(0)

    def get_eta(self, remaining_tiles: int) -> float:
        """
        Estimate remaining time.

        Args:
            remaining_tiles: Number of tiles remaining

        Returns:
            Estimated seconds remaining
        """
        if not self._times or remaining_tiles <= 0:
            return 0.0

        # Use harmonic mean for robustness to outliers
        inv_sum = sum(1.0 / t for t in self._times if t > 0)
        if inv_sum == 0:
            return 0.0

        avg_time = len(self._times) / inv_sum
        return avg_time * remaining_tiles

    def get_rate(self) -> float:
        """Get tiles per second."""
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


class ConsoleProgressDisplay:
    """
    Rich console progress display.

    Shows:
    - Overall progress bar
    - Current phase/step
    - Live metrics (speed, memory, etc.)
    - Recent log messages
    """

    def __init__(self, console: Console | None = None):
        self.console = console or default_console
        self._progress: Progress | None = None
        self._live: Live | None = None
        self._task_id: TaskID | None = None

        self._current_phase = ""
        self._current_step = ""
        self._metrics: dict[str, Any] = {}
        self._recent_logs: list[str] = []
        self._max_logs = 5

        self._eta = ETACalculator()

    def start(self, total_tiles: int) -> None:
        """Start the progress display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._task_id = self._progress.add_task("Migrating...", total=total_tiles)
        self._live = Live(self._make_layout(), console=self.console, refresh_per_second=4)
        self._live.start()
        self._eta.start()

    def stop(self) -> None:
        """Stop the progress display."""
        if self._live:
            self._live.stop()
            self._live = None

    def handle_event(self, event: ProgressEvent) -> None:
        """Handle a progress event."""
        if event.event_type == EventType.PHASE_START:
            self._current_phase = event.phase

        elif event.event_type == EventType.STEP_START:
            self._current_step = event.step

        elif event.event_type == EventType.TILE_COMPLETE:
            self._eta.record_tile()
            if self._progress and self._task_id is not None:
                self._progress.update(self._task_id, completed=event.current)

        elif event.event_type == EventType.METRICS:
            self._metrics.update(event.metrics)

        elif event.event_type in (EventType.LOG, EventType.WARNING, EventType.ERROR):
            self._add_log(event.message)

        # Update display
        if self._live:
            self._live.update(self._make_layout())

    def _add_log(self, message: str) -> None:
        """Add a log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._recent_logs.append(f"[dim]{timestamp}[/dim] {message}")
        if len(self._recent_logs) > self._max_logs:
            self._recent_logs.pop(0)

    def _make_layout(self) -> Panel:
        """Create the display layout."""
        table = Table.grid(expand=True)
        table.add_column()

        # Progress bar
        if self._progress:
            table.add_row(self._progress)

        # Status line
        status = f"[bold]{self._current_phase}[/bold]"
        if self._current_step:
            status += f" - {self._current_step}"
        table.add_row(status)

        # Metrics
        if self._metrics:
            metrics_str = " | ".join(
                f"[cyan]{k}[/cyan]: {v}" for k, v in self._metrics.items()
            )
            table.add_row(metrics_str)

        # Recent logs
        if self._recent_logs:
            table.add_row("")
            for log in self._recent_logs:
                table.add_row(log)

        return Panel(table, title="PSTM Progress", border_style="blue")


def create_progress_callback(display: ConsoleProgressDisplay) -> Callable:
    """Create a progress callback that updates the display."""

    def callback(phase, current, total, message):
        event = ProgressEvent(
            event_type=EventType.TILE_COMPLETE if current > 0 else EventType.PHASE_START,
            phase=str(phase.value) if hasattr(phase, "value") else str(phase),
            current=current,
            total=total,
            message=message,
        )
        display.handle_event(event)

    return callback
