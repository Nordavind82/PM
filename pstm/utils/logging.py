"""
Logging configuration for PSTM.

Provides Rich-based console logging and optional file logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for PSTM
PSTM_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "critical": "red bold reverse",
        "success": "green",
        "progress": "blue",
        "metric": "magenta",
    }
)

# Global console instance
console = Console(theme=PSTM_THEME)


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_file: Path | str | None = None,
    rich_tracebacks: bool = True,
    show_time: bool = True,
    show_path: bool = False,
) -> logging.Logger:
    """
    Set up logging with Rich console handler and optional file handler.

    Args:
        level: Logging level
        log_file: Optional path to log file
        rich_tracebacks: Use Rich for traceback formatting
        show_time: Show timestamp in console output
        show_path: Show file path in console output

    Returns:
        Configured root logger for pstm
    """
    # Get the pstm logger
    logger = logging.getLogger("pstm")
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    logger.handlers.clear()

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=True,
        markup=True,
    )
    rich_handler.setLevel(getattr(logging, level))
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    # File handler (if specified)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"pstm.{name}")


class LogCapture:
    """
    Context manager to capture log messages.

    Useful for testing or collecting messages for display.
    """

    def __init__(self, logger_name: str = "pstm", level: int = logging.DEBUG):
        self.logger_name = logger_name
        self.level = level
        self.messages: list[logging.LogRecord] = []
        self._handler: logging.Handler | None = None

    def __enter__(self) -> "LogCapture":
        """Start capturing log messages."""

        class CaptureHandler(logging.Handler):
            def __init__(handler_self, capture: LogCapture):
                super().__init__()
                handler_self.capture = capture

            def emit(handler_self, record: logging.LogRecord) -> None:
                handler_self.capture.messages.append(record)

        logger = logging.getLogger(self.logger_name)
        self._handler = CaptureHandler(self)
        self._handler.setLevel(self.level)
        logger.addHandler(self._handler)
        return self

    def __exit__(self, *args) -> None:
        """Stop capturing log messages."""
        if self._handler:
            logger = logging.getLogger(self.logger_name)
            logger.removeHandler(self._handler)

    def get_messages(self, level: int | None = None) -> list[str]:
        """
        Get captured messages, optionally filtered by level.

        Args:
            level: Filter to this level (None = all)

        Returns:
            List of message strings
        """
        records = self.messages
        if level is not None:
            records = [r for r in records if r.levelno >= level]
        return [r.getMessage() for r in records]


def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    """
    Log an exception with context and rich formatting.

    Args:
        logger: Logger to use
        exc: Exception to log
        context: Additional context string
    """
    if context:
        logger.error(f"[error]{context}[/error]: {type(exc).__name__}: {exc}")
    else:
        logger.error(f"[error]{type(exc).__name__}[/error]: {exc}")


def print_banner() -> None:
    """Print the PSTM banner to console."""
    banner = """
[bold blue]╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██████╗ ███████╗████████╗███╗   ███╗                                    ║
║   ██╔══██╗██╔════╝╚══██╔══╝████╗ ████║                                    ║
║   ██████╔╝███████╗   ██║   ██╔████╔██║                                    ║
║   ██╔═══╝ ╚════██║   ██║   ██║╚██╔╝██║                                    ║
║   ██║     ███████║   ██║   ██║ ╚═╝ ██║                                    ║
║   ╚═╝     ╚══════╝   ╚═╝   ╚═╝     ╚═╝                                    ║
║                                                                           ║
║   [cyan]3D Prestack Kirchhoff Time Migration[/cyan]                                   ║
║   [dim]Optimized for Apple Silicon[/dim]                                            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝[/bold blue]
"""
    console.print(banner)


def print_section(title: str) -> None:
    """Print a section header."""
    console.print(f"\n[bold cyan]{'─' * 60}[/bold cyan]")
    console.print(f"[bold cyan]  {title}[/bold cyan]")
    console.print(f"[bold cyan]{'─' * 60}[/bold cyan]\n")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[success]✓[/success] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[warning]⚠[/warning] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[error]✗[/error] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[info]ℹ[/info] {message}")


def print_metric(name: str, value: str | float | int, unit: str = "") -> None:
    """Print a metric value."""
    if isinstance(value, float):
        value_str = f"{value:.2f}"
    else:
        value_str = str(value)

    if unit:
        console.print(f"  [dim]{name}:[/dim] [metric]{value_str}[/metric] {unit}")
    else:
        console.print(f"  [dim]{name}:[/dim] [metric]{value_str}[/metric]")
