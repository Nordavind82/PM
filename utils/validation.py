"""
Edge case handling and validation for PSTM.

Provides robust error handling, input validation, and graceful degradation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pstm.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.category}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validation checks."""
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_issue(
        self,
        severity: ValidationSeverity,
        category: str,
        message: str,
        **details,
    ) -> None:
        """Add a validation issue."""
        self.issues.append(ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            details=details,
        ))
        if severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.valid = False

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in self.issues
        )

    def get_by_severity(self, severity: ValidationSeverity) -> list[ValidationIssue]:
        """Get issues by severity."""
        return [i for i in self.issues if i.severity == severity]

    def summary(self) -> str:
        """Get summary of validation results."""
        counts = {s: 0 for s in ValidationSeverity}
        for issue in self.issues:
            counts[issue.severity] += 1

        parts = []
        for sev in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR,
                    ValidationSeverity.WARNING, ValidationSeverity.INFO]:
            if counts[sev] > 0:
                parts.append(f"{counts[sev]} {sev.value}(s)")

        status = "VALID" if self.valid else "INVALID"
        return f"{status}: " + ", ".join(parts) if parts else f"{status}"


# =============================================================================
# Input Validation
# =============================================================================


def validate_file_exists(path: Path | str, file_type: str = "file") -> ValidationResult:
    """
    Validate that a file exists and is accessible.

    Args:
        path: File path to validate
        file_type: Description of file type for error messages

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)
    path = Path(path)

    if not path.exists():
        result.add_issue(
            ValidationSeverity.ERROR,
            "file_access",
            f"{file_type} not found: {path}",
            path=str(path),
        )
    elif not os.access(path, os.R_OK):
        result.add_issue(
            ValidationSeverity.ERROR,
            "file_access",
            f"{file_type} not readable: {path}",
            path=str(path),
        )

    return result


def validate_zarr_file(path: Path | str) -> ValidationResult:
    """
    Validate a Zarr file structure.

    Args:
        path: Path to Zarr array/group

    Returns:
        ValidationResult
    """
    import zarr

    result = ValidationResult(valid=True)
    path = Path(path)

    # Check existence
    if not path.exists():
        result.add_issue(
            ValidationSeverity.ERROR,
            "zarr",
            f"Zarr path not found: {path}",
        )
        return result

    try:
        z = zarr.open(str(path), mode='r')

        # Check if it's an array
        if hasattr(z, 'shape'):
            if len(z.shape) != 2:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "zarr",
                    f"Expected 2D array, got shape {z.shape}",
                )

            # Check for reasonable dimensions
            n_traces, n_samples = z.shape
            if n_traces == 0:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "zarr",
                    "Array has zero traces",
                )
            if n_samples == 0:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "zarr",
                    "Array has zero samples",
                )

            # Check dtype
            if z.dtype not in (np.float32, np.float64, np.int16, np.int32):
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "zarr",
                    f"Unusual dtype: {z.dtype}",
                )

            result.add_issue(
                ValidationSeverity.INFO,
                "zarr",
                f"Valid Zarr array: {z.shape}, {z.dtype}",
            )

    except Exception as e:
        result.add_issue(
            ValidationSeverity.ERROR,
            "zarr",
            f"Failed to open Zarr file: {e}",
        )

    return result


def validate_parquet_file(path: Path | str, required_columns: list[str] | None = None) -> ValidationResult:
    """
    Validate a Parquet file.

    Args:
        path: Path to Parquet file
        required_columns: List of required column names

    Returns:
        ValidationResult
    """
    import polars as pl

    result = ValidationResult(valid=True)
    path = Path(path)

    if not path.exists():
        result.add_issue(
            ValidationSeverity.ERROR,
            "parquet",
            f"Parquet file not found: {path}",
        )
        return result

    try:
        # Read schema only
        schema = pl.read_parquet_schema(str(path))
        columns = list(schema.keys())

        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(columns)
            if missing:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "parquet",
                    f"Missing required columns: {missing}",
                    missing_columns=list(missing),
                    available_columns=columns,
                )

        # Quick row count check
        df = pl.scan_parquet(str(path))
        n_rows = df.select(pl.count()).collect().item()

        if n_rows == 0:
            result.add_issue(
                ValidationSeverity.ERROR,
                "parquet",
                "Parquet file has zero rows",
            )
        else:
            result.add_issue(
                ValidationSeverity.INFO,
                "parquet",
                f"Valid Parquet file: {n_rows} rows, {len(columns)} columns",
            )

    except Exception as e:
        result.add_issue(
            ValidationSeverity.ERROR,
            "parquet",
            f"Failed to read Parquet file: {e}",
        )

    return result


def validate_velocity_cube(path: Path | str) -> ValidationResult:
    """
    Validate a 3D velocity cube.

    Args:
        path: Path to velocity Zarr file

    Returns:
        ValidationResult
    """
    import zarr

    result = ValidationResult(valid=True)
    path = Path(path)

    if not path.exists():
        result.add_issue(
            ValidationSeverity.ERROR,
            "velocity",
            f"Velocity cube not found: {path}",
        )
        return result

    try:
        z = zarr.open(str(path), mode='r')

        if hasattr(z, 'shape'):
            if len(z.shape) != 3:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "velocity",
                    f"Expected 3D velocity cube, got shape {z.shape}",
                )
                return result

            # Sample some values to check range
            sample = z[::max(1, z.shape[0]//10),
                       ::max(1, z.shape[1]//10),
                       ::max(1, z.shape[2]//10)]

            v_min, v_max = sample.min(), sample.max()

            if v_min < 500:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "velocity",
                    f"Very low velocity detected: {v_min:.0f} m/s",
                )

            if v_max > 10000:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "velocity",
                    f"Very high velocity detected: {v_max:.0f} m/s",
                )

            if np.any(np.isnan(sample)):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "velocity",
                    "Velocity cube contains NaN values",
                )

            result.add_issue(
                ValidationSeverity.INFO,
                "velocity",
                f"Velocity cube: {z.shape}, range [{v_min:.0f}, {v_max:.0f}] m/s",
            )

    except Exception as e:
        result.add_issue(
            ValidationSeverity.ERROR,
            "velocity",
            f"Failed to open velocity cube: {e}",
        )

    return result


# =============================================================================
# Data Quality Validation
# =============================================================================


def validate_trace_data(
    data: NDArray,
    max_nan_fraction: float = 0.01,
    max_zero_fraction: float = 0.5,
) -> ValidationResult:
    """
    Validate trace data quality.

    Args:
        data: Trace data array (n_traces, n_samples)
        max_nan_fraction: Maximum allowed NaN fraction
        max_zero_fraction: Maximum allowed zero fraction

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    n_traces, n_samples = data.shape

    # Check for NaN
    n_nan = np.sum(np.isnan(data))
    nan_fraction = n_nan / data.size

    if nan_fraction > max_nan_fraction:
        result.add_issue(
            ValidationSeverity.ERROR,
            "data_quality",
            f"Too many NaN values: {nan_fraction*100:.2f}%",
            n_nan=int(n_nan),
            fraction=nan_fraction,
        )
    elif n_nan > 0:
        result.add_issue(
            ValidationSeverity.WARNING,
            "data_quality",
            f"Data contains {n_nan} NaN values ({nan_fraction*100:.4f}%)",
        )

    # Check for Inf
    n_inf = np.sum(np.isinf(data))
    if n_inf > 0:
        result.add_issue(
            ValidationSeverity.ERROR,
            "data_quality",
            f"Data contains {n_inf} infinite values",
        )

    # Check for all-zero traces
    trace_sums = np.sum(np.abs(data), axis=1)
    n_zero_traces = np.sum(trace_sums == 0)
    zero_trace_fraction = n_zero_traces / n_traces

    if zero_trace_fraction > max_zero_fraction:
        result.add_issue(
            ValidationSeverity.ERROR,
            "data_quality",
            f"Too many zero traces: {zero_trace_fraction*100:.1f}%",
        )
    elif n_zero_traces > 0:
        result.add_issue(
            ValidationSeverity.WARNING,
            "data_quality",
            f"{n_zero_traces} traces ({zero_trace_fraction*100:.1f}%) are all zeros",
        )

    # Check amplitude range
    finite_data = data[np.isfinite(data)]
    if len(finite_data) > 0:
        amp_range = finite_data.max() - finite_data.min()
        if amp_range < 1e-10:
            result.add_issue(
                ValidationSeverity.WARNING,
                "data_quality",
                "Very small amplitude range - data may be constant",
            )

    return result


def validate_geometry(
    source_x: NDArray,
    source_y: NDArray,
    receiver_x: NDArray,
    receiver_y: NDArray,
    offset: NDArray | None = None,
) -> ValidationResult:
    """
    Validate geometry data.

    Args:
        source_x, source_y: Source coordinates
        receiver_x, receiver_y: Receiver coordinates
        offset: Optional pre-computed offsets

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    n_traces = len(source_x)

    # Check array lengths
    if not all(len(arr) == n_traces for arr in [source_y, receiver_x, receiver_y]):
        result.add_issue(
            ValidationSeverity.ERROR,
            "geometry",
            "Coordinate arrays have inconsistent lengths",
        )
        return result

    # Check for NaN/Inf in coordinates
    for name, arr in [("source_x", source_x), ("source_y", source_y),
                       ("receiver_x", receiver_x), ("receiver_y", receiver_y)]:
        if np.any(~np.isfinite(arr)):
            result.add_issue(
                ValidationSeverity.ERROR,
                "geometry",
                f"{name} contains NaN or Inf values",
            )

    # Compute offsets if not provided
    if offset is None:
        offset = np.sqrt((receiver_x - source_x)**2 + (receiver_y - source_y)**2)

    # Check for zero offsets
    n_zero_offset = np.sum(offset < 1.0)
    if n_zero_offset > n_traces * 0.01:  # More than 1%
        result.add_issue(
            ValidationSeverity.WARNING,
            "geometry",
            f"{n_zero_offset} traces ({n_zero_offset/n_traces*100:.1f}%) have near-zero offset",
        )

    # Check for duplicate positions
    positions = np.column_stack([source_x, source_y, receiver_x, receiver_y])
    unique_positions = np.unique(positions, axis=0)
    n_duplicates = n_traces - len(unique_positions)

    if n_duplicates > 0:
        result.add_issue(
            ValidationSeverity.WARNING,
            "geometry",
            f"{n_duplicates} duplicate trace positions detected",
        )

    # Check coordinate range is reasonable
    x_range = max(source_x.max(), receiver_x.max()) - min(source_x.min(), receiver_x.min())
    y_range = max(source_y.max(), receiver_y.max()) - min(source_y.min(), receiver_y.min())

    if x_range > 1e6 or y_range > 1e6:
        result.add_issue(
            ValidationSeverity.WARNING,
            "geometry",
            f"Very large survey extent: {x_range/1000:.0f} x {y_range/1000:.0f} km",
        )

    result.add_issue(
        ValidationSeverity.INFO,
        "geometry",
        f"Geometry valid: {n_traces} traces, offset range [{offset.min():.0f}, {offset.max():.0f}] m",
    )

    return result


# =============================================================================
# Configuration Validation
# =============================================================================


def validate_output_grid(
    x_min: float,
    x_max: float,
    dx: float,
    y_min: float,
    y_max: float,
    dy: float,
    t_min_ms: float,
    t_max_ms: float,
    dt_ms: float,
    max_memory_gb: float = 64.0,
) -> ValidationResult:
    """
    Validate output grid parameters.

    Args:
        x_min, x_max, dx: X axis parameters
        y_min, y_max, dy: Y axis parameters
        t_min_ms, t_max_ms, dt_ms: Time axis parameters
        max_memory_gb: Maximum allowed memory usage

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    # Check ranges
    if x_max <= x_min:
        result.add_issue(
            ValidationSeverity.ERROR,
            "grid",
            "X range invalid: x_max must be greater than x_min",
        )

    if y_max <= y_min:
        result.add_issue(
            ValidationSeverity.ERROR,
            "grid",
            "Y range invalid: y_max must be greater than y_min",
        )

    if t_max_ms <= t_min_ms:
        result.add_issue(
            ValidationSeverity.ERROR,
            "grid",
            "Time range invalid: t_max must be greater than t_min",
        )

    # Check step sizes
    if dx <= 0:
        result.add_issue(ValidationSeverity.ERROR, "grid", "dx must be positive")
    if dy <= 0:
        result.add_issue(ValidationSeverity.ERROR, "grid", "dy must be positive")
    if dt_ms <= 0:
        result.add_issue(ValidationSeverity.ERROR, "grid", "dt must be positive")

    if not result.valid:
        return result

    # Calculate dimensions
    nx = int((x_max - x_min) / dx) + 1
    ny = int((y_max - y_min) / dy) + 1
    nt = int((t_max_ms - t_min_ms) / dt_ms) + 1

    # Check dimension limits
    if nx > 10000:
        result.add_issue(
            ValidationSeverity.WARNING,
            "grid",
            f"Very large X dimension: {nx}",
        )

    if ny > 10000:
        result.add_issue(
            ValidationSeverity.WARNING,
            "grid",
            f"Very large Y dimension: {ny}",
        )

    if nt > 20000:
        result.add_issue(
            ValidationSeverity.WARNING,
            "grid",
            f"Very large time dimension: {nt}",
        )

    # Check memory usage
    n_samples = nx * ny * nt
    memory_gb = n_samples * 4 / (1024**3)  # float32

    if memory_gb > max_memory_gb:
        result.add_issue(
            ValidationSeverity.ERROR,
            "grid",
            f"Output size ({memory_gb:.1f} GB) exceeds limit ({max_memory_gb} GB)",
            size_gb=memory_gb,
        )
    elif memory_gb > max_memory_gb * 0.5:
        result.add_issue(
            ValidationSeverity.WARNING,
            "grid",
            f"Large output size: {memory_gb:.1f} GB",
        )

    result.add_issue(
        ValidationSeverity.INFO,
        "grid",
        f"Output grid: {nx}×{ny}×{nt} = {n_samples:,} samples ({memory_gb:.2f} GB)",
    )

    return result


# =============================================================================
# Safe Operations
# =============================================================================


def safe_divide(
    numerator: NDArray,
    denominator: NDArray,
    default: float = 0.0,
) -> NDArray:
    """
    Safe division handling zero denominator.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        default: Value to use when denominator is zero

    Returns:
        Result array with default values where denominator is zero
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(denominator != 0, numerator / denominator, default)
    return result


def safe_sqrt(x: NDArray, min_value: float = 0.0) -> NDArray:
    """
    Safe square root handling negative values.

    Args:
        x: Input array
        min_value: Minimum value to clamp to before sqrt

    Returns:
        Square root of clamped values
    """
    return np.sqrt(np.maximum(x, min_value))


def clip_to_valid(
    data: NDArray,
    min_val: float | None = None,
    max_val: float | None = None,
    replace_nan: float | None = None,
    replace_inf: float | None = None,
) -> NDArray:
    """
    Clip data to valid range and handle NaN/Inf.

    Args:
        data: Input array
        min_val: Minimum valid value
        max_val: Maximum valid value
        replace_nan: Value to replace NaN with
        replace_inf: Value to replace Inf with

    Returns:
        Cleaned array
    """
    result = data.copy()

    if replace_nan is not None:
        result = np.where(np.isnan(result), replace_nan, result)

    if replace_inf is not None:
        result = np.where(np.isinf(result), replace_inf, result)

    if min_val is not None or max_val is not None:
        result = np.clip(result, min_val, max_val)

    return result


def handle_empty_tile(
    tile_shape: tuple[int, int, int],
    dtype: np.dtype = np.float64,
) -> tuple[NDArray, NDArray]:
    """
    Create empty output for a tile with no contributing traces.

    Args:
        tile_shape: (nx, ny, nt) shape
        dtype: Output dtype

    Returns:
        Tuple of (image, fold) arrays
    """
    image = np.zeros(tile_shape, dtype=dtype)
    fold = np.zeros(tile_shape[:2], dtype=np.int32)
    return image, fold


# =============================================================================
# Error Recovery
# =============================================================================


class RecoverableError(Exception):
    """An error that can be recovered from."""

    def __init__(self, message: str, recovery_action: str = "skip"):
        super().__init__(message)
        self.recovery_action = recovery_action


def with_recovery(
    func,
    *args,
    default_return=None,
    log_errors: bool = True,
    **kwargs,
):
    """
    Execute function with automatic error recovery.

    Args:
        func: Function to execute
        *args: Positional arguments
        default_return: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except RecoverableError as e:
        if log_errors:
            logger.warning(f"Recoverable error in {func.__name__}: {e}")
        return default_return
    except Exception as e:
        if log_errors:
            logger.error(f"Error in {func.__name__}: {e}")
        return default_return
