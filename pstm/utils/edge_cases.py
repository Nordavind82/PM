"""
Edge case handling and validation module for PSTM.

Provides robust input validation, error recovery, and edge case handling.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

from pstm.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A validation issue found during checking."""
    
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    suggestion: str | None = None

    def __str__(self) -> str:
        s = f"[{self.severity.value.upper()}] {self.field}: {self.message}"
        if self.suggestion:
            s += f" (Suggestion: {self.suggestion})"
        return s


@dataclass
class ValidationResult:
    """Result of validation checks."""
    
    issues: list[ValidationIssue] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if no errors or critical issues."""
        return not any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in self.issues
        )
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)
    
    @property
    def errors(self) -> list[ValidationIssue]:
        """Get error-level issues."""
        return [i for i in self.issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)]
    
    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    def add(
        self,
        field: str,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        value: Any = None,
        suggestion: str | None = None,
    ) -> None:
        """Add a validation issue."""
        self.issues.append(ValidationIssue(field, message, severity, value, suggestion))
    
    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.issues.extend(other.issues)
    
    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            errors = "\n".join(str(e) for e in self.errors)
            raise ValueError(f"Validation failed:\n{errors}")


# =============================================================================
# Input Data Validation
# =============================================================================


def validate_trace_data(
    data: NDArray,
    expected_dtype: np.dtype | None = None,
    max_nan_fraction: float | None = None,
    max_inf_fraction: float | None = None,
    check_zero_traces: bool = True,
) -> ValidationResult:
    """
    Validate trace amplitude data.
    
    Args:
        data: Trace data array (n_traces, n_samples)
        expected_dtype: Expected data type
        max_nan_fraction: Maximum allowed NaN fraction (default from settings)
        max_inf_fraction: Maximum allowed Inf fraction (default from settings)
        check_zero_traces: Check for all-zero traces
        
    Returns:
        ValidationResult
    """
    from pstm.settings import get_settings
    s = get_settings()
    
    if max_nan_fraction is None:
        max_nan_fraction = s.io.max_nan_fraction
    if max_inf_fraction is None:
        max_inf_fraction = s.io.max_inf_fraction
    
    result = ValidationResult()
    
    # Check shape
    if data.ndim != 2:
        result.add(
            "trace_data",
            f"Expected 2D array, got {data.ndim}D",
            ValidationSeverity.ERROR,
            suggestion="Reshape data to (n_traces, n_samples)",
        )
        return result
    
    n_traces, n_samples = data.shape
    
    if n_traces == 0:
        result.add("trace_data", "Empty trace data (0 traces)", ValidationSeverity.ERROR)
        return result
    
    if n_samples == 0:
        result.add("trace_data", "Empty trace data (0 samples)", ValidationSeverity.ERROR)
        return result
    
    # Check dtype
    if expected_dtype is not None and data.dtype != expected_dtype:
        result.add(
            "trace_data.dtype",
            f"Expected {expected_dtype}, got {data.dtype}",
            ValidationSeverity.WARNING,
            suggestion=f"Convert with data.astype({expected_dtype})",
        )
    
    # Check for NaN
    nan_count = np.sum(np.isnan(data))
    nan_fraction = nan_count / data.size
    if nan_fraction > max_nan_fraction:
        result.add(
            "trace_data",
            f"Too many NaN values: {nan_fraction*100:.2f}% (max: {max_nan_fraction*100:.2f}%)",
            ValidationSeverity.ERROR,
            value=nan_count,
            suggestion="Interpolate or remove NaN values",
        )
    elif nan_count > 0:
        result.add(
            "trace_data",
            f"Contains {nan_count} NaN values ({nan_fraction*100:.4f}%)",
            ValidationSeverity.WARNING,
        )
    
    # Check for Inf
    inf_count = np.sum(np.isinf(data))
    inf_fraction = inf_count / data.size
    if inf_fraction > max_inf_fraction:
        result.add(
            "trace_data",
            f"Contains Inf values: {inf_count}",
            ValidationSeverity.ERROR,
            suggestion="Replace Inf values with finite values",
        )
    
    # Check for all-zero traces
    if check_zero_traces:
        zero_traces = np.sum(np.all(data == 0, axis=1))
        if zero_traces > 0:
            zero_fraction = zero_traces / n_traces
            if zero_fraction > 0.5:
                result.add(
                    "trace_data",
                    f"{zero_traces} all-zero traces ({zero_fraction*100:.1f}%)",
                    ValidationSeverity.WARNING,
                    suggestion="Check data loading or apply muting after migration",
                )
    
    # Check amplitude range
    finite_data = data[np.isfinite(data)]
    if len(finite_data) > 0:
        amp_range = finite_data.max() - finite_data.min()
        if amp_range == 0:
            result.add(
                "trace_data",
                "All trace values are identical (zero dynamic range)",
                ValidationSeverity.ERROR,
            )
        elif amp_range < 1e-20:
            result.add(
                "trace_data",
                f"Very small amplitude range: {amp_range:.2e}",
                ValidationSeverity.WARNING,
                suggestion="Check data scaling",
            )
    
    # Check contiguity
    if not data.flags['C_CONTIGUOUS']:
        result.add(
            "trace_data",
            "Array is not C-contiguous",
            ValidationSeverity.WARNING,
            suggestion="Use np.ascontiguousarray() for better performance",
        )
    
    return result


def validate_coordinates(
    x: NDArray,
    y: NDArray,
    name: str = "coordinates",
    expected_unit: str = "meters",
    max_value: float = 1e8,
) -> ValidationResult:
    """
    Validate coordinate arrays.
    
    Args:
        x: X coordinates
        y: Y coordinates
        name: Name for error messages
        expected_unit: Expected coordinate unit
        max_value: Maximum reasonable coordinate value
        
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    # Check lengths match
    if len(x) != len(y):
        result.add(
            name,
            f"X and Y lengths differ: {len(x)} vs {len(y)}",
            ValidationSeverity.ERROR,
        )
        return result
    
    if len(x) == 0:
        result.add(name, "Empty coordinate arrays", ValidationSeverity.ERROR)
        return result
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(x)):
        n_bad = np.sum(~np.isfinite(x))
        result.add(
            f"{name}.x",
            f"Contains {n_bad} non-finite values",
            ValidationSeverity.ERROR,
        )
    
    if np.any(~np.isfinite(y)):
        n_bad = np.sum(~np.isfinite(y))
        result.add(
            f"{name}.y",
            f"Contains {n_bad} non-finite values",
            ValidationSeverity.ERROR,
        )
    
    # Check range
    x_range = (np.nanmin(x), np.nanmax(x))
    y_range = (np.nanmin(y), np.nanmax(y))
    
    if abs(x_range[0]) > max_value or abs(x_range[1]) > max_value:
        result.add(
            f"{name}.x",
            f"X range [{x_range[0]:.0f}, {x_range[1]:.0f}] exceeds reasonable bounds",
            ValidationSeverity.WARNING,
            suggestion=f"Check coordinate units (expected: {expected_unit})",
        )
    
    if abs(y_range[0]) > max_value or abs(y_range[1]) > max_value:
        result.add(
            f"{name}.y",
            f"Y range [{y_range[0]:.0f}, {y_range[1]:.0f}] exceeds reasonable bounds",
            ValidationSeverity.WARNING,
        )
    
    # Check for duplicate locations
    coords = np.column_stack([x, y])
    unique_coords = np.unique(coords, axis=0)
    n_duplicates = len(coords) - len(unique_coords)
    if n_duplicates > 0:
        dup_fraction = n_duplicates / len(coords)
        if dup_fraction > 0.5:
            result.add(
                name,
                f"{n_duplicates} duplicate coordinate pairs ({dup_fraction*100:.1f}%)",
                ValidationSeverity.WARNING,
            )
    
    return result


def validate_velocity(
    velocity: NDArray | float,
    t_axis_ms: NDArray | None = None,
    min_valid: float | None = None,
    max_valid: float | None = None,
    max_gradient: float | None = None,
) -> ValidationResult:
    """
    Validate velocity model.
    
    Args:
        velocity: Velocity values (scalar, 1D, or 3D)
        t_axis_ms: Time axis for gradient check
        min_valid: Minimum valid velocity (default from settings)
        max_valid: Maximum valid velocity (default from settings)
        max_gradient: Maximum velocity gradient (default from settings)
        
    Returns:
        ValidationResult
    """
    from pstm.settings import get_settings
    s = get_settings()
    
    if min_valid is None:
        min_valid = s.velocity.min_velocity_ms
    if max_valid is None:
        max_valid = s.velocity.max_velocity_ms
    if max_gradient is None:
        # Convert from m/s per second to m/s per ms
        max_gradient = s.velocity.max_gradient_ms_per_s / 1000.0
    
    result = ValidationResult()
    
    # Handle scalar
    if np.isscalar(velocity):
        velocity = np.array([velocity])
    
    # Check for NaN/Inf
    if np.any(np.isnan(velocity)):
        result.add("velocity", "Contains NaN values", ValidationSeverity.ERROR)
    
    if np.any(np.isinf(velocity)):
        result.add("velocity", "Contains Inf values", ValidationSeverity.ERROR)
    
    # Check range
    v_min = np.nanmin(velocity)
    v_max = np.nanmax(velocity)
    
    if v_min < min_valid:
        result.add(
            "velocity",
            f"Minimum velocity {v_min:.0f} m/s below valid range ({min_valid:.0f} m/s)",
            ValidationSeverity.ERROR,
        )
    
    if v_max > max_valid:
        result.add(
            "velocity",
            f"Maximum velocity {v_max:.0f} m/s above valid range ({max_valid:.0f} m/s)",
            ValidationSeverity.ERROR,
        )
    
    # Check gradient (1D case)
    if velocity.ndim == 1 and len(velocity) > 1 and t_axis_ms is not None:
        dt_ms = t_axis_ms[1] - t_axis_ms[0] if len(t_axis_ms) > 1 else 1.0
        dv = np.diff(velocity)
        gradient = dv / dt_ms
        
        max_grad = np.abs(gradient).max()
        if max_grad > max_gradient:
            result.add(
                "velocity",
                f"High velocity gradient: {max_grad:.2f} m/s per ms",
                ValidationSeverity.WARNING,
                suggestion="Check for velocity artifacts or use smoothing",
            )
        
        # Check for inversions
        n_inversions = np.sum(dv < -50)  # 50 m/s threshold
        if n_inversions > 0:
            result.add(
                "velocity",
                f"Detected {n_inversions} velocity inversions",
                ValidationSeverity.WARNING,
            )
    
    return result


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
    max_output_gb: float = 100.0,
) -> ValidationResult:
    """
    Validate output grid parameters.
    
    Args:
        x_min, x_max, dx: X grid parameters
        y_min, y_max, dy: Y grid parameters
        t_min_ms, t_max_ms, dt_ms: Time grid parameters
        max_output_gb: Maximum output size warning threshold
        
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    # Check ranges
    if x_max <= x_min:
        result.add("output_grid.x", "x_max must be greater than x_min", ValidationSeverity.ERROR)
    
    if y_max <= y_min:
        result.add("output_grid.y", "y_max must be greater than y_min", ValidationSeverity.ERROR)
    
    if t_max_ms <= t_min_ms:
        result.add("output_grid.t", "t_max_ms must be greater than t_min_ms", ValidationSeverity.ERROR)
    
    # Check spacing
    if dx <= 0:
        result.add("output_grid.dx", "dx must be positive", ValidationSeverity.ERROR)
    
    if dy <= 0:
        result.add("output_grid.dy", "dy must be positive", ValidationSeverity.ERROR)
    
    if dt_ms <= 0:
        result.add("output_grid.dt_ms", "dt_ms must be positive", ValidationSeverity.ERROR)
    
    if not result.is_valid:
        return result
    
    # Check grid dimensions
    nx = int((x_max - x_min) / dx) + 1
    ny = int((y_max - y_min) / dy) + 1
    nt = int((t_max_ms - t_min_ms) / dt_ms) + 1
    
    if nx > 10000:
        result.add(
            "output_grid.nx",
            f"Very large X dimension: {nx}",
            ValidationSeverity.WARNING,
            suggestion="Consider increasing dx or using tiling",
        )
    
    if ny > 10000:
        result.add(
            "output_grid.ny",
            f"Very large Y dimension: {ny}",
            ValidationSeverity.WARNING,
        )
    
    if nt > 10000:
        result.add(
            "output_grid.nt",
            f"Very large T dimension: {nt}",
            ValidationSeverity.WARNING,
            suggestion="Consider increasing dt_ms",
        )
    
    # Check output size
    output_size_gb = (nx * ny * nt * 4) / (1024**3)  # float32
    if output_size_gb > max_output_gb:
        result.add(
            "output_grid",
            f"Output size {output_size_gb:.1f} GB exceeds {max_output_gb:.1f} GB",
            ValidationSeverity.WARNING,
            suggestion="Consider coarser grid or output compression",
        )
    
    # Check for reasonable spacing
    if dx < 1.0:
        result.add(
            "output_grid.dx",
            f"Very small dx ({dx} m) may cause performance issues",
            ValidationSeverity.WARNING,
        )
    
    if dt_ms < 0.5:
        result.add(
            "output_grid.dt_ms",
            f"Very small dt ({dt_ms} ms) may cause performance issues",
            ValidationSeverity.WARNING,
        )
    
    return result


def validate_file_path(
    path: Path | str,
    must_exist: bool = True,
    expected_suffix: str | list[str] | None = None,
    writable: bool = False,
) -> ValidationResult:
    """
    Validate file path.
    
    Args:
        path: File path to validate
        must_exist: Path must exist
        expected_suffix: Expected file extension(s)
        writable: Check if path is writable
        
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    path = Path(path)
    
    if must_exist and not path.exists():
        result.add(
            "path",
            f"Path does not exist: {path}",
            ValidationSeverity.ERROR,
        )
        return result
    
    if expected_suffix:
        if isinstance(expected_suffix, str):
            expected_suffix = [expected_suffix]
        
        if path.suffix.lower() not in [s.lower() for s in expected_suffix]:
            result.add(
                "path",
                f"Unexpected file extension: {path.suffix} (expected: {expected_suffix})",
                ValidationSeverity.WARNING,
            )
    
    if writable and path.exists():
        try:
            # Try to open for writing
            if path.is_dir():
                test_file = path / ".write_test"
                test_file.touch()
                test_file.unlink()
            else:
                with open(path, "a"):
                    pass
        except PermissionError:
            result.add(
                "path",
                f"Path is not writable: {path}",
                ValidationSeverity.ERROR,
            )
    
    return result


# =============================================================================
# Error Recovery
# =============================================================================


def safe_divide(
    numerator: NDArray,
    denominator: NDArray,
    fill_value: float = 0.0,
) -> NDArray:
    """
    Safe division handling zero denominators.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        fill_value: Value to use where denominator is zero
        
    Returns:
        Division result with fill_value where denominator is zero
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(denominator != 0, numerator / denominator, fill_value)
    return result


def clip_to_valid_range(
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
        min_val: Minimum clip value
        max_val: Maximum clip value
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


def handle_empty_input(
    data: NDArray | None,
    default_shape: tuple[int, ...],
    default_dtype: np.dtype = np.float32,
    fill_value: float = 0.0,
) -> NDArray:
    """
    Handle empty or None input arrays.
    
    Args:
        data: Input array (may be None or empty)
        default_shape: Default shape if empty
        default_dtype: Default dtype
        fill_value: Fill value for empty array
        
    Returns:
        Valid array (original or default)
    """
    if data is None:
        return np.full(default_shape, fill_value, dtype=default_dtype)
    
    if isinstance(data, np.ndarray) and data.size == 0:
        return np.full(default_shape, fill_value, dtype=default_dtype)
    
    return data


def robust_mean(
    data: NDArray,
    axis: int | None = None,
    percentile_range: tuple[float, float] = (5, 95),
) -> float | NDArray:
    """
    Compute robust mean excluding outliers.
    
    Args:
        data: Input array
        axis: Axis for computation
        percentile_range: Percentile range to include
        
    Returns:
        Robust mean value(s)
    """
    low = np.percentile(data, percentile_range[0], axis=axis, keepdims=True)
    high = np.percentile(data, percentile_range[1], axis=axis, keepdims=True)
    
    mask = (data >= low) & (data <= high)
    
    if axis is None:
        return np.mean(data[mask])
    else:
        # This is simplified - proper implementation would be more complex
        return np.mean(np.where(mask, data, np.nan), axis=axis)


# =============================================================================
# Edge Case Handlers for Migration
# =============================================================================


def handle_zero_traces_in_aperture(
    n_traces: int,
    tile_id: int,
) -> dict[str, Any]:
    """
    Handle case when no traces fall within aperture.
    
    Args:
        n_traces: Number of traces (0)
        tile_id: Tile identifier
        
    Returns:
        Empty result dict with appropriate flags
    """
    logger.debug(f"Tile {tile_id}: No traces in aperture")
    return {
        'skip': True,
        'reason': 'no_traces_in_aperture',
        'n_traces': 0,
    }


def handle_invalid_travel_time(
    travel_time_s: float,
    max_valid_time_s: float = 20.0,
) -> tuple[bool, float]:
    """
    Handle invalid travel time calculations.
    
    Args:
        travel_time_s: Calculated travel time
        max_valid_time_s: Maximum valid travel time
        
    Returns:
        Tuple of (is_valid, clamped_time)
    """
    if not np.isfinite(travel_time_s):
        return False, 0.0
    
    if travel_time_s < 0:
        return False, 0.0
    
    if travel_time_s > max_valid_time_s:
        return False, max_valid_time_s
    
    return True, travel_time_s


def handle_interpolation_bounds(
    sample_idx: float,
    n_samples: int,
    method: str = "clamp",
) -> tuple[bool, float]:
    """
    Handle out-of-bounds interpolation indices.
    
    Args:
        sample_idx: Fractional sample index
        n_samples: Number of samples
        method: "clamp", "skip", or "extrapolate"
        
    Returns:
        Tuple of (is_valid, adjusted_index)
    """
    if method == "skip":
        if sample_idx < 0 or sample_idx >= n_samples - 1:
            return False, sample_idx
        return True, sample_idx
    
    elif method == "clamp":
        clamped = np.clip(sample_idx, 0, n_samples - 1 - 1e-6)
        return True, clamped
    
    elif method == "extrapolate":
        # Allow small extrapolation
        if sample_idx < -1 or sample_idx >= n_samples:
            return False, sample_idx
        return True, np.clip(sample_idx, 0, n_samples - 1 - 1e-6)
    
    return True, sample_idx


def handle_memory_overflow(
    requested_bytes: int,
    available_bytes: int,
    strategy: str = "reduce",
) -> dict[str, Any]:
    """
    Handle memory overflow conditions.
    
    Args:
        requested_bytes: Requested memory
        available_bytes: Available memory
        strategy: "reduce", "error", or "warn"
        
    Returns:
        Action dict
    """
    if requested_bytes <= available_bytes:
        return {'action': 'proceed', 'bytes': requested_bytes}
    
    if strategy == "error":
        raise MemoryError(
            f"Requested {requested_bytes / 1e9:.2f} GB but only "
            f"{available_bytes / 1e9:.2f} GB available"
        )
    
    elif strategy == "reduce":
        # Calculate reduction factor
        reduction = available_bytes / requested_bytes * 0.8  # 80% safety margin
        return {
            'action': 'reduce',
            'reduction_factor': reduction,
            'original_bytes': requested_bytes,
            'reduced_bytes': int(requested_bytes * reduction),
        }
    
    else:  # warn
        warnings.warn(
            f"Memory warning: requested {requested_bytes / 1e9:.2f} GB, "
            f"available {available_bytes / 1e9:.2f} GB"
        )
        return {'action': 'proceed', 'bytes': requested_bytes, 'warning': True}
