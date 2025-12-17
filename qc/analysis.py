"""
Quality Control module for PSTM.

Provides geometry, velocity, and output QC functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pstm.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Geometry QC
# =============================================================================


@dataclass
class GeometryQCReport:
    """Report from geometry QC analysis."""

    n_traces: int
    n_shots: int

    # Survey extent
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    midpoint_x_range: tuple[float, float]
    midpoint_y_range: tuple[float, float]

    # Offset statistics
    offset_min: float
    offset_max: float
    offset_mean: float
    offset_std: float
    offset_histogram: tuple[NDArray, NDArray]  # counts, bin_edges

    # Azimuth statistics
    azimuth_histogram: tuple[NDArray, NDArray] | None

    # Fold statistics
    max_fold: int
    mean_fold: float
    fold_map: NDArray | None  # 2D fold map

    # Issues detected
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_traces": self.n_traces,
            "n_shots": self.n_shots,
            "x_range": self.x_range,
            "y_range": self.y_range,
            "offset_range": (self.offset_min, self.offset_max),
            "offset_mean": self.offset_mean,
            "max_fold": self.max_fold,
            "mean_fold": self.mean_fold,
            "n_warnings": len(self.warnings),
        }


def compute_fold_map(
    midpoint_x: NDArray[np.float64],
    midpoint_y: NDArray[np.float64],
    bin_size: float,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute fold map from midpoint coordinates.

    Args:
        midpoint_x: Midpoint X coordinates
        midpoint_y: Midpoint Y coordinates
        bin_size: Bin size for fold calculation
        x_range: Optional X range (auto-computed if None)
        y_range: Optional Y range (auto-computed if None)

    Returns:
        Tuple of (fold_map, x_bins, y_bins)
    """
    if x_range is None:
        x_range = (midpoint_x.min(), midpoint_x.max())
    if y_range is None:
        y_range = (midpoint_y.min(), midpoint_y.max())

    # Create bin edges
    x_bins = np.arange(x_range[0], x_range[1] + bin_size, bin_size)
    y_bins = np.arange(y_range[0], y_range[1] + bin_size, bin_size)

    # Compute 2D histogram
    fold_map, _, _ = np.histogram2d(
        midpoint_x, midpoint_y,
        bins=[x_bins, y_bins],
    )

    return fold_map.astype(np.int32), x_bins, y_bins


def analyze_offsets(
    offset: NDArray[np.float64],
    n_bins: int = 50,
) -> dict[str, Any]:
    """
    Analyze offset distribution.

    Args:
        offset: Offset values
        n_bins: Number of histogram bins

    Returns:
        Dictionary with offset statistics
    """
    return {
        "min": float(offset.min()),
        "max": float(offset.max()),
        "mean": float(offset.mean()),
        "std": float(offset.std()),
        "median": float(np.median(offset)),
        "histogram": np.histogram(offset, bins=n_bins),
    }


def analyze_azimuths(
    source_x: NDArray[np.float64],
    source_y: NDArray[np.float64],
    receiver_x: NDArray[np.float64],
    receiver_y: NDArray[np.float64],
    n_bins: int = 36,
) -> dict[str, Any]:
    """
    Analyze azimuth distribution.

    Args:
        source_x, source_y: Source coordinates
        receiver_x, receiver_y: Receiver coordinates
        n_bins: Number of azimuth bins (default: 36 = 10° bins)

    Returns:
        Dictionary with azimuth statistics
    """
    dx = receiver_x - source_x
    dy = receiver_y - source_y

    # Compute azimuth (clockwise from north)
    azimuth = np.degrees(np.arctan2(dx, dy))
    azimuth = np.mod(azimuth, 360.0)

    hist, bin_edges = np.histogram(azimuth, bins=n_bins, range=(0, 360))

    return {
        "histogram": (hist, bin_edges),
        "mean": float(np.mean(azimuth)),
        "std": float(np.std(azimuth)),
        "dominant": float(bin_edges[np.argmax(hist)]),
    }


def run_geometry_qc(
    midpoint_x: NDArray[np.float64],
    midpoint_y: NDArray[np.float64],
    source_x: NDArray[np.float64],
    source_y: NDArray[np.float64],
    receiver_x: NDArray[np.float64],
    receiver_y: NDArray[np.float64],
    offset: NDArray[np.float64],
    shot_ids: NDArray[np.int32] | None = None,
    fold_bin_size: float = 25.0,
) -> GeometryQCReport:
    """
    Run comprehensive geometry QC.

    Args:
        midpoint_x, midpoint_y: Midpoint coordinates
        source_x, source_y: Source coordinates
        receiver_x, receiver_y: Receiver coordinates
        offset: Offset values
        shot_ids: Shot IDs (optional)
        fold_bin_size: Bin size for fold map

    Returns:
        GeometryQCReport
    """
    logger.info("Running geometry QC...")

    n_traces = len(midpoint_x)
    n_shots = len(np.unique(shot_ids)) if shot_ids is not None else 0

    warnings = []

    # Survey extent
    x_min = min(source_x.min(), receiver_x.min())
    x_max = max(source_x.max(), receiver_x.max())
    y_min = min(source_y.min(), receiver_y.min())
    y_max = max(source_y.max(), receiver_y.max())

    # Offset analysis
    offset_stats = analyze_offsets(offset)

    # Check for zero offsets
    n_zero_offset = np.sum(offset < 1.0)
    if n_zero_offset > 0:
        warnings.append(f"{n_zero_offset} traces with near-zero offset")

    # Azimuth analysis
    azimuth_stats = analyze_azimuths(source_x, source_y, receiver_x, receiver_y)

    # Fold map
    fold_map, x_bins, y_bins = compute_fold_map(
        midpoint_x, midpoint_y, fold_bin_size
    )

    max_fold = int(fold_map.max())
    mean_fold = float(fold_map[fold_map > 0].mean()) if np.any(fold_map > 0) else 0.0

    # Check fold coverage
    zero_fold_pct = 100 * np.sum(fold_map == 0) / fold_map.size
    if zero_fold_pct > 10:
        warnings.append(f"{zero_fold_pct:.1f}% of bins have zero fold")

    logger.info(f"Geometry QC complete: {n_traces} traces, {len(warnings)} warnings")

    return GeometryQCReport(
        n_traces=n_traces,
        n_shots=n_shots,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        midpoint_x_range=(midpoint_x.min(), midpoint_x.max()),
        midpoint_y_range=(midpoint_y.min(), midpoint_y.max()),
        offset_min=offset_stats["min"],
        offset_max=offset_stats["max"],
        offset_mean=offset_stats["mean"],
        offset_std=offset_stats["std"],
        offset_histogram=offset_stats["histogram"],
        azimuth_histogram=azimuth_stats["histogram"],
        max_fold=max_fold,
        mean_fold=mean_fold,
        fold_map=fold_map,
        warnings=warnings,
    )


# =============================================================================
# Velocity QC
# =============================================================================


@dataclass
class VelocityQCReport:
    """Report from velocity QC analysis."""

    velocity_type: str
    is_3d: bool

    # Value range
    v_min: float
    v_max: float
    v_mean: float

    # Gradient analysis
    max_gradient: float  # m/s per s
    n_inversions: int

    # Issues
    warnings: list[str]


def run_velocity_qc(
    vrms: NDArray[np.float64],
    t_axis_ms: NDArray[np.float64],
    min_valid: float = 1000.0,
    max_valid: float = 8000.0,
) -> VelocityQCReport:
    """
    Run velocity model QC.

    Args:
        vrms: Velocity values (can be 1D or 3D)
        t_axis_ms: Time axis in milliseconds
        min_valid: Minimum valid velocity
        max_valid: Maximum valid velocity

    Returns:
        VelocityQCReport
    """
    logger.info("Running velocity QC...")

    warnings = []
    is_3d = vrms.ndim > 1

    v_min = float(vrms.min())
    v_max = float(vrms.max())
    v_mean = float(vrms.mean())

    # Check bounds
    if v_min < min_valid:
        warnings.append(f"Velocity below minimum: {v_min:.0f} < {min_valid:.0f} m/s")
    if v_max > max_valid:
        warnings.append(f"Velocity above maximum: {v_max:.0f} > {max_valid:.0f} m/s")

    # Check for NaN/Inf
    if np.any(np.isnan(vrms)):
        warnings.append("Velocity contains NaN values")
    if np.any(np.isinf(vrms)):
        warnings.append("Velocity contains Inf values")

    # Gradient analysis (1D or sample 1D profile)
    if is_3d:
        # Use center profile
        nx, ny, nt = vrms.shape
        v_profile = vrms[nx // 2, ny // 2, :]
    else:
        v_profile = vrms

    dt_s = (t_axis_ms[1] - t_axis_ms[0]) / 1000.0 if len(t_axis_ms) > 1 else 0.001
    dv = np.diff(v_profile)
    gradient = dv / dt_s

    max_gradient = float(np.abs(gradient).max())
    n_inversions = int(np.sum(dv < -100))  # Significant inversions

    if n_inversions > 0:
        warnings.append(f"Detected {n_inversions} velocity inversions")

    if max_gradient > 5000:
        warnings.append(f"Very high velocity gradient: {max_gradient:.0f} m/s per s")

    logger.info(f"Velocity QC complete: range [{v_min:.0f}, {v_max:.0f}] m/s, {len(warnings)} warnings")

    return VelocityQCReport(
        velocity_type="3D cube" if is_3d else "1D profile",
        is_3d=is_3d,
        v_min=v_min,
        v_max=v_max,
        v_mean=v_mean,
        max_gradient=max_gradient,
        n_inversions=n_inversions,
        warnings=warnings,
    )


# =============================================================================
# Output QC
# =============================================================================


@dataclass
class OutputQCReport:
    """Report from output image QC."""

    shape: tuple[int, ...]
    dtype: str

    # Amplitude statistics
    amp_min: float
    amp_max: float
    amp_mean: float
    amp_std: float
    amp_rms: float

    # Data quality
    n_nan: int
    n_inf: int
    n_zero: int
    zero_percent: float

    # Fold statistics
    fold_max: int
    fold_mean: float

    # Issues
    warnings: list[str]


def run_output_qc(
    image: NDArray,
    fold: NDArray | None = None,
) -> OutputQCReport:
    """
    Run output image QC.

    Args:
        image: Migration output image
        fold: Fold volume (optional)

    Returns:
        OutputQCReport
    """
    logger.info("Running output QC...")

    warnings = []

    shape = image.shape
    dtype = str(image.dtype)

    # Amplitude statistics
    finite_mask = np.isfinite(image)
    finite_values = image[finite_mask]

    if len(finite_values) == 0:
        warnings.append("No finite values in output")
        return OutputQCReport(
            shape=shape,
            dtype=dtype,
            amp_min=0, amp_max=0, amp_mean=0, amp_std=0, amp_rms=0,
            n_nan=int(np.sum(np.isnan(image))),
            n_inf=int(np.sum(np.isinf(image))),
            n_zero=0,
            zero_percent=100.0,
            fold_max=0,
            fold_mean=0,
            warnings=warnings,
        )

    amp_min = float(finite_values.min())
    amp_max = float(finite_values.max())
    amp_mean = float(finite_values.mean())
    amp_std = float(finite_values.std())
    amp_rms = float(np.sqrt(np.mean(finite_values ** 2)))

    # Data quality checks
    n_nan = int(np.sum(np.isnan(image)))
    n_inf = int(np.sum(np.isinf(image)))
    n_zero = int(np.sum(np.abs(image) < 1e-30))
    zero_percent = 100.0 * n_zero / image.size

    if n_nan > 0:
        warnings.append(f"{n_nan} NaN values in output")
    if n_inf > 0:
        warnings.append(f"{n_inf} Inf values in output")
    if zero_percent > 90:
        warnings.append(f"{zero_percent:.1f}% of output is zero")

    # Fold statistics
    fold_max = 0
    fold_mean = 0.0
    if fold is not None:
        fold_max = int(fold.max())
        fold_mean = float(fold[fold > 0].mean()) if np.any(fold > 0) else 0.0

        if fold_max == 0:
            warnings.append("Maximum fold is zero - no data migrated")

    logger.info(f"Output QC complete: shape {shape}, {len(warnings)} warnings")

    return OutputQCReport(
        shape=shape,
        dtype=dtype,
        amp_min=amp_min,
        amp_max=amp_max,
        amp_mean=amp_mean,
        amp_std=amp_std,
        amp_rms=amp_rms,
        n_nan=n_nan,
        n_inf=n_inf,
        n_zero=n_zero,
        zero_percent=zero_percent,
        fold_max=fold_max,
        fold_mean=fold_mean,
        warnings=warnings,
    )


def extract_slice(
    volume: NDArray,
    axis: int,
    index: int,
) -> NDArray:
    """
    Extract a slice from a 3D volume.

    Args:
        volume: 3D array (nx, ny, nt)
        axis: Axis to slice (0=inline, 1=crossline, 2=time)
        index: Index along axis

    Returns:
        2D slice array
    """
    if axis == 0:
        return volume[index, :, :]
    elif axis == 1:
        return volume[:, index, :]
    elif axis == 2:
        return volume[:, :, index]
    else:
        raise ValueError(f"Invalid axis: {axis}")


# =============================================================================
# Migration Verification
# =============================================================================


@dataclass
class MigrationVerificationReport:
    """Report from migration verification tests."""

    test_name: str
    passed: bool
    expected_value: float
    actual_value: float
    tolerance: float
    message: str


def verify_diffractor_focus(
    image: NDArray,
    x_axis: NDArray,
    y_axis: NDArray,
    t_axis_ms: NDArray,
    expected_x: float,
    expected_y: float,
    expected_t_ms: float,
    tolerance_xy: float = 50.0,
    tolerance_t_ms: float = 20.0,
) -> MigrationVerificationReport:
    """
    Verify point diffractor focuses correctly.

    Args:
        image: Migrated image (nx, ny, nt)
        x_axis, y_axis, t_axis_ms: Output axes
        expected_x, expected_y, expected_t_ms: Expected focus location
        tolerance_xy: Spatial tolerance (meters)
        tolerance_t_ms: Time tolerance (ms)

    Returns:
        MigrationVerificationReport
    """
    # Find peak location
    peak_idx = np.unravel_index(np.argmax(np.abs(image)), image.shape)
    ix, iy, it = peak_idx

    actual_x = x_axis[ix]
    actual_y = y_axis[iy]
    actual_t = t_axis_ms[it]

    # Check distances
    xy_error = np.sqrt((actual_x - expected_x) ** 2 + (actual_y - expected_y) ** 2)
    t_error = abs(actual_t - expected_t_ms)

    passed = (xy_error <= tolerance_xy) and (t_error <= tolerance_t_ms)

    message = (
        f"Peak at ({actual_x:.0f}, {actual_y:.0f}, {actual_t:.0f}ms), "
        f"expected ({expected_x:.0f}, {expected_y:.0f}, {expected_t_ms:.0f}ms). "
        f"XY error: {xy_error:.0f}m, T error: {t_error:.0f}ms"
    )

    return MigrationVerificationReport(
        test_name="diffractor_focus",
        passed=passed,
        expected_value=expected_t_ms,
        actual_value=actual_t,
        tolerance=tolerance_t_ms,
        message=message,
    )


def verify_flat_reflector_depth(
    image: NDArray,
    t_axis_ms: NDArray,
    expected_t_ms: float,
    tolerance_ms: float = 10.0,
) -> MigrationVerificationReport:
    """
    Verify flat reflector is at correct depth.

    Args:
        image: Migrated image (nx, ny, nt)
        t_axis_ms: Time axis
        expected_t_ms: Expected reflector time
        tolerance_ms: Time tolerance

    Returns:
        MigrationVerificationReport
    """
    # Stack horizontally and find peak time
    stacked = np.abs(image).mean(axis=(0, 1))
    peak_it = np.argmax(stacked)
    actual_t = t_axis_ms[peak_it]

    t_error = abs(actual_t - expected_t_ms)
    passed = t_error <= tolerance_ms

    message = (
        f"Reflector at {actual_t:.0f}ms, expected {expected_t_ms:.0f}ms. "
        f"Error: {t_error:.0f}ms"
    )

    return MigrationVerificationReport(
        test_name="flat_reflector_depth",
        passed=passed,
        expected_value=expected_t_ms,
        actual_value=actual_t,
        tolerance=tolerance_ms,
        message=message,
    )
