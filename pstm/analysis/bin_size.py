"""
Bin size auto-calculation algorithms.

This module provides algorithms to automatically determine optimal bin sizes
from input seismic data based on midpoint distribution analysis.

Methods:
- Histogram-based: Finds modal spacing from sorted coordinate differences
- KDE (Kernel Density Estimation): Smoothed peak detection
- FFT-based: Frequency analysis for regular grids

The algorithms analyze midpoint (CMP) distribution to determine natural
acquisition spacing and recommend appropriate output bin sizes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class BinSizeMethod(str, Enum):
    """Method for bin size calculation."""
    HISTOGRAM = "histogram"
    KDE = "kde"
    FFT = "fft"
    NEAREST_NEIGHBOR = "nearest_neighbor"


# Standard bin sizes used in seismic processing (meters)
STANDARD_BIN_SIZES = [
    6.25, 12.5, 18.75, 25.0, 37.5, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0
]


@dataclass
class BinSizeResult:
    """Result of bin size auto-calculation."""

    # Recommended bin sizes
    dx: float  # Inline bin size (meters)
    dy: float  # Crossline bin size (meters)

    # Raw detected spacings before rounding
    raw_dx: float
    raw_dy: float

    # Method used for detection
    method: BinSizeMethod

    # Quality metrics
    confidence: float  # 0-1, how confident in the result
    fold_estimate: float  # Estimated fold per bin
    coverage_ratio: float  # Ratio of bins with data

    # Diagnostics
    n_points_analyzed: int
    detection_details: dict

    # Warnings
    warnings: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "dx": self.dx,
            "dy": self.dy,
            "raw_dx": self.raw_dx,
            "raw_dy": self.raw_dy,
            "method": self.method.value,
            "confidence": self.confidence,
            "fold_estimate": self.fold_estimate,
            "coverage_ratio": self.coverage_ratio,
            "n_points_analyzed": self.n_points_analyzed,
            "detection_details": self.detection_details,
            "warnings": self.warnings,
        }


def round_to_standard_bin_size(value: float, tolerance: float = 0.15) -> float:
    """
    Round a bin size to the nearest standard value.

    Args:
        value: Raw bin size in meters
        tolerance: Fractional tolerance for rounding (default 15%)

    Returns:
        Nearest standard bin size
    """
    if value <= 0:
        return STANDARD_BIN_SIZES[0]

    # Find nearest standard size
    best_match = min(STANDARD_BIN_SIZES, key=lambda s: abs(s - value))

    # Check if within tolerance
    if abs(best_match - value) / value <= tolerance:
        return best_match

    # Otherwise return the original rounded to reasonable precision
    return round(value, 2)


def find_histogram_peak(
    values: NDArray[np.floating],
    bin_width: float = 5.0,
    min_count_ratio: float = 0.02,
) -> tuple[float, float]:
    """
    Find the modal (most common) value using histogram analysis.

    Args:
        values: Array of spacing values
        bin_width: Histogram bin width
        min_count_ratio: Minimum fraction of total counts for a valid peak

    Returns:
        Tuple of (modal_value, confidence)
    """
    if len(values) == 0:
        return 0.0, 0.0

    # Create histogram
    v_min, v_max = values.min(), values.max()
    if v_max <= v_min:
        return float(v_min), 0.5

    n_bins = max(10, int((v_max - v_min) / bin_width))
    counts, edges = np.histogram(values, bins=n_bins)

    # Find peak
    peak_idx = np.argmax(counts)
    peak_value = (edges[peak_idx] + edges[peak_idx + 1]) / 2

    # Calculate confidence based on peak prominence
    total_counts = counts.sum()
    peak_ratio = counts[peak_idx] / total_counts if total_counts > 0 else 0

    # Higher confidence if peak is more pronounced
    confidence = min(1.0, peak_ratio * 5)  # Scale to 0-1

    return float(peak_value), float(confidence)


def compute_coordinate_spacings(
    coords: NDArray[np.floating],
    sample_size: int = 100000,
) -> NDArray[np.floating]:
    """
    Compute spacings between consecutive sorted coordinates.

    Args:
        coords: Array of coordinate values
        sample_size: Maximum number of points to use

    Returns:
        Array of positive spacing values
    """
    # Sample if too many points
    if len(coords) > sample_size:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(coords), sample_size, replace=False)
        coords = coords[indices]

    # Sort and compute differences
    sorted_coords = np.sort(coords)
    diffs = np.diff(sorted_coords)

    # Filter out zero and very small values (duplicates)
    min_spacing = 1.0  # 1 meter minimum
    valid_diffs = diffs[diffs >= min_spacing]

    return valid_diffs


def calculate_bin_size_histogram(
    mx: NDArray[np.floating],
    my: NDArray[np.floating],
    azimuth_deg: float = 0.0,
) -> BinSizeResult:
    """
    Calculate optimal bin size using histogram analysis of coordinate spacings.

    This method:
    1. Optionally rotates coordinates to align with acquisition azimuth
    2. Computes sorted coordinate differences in X and Y
    3. Finds the modal (most common) spacing using histogram peak detection
    4. Rounds to nearest standard bin size

    Args:
        mx: Midpoint X coordinates
        my: Midpoint Y coordinates
        azimuth_deg: Acquisition azimuth in degrees (for rotation)

    Returns:
        BinSizeResult with recommended bin sizes
    """
    warnings = []
    n_points = len(mx)

    if n_points < 100:
        warnings.append(f"Very few points ({n_points}), bin size estimate may be unreliable")

    # Rotate coordinates if azimuth is non-zero
    if abs(azimuth_deg) > 0.1:
        theta = np.radians(azimuth_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        mx_rot = mx * cos_t + my * sin_t
        my_rot = -mx * sin_t + my * cos_t
    else:
        mx_rot, my_rot = mx, my

    # Compute spacings
    dx_spacings = compute_coordinate_spacings(mx_rot)
    dy_spacings = compute_coordinate_spacings(my_rot)

    if len(dx_spacings) == 0 or len(dy_spacings) == 0:
        warnings.append("Could not compute coordinate spacings")
        return BinSizeResult(
            dx=25.0, dy=25.0,
            raw_dx=25.0, raw_dy=25.0,
            method=BinSizeMethod.HISTOGRAM,
            confidence=0.0,
            fold_estimate=0.0,
            coverage_ratio=0.0,
            n_points_analyzed=n_points,
            detection_details={},
            warnings=warnings,
        )

    # Find modal spacings
    raw_dx, conf_x = find_histogram_peak(dx_spacings)
    raw_dy, conf_y = find_histogram_peak(dy_spacings)

    # Round to standard sizes
    dx = round_to_standard_bin_size(raw_dx)
    dy = round_to_standard_bin_size(raw_dy)

    # Estimate fold
    x_range = mx_rot.max() - mx_rot.min()
    y_range = my_rot.max() - my_rot.min()
    n_bins_x = max(1, int(x_range / dx))
    n_bins_y = max(1, int(y_range / dy))
    total_bins = n_bins_x * n_bins_y
    fold_estimate = n_points / total_bins if total_bins > 0 else 0

    # Compute coverage
    coverage_ratio = min(1.0, n_points / total_bins) if total_bins > 0 else 0

    # Overall confidence
    confidence = (conf_x + conf_y) / 2

    # Quality checks
    if fold_estimate < 5:
        warnings.append(f"Low fold estimate ({fold_estimate:.1f}), data may be sparse")
    if fold_estimate > 500:
        warnings.append(f"Very high fold ({fold_estimate:.1f}), consider larger bins")

    details = {
        "x_spacing_stats": {
            "median": float(np.median(dx_spacings)),
            "mean": float(np.mean(dx_spacings)),
            "std": float(np.std(dx_spacings)),
            "modal": raw_dx,
        },
        "y_spacing_stats": {
            "median": float(np.median(dy_spacings)),
            "mean": float(np.mean(dy_spacings)),
            "std": float(np.std(dy_spacings)),
            "modal": raw_dy,
        },
        "n_x_spacings": len(dx_spacings),
        "n_y_spacings": len(dy_spacings),
        "azimuth_applied": azimuth_deg,
    }

    return BinSizeResult(
        dx=dx,
        dy=dy,
        raw_dx=raw_dx,
        raw_dy=raw_dy,
        method=BinSizeMethod.HISTOGRAM,
        confidence=confidence,
        fold_estimate=fold_estimate,
        coverage_ratio=coverage_ratio,
        n_points_analyzed=n_points,
        detection_details=details,
        warnings=warnings,
    )


def calculate_bin_size_nearest_neighbor(
    mx: NDArray[np.floating],
    my: NDArray[np.floating],
    sample_size: int = 10000,
) -> BinSizeResult:
    """
    Calculate optimal bin size using nearest-neighbor distance analysis.

    This method:
    1. Samples points if dataset is large
    2. For each point, finds distance to nearest neighbor
    3. Uses median NN distance as basis for bin size

    Args:
        mx: Midpoint X coordinates
        my: Midpoint Y coordinates
        sample_size: Maximum points to analyze (for performance)

    Returns:
        BinSizeResult with recommended bin sizes
    """
    warnings = []
    n_points = len(mx)

    if n_points < 100:
        warnings.append(f"Very few points ({n_points}), using default bin size")
        return BinSizeResult(
            dx=25.0, dy=25.0,
            raw_dx=25.0, raw_dy=25.0,
            method=BinSizeMethod.NEAREST_NEIGHBOR,
            confidence=0.0,
            fold_estimate=0.0,
            coverage_ratio=0.0,
            n_points_analyzed=n_points,
            detection_details={},
            warnings=warnings,
        )

    # Sample if needed
    if n_points > sample_size:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(n_points, sample_size, replace=False)
        mx_sample = mx[indices]
        my_sample = my[indices]
        n_analyzed = sample_size
    else:
        mx_sample = mx
        my_sample = my
        n_analyzed = n_points

    # Compute pairwise distances using KDTree for efficiency
    try:
        from scipy.spatial import cKDTree

        points = np.column_stack([mx_sample, my_sample])
        tree = cKDTree(points)

        # Query nearest neighbor (k=2 because first is self)
        distances, _ = tree.query(points, k=2)
        nn_distances = distances[:, 1]  # Second column is nearest neighbor

    except ImportError:
        # Fallback: brute force on smaller sample
        warnings.append("scipy not available, using approximate method")

        sub_sample = min(1000, len(mx_sample))
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(mx_sample), sub_sample, replace=False)

        nn_distances = []
        for i in idx:
            dx = mx_sample - mx_sample[i]
            dy = my_sample - my_sample[i]
            dists = np.sqrt(dx**2 + dy**2)
            dists[i] = np.inf  # Exclude self
            nn_distances.append(dists.min())
        nn_distances = np.array(nn_distances)

    # Use median NN distance as bin size basis
    median_nn = float(np.median(nn_distances))

    # Typical bin size is NN distance / sqrt(2) for Nyquist-like sampling
    raw_bin = median_nn / np.sqrt(2)

    # Round to standard size
    bin_size = round_to_standard_bin_size(raw_bin)

    # Estimate fold
    x_range = mx.max() - mx.min()
    y_range = my.max() - my.min()
    n_bins = (x_range / bin_size) * (y_range / bin_size)
    fold_estimate = n_points / n_bins if n_bins > 0 else 0

    # Confidence based on NN distance consistency
    nn_std = float(np.std(nn_distances))
    confidence = max(0, 1 - nn_std / median_nn) if median_nn > 0 else 0

    details = {
        "nn_distance_stats": {
            "median": median_nn,
            "mean": float(np.mean(nn_distances)),
            "std": nn_std,
            "min": float(np.min(nn_distances)),
            "max": float(np.max(nn_distances)),
        },
        "raw_bin_size": raw_bin,
    }

    return BinSizeResult(
        dx=bin_size,
        dy=bin_size,
        raw_dx=raw_bin,
        raw_dy=raw_bin,
        method=BinSizeMethod.NEAREST_NEIGHBOR,
        confidence=confidence,
        fold_estimate=fold_estimate,
        coverage_ratio=min(1.0, n_points / n_bins) if n_bins > 0 else 0,
        n_points_analyzed=n_analyzed,
        detection_details=details,
        warnings=warnings,
    )


def calculate_bin_size_fft(
    mx: NDArray[np.floating],
    my: NDArray[np.floating],
    grid_resolution: float = 5.0,
) -> BinSizeResult:
    """
    Calculate optimal bin size using FFT analysis of point density.

    This method:
    1. Creates a density grid from midpoint locations
    2. Applies 2D FFT to find dominant spatial frequencies
    3. Converts frequencies to spatial periods (bin sizes)

    Best for regular acquisition geometries (orthogonal 3D).

    Args:
        mx: Midpoint X coordinates
        my: Midpoint Y coordinates
        grid_resolution: Resolution for density grid (meters)

    Returns:
        BinSizeResult with recommended bin sizes
    """
    warnings = []
    n_points = len(mx)

    if n_points < 1000:
        warnings.append("FFT method works best with large datasets, using histogram fallback")
        return calculate_bin_size_histogram(mx, my)

    # Create density grid
    x_min, x_max = mx.min(), mx.max()
    y_min, y_max = my.min(), my.max()

    x_bins = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    y_bins = np.arange(y_min, y_max + grid_resolution, grid_resolution)

    if len(x_bins) < 32 or len(y_bins) < 32:
        warnings.append("Grid too small for FFT analysis, using histogram fallback")
        return calculate_bin_size_histogram(mx, my)

    # 2D histogram = density grid
    density, _, _ = np.histogram2d(mx, my, bins=[x_bins, y_bins])

    # Apply FFT
    fft_result = np.fft.fft2(density)
    fft_power = np.abs(fft_result) ** 2

    # Shift zero frequency to center
    fft_power_shifted = np.fft.fftshift(fft_power)

    # Find peak frequencies (excluding DC component)
    center_x, center_y = fft_power_shifted.shape[0] // 2, fft_power_shifted.shape[1] // 2

    # Mask out DC and very low frequencies
    mask_radius = 3
    y_grid, x_grid = np.ogrid[:fft_power_shifted.shape[0], :fft_power_shifted.shape[1]]
    mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 > mask_radius**2

    fft_masked = fft_power_shifted * mask

    # Find peaks along X and Y axes
    x_profile = fft_masked[center_x, center_y:]
    y_profile = fft_masked[center_x:, center_y]

    # Find first significant peak
    x_peak_idx = np.argmax(x_profile[1:]) + 1 if len(x_profile) > 1 else 1
    y_peak_idx = np.argmax(y_profile[1:]) + 1 if len(y_profile) > 1 else 1

    # Convert frequency index to spatial period
    freq_x = x_peak_idx / (len(x_bins) * grid_resolution)
    freq_y = y_peak_idx / (len(y_bins) * grid_resolution)

    raw_dx = 1.0 / freq_x if freq_x > 0 else 25.0
    raw_dy = 1.0 / freq_y if freq_y > 0 else 25.0

    # Sanity check
    if raw_dx < 5 or raw_dx > 500:
        raw_dx = 25.0
        warnings.append("X frequency detection unreliable")
    if raw_dy < 5 or raw_dy > 500:
        raw_dy = 25.0
        warnings.append("Y frequency detection unreliable")

    dx = round_to_standard_bin_size(raw_dx)
    dy = round_to_standard_bin_size(raw_dy)

    # Estimate fold
    n_bins = ((x_max - x_min) / dx) * ((y_max - y_min) / dy)
    fold_estimate = n_points / n_bins if n_bins > 0 else 0

    # Confidence based on peak prominence
    x_prominence = x_profile[x_peak_idx] / x_profile.mean() if x_profile.mean() > 0 else 0
    y_prominence = y_profile[y_peak_idx] / y_profile.mean() if y_profile.mean() > 0 else 0
    confidence = min(1.0, (x_prominence + y_prominence) / 20)

    details = {
        "grid_size": (len(x_bins), len(y_bins)),
        "peak_frequencies": (freq_x, freq_y),
        "peak_prominences": (float(x_prominence), float(y_prominence)),
    }

    return BinSizeResult(
        dx=dx,
        dy=dy,
        raw_dx=raw_dx,
        raw_dy=raw_dy,
        method=BinSizeMethod.FFT,
        confidence=confidence,
        fold_estimate=fold_estimate,
        coverage_ratio=min(1.0, n_points / n_bins) if n_bins > 0 else 0,
        n_points_analyzed=n_points,
        detection_details=details,
        warnings=warnings,
    )


def auto_calculate_bin_size(
    mx: NDArray[np.floating],
    my: NDArray[np.floating],
    method: BinSizeMethod | str = BinSizeMethod.HISTOGRAM,
    azimuth_deg: float = 0.0,
) -> BinSizeResult:
    """
    Auto-calculate optimal bin size from midpoint distribution.

    This is the main entry point for bin size auto-calculation. It dispatches
    to the appropriate algorithm based on the method parameter.

    Args:
        mx: Midpoint X coordinates
        my: Midpoint Y coordinates
        method: Algorithm to use (histogram, kde, fft, nearest_neighbor)
        azimuth_deg: Acquisition azimuth for rotation (only used by histogram)

    Returns:
        BinSizeResult with recommended bin sizes and diagnostics
    """
    if isinstance(method, str):
        method = BinSizeMethod(method)

    if method == BinSizeMethod.HISTOGRAM:
        return calculate_bin_size_histogram(mx, my, azimuth_deg)
    elif method == BinSizeMethod.NEAREST_NEIGHBOR:
        return calculate_bin_size_nearest_neighbor(mx, my)
    elif method == BinSizeMethod.FFT:
        return calculate_bin_size_fft(mx, my)
    else:
        # Default to histogram
        return calculate_bin_size_histogram(mx, my, azimuth_deg)


def auto_calculate_bin_size_ensemble(
    mx: NDArray[np.floating],
    my: NDArray[np.floating],
    azimuth_deg: float = 0.0,
) -> BinSizeResult:
    """
    Calculate bin size using multiple methods and return consensus.

    Runs histogram and nearest-neighbor methods, then combines results
    weighted by their confidence scores.

    Args:
        mx: Midpoint X coordinates
        my: Midpoint Y coordinates
        azimuth_deg: Acquisition azimuth

    Returns:
        BinSizeResult with consensus recommendation
    """
    results = [
        calculate_bin_size_histogram(mx, my, azimuth_deg),
        calculate_bin_size_nearest_neighbor(mx, my),
    ]

    # Add FFT if we have enough points
    if len(mx) >= 10000:
        results.append(calculate_bin_size_fft(mx, my))

    # Weight by confidence
    total_conf = sum(r.confidence for r in results)
    if total_conf <= 0:
        # All methods failed, return histogram result
        return results[0]

    # Weighted average of raw values
    weighted_dx = sum(r.raw_dx * r.confidence for r in results) / total_conf
    weighted_dy = sum(r.raw_dy * r.confidence for r in results) / total_conf

    # Round to standard
    dx = round_to_standard_bin_size(weighted_dx)
    dy = round_to_standard_bin_size(weighted_dy)

    # Combine warnings
    all_warnings = []
    for r in results:
        all_warnings.extend(r.warnings)
    all_warnings = list(set(all_warnings))

    # Use highest confidence
    best_result = max(results, key=lambda r: r.confidence)

    details = {
        "methods_used": [r.method.value for r in results],
        "individual_results": [
            {"method": r.method.value, "dx": r.dx, "dy": r.dy, "confidence": r.confidence}
            for r in results
        ],
        "weighted_raw": {"dx": weighted_dx, "dy": weighted_dy},
    }

    return BinSizeResult(
        dx=dx,
        dy=dy,
        raw_dx=weighted_dx,
        raw_dy=weighted_dy,
        method=best_result.method,  # Report method with highest confidence
        confidence=max(r.confidence for r in results),
        fold_estimate=best_result.fold_estimate,
        coverage_ratio=best_result.coverage_ratio,
        n_points_analyzed=best_result.n_points_analyzed,
        detection_details=details,
        warnings=all_warnings,
    )
