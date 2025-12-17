"""
Unit conversion utilities for PSTM.

Provides functions for converting between common seismic units.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Conversion constants
FEET_PER_METER = 3.28084
METERS_PER_FOOT = 0.3048
MS_PER_SECOND = 1000.0
SECONDS_PER_MS = 0.001


def meters_to_feet(value: ArrayLike) -> NDArray[np.float64]:
    """
    Convert meters to feet.

    Args:
        value: Value(s) in meters

    Returns:
        Value(s) in feet
    """
    return np.asarray(value) * FEET_PER_METER


def feet_to_meters(value: ArrayLike) -> NDArray[np.float64]:
    """
    Convert feet to meters.

    Args:
        value: Value(s) in feet

    Returns:
        Value(s) in meters
    """
    return np.asarray(value) * METERS_PER_FOOT


def ms_to_s(value: ArrayLike) -> NDArray[np.float64]:
    """
    Convert milliseconds to seconds.

    Args:
        value: Value(s) in milliseconds

    Returns:
        Value(s) in seconds
    """
    return np.asarray(value) * SECONDS_PER_MS


def s_to_ms(value: ArrayLike) -> NDArray[np.float64]:
    """
    Convert seconds to milliseconds.

    Args:
        value: Value(s) in seconds

    Returns:
        Value(s) in milliseconds
    """
    return np.asarray(value) * MS_PER_SECOND


def samples_to_time(
    samples: ArrayLike,
    sample_rate_ms: float,
    start_time_ms: float = 0.0,
) -> NDArray[np.float64]:
    """
    Convert sample indices to time values.

    Args:
        samples: Sample index or indices
        sample_rate_ms: Sample rate in milliseconds
        start_time_ms: Recording start time in milliseconds

    Returns:
        Time value(s) in milliseconds
    """
    return np.asarray(samples) * sample_rate_ms + start_time_ms


def time_to_samples(
    time_ms: ArrayLike,
    sample_rate_ms: float,
    start_time_ms: float = 0.0,
) -> NDArray[np.float64]:
    """
    Convert time values to sample indices (continuous).

    Args:
        time_ms: Time value(s) in milliseconds
        sample_rate_ms: Sample rate in milliseconds
        start_time_ms: Recording start time in milliseconds

    Returns:
        Sample index/indices (may be fractional)
    """
    return (np.asarray(time_ms) - start_time_ms) / sample_rate_ms


def time_to_sample_indices(
    time_ms: ArrayLike,
    sample_rate_ms: float,
    start_time_ms: float = 0.0,
    num_samples: int | None = None,
) -> NDArray[np.int64]:
    """
    Convert time values to integer sample indices.

    Args:
        time_ms: Time value(s) in milliseconds
        sample_rate_ms: Sample rate in milliseconds
        start_time_ms: Recording start time in milliseconds
        num_samples: Maximum number of samples (for clipping)

    Returns:
        Sample indices (integer, clipped to valid range)
    """
    indices = np.floor(time_to_samples(time_ms, sample_rate_ms, start_time_ms)).astype(np.int64)

    if num_samples is not None:
        indices = np.clip(indices, 0, num_samples - 1)

    return indices


def offset_to_midpoint(
    source_x: ArrayLike,
    source_y: ArrayLike,
    receiver_x: ArrayLike,
    receiver_y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate midpoint coordinates from source and receiver positions.

    Args:
        source_x: Source X coordinates
        source_y: Source Y coordinates
        receiver_x: Receiver X coordinates
        receiver_y: Receiver Y coordinates

    Returns:
        Tuple of (midpoint_x, midpoint_y) arrays
    """
    mid_x = (np.asarray(source_x) + np.asarray(receiver_x)) / 2.0
    mid_y = (np.asarray(source_y) + np.asarray(receiver_y)) / 2.0
    return mid_x, mid_y


def compute_offset(
    source_x: ArrayLike,
    source_y: ArrayLike,
    receiver_x: ArrayLike,
    receiver_y: ArrayLike,
) -> NDArray[np.float64]:
    """
    Calculate offset (source-receiver distance).

    Args:
        source_x: Source X coordinates
        source_y: Source Y coordinates
        receiver_x: Receiver X coordinates
        receiver_y: Receiver Y coordinates

    Returns:
        Offset distances
    """
    dx = np.asarray(receiver_x) - np.asarray(source_x)
    dy = np.asarray(receiver_y) - np.asarray(source_y)
    return np.sqrt(dx**2 + dy**2)


def compute_azimuth(
    source_x: ArrayLike,
    source_y: ArrayLike,
    receiver_x: ArrayLike,
    receiver_y: ArrayLike,
) -> NDArray[np.float64]:
    """
    Calculate azimuth (source to receiver direction).

    Args:
        source_x: Source X coordinates
        source_y: Source Y coordinates
        receiver_x: Receiver X coordinates
        receiver_y: Receiver Y coordinates

    Returns:
        Azimuth in degrees (0-360, clockwise from north)
    """
    dx = np.asarray(receiver_x) - np.asarray(source_x)
    dy = np.asarray(receiver_y) - np.asarray(source_y)

    # atan2 gives angle from positive X axis, counterclockwise
    # Convert to azimuth: clockwise from north (positive Y)
    azimuth = np.degrees(np.arctan2(dx, dy))

    # Normalize to 0-360
    azimuth = np.mod(azimuth, 360.0)

    return azimuth


def apply_seg_y_scalar(
    values: ArrayLike,
    scalar: int,
) -> NDArray[np.float64]:
    """
    Apply SEG-Y coordinate scalar.

    SEG-Y convention:
    - If scalar > 0: multiply by scalar
    - If scalar < 0: divide by abs(scalar)
    - If scalar == 0: no change (treat as 1)

    Args:
        values: Coordinate values
        scalar: SEG-Y scalar value

    Returns:
        Scaled coordinate values
    """
    values = np.asarray(values, dtype=np.float64)

    if scalar > 0:
        return values * scalar
    elif scalar < 0:
        return values / abs(scalar)
    else:
        return values


def format_bytes(size_bytes: int | float) -> str:
    """
    Format byte size to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.23 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
