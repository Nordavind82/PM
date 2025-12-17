"""
Sample interpolation methods for seismic trace resampling.

Provides multiple interpolation algorithms for computing amplitude values
at non-integer sample positions during Kirchhoff migration.

Available methods:
- nearest: Nearest neighbor (fastest, lowest quality)
- linear: Linear interpolation (fast, good quality)
- cubic: Cubic spline interpolation (good quality, smooth)
- sinc4: 4-point sinc with Hanning window (balanced)
- sinc8: 8-point sinc with Hanning window (high quality)
- sinc16: 16-point sinc with Hanning window (highest quality)
- lanczos3: 3-lobe Lanczos interpolation (sharp)
- lanczos5: 5-lobe Lanczos interpolation (sharpest)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


class InterpolationMethod(str, Enum):
    """Available interpolation methods."""
    
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    SINC4 = "sinc4"
    SINC8 = "sinc8"
    SINC16 = "sinc16"
    LANCZOS3 = "lanczos3"
    LANCZOS5 = "lanczos5"
    
    @classmethod
    def from_string(cls, s: str) -> "InterpolationMethod":
        """Parse string to enum, with aliases."""
        s = s.lower().strip()
        aliases = {
            "sinc": "sinc8",
            "spline": "cubic",
            "lanczos": "lanczos3",
        }
        s = aliases.get(s, s)
        return cls(s)


# =============================================================================
# Numba-accelerated interpolation kernels
# =============================================================================


@njit(cache=True, fastmath=True, inline='always')
def _nearest_interp(trace: np.ndarray, t_sample: float) -> float:
    """Nearest neighbor interpolation."""
    n_samples = len(trace)
    i = int(t_sample + 0.5)
    if i < 0 or i >= n_samples:
        return 0.0
    return trace[i]


@njit(cache=True, fastmath=True, inline='always')
def _linear_interp(trace: np.ndarray, t_sample: float) -> float:
    """
    Linear interpolation of trace amplitude.

    Args:
        trace: Trace amplitude array
        t_sample: Fractional sample index

    Returns:
        Interpolated amplitude
    """
    n_samples = len(trace)

    if t_sample < 0.0 or t_sample >= n_samples - 1:
        return 0.0

    i0 = int(t_sample)
    i1 = i0 + 1

    if i1 >= n_samples:
        return trace[i0]

    frac = t_sample - i0
    return trace[i0] * (1.0 - frac) + trace[i1] * frac


@njit(cache=True, fastmath=True, inline='always')
def _cubic_interp(trace: np.ndarray, t_sample: float) -> float:
    """
    Cubic (Catmull-Rom) spline interpolation.
    
    Uses 4 points centered on the sample location.
    Provides C1 continuity (continuous first derivative).
    
    Args:
        trace: Trace amplitude array
        t_sample: Fractional sample index
        
    Returns:
        Interpolated amplitude
    """
    n_samples = len(trace)
    
    if t_sample < 1.0 or t_sample >= n_samples - 2:
        # Fall back to linear at edges
        return _linear_interp(trace, t_sample)
    
    i = int(t_sample)
    t = t_sample - i
    
    # Get 4 surrounding points
    p0 = trace[i - 1]
    p1 = trace[i]
    p2 = trace[i + 1]
    p3 = trace[i + 2]
    
    # Catmull-Rom coefficients
    t2 = t * t
    t3 = t2 * t
    
    # Catmull-Rom spline with tension = 0.5
    a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    a1 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
    a2 = -0.5 * p0 + 0.5 * p2
    a3 = p1
    
    return a0 * t3 + a1 * t2 + a2 * t + a3


@njit(cache=True, fastmath=True, inline='always')
def _sinc_value(x: float) -> float:
    """Compute sinc(x) = sin(pi*x) / (pi*x)."""
    if abs(x) < 1e-10:
        return 1.0
    return np.sin(np.pi * x) / (np.pi * x)


@njit(cache=True, fastmath=True, inline='always')
def _hanning_window(x: float, half_width: int) -> float:
    """Hanning window for sinc interpolation."""
    if abs(x) >= half_width:
        return 0.0
    return 0.5 * (1.0 + np.cos(np.pi * x / half_width))


@njit(cache=True, fastmath=True)
def _sinc_interp(trace: np.ndarray, t_sample: float, half_width: int) -> float:
    """
    Windowed sinc interpolation with configurable width.
    
    Args:
        trace: Trace amplitude array
        t_sample: Fractional sample index
        half_width: Half-width of sinc kernel (2, 4, 8)
        
    Returns:
        Interpolated amplitude
    """
    n_samples = len(trace)
    
    if t_sample < half_width or t_sample >= n_samples - half_width:
        # Fall back to linear near edges
        return _linear_interp(trace, t_sample)
    
    i_center = int(t_sample + 0.5)  # Nearest sample
    frac = t_sample - i_center  # Fractional offset from center
    
    result = 0.0
    norm = 0.0
    
    for k in range(-half_width + 1, half_width + 1):
        i = i_center + k
        if 0 <= i < n_samples:
            x = frac - k
            sinc_val = _sinc_value(x)
            window = _hanning_window(x, half_width)
            w = sinc_val * window
            result += trace[i] * w
            norm += w
    
    if norm > 1e-10:
        return result / norm
    return 0.0


@njit(cache=True, fastmath=True, inline='always')
def _sinc4_interp(trace: np.ndarray, t_sample: float) -> float:
    """4-point sinc interpolation."""
    return _sinc_interp(trace, t_sample, 2)


@njit(cache=True, fastmath=True, inline='always')
def _sinc8_interp(trace: np.ndarray, t_sample: float) -> float:
    """8-point sinc interpolation."""
    return _sinc_interp(trace, t_sample, 4)


@njit(cache=True, fastmath=True, inline='always')
def _sinc16_interp(trace: np.ndarray, t_sample: float) -> float:
    """16-point sinc interpolation."""
    return _sinc_interp(trace, t_sample, 8)


@njit(cache=True, fastmath=True, inline='always')
def _lanczos_kernel(x: float, a: int) -> float:
    """Lanczos kernel L(x) = sinc(x) * sinc(x/a) for |x| < a."""
    if abs(x) < 1e-10:
        return 1.0
    if abs(x) >= a:
        return 0.0
    return _sinc_value(x) * _sinc_value(x / a)


@njit(cache=True, fastmath=True)
def _lanczos_interp(trace: np.ndarray, t_sample: float, a: int) -> float:
    """
    Lanczos interpolation.
    
    Args:
        trace: Trace amplitude array
        t_sample: Fractional sample index
        a: Lanczos parameter (number of lobes, typically 2, 3, or 5)
        
    Returns:
        Interpolated amplitude
    """
    n_samples = len(trace)
    
    if t_sample < a or t_sample >= n_samples - a:
        return _linear_interp(trace, t_sample)
    
    i_center = int(t_sample)
    frac = t_sample - i_center
    
    result = 0.0
    norm = 0.0
    
    for k in range(-a + 1, a + 1):
        i = i_center + k
        if 0 <= i < n_samples:
            x = frac - k
            w = _lanczos_kernel(x, a)
            result += trace[i] * w
            norm += w
    
    if norm > 1e-10:
        return result / norm
    return 0.0


@njit(cache=True, fastmath=True, inline='always')
def _lanczos3_interp(trace: np.ndarray, t_sample: float) -> float:
    """3-lobe Lanczos interpolation."""
    return _lanczos_interp(trace, t_sample, 3)


@njit(cache=True, fastmath=True, inline='always')
def _lanczos5_interp(trace: np.ndarray, t_sample: float) -> float:
    """5-lobe Lanczos interpolation."""
    return _lanczos_interp(trace, t_sample, 5)


# =============================================================================
# Dispatcher function
# =============================================================================


@njit(cache=True, fastmath=True)
def interpolate_sample(
    trace: np.ndarray,
    t_sample: float,
    method: int,
) -> float:
    """
    Interpolate trace amplitude at fractional sample index.
    
    Args:
        trace: Trace amplitude array
        t_sample: Fractional sample index
        method: Interpolation method code:
            0 = nearest
            1 = linear
            2 = cubic
            3 = sinc4
            4 = sinc8
            5 = sinc16
            6 = lanczos3
            7 = lanczos5
            
    Returns:
        Interpolated amplitude
    """
    if method == 0:
        return _nearest_interp(trace, t_sample)
    elif method == 1:
        return _linear_interp(trace, t_sample)
    elif method == 2:
        return _cubic_interp(trace, t_sample)
    elif method == 3:
        return _sinc4_interp(trace, t_sample)
    elif method == 4:
        return _sinc8_interp(trace, t_sample)
    elif method == 5:
        return _sinc16_interp(trace, t_sample)
    elif method == 6:
        return _lanczos3_interp(trace, t_sample)
    elif method == 7:
        return _lanczos5_interp(trace, t_sample)
    else:
        # Default to linear
        return _linear_interp(trace, t_sample)


def get_method_code(method: str | InterpolationMethod) -> int:
    """
    Convert method name to numeric code for Numba.
    
    Args:
        method: Method name or enum
        
    Returns:
        Integer code for interpolate_sample()
    """
    if isinstance(method, InterpolationMethod):
        method = method.value
    
    method = method.lower().strip()
    
    codes = {
        "nearest": 0,
        "linear": 1,
        "cubic": 2,
        "spline": 2,
        "sinc4": 3,
        "sinc8": 4,
        "sinc": 4,  # Default sinc = sinc8
        "sinc16": 5,
        "lanczos3": 6,
        "lanczos": 6,  # Default lanczos = lanczos3
        "lanczos5": 7,
    }
    
    return codes.get(method, 1)  # Default to linear


# =============================================================================
# Vectorized interpolation for batch processing
# =============================================================================


@njit(cache=True, parallel=True, fastmath=True)
def interpolate_traces_batch(
    traces: np.ndarray,  # (n_traces, n_samples)
    t_samples: np.ndarray,  # (n_traces,) fractional sample indices
    method: int,
) -> np.ndarray:
    """
    Batch interpolate multiple traces in parallel.
    
    Args:
        traces: 2D array of traces (n_traces, n_samples)
        t_samples: Array of fractional sample indices
        method: Interpolation method code
        
    Returns:
        Array of interpolated amplitudes
    """
    n_traces = traces.shape[0]
    result = np.zeros(n_traces, dtype=np.float64)
    
    for i in prange(n_traces):
        result[i] = interpolate_sample(traces[i], t_samples[i], method)
    
    return result


# =============================================================================
# Information functions
# =============================================================================


def get_available_methods() -> list[str]:
    """Get list of available interpolation methods."""
    return [m.value for m in InterpolationMethod]


def get_method_info(method: str) -> dict:
    """
    Get information about an interpolation method.
    
    Args:
        method: Method name
        
    Returns:
        Dictionary with method properties
    """
    info = {
        "nearest": {
            "name": "Nearest Neighbor",
            "points": 1,
            "edge_samples": 0,
            "quality": "low",
            "speed": "fastest",
            "description": "Uses nearest sample value. Fast but introduces aliasing.",
        },
        "linear": {
            "name": "Linear",
            "points": 2,
            "edge_samples": 1,
            "quality": "medium",
            "speed": "fast",
            "description": "Linear interpolation between adjacent samples. Good balance of speed and quality.",
        },
        "cubic": {
            "name": "Cubic Spline (Catmull-Rom)",
            "points": 4,
            "edge_samples": 2,
            "quality": "good",
            "speed": "medium",
            "description": "Catmull-Rom cubic spline. Smooth with continuous first derivative.",
        },
        "sinc4": {
            "name": "4-point Sinc",
            "points": 4,
            "edge_samples": 2,
            "quality": "good",
            "speed": "medium",
            "description": "4-point windowed sinc. Good frequency response.",
        },
        "sinc8": {
            "name": "8-point Sinc",
            "points": 8,
            "edge_samples": 4,
            "quality": "high",
            "speed": "medium",
            "description": "8-point windowed sinc. Excellent frequency response, minimal aliasing.",
        },
        "sinc16": {
            "name": "16-point Sinc",
            "points": 16,
            "edge_samples": 8,
            "quality": "highest",
            "speed": "slow",
            "description": "16-point windowed sinc. Best frequency response, most accurate.",
        },
        "lanczos3": {
            "name": "Lanczos-3",
            "points": 6,
            "edge_samples": 3,
            "quality": "high",
            "speed": "medium",
            "description": "3-lobe Lanczos. Sharp reconstruction with minimal ringing.",
        },
        "lanczos5": {
            "name": "Lanczos-5",
            "points": 10,
            "edge_samples": 5,
            "quality": "highest",
            "speed": "slow",
            "description": "5-lobe Lanczos. Sharpest reconstruction.",
        },
    }
    
    method = method.lower().strip()
    return info.get(method, info["linear"])


def recommend_method(
    priority: Literal["speed", "quality", "balanced"] = "balanced",
    data_frequency_hz: float | None = None,
    sample_rate_ms: float | None = None,
) -> str:
    """
    Recommend interpolation method based on requirements.
    
    Args:
        priority: Optimization priority
        data_frequency_hz: Dominant frequency of seismic data
        sample_rate_ms: Sample interval in ms
        
    Returns:
        Recommended method name
    """
    if priority == "speed":
        return "linear"
    elif priority == "quality":
        return "sinc8"
    else:
        # Balanced - check Nyquist if frequency info available
        if data_frequency_hz is not None and sample_rate_ms is not None:
            nyquist_hz = 500.0 / sample_rate_ms
            freq_ratio = data_frequency_hz / nyquist_hz
            
            if freq_ratio > 0.7:
                # High frequency content - need better interpolation
                return "sinc8"
            elif freq_ratio > 0.4:
                return "cubic"
            else:
                return "linear"
        
        return "sinc4"  # Default balanced choice
