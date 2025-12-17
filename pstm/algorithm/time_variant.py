"""
Time-Variant Sampling for PSTM Migration.

Implements time-dependent sample rate optimization where coarser sampling
is used at deeper times where high frequencies are naturally attenuated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator


@dataclass
class FrequencyTimeTable:
    """
    Table of (time, frequency) pairs defining maximum frequency vs time.

    Times must be monotonically increasing.
    Frequencies should generally decrease with time (physical reality).
    """

    times_ms: List[float] = field(default_factory=list)
    frequencies_hz: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Validate table on creation."""
        if len(self.times_ms) != len(self.frequencies_hz):
            raise ValueError("times and frequencies must have same length")
        if len(self.times_ms) < 2:
            raise ValueError("Table must have at least 2 entries")

        # Check monotonically increasing times
        for i in range(1, len(self.times_ms)):
            if self.times_ms[i] <= self.times_ms[i-1]:
                raise ValueError(f"Times must be monotonically increasing: {self.times_ms}")

        # Check positive frequencies
        for f in self.frequencies_hz:
            if f <= 0:
                raise ValueError(f"Frequencies must be positive: {self.frequencies_hz}")

    @classmethod
    def from_list(cls, pairs: List[Tuple[float, float]]) -> "FrequencyTimeTable":
        """Create from list of (time_ms, freq_hz) tuples."""
        if not pairs:
            raise ValueError("pairs list cannot be empty")

        # Sort by time
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        times = [t for t, f in sorted_pairs]
        freqs = [f for t, f in sorted_pairs]

        return cls(times_ms=times, frequencies_hz=freqs)

    def to_list(self) -> List[Tuple[float, float]]:
        """Convert to list of (time_ms, freq_hz) tuples."""
        return list(zip(self.times_ms, self.frequencies_hz))

    @classmethod
    def default(cls) -> "FrequencyTimeTable":
        """Create default table suitable for typical seismic data."""
        return cls(
            times_ms=[0, 500, 1500, 3000, 5000],
            frequencies_hz=[80, 60, 40, 25, 15],
        )

    @property
    def t_min(self) -> float:
        """Minimum time in table."""
        return self.times_ms[0]

    @property
    def t_max(self) -> float:
        """Maximum time in table."""
        return self.times_ms[-1]

    @property
    def n_entries(self) -> int:
        """Number of entries in table."""
        return len(self.times_ms)


@dataclass
class TimeWindow:
    """
    A time window with uniform effective sampling rate.

    Migration uses coarser sampling within this window based on
    the maximum frequency content at this time range.
    """

    t_start_ms: float
    t_end_ms: float
    dt_effective_ms: float
    downsample_factor: int
    sample_start: int  # Start index in output array
    sample_end: int    # End index in output array (exclusive)
    f_max_hz: float    # Maximum frequency in this window

    @property
    def n_samples(self) -> int:
        """Number of samples computed in this window."""
        return self.sample_end - self.sample_start

    @property
    def duration_ms(self) -> float:
        """Duration of window in milliseconds."""
        return self.t_end_ms - self.t_start_ms

    def __repr__(self) -> str:
        return (
            f"TimeWindow(t={self.t_start_ms:.0f}-{self.t_end_ms:.0f}ms, "
            f"dt={self.dt_effective_ms:.1f}ms, "
            f"factor={self.downsample_factor}x, "
            f"n={self.n_samples}, f_max={self.f_max_hz:.0f}Hz)"
        )


def interpolate_fmax(t_ms: float, table: FrequencyTimeTable) -> float:
    """
    Interpolate maximum frequency at given time.

    Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) to ensure:
    - Smooth transitions (C1 continuous)
    - Shape-preserving (no overshoots)
    - Monotonic if input is monotonic

    Args:
        t_ms: Time in milliseconds
        table: Frequency-time table

    Returns:
        Maximum frequency in Hz at given time
    """
    # Clamp to table bounds
    if t_ms <= table.t_min:
        return table.frequencies_hz[0]
    if t_ms >= table.t_max:
        return table.frequencies_hz[-1]

    # PCHIP interpolation
    interp = PchipInterpolator(table.times_ms, table.frequencies_hz)
    return float(interp(t_ms))


def nearest_power_of_2(n: int) -> int:
    """Round to nearest power of 2 (minimum 1)."""
    if n <= 1:
        return 1
    # Find power of 2 <= n
    p = 1
    while p * 2 <= n:
        p *= 2
    # Check if p*2 is closer
    if abs(n - p) > abs(n - p * 2):
        return p * 2
    return p


def compute_downsample_factor(
    f_max_hz: float,
    base_dt_ms: float,
    min_factor: int = 1,
    max_factor: int = 8,
) -> int:
    """
    Compute optimal downsample factor for given frequency and base dt.

    Based on Nyquist criterion: dt_max = 1000 / (2 * f_max)

    Args:
        f_max_hz: Maximum frequency to preserve
        base_dt_ms: Base sample rate in milliseconds
        min_factor: Minimum downsample factor (default 1)
        max_factor: Maximum downsample factor (default 8)

    Returns:
        Downsample factor (power of 2)
    """
    # Nyquist: need at least 2 samples per period
    # dt_max = 1000 / (2 * f_max) ms
    dt_nyquist_ms = 1000.0 / (2.0 * f_max_hz)

    # How many base samples fit in Nyquist interval?
    factor = int(dt_nyquist_ms / base_dt_ms)

    # Round to power of 2
    factor = nearest_power_of_2(factor)

    # Clamp to bounds
    factor = max(min_factor, min(max_factor, factor))

    return factor


def compute_time_windows(
    t_min_ms: float,
    t_max_ms: float,
    base_dt_ms: float,
    freq_table: FrequencyTimeTable,
    min_factor: int = 1,
    max_factor: int = 8,
) -> List[TimeWindow]:
    """
    Compute time windows with optimal sampling for migration.

    Divides the output time axis into windows where each window
    uses a constant (coarser) sample rate appropriate for the
    frequency content at that time.

    Args:
        t_min_ms: Minimum output time
        t_max_ms: Maximum output time
        base_dt_ms: Base sample rate (finest resolution)
        freq_table: Frequency-time table
        min_factor: Minimum downsample factor
        max_factor: Maximum downsample factor

    Returns:
        List of TimeWindow objects covering [t_min_ms, t_max_ms]
    """
    windows = []
    t = t_min_ms
    sample_idx = 0

    while t < t_max_ms:
        # Get frequency at this time
        f_max = interpolate_fmax(t, freq_table)

        # Compute downsample factor
        factor = compute_downsample_factor(
            f_max, base_dt_ms, min_factor, max_factor
        )

        # Effective sample rate
        dt_eff = base_dt_ms * factor

        # Find where this factor remains valid
        # (where frequency hasn't dropped enough to allow more downsampling)
        t_end = t
        while t_end < t_max_ms:
            f_check = interpolate_fmax(t_end + dt_eff, freq_table)
            factor_check = compute_downsample_factor(
                f_check, base_dt_ms, min_factor, max_factor
            )
            if factor_check != factor:
                break
            t_end += dt_eff

        # Ensure we advance at least one step
        if t_end <= t:
            t_end = min(t + dt_eff, t_max_ms)

        # Clamp to max time
        t_end = min(t_end, t_max_ms)

        # Compute number of samples in this window
        n_samples = max(1, int((t_end - t) / dt_eff))

        # Adjust t_end to align with samples
        t_end = t + n_samples * dt_eff
        if t_end > t_max_ms:
            t_end = t_max_ms
            n_samples = max(1, int((t_end - t) / dt_eff))

        window = TimeWindow(
            t_start_ms=t,
            t_end_ms=t_end,
            dt_effective_ms=dt_eff,
            downsample_factor=factor,
            sample_start=sample_idx,
            sample_end=sample_idx + n_samples,
            f_max_hz=f_max,
        )
        windows.append(window)

        # Advance
        sample_idx += n_samples
        t = t_end

    return windows


def estimate_speedup(
    t_min_ms: float,
    t_max_ms: float,
    base_dt_ms: float,
    freq_table: FrequencyTimeTable,
) -> float:
    """
    Estimate performance speedup from time-variant sampling.

    Args:
        t_min_ms: Minimum output time
        t_max_ms: Maximum output time
        base_dt_ms: Base sample rate
        freq_table: Frequency-time table

    Returns:
        Speedup factor (e.g., 3.5 means 3.5x faster)
    """
    # Uniform samples
    uniform_samples = int((t_max_ms - t_min_ms) / base_dt_ms)

    # Time-variant samples
    windows = compute_time_windows(t_min_ms, t_max_ms, base_dt_ms, freq_table)
    tv_samples = sum(w.n_samples for w in windows)

    if tv_samples == 0:
        return 1.0

    return uniform_samples / tv_samples


def create_output_sample_map(
    windows: List[TimeWindow],
    base_dt_ms: float,
    output_nt: int,
) -> NDArray[np.int32]:
    """
    Create mapping from compute samples to output samples.

    For each computed sample (at effective dt), determines which
    output sample (at base dt) it corresponds to.

    Args:
        windows: List of time windows
        base_dt_ms: Base sample rate
        output_nt: Number of output samples

    Returns:
        Array of output sample indices for each compute sample
    """
    total_compute_samples = sum(w.n_samples for w in windows)
    sample_map = np.zeros(total_compute_samples, dtype=np.int32)

    idx = 0
    for win in windows:
        for i in range(win.n_samples):
            t_ms = win.t_start_ms + i * win.dt_effective_ms
            out_idx = int(t_ms / base_dt_ms)
            out_idx = min(out_idx, output_nt - 1)
            sample_map[idx] = out_idx
            idx += 1

    return sample_map


def resample_to_uniform(
    tv_image: NDArray[np.float64],
    windows: List[TimeWindow],
    base_dt_ms: float,
    output_nt: int,
) -> NDArray[np.float64]:
    """
    Resample time-variant image to uniform sampling.

    Takes an image computed with time-variant sampling and
    resamples to uniform dt using sinc interpolation.

    Args:
        tv_image: Time-variant image (nx, ny, n_tv_samples)
        windows: Time windows used for computation
        base_dt_ms: Target uniform sample rate
        output_nt: Number of output samples

    Returns:
        Uniformly sampled image (nx, ny, output_nt)
    """
    nx, ny, n_tv = tv_image.shape
    output = np.zeros((nx, ny, output_nt), dtype=np.float64)

    # Build time axis for TV samples
    tv_times = []
    for win in windows:
        for i in range(win.n_samples):
            tv_times.append(win.t_start_ms + i * win.dt_effective_ms)
    tv_times = np.array(tv_times)

    # Output time axis
    out_times = np.arange(output_nt) * base_dt_ms

    # Simple linear interpolation for each trace
    # (Could use sinc for higher quality)
    for ix in range(nx):
        for iy in range(ny):
            if len(tv_times) > 1:
                output[ix, iy, :] = np.interp(
                    out_times, tv_times, tv_image[ix, iy, :]
                )

    return output


def get_window_info_string(windows: List[TimeWindow]) -> str:
    """Get human-readable string describing windows."""
    lines = [f"Time-Variant Sampling: {len(windows)} windows"]
    lines.append("-" * 50)

    total_samples = 0
    for i, win in enumerate(windows):
        lines.append(
            f"  Window {i+1}: {win.t_start_ms:.0f}-{win.t_end_ms:.0f}ms, "
            f"dt={win.dt_effective_ms:.1f}ms ({win.downsample_factor}x), "
            f"{win.n_samples} samples, f_max={win.f_max_hz:.0f}Hz"
        )
        total_samples += win.n_samples

    lines.append("-" * 50)
    lines.append(f"  Total compute samples: {total_samples}")

    return "\n".join(lines)
