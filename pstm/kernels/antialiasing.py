"""
Anti-aliasing filter implementation for PSTM.

Provides triangle filter banks and offset-dependent filtering
to prevent operator aliasing at far offsets.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from pstm.utils.logging import get_logger

logger = get_logger(__name__)


def compute_fmax_gray(
    offset: float,
    vrms: float,
    t_ms: float,
    dx: float,
) -> float:
    """
    Compute maximum unaliased frequency using Gray's formula.

    f_max = v / (4 * dx * sin(theta))

    where theta is the emergence angle: sin(theta) ≈ x / (v * t)

    For prestack: x = offset/2 (half-offset)

    Args:
        offset: Source-receiver offset in meters
        vrms: RMS velocity in m/s
        t_ms: Two-way time in milliseconds
        dx: Output grid spacing in meters

    Returns:
        Maximum unaliased frequency in Hz
    """
    t_s = t_ms / 1000.0

    if t_s < 0.001:
        return np.inf

    # Half-offset
    half_offset = offset / 2.0

    # Emergence angle sine: sin(theta) = x / (v * t / 2) for one-way
    # Using v * t for two-way simplifies to: sin(theta) ≈ offset / (v * t)
    sin_theta = half_offset / (vrms * t_s / 2.0)

    # Clamp to valid range
    sin_theta = min(abs(sin_theta), 0.999)

    if sin_theta < 0.001:
        return np.inf

    # Gray's formula
    f_max = vrms / (4.0 * dx * sin_theta)

    return f_max


def compute_fmax_array(
    offsets: NDArray[np.float64],
    vrms: float,
    t_ms: float,
    dx: float,
) -> NDArray[np.float64]:
    """
    Compute f_max for array of offsets.

    Args:
        offsets: Offset array in meters
        vrms: RMS velocity in m/s
        t_ms: Two-way time in milliseconds
        dx: Output grid spacing in meters

    Returns:
        Array of f_max values in Hz
    """
    f_max = np.empty_like(offsets)

    for i, offset in enumerate(offsets):
        f_max[i] = compute_fmax_gray(offset, vrms, t_ms, dx)

    return f_max


class TriangleFilterBank:
    """
    Bank of triangle (running average) filters for anti-aliasing.

    Creates a set of filters with progressively lower cutoff frequencies.
    """

    def __init__(
        self,
        n_filters: int,
        f_min_hz: float,
        f_max_hz: float,
        sample_rate_ms: float,
        max_length: int = 64,
    ):
        """
        Initialize filter bank.

        Args:
            n_filters: Number of filters in bank
            f_min_hz: Minimum cutoff frequency
            f_max_hz: Maximum cutoff frequency
            sample_rate_ms: Sample rate in milliseconds
            max_length: Maximum filter length in samples
        """
        self.n_filters = n_filters
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.sample_rate_ms = sample_rate_ms
        self.sample_rate_hz = 1000.0 / sample_rate_ms

        # Compute cutoff frequencies (logarithmically spaced)
        self.cutoffs_hz = np.logspace(
            np.log10(f_min_hz),
            np.log10(f_max_hz),
            n_filters,
        )

        # Build filters
        self.filters: list[NDArray[np.float32]] = []
        self.filter_lengths: NDArray[np.int32] = np.empty(n_filters, dtype=np.int32)

        for i, f_cut in enumerate(self.cutoffs_hz):
            # Triangle filter length based on cutoff
            # Length ≈ sample_rate / (2 * f_cut)
            length = int(self.sample_rate_hz / (2 * f_cut))
            length = max(1, min(length, max_length))

            # Make odd for symmetry
            if length % 2 == 0:
                length += 1

            # Create triangle filter
            half = length // 2
            filt = np.zeros(length, dtype=np.float32)
            for j in range(length):
                filt[j] = 1.0 - abs(j - half) / (half + 1)

            # Normalize
            filt /= filt.sum()

            self.filters.append(filt)
            self.filter_lengths[i] = length

        logger.info(
            f"Created triangle filter bank: {n_filters} filters, "
            f"f=[{f_min_hz:.0f}, {f_max_hz:.0f}] Hz, "
            f"lengths=[{self.filter_lengths.min()}, {self.filter_lengths.max()}]"
        )

    def get_filter_index(self, f_max: float) -> int:
        """
        Get filter index for given maximum frequency.

        Args:
            f_max: Maximum unaliased frequency in Hz

        Returns:
            Filter index (0 = most aggressive, n_filters-1 = least filtering)
        """
        if f_max >= self.f_max_hz:
            return self.n_filters - 1  # No filtering needed

        if f_max <= self.f_min_hz:
            return 0  # Maximum filtering

        # Find appropriate filter
        idx = np.searchsorted(self.cutoffs_hz, f_max)
        return min(idx, self.n_filters - 1)

    def get_filter_indices(self, f_max_array: NDArray[np.float64]) -> NDArray[np.int32]:
        """
        Get filter indices for array of f_max values.

        Args:
            f_max_array: Array of maximum frequencies

        Returns:
            Array of filter indices
        """
        indices = np.searchsorted(self.cutoffs_hz, f_max_array)
        return np.clip(indices, 0, self.n_filters - 1).astype(np.int32)

    def apply_filter(self, trace: NDArray[np.float32], filter_idx: int) -> NDArray[np.float32]:
        """
        Apply filter to trace.

        Args:
            trace: Input trace
            filter_idx: Filter index

        Returns:
            Filtered trace
        """
        if filter_idx >= self.n_filters - 1:
            return trace  # No filtering

        filt = self.filters[filter_idx]
        return np.convolve(trace, filt, mode="same").astype(np.float32)

    def to_padded_array(self) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """
        Convert filter bank to padded 2D array for kernel use.

        Returns:
            Tuple of (filter_coefficients, filter_lengths)
            filter_coefficients shape: (n_filters, max_length)
        """
        max_len = max(len(f) for f in self.filters)
        coeffs = np.zeros((self.n_filters, max_len), dtype=np.float32)

        for i, filt in enumerate(self.filters):
            coeffs[i, : len(filt)] = filt

        return coeffs, self.filter_lengths


@njit(cache=True)
def apply_triangle_filter_inline(
    trace: np.ndarray,
    filter_length: int,
    output: np.ndarray,
) -> None:
    """
    Apply triangle filter inline (Numba JIT).

    Args:
        trace: Input trace
        filter_length: Filter length (odd)
        output: Output array (same length as trace)
    """
    n = len(trace)
    half = filter_length // 2

    for i in range(n):
        acc = 0.0
        weight_sum = 0.0

        for j in range(-half, half + 1):
            idx = i + j
            if 0 <= idx < n:
                # Triangle weight
                w = 1.0 - abs(j) / (half + 1)
                acc += trace[idx] * w
                weight_sum += w

        if weight_sum > 0:
            output[i] = acc / weight_sum
        else:
            output[i] = trace[i]


class OffsetSectorManager:
    """
    Manager for offset-based anti-aliasing sectoring.

    Divides traces into offset ranges and applies appropriate
    filtering to each sector.
    """

    def __init__(
        self,
        offset_bins: list[tuple[float, float]],
        f_max_per_bin: list[float],
    ):
        """
        Initialize offset sector manager.

        Args:
            offset_bins: List of (min_offset, max_offset) tuples
            f_max_per_bin: Maximum frequency for each bin
        """
        self.offset_bins = offset_bins
        self.f_max_per_bin = f_max_per_bin
        self.n_sectors = len(offset_bins)

    @classmethod
    def create_automatic(
        cls,
        min_offset: float,
        max_offset: float,
        n_sectors: int,
        vrms_typical: float,
        t_max_ms: float,
        dx: float,
    ) -> "OffsetSectorManager":
        """
        Create offset sectors automatically.

        Args:
            min_offset: Minimum offset in data
            max_offset: Maximum offset in data
            n_sectors: Number of sectors to create
            vrms_typical: Typical RMS velocity
            t_max_ms: Maximum time (for f_max calculation)
            dx: Output grid spacing

        Returns:
            Configured OffsetSectorManager
        """
        # Create linearly spaced offset bins
        edges = np.linspace(min_offset, max_offset, n_sectors + 1)
        bins = [(edges[i], edges[i + 1]) for i in range(n_sectors)]

        # Compute f_max for center of each bin
        f_max_list = []
        for omin, omax in bins:
            center_offset = (omin + omax) / 2
            f_max = compute_fmax_gray(center_offset, vrms_typical, t_max_ms, dx)
            f_max_list.append(f_max)

        return cls(bins, f_max_list)

    def get_sector(self, offset: float) -> int:
        """
        Get sector index for given offset.

        Args:
            offset: Offset value

        Returns:
            Sector index
        """
        for i, (omin, omax) in enumerate(self.offset_bins):
            if omin <= offset < omax:
                return i

        # Default to last sector for offsets beyond range
        return self.n_sectors - 1

    def get_sectors(self, offsets: NDArray[np.float64]) -> NDArray[np.int32]:
        """
        Get sector indices for array of offsets.

        Args:
            offsets: Offset array

        Returns:
            Array of sector indices
        """
        sectors = np.empty(len(offsets), dtype=np.int32)
        for i, offset in enumerate(offsets):
            sectors[i] = self.get_sector(offset)
        return sectors
