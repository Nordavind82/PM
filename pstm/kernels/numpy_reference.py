"""
NumPy reference kernel for PSTM.

This is a pure NumPy implementation used for:
- Correctness verification of optimized kernels
- Fallback when Numba is not available
- Educational/debugging purposes

Performance is ~100x slower than Numba but guaranteed correct.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pstm.kernels.base import (
    KernelCapability,
    KernelConfig,
    KernelMetrics,
    OutputTile,
    TraceBlock,
    VelocitySlice,
)
from pstm.utils.logging import get_logger

logger = get_logger(__name__)


def _dsr_travel_time_numpy(
    sx: NDArray,
    sy: NDArray,
    rx: NDArray,
    ry: NDArray,
    ox: float,
    oy: float,
    t0_s: float,
    vrms: float,
) -> NDArray:
    """
    Vectorized DSR travel time computation.

    Args:
        sx, sy: Source coordinates (arrays)
        rx, ry: Receiver coordinates (arrays)
        ox, oy: Output point coordinates (scalars)
        t0_s: Zero-offset time in seconds
        vrms: RMS velocity in m/s

    Returns:
        Travel times for all traces
    """
    # Source-to-output distance squared
    ds2 = (ox - sx) ** 2 + (oy - sy) ** 2

    # Receiver-to-output distance squared
    dr2 = (ox - rx) ** 2 + (oy - ry) ** 2

    t0_half = t0_s / 2.0
    t0_half_sq = t0_half ** 2
    v_sq = vrms ** 2

    # Source leg
    t_src = np.sqrt(t0_half_sq + ds2 / v_sq)

    # Receiver leg
    t_rec = np.sqrt(t0_half_sq + dr2 / v_sq)

    return t_src + t_rec


def _linear_interp_numpy(
    traces: NDArray,
    t_samples: NDArray,
) -> NDArray:
    """
    Vectorized linear interpolation.

    Args:
        traces: Trace data (n_traces, n_samples)
        t_samples: Sample indices (n_traces,) - may be fractional

    Returns:
        Interpolated amplitudes (n_traces,)
    """
    n_traces, n_samples = traces.shape

    # Clamp to valid range
    t_samples = np.clip(t_samples, 0, n_samples - 1.001)

    i0 = np.floor(t_samples).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_samples - 1)
    frac = t_samples - i0

    # Gather amplitudes
    amp0 = traces[np.arange(n_traces), i0]
    amp1 = traces[np.arange(n_traces), i1]

    return amp0 * (1.0 - frac) + amp1 * frac


def _compute_aperture_numpy(
    t_ms: float,
    vrms: float,
    max_dip_deg: float,
    min_aperture: float,
    max_aperture: float,
) -> float:
    """Compute time-dependent aperture."""
    t_s = t_ms / 1000.0
    tan_dip = np.tan(np.radians(max_dip_deg))
    aperture = vrms * t_s * 0.5 * tan_dip
    return float(np.clip(aperture, min_aperture, max_aperture))


def _compute_taper_numpy(
    distances: NDArray,
    aperture: float,
    taper_fraction: float,
) -> NDArray:
    """Compute aperture taper weights."""
    taper_start = aperture * (1.0 - taper_fraction)

    weights = np.ones_like(distances)

    # Taper region
    taper_mask = (distances > taper_start) & (distances <= aperture)
    if np.any(taper_mask):
        x = (distances[taper_mask] - taper_start) / (aperture - taper_start)
        weights[taper_mask] = 0.5 * (1.0 + np.cos(np.pi * x))

    # Outside aperture
    weights[distances > aperture] = 0.0

    return weights


class NumpyReferenceKernel:
    """
    Pure NumPy reference implementation.

    Slow but guaranteed correct - use for validation.
    """

    def __init__(self):
        self._config: KernelConfig | None = None
        self._initialized: bool = False

    @property
    def name(self) -> str:
        return "numpy_reference"

    @property
    def capabilities(self) -> set[KernelCapability]:
        return {KernelCapability.FP64}

    def initialize(self, config: KernelConfig) -> None:
        """Initialize kernel."""
        self._config = config
        self._initialized = True
        logger.info("NumPy reference kernel initialized")

    def migrate_tile(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: "KernelConfig | None" = None,
    ) -> KernelMetrics:
        """
        Migrate traces using pure NumPy.

        This is the reference implementation for correctness verification.
        """
        if not self._initialized or self._config is None:
            raise RuntimeError("Kernel not initialized")

        # Use provided config or fall back to initialized config
        active_config = config if config is not None else self._config

        start_time = time.perf_counter()

        # Get velocity
        if not velocity.is_1d:
            vrms_1d = velocity.get_velocity_1d()
        else:
            vrms_1d = velocity.vrms

        # Interpolate velocity to output times if needed
        if len(vrms_1d) != output.nt:
            vel_t = velocity.t_coords_ms if velocity.t_coords_ms is not None else output.t_coords_ms
            vrms_1d = np.interp(output.t_coords_ms, vel_t, vrms_1d)

        # Get trace weights
        trace_weights = traces.weights
        if trace_weights is None:
            trace_weights = np.ones(traces.n_traces, dtype=np.float64)

        # Compute distances from all traces to all output points
        # This is memory-intensive but clear

        n_contributions = 0

        # Loop over output points (not parallelized in reference impl)
        for ix, ox in enumerate(output.x_coords):
            for iy, oy in enumerate(output.y_coords):
                # Distance from output point to all trace midpoints
                distances = np.sqrt(
                    (ox - traces.midpoint_x) ** 2 + (oy - traces.midpoint_y) ** 2
                )

                # Quick rejection: traces outside max aperture
                max_dist_mask = distances <= active_config.max_aperture_m

                if not np.any(max_dist_mask):
                    continue

                # Update fold (count traces in aperture)
                output.fold[ix, iy] += np.sum(max_dist_mask)

                # Loop over output times
                for iot, t0_ms in enumerate(output.t_coords_ms):
                    t0_s = t0_ms / 1000.0
                    vrms = vrms_1d[iot]

                    # Time-dependent aperture
                    aperture = _compute_aperture_numpy(
                        t0_ms,
                        vrms,
                        active_config.max_dip_degrees,
                        active_config.min_aperture_m,
                        active_config.max_aperture_m,
                    )

                    # Mask for traces within aperture
                    in_aperture = distances <= aperture

                    if not np.any(in_aperture):
                        continue

                    # Get indices of traces in aperture
                    trace_idx = np.where(in_aperture)[0]
                    n_valid = len(trace_idx)
                    n_contributions += n_valid

                    # Compute DSR travel times
                    t_total = _dsr_travel_time_numpy(
                        traces.source_x[trace_idx],
                        traces.source_y[trace_idx],
                        traces.receiver_x[trace_idx],
                        traces.receiver_y[trace_idx],
                        ox,
                        oy,
                        t0_s,
                        vrms,
                    )

                    # Convert to sample indices
                    t_total_ms = t_total * 1000.0
                    t_samples = (t_total_ms - traces.start_time_ms) / traces.sample_rate_ms

                    # Filter valid sample indices
                    valid_samples = (t_samples >= 0) & (t_samples < traces.n_samples - 1)

                    if not np.any(valid_samples):
                        continue

                    # Apply sample mask
                    valid_idx = trace_idx[valid_samples]
                    valid_t_samples = t_samples[valid_samples]
                    valid_t_total = t_total[valid_samples]
                    valid_distances = distances[valid_idx]

                    # Interpolate amplitudes
                    amps = _linear_interp_numpy(
                        traces.amplitudes[valid_idx],
                        valid_t_samples,
                    )

                    # Compute weights
                    weights = trace_weights[valid_idx].copy()

                    # Aperture taper
                    taper = _compute_taper_numpy(
                        valid_distances,
                        aperture,
                        active_config.taper_fraction,
                    )
                    weights *= taper

                    # Geometrical spreading
                    if active_config.apply_spreading:
                        spreading = 1.0 / (vrms * valid_t_total + 1e-10)
                        weights *= spreading

                    # Obliquity factor
                    if active_config.apply_obliquity:
                        obliquity = t0_s / (valid_t_total + 1e-10)
                        weights *= obliquity

                    # Accumulate
                    output.image[ix, iy, iot] += np.sum(amps * weights)

        elapsed = time.perf_counter() - start_time

        return KernelMetrics(
            n_traces_processed=traces.n_traces,
            n_samples_output=output.nx * output.ny * output.nt,
            n_contributions=n_contributions,
            elapsed_time_s=elapsed,
        )

    def synchronize(self) -> None:
        """No-op for CPU kernel."""
        pass

    def cleanup(self) -> None:
        """Clean up kernel."""
        self._initialized = False
        self._config = None


def create_numpy_kernel() -> NumpyReferenceKernel:
    """Factory function to create NumPy reference kernel."""
    return NumpyReferenceKernel()
