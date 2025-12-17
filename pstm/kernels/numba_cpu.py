"""
Numba CPU migration kernel for PSTM.

This is the primary compute kernel, optimized for multi-core CPU execution.
Uses Numba JIT compilation with parallel loops.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from pstm.kernels.base import (
    KernelCapability,
    KernelConfig,
    KernelMetrics,
    MigrationKernel,
    OutputTile,
    TraceBlock,
    VelocitySlice,
)
from pstm.kernels.interpolation import (
    get_method_code,
    interpolate_sample,
    _linear_interp,
    _sinc8_interp,
    _cubic_interp,
    _lanczos3_interp,
)
from pstm.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# =============================================================================
# Low-level Numba JIT functions
# =============================================================================


@njit(cache=True, fastmath=True)
def _dsr_travel_time(
    sx: float,
    sy: float,
    rx: float,
    ry: float,
    ox: float,
    oy: float,
    t0_s: float,
    vrms: float,
) -> float:
    """
    Compute double-square-root (DSR) travel time for prestack migration.

    DSR equation:
        t = sqrt(t0^2 + (4/v^2) * ((ox-sx)^2 + (oy-sy)^2)) / 2
          + sqrt(t0^2 + (4/v^2) * ((ox-rx)^2 + (oy-ry)^2)) / 2

    This is the correct prestack formulation where:
    - t0 is the zero-offset two-way time
    - We compute source-to-image and image-to-receiver times separately

    Args:
        sx, sy: Source coordinates
        rx, ry: Receiver coordinates
        ox, oy: Output image point coordinates
        t0_s: Zero-offset two-way time in seconds
        vrms: RMS velocity in m/s

    Returns:
        Two-way travel time in seconds
    """
    # Source to image point distance squared
    ds2 = (ox - sx) ** 2 + (oy - sy) ** 2

    # Receiver to image point distance squared
    dr2 = (ox - rx) ** 2 + (oy - ry) ** 2

    # Velocity squared factor: (2/v)^2 = 4/v^2
    # But we use t0 as two-way time, so the vertical component is (t0/2)^2
    # Actually for DSR: t = t_source + t_receiver
    # where t_source = sqrt((t0/2)^2 + ds2/v^2)
    # and t_receiver = sqrt((t0/2)^2 + dr2/v^2)

    t0_half = t0_s * 0.5
    t0_half_sq = t0_half * t0_half
    v_sq = vrms * vrms

    # Source leg travel time
    t_src = np.sqrt(t0_half_sq + ds2 / v_sq)

    # Receiver leg travel time
    t_rec = np.sqrt(t0_half_sq + dr2 / v_sq)

    return t_src + t_rec


# Note: Interpolation functions (_linear_interp, _sinc8_interp, etc.)
# are now imported from pstm.kernels.interpolation module


@njit(cache=True, fastmath=True)
def _compute_spreading_weight(t_s: float, vrms: float) -> float:
    """
    Compute geometrical spreading correction weight.

    Weight = 1 / (v * t) to compensate for amplitude decay.

    Args:
        t_s: Travel time in seconds
        vrms: RMS velocity in m/s

    Returns:
        Spreading weight
    """
    if t_s < 0.001:  # Avoid division by very small times
        return 0.0
    return 1.0 / (vrms * t_s)


@njit(cache=True, fastmath=True)
def _compute_obliquity_weight(
    t0_s: float,
    t_total_s: float,
) -> float:
    """
    Compute obliquity (cosine) factor.

    This accounts for the angle of incidence.
    Weight = t0 / t_total = cos(angle)

    Args:
        t0_s: Zero-offset time in seconds
        t_total_s: Total travel time in seconds

    Returns:
        Obliquity weight
    """
    if t_total_s < 0.001:
        return 0.0
    return t0_s / t_total_s


@njit(cache=True, fastmath=True)
def _compute_taper_weight(
    distance: float,
    aperture_radius: float,
    taper_fraction: float,
) -> float:
    """
    Compute cosine taper weight at aperture edge.

    Args:
        distance: Distance from output point to trace midpoint
        aperture_radius: Migration aperture radius
        taper_fraction: Fraction of aperture to taper

    Returns:
        Taper weight (0 to 1)
    """
    if distance >= aperture_radius:
        return 0.0

    taper_start = aperture_radius * (1.0 - taper_fraction)

    if distance <= taper_start:
        return 1.0

    # Cosine taper
    x = (distance - taper_start) / (aperture_radius - taper_start)
    return 0.5 * (1.0 + np.cos(np.pi * x))


@njit(cache=True, fastmath=True)
def _compute_aperture(
    t_ms: float,
    vrms: float,
    max_dip_deg: float,
    min_aperture: float,
    max_aperture: float,
) -> float:
    """
    Compute time-dependent aperture radius.

    Args:
        t_ms: Two-way time in milliseconds
        vrms: RMS velocity in m/s
        max_dip_deg: Maximum dip angle in degrees
        min_aperture: Minimum aperture radius
        max_aperture: Maximum aperture radius

    Returns:
        Aperture radius in meters
    """
    t_s = t_ms / 1000.0
    tan_dip = np.tan(max_dip_deg * np.pi / 180.0)

    # Aperture = depth * tan(dip) = v * t/2 * tan(dip)
    aperture = vrms * t_s * 0.5 * tan_dip

    if aperture < min_aperture:
        return min_aperture
    if aperture > max_aperture:
        return max_aperture
    return aperture


# =============================================================================
# Main migration kernel (parallelized over output points)
# =============================================================================


@njit(parallel=True, cache=True, fastmath=True)
def _migrate_tile_kernel(
    # Trace data
    amplitudes: np.ndarray,  # (n_traces, n_samples_in)
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    midpoint_x: np.ndarray,
    midpoint_y: np.ndarray,
    trace_weights: np.ndarray,  # (n_traces,)
    # Timing
    dt_in_ms: float,
    t_start_in_ms: float,
    # Output grid
    image: np.ndarray,  # (nx, ny, nt) - accumulator
    fold: np.ndarray,  # (nx, ny) - fold counter
    ox_coords: np.ndarray,  # (nx,)
    oy_coords: np.ndarray,  # (ny,)
    ot_coords_ms: np.ndarray,  # (nt,)
    # Velocity (1D or 3D)
    vrms_1d: np.ndarray,  # (nt,)
    # Algorithm parameters
    max_dip_deg: float,
    min_aperture: float,
    max_aperture: float,
    taper_fraction: float,
    apply_spreading: bool,
    apply_obliquity: bool,
    interp_method: int,  # 0=nearest, 1=linear, 2=cubic, 3=sinc4, 4=sinc8, etc.
) -> int:
    """
    Core migration kernel with parallel loop over ALL output pillars.

    Uses flattened loop over nx*ny for maximum parallelism (e.g., 1024 work units
    for 32x32 tile instead of just 32).

    Interpolation methods:
        0 = nearest neighbor
        1 = linear (default)
        2 = cubic spline
        3 = sinc4 (4-point)
        4 = sinc8 (8-point)
        5 = sinc16 (16-point)
        6 = lanczos3
        7 = lanczos5

    Returns:
        Total number of trace contributions
    """
    n_traces = amplitudes.shape[0]
    n_samples_in = amplitudes.shape[1]
    nx = len(ox_coords)
    ny = len(oy_coords)
    nt = len(ot_coords_ms)
    n_pillars = nx * ny  # Total output pillars (e.g., 1024 for 32x32)

    total_contributions = 0

    # FLATTENED parallel loop over ALL output pillars
    # This gives nx*ny parallel work units instead of just nx
    # For 32x32 tile: 1024 parallel work units instead of 32!
    for idx in prange(n_pillars):
        # Convert flat index to 2D coordinates
        ix = idx // ny
        iy = idx % ny

        ox = ox_coords[ix]
        oy = oy_coords[iy]

        local_fold = 0

        # Loop over all traces
        for it in range(n_traces):
            # Get trace geometry
            sx = source_x[it]
            sy = source_y[it]
            rx = receiver_x[it]
            ry = receiver_y[it]
            mx = midpoint_x[it]
            my = midpoint_y[it]
            w_trace = trace_weights[it]

            # Distance from output point to trace midpoint
            dist = np.sqrt((ox - mx) ** 2 + (oy - my) ** 2)

            # Check if trace is within maximum aperture
            if dist > max_aperture:
                continue

            # This trace contributes - increment fold once
            local_fold += 1

            # Loop over output times
            for iot in range(nt):
                t0_ms = ot_coords_ms[iot]
                t0_s = t0_ms / 1000.0

                # Get velocity at this time
                vrms = vrms_1d[iot]

                # Compute time-dependent aperture
                aperture = _compute_aperture(
                    t0_ms, vrms, max_dip_deg, min_aperture, max_aperture
                )

                # Check aperture
                if dist > aperture:
                    continue

                # Compute DSR travel time
                t_total_s = _dsr_travel_time(sx, sy, rx, ry, ox, oy, t0_s, vrms)
                t_total_ms = t_total_s * 1000.0

                # Convert to input sample index
                t_sample = (t_total_ms - t_start_in_ms) / dt_in_ms

                # Check bounds
                if t_sample < 0.0 or t_sample >= n_samples_in - 1:
                    continue

                # Interpolate trace amplitude
                amp = interpolate_sample(amplitudes[it], t_sample, interp_method)

                # Compute weights
                weight = w_trace

                # Aperture taper
                taper = _compute_taper_weight(dist, aperture, taper_fraction)
                weight *= taper

                # Geometrical spreading
                if apply_spreading:
                    spreading = _compute_spreading_weight(t_total_s, vrms)
                    weight *= spreading

                # Obliquity factor
                if apply_obliquity:
                    obliquity = _compute_obliquity_weight(t0_s, t_total_s)
                    weight *= obliquity

                # Accumulate to output
                image[ix, iy, iot] += amp * weight

        # Update fold counter
        fold[ix, iy] += local_fold

    return total_contributions


# =============================================================================
# Alternative kernel: Loop order optimized for trace access
# =============================================================================


@njit(parallel=True, cache=True, fastmath=True)
def _migrate_tile_kernel_trace_outer(
    amplitudes: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    midpoint_x: np.ndarray,
    midpoint_y: np.ndarray,
    trace_weights: np.ndarray,
    dt_in_ms: float,
    t_start_in_ms: float,
    image: np.ndarray,
    fold: np.ndarray,
    ox_coords: np.ndarray,
    oy_coords: np.ndarray,
    ot_coords_ms: np.ndarray,
    vrms_1d: np.ndarray,
    max_dip_deg: float,
    min_aperture: float,
    max_aperture: float,
    taper_fraction: float,
    apply_spreading: bool,
    apply_obliquity: bool,
    interp_method: int,
) -> int:
    """
    Migration kernel with parallel loop over traces.

    This version may be faster when there are many traces and small tiles.
    Uses atomic-like accumulation (Numba handles race conditions in prange).

    Note: Numba's prange with reductions handles accumulation correctly,
    but for safety we use a local accumulator per trace.
    """
    n_traces = amplitudes.shape[0]
    n_samples_in = amplitudes.shape[1]
    nx = len(ox_coords)
    ny = len(oy_coords)
    nt = len(ot_coords_ms)

    # Pre-compute velocity-dependent apertures for all output times
    apertures = np.empty(nt)
    for iot in range(nt):
        vrms = vrms_1d[iot]
        apertures[iot] = _compute_aperture(
            ot_coords_ms[iot], vrms, max_dip_deg, min_aperture, max_aperture
        )

    # Create thread-local accumulators
    # This is a workaround for atomic operations in Numba
    # Actually, let's just use sequential accumulation which is safer

    for it in range(n_traces):
        # Get trace geometry
        sx = source_x[it]
        sy = source_y[it]
        rx = receiver_x[it]
        ry = receiver_y[it]
        mx = midpoint_x[it]
        my = midpoint_y[it]
        w_trace = trace_weights[it]
        trace = amplitudes[it]

        # Parallel over output points
        for ix in prange(nx):
            ox = ox_coords[ix]

            for iy in range(ny):
                oy = oy_coords[iy]

                # Distance from output point to trace midpoint
                dist = np.sqrt((ox - mx) ** 2 + (oy - my) ** 2)

                # Quick rejection
                if dist > max_aperture:
                    continue

                # Loop over output times
                for iot in range(nt):
                    t0_ms = ot_coords_ms[iot]
                    t0_s = t0_ms / 1000.0
                    vrms = vrms_1d[iot]
                    aperture = apertures[iot]

                    if dist > aperture:
                        continue

                    # DSR travel time
                    t_total_s = _dsr_travel_time(sx, sy, rx, ry, ox, oy, t0_s, vrms)
                    t_total_ms = t_total_s * 1000.0

                    # Sample index
                    t_sample = (t_total_ms - t_start_in_ms) / dt_in_ms

                    if t_sample < 0.0 or t_sample >= n_samples_in - 1:
                        continue

                    # Interpolate
                    amp = interpolate_sample(trace, t_sample, interp_method)

                    # Weights
                    weight = w_trace
                    weight *= _compute_taper_weight(dist, aperture, taper_fraction)

                    if apply_spreading:
                        weight *= _compute_spreading_weight(t_total_s, vrms)
                    if apply_obliquity:
                        weight *= _compute_obliquity_weight(t0_s, t_total_s)

                    # Accumulate (Numba handles this safely in prange)
                    image[ix, iy, iot] += amp * weight

    # Update fold (sequential to avoid races)
    for it in range(n_traces):
        mx = midpoint_x[it]
        my = midpoint_y[it]

        for ix in range(nx):
            ox = ox_coords[ix]
            for iy in range(ny):
                oy = oy_coords[iy]
                dist = np.sqrt((ox - mx) ** 2 + (oy - my) ** 2)
                if dist <= max_aperture:
                    fold[ix, iy] += 1

    return 0


# =============================================================================
# NumbaKernel class
# =============================================================================


class NumbaKernel:
    """
    Numba CPU migration kernel.

    This is the primary compute backend, using Numba JIT compilation
    with parallel loops for multi-core execution.
    """

    def __init__(self):
        """Initialize Numba kernel."""
        self._config: KernelConfig | None = None
        self._initialized: bool = False
        self._use_trace_outer: bool = False

    @property
    def name(self) -> str:
        return "numba_cpu"

    @property
    def capabilities(self) -> set[KernelCapability]:
        return {KernelCapability.FP64, KernelCapability.BATCH}

    def initialize(self, config: KernelConfig) -> None:
        """
        Initialize kernel with configuration.

        Triggers JIT compilation of kernels.
        """
        self._config = config
        logger.info("Initializing Numba CPU kernel...")

        # Warm up JIT compilation with small arrays
        logger.debug("Warming up JIT compilation...")
        n_warmup = 10
        warmup_amp = np.random.randn(n_warmup, 100).astype(np.float32)
        warmup_coords = np.random.randn(n_warmup).astype(np.float64) * 1000
        warmup_weights = np.ones(n_warmup, dtype=np.float64)
        warmup_image = np.zeros((2, 2, 50), dtype=np.float64)
        warmup_fold = np.zeros((2, 2), dtype=np.int32)
        warmup_ox = np.array([0.0, 100.0])
        warmup_oy = np.array([0.0, 100.0])
        warmup_ot = np.linspace(0, 1000, 50)
        warmup_vrms = np.full(50, 2000.0)

        # Run once to trigger compilation
        _migrate_tile_kernel(
            warmup_amp,
            warmup_coords,
            warmup_coords,
            warmup_coords,
            warmup_coords,
            warmup_coords,
            warmup_coords,
            warmup_weights,
            4.0,
            0.0,
            warmup_image,
            warmup_fold,
            warmup_ox,
            warmup_oy,
            warmup_ot,
            warmup_vrms,
            45.0,
            100.0,
            5000.0,
            0.1,
            True,
            True,
            False,
        )

        self._initialized = True
        logger.info("Numba kernel initialized and JIT-compiled")

    def migrate_tile(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: KernelConfig | None = None,
    ) -> KernelMetrics:
        """
        Migrate traces to output tile.

        Args:
            traces: Input trace data
            output: Output tile (modified in place)
            velocity: Velocity model
            config: Kernel config (optional, uses initialized config if not provided)

        Returns:
            Execution statistics
        """
        if not self._initialized or self._config is None:
            raise RuntimeError("Kernel not initialized")

        # Use provided config or fall back to initialized config
        active_config = config if config is not None else self._config

        start_time = time.perf_counter()

        # Ensure contiguous arrays
        traces = traces.ensure_contiguous()

        # Get trace weights or create ones
        trace_weights = traces.weights
        if trace_weights is None:
            trace_weights = np.ones(traces.n_traces, dtype=np.float64)

        # Get 1D velocity profile
        if not velocity.is_1d:
            # For 3D velocity, use center profile
            # (Full 3D support would require modified kernel)
            vrms_1d = velocity.get_velocity_1d()
        else:
            vrms_1d = velocity.vrms

        # Ensure velocity matches output time axis
        if len(vrms_1d) != output.nt:
            # Interpolate to output times
            vel_t = velocity.t_coords_ms if velocity.t_coords_ms is not None else output.t_coords_ms
            vrms_1d = np.interp(output.t_coords_ms, vel_t, vrms_1d)

        # Get interpolation method code
        interp_method_code = get_method_code(active_config.interpolation_method)

        # Call kernel
        _migrate_tile_kernel(
            traces.amplitudes,
            traces.source_x,
            traces.source_y,
            traces.receiver_x,
            traces.receiver_y,
            traces.midpoint_x,
            traces.midpoint_y,
            trace_weights,
            traces.sample_rate_ms,
            traces.start_time_ms,
            output.image,
            output.fold,
            output.x_coords,
            output.y_coords,
            output.t_coords_ms,
            vrms_1d,
            active_config.max_dip_degrees,
            active_config.min_aperture_m,
            active_config.max_aperture_m,
            active_config.taper_fraction,
            active_config.apply_spreading,
            active_config.apply_obliquity,
            interp_method_code,
        )

        elapsed = time.perf_counter() - start_time

        return KernelMetrics(
            n_traces_processed=traces.n_traces,
            n_samples_output=output.nx * output.ny * output.nt,
            compute_time_s=elapsed,
        )

    def synchronize(self) -> None:
        """No-op for CPU kernel."""
        pass

    def cleanup(self) -> None:
        """Clean up kernel resources."""
        self._initialized = False
        self._config = None


def create_numba_kernel() -> NumbaKernel:
    """
    Factory function to create Numba kernel.

    Returns:
        Initialized NumbaKernel instance
    """
    return NumbaKernel()
