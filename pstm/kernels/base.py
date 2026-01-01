"""
Kernel abstraction layer for PSTM.

Defines the Protocol and data structures for migration kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class KernelCapability(Enum):
    """Capabilities that kernels may support."""

    FP16 = "fp16"  # Half precision support
    FP64 = "fp64"  # Double precision support
    COMPLEX = "complex"  # Complex number support
    ASYNC = "async"  # Asynchronous execution
    BATCH = "batch"  # Batch processing


@dataclass
class TraceBlock:
    """
    Input data block for migration kernel.

    Contains trace amplitudes and geometry for a batch of traces.
    All arrays must be C-contiguous.
    """

    # Trace data: (n_traces, n_samples)
    amplitudes: NDArray[np.float32]

    # Source coordinates: (n_traces,)
    source_x: NDArray[np.float64]
    source_y: NDArray[np.float64]

    # Receiver coordinates: (n_traces,)
    receiver_x: NDArray[np.float64]
    receiver_y: NDArray[np.float64]

    # Derived geometry: (n_traces,)
    offset: NDArray[np.float64]
    midpoint_x: NDArray[np.float64]
    midpoint_y: NDArray[np.float64]

    # Trace weights (optional): (n_traces,)
    weights: NDArray[np.float64] | None = None

    # Time axis parameters
    sample_rate_ms: float = 2.0
    start_time_ms: float = 0.0

    @property
    def n_traces(self) -> int:
        """Number of traces."""
        return self.amplitudes.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of samples per trace."""
        return self.amplitudes.shape[1]

    @property
    def time_axis_ms(self) -> NDArray[np.float64]:
        """Time axis in milliseconds."""
        return np.arange(self.n_samples) * self.sample_rate_ms + self.start_time_ms

    def validate(self) -> list[str]:
        """Validate data consistency. Returns list of errors."""
        errors = []
        n = self.n_traces

        if len(self.source_x) != n:
            errors.append(f"source_x length mismatch: {len(self.source_x)} != {n}")
        if len(self.source_y) != n:
            errors.append(f"source_y length mismatch: {len(self.source_y)} != {n}")
        if len(self.receiver_x) != n:
            errors.append(f"receiver_x length mismatch: {len(self.receiver_x)} != {n}")
        if len(self.receiver_y) != n:
            errors.append(f"receiver_y length mismatch: {len(self.receiver_y)} != {n}")
        if len(self.offset) != n:
            errors.append(f"offset length mismatch: {len(self.offset)} != {n}")
        if len(self.midpoint_x) != n:
            errors.append(f"midpoint_x length mismatch: {len(self.midpoint_x)} != {n}")
        if len(self.midpoint_y) != n:
            errors.append(f"midpoint_y length mismatch: {len(self.midpoint_y)} != {n}")

        if not self.amplitudes.flags["C_CONTIGUOUS"]:
            errors.append("amplitudes must be C-contiguous")

        if np.any(~np.isfinite(self.amplitudes)):
            errors.append("amplitudes contains NaN or Inf values")

        return errors

    def ensure_contiguous(self) -> "TraceBlock":
        """Return a copy with all arrays contiguous in memory."""
        return TraceBlock(
            amplitudes=np.ascontiguousarray(self.amplitudes),
            source_x=np.ascontiguousarray(self.source_x),
            source_y=np.ascontiguousarray(self.source_y),
            receiver_x=np.ascontiguousarray(self.receiver_x),
            receiver_y=np.ascontiguousarray(self.receiver_y),
            offset=np.ascontiguousarray(self.offset),
            midpoint_x=np.ascontiguousarray(self.midpoint_x),
            midpoint_y=np.ascontiguousarray(self.midpoint_y),
            weights=np.ascontiguousarray(self.weights) if self.weights is not None else None,
            sample_rate_ms=self.sample_rate_ms,
            start_time_ms=self.start_time_ms,
        )


@dataclass
class OutputTile:
    """Output tile for migration kernel."""

    image: NDArray[np.float64]  # (nx, ny, nt)
    fold: NDArray[np.int32]  # (nx, ny, nt) - 3D fold per sample
    x_axis: NDArray[np.float64]  # 1D inline coordinates
    y_axis: NDArray[np.float64]  # 1D crossline coordinates
    t_axis_ms: NDArray[np.float64]
    # Optional 2D coordinate grids for rotated grids
    x_grid: NDArray[np.float64] | None = None  # (nx, ny) X coordinates
    y_grid: NDArray[np.float64] | None = None  # (nx, ny) Y coordinates

    @property
    def nx(self) -> int:
        return len(self.x_axis)

    @property
    def ny(self) -> int:
        return len(self.y_axis)

    @property
    def nt(self) -> int:
        return len(self.t_axis_ms)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nt)

    @property
    def x_range(self) -> tuple[float, float]:
        return (float(self.x_axis[0]), float(self.x_axis[-1]))

    @property
    def y_range(self) -> tuple[float, float]:
        return (float(self.y_axis[0]), float(self.y_axis[-1]))

    @property
    def dt_ms(self) -> float:
        if len(self.t_axis_ms) < 2:
            return 1.0
        return float(self.t_axis_ms[1] - self.t_axis_ms[0])

    # Aliases for compatibility with different naming conventions
    @property
    def x_coords(self) -> NDArray[np.float64]:
        return self.x_axis

    @property
    def y_coords(self) -> NDArray[np.float64]:
        return self.y_axis

    @property
    def t_coords_ms(self) -> NDArray[np.float64]:
        return self.t_axis_ms

    def reset(self) -> None:
        self.image[:] = 0.0
        self.fold[:] = 0


@dataclass
class VelocitySlice:
    """Velocity data for migration kernel."""

    vrms: NDArray[np.float64]  # (nt,) for 1D or (nx, ny, nt) for 3D
    is_1d: bool = True
    x_axis: NDArray[np.float64] | None = None
    y_axis: NDArray[np.float64] | None = None
    t_axis_ms: NDArray[np.float64] | None = None

    def get_velocity_at(self, ix: int, iy: int, it: int) -> float:
        if self.is_1d:
            return float(self.vrms[it])
        else:
            return float(self.vrms[ix, iy, it])


@dataclass
class KernelConfig:
    """Configuration for migration kernel."""

    max_aperture_m: float | None = None
    min_aperture_m: float | None = None
    max_dip_degrees: float | None = None
    taper_fraction: float | None = None

    aa_enabled: bool | None = None
    aa_filter_indices: NDArray[np.int32] | None = None
    aa_filter_bank: NDArray[np.float32] | None = None

    apply_spreading: bool | None = None
    apply_obliquity: bool = True

    interpolation_method: str | None = None
    output_dt_ms: float | None = None
    accumulate_fold: bool = True

    # Time-variant sampling
    time_variant_enabled: bool = False
    time_variant_windows: list | None = None  # List of TimeWindow objects

    # Migration kernel type selection
    kernel_type: str = "straight_ray"  # "straight_ray", "curved_ray", "anisotropic_vti"

    # Curved ray parameters (V(z) = V0 + k*z linear gradient model)
    curved_ray_enabled: bool = False
    curved_ray_v0: float = 1500.0  # Surface velocity (m/s)
    curved_ray_k: float = 0.5  # Velocity gradient (1/s), typically 0.3-0.6

    # VTI anisotropy parameters (Alkhalifah-Tsvankin eta formulation)
    vti_enabled: bool = False
    vti_eta_constant: float = 0.0  # Constant eta value (used if eta_array is None)
    vti_eta_array: NDArray[np.float64] | None = None  # 1D eta(t) or 3D eta(x,y,t)
    vti_eta_is_1d: bool = True  # True if eta_array is 1D, False if 3D

    # Grid bin spacing for AA filter (actual bin size, not linspace-derived)
    grid_dx: float = 25.0  # Inline bin size in meters
    grid_dy: float = 12.5  # Crossline bin size in meters

    def __post_init__(self):
        """Apply defaults from settings."""
        from pstm.settings import get_settings
        s = get_settings()
        
        if self.max_aperture_m is None:
            self.max_aperture_m = s.aperture.max_aperture_m
        if self.min_aperture_m is None:
            self.min_aperture_m = s.aperture.min_aperture_m
        if self.max_dip_degrees is None:
            self.max_dip_degrees = s.aperture.max_dip_degrees
        if self.taper_fraction is None:
            self.taper_fraction = s.aperture.taper_fraction
        if self.aa_enabled is None:
            self.aa_enabled = s.kernel.enable_antialiasing
        if self.apply_spreading is None:
            self.apply_spreading = s.kernel.apply_spreading_correction
        if self.interpolation_method is None:
            self.interpolation_method = s.kernel.default_interpolation
        if self.output_dt_ms is None:
            self.output_dt_ms = s.grid.dt_ms

    def get_aperture_at_time(self, t_ms: float, velocity: float) -> float:
        t_s = t_ms / 1000.0
        tan_dip = np.tan(np.radians(self.max_dip_degrees))
        aperture = velocity * t_s * tan_dip
        return np.clip(aperture, self.min_aperture_m, self.max_aperture_m)


@dataclass
class KernelMetrics:
    """Performance metrics from kernel execution."""

    n_traces_processed: int = 0
    n_samples_output: int = 0
    compute_time_s: float = 0.0
    memory_peak_mb: float = 0.0

    @property
    def traces_per_second(self) -> float:
        if self.compute_time_s > 0:
            return self.n_traces_processed / self.compute_time_s
        return 0.0

    @property
    def samples_per_second(self) -> float:
        if self.compute_time_s > 0:
            return self.n_samples_output / self.compute_time_s
        return 0.0


@runtime_checkable
class MigrationKernel(Protocol):
    """Protocol defining the interface for migration kernels."""

    @property
    def name(self) -> str:
        ...

    @property
    def capabilities(self) -> set[KernelCapability]:
        ...

    def initialize(self, config: KernelConfig) -> None:
        ...

    def migrate_tile(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: KernelConfig,
    ) -> KernelMetrics:
        ...

    def synchronize(self) -> None:
        ...

    def cleanup(self) -> None:
        ...


def create_trace_block(
    amplitudes: NDArray,
    source_x: NDArray,
    source_y: NDArray,
    receiver_x: NDArray,
    receiver_y: NDArray,
    sample_rate_ms: float,
    start_time_ms: float = 0.0,
    weights: NDArray | None = None,
) -> TraceBlock:
    """Factory function to create TraceBlock with derived quantities."""
    amplitudes = np.ascontiguousarray(amplitudes, dtype=np.float32)
    source_x = np.asarray(source_x, dtype=np.float64)
    source_y = np.asarray(source_y, dtype=np.float64)
    receiver_x = np.asarray(receiver_x, dtype=np.float64)
    receiver_y = np.asarray(receiver_y, dtype=np.float64)

    offset = np.sqrt((receiver_x - source_x) ** 2 + (receiver_y - source_y) ** 2)
    midpoint_x = (source_x + receiver_x) / 2.0
    midpoint_y = (source_y + receiver_y) / 2.0

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)

    return TraceBlock(
        amplitudes=amplitudes,
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        offset=offset,
        midpoint_x=midpoint_x,
        midpoint_y=midpoint_y,
        weights=weights,
        sample_rate_ms=sample_rate_ms,
        start_time_ms=start_time_ms,
    )


def create_output_tile(
    x_min: float,
    x_max: float,
    dx: float,
    y_min: float,
    y_max: float,
    dy: float,
    t_min_ms: float,
    t_max_ms: float,
    dt_ms: float,
) -> OutputTile:
    """Factory function to create OutputTile."""
    x_axis = np.arange(x_min, x_max + dx / 2, dx, dtype=np.float64)
    y_axis = np.arange(y_min, y_max + dy / 2, dy, dtype=np.float64)
    t_axis_ms = np.arange(t_min_ms, t_max_ms + dt_ms / 2, dt_ms, dtype=np.float64)

    nx, ny, nt = len(x_axis), len(y_axis), len(t_axis_ms)

    return OutputTile(
        image=np.zeros((nx, ny, nt), dtype=np.float64),
        fold=np.zeros((nx, ny, nt), dtype=np.int32),  # 3D fold per sample
        x_axis=x_axis,
        y_axis=y_axis,
        t_axis_ms=t_axis_ms,
    )
