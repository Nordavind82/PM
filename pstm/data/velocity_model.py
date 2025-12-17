"""
Velocity model handler for PSTM.

Supports various velocity sources: constant, 1D function, 1D table, 3D cube.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from pstm.config.models import VelocityConfig, VelocitySource
from pstm.utils.logging import get_logger

logger = get_logger(__name__)


class VelocityModel(ABC):
    """Abstract base class for velocity models."""

    @abstractmethod
    def get_vrms_1d(self, t_axis_ms: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get 1D Vrms profile at given times.

        Args:
            t_axis_ms: Time axis in milliseconds

        Returns:
            Vrms values at each time
        """
        pass

    @abstractmethod
    def get_vrms_at_point(
        self,
        x: float,
        y: float,
        t_axis_ms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Get Vrms profile at a specific (x, y) location.

        Args:
            x: X coordinate
            y: Y coordinate
            t_axis_ms: Time axis in milliseconds

        Returns:
            Vrms values at each time
        """
        pass

    @abstractmethod
    def get_vrms_volume(
        self,
        x_axis: NDArray[np.float64],
        y_axis: NDArray[np.float64],
        t_axis_ms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Get Vrms volume interpolated to output grid.

        Args:
            x_axis: Output X coordinates
            y_axis: Output Y coordinates
            t_axis_ms: Output time axis in milliseconds

        Returns:
            Vrms volume of shape (nx, ny, nt)
        """
        pass

    @property
    @abstractmethod
    def is_laterally_constant(self) -> bool:
        """Whether velocity varies only with time (not x, y)."""
        pass


class ConstantVelocityModel(VelocityModel):
    """Constant velocity model."""

    def __init__(self, velocity: float):
        """
        Initialize constant velocity model.

        Args:
            velocity: Constant velocity in m/s
        """
        self.velocity = velocity
        logger.info(f"Created constant velocity model: {velocity} m/s")

    def get_vrms_1d(self, t_axis_ms: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full_like(t_axis_ms, self.velocity, dtype=np.float64)

    def get_vrms_at_point(
        self,
        x: float,
        y: float,
        t_axis_ms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.get_vrms_1d(t_axis_ms)

    def get_vrms_volume(
        self,
        x_axis: NDArray[np.float64],
        y_axis: NDArray[np.float64],
        t_axis_ms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        nx, ny, nt = len(x_axis), len(y_axis), len(t_axis_ms)
        return np.full((nx, ny, nt), self.velocity, dtype=np.float64)

    @property
    def is_laterally_constant(self) -> bool:
        return True


class LinearVelocityModel(VelocityModel):
    """Linear velocity model: V(t) = V0 + k * t."""

    def __init__(self, v0: float, k: float):
        """
        Initialize linear velocity model.

        Args:
            v0: Velocity at t=0 in m/s
            k: Velocity gradient in m/s per second
        """
        self.v0 = v0
        self.k = k
        logger.info(f"Created linear velocity model: V(t) = {v0} + {k} * t")

    def get_vrms_1d(self, t_axis_ms: NDArray[np.float64]) -> NDArray[np.float64]:
        t_s = t_axis_ms / 1000.0  # Convert to seconds
        return self.v0 + self.k * t_s

    def get_vrms_at_point(
        self,
        x: float,
        y: float,
        t_axis_ms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.get_vrms_1d(t_axis_ms)

    def get_vrms_volume(
        self,
        x_axis: NDArray[np.float64],
        y_axis: NDArray[np.float64],
        t_axis_ms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        vrms_1d = self.get_vrms_1d(t_axis_ms)
        nx, ny = len(x_axis), len(y_axis)
        # Broadcast to 3D
        return np.broadcast_to(vrms_1d, (nx, ny, len(t_axis_ms))).copy()

    @property
    def is_laterally_constant(self) -> bool:
        return True


class TableVelocityModel(VelocityModel):
    """1D velocity model from time-velocity pairs."""

    def __init__(self, time_ms: NDArray[np.float64], velocity: NDArray[np.float64]):
        """
        Initialize table velocity model.

        Args:
            time_ms: Time values in milliseconds
            velocity: Velocity values in m/s
        """
        self.time_ms = np.asarray(time_ms, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)

        if len(self.time_ms) != len(self.velocity):
            raise ValueError("Time and velocity arrays must have same length")

        # Ensure sorted
        sort_idx = np.argsort(self.time_ms)
        self.time_ms = self.time_ms[sort_idx]
        self.velocity = self.velocity[sort_idx]

        logger.info(
            f"Created table velocity model: {len(self.time_ms)} points, "
            f"T: [{self.time_ms[0]:.0f}, {self.time_ms[-1]:.0f}] ms, "
            f"V: [{self.velocity.min():.0f}, {self.velocity.max():.0f}] m/s"
        )

    def get_vrms_1d(self, t_axis_ms: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.interp(t_axis_ms, self.time_ms, self.velocity)

    def get_vrms_at_point(
        self,
        x: float,
        y: float,
        t_axis_ms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.get_vrms_1d(t_axis_ms)

    def get_vrms_volume(
        self,
        x_axis: NDArray[np.float64],
        y_axis: NDArray[np.float64],
        t_axis_ms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        vrms_1d = self.get_vrms_1d(t_axis_ms)
        nx, ny = len(x_axis), len(y_axis)
        return np.broadcast_to(vrms_1d, (nx, ny, len(t_axis_ms))).copy()

    @property
    def is_laterally_constant(self) -> bool:
        return True


class CubeVelocityModel(VelocityModel):
    """3D velocity cube model."""

    def __init__(
        self,
        path: Path | str,
        x_axis: NDArray[np.float64] | None = None,
        y_axis: NDArray[np.float64] | None = None,
        t_axis_ms: NDArray[np.float64] | None = None,
    ):
        """
        Initialize 3D velocity cube model.

        Args:
            path: Path to Zarr velocity cube
            x_axis: X axis coordinates (read from attrs if None)
            y_axis: Y axis coordinates (read from attrs if None)
            t_axis_ms: Time axis in ms (read from attrs if None)
        """
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Velocity cube not found: {self.path}")

        # Open Zarr
        self._zarr = zarr.open(str(self.path), mode="r")

        # Get or set axes
        self.x_axis = self._get_axis(x_axis, "x_axis", "x")
        self.y_axis = self._get_axis(y_axis, "y_axis", "y")
        self.t_axis_ms = self._get_axis(t_axis_ms, "t_axis_ms", "t")

        # Validate shape
        expected_shape = (len(self.x_axis), len(self.y_axis), len(self.t_axis_ms))
        if self._zarr.shape != expected_shape:
            # Try transposed
            if self._zarr.shape == expected_shape[::-1]:
                logger.warning("Velocity cube appears transposed, will handle internally")
                self._transposed = True
            else:
                raise ValueError(
                    f"Velocity cube shape {self._zarr.shape} doesn't match "
                    f"axes: {expected_shape}"
                )
        else:
            self._transposed = False

        # Create interpolator
        self._interpolator: RegularGridInterpolator | None = None

        logger.info(
            f"Created 3D velocity model: {self._zarr.shape}, "
            f"X: [{self.x_axis[0]:.0f}, {self.x_axis[-1]:.0f}], "
            f"Y: [{self.y_axis[0]:.0f}, {self.y_axis[-1]:.0f}], "
            f"T: [{self.t_axis_ms[0]:.0f}, {self.t_axis_ms[-1]:.0f}] ms"
        )

    def _get_axis(
        self,
        provided: NDArray[np.float64] | None,
        attr_name: str,
        short_name: str,
    ) -> NDArray[np.float64]:
        """Get axis from provided value or Zarr attributes."""
        if provided is not None:
            return np.asarray(provided, dtype=np.float64)

        # Try to read from attributes
        if attr_name in self._zarr.attrs:
            return np.asarray(self._zarr.attrs[attr_name], dtype=np.float64)

        # Try alternative names
        alternatives = [
            f"{short_name}_coords",
            f"{short_name}",
            f"{short_name.upper()}",
        ]
        for alt in alternatives:
            if alt in self._zarr.attrs:
                return np.asarray(self._zarr.attrs[alt], dtype=np.float64)

        raise ValueError(
            f"Could not determine {attr_name} - provide explicitly or add to Zarr attributes"
        )

    def _ensure_interpolator(self) -> None:
        """Ensure interpolator is built."""
        if self._interpolator is not None:
            return

        logger.debug("Building velocity interpolator...")

        # Load full cube into memory for interpolation
        data = self._zarr[:]
        if self._transposed:
            data = data.T

        self._interpolator = RegularGridInterpolator(
            (self.x_axis, self.y_axis, self.t_axis_ms),
            data,
            method="linear",
            bounds_error=False,
            fill_value=None,  # Extrapolate
        )

    def get_vrms_1d(self, t_axis_ms: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get 1D profile at center of model."""
        x_center = (self.x_axis[0] + self.x_axis[-1]) / 2
        y_center = (self.y_axis[0] + self.y_axis[-1]) / 2
        return self.get_vrms_at_point(x_center, y_center, t_axis_ms)

    def get_vrms_at_point(
        self,
        x: float,
        y: float,
        t_axis_ms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self._ensure_interpolator()
        assert self._interpolator is not None

        points = np.column_stack([
            np.full_like(t_axis_ms, x),
            np.full_like(t_axis_ms, y),
            t_axis_ms,
        ])

        return self._interpolator(points)

    def get_vrms_volume(
        self,
        x_axis: NDArray[np.float64],
        y_axis: NDArray[np.float64],
        t_axis_ms: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self._ensure_interpolator()
        assert self._interpolator is not None

        nx, ny, nt = len(x_axis), len(y_axis), len(t_axis_ms)

        # Create meshgrid of output coordinates
        xx, yy, tt = np.meshgrid(x_axis, y_axis, t_axis_ms, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])

        # Interpolate
        vrms_flat = self._interpolator(points)

        return vrms_flat.reshape((nx, ny, nt))

    @property
    def is_laterally_constant(self) -> bool:
        return False


def create_velocity_model(config: VelocityConfig) -> VelocityModel:
    """
    Factory function to create appropriate velocity model from config.

    Args:
        config: Velocity configuration

    Returns:
        Appropriate VelocityModel subclass instance
    """
    if config.source == VelocitySource.CONSTANT:
        if config.constant_velocity is None:
            raise ValueError("constant_velocity required for CONSTANT source")
        return ConstantVelocityModel(config.constant_velocity)

    elif config.source == VelocitySource.LINEAR_V0K:
        if config.v0 is None or config.k is None:
            raise ValueError("v0 and k required for LINEAR_V0K source")
        return LinearVelocityModel(config.v0, config.k)

    elif config.source == VelocitySource.TABLE_1D:
        if not config.velocity_table:
            raise ValueError("velocity_table required for TABLE_1D source")
        times = np.array([t for t, _ in config.velocity_table])
        velocities = np.array([v for _, v in config.velocity_table])
        return TableVelocityModel(times, velocities)

    elif config.source == VelocitySource.CUBE_3D:
        if config.velocity_path is None:
            raise ValueError("velocity_path required for CUBE_3D source")
        return CubeVelocityModel(config.velocity_path)

    else:
        raise ValueError(f"Unknown velocity source: {config.source}")


def validate_velocity_range(
    model: VelocityModel,
    t_axis_ms: NDArray[np.float64],
    min_velocity: float | None = None,
    max_velocity: float | None = None,
) -> list[str]:
    """
    Validate velocity model values are within expected range.

    Args:
        model: Velocity model to validate
        t_axis_ms: Time axis for sampling
        min_velocity: Minimum valid velocity (default from settings)
        max_velocity: Maximum valid velocity (default from settings)

    Returns:
        List of warning messages (empty if valid)
    """
    from pstm.settings import get_settings
    s = get_settings()
    
    if min_velocity is None:
        min_velocity = s.velocity.qc_min_velocity_ms
    if max_velocity is None:
        max_velocity = s.velocity.qc_max_velocity_ms
    
    warnings = []

    vrms = model.get_vrms_1d(t_axis_ms)

    if np.any(vrms < min_velocity):
        warnings.append(
            f"Velocity below minimum ({min_velocity} m/s): min = {vrms.min():.0f} m/s"
        )

    if np.any(vrms > max_velocity):
        warnings.append(
            f"Velocity above maximum ({max_velocity} m/s): max = {vrms.max():.0f} m/s"
        )

    if np.any(np.isnan(vrms)):
        warnings.append("Velocity contains NaN values")

    if np.any(np.isinf(vrms)):
        warnings.append("Velocity contains infinite values")

    # Check for velocity inversions (decreasing velocity with depth)
    dv = np.diff(vrms)
    inversion_threshold = s.velocity.inversion_threshold_ms
    if np.any(dv < inversion_threshold):
        n_inversions = np.sum(dv < inversion_threshold)
        warnings.append(f"Velocity inversions detected: {n_inversions} locations")

    return warnings


class VelocityManager:
    """
    Manages velocity model for migration, including tile extraction.
    
    Handles:
    - Pre-interpolation to output grid
    - Tile-based velocity extraction
    - Memory-efficient velocity access
    """
    
    def __init__(
        self,
        model: VelocityModel,
        output_x_axis: NDArray[np.float64],
        output_y_axis: NDArray[np.float64],
        output_t_axis_ms: NDArray[np.float64],
        precompute: bool = True,
    ):
        """
        Initialize velocity manager.
        
        Args:
            model: Velocity model
            output_x_axis: Output X coordinates
            output_y_axis: Output Y coordinates
            output_t_axis_ms: Output time axis (ms)
            precompute: Pre-interpolate to output grid
        """
        self.model = model
        self.output_x_axis = output_x_axis
        self.output_y_axis = output_y_axis
        self.output_t_axis_ms = output_t_axis_ms
        
        self._precomputed: NDArray[np.float64] | None = None
        self._is_1d = model.is_laterally_constant
        
        if precompute and not self._is_1d:
            self._precompute_velocity()
    
    def _precompute_velocity(self) -> None:
        """Pre-interpolate 3D velocity to output grid."""
        logger.info("Pre-computing velocity on output grid...")
        
        self._precomputed = self.model.get_vrms_volume(
            self.output_x_axis,
            self.output_y_axis,
            self.output_t_axis_ms,
        )
        
        logger.info(
            f"Velocity grid: {self._precomputed.shape}, "
            f"range: [{self._precomputed.min():.0f}, {self._precomputed.max():.0f}] m/s"
        )
    
    @property
    def is_laterally_constant(self) -> bool:
        """Whether velocity is 1D (laterally constant)."""
        return self._is_1d
    
    def get_velocity_1d(self) -> NDArray[np.float64]:
        """
        Get 1D velocity profile.
        
        Returns:
            Velocity array (nt,)
        """
        return self.model.get_vrms_1d(self.output_t_axis_ms)
    
    def get_velocity_for_tile(
        self,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
    ) -> tuple[NDArray[np.float64], bool]:
        """
        Get velocity for a tile region.
        
        Args:
            x_start, x_end: X index range (exclusive end)
            y_start, y_end: Y index range (exclusive end)
            
        Returns:
            Tuple of (velocity_array, is_1d)
            - If 1D: shape (nt,)
            - If 3D: shape (tile_nx, tile_ny, nt)
        """
        if self._is_1d:
            return self.get_velocity_1d(), True
        
        if self._precomputed is not None:
            # Extract tile from precomputed volume
            return self._precomputed[x_start:x_end, y_start:y_end, :].copy(), False
        
        # Compute on-the-fly for this tile
        tile_x = self.output_x_axis[x_start:x_end]
        tile_y = self.output_y_axis[y_start:y_end]
        
        return self.model.get_vrms_volume(tile_x, tile_y, self.output_t_axis_ms), False
    
    def get_velocity_slice_for_tile(
        self,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
    ):
        """
        Get VelocitySlice object for a tile.
        
        Args:
            x_start, x_end: X index range
            y_start, y_end: Y index range
            
        Returns:
            VelocitySlice for kernel consumption
        """
        from pstm.kernels.base import VelocitySlice
        
        vrms, is_1d = self.get_velocity_for_tile(x_start, x_end, y_start, y_end)
        
        if is_1d:
            return VelocitySlice(vrms=vrms, is_1d=True)
        else:
            return VelocitySlice(
                vrms=vrms,
                is_1d=False,
                x_axis=self.output_x_axis[x_start:x_end],
                y_axis=self.output_y_axis[y_start:y_end],
                t_axis_ms=self.output_t_axis_ms,
            )
    
    @property
    def memory_usage_gb(self) -> float:
        """Memory usage of precomputed velocity in GB."""
        if self._precomputed is None:
            return 0.0
        return self._precomputed.nbytes / (1024**3)


def create_velocity_manager(
    config,  # VelocityConfig
    output_grid,  # OutputGridConfig
) -> VelocityManager:
    """
    Create a velocity manager from configuration.
    
    Args:
        config: Velocity configuration
        output_grid: Output grid configuration
        
    Returns:
        Configured VelocityManager
    """
    # Create velocity model
    model = create_velocity_model(config)
    
    # Create output axes
    x_axis = np.arange(output_grid.x_min, output_grid.x_max + output_grid.dx / 2, output_grid.dx)
    y_axis = np.arange(output_grid.y_min, output_grid.y_max + output_grid.dy / 2, output_grid.dy)
    t_axis_ms = np.arange(
        output_grid.t_min_ms, 
        output_grid.t_max_ms + output_grid.dt_ms / 2, 
        output_grid.dt_ms
    )
    
    return VelocityManager(
        model=model,
        output_x_axis=x_axis,
        output_y_axis=y_axis,
        output_t_axis_ms=t_axis_ms,
        precompute=config.precompute_to_output_grid,
    )
