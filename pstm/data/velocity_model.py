"""
Velocity model handler for PSTM.

Supports various velocity sources: constant, 1D function, 1D table, 3D cube.
Includes coverage validation for rotated grids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from pstm.config.models import VelocityConfig, VelocitySource
from pstm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VelocityCoverageReport:
    """Report on velocity cube coverage of output grid."""

    # Coverage status
    is_valid: bool

    # Velocity cube bounds
    vel_x_min: float
    vel_x_max: float
    vel_y_min: float
    vel_y_max: float

    # Required bounds (from output grid)
    grid_x_min: float
    grid_x_max: float
    grid_y_min: float
    grid_y_max: float

    # Coverage gaps (positive = gap exists)
    gap_x_min: float  # How much grid extends below velocity X min
    gap_x_max: float  # How much grid extends above velocity X max
    gap_y_min: float  # How much grid extends below velocity Y min
    gap_y_max: float  # How much grid extends above velocity Y max

    # Percentage of output points outside velocity coverage
    points_outside_percent: float

    # Detailed message
    message: str

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"VelocityCoverageReport({status}): {self.message}"

    def print_report(self) -> None:
        """Print detailed coverage report."""
        print("=" * 70)
        print("VELOCITY COVERAGE REPORT")
        print("=" * 70)
        print(f"Status: {'VALID' if self.is_valid else 'INVALID - COVERAGE GAP DETECTED'}")
        print()
        print("Velocity Cube Bounds:")
        print(f"  X: [{self.vel_x_min:.1f}, {self.vel_x_max:.1f}]")
        print(f"  Y: [{self.vel_y_min:.1f}, {self.vel_y_max:.1f}]")
        print()
        print("Output Grid Bounds (required):")
        print(f"  X: [{self.grid_x_min:.1f}, {self.grid_x_max:.1f}]")
        print(f"  Y: [{self.grid_y_min:.1f}, {self.grid_y_max:.1f}]")
        print()
        if not self.is_valid:
            print("Coverage Gaps:")
            if self.gap_x_min > 0:
                print(f"  X min: {self.gap_x_min:.1f}m below velocity coverage")
            if self.gap_x_max > 0:
                print(f"  X max: {self.gap_x_max:.1f}m above velocity coverage")
            if self.gap_y_min > 0:
                print(f"  Y min: {self.gap_y_min:.1f}m below velocity coverage")
            if self.gap_y_max > 0:
                print(f"  Y max: {self.gap_y_max:.1f}m above velocity coverage")
            print()
            print(f"Points outside coverage: {self.points_outside_percent:.1f}%")
        print()
        print(f"Message: {self.message}")
        print("=" * 70)


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

        # Store data for boundary clamping
        self._data = data

        self._interpolator = RegularGridInterpolator(
            (self.x_axis, self.y_axis, self.t_axis_ms),
            data,
            method="linear",
            bounds_error=False,
            fill_value=None,  # Will use clamping instead
        )

    def _clamp_coordinates(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Clamp query coordinates to velocity cube bounds.

        This prevents extrapolation artifacts by using edge velocities
        for points outside the velocity cube coverage.

        Args:
            points: Query points of shape (N, 3) with columns [x, y, t]

        Returns:
            Clamped points array
        """
        clamped = points.copy()

        # Clamp X
        clamped[:, 0] = np.clip(clamped[:, 0], self.x_axis[0], self.x_axis[-1])
        # Clamp Y
        clamped[:, 1] = np.clip(clamped[:, 1], self.y_axis[0], self.y_axis[-1])
        # Clamp T
        clamped[:, 2] = np.clip(clamped[:, 2], self.t_axis_ms[0], self.t_axis_ms[-1])

        return clamped

    def check_coverage(
        self,
        x_grid: NDArray[np.float64] | None = None,
        y_grid: NDArray[np.float64] | None = None,
        x_axis: NDArray[np.float64] | None = None,
        y_axis: NDArray[np.float64] | None = None,
        tolerance_m: float = 1.0,
    ) -> VelocityCoverageReport:
        """
        Check if velocity cube covers the output grid.

        Args:
            x_grid: 2D X coordinate grid (nx, ny) for rotated grids
            y_grid: 2D Y coordinate grid (nx, ny) for rotated grids
            x_axis: 1D X axis for axis-aligned grids
            y_axis: 1D Y axis for axis-aligned grids
            tolerance_m: Tolerance in meters for boundary checks

        Returns:
            VelocityCoverageReport with detailed coverage analysis
        """
        # Determine output grid bounds
        if x_grid is not None and y_grid is not None:
            # Rotated grid - use 2D coordinates
            grid_x_min = float(x_grid.min())
            grid_x_max = float(x_grid.max())
            grid_y_min = float(y_grid.min())
            grid_y_max = float(y_grid.max())
            n_points = x_grid.size

            # Count points outside
            outside_x = (x_grid < self.x_axis[0]) | (x_grid > self.x_axis[-1])
            outside_y = (y_grid < self.y_axis[0]) | (y_grid > self.y_axis[-1])
            outside = outside_x | outside_y
            points_outside = np.sum(outside)
        elif x_axis is not None and y_axis is not None:
            # Axis-aligned grid
            grid_x_min = float(x_axis.min())
            grid_x_max = float(x_axis.max())
            grid_y_min = float(y_axis.min())
            grid_y_max = float(y_axis.max())
            n_points = len(x_axis) * len(y_axis)
            points_outside = 0  # Axis-aligned, count below
        else:
            raise ValueError("Must provide either (x_grid, y_grid) or (x_axis, y_axis)")

        # Velocity cube bounds
        vel_x_min = float(self.x_axis[0])
        vel_x_max = float(self.x_axis[-1])
        vel_y_min = float(self.y_axis[0])
        vel_y_max = float(self.y_axis[-1])

        # Calculate gaps
        gap_x_min = max(0, vel_x_min - grid_x_min)
        gap_x_max = max(0, grid_x_max - vel_x_max)
        gap_y_min = max(0, vel_y_min - grid_y_min)
        gap_y_max = max(0, grid_y_max - vel_y_max)

        # For axis-aligned, estimate points outside
        if x_axis is not None and y_axis is not None:
            nx_outside = np.sum((x_axis < vel_x_min) | (x_axis > vel_x_max))
            ny_outside = np.sum((y_axis < vel_y_min) | (y_axis > vel_y_max))
            # Rough estimate
            points_outside = nx_outside * len(y_axis) + ny_outside * len(x_axis)

        points_outside_percent = 100.0 * points_outside / n_points if n_points > 0 else 0

        # Determine validity
        is_valid = (
            gap_x_min <= tolerance_m
            and gap_x_max <= tolerance_m
            and gap_y_min <= tolerance_m
            and gap_y_max <= tolerance_m
        )

        # Build message
        if is_valid:
            message = "Velocity cube fully covers output grid"
        else:
            gaps = []
            if gap_x_min > tolerance_m:
                gaps.append(f"X min gap: {gap_x_min:.1f}m")
            if gap_x_max > tolerance_m:
                gaps.append(f"X max gap: {gap_x_max:.1f}m")
            if gap_y_min > tolerance_m:
                gaps.append(f"Y min gap: {gap_y_min:.1f}m")
            if gap_y_max > tolerance_m:
                gaps.append(f"Y max gap: {gap_y_max:.1f}m")
            message = f"Coverage gaps detected: {', '.join(gaps)}"

        return VelocityCoverageReport(
            is_valid=is_valid,
            vel_x_min=vel_x_min,
            vel_x_max=vel_x_max,
            vel_y_min=vel_y_min,
            vel_y_max=vel_y_max,
            grid_x_min=grid_x_min,
            grid_x_max=grid_x_max,
            grid_y_min=grid_y_min,
            grid_y_max=grid_y_max,
            gap_x_min=gap_x_min,
            gap_x_max=gap_x_max,
            gap_y_min=gap_y_min,
            gap_y_max=gap_y_max,
            points_outside_percent=points_outside_percent,
            message=message,
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

        # Clamp coordinates to velocity cube bounds
        points = self._clamp_coordinates(points)

        return self._interpolator(points)

    def get_vrms_volume(
        self,
        x_axis: NDArray[np.float64],
        y_axis: NDArray[np.float64],
        t_axis_ms: NDArray[np.float64],
        chunk_size: int = 50,
        x_grid: NDArray[np.float64] | None = None,
        y_grid: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Get Vrms volume interpolated to output grid.

        Uses chunk-based processing to avoid massive memory allocations.
        For a 511x427x1001 grid, naive meshgrid would allocate ~17 GB of
        temporary arrays. Chunking reduces peak memory to ~1-2 GB.

        Args:
            x_axis: Output X coordinates (1D, used for axis-aligned grids)
            y_axis: Output Y coordinates (1D, used for axis-aligned grids)
            t_axis_ms: Output time axis in milliseconds
            chunk_size: Number of X/Y points to process at once (default 50)
            x_grid: Optional 2D X coordinate grid for rotated grids (nx, ny)
            y_grid: Optional 2D Y coordinate grid for rotated grids (nx, ny)

        Returns:
            Vrms volume of shape (nx, ny, nt)

        Note:
            For rotated grids, x_grid and y_grid MUST be provided to get
            correct velocity sampling. Using only x_axis/y_axis with meshgrid
            would sample velocity from wrong locations!
        """
        self._ensure_interpolator()
        assert self._interpolator is not None

        # Use 2D grids if provided (for rotated grids)
        use_2d_grids = x_grid is not None and y_grid is not None

        if use_2d_grids:
            nx, ny = x_grid.shape
            nt = len(t_axis_ms)
            logger.info(f"Using 2D coordinate grids for velocity interpolation (rotated grid support)")
        else:
            nx, ny, nt = len(x_axis), len(y_axis), len(t_axis_ms)

        # Estimate memory for full approach
        full_meshgrid_gb = (nx * ny * nt * 8 * 4) / (1024**3)  # 4 arrays

        if full_meshgrid_gb < 1.0:
            # Small enough to do in one shot
            if use_2d_grids:
                # For 2D grids, broadcast X and Y with time
                # X_grid and Y_grid are (nx, ny), we need (nx, ny, nt)
                xx = np.broadcast_to(x_grid[:, :, np.newaxis], (nx, ny, nt))
                yy = np.broadcast_to(y_grid[:, :, np.newaxis], (nx, ny, nt))
                tt = np.broadcast_to(t_axis_ms[np.newaxis, np.newaxis, :], (nx, ny, nt))
                points = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])
            else:
                xx, yy, tt = np.meshgrid(x_axis, y_axis, t_axis_ms, indexing="ij")
                points = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])
            # Clamp coordinates to velocity cube bounds
            points = self._clamp_coordinates(points)
            vrms_flat = self._interpolator(points)
            return vrms_flat.reshape((nx, ny, nt))

        # Use chunked processing for large grids
        logger.info(
            f"Using chunked velocity interpolation: {nx}x{ny}x{nt} grid, "
            f"chunk_size={chunk_size} (avoids {full_meshgrid_gb:.1f} GB peak allocation)"
        )

        # Pre-allocate output array
        result = np.empty((nx, ny, nt), dtype=np.float64)

        # Process in chunks to limit peak memory
        total_chunks = ((nx + chunk_size - 1) // chunk_size) * ((ny + chunk_size - 1) // chunk_size)
        chunk_count = 0

        for i_start in range(0, nx, chunk_size):
            i_end = min(i_start + chunk_size, nx)

            for j_start in range(0, ny, chunk_size):
                j_end = min(j_start + chunk_size, ny)

                chunk_nx = i_end - i_start
                chunk_ny = j_end - j_start

                if use_2d_grids:
                    # Extract chunk from 2D grids
                    chunk_x_grid = x_grid[i_start:i_end, j_start:j_end]
                    chunk_y_grid = y_grid[i_start:i_end, j_start:j_end]

                    # Broadcast with time axis
                    xx = np.broadcast_to(chunk_x_grid[:, :, np.newaxis], (chunk_nx, chunk_ny, nt))
                    yy = np.broadcast_to(chunk_y_grid[:, :, np.newaxis], (chunk_nx, chunk_ny, nt))
                    tt = np.broadcast_to(t_axis_ms[np.newaxis, np.newaxis, :], (chunk_nx, chunk_ny, nt))
                else:
                    # Use 1D axes with meshgrid (axis-aligned grids)
                    chunk_x = x_axis[i_start:i_end]
                    chunk_y = y_axis[j_start:j_end]
                    xx, yy, tt = np.meshgrid(chunk_x, chunk_y, t_axis_ms, indexing="ij")

                points = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])

                # Clamp coordinates to velocity cube bounds
                points = self._clamp_coordinates(points)

                # Interpolate chunk
                vrms_chunk = self._interpolator(points).reshape((chunk_nx, chunk_ny, nt))

                # Store in result
                result[i_start:i_end, j_start:j_end, :] = vrms_chunk

                # Explicit cleanup of temporary arrays
                del xx, yy, tt, points, vrms_chunk

                chunk_count += 1

        logger.debug(f"Velocity interpolation complete: processed {chunk_count} chunks")

        return result

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
    - Proper handling of rotated grids via 2D coordinate grids
    """

    def __init__(
        self,
        model: VelocityModel,
        output_x_axis: NDArray[np.float64],
        output_y_axis: NDArray[np.float64],
        output_t_axis_ms: NDArray[np.float64],
        precompute: bool = True,
        output_x_grid: NDArray[np.float64] | None = None,
        output_y_grid: NDArray[np.float64] | None = None,
    ):
        """
        Initialize velocity manager.

        Args:
            model: Velocity model
            output_x_axis: Output X coordinates (1D)
            output_y_axis: Output Y coordinates (1D)
            output_t_axis_ms: Output time axis (ms)
            precompute: Pre-interpolate to output grid
            output_x_grid: 2D X coordinate grid (nx, ny) for rotated grids
            output_y_grid: 2D Y coordinate grid (nx, ny) for rotated grids

        Note:
            For rotated grids, output_x_grid and output_y_grid MUST be provided
            to ensure velocity is sampled at the correct spatial locations.
            Without these, velocity would be sampled from wrong locations,
            causing residual moveout in CIGs.
        """
        self.model = model
        self.output_x_axis = output_x_axis
        self.output_y_axis = output_y_axis
        self.output_t_axis_ms = output_t_axis_ms
        self.output_x_grid = output_x_grid
        self.output_y_grid = output_y_grid

        self._precomputed: NDArray[np.float64] | None = None
        self._is_1d = model.is_laterally_constant
        self._is_rotated = output_x_grid is not None and output_y_grid is not None

        # Cache for velocity slices (lazy slicing optimization)
        # Key: (x_start, x_end, y_start, y_end) -> VelocitySlice
        self._slice_cache: dict[tuple[int, int, int, int], object] = {}
        self._slice_cache_hits = 0
        self._slice_cache_misses = 0

        if self._is_rotated:
            logger.info("VelocityManager: Using 2D coordinate grids for rotated grid support")

        if precompute and not self._is_1d:
            self._precompute_velocity()

    def _precompute_velocity(self) -> None:
        """Pre-interpolate 3D velocity to output grid."""
        logger.info("Pre-computing velocity on output grid...")

        self._precomputed = self.model.get_vrms_volume(
            self.output_x_axis,
            self.output_y_axis,
            self.output_t_axis_ms,
            x_grid=self.output_x_grid,
            y_grid=self.output_y_grid,
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
        Get VelocitySlice object for a tile (cached).

        Uses lazy slicing with caching to avoid redundant slice creation
        when the same tile bounds are requested multiple times.

        Args:
            x_start, x_end: X index range
            y_start, y_end: Y index range

        Returns:
            VelocitySlice for kernel consumption
        """
        from pstm.kernels.base import VelocitySlice

        # Check cache first
        cache_key = (x_start, x_end, y_start, y_end)
        if cache_key in self._slice_cache:
            self._slice_cache_hits += 1
            return self._slice_cache[cache_key]

        self._slice_cache_misses += 1

        vrms, is_1d = self.get_velocity_for_tile(x_start, x_end, y_start, y_end)

        if is_1d:
            vel_slice = VelocitySlice(vrms=vrms, is_1d=True)
        else:
            vel_slice = VelocitySlice(
                vrms=vrms,
                is_1d=False,
                x_axis=self.output_x_axis[x_start:x_end],
                y_axis=self.output_y_axis[y_start:y_end],
                t_axis_ms=self.output_t_axis_ms,
            )

        # Cache the slice (limited to ~100 entries to prevent memory issues)
        if len(self._slice_cache) < 100:
            self._slice_cache[cache_key] = vel_slice

        return vel_slice
    
    @property
    def memory_usage_gb(self) -> float:
        """Memory usage of precomputed velocity in GB."""
        if self._precomputed is None:
            return 0.0
        return self._precomputed.nbytes / (1024**3)

    def get_slice_cache_stats(self) -> dict:
        """Get velocity slice cache statistics."""
        total = self._slice_cache_hits + self._slice_cache_misses
        hit_rate = self._slice_cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._slice_cache_hits,
            "misses": self._slice_cache_misses,
            "total": total,
            "hit_rate": hit_rate,
            "cached_slices": len(self._slice_cache),
        }


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

    # Get output coordinates (handles both bounding-box and corner-point grids)
    coords = output_grid.get_output_coordinates()
    x_axis = coords['x']
    y_axis = coords['y']
    t_axis_ms = coords['t_ms']

    # Get 2D coordinate grids for rotated grids
    # These are None for axis-aligned grids but essential for rotated grids
    # to ensure velocity is sampled at correct spatial locations
    x_grid = coords.get('X')  # 2D grid (nx, ny)
    y_grid = coords.get('Y')  # 2D grid (nx, ny)

    # Check if velocity cube is in IL/XL space (indexed by inline/crossline numbers)
    # rather than X/Y UTM coordinates. This is detected by checking if the velocity
    # axes look like integer indices (1, 2, 3, ...) rather than UTM coordinates.
    if isinstance(model, CubeVelocityModel):
        vel_x_axis = model.x_axis
        vel_y_axis = model.y_axis

        # Detect IL/XL indexed velocity cube:
        # - Axes start at 1 (or close to it)
        # - Axes are integers or very close to integers
        # - Range is small (< 1000) rather than UTM scale (> 100000)
        is_ilxl_indexed = (
            vel_x_axis[0] < 10 and  # Starts near 1
            vel_y_axis[0] < 10 and
            vel_x_axis[-1] < 1000 and  # Small range
            vel_y_axis[-1] < 1000 and
            np.allclose(vel_x_axis, np.round(vel_x_axis)) and  # Integer values
            np.allclose(vel_y_axis, np.round(vel_y_axis))
        )

        if is_ilxl_indexed:
            logger.info(
                f"Detected IL/XL indexed velocity cube: "
                f"IL=[{vel_x_axis[0]:.0f}, {vel_x_axis[-1]:.0f}], "
                f"XL=[{vel_y_axis[0]:.0f}, {vel_y_axis[-1]:.0f}]"
            )

            # Create IL/XL coordinate grids for the output grid
            # Output grid point (ix, iy) corresponds to IL=ix+1, XL=iy+1
            nx, ny = len(x_axis), len(y_axis)
            il_grid = np.arange(1, nx + 1, dtype=np.float64)[:, np.newaxis] * np.ones(ny)
            xl_grid = np.ones(nx)[:, np.newaxis] * np.arange(1, ny + 1, dtype=np.float64)

            logger.info(
                f"Created IL/XL grids for velocity sampling: "
                f"IL=[{il_grid.min():.0f}, {il_grid.max():.0f}], "
                f"XL=[{xl_grid.min():.0f}, {xl_grid.max():.0f}]"
            )

            # Use IL/XL grids instead of X/Y grids for velocity sampling
            x_grid = il_grid
            y_grid = xl_grid

    return VelocityManager(
        model=model,
        output_x_axis=x_axis,
        output_y_axis=y_axis,
        output_t_axis_ms=t_axis_ms,
        precompute=config.precompute_to_output_grid,
        output_x_grid=x_grid,
        output_y_grid=y_grid,
    )


def validate_velocity_coverage(
    velocity_path: Path | str,
    output_grid,  # OutputGridConfig
    raise_on_error: bool = False,
) -> VelocityCoverageReport:
    """
    Validate that a velocity cube covers the output grid.

    This should be called before migration to detect coverage gaps
    that would cause artifacts (diagonal lines in time/crossline slices).

    Args:
        velocity_path: Path to velocity cube (zarr)
        output_grid: Output grid configuration
        raise_on_error: If True, raise ValueError on coverage gap

    Returns:
        VelocityCoverageReport with detailed analysis

    Raises:
        ValueError: If raise_on_error=True and coverage is invalid
    """
    # Load velocity model
    model = CubeVelocityModel(velocity_path)

    # Get output coordinates
    coords = output_grid.get_output_coordinates()
    x_grid = coords.get('X')
    y_grid = coords.get('Y')
    x_axis = coords.get('x')
    y_axis = coords.get('y')

    # Check if velocity cube is IL/XL indexed
    vel_x_axis = model.x_axis
    vel_y_axis = model.y_axis
    is_ilxl_indexed = (
        vel_x_axis[0] < 10 and
        vel_y_axis[0] < 10 and
        vel_x_axis[-1] < 1000 and
        vel_y_axis[-1] < 1000 and
        np.allclose(vel_x_axis, np.round(vel_x_axis)) and
        np.allclose(vel_y_axis, np.round(vel_y_axis))
    )

    if is_ilxl_indexed:
        # For IL/XL indexed cubes, check that IL/XL ranges cover output grid
        nx, ny = len(x_axis), len(y_axis)
        il_min, il_max = 1, nx  # Output grid IL range
        xl_min, xl_max = 1, ny  # Output grid XL range

        # Create IL/XL grids for coverage check
        il_grid = np.arange(1, nx + 1, dtype=np.float64)[:, np.newaxis] * np.ones(ny)
        xl_grid = np.ones(nx)[:, np.newaxis] * np.arange(1, ny + 1, dtype=np.float64)

        logger.info(
            f"Validating IL/XL indexed velocity cube: "
            f"Velocity IL=[{vel_x_axis[0]:.0f}, {vel_x_axis[-1]:.0f}], "
            f"XL=[{vel_y_axis[0]:.0f}, {vel_y_axis[-1]:.0f}]; "
            f"Required IL=[{il_min}, {il_max}], XL=[{xl_min}, {xl_max}]"
        )

        report = model.check_coverage(
            x_grid=il_grid,
            y_grid=xl_grid,
        )
    else:
        # Standard X/Y coverage check
        report = model.check_coverage(
            x_grid=x_grid,
            y_grid=y_grid,
            x_axis=x_axis,
            y_axis=y_axis,
        )

    # Log result
    if report.is_valid:
        logger.info(f"Velocity coverage check PASSED: {report.message}")
    else:
        logger.error(f"Velocity coverage check FAILED: {report.message}")
        if is_ilxl_indexed:
            logger.error(
                f"  Velocity bounds: IL [{report.vel_x_min:.0f}, {report.vel_x_max:.0f}], "
                f"XL [{report.vel_y_min:.0f}, {report.vel_y_max:.0f}]"
            )
            logger.error(
                f"  Grid bounds:     IL [{report.grid_x_min:.0f}, {report.grid_x_max:.0f}], "
                f"XL [{report.grid_y_min:.0f}, {report.grid_y_max:.0f}]"
            )
        else:
            logger.error(
                f"  Velocity bounds: X [{report.vel_x_min:.1f}, {report.vel_x_max:.1f}], "
                f"Y [{report.vel_y_min:.1f}, {report.vel_y_max:.1f}]"
            )
            logger.error(
                f"  Grid bounds:     X [{report.grid_x_min:.1f}, {report.grid_x_max:.1f}], "
                f"Y [{report.grid_y_min:.1f}, {report.grid_y_max:.1f}]"
            )
        logger.error(f"  Points outside coverage: {report.points_outside_percent:.1f}%")

        if raise_on_error:
            raise ValueError(f"Velocity coverage check failed: {report.message}")

    return report


def get_required_velocity_bounds(output_grid) -> dict:
    """
    Get the required velocity cube bounds for an output grid.

    Use this to determine what bounds a velocity cube needs to cover
    the entire output grid (including rotated grids).

    Args:
        output_grid: Output grid configuration

    Returns:
        Dictionary with x_min, x_max, y_min, y_max required bounds
    """
    coords = output_grid.get_output_coordinates()
    x_grid = coords.get('X')
    y_grid = coords.get('Y')

    if x_grid is not None and y_grid is not None:
        # Rotated grid
        return {
            'x_min': float(x_grid.min()),
            'x_max': float(x_grid.max()),
            'y_min': float(y_grid.min()),
            'y_max': float(y_grid.max()),
        }
    else:
        # Axis-aligned grid
        x_axis = coords['x']
        y_axis = coords['y']
        return {
            'x_min': float(x_axis.min()),
            'x_max': float(x_axis.max()),
            'y_min': float(y_axis.min()),
            'y_max': float(y_axis.max()),
        }
