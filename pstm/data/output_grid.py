"""
Output Grid Definition Module.

Supports defining output grids using corner points, enabling rotated grids
with bin sizes independent of input data spacing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class CornerPoint:
    """A 2D coordinate point."""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y])
    
    def distance_to(self, other: "CornerPoint") -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __iter__(self):
        yield self.x
        yield self.y


@dataclass
class OutputGridDefinition:
    """
    Output grid defined by 4 corner points and bin sizes.
    
    Corner arrangement:
        C4 ─────────────────── C3
        │                       │
        │     OUTPUT GRID       │
        │                       │
        C1 ─────────────────── C2  (C1 = Origin)
    
    The grid can be rotated. Inline direction is C1->C2, 
    crossline direction is C1->C4.
    """
    
    # Corner points
    corner1: CornerPoint  # Origin (typically SW)
    corner2: CornerPoint  # End of inline direction (typically SE)
    corner3: CornerPoint  # Opposite corner (typically NE)
    corner4: CornerPoint  # End of crossline direction (typically NW)
    
    # Bin sizes (independent of input)
    dx: float = 25.0  # Inline bin size (m)
    dy: float = 25.0  # Crossline bin size (m)
    dt_ms: float = 2.0  # Time sample interval (ms)
    
    # Time range
    t_min_ms: float = 0.0
    t_max_ms: float = 4000.0
    
    @classmethod
    def from_bounding_box(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        dx: float = 25.0,
        dy: float = 25.0,
        dt_ms: float = 2.0,
        t_min_ms: float = 0.0,
        t_max_ms: float = 4000.0,
    ) -> "OutputGridDefinition":
        """Create grid from axis-aligned bounding box."""
        return cls(
            corner1=CornerPoint(x_min, y_min),
            corner2=CornerPoint(x_max, y_min),
            corner3=CornerPoint(x_max, y_max),
            corner4=CornerPoint(x_min, y_max),
            dx=dx,
            dy=dy,
            dt_ms=dt_ms,
            t_min_ms=t_min_ms,
            t_max_ms=t_max_ms,
        )
    
    @classmethod
    def from_origin_and_dimensions(
        cls,
        origin_x: float,
        origin_y: float,
        inline_length: float,
        crossline_length: float,
        rotation_degrees: float = 0.0,
        dx: float = 25.0,
        dy: float = 25.0,
        dt_ms: float = 2.0,
        t_min_ms: float = 0.0,
        t_max_ms: float = 4000.0,
    ) -> "OutputGridDefinition":
        """Create grid from origin, dimensions, and rotation."""
        # Rotation angle (clockwise from North/+Y)
        theta = math.radians(rotation_degrees)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # Inline direction vector (rotated X axis)
        il_dx = inline_length * cos_t
        il_dy = inline_length * sin_t
        
        # Crossline direction vector (rotated Y axis)
        xl_dx = -crossline_length * sin_t
        xl_dy = crossline_length * cos_t
        
        c1 = CornerPoint(origin_x, origin_y)
        c2 = CornerPoint(origin_x + il_dx, origin_y + il_dy)
        c3 = CornerPoint(origin_x + il_dx + xl_dx, origin_y + il_dy + xl_dy)
        c4 = CornerPoint(origin_x + xl_dx, origin_y + xl_dy)
        
        return cls(
            corner1=c1, corner2=c2, corner3=c3, corner4=c4,
            dx=dx, dy=dy, dt_ms=dt_ms,
            t_min_ms=t_min_ms, t_max_ms=t_max_ms,
        )
    
    @property
    def inline_length(self) -> float:
        """Length along inline direction (C1 to C2)."""
        return self.corner1.distance_to(self.corner2)
    
    @property
    def crossline_length(self) -> float:
        """Length along crossline direction (C1 to C4)."""
        return self.corner1.distance_to(self.corner4)
    
    @property
    def rotation_degrees(self) -> float:
        """Grid rotation in degrees (clockwise from North)."""
        dx = self.corner2.x - self.corner1.x
        dy = self.corner2.y - self.corner1.y
        # Angle of inline direction from East (+X)
        angle_from_east = math.atan2(dy, dx)
        # Convert to clockwise from North
        angle_from_north = 90 - math.degrees(angle_from_east)
        return angle_from_north % 360
    
    @property
    def nx(self) -> int:
        """Number of inline bins."""
        return max(1, int(round(self.inline_length / self.dx)) + 1)
    
    @property
    def ny(self) -> int:
        """Number of crossline bins."""
        return max(1, int(round(self.crossline_length / self.dy)) + 1)
    
    @property
    def nt(self) -> int:
        """Number of time samples."""
        return max(1, int(round((self.t_max_ms - self.t_min_ms) / self.dt_ms)) + 1)
    
    @property
    def total_points(self) -> int:
        """Total number of grid points."""
        return self.nx * self.ny * self.nt
    
    @property
    def estimated_size_bytes(self) -> int:
        """Estimated output size in bytes (float32)."""
        return self.total_points * 4
    
    @property
    def estimated_size_gb(self) -> float:
        """Estimated output size in GB (float32)."""
        return self.estimated_size_bytes / (1024**3)
    
    @property
    def area_km2(self) -> float:
        """Grid area in square kilometers."""
        return (self.inline_length * self.crossline_length) / 1e6
    
    @property
    def inline_unit_vector(self) -> NDArray[np.float64]:
        """Unit vector in inline direction."""
        dx = self.corner2.x - self.corner1.x
        dy = self.corner2.y - self.corner1.y
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-10:
            return np.array([1.0, 0.0])
        return np.array([dx / length, dy / length])
    
    @property
    def crossline_unit_vector(self) -> NDArray[np.float64]:
        """Unit vector in crossline direction."""
        dx = self.corner4.x - self.corner1.x
        dy = self.corner4.y - self.corner1.y
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-10:
            return np.array([0.0, 1.0])
        return np.array([dx / length, dy / length])
    
    def get_x_axis(self) -> NDArray[np.float64]:
        """Get X coordinates along inline direction."""
        il_vec = self.inline_unit_vector
        return np.array([
            self.corner1.x + i * self.dx * il_vec[0]
            for i in range(self.nx)
        ])
    
    def get_y_axis(self) -> NDArray[np.float64]:
        """Get Y coordinates along crossline direction."""
        xl_vec = self.crossline_unit_vector
        return np.array([
            self.corner1.y + i * self.dy * xl_vec[1]
            for i in range(self.ny)
        ])
    
    def get_t_axis_ms(self) -> NDArray[np.float64]:
        """Get time axis in milliseconds."""
        return np.arange(self.t_min_ms, self.t_max_ms + self.dt_ms/2, self.dt_ms)
    
    def get_grid_coordinates(self) -> Tuple[NDArray, NDArray]:
        """
        Get 2D arrays of X and Y coordinates for all grid points.
        
        Returns:
            (x_grid, y_grid) each of shape (nx, ny)
        """
        il_vec = self.inline_unit_vector
        xl_vec = self.crossline_unit_vector
        
        x_grid = np.zeros((self.nx, self.ny))
        y_grid = np.zeros((self.nx, self.ny))
        
        for i in range(self.nx):
            for j in range(self.ny):
                x_grid[i, j] = (self.corner1.x + 
                               i * self.dx * il_vec[0] + 
                               j * self.dy * xl_vec[0])
                y_grid[i, j] = (self.corner1.y + 
                               i * self.dx * il_vec[1] + 
                               j * self.dy * xl_vec[1])
        
        return x_grid, y_grid
    
    def point_to_grid_indices(
        self, x: float, y: float
    ) -> Tuple[float, float]:
        """
        Convert world coordinates to fractional grid indices.
        
        Args:
            x, y: World coordinates
            
        Returns:
            (inline_index, crossline_index) as floats
        """
        # Vector from origin to point
        px = x - self.corner1.x
        py = y - self.corner1.y
        
        # Project onto inline and crossline directions
        il_vec = self.inline_unit_vector
        xl_vec = self.crossline_unit_vector
        
        il_dist = px * il_vec[0] + py * il_vec[1]
        xl_dist = px * xl_vec[0] + py * xl_vec[1]
        
        il_idx = il_dist / self.dx
        xl_idx = xl_dist / self.dy
        
        return (il_idx, xl_idx)
    
    def grid_indices_to_point(
        self, il_idx: float, xl_idx: float
    ) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates.
        
        Args:
            il_idx: Inline index
            xl_idx: Crossline index
            
        Returns:
            (x, y) world coordinates
        """
        il_vec = self.inline_unit_vector
        xl_vec = self.crossline_unit_vector
        
        x = (self.corner1.x + 
             il_idx * self.dx * il_vec[0] + 
             xl_idx * self.dy * xl_vec[0])
        y = (self.corner1.y + 
             il_idx * self.dx * il_vec[1] + 
             xl_idx * self.dy * xl_vec[1])
        
        return (x, y)
    
    def get_corner_array(self) -> NDArray[np.float64]:
        """Get corners as (4, 2) array for plotting."""
        return np.array([
            [self.corner1.x, self.corner1.y],
            [self.corner2.x, self.corner2.y],
            [self.corner3.x, self.corner3.y],
            [self.corner4.x, self.corner4.y],
        ])
    
    def get_polygon_coords(self) -> Tuple[list, list]:
        """Get corner coordinates as lists for plotting (closed polygon)."""
        xs = [self.corner1.x, self.corner2.x, self.corner3.x, 
              self.corner4.x, self.corner1.x]
        ys = [self.corner1.y, self.corner2.y, self.corner3.y, 
              self.corner4.y, self.corner1.y]
        return xs, ys
    
    def to_output_grid_config(self):
        """Convert to OutputGridConfig for pipeline compatibility."""
        from pstm.config.models import OutputGridConfig
        
        # For rotated grids, we need to compute the bounding box
        corners = self.get_corner_array()
        x_min, y_min = corners.min(axis=0)
        x_max, y_max = corners.max(axis=0)
        
        # Note: This loses rotation info - full support requires
        # updating OutputGridConfig to support rotation
        return OutputGridConfig(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            dx=self.dx,
            dy=self.dy,
            t_min_ms=self.t_min_ms,
            t_max_ms=self.t_max_ms,
            dt_ms=self.dt_ms,
        )
    
    def validate(self) -> list[str]:
        """Validate grid definition. Returns list of issues."""
        issues = []
        
        if self.dx <= 0:
            issues.append(f"Inline bin size must be positive: {self.dx}")
        if self.dy <= 0:
            issues.append(f"Crossline bin size must be positive: {self.dy}")
        if self.dt_ms <= 0:
            issues.append(f"Time sample interval must be positive: {self.dt_ms}")
        if self.t_max_ms <= self.t_min_ms:
            issues.append(f"Time max must be greater than min: {self.t_min_ms} to {self.t_max_ms}")
        if self.inline_length < self.dx:
            issues.append(f"Inline length ({self.inline_length:.1f}m) less than bin size ({self.dx}m)")
        if self.crossline_length < self.dy:
            issues.append(f"Crossline length ({self.crossline_length:.1f}m) less than bin size ({self.dy}m)")
        
        # Check for degenerate grid (corners too close)
        if self.inline_length < 1.0:
            issues.append("Grid has near-zero inline extent")
        if self.crossline_length < 1.0:
            issues.append("Grid has near-zero crossline extent")
        
        return issues
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "corner1": {"x": self.corner1.x, "y": self.corner1.y},
            "corner2": {"x": self.corner2.x, "y": self.corner2.y},
            "corner3": {"x": self.corner3.x, "y": self.corner3.y},
            "corner4": {"x": self.corner4.x, "y": self.corner4.y},
            "dx": self.dx,
            "dy": self.dy,
            "dt_ms": self.dt_ms,
            "t_min_ms": self.t_min_ms,
            "t_max_ms": self.t_max_ms,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "OutputGridDefinition":
        """Deserialize from dictionary."""
        return cls(
            corner1=CornerPoint(**data["corner1"]),
            corner2=CornerPoint(**data["corner2"]),
            corner3=CornerPoint(**data["corner3"]),
            corner4=CornerPoint(**data["corner4"]),
            dx=data.get("dx", 25.0),
            dy=data.get("dy", 25.0),
            dt_ms=data.get("dt_ms", 2.0),
            t_min_ms=data.get("t_min_ms", 0.0),
            t_max_ms=data.get("t_max_ms", 4000.0),
        )
    
    def __str__(self) -> str:
        return (
            f"OutputGrid: {self.nx}×{self.ny}×{self.nt} "
            f"({self.inline_length:.0f}m × {self.crossline_length:.0f}m × "
            f"{self.t_max_ms - self.t_min_ms:.0f}ms) "
            f"@ {self.dx}m×{self.dy}m×{self.dt_ms}ms, "
            f"rotation={self.rotation_degrees:.1f}°"
        )
