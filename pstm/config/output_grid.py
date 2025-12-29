"""
Output grid configuration with corner-point support.

This module provides flexible output grid definition supporting:
- Traditional bounding box (x_min/x_max, y_min/y_max)
- Corner point definition (4 corners for rotated grids)
- Independent bin sizes from input data

The output grid is defined BEFORE velocity model initialization,
ensuring velocity is interpolated to the correct output coordinates.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pstm.analysis.bin_size import BinSizeResult
    from pstm.analysis.grid_outliers import OutlierReport

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


# =============================================================================
# Type Aliases
# =============================================================================

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class GridDefinitionMethod(str, Enum):
    """Method used to define the output grid."""
    BOUNDING_BOX = "bounding_box"       # Traditional min/max
    CORNER_POINTS = "corner_points"     # 4 corners (supports rotation)
    CENTER_DIMENSIONS = "center_dimensions"  # Center point + dimensions
    IMPORTED = "imported"               # Loaded from external file


# =============================================================================
# Corner Point Grid
# =============================================================================


class Point2D(BaseModel):
    """A 2D coordinate point."""
    model_config = ConfigDict(frozen=True)
    
    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")
    
    def __iter__(self):
        return iter((self.x, self.y))
    
    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)
    
    def distance_to(self, other: "Point2D") -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class CornerPoints(BaseModel):
    """
    Four corner points defining a (potentially rotated) output grid.
    
    Convention:
        C4 (NW) ─────────────── C3 (NE)
        │                        │
        │     OUTPUT GRID        │
        │                        │
        C1 (Origin/SW) ──────── C2 (SE)
    
    The grid inline direction is C1 → C2.
    The grid crossline direction is C1 → C4.
    """
    model_config = ConfigDict(validate_assignment=True)
    
    corner1: Point2D = Field(description="Origin corner (typically SW)")
    corner2: Point2D = Field(description="Inline end corner (typically SE)")
    corner3: Point2D = Field(description="Opposite corner (typically NE)")
    corner4: Point2D = Field(description="Crossline end corner (typically NW)")
    
    @classmethod
    def from_tuples(
        cls,
        c1: tuple[float, float],
        c2: tuple[float, float],
        c3: tuple[float, float],
        c4: tuple[float, float],
    ) -> "CornerPoints":
        """Create from coordinate tuples."""
        return cls(
            corner1=Point2D(x=c1[0], y=c1[1]),
            corner2=Point2D(x=c2[0], y=c2[1]),
            corner3=Point2D(x=c3[0], y=c3[1]),
            corner4=Point2D(x=c4[0], y=c4[1]),
        )
    
    @classmethod
    def from_bounding_box(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> "CornerPoints":
        """Create axis-aligned corners from bounding box."""
        return cls.from_tuples(
            c1=(x_min, y_min),  # SW / origin
            c2=(x_max, y_min),  # SE
            c3=(x_max, y_max),  # NE
            c4=(x_min, y_max),  # NW
        )
    
    @classmethod
    def from_origin_and_dimensions(
        cls,
        origin: tuple[float, float],
        inline_length: float,
        crossline_length: float,
        rotation_deg: float = 0.0,
    ) -> "CornerPoints":
        """
        Create from origin point, dimensions, and rotation.
        
        Args:
            origin: Origin point (x, y)
            inline_length: Length in inline direction (meters)
            crossline_length: Length in crossline direction (meters)
            rotation_deg: Rotation angle in degrees (clockwise from North/+Y)
        """
        # Convert to radians (clockwise from North)
        theta = math.radians(rotation_deg)
        
        # Inline direction vector (rotated East direction)
        inline_dx = inline_length * math.sin(theta)
        inline_dy = inline_length * math.cos(theta)
        
        # Crossline direction vector (rotated North direction, perpendicular)
        xline_dx = crossline_length * math.cos(theta)
        xline_dy = -crossline_length * math.sin(theta)
        
        ox, oy = origin
        
        return cls.from_tuples(
            c1=(ox, oy),
            c2=(ox + inline_dx, oy + inline_dy),
            c3=(ox + inline_dx + xline_dx, oy + inline_dy + xline_dy),
            c4=(ox + xline_dx, oy + xline_dy),
        )
    
    @computed_field
    @property
    def inline_length(self) -> float:
        """Length of inline direction (C1 to C2)."""
        return self.corner1.distance_to(self.corner2)
    
    @computed_field
    @property
    def crossline_length(self) -> float:
        """Length of crossline direction (C1 to C4)."""
        return self.corner1.distance_to(self.corner4)
    
    @computed_field
    @property
    def rotation_degrees(self) -> float:
        """
        Grid rotation in degrees (clockwise from North/+Y axis).
        
        Computed from the inline direction (C1 → C2).
        """
        dx = self.corner2.x - self.corner1.x
        dy = self.corner2.y - self.corner1.y
        
        # atan2 gives angle from +X axis, we want from +Y (North)
        angle_rad = math.atan2(dx, dy)
        return math.degrees(angle_rad)
    
    @computed_field
    @property
    def is_axis_aligned(self) -> bool:
        """Check if grid is aligned with coordinate axes."""
        rot = abs(self.rotation_degrees % 90)
        return rot < 0.01 or rot > 89.99
    
    @computed_field
    @property
    def center(self) -> Point2D:
        """Center point of the grid."""
        cx = (self.corner1.x + self.corner2.x + self.corner3.x + self.corner4.x) / 4
        cy = (self.corner1.y + self.corner2.y + self.corner3.y + self.corner4.y) / 4
        return Point2D(x=cx, y=cy)
    
    @computed_field
    @property
    def bounding_box(self) -> dict[str, float]:
        """Axis-aligned bounding box containing all corners."""
        xs = [self.corner1.x, self.corner2.x, self.corner3.x, self.corner4.x]
        ys = [self.corner1.y, self.corner2.y, self.corner3.y, self.corner4.y]
        return {
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
        }
    
    def to_numpy(self) -> np.ndarray:
        """Return corners as (4, 2) numpy array."""
        return np.array([
            [self.corner1.x, self.corner1.y],
            [self.corner2.x, self.corner2.y],
            [self.corner3.x, self.corner3.y],
            [self.corner4.x, self.corner4.y],
        ])
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the grid (using cross-product method)."""
        corners = self.to_numpy()

        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        p = np.array([x, y])

        # Check if point is inside quadrilateral using triangulation
        # Split quad into two triangles and check both
        d1 = sign(p, corners[0], corners[1])
        d2 = sign(p, corners[1], corners[2])
        d3 = sign(p, corners[2], corners[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        in_tri1 = not (has_neg and has_pos)

        d1 = sign(p, corners[0], corners[2])
        d2 = sign(p, corners[2], corners[3])
        d3 = sign(p, corners[3], corners[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        in_tri2 = not (has_neg and has_pos)

        return in_tri1 or in_tri2

    def classify_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        buffer_m: float = 0.0,
    ) -> dict[str, Any]:
        """
        Classify points relative to grid boundary.

        Uses the grid_outliers module for efficient batch classification.

        Args:
            x: X coordinates of points
            y: Y coordinates of points
            buffer_m: Optional buffer distance (positive = expand grid)

        Returns:
            Dictionary with:
            - inside_mask: boolean array
            - signed_distances: float array (negative = inside)
            - n_inside, n_outside: counts
            - inside_ratio: fraction inside
            - suggested_buffer_m: buffer to include all points
        """
        from pstm.analysis.grid_outliers import classify_points_against_grid

        corners = self.to_numpy()
        result = classify_points_against_grid(x, y, corners, buffer_m)

        return {
            "inside_mask": result.inside_mask,
            "signed_distances": result.signed_distances,
            "nearest_edge": result.nearest_edge,
            "n_inside": result.n_inside,
            "n_outside": result.n_outside,
            "inside_ratio": result.inside_ratio,
            "max_distance_outside": result.max_distance_outside,
            "mean_distance_outside": result.mean_distance_outside,
            "outside_by_quadrant": result.outside_by_quadrant,
            "suggested_buffer_m": result.suggested_buffer_m,
        }

    def extend_by_buffer(self, buffer_m: float) -> "CornerPoints":
        """
        Create new CornerPoints extended outward by buffer distance.

        Each corner is moved outward along the diagonal from center.

        Args:
            buffer_m: Buffer distance in meters

        Returns:
            New CornerPoints instance with extended boundaries
        """
        from pstm.analysis.grid_outliers import compute_extended_corners

        corners = self.to_numpy()
        extended = compute_extended_corners(corners, buffer_m)

        return CornerPoints.from_tuples(
            c1=tuple(extended[0]),
            c2=tuple(extended[1]),
            c3=tuple(extended[2]),
            c4=tuple(extended[3]),
        )

    def extend_by_aperture(
        self,
        max_offset_m: float,
        max_dip_deg: float,
    ) -> "CornerPoints":
        """
        Create new CornerPoints extended by migration aperture.

        Traces within aperture distance outside grid can contribute
        to output points near the edge.

        Args:
            max_offset_m: Maximum offset in meters
            max_dip_deg: Maximum dip angle in degrees

        Returns:
            New CornerPoints instance with aperture-extended boundaries
        """
        from pstm.analysis.grid_outliers import compute_aperture_extended_corners

        corners = self.to_numpy()
        extended = compute_aperture_extended_corners(corners, max_offset_m, max_dip_deg)

        return CornerPoints.from_tuples(
            c1=tuple(extended[0]),
            c2=tuple(extended[1]),
            c3=tuple(extended[2]),
            c4=tuple(extended[3]),
        )


# =============================================================================
# Enhanced Output Grid Configuration
# =============================================================================


class OutputGridConfig(BaseModel):
    """
    Enhanced output grid configuration supporting corner-point definition.
    
    The output grid can be defined by:
    1. Bounding box (traditional min/max) - axis-aligned
    2. Corner points (4 corners) - supports rotation
    3. Center + dimensions + rotation
    
    Bin sizes (dx, dy, dt_ms) are INDEPENDENT of input data spacing,
    allowing output at different resolution than input.
    """
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    
    # Definition method
    definition_method: GridDefinitionMethod = Field(
        default=GridDefinitionMethod.BOUNDING_BOX,
        description="Method used to define the grid",
    )
    
    # Corner points (used when definition_method is CORNER_POINTS)
    corners: CornerPoints | None = Field(
        default=None,
        description="Four corner points for rotated grid",
    )
    
    # Bounding box (used when definition_method is BOUNDING_BOX)
    x_min: float | None = Field(default=None, description="Grid X minimum")
    x_max: float | None = Field(default=None, description="Grid X maximum")
    y_min: float | None = Field(default=None, description="Grid Y minimum")
    y_max: float | None = Field(default=None, description="Grid Y maximum")
    
    # Bin sizes (INDEPENDENT of input)
    dx: PositiveFloat = Field(
        default=25.0,
        description="Output inline bin size (meters)",
    )
    dy: PositiveFloat = Field(
        default=25.0,
        description="Output crossline bin size (meters)",
    )
    
    # Time axis
    t_min_ms: NonNegativeFloat = Field(
        default=0.0,
        description="Output time minimum (ms)",
    )
    t_max_ms: PositiveFloat = Field(
        default=4000.0,
        description="Output time maximum (ms)",
    )
    dt_ms: PositiveFloat = Field(
        default=2.0,
        description="Output time sample interval (ms)",
    )
    
    @model_validator(mode="after")
    def validate_grid_definition(self) -> "OutputGridConfig":
        """Ensure grid is properly defined based on method."""
        if self.definition_method == GridDefinitionMethod.CORNER_POINTS:
            if self.corners is None:
                raise ValueError("corners required for CORNER_POINTS method")
        elif self.definition_method == GridDefinitionMethod.BOUNDING_BOX:
            if any(v is None for v in [self.x_min, self.x_max, self.y_min, self.y_max]):
                raise ValueError("x_min, x_max, y_min, y_max required for BOUNDING_BOX method")
            if self.x_max <= self.x_min:
                raise ValueError("x_max must be > x_min")
            if self.y_max <= self.y_min:
                raise ValueError("y_max must be > y_min")
        
        if self.t_max_ms <= self.t_min_ms:
            raise ValueError("t_max_ms must be > t_min_ms")
        
        return self
    
    @classmethod
    def from_bounding_box(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        t_min_ms: float = 0.0,
        t_max_ms: float = 4000.0,
        dx: float = 25.0,
        dy: float = 25.0,
        dt_ms: float = 2.0,
    ) -> "OutputGridConfig":
        """Create grid from traditional bounding box."""
        return cls(
            definition_method=GridDefinitionMethod.BOUNDING_BOX,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            t_min_ms=t_min_ms,
            t_max_ms=t_max_ms,
            dx=dx,
            dy=dy,
            dt_ms=dt_ms,
        )
    
    @classmethod
    def from_corners(
        cls,
        corner1: tuple[float, float],
        corner2: tuple[float, float],
        corner3: tuple[float, float],
        corner4: tuple[float, float],
        t_min_ms: float = 0.0,
        t_max_ms: float = 4000.0,
        dx: float = 25.0,
        dy: float = 25.0,
        dt_ms: float = 2.0,
    ) -> "OutputGridConfig":
        """Create grid from four corner points."""
        corners = CornerPoints.from_tuples(corner1, corner2, corner3, corner4)
        return cls(
            definition_method=GridDefinitionMethod.CORNER_POINTS,
            corners=corners,
            t_min_ms=t_min_ms,
            t_max_ms=t_max_ms,
            dx=dx,
            dy=dy,
            dt_ms=dt_ms,
        )
    
    @classmethod
    def from_origin_rotation(
        cls,
        origin: tuple[float, float],
        inline_length: float,
        crossline_length: float,
        rotation_deg: float = 0.0,
        t_min_ms: float = 0.0,
        t_max_ms: float = 4000.0,
        dx: float = 25.0,
        dy: float = 25.0,
        dt_ms: float = 2.0,
    ) -> "OutputGridConfig":
        """Create grid from origin, dimensions, and rotation."""
        corners = CornerPoints.from_origin_and_dimensions(
            origin=origin,
            inline_length=inline_length,
            crossline_length=crossline_length,
            rotation_deg=rotation_deg,
        )
        return cls(
            definition_method=GridDefinitionMethod.CORNER_POINTS,
            corners=corners,
            t_min_ms=t_min_ms,
            t_max_ms=t_max_ms,
            dx=dx,
            dy=dy,
            dt_ms=dt_ms,
        )
    
    def _get_corners(self) -> CornerPoints:
        """Get corner points, creating from bounding box if needed."""
        if self.corners is not None:
            return self.corners
        return CornerPoints.from_bounding_box(
            self.x_min, self.x_max, self.y_min, self.y_max
        )
    
    @computed_field
    @property
    def inline_length(self) -> float:
        """Total length in inline direction (meters)."""
        corners = self._get_corners()
        return corners.inline_length
    
    @computed_field
    @property
    def crossline_length(self) -> float:
        """Total length in crossline direction (meters)."""
        corners = self._get_corners()
        return corners.crossline_length
    
    @computed_field
    @property
    def rotation_degrees(self) -> float:
        """Grid rotation in degrees."""
        corners = self._get_corners()
        return corners.rotation_degrees
    
    @computed_field
    @property
    def nx(self) -> int:
        """Number of output grid points in inline direction."""
        return int(self.inline_length / self.dx) + 1
    
    @computed_field
    @property
    def ny(self) -> int:
        """Number of output grid points in crossline direction."""
        return int(self.crossline_length / self.dy) + 1
    
    @computed_field
    @property
    def nt(self) -> int:
        """Number of output grid points in time."""
        return int((self.t_max_ms - self.t_min_ms) / self.dt_ms) + 1
    
    @computed_field
    @property
    def shape(self) -> tuple[int, int, int]:
        """Output grid shape (nx, ny, nt)."""
        return (self.nx, self.ny, self.nt)
    
    @computed_field
    @property
    def total_samples(self) -> int:
        """Total number of output samples."""
        return self.nx * self.ny * self.nt
    
    @computed_field
    @property
    def size_gb(self) -> float:
        """Estimated output size in GB (float32)."""
        return self.total_samples * 4 / (1024**3)
    
    def get_output_coordinates(self) -> dict[str, np.ndarray]:
        """
        Generate output grid coordinates.
        
        Returns:
            Dictionary with:
            - 'x': 1D array of inline coordinates
            - 'y': 1D array of crossline coordinates
            - 't_ms': 1D array of time samples
            - 'X': 2D meshgrid of X coordinates
            - 'Y': 2D meshgrid of Y coordinates
        """
        corners = self._get_corners()
        
        # Time axis is always straightforward
        t_ms = np.linspace(self.t_min_ms, self.t_max_ms, self.nt)
        
        if corners.is_axis_aligned:
            # Simple case: axis-aligned grid
            bbox = corners.bounding_box
            x = np.linspace(bbox['x_min'], bbox['x_max'], self.nx)
            y = np.linspace(bbox['y_min'], bbox['y_max'], self.ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
        else:
            # Rotated grid: generate coordinates along inline/crossline axes
            c1 = np.array([corners.corner1.x, corners.corner1.y])
            c2 = np.array([corners.corner2.x, corners.corner2.y])
            c4 = np.array([corners.corner4.x, corners.corner4.y])
            
            # Unit vectors
            inline_vec = (c2 - c1) / np.linalg.norm(c2 - c1)
            xline_vec = (c4 - c1) / np.linalg.norm(c4 - c1)
            
            # Generate 2D grid
            inline_pos = np.linspace(0, self.inline_length, self.nx)
            xline_pos = np.linspace(0, self.crossline_length, self.ny)
            
            IL, XL = np.meshgrid(inline_pos, xline_pos, indexing='ij')
            
            X = c1[0] + IL * inline_vec[0] + XL * xline_vec[0]
            Y = c1[1] + IL * inline_vec[1] + XL * xline_vec[1]
            
            x = X[:, 0]  # Inline coordinates at first crossline
            y = Y[0, :]  # Crossline coordinates at first inline
        
        return {
            'x': x,
            'y': y,
            't_ms': t_ms,
            'X': X,
            'Y': Y,
        }
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the output grid configuration."""
        corners = self._get_corners()
        return {
            "method": self.definition_method.value,
            "shape": f"{self.nx} × {self.ny} × {self.nt}",
            "inline_extent": f"{self.inline_length:.1f} m",
            "crossline_extent": f"{self.crossline_length:.1f} m",
            "time_extent": f"{self.t_min_ms:.0f} - {self.t_max_ms:.0f} ms",
            "bin_size": f"{self.dx:.1f} × {self.dy:.1f} m",
            "sample_rate": f"{self.dt_ms:.1f} ms",
            "rotation": f"{self.rotation_degrees:.1f}°",
            "total_samples": f"{self.total_samples:,}",
            "size_gb": f"{self.size_gb:.2f}",
            "corners": {
                "C1": corners.corner1.to_tuple(),
                "C2": corners.corner2.to_tuple(),
                "C3": corners.corner3.to_tuple(),
                "C4": corners.corner4.to_tuple(),
            }
        }

    def analyze_coverage(
        self,
        mx: np.ndarray,
        my: np.ndarray,
        max_offset_m: float = 5000.0,
        max_dip_deg: float = 45.0,
    ) -> "OutlierReport":
        """
        Analyze midpoint coverage relative to this output grid.

        Args:
            mx: Midpoint X coordinates
            my: Midpoint Y coordinates
            max_offset_m: Maximum offset for aperture calculation
            max_dip_deg: Maximum dip for aperture calculation

        Returns:
            OutlierReport with classification and recommendations
        """
        from pstm.analysis.grid_outliers import generate_outlier_report

        corners = self._get_corners()
        return generate_outlier_report(
            mx, my, corners.to_numpy(),
            max_offset_m=max_offset_m,
            max_dip_deg=max_dip_deg,
        )

    @classmethod
    def auto_calculate_bin_size(
        cls,
        mx: np.ndarray,
        my: np.ndarray,
        method: str = "histogram",
        azimuth_deg: float = 0.0,
    ) -> "BinSizeResult":
        """
        Auto-calculate optimal bin size from midpoint distribution.

        This is a convenience class method that wraps the bin_size module.

        Args:
            mx: Midpoint X coordinates
            my: Midpoint Y coordinates
            method: Algorithm - "histogram", "nearest_neighbor", "fft", or "ensemble"
            azimuth_deg: Acquisition azimuth for rotation alignment

        Returns:
            BinSizeResult with recommended dx, dy and diagnostics
        """
        from pstm.analysis.bin_size import (
            auto_calculate_bin_size,
            auto_calculate_bin_size_ensemble,
            BinSizeMethod,
        )

        if method == "ensemble":
            return auto_calculate_bin_size_ensemble(mx, my, azimuth_deg)
        else:
            return auto_calculate_bin_size(mx, my, BinSizeMethod(method), azimuth_deg)

    def with_auto_bin_size(
        self,
        mx: np.ndarray,
        my: np.ndarray,
        method: str = "histogram",
    ) -> "OutputGridConfig":
        """
        Create a new OutputGridConfig with auto-calculated bin sizes.

        Args:
            mx: Midpoint X coordinates
            my: Midpoint Y coordinates
            method: Algorithm for bin size calculation

        Returns:
            New OutputGridConfig with updated dx, dy
        """
        result = self.auto_calculate_bin_size(
            mx, my, method, self.rotation_degrees
        )

        # Create new config with updated bin sizes
        return OutputGridConfig(
            definition_method=self.definition_method,
            corners=self.corners,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            dx=result.dx,
            dy=result.dy,
            t_min_ms=self.t_min_ms,
            t_max_ms=self.t_max_ms,
            dt_ms=self.dt_ms,
        )

    def extend_for_migration(
        self,
        max_offset_m: float,
        max_dip_deg: float,
    ) -> "OutputGridConfig":
        """
        Create extended grid that includes migration aperture zone.

        Traces within aperture distance outside the grid can contribute
        energy to output points near edges. This creates a grid that
        includes those zones for proper edge handling.

        Args:
            max_offset_m: Maximum offset in meters
            max_dip_deg: Maximum dip angle in degrees

        Returns:
            New OutputGridConfig with extended boundaries
        """
        corners = self._get_corners()
        extended_corners = corners.extend_by_aperture(max_offset_m, max_dip_deg)

        return OutputGridConfig(
            definition_method=GridDefinitionMethod.CORNER_POINTS,
            corners=extended_corners,
            dx=self.dx,
            dy=self.dy,
            t_min_ms=self.t_min_ms,
            t_max_ms=self.t_max_ms,
            dt_ms=self.dt_ms,
        )
