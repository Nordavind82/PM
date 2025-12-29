"""
Data selection configuration for flexible trace filtering.

This module provides flexible data selection based on:
- Offset ranges (single or multiple)
- Offset-azimuth sectors
- Offset vector (OVT style) with signed components
- Custom expressions

IMPORTANT: No validation or limits are enforced. 
The user takes full responsibility for selection parameters.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Callable

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)


# =============================================================================
# Enumerations
# =============================================================================


class SelectionMode(str, Enum):
    """Data selection mode."""
    ALL = "all"                         # Use all data
    OFFSET_RANGE = "offset_range"       # Filter by offset range
    OFFSET_AZIMUTH = "offset_azimuth"   # Filter by offset-azimuth sectors
    OFFSET_VECTOR = "offset_vector"     # OVT style with signed offset_x, offset_y
    CUSTOM = "custom"                   # Custom expression


class AzimuthConvention(str, Enum):
    """Azimuth angle convention."""
    NORTH_0_360 = "north_0_360"         # 0-360°, North=0, clockwise
    NORTH_PLUS_MINUS = "north_pm180"    # -180 to +180°, North=0
    RECEIVER_RELATIVE = "receiver_rel"  # As recorded (receiver azimuth)
    SOURCE_RELATIVE = "source_rel"      # Source azimuth


class RangeMode(str, Enum):
    """How to treat multiple ranges."""
    INCLUDE = "include"  # Include traces matching ranges
    EXCLUDE = "exclude"  # Exclude traces matching ranges


# =============================================================================
# Base Configuration
# =============================================================================


class BaseConfig(BaseModel):
    """Base configuration with common settings."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )


# =============================================================================
# Offset Range Selection
# =============================================================================


class OffsetRange(BaseConfig):
    """A single offset range."""
    min_offset: float | None = Field(
        default=None,
        description="Minimum offset (meters). None = no limit.",
    )
    max_offset: float | None = Field(
        default=None,
        description="Maximum offset (meters). None = no limit.",
    )
    
    def matches(self, offset: np.ndarray) -> np.ndarray:
        """Return boolean mask of offsets matching this range."""
        mask = np.ones(len(offset), dtype=bool)
        if self.min_offset is not None:
            mask &= (offset >= self.min_offset)
        if self.max_offset is not None:
            mask &= (offset <= self.max_offset)
        return mask


class OffsetRangeSelector(BaseConfig):
    """
    Select traces based on offset ranges.
    
    Can specify a single range or multiple ranges.
    No validation is performed - user takes responsibility.
    """
    # Single range (simple case)
    min_offset: float | None = Field(
        default=None,
        description="Minimum offset (meters). None = no limit.",
    )
    max_offset: float | None = Field(
        default=None,
        description="Maximum offset (meters). None = no limit.",
    )
    
    # Multiple ranges (advanced)
    ranges: list[OffsetRange] | None = Field(
        default=None,
        description="List of offset ranges for complex selection.",
    )
    
    # How to treat ranges
    mode: RangeMode = Field(
        default=RangeMode.INCLUDE,
        description="Include or exclude traces matching ranges.",
    )
    
    # Sign handling
    include_negative: bool = Field(
        default=True,
        description="Include negative offsets (if present in data).",
    )
    use_absolute: bool = Field(
        default=True,
        description="Use absolute offset values for comparison.",
    )
    
    def apply(self, offset: np.ndarray) -> np.ndarray:
        """
        Apply offset selection and return boolean mask.
        
        Args:
            offset: Array of offset values
            
        Returns:
            Boolean mask (True = include trace)
        """
        # Handle negative offsets
        if self.use_absolute:
            offset_check = np.abs(offset)
        else:
            offset_check = offset
        
        if not self.include_negative and not self.use_absolute:
            negative_mask = offset >= 0
        else:
            negative_mask = np.ones(len(offset), dtype=bool)
        
        # Apply range filters
        if self.ranges is not None and len(self.ranges) > 0:
            # Multiple ranges: OR them together
            range_mask = np.zeros(len(offset), dtype=bool)
            for r in self.ranges:
                range_mask |= r.matches(offset_check)
        else:
            # Single range from min/max
            range_mask = OffsetRange(
                min_offset=self.min_offset,
                max_offset=self.max_offset
            ).matches(offset_check)
        
        # Apply mode
        if self.mode == RangeMode.EXCLUDE:
            range_mask = ~range_mask
        
        return negative_mask & range_mask


# =============================================================================
# Offset-Azimuth Sector Selection
# =============================================================================


class AzimuthSector(BaseConfig):
    """A single sector in offset-azimuth space."""
    
    # Offset range for this sector
    offset_min: float | None = Field(
        default=None,
        description="Minimum offset for sector (meters).",
    )
    offset_max: float | None = Field(
        default=None,
        description="Maximum offset for sector (meters).",
    )
    
    # Azimuth range for this sector
    azimuth_min: float = Field(
        default=0.0,
        description="Minimum azimuth (degrees).",
    )
    azimuth_max: float = Field(
        default=360.0,
        description="Maximum azimuth (degrees).",
    )
    
    # Active flag
    active: bool = Field(
        default=True,
        description="Whether this sector is active.",
    )
    
    def matches(self, offset: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
        """Return boolean mask of traces matching this sector."""
        if not self.active:
            return np.zeros(len(offset), dtype=bool)
        
        mask = np.ones(len(offset), dtype=bool)
        
        # Offset filter
        if self.offset_min is not None:
            mask &= (np.abs(offset) >= self.offset_min)
        if self.offset_max is not None:
            mask &= (np.abs(offset) <= self.offset_max)
        
        # Azimuth filter (handle wrap-around)
        if self.azimuth_min <= self.azimuth_max:
            # Normal case
            mask &= (azimuth >= self.azimuth_min) & (azimuth <= self.azimuth_max)
        else:
            # Wrap-around case (e.g., 350° to 10°)
            mask &= (azimuth >= self.azimuth_min) | (azimuth <= self.azimuth_max)
        
        return mask


class OffsetAzimuthSelector(BaseConfig):
    """
    Select traces based on offset-azimuth sectors.
    
    Allows defining arbitrary sectors in the offset-azimuth polar space.
    Multiple sectors are OR'd together.
    """
    
    sectors: list[AzimuthSector] = Field(
        default_factory=list,
        description="List of azimuth sectors.",
    )
    
    azimuth_convention: AzimuthConvention = Field(
        default=AzimuthConvention.RECEIVER_RELATIVE,
        description="Azimuth angle convention.",
    )
    
    def apply(
        self,
        offset: np.ndarray,
        azimuth: np.ndarray,
    ) -> np.ndarray:
        """
        Apply offset-azimuth selection and return boolean mask.
        
        Args:
            offset: Array of offset values
            azimuth: Array of azimuth values
            
        Returns:
            Boolean mask (True = include trace)
        """
        if len(self.sectors) == 0:
            return np.ones(len(offset), dtype=bool)
        
        # Normalize azimuth based on convention
        az = self._normalize_azimuth(azimuth)
        
        # OR all sectors together
        mask = np.zeros(len(offset), dtype=bool)
        for sector in self.sectors:
            mask |= sector.matches(offset, az)
        
        return mask
    
    def _normalize_azimuth(self, azimuth: np.ndarray) -> np.ndarray:
        """Normalize azimuth to 0-360 range."""
        az = azimuth.copy()
        
        if self.azimuth_convention == AzimuthConvention.NORTH_PLUS_MINUS:
            # Convert -180..180 to 0..360
            az = np.where(az < 0, az + 360, az)
        
        # Ensure 0-360 range
        az = az % 360
        
        return az


# =============================================================================
# Offset Vector (OVT) Selection
# =============================================================================


class OffsetVectorSelector(BaseConfig):
    """
    Select traces using signed offset components (OVT style).

    In OVT (Offset Vector Tile) selection:
    - offset_x = Rx - Sx (positive = receiver East of source)
    - offset_y = Ry - Sy (positive = receiver North of source)

    Signs change according to azimuth quadrant, allowing selection
    of specific source-receiver geometries.

    Supports two modes:
    1. Continuous range: offset_x_min/max and offset_y_min/max
    2. Vector-based tiles: Generate tiles from X/Y edge vectors

    NO VALIDATION - user takes full responsibility for parameters.
    """

    # Offset_X range (signed)
    offset_x_min: float | None = Field(
        default=None,
        description="Minimum offset_x (meters). Negative = receiver West of source.",
    )
    offset_x_max: float | None = Field(
        default=None,
        description="Maximum offset_x (meters). Positive = receiver East of source.",
    )

    # Offset_Y range (signed)
    offset_y_min: float | None = Field(
        default=None,
        description="Minimum offset_y (meters). Negative = receiver South of source.",
    )
    offset_y_max: float | None = Field(
        default=None,
        description="Maximum offset_y (meters). Positive = receiver North of source.",
    )

    # OVT tile definition
    use_tiles: bool = Field(
        default=False,
        description="Select by OVT tiles instead of continuous range.",
    )
    tile_size_x: float = Field(
        default=500.0,
        description="OVT tile size in X direction (meters).",
    )
    tile_size_y: float = Field(
        default=500.0,
        description="OVT tile size in Y direction (meters).",
    )
    selected_tiles: list[tuple[int, int]] | None = Field(
        default=None,
        description="List of (tile_ix, tile_iy) indices to select. None = all tiles in range.",
    )

    # Vector-based tile definition (alternative to fixed tile_size)
    use_vector_tiles: bool = Field(
        default=False,
        description="Use vector-based tile edges instead of uniform tile size.",
    )
    vector_x_edges: list[float] | None = Field(
        default=None,
        description="X offset edge values for tile boundaries (e.g., [-1000,-600,-200,200,600,1000]).",
    )
    vector_y_edges: list[float] | None = Field(
        default=None,
        description="Y offset edge values for tile boundaries (e.g., [-1000,-600,-200,200,600,1000]).",
    )
    
    # Quadrant selection (convenience shortcuts)
    select_quadrant_1: bool = Field(
        default=True,
        description="Select Q1: +offset_x, +offset_y (NE)",
    )
    select_quadrant_2: bool = Field(
        default=True,
        description="Select Q2: -offset_x, +offset_y (NW)",
    )
    select_quadrant_3: bool = Field(
        default=True,
        description="Select Q3: -offset_x, -offset_y (SW)",
    )
    select_quadrant_4: bool = Field(
        default=True,
        description="Select Q4: +offset_x, -offset_y (SE)",
    )
    
    def apply(
        self,
        offset_x: np.ndarray,
        offset_y: np.ndarray,
    ) -> np.ndarray:
        """
        Apply OVT selection and return boolean mask.

        Args:
            offset_x: Array of signed X offset values (Rx - Sx)
            offset_y: Array of signed Y offset values (Ry - Sy)

        Returns:
            Boolean mask (True = include trace)
        """
        n = len(offset_x)

        # Start with all True
        mask = np.ones(n, dtype=bool)

        # Apply continuous range filters
        if self.offset_x_min is not None:
            mask &= (offset_x >= self.offset_x_min)
        if self.offset_x_max is not None:
            mask &= (offset_x <= self.offset_x_max)
        if self.offset_y_min is not None:
            mask &= (offset_y >= self.offset_y_min)
        if self.offset_y_max is not None:
            mask &= (offset_y <= self.offset_y_max)

        # Apply quadrant selection
        if not all([self.select_quadrant_1, self.select_quadrant_2,
                    self.select_quadrant_3, self.select_quadrant_4]):
            quadrant_mask = np.zeros(n, dtype=bool)

            if self.select_quadrant_1:  # +X, +Y
                quadrant_mask |= (offset_x >= 0) & (offset_y >= 0)
            if self.select_quadrant_2:  # -X, +Y
                quadrant_mask |= (offset_x < 0) & (offset_y >= 0)
            if self.select_quadrant_3:  # -X, -Y
                quadrant_mask |= (offset_x < 0) & (offset_y < 0)
            if self.select_quadrant_4:  # +X, -Y
                quadrant_mask |= (offset_x >= 0) & (offset_y < 0)

            mask &= quadrant_mask

        # Apply vector-based tile selection (irregular grid)
        if self.use_vector_tiles and self.vector_x_edges and self.vector_y_edges:
            if self.selected_tiles:
                # Use vector edges for irregular tile boundaries
                tile_ix, tile_iy = self.get_vector_tile_indices(offset_x, offset_y)
                tile_mask = np.zeros(n, dtype=bool)
                for tix, tiy in self.selected_tiles:
                    tile_mask |= (tile_ix == tix) & (tile_iy == tiy)
                mask &= tile_mask
        # Apply regular OVT tile selection
        elif self.use_tiles and self.selected_tiles:
            tile_ix = np.floor(offset_x / self.tile_size_x).astype(int)
            tile_iy = np.floor(offset_y / self.tile_size_y).astype(int)

            tile_mask = np.zeros(n, dtype=bool)
            for tix, tiy in self.selected_tiles:
                tile_mask |= (tile_ix == tix) & (tile_iy == tiy)

            mask &= tile_mask

        return mask

    def get_tile_indices(
        self,
        offset_x: np.ndarray,
        offset_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute tile indices for all traces (regular grid).

        Returns:
            (tile_ix, tile_iy) arrays of tile indices
        """
        tile_ix = np.floor(offset_x / self.tile_size_x).astype(int)
        tile_iy = np.floor(offset_y / self.tile_size_y).astype(int)
        return tile_ix, tile_iy

    def get_vector_tile_indices(
        self,
        offset_x: np.ndarray,
        offset_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute tile indices for vector-based (irregular) tile grid.

        Uses vector_x_edges and vector_y_edges to define irregular tile boundaries.
        Tile index is the bin number (0 to n_edges-2).

        Returns:
            (tile_ix, tile_iy) arrays of tile indices. Values are -1 for out-of-range.
        """
        if not self.vector_x_edges or not self.vector_y_edges:
            raise ValueError("Vector edges not defined")

        x_edges = np.array(sorted(self.vector_x_edges))
        y_edges = np.array(sorted(self.vector_y_edges))

        # np.searchsorted returns bin index (0 = below first edge, n = above last)
        # Subtract 1 to get 0-based tile index, -1 for out of range
        tile_ix = np.searchsorted(x_edges, offset_x, side='right') - 1
        tile_iy = np.searchsorted(y_edges, offset_y, side='right') - 1

        # Mark out-of-range as -1
        tile_ix = np.where((tile_ix < 0) | (tile_ix >= len(x_edges) - 1), -1, tile_ix)
        tile_iy = np.where((tile_iy < 0) | (tile_iy >= len(y_edges) - 1), -1, tile_iy)

        return tile_ix, tile_iy

    def generate_tiles_from_vectors(self) -> list[tuple[int, int, float, float, float, float]]:
        """
        Generate all tile definitions from vector edges.

        Returns:
            List of (tile_ix, tile_iy, x_min, x_max, y_min, y_max) tuples
        """
        if not self.vector_x_edges or not self.vector_y_edges:
            return []

        x_edges = sorted(self.vector_x_edges)
        y_edges = sorted(self.vector_y_edges)

        tiles = []
        for ix in range(len(x_edges) - 1):
            for iy in range(len(y_edges) - 1):
                tiles.append((
                    ix, iy,
                    x_edges[ix], x_edges[ix + 1],
                    y_edges[iy], y_edges[iy + 1],
                ))
        return tiles

    def get_tile_info(self) -> list[dict]:
        """
        Get detailed tile information for display/export.

        Returns:
            List of dicts with tile_ix, tile_iy, x_min, x_max, y_min, y_max, center_x, center_y
        """
        if self.use_vector_tiles and self.vector_x_edges and self.vector_y_edges:
            tiles = self.generate_tiles_from_vectors()
            return [
                {
                    "tile_ix": t[0],
                    "tile_iy": t[1],
                    "x_min": t[2],
                    "x_max": t[3],
                    "y_min": t[4],
                    "y_max": t[5],
                    "center_x": (t[2] + t[3]) / 2,
                    "center_y": (t[4] + t[5]) / 2,
                }
                for t in tiles
            ]
        elif self.selected_tiles:
            return [
                {
                    "tile_ix": tix,
                    "tile_iy": tiy,
                    "x_min": tix * self.tile_size_x,
                    "x_max": (tix + 1) * self.tile_size_x,
                    "y_min": tiy * self.tile_size_y,
                    "y_max": (tiy + 1) * self.tile_size_y,
                    "center_x": (tix + 0.5) * self.tile_size_x,
                    "center_y": (tiy + 0.5) * self.tile_size_y,
                }
                for tix, tiy in self.selected_tiles
            ]
        return []

    @classmethod
    def from_vectors(
        cls,
        x_vector: list[float],
        y_vector: list[float],
        select_all: bool = True,
    ) -> "OffsetVectorSelector":
        """
        Create OVT selector from X and Y offset vectors.

        Vectors define tile edges. For example:
            x_vector = [-1000, -600, -200, 200, 600, 1000]
        Creates 5 tiles in X direction: [-1000,-600], [-600,-200], [-200,200], [200,600], [600,1000]

        Args:
            x_vector: Sorted list of X offset edge values
            y_vector: Sorted list of Y offset edge values
            select_all: If True, select all tiles; if False, selected_tiles must be set manually

        Returns:
            Configured OffsetVectorSelector
        """
        x_edges = sorted(x_vector)
        y_edges = sorted(y_vector)

        # Generate all tile indices if select_all
        selected = None
        if select_all:
            selected = [
                (ix, iy)
                for ix in range(len(x_edges) - 1)
                for iy in range(len(y_edges) - 1)
            ]

        return cls(
            use_vector_tiles=True,
            use_tiles=True,
            vector_x_edges=x_edges,
            vector_y_edges=y_edges,
            selected_tiles=selected,
            offset_x_min=min(x_edges),
            offset_x_max=max(x_edges),
            offset_y_min=min(y_edges),
            offset_y_max=max(y_edges),
        )


# =============================================================================
# Custom Expression Selection
# =============================================================================


class CustomExpressionSelector(BaseConfig):
    """
    Select traces using a custom Python expression.
    
    The expression is evaluated with numpy and has access to:
    - offset: Absolute offset
    - azimuth: Azimuth angle
    - offset_x: Signed X offset (Rx - Sx)
    - offset_y: Signed Y offset (Ry - Sy)
    - sx, sy: Source coordinates
    - rx, ry: Receiver coordinates
    - mx, my: Midpoint coordinates
    
    Example expressions:
    - "(offset >= 500) & (offset <= 3000)"
    - "(azimuth >= 45) & (azimuth <= 135)"
    - "(offset_x > 0) & (offset_y > 0)"  # Quadrant 1
    - "(offset < 2000) | ((offset > 3000) & (azimuth < 90))"
    
    NO VALIDATION - user takes full responsibility.
    """
    
    expression: str = Field(
        description="Python/numpy expression that evaluates to boolean array.",
    )
    
    def apply(self, variables: dict[str, np.ndarray]) -> np.ndarray:
        """
        Evaluate expression and return boolean mask.
        
        Args:
            variables: Dictionary with arrays for each variable
            
        Returns:
            Boolean mask (True = include trace)
        """
        # Build namespace with numpy functions
        namespace = {
            'np': np,
            'abs': np.abs,
            'sqrt': np.sqrt,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'min': np.minimum,
            'max': np.maximum,
            'floor': np.floor,
            'ceil': np.ceil,
            'mod': np.mod,
        }
        
        # Add variable arrays
        namespace.update(variables)
        
        # Evaluate expression
        try:
            result = eval(self.expression, {"__builtins__": {}}, namespace)
            return np.asarray(result, dtype=bool)
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}")


# =============================================================================
# Main Data Selection Configuration
# =============================================================================


class DataSelectionConfig(BaseConfig):
    """
    Root configuration for data selection.
    
    This defines which input traces are used for migration based on:
    - Offset ranges
    - Offset-azimuth sectors
    - OVT (offset vector) tiles
    - Custom expressions
    
    IMPORTANT: No validation is enforced. The user takes full 
    responsibility for selection parameters and their impact on
    migration results (fold, coverage, artifacts).
    
    The selection is applied BEFORE migration starts, so only
    traces passing the selection criteria are loaded and processed.
    """
    
    # Selection mode
    mode: SelectionMode = Field(
        default=SelectionMode.ALL,
        description="Data selection mode.",
    )
    
    # Mode-specific selectors
    offset_selector: OffsetRangeSelector | None = Field(
        default=None,
        description="Offset range selection parameters.",
    )
    
    offset_azimuth_selector: OffsetAzimuthSelector | None = Field(
        default=None,
        description="Offset-azimuth sector selection parameters.",
    )
    
    offset_vector_selector: OffsetVectorSelector | None = Field(
        default=None,
        description="OVT-style offset vector selection parameters.",
    )
    
    custom_selector: CustomExpressionSelector | None = Field(
        default=None,
        description="Custom expression selection parameters.",
    )
    
    # Combination mode (when multiple selectors active)
    combine_mode: str = Field(
        default="and",
        pattern="^(and|or)$",
        description="How to combine multiple selectors: 'and' or 'or'.",
    )
    
    # User acknowledgment (for GUI)
    user_acknowledged: bool = Field(
        default=False,
        description="User has acknowledged responsibility for selection.",
    )
    
    @classmethod
    def use_all(cls) -> "DataSelectionConfig":
        """Create config that uses all data (no filtering)."""
        return cls(mode=SelectionMode.ALL)
    
    @classmethod
    def by_offset_range(
        cls,
        min_offset: float | None = None,
        max_offset: float | None = None,
    ) -> "DataSelectionConfig":
        """Create config for simple offset range selection."""
        return cls(
            mode=SelectionMode.OFFSET_RANGE,
            offset_selector=OffsetRangeSelector(
                min_offset=min_offset,
                max_offset=max_offset,
            ),
        )
    
    @classmethod
    def by_expression(cls, expression: str) -> "DataSelectionConfig":
        """Create config for custom expression selection."""
        return cls(
            mode=SelectionMode.CUSTOM,
            custom_selector=CustomExpressionSelector(expression=expression),
        )
    
    def apply(self, headers: dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply data selection and return boolean mask.
        
        Args:
            headers: Dictionary with header arrays:
                - offset: Absolute offset
                - azimuth: Azimuth (if available)
                - offset_x: Signed X offset (if available)
                - offset_y: Signed Y offset (if available)
                - sx, sy, rx, ry, mx, my: Coordinates (if available)
        
        Returns:
            Boolean mask (True = include trace)
        """
        n = len(headers.get('offset', headers.get('sx', [])))
        
        if self.mode == SelectionMode.ALL:
            return np.ones(n, dtype=bool)
        
        masks = []
        
        # Offset range selection
        if self.mode == SelectionMode.OFFSET_RANGE and self.offset_selector:
            if 'offset' in headers:
                masks.append(self.offset_selector.apply(headers['offset']))
        
        # Offset-azimuth selection
        if self.mode == SelectionMode.OFFSET_AZIMUTH and self.offset_azimuth_selector:
            if 'offset' in headers and 'azimuth' in headers:
                masks.append(self.offset_azimuth_selector.apply(
                    headers['offset'],
                    headers['azimuth']
                ))
        
        # OVT selection
        if self.mode == SelectionMode.OFFSET_VECTOR and self.offset_vector_selector:
            offset_x = headers.get('offset_x')
            offset_y = headers.get('offset_y')
            
            # Compute offset_x/offset_y if not provided
            if offset_x is None and 'rx' in headers and 'sx' in headers:
                offset_x = headers['rx'] - headers['sx']
            if offset_y is None and 'ry' in headers and 'sy' in headers:
                offset_y = headers['ry'] - headers['sy']
            
            if offset_x is not None and offset_y is not None:
                masks.append(self.offset_vector_selector.apply(offset_x, offset_y))
        
        # Custom expression
        if self.mode == SelectionMode.CUSTOM and self.custom_selector:
            masks.append(self.custom_selector.apply(headers))
        
        # Combine masks
        if len(masks) == 0:
            return np.ones(n, dtype=bool)
        elif len(masks) == 1:
            return masks[0]
        else:
            if self.combine_mode == "and":
                result = masks[0]
                for m in masks[1:]:
                    result = result & m
                return result
            else:  # "or"
                result = masks[0]
                for m in masks[1:]:
                    result = result | m
                return result
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of selection configuration."""
        summary = {
            "mode": self.mode.value,
        }
        
        if self.mode == SelectionMode.OFFSET_RANGE and self.offset_selector:
            s = self.offset_selector
            summary["offset_range"] = f"{s.min_offset or '-∞'} to {s.max_offset or '+∞'} m"
            if s.ranges:
                summary["multiple_ranges"] = len(s.ranges)
        
        if self.mode == SelectionMode.OFFSET_AZIMUTH and self.offset_azimuth_selector:
            s = self.offset_azimuth_selector
            summary["sectors"] = len(s.sectors)
            summary["azimuth_convention"] = s.azimuth_convention.value
        
        if self.mode == SelectionMode.OFFSET_VECTOR and self.offset_vector_selector:
            s = self.offset_vector_selector
            summary["offset_x_range"] = f"{s.offset_x_min or '-∞'} to {s.offset_x_max or '+∞'} m"
            summary["offset_y_range"] = f"{s.offset_y_min or '-∞'} to {s.offset_y_max or '+∞'} m"
            if s.use_tiles:
                summary["tile_size"] = f"{s.tile_size_x} × {s.tile_size_y} m"
                if s.selected_tiles:
                    summary["selected_tiles"] = len(s.selected_tiles)
        
        if self.mode == SelectionMode.CUSTOM and self.custom_selector:
            summary["expression"] = self.custom_selector.expression[:50] + "..."
        
        return summary


# =============================================================================
# Preset Factories
# =============================================================================


def near_offset_selection(max_offset: float = 1500.0) -> DataSelectionConfig:
    """Create selection for near offsets only."""
    return DataSelectionConfig.by_offset_range(max_offset=max_offset)


def far_offset_selection(min_offset: float = 2500.0) -> DataSelectionConfig:
    """Create selection for far offsets only."""
    return DataSelectionConfig.by_offset_range(min_offset=min_offset)


def north_south_azimuth_selection() -> DataSelectionConfig:
    """Create selection for North-South trending azimuths."""
    return DataSelectionConfig(
        mode=SelectionMode.OFFSET_AZIMUTH,
        offset_azimuth_selector=OffsetAzimuthSelector(
            sectors=[
                AzimuthSector(azimuth_min=0, azimuth_max=45),
                AzimuthSector(azimuth_min=315, azimuth_max=360),
                AzimuthSector(azimuth_min=135, azimuth_max=225),
            ]
        )
    )


def east_west_azimuth_selection() -> DataSelectionConfig:
    """Create selection for East-West trending azimuths."""
    return DataSelectionConfig(
        mode=SelectionMode.OFFSET_AZIMUTH,
        offset_azimuth_selector=OffsetAzimuthSelector(
            sectors=[
                AzimuthSector(azimuth_min=45, azimuth_max=135),
                AzimuthSector(azimuth_min=225, azimuth_max=315),
            ]
        )
    )


def quadrant_selection(
    q1: bool = True,
    q2: bool = True,
    q3: bool = True,
    q4: bool = True,
) -> DataSelectionConfig:
    """Create selection for specific OVT quadrants."""
    return DataSelectionConfig(
        mode=SelectionMode.OFFSET_VECTOR,
        offset_vector_selector=OffsetVectorSelector(
            select_quadrant_1=q1,
            select_quadrant_2=q2,
            select_quadrant_3=q3,
            select_quadrant_4=q4,
        )
    )
