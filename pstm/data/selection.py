"""
Data Selection Module.

Flexible trace filtering based on offset, azimuth, and OVT parameters.
No validation - user has full control and responsibility.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class SelectionMode(str, Enum):
    """Data selection mode."""
    ALL = "all"
    OFFSET_RANGE = "offset_range"
    AZIMUTH_SECTOR = "azimuth_sector"
    OVT = "ovt"
    CUSTOM = "custom"


class AzimuthConvention(str, Enum):
    """Azimuth angle convention."""
    NORTH_0_360 = "north_0_360"  # 0-360°, North=0, clockwise
    NORTH_PM180 = "north_pm180"  # -180 to +180°, North=0
    RECEIVER_RELATIVE = "receiver_relative"  # As recorded in data


# =============================================================================
# Selector Base Class
# =============================================================================


class BaseSelector(ABC):
    """Abstract base class for data selectors."""
    
    @abstractmethod
    def apply(self, headers: pd.DataFrame) -> NDArray[np.bool_]:
        """
        Apply selection to header data.
        
        Args:
            headers: DataFrame with trace headers
            
        Returns:
            Boolean mask (True = include trace)
        """
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """Return human-readable description of selection."""
        pass
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"type": self.__class__.__name__}
    
    @classmethod
    def from_dict(cls, data: dict) -> "BaseSelector":
        """Deserialize from dictionary."""
        raise NotImplementedError


# =============================================================================
# All Data Selector (No Filtering)
# =============================================================================


@dataclass
class AllDataSelector(BaseSelector):
    """Select all traces (no filtering)."""
    
    def apply(self, headers: pd.DataFrame) -> NDArray[np.bool_]:
        return np.ones(len(headers), dtype=bool)
    
    def describe(self) -> str:
        return "All data (no filtering)"
    
    def to_dict(self) -> dict:
        return {"type": "AllDataSelector"}
    
    @classmethod
    def from_dict(cls, data: dict) -> "AllDataSelector":
        return cls()


# =============================================================================
# Offset Range Selector
# =============================================================================


@dataclass
class OffsetRange:
    """A single offset range."""
    min_offset: float | None = None  # None = no limit
    max_offset: float | None = None  # None = no limit
    
    def contains(self, offset: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Check if offsets are within range."""
        mask = np.ones(len(offset), dtype=bool)
        if self.min_offset is not None:
            mask &= offset >= self.min_offset
        if self.max_offset is not None:
            mask &= offset <= self.max_offset
        return mask
    
    def describe(self) -> str:
        min_str = f"{self.min_offset:.0f}" if self.min_offset is not None else "-∞"
        max_str = f"{self.max_offset:.0f}" if self.max_offset is not None else "∞"
        return f"[{min_str}, {max_str}] m"


@dataclass
class OffsetRangeSelector(BaseSelector):
    """
    Select traces by offset range(s).
    
    Supports multiple ranges and include/exclude mode.
    """
    
    ranges: list[OffsetRange] = field(default_factory=list)
    include_negative: bool = True  # Whether to include negative offsets
    exclude_mode: bool = False  # If True, exclude ranges instead of include
    
    # Header column name for offset
    offset_column: str = "offset"
    
    def apply(self, headers: pd.DataFrame) -> NDArray[np.bool_]:
        if self.offset_column not in headers.columns:
            # Try common alternatives
            for col in ["OFFSET", "Offset", "offset_m", "OFFSET_M"]:
                if col in headers.columns:
                    self.offset_column = col
                    break
            else:
                raise KeyError(f"Offset column '{self.offset_column}' not found")
        
        offset = headers[self.offset_column].values.astype(np.float64)
        
        # Handle negative offsets
        if not self.include_negative:
            offset = np.abs(offset)
        
        if not self.ranges:
            # No ranges specified = all data
            return np.ones(len(headers), dtype=bool)
        
        # Combine all ranges
        combined_mask = np.zeros(len(headers), dtype=bool)
        for r in self.ranges:
            combined_mask |= r.contains(offset)
        
        # Apply exclude mode
        if self.exclude_mode:
            combined_mask = ~combined_mask
        
        return combined_mask
    
    def describe(self) -> str:
        if not self.ranges:
            return "All offsets"
        
        mode = "Exclude" if self.exclude_mode else "Include"
        ranges_str = ", ".join(r.describe() for r in self.ranges)
        return f"{mode} offset ranges: {ranges_str}"
    
    def to_dict(self) -> dict:
        return {
            "type": "OffsetRangeSelector",
            "ranges": [
                {"min": r.min_offset, "max": r.max_offset}
                for r in self.ranges
            ],
            "include_negative": self.include_negative,
            "exclude_mode": self.exclude_mode,
            "offset_column": self.offset_column,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "OffsetRangeSelector":
        ranges = [
            OffsetRange(min_offset=r.get("min"), max_offset=r.get("max"))
            for r in data.get("ranges", [])
        ]
        return cls(
            ranges=ranges,
            include_negative=data.get("include_negative", True),
            exclude_mode=data.get("exclude_mode", False),
            offset_column=data.get("offset_column", "offset"),
        )


# =============================================================================
# Azimuth Sector Selector
# =============================================================================


@dataclass
class AzimuthSector:
    """A sector defined by offset and azimuth ranges."""
    
    offset_min: float | None = None
    offset_max: float | None = None
    azimuth_min: float | None = None  # Degrees
    azimuth_max: float | None = None  # Degrees
    active: bool = True
    
    def contains(
        self,
        offset: NDArray[np.float64],
        azimuth: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Check if offset/azimuth pairs are within sector."""
        if not self.active:
            return np.zeros(len(offset), dtype=bool)
        
        mask = np.ones(len(offset), dtype=bool)
        
        # Offset check
        if self.offset_min is not None:
            mask &= offset >= self.offset_min
        if self.offset_max is not None:
            mask &= offset <= self.offset_max
        
        # Azimuth check (handle wraparound)
        if self.azimuth_min is not None and self.azimuth_max is not None:
            if self.azimuth_min <= self.azimuth_max:
                # Normal range
                mask &= (azimuth >= self.azimuth_min) & (azimuth <= self.azimuth_max)
            else:
                # Wraparound (e.g., 350° to 10°)
                mask &= (azimuth >= self.azimuth_min) | (azimuth <= self.azimuth_max)
        elif self.azimuth_min is not None:
            mask &= azimuth >= self.azimuth_min
        elif self.azimuth_max is not None:
            mask &= azimuth <= self.azimuth_max
        
        return mask
    
    def describe(self) -> str:
        parts = []
        if self.offset_min is not None or self.offset_max is not None:
            omin = f"{self.offset_min:.0f}" if self.offset_min is not None else "-∞"
            omax = f"{self.offset_max:.0f}" if self.offset_max is not None else "∞"
            parts.append(f"offset=[{omin},{omax}]m")
        if self.azimuth_min is not None or self.azimuth_max is not None:
            amin = f"{self.azimuth_min:.0f}" if self.azimuth_min is not None else "-∞"
            amax = f"{self.azimuth_max:.0f}" if self.azimuth_max is not None else "∞"
            parts.append(f"azimuth=[{amin},{amax}]°")
        return " & ".join(parts) if parts else "All"


@dataclass
class AzimuthSectorSelector(BaseSelector):
    """
    Select traces by offset-azimuth sectors.
    
    Supports multiple sectors with independent offset/azimuth ranges.
    """
    
    sectors: list[AzimuthSector] = field(default_factory=list)
    azimuth_convention: AzimuthConvention = AzimuthConvention.RECEIVER_RELATIVE
    
    # Header column names
    offset_column: str = "offset"
    azimuth_column: str = "azimuth"
    
    def _find_columns(self, headers: pd.DataFrame) -> tuple[str, str]:
        """Find offset and azimuth columns."""
        offset_col = self.offset_column
        azimuth_col = self.azimuth_column
        
        # Offset column
        if offset_col not in headers.columns:
            for col in ["OFFSET", "Offset", "offset_m"]:
                if col in headers.columns:
                    offset_col = col
                    break
        
        # Azimuth column
        if azimuth_col not in headers.columns:
            for col in ["AZIMUTH", "Azimuth", "azimuth_deg", "azi"]:
                if col in headers.columns:
                    azimuth_col = col
                    break
        
        return offset_col, azimuth_col
    
    def apply(self, headers: pd.DataFrame) -> NDArray[np.bool_]:
        offset_col, azimuth_col = self._find_columns(headers)
        
        if offset_col not in headers.columns:
            raise KeyError(f"Offset column not found")
        if azimuth_col not in headers.columns:
            raise KeyError(f"Azimuth column not found")
        
        offset = headers[offset_col].values.astype(np.float64)
        azimuth = headers[azimuth_col].values.astype(np.float64)
        
        # Normalize azimuth based on convention
        if self.azimuth_convention == AzimuthConvention.NORTH_0_360:
            azimuth = azimuth % 360
        elif self.azimuth_convention == AzimuthConvention.NORTH_PM180:
            azimuth = ((azimuth + 180) % 360) - 180
        
        if not self.sectors:
            return np.ones(len(headers), dtype=bool)
        
        # OR all active sectors
        combined_mask = np.zeros(len(headers), dtype=bool)
        for sector in self.sectors:
            combined_mask |= sector.contains(offset, azimuth)
        
        return combined_mask
    
    def describe(self) -> str:
        active = [s for s in self.sectors if s.active]
        if not active:
            return "No active sectors"
        return f"{len(active)} azimuth sectors"
    
    def to_dict(self) -> dict:
        return {
            "type": "AzimuthSectorSelector",
            "sectors": [
                {
                    "offset_min": s.offset_min,
                    "offset_max": s.offset_max,
                    "azimuth_min": s.azimuth_min,
                    "azimuth_max": s.azimuth_max,
                    "active": s.active,
                }
                for s in self.sectors
            ],
            "azimuth_convention": self.azimuth_convention.value,
            "offset_column": self.offset_column,
            "azimuth_column": self.azimuth_column,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AzimuthSectorSelector":
        sectors = [
            AzimuthSector(
                offset_min=s.get("offset_min"),
                offset_max=s.get("offset_max"),
                azimuth_min=s.get("azimuth_min"),
                azimuth_max=s.get("azimuth_max"),
                active=s.get("active", True),
            )
            for s in data.get("sectors", [])
        ]
        return cls(
            sectors=sectors,
            azimuth_convention=AzimuthConvention(
                data.get("azimuth_convention", "receiver_relative")
            ),
            offset_column=data.get("offset_column", "offset"),
            azimuth_column=data.get("azimuth_column", "azimuth"),
        )


# =============================================================================
# OVT (Offset Vector Tile) Selector
# =============================================================================


@dataclass
class OVTSelector(BaseSelector):
    """
    Select traces by signed offset components (OVT style).
    
    Offset_X = Rx - Sx (positive = receiver East of source)
    Offset_Y = Ry - Sy (positive = receiver North of source)
    
    Signs change according to azimuth quadrant.
    """
    
    # Offset_X range
    offset_x_min: float | None = None
    offset_x_max: float | None = None
    
    # Offset_Y range
    offset_y_min: float | None = None
    offset_y_max: float | None = None
    
    # OVT tile mode
    use_tiles: bool = False
    tile_size_x: float = 500.0  # Offset_X tile size
    tile_size_y: float = 500.0  # Offset_Y tile size
    selected_tiles: list[tuple[int, int]] = field(default_factory=list)
    
    # Header column names (or computed from source/receiver)
    offset_x_column: str | None = None  # If None, compute from SX/RX
    offset_y_column: str | None = None  # If None, compute from SY/RY
    source_x_column: str = "source_x"
    source_y_column: str = "source_y"
    receiver_x_column: str = "receiver_x"
    receiver_y_column: str = "receiver_y"
    
    def _get_offset_components(
        self, headers: pd.DataFrame
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get or compute signed offset components."""
        
        # Try direct columns first
        if self.offset_x_column and self.offset_x_column in headers.columns:
            offset_x = headers[self.offset_x_column].values.astype(np.float64)
        else:
            # Compute from coordinates
            sx_col = self._find_column(headers, self.source_x_column, 
                                        ["source_x", "SOU_X", "sx", "SX"])
            rx_col = self._find_column(headers, self.receiver_x_column,
                                        ["receiver_x", "REC_X", "rx", "RX"])
            offset_x = headers[rx_col].values - headers[sx_col].values
        
        if self.offset_y_column and self.offset_y_column in headers.columns:
            offset_y = headers[self.offset_y_column].values.astype(np.float64)
        else:
            sy_col = self._find_column(headers, self.source_y_column,
                                        ["source_y", "SOU_Y", "sy", "SY"])
            ry_col = self._find_column(headers, self.receiver_y_column,
                                        ["receiver_y", "REC_Y", "ry", "RY"])
            offset_y = headers[ry_col].values - headers[sy_col].values
        
        return offset_x.astype(np.float64), offset_y.astype(np.float64)
    
    def _find_column(
        self, headers: pd.DataFrame, preferred: str, alternatives: list[str]
    ) -> str:
        """Find column in headers."""
        if preferred in headers.columns:
            return preferred
        for col in alternatives:
            if col in headers.columns:
                return col
        raise KeyError(f"Column not found: {preferred} or {alternatives}")
    
    def apply(self, headers: pd.DataFrame) -> NDArray[np.bool_]:
        offset_x, offset_y = self._get_offset_components(headers)
        
        if self.use_tiles and self.selected_tiles:
            # Tile-based selection
            tile_ix = np.floor(offset_x / self.tile_size_x).astype(int)
            tile_iy = np.floor(offset_y / self.tile_size_y).astype(int)
            
            mask = np.zeros(len(headers), dtype=bool)
            for tx, ty in self.selected_tiles:
                mask |= (tile_ix == tx) & (tile_iy == ty)
            return mask
        
        # Range-based selection
        mask = np.ones(len(headers), dtype=bool)
        
        if self.offset_x_min is not None:
            mask &= offset_x >= self.offset_x_min
        if self.offset_x_max is not None:
            mask &= offset_x <= self.offset_x_max
        if self.offset_y_min is not None:
            mask &= offset_y >= self.offset_y_min
        if self.offset_y_max is not None:
            mask &= offset_y <= self.offset_y_max
        
        return mask
    
    def describe(self) -> str:
        if self.use_tiles and self.selected_tiles:
            return f"OVT: {len(self.selected_tiles)} tiles selected"
        
        parts = []
        if self.offset_x_min is not None or self.offset_x_max is not None:
            xmin = f"{self.offset_x_min:.0f}" if self.offset_x_min is not None else "-∞"
            xmax = f"{self.offset_x_max:.0f}" if self.offset_x_max is not None else "∞"
            parts.append(f"Offset_X=[{xmin},{xmax}]")
        if self.offset_y_min is not None or self.offset_y_max is not None:
            ymin = f"{self.offset_y_min:.0f}" if self.offset_y_min is not None else "-∞"
            ymax = f"{self.offset_y_max:.0f}" if self.offset_y_max is not None else "∞"
            parts.append(f"Offset_Y=[{ymin},{ymax}]")
        
        return "OVT: " + " & ".join(parts) if parts else "OVT: All"
    
    def to_dict(self) -> dict:
        return {
            "type": "OVTSelector",
            "offset_x_min": self.offset_x_min,
            "offset_x_max": self.offset_x_max,
            "offset_y_min": self.offset_y_min,
            "offset_y_max": self.offset_y_max,
            "use_tiles": self.use_tiles,
            "tile_size_x": self.tile_size_x,
            "tile_size_y": self.tile_size_y,
            "selected_tiles": self.selected_tiles,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "OVTSelector":
        return cls(
            offset_x_min=data.get("offset_x_min"),
            offset_x_max=data.get("offset_x_max"),
            offset_y_min=data.get("offset_y_min"),
            offset_y_max=data.get("offset_y_max"),
            use_tiles=data.get("use_tiles", False),
            tile_size_x=data.get("tile_size_x", 500.0),
            tile_size_y=data.get("tile_size_y", 500.0),
            selected_tiles=data.get("selected_tiles", []),
        )


# =============================================================================
# Custom Expression Selector
# =============================================================================


@dataclass
class CustomExpressionSelector(BaseSelector):
    """
    Select traces using a custom Python expression.
    
    Available variables:
        offset, azimuth, offset_x, offset_y,
        sx, sy, rx, ry, mx, my, shot_id, trace_id
    
    Example: "(offset >= 500 and offset <= 3000) and (azimuth >= 45 and azimuth <= 135)"
    """
    
    expression: str = ""
    
    # Allowed functions for safe evaluation
    ALLOWED_FUNCTIONS = {
        "abs": np.abs,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "min": np.minimum,
        "max": np.maximum,
        "log": np.log,
        "log10": np.log10,
        "exp": np.exp,
        "floor": np.floor,
        "ceil": np.ceil,
        "round": np.round,
    }
    
    def _find_column(self, headers: pd.DataFrame, name: str) -> NDArray | None:
        """Find column by name or common alternatives."""
        alternatives = {
            "offset": ["offset", "OFFSET", "Offset", "offset_m"],
            "azimuth": ["azimuth", "AZIMUTH", "Azimuth", "azimuth_deg", "azi"],
            "sx": ["source_x", "SOU_X", "sx", "SX", "source_easting"],
            "sy": ["source_y", "SOU_Y", "sy", "SY", "source_northing"],
            "rx": ["receiver_x", "REC_X", "rx", "RX", "receiver_easting"],
            "ry": ["receiver_y", "REC_Y", "ry", "RY", "receiver_northing"],
            "mx": ["midpoint_x", "CDP_X", "mx", "MX", "cdp_x"],
            "my": ["midpoint_y", "CDP_Y", "my", "MY", "cdp_y"],
            "shot_id": ["shot_id", "SHOT_ID", "ffid", "FFID", "shot"],
            "trace_id": ["trace_id", "TRACE_ID", "trace", "tracf"],
        }
        
        cols_to_try = alternatives.get(name, [name])
        for col in cols_to_try:
            if col in headers.columns:
                return headers[col].values
        return None
    
    def _compute_offset_components(
        self, headers: pd.DataFrame
    ) -> tuple[NDArray | None, NDArray | None]:
        """Compute offset_x and offset_y from source/receiver coords."""
        sx = self._find_column(headers, "sx")
        sy = self._find_column(headers, "sy")
        rx = self._find_column(headers, "rx")
        ry = self._find_column(headers, "ry")
        
        offset_x = (rx - sx) if (sx is not None and rx is not None) else None
        offset_y = (ry - sy) if (sy is not None and ry is not None) else None
        
        return offset_x, offset_y
    
    def validate_expression(self) -> tuple[bool, str]:
        """
        Validate expression syntax.
        
        Returns:
            (is_valid, error_message)
        """
        if not self.expression.strip():
            return False, "Expression is empty"
        
        # Check for dangerous constructs
        dangerous = ["import", "exec", "eval", "__", "open", "file", "os.", "sys."]
        expr_lower = self.expression.lower()
        for d in dangerous:
            if d in expr_lower:
                return False, f"Forbidden construct: {d}"
        
        # Try parsing
        try:
            compile(self.expression, "<expression>", "eval")
            return True, "Valid"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    def apply(self, headers: pd.DataFrame) -> NDArray[np.bool_]:
        if not self.expression.strip():
            return np.ones(len(headers), dtype=bool)
        
        # Build namespace with available variables
        namespace = dict(self.ALLOWED_FUNCTIONS)
        namespace["np"] = np
        namespace["True"] = True
        namespace["False"] = False
        
        # Add header columns
        for var in ["offset", "azimuth", "sx", "sy", "rx", "ry", "mx", "my",
                    "shot_id", "trace_id"]:
            col = self._find_column(headers, var)
            if col is not None:
                namespace[var] = col
        
        # Compute offset components
        offset_x, offset_y = self._compute_offset_components(headers)
        if offset_x is not None:
            namespace["offset_x"] = offset_x
        if offset_y is not None:
            namespace["offset_y"] = offset_y
        
        # Evaluate expression
        try:
            result = eval(self.expression, {"__builtins__": {}}, namespace)
            return np.asarray(result, dtype=bool)
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}")
    
    def describe(self) -> str:
        if not self.expression.strip():
            return "No filter expression"
        # Truncate long expressions
        if len(self.expression) > 50:
            return f"Custom: {self.expression[:50]}..."
        return f"Custom: {self.expression}"
    
    def to_dict(self) -> dict:
        return {
            "type": "CustomExpressionSelector",
            "expression": self.expression,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CustomExpressionSelector":
        return cls(expression=data.get("expression", ""))


# =============================================================================
# Combined Data Selector
# =============================================================================


@dataclass
class DataSelector:
    """
    Combined data selector supporting multiple selection modes.
    
    User has full control - no validation or restrictions.
    """
    
    mode: SelectionMode = SelectionMode.ALL
    
    # Individual selectors
    offset_selector: OffsetRangeSelector = field(default_factory=OffsetRangeSelector)
    azimuth_selector: AzimuthSectorSelector = field(default_factory=AzimuthSectorSelector)
    ovt_selector: OVTSelector = field(default_factory=OVTSelector)
    custom_selector: CustomExpressionSelector = field(default_factory=CustomExpressionSelector)
    
    def get_active_selector(self) -> BaseSelector:
        """Get the currently active selector based on mode."""
        if self.mode == SelectionMode.ALL:
            return AllDataSelector()
        elif self.mode == SelectionMode.OFFSET_RANGE:
            return self.offset_selector
        elif self.mode == SelectionMode.AZIMUTH_SECTOR:
            return self.azimuth_selector
        elif self.mode == SelectionMode.OVT:
            return self.ovt_selector
        elif self.mode == SelectionMode.CUSTOM:
            return self.custom_selector
        else:
            return AllDataSelector()
    
    def apply(self, headers: pd.DataFrame) -> NDArray[np.bool_]:
        """Apply current selection to headers."""
        return self.get_active_selector().apply(headers)
    
    def describe(self) -> str:
        """Get description of current selection."""
        return self.get_active_selector().describe()
    
    def count_selected(self, headers: pd.DataFrame) -> tuple[int, int]:
        """
        Count selected traces.
        
        Returns:
            (n_selected, n_total)
        """
        mask = self.apply(headers)
        return int(mask.sum()), len(headers)
    
    def get_selection_stats(
        self, headers: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Get statistics about the selection.
        
        Returns dict with counts, percentages, and warnings.
        """
        mask = self.apply(headers)
        n_selected = int(mask.sum())
        n_total = len(headers)
        pct = 100 * n_selected / n_total if n_total > 0 else 0
        
        stats = {
            "n_selected": n_selected,
            "n_total": n_total,
            "n_excluded": n_total - n_selected,
            "percent_selected": pct,
            "mode": self.mode.value,
            "description": self.describe(),
            "warnings": [],
        }
        
        # Generate warnings (informational only, user is responsible)
        if pct < 10:
            stats["warnings"].append(f"Low selection: only {pct:.1f}% of traces")
        if n_selected == 0:
            stats["warnings"].append("No traces selected!")
        
        return stats
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "mode": self.mode.value,
            "offset_selector": self.offset_selector.to_dict(),
            "azimuth_selector": self.azimuth_selector.to_dict(),
            "ovt_selector": self.ovt_selector.to_dict(),
            "custom_selector": self.custom_selector.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DataSelector":
        """Deserialize from dictionary."""
        return cls(
            mode=SelectionMode(data.get("mode", "all")),
            offset_selector=OffsetRangeSelector.from_dict(
                data.get("offset_selector", {})
            ),
            azimuth_selector=AzimuthSectorSelector.from_dict(
                data.get("azimuth_selector", {})
            ),
            ovt_selector=OVTSelector.from_dict(
                data.get("ovt_selector", {})
            ),
            custom_selector=CustomExpressionSelector.from_dict(
                data.get("custom_selector", {})
            ),
        )


# =============================================================================
# Preset Expressions
# =============================================================================


PRESET_EXPRESSIONS = {
    "near_offsets": "offset < 1500",
    "far_offsets": "offset > 2500",
    "mid_offsets": "(offset >= 1000) and (offset <= 3000)",
    "ns_azimuths": "(azimuth < 45) or (azimuth > 315)",
    "ew_azimuths": "(azimuth >= 45) and (azimuth <= 135)",
    "quadrant_1_ne": "(offset_x > 0) and (offset_y > 0)",
    "quadrant_2_nw": "(offset_x < 0) and (offset_y > 0)",
    "quadrant_3_sw": "(offset_x < 0) and (offset_y < 0)",
    "quadrant_4_se": "(offset_x > 0) and (offset_y < 0)",
    "positive_offset_x": "offset_x > 0",
    "negative_offset_x": "offset_x < 0",
}


def get_preset_expression(name: str) -> str:
    """Get a preset expression by name."""
    return PRESET_EXPRESSIONS.get(name, "")


def list_preset_expressions() -> list[tuple[str, str]]:
    """List all preset expressions as (name, expression) tuples."""
    return list(PRESET_EXPRESSIONS.items())
