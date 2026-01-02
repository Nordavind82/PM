"""Core data types for seismic viewer."""

from dataclasses import dataclass


@dataclass
class AxisConfig:
    """Configuration for an axis."""
    label: str
    min_val: float
    max_val: float
    unit: str = ""

    @property
    def range(self) -> float:
        return self.max_val - self.min_val


@dataclass
class ViewState:
    """Current view state (zoom and pan in data coordinates)."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def copy(self) -> 'ViewState':
        return ViewState(self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def x_range(self) -> float:
        return self.x_max - self.x_min

    @property
    def y_range(self) -> float:
        return self.y_max - self.y_min
