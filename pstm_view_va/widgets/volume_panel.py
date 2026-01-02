"""Volume panel widget for seismic cube visualization."""

from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import zarr

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog
)
from PyQt6.QtCore import pyqtSignal

from ..core import AxisConfig, ViewState
from .canvas import SeismicCanvas


class VolumePanel(QFrame):
    """Panel for displaying one seismic volume."""

    slice_changed = pyqtSignal(int)
    view_changed = pyqtSignal(object)  # ViewState

    def __init__(self, title: str = "Volume", parent=None):
        super().__init__(parent)
        self.title = title
        self.setFrameStyle(QFrame.Shape.StyledPanel)

        # Data
        self.cube: Optional[np.ndarray] = None
        self.cube_shape: Tuple[int, int, int] = (0, 0, 0)
        self.current_direction = "inline"
        self.current_index = 0
        self.slice_value = 0.0  # Current slice coordinate value
        self.file_path: Optional[str] = None  # Loaded file path for state recovery

        # Coordinate arrays (optional, for proper axis values)
        self.x_coords: Optional[np.ndarray] = None  # Inline coordinates
        self.y_coords: Optional[np.ndarray] = None  # Crossline coordinates
        self.t_coords: Optional[np.ndarray] = None  # Time coordinates (ms)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with file info
        header = QHBoxLayout()
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header.addWidget(self.title_label)

        self.open_btn = QPushButton("Open...")
        self.open_btn.setMaximumWidth(80)
        self.open_btn.clicked.connect(self.open_file)
        header.addWidget(self.open_btn)

        layout.addLayout(header)

        # File label
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.file_label)

        # Canvas
        self.canvas = SeismicCanvas()
        self.canvas.slice_changed.connect(self._on_canvas_slice_changed)
        self.canvas.view_changed.connect(self.view_changed.emit)
        layout.addWidget(self.canvas, 1)

    def open_file(self):
        """Open a zarr file."""
        path = QFileDialog.getExistingDirectory(
            self, "Open Zarr Directory",
            str(Path.home() / "SeismicData")
        )
        if path:
            self.load_zarr(path)

    def load_zarr(self, path: str):
        """Load zarr data."""
        try:
            z = zarr.open(path, mode='r')

            # Try different array names
            if isinstance(z, zarr.Array):
                data = np.asarray(z)
            elif 'cube' in z:
                data = np.asarray(z['cube'])
            elif 'migrated' in z:
                data = np.asarray(z['migrated'])
            elif 'data' in z:
                data = np.asarray(z['data'])
            elif 'c' in z:
                data = np.asarray(z['c'])
            else:
                data = np.asarray(z)

            if data.ndim != 3:
                raise ValueError(f"Expected 3D data, got {data.ndim}D")

            self.cube = data
            self.cube_shape = data.shape
            self.file_path = path  # Store path for state recovery

            # Try to load coordinate arrays
            self._load_coordinates(z, path)

            self.file_label.setText(f"{Path(path).name} | {data.shape}")
            self.title_label.setText(f"{self.title}: {Path(path).name}")

            # Update display
            self.update_display()

        except Exception as e:
            self.file_label.setText(f"Error: {e}")

    def _load_coordinates(self, z, path: str):
        """Set up index-based inline/crossline/time coordinates."""
        nx, ny, nt = self.cube_shape

        # Use simple index-based coordinates (inline, crossline numbers)
        self.x_coords = np.arange(nx)  # Inline indices
        self.y_coords = np.arange(ny)  # Crossline indices

        # Time in ms (try to get dt from attrs, default 2ms)
        dt_ms = 2.0
        try:
            if hasattr(z, 'attrs') and 'dt_ms' in z.attrs:
                dt_ms = float(z.attrs['dt_ms'])
        except:
            pass
        self.t_coords = np.arange(nt) * dt_ms

    def set_direction(self, direction: str):
        """Set slice direction."""
        self.current_direction = direction
        if self.cube is not None:
            if direction == "inline":
                max_idx = self.cube_shape[0] - 1
            elif direction == "crossline":
                max_idx = self.cube_shape[1] - 1
            else:
                max_idx = self.cube_shape[2] - 1

            self.current_index = min(self.current_index, max_idx)
            self.update_display()

    def set_slice_index(self, index: int):
        """Set slice index."""
        self.current_index = index
        self.update_display()

    def set_view(self, view: ViewState):
        """Set view state for synchronization."""
        self.canvas.set_view(view)

    def set_palette(self, name: str):
        """Set color palette."""
        self.canvas.set_palette(name)

    def set_gain(self, gain: float):
        """Set display gain."""
        self.canvas.set_gain(gain)

    def set_clip_percentile(self, pct: float):
        """Set clip percentile."""
        self.canvas.set_clip_percentile(pct)

    def reset_view(self):
        """Reset view to full extent."""
        self.canvas.reset_view()

    def _on_canvas_slice_changed(self, new_idx: int):
        """Handle slice change from canvas."""
        direction = self.current_direction
        if self.cube is not None:
            if direction == "inline":
                max_idx = self.cube_shape[0] - 1
            elif direction == "crossline":
                max_idx = self.cube_shape[1] - 1
            else:
                max_idx = self.cube_shape[2] - 1

            # Apply step (could be > 1)
            if new_idx > self.current_index:
                self.current_index = min(self.current_index + 1, max_idx)
            else:
                self.current_index = max(self.current_index - 1, 0)

            self.slice_changed.emit(self.current_index)
            self.update_display()

    def update_display(self):
        """Update the canvas with current slice."""
        if self.cube is None:
            return

        idx = self.current_index
        direction = self.current_direction

        if direction == "inline":
            if idx >= self.cube_shape[0]:
                idx = self.cube_shape[0] - 1
            data = self.cube[idx, :, :].T  # (crossline, time) -> time vertical
            x_axis = AxisConfig("XL", float(self.y_coords[0]), float(self.y_coords[-1]))
            y_axis = AxisConfig("Time", float(self.t_coords[0]), float(self.t_coords[-1]), "ms")
            self.slice_value = float(self.x_coords[idx]) if idx < len(self.x_coords) else idx
            max_idx = self.cube_shape[0] - 1
        elif direction == "crossline":
            if idx >= self.cube_shape[1]:
                idx = self.cube_shape[1] - 1
            data = self.cube[:, idx, :].T  # (inline, time) -> time vertical
            x_axis = AxisConfig("IL", float(self.x_coords[0]), float(self.x_coords[-1]))
            y_axis = AxisConfig("Time", float(self.t_coords[0]), float(self.t_coords[-1]), "ms")
            self.slice_value = float(self.y_coords[idx]) if idx < len(self.y_coords) else idx
            max_idx = self.cube_shape[1] - 1
        else:  # time
            if idx >= self.cube_shape[2]:
                idx = self.cube_shape[2] - 1
            data = self.cube[:, :, idx].T  # (inline, crossline)
            x_axis = AxisConfig("IL", float(self.x_coords[0]), float(self.x_coords[-1]))
            y_axis = AxisConfig("XL", float(self.y_coords[0]), float(self.y_coords[-1]))
            self.slice_value = float(self.t_coords[idx]) if idx < len(self.t_coords) else idx
            max_idx = self.cube_shape[2] - 1

        self.canvas.set_data(data, direction, idx, max_idx, x_axis, y_axis, self.slice_value)
