"""Gather panel widget for CIG and common offset gather visualization."""

import json
from typing import Optional, Tuple, List
from pathlib import Path
import numpy as np
import zarr

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox
)

from ..core import AxisConfig
from .canvas import SeismicCanvas


class GatherPanel(QFrame):
    """Panel for displaying seismic gathers (CIG or extracted from common offset)."""

    def __init__(self, title: str = "Gathers", parent=None):
        super().__init__(parent)
        self.title = title
        self.setFrameStyle(QFrame.Shape.StyledPanel)

        # Data - can be either CIG or common offset gathers
        self.gather_data: Optional[zarr.Array] = None
        self.gather_store = None  # Keep zarr store reference
        self.gather_shape: Tuple = ()
        self.gather_type = "cig"  # "cig" or "common_offset_folder"
        self.file_path: Optional[str] = None  # Loaded file path for state recovery

        # For folder-based common offset gathers
        self.offset_bins: List[zarr.Array] = []  # List of zarr arrays per offset
        self.metadata: Optional[dict] = None

        # Coordinate arrays
        self.il_coords: Optional[np.ndarray] = None
        self.xl_coords: Optional[np.ndarray] = None
        self.offset_coords: Optional[np.ndarray] = None
        self.t_coords: Optional[np.ndarray] = None

        # Current position
        self.current_il_idx = 0
        self.current_xl_idx = 0

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Ensure panel has minimum size
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        # Header with file info
        header = QHBoxLayout()
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header.addWidget(self.title_label)

        self.open_btn = QPushButton("Open Gathers...")
        self.open_btn.setMaximumWidth(120)
        self.open_btn.clicked.connect(self.open_file)
        header.addWidget(self.open_btn)

        layout.addLayout(header)

        # Gather type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["CIG (4D zarr)", "Common Offset (folder)"])
        self.type_combo.setCurrentIndex(1)  # Default to folder-based
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # File label
        self.file_label = QLabel("No gathers loaded")
        self.file_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.file_label)

        # Position label
        self.position_label = QLabel("Position: IL=- XL=-")
        self.position_label.setStyleSheet("color: #4a9eff; font-size: 11px;")
        layout.addWidget(self.position_label)

        # Canvas
        self.canvas = SeismicCanvas()
        layout.addWidget(self.canvas, 1)

    def open_file(self):
        """Open a zarr file containing gathers."""
        path = QFileDialog.getExistingDirectory(
            self, "Open Zarr Gathers Directory",
            str(Path.home() / "SeismicData")
        )
        if path:
            self.load_zarr(path)

    def load_zarr(self, path: str):
        """Load zarr gather data - supports both 4D zarr and folder-based common offset."""
        path = Path(path)
        self.file_path = str(path)  # Store path for state recovery

        try:
            # Check for folder-based common offset gathers (has metadata.json)
            metadata_path = path / 'gather_metadata.json'
            if metadata_path.exists():
                self._load_folder_gathers(path, metadata_path)
                return

            # Try loading as single 4D zarr
            z = zarr.open(str(path))

            # Try different array names
            if isinstance(z, zarr.Array):
                self.gather_data = z
            elif 'gathers' in z:
                self.gather_data = z['gathers']
            elif 'data' in z:
                self.gather_data = z['data']
            elif 'cube' in z:
                self.gather_data = z['cube']
            else:
                # Try to find any array
                for key in z.keys():
                    if isinstance(z[key], zarr.Array):
                        self.gather_data = z[key]
                        break
                if self.gather_data is None:
                    self.gather_data = z

            self.gather_store = z
            self.gather_shape = self.gather_data.shape
            self.gather_type = "cig"
            self.type_combo.setCurrentIndex(0)

            if len(self.gather_shape) != 4:
                raise ValueError(f"Expected 4D gather data, got {len(self.gather_shape)}D")

            # Load coordinates
            self._load_coordinates(z, str(path))

            self.file_label.setText(f"{path.name} | {self.gather_shape}")
            self.title_label.setText(f"{self.title}: {path.name}")

            # Update display at center position
            self._set_center_position()

        except Exception as e:
            self.file_label.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _load_folder_gathers(self, base_path: Path, metadata_path: Path):
        """Load folder-based common offset gathers using metadata JSON."""
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.gather_type = "common_offset_folder"
        self.type_combo.setCurrentIndex(1)

        # Load all offset bins (lazy - just open zarr references)
        self.offset_bins = []
        data_array_name = self.metadata.get('data_array', 'migrated_stack.zarr')

        for offset_info in self.metadata['offsets']:
            bin_path = base_path / offset_info['bin_name'] / data_array_name
            z = zarr.open(str(bin_path))
            self.offset_bins.append(z)

        # Set coordinates - use indices for inline/crossline
        dims = self.metadata['dimensions']

        self.il_coords = np.arange(dims['n_inline'])    # Inline indices
        self.xl_coords = np.arange(dims['n_crossline']) # Crossline indices
        self.t_coords = np.linspace(
            self.metadata['coordinates']['time']['min'],
            self.metadata['coordinates']['time']['max'],
            dims['n_time']
        )
        self.offset_coords = np.array([o['offset_m'] for o in self.metadata['offsets']])

        self.gather_shape = (
            dims['n_offsets'],
            dims['n_inline'],
            dims['n_crossline'],
            dims['n_time']
        )

        n_off = len(self.offset_bins)
        self.file_label.setText(
            f"{base_path.name} | {n_off} offsets, "
            f"{dims['n_inline']}x{dims['n_crossline']}x{dims['n_time']}"
        )
        self.title_label.setText(f"{self.title}: {base_path.name}")

        # Update display at center position
        self._set_center_position()

    def _load_coordinates(self, z, path: str):
        """Load coordinate arrays from zarr attributes."""
        # Determine dimensions based on gather type
        if self.gather_type == "cig":
            # CIG: (inline, crossline, offset, time)
            n_il, n_xl, n_off, n_t = self.gather_shape
        else:
            # Common offset: (offset, inline, crossline, time)
            n_off, n_il, n_xl, n_t = self.gather_shape

        # Try to get from zarr attributes
        try:
            if hasattr(z, 'attrs'):
                attrs = dict(z.attrs)
                if 'il_coords' in attrs:
                    self.il_coords = np.array(attrs['il_coords'])
                if 'xl_coords' in attrs:
                    self.xl_coords = np.array(attrs['xl_coords'])
                if 'offset_coords' in attrs:
                    self.offset_coords = np.array(attrs['offset_coords'])
                if 't_coords' in attrs:
                    self.t_coords = np.array(attrs['t_coords'])
        except:
            pass

        # Use index-based coordinates if not found
        if self.il_coords is None:
            self.il_coords = np.arange(n_il)
        if self.xl_coords is None:
            self.xl_coords = np.arange(n_xl)
        if self.offset_coords is None:
            self.offset_coords = np.arange(n_off) * 100  # Assume 100m offset increment
        if self.t_coords is None:
            self.t_coords = np.arange(n_t) * 2.0  # Assume 2ms sampling

    def _set_center_position(self):
        """Set position to center of survey."""
        if self.gather_type == "cig":
            n_il, n_xl = self.gather_shape[0], self.gather_shape[1]
        elif self.gather_type == "common_offset_folder":
            n_il = self.metadata['dimensions']['n_inline']
            n_xl = self.metadata['dimensions']['n_crossline']
        else:
            n_il, n_xl = self.gather_shape[1], self.gather_shape[2]

        self.current_il_idx = n_il // 2
        self.current_xl_idx = n_xl // 2
        self.update_display()

    def _on_type_changed(self, index: int):
        """Handle gather type change."""
        self.gather_type = "cig" if index == 0 else "common_offset"
        if self.gather_data is not None:
            # Re-parse coordinates
            self._load_coordinates(self.gather_store, "")
            self._set_center_position()

    def set_position(self, il_val: float, xl_val: float):
        """Set CIG position by inline/crossline value (finds nearest)."""
        if self.gather_type == "common_offset_folder":
            if len(self.offset_bins) == 0:
                return
        elif self.gather_data is None:
            return

        # Find nearest indices
        if self.il_coords is not None:
            self.current_il_idx = int(np.argmin(np.abs(self.il_coords - il_val)))
        if self.xl_coords is not None:
            self.current_xl_idx = int(np.argmin(np.abs(self.xl_coords - xl_val)))

        self.update_display()

    def set_position_by_index(self, il_idx: int, xl_idx: int):
        """Set CIG position by index."""
        if self.gather_type == "cig":
            if self.gather_data is None:
                return
            max_il = self.gather_shape[0] - 1
            max_xl = self.gather_shape[1] - 1
        elif self.gather_type == "common_offset_folder":
            if len(self.offset_bins) == 0:
                return
            max_il = self.metadata['dimensions']['n_inline'] - 1
            max_xl = self.metadata['dimensions']['n_crossline'] - 1
        else:
            if self.gather_data is None:
                return
            max_il = self.gather_shape[1] - 1
            max_xl = self.gather_shape[2] - 1

        self.current_il_idx = max(0, min(il_idx, max_il))
        self.current_xl_idx = max(0, min(xl_idx, max_xl))
        self.update_display()

    def extract_cig(self) -> Optional[np.ndarray]:
        """Extract CIG at current position.
        For folder-based common offset gathers, this performs near-online extraction."""
        il_idx = self.current_il_idx
        xl_idx = self.current_xl_idx

        if self.gather_type == "cig":
            if self.gather_data is None:
                return None
            # Direct extraction: (offset, time) at (il, xl)
            cig = np.asarray(self.gather_data[il_idx, xl_idx, :, :])

        elif self.gather_type == "common_offset_folder":
            if len(self.offset_bins) == 0:
                return None
            # Folder-based: read from each offset bin
            n_offsets = len(self.offset_bins)
            n_time = self.offset_bins[0].shape[2]

            cig = np.zeros((n_offsets, n_time), dtype=np.float32)
            for i, offset_zarr in enumerate(self.offset_bins):
                # Each bin is (il, xl, time)
                cig[i, :] = np.asarray(offset_zarr[il_idx, xl_idx, :])

        else:
            # Legacy 4D common offset: (offset, inline, crossline, time)
            if self.gather_data is None:
                return None
            cig = np.asarray(self.gather_data[:, il_idx, xl_idx, :])

        return cig

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

    def update_display(self):
        """Update the canvas with current CIG."""
        cig = self.extract_cig()
        if cig is None:
            return

        # CIG is (offset, time) - display with offset on X, time on Y
        data = cig.T  # (time, offset)

        # Get axis ranges
        off_min = float(self.offset_coords[0]) if self.offset_coords is not None else 0
        off_max = float(self.offset_coords[-1]) if self.offset_coords is not None else cig.shape[0]
        t_min = float(self.t_coords[0]) if self.t_coords is not None else 0
        t_max = float(self.t_coords[-1]) if self.t_coords is not None else cig.shape[1] * 2

        x_axis = AxisConfig("Offset", off_min, off_max, "m")
        y_axis = AxisConfig("Time", t_min, t_max, "ms")

        # Show IL/XL indices
        self.position_label.setText(f"IL={self.current_il_idx}, XL={self.current_xl_idx}")

        self.canvas.set_data(
            data, "cig", 0, 0, x_axis, y_axis,
            slice_value=self.current_il_idx
        )
        self.canvas.slice_direction = f"CIG @ IL={self.current_il_idx}, XL={self.current_xl_idx}"
