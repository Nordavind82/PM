"""
Step 9: Visualization - Interactive seismic data viewer.

Features:
- PyQtGraph-based high-performance visualization
- Inline, crossline, and time slice views
- Interactive spectrum analysis (ISA)
- Zoom, pan, and colormap controls
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QFrame, QSlider, QSpinBox,
    QSplitter, QComboBox, QDoubleSpinBox, QTabWidget,
    QCheckBox, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

from pstm.gui.steps.base import WizardStepWidget

if TYPE_CHECKING:
    from pstm.gui.state import WizardController

# Try to import pyqtgraph
try:
    import pyqtgraph as pg
    from pyqtgraph import ImageView, GraphicsLayoutWidget
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    pg = None

# Try to import zarr
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

# Try to import polars for parquet
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


class ViewportLimits:
    """Current viewport limits."""
    def __init__(self, time_min=0.0, time_max=1000.0, x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0):
        self.time_min = time_min
        self.time_max = time_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


class ViewportState(QObject):
    """Manages synchronized viewport state across panels."""

    limits_changed = pyqtSignal(object)  # ViewportLimits
    amplitude_range_changed = pyqtSignal(float, float)
    colormap_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._limits = ViewportLimits()
        self._min_amplitude = -1.0
        self._max_amplitude = 1.0
        self._colormap = 'seismic'

    @property
    def limits(self) -> ViewportLimits:
        return self._limits

    @property
    def min_amplitude(self) -> float:
        return self._min_amplitude

    @property
    def max_amplitude(self) -> float:
        return self._max_amplitude

    @property
    def colormap(self) -> str:
        return self._colormap

    def set_limits(self, time_min: float, time_max: float, x_min: float, x_max: float):
        self._limits = ViewportLimits(time_min, time_max, x_min, x_max)
        self.limits_changed.emit(self._limits)

    def set_amplitude_range(self, min_amp: float, max_amp: float):
        self._min_amplitude = min_amp
        self._max_amplitude = max_amp
        self.amplitude_range_changed.emit(min_amp, max_amp)

    def set_colormap(self, colormap: str):
        self._colormap = colormap
        self.colormap_changed.emit(colormap)


class SeismicSliceViewer(QWidget):
    """Single slice viewer panel using PyQtGraph."""

    def __init__(self, title: str, axis_labels: tuple = ('X', 'Y'), parent=None):
        super().__init__(parent)
        self.title = title
        self.axis_labels = axis_labels
        self._data = None
        self._clip_percent = 99
        self._colormap = 'seismic'

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Title
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px; color: white;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        if not PYQTGRAPH_AVAILABLE:
            error_label = QLabel("PyQtGraph not available.\nInstall with: pip install pyqtgraph")
            error_label.setStyleSheet("color: #ff6b6b;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(error_label)
            return

        # Create graphics widget
        self.graphics_widget = GraphicsLayoutWidget()
        self.graphics_widget.setBackground('#1a1a1a')

        # Create plot
        self.plot_item = self.graphics_widget.addPlot(
            labels={'left': self.axis_labels[1], 'bottom': self.axis_labels[0]}
        )
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.setMenuEnabled(False)

        # Create image item
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        # Setup colormap
        self._apply_colormap('seismic')

        # Configure mouse
        self.view_box = self.plot_item.getViewBox()
        self.view_box.setMouseEnabled(x=True, y=True)
        self.view_box.setMouseMode(pg.ViewBox.RectMode)

        layout.addWidget(self.graphics_widget)

    def _apply_colormap(self, colormap_name: str):
        """Apply colormap to image."""
        if not PYQTGRAPH_AVAILABLE:
            return

        if colormap_name == 'seismic':
            positions = np.array([0.0, 0.45, 0.50, 0.55, 1.0])
            colors = np.array([
                [0, 0, 255, 255],
                [135, 206, 250, 255],
                [245, 245, 245, 255],
                [255, 160, 122, 255],
                [255, 0, 0, 255]
            ], dtype=np.float32)
        elif colormap_name == 'grayscale':
            positions = np.array([0.0, 1.0])
            colors = np.array([
                [0, 0, 0, 255],
                [255, 255, 255, 255]
            ], dtype=np.float32)
        elif colormap_name == 'viridis':
            positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            colors = np.array([
                [68, 1, 84, 255],
                [59, 82, 139, 255],
                [33, 145, 140, 255],
                [94, 201, 98, 255],
                [253, 231, 37, 255]
            ], dtype=np.float32)
        else:
            # Default seismic
            positions = np.array([0.0, 0.5, 1.0])
            colors = np.array([
                [0, 0, 255, 255],
                [255, 255, 255, 255],
                [255, 0, 0, 255]
            ], dtype=np.float32)

        # Generate LUT
        lut = np.zeros((256, 4), dtype=np.uint8)
        for i in range(256):
            pos = i / 255.0
            idx = np.searchsorted(positions, pos)
            if idx == 0:
                lut[i] = colors[0].astype(np.uint8)
            elif idx >= len(positions):
                lut[i] = colors[-1].astype(np.uint8)
            else:
                t = (pos - positions[idx - 1]) / (positions[idx] - positions[idx - 1])
                interpolated = colors[idx - 1] * (1 - t) + colors[idx] * t
                lut[i] = interpolated.astype(np.uint8)

        self.image_item.setLookupTable(lut)
        self._colormap = colormap_name

    def set_data(self, data: np.ndarray, clip_percent: float = 99,
                 x_axis: np.ndarray = None, y_axis: np.ndarray = None,
                 vmin: float = None, vmax: float = None):
        """Set display data.

        Args:
            data: 2D array of amplitudes
            clip_percent: Percentile for auto-clipping (used if vmin/vmax not set)
            x_axis: X coordinate axis
            y_axis: Y coordinate axis
            vmin: Manual minimum value for color scale
            vmax: Manual maximum value for color scale
        """
        if not PYQTGRAPH_AVAILABLE or data is None:
            return

        self._data = data
        self._clip_percent = clip_percent

        # Determine scale range
        if vmin is not None and vmax is not None:
            # Use manual min/max
            scale_min = vmin
            scale_max = vmax
        else:
            # Auto-clip based on percentile
            valid_data = data[~np.isnan(data)]
            if valid_data.size > 0:
                clip_val = np.percentile(np.abs(valid_data), clip_percent)
                clip_val = max(clip_val, 1e-15)  # Prevent zero range
            else:
                clip_val = 1.0
            scale_min = -clip_val
            scale_max = clip_val

        # Clip and normalize to 0-1 range for colormap
        scale_range = scale_max - scale_min
        if scale_range < 1e-15:
            scale_range = 1e-15  # Prevent division by zero

        display_data = np.clip(data, scale_min, scale_max)
        display_data = (display_data - scale_min) / scale_range

        # Set image
        self.image_item.setImage(display_data.T, autoLevels=False, levels=(0, 1))

        # Set coordinate transform if axes provided
        if x_axis is not None and y_axis is not None:
            self.image_item.setRect(
                x_axis[0], y_axis[0],
                x_axis[-1] - x_axis[0], y_axis[-1] - y_axis[0]
            )

    def set_colormap(self, colormap: str):
        """Change colormap."""
        self._apply_colormap(colormap)
        if self._data is not None:
            self.set_data(self._data, self._clip_percent)

    def clear(self):
        """Clear display."""
        if PYQTGRAPH_AVAILABLE:
            self.image_item.clear()
        self._data = None


class SpectrumPanel(QWidget):
    """Panel for displaying amplitude spectrum."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel("Amplitude Spectrum")
        title.setStyleSheet("font-weight: bold; color: white;")
        layout.addWidget(title)

        if not PYQTGRAPH_AVAILABLE:
            error_label = QLabel("PyQtGraph required for spectrum display")
            error_label.setStyleSheet("color: #ff6b6b;")
            layout.addWidget(error_label)
            return

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1a1a1a')
        self.plot_widget.setLabel('left', 'Amplitude (dB)')
        self.plot_widget.setLabel('bottom', 'Frequency (Hz)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        self.spectrum_curve = self.plot_widget.plot(pen=pg.mkPen('c', width=2))

        layout.addWidget(self.plot_widget)

    def set_spectrum(self, frequencies: np.ndarray, amplitudes_db: np.ndarray):
        """Set spectrum data."""
        if PYQTGRAPH_AVAILABLE:
            self.spectrum_curve.setData(frequencies, amplitudes_db)

    def clear(self):
        """Clear spectrum."""
        if PYQTGRAPH_AVAILABLE:
            self.spectrum_curve.clear()


class VisualizationStep(WizardStepWidget):
    """Step 9: Interactive visualization of migration results."""

    @property
    def title(self) -> str:
        return "Visualization"

    def _setup_ui(self) -> None:
        """Set up the UI."""
        # Header
        header_frame, header_layout = self.create_section("Step 9: Visualization")
        desc = QLabel(
            "Interactive visualization of migration results. "
            "View inline, crossline, and time slices. Analyze frequency content."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888888; border: none; background: transparent;")
        header_layout.addWidget(desc)
        self.content_layout.addWidget(header_frame)

        # Initialize state
        self._data = None
        self._fold = None
        self._sample_rate_ms = 2.0
        self._bin_headers = None  # Polars DataFrame from bin_headers.parquet
        self._cig_data = None  # CIG volume if available
        self._is_prestack = False  # Flag for prestack data

        # Main content - horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Viewer panels
        viewer_widget = self._create_viewer_panels()
        main_splitter.addWidget(viewer_widget)

        # Right: Control panel
        control_panel = self._create_control_panel()
        main_splitter.addWidget(control_panel)

        # Set splitter sizes (80% viewer, 20% controls)
        main_splitter.setSizes([800, 200])

        self.content_layout.addWidget(main_splitter, 1)

    def _create_viewer_panels(self) -> QWidget:
        """Create the viewer panels."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create tabs for different view modes
        self.view_tabs = QTabWidget()

        # Tab 1: Slice Views (inline + time slice side by side)
        slice_widget = QWidget()
        slice_layout = QHBoxLayout(slice_widget)

        # Vertical splitter for inline and crossline
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        self.inline_viewer = SeismicSliceViewer("Inline Section", ('Crossline', 'Time (ms)'))
        self.crossline_viewer = SeismicSliceViewer("Crossline Section", ('Inline', 'Time (ms)'))

        left_splitter.addWidget(self.inline_viewer)
        left_splitter.addWidget(self.crossline_viewer)

        slice_layout.addWidget(left_splitter, 2)

        # Time slice viewer
        self.time_slice_viewer = SeismicSliceViewer("Time Slice", ('Inline', 'Crossline'))
        slice_layout.addWidget(self.time_slice_viewer, 1)

        self.view_tabs.addTab(slice_widget, "Slice Views")

        # Tab 2: Spectrum Analysis
        spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_widget)

        # Data viewer at top
        self.spectrum_data_viewer = SeismicSliceViewer("Data (click trace for spectrum)", ('Trace', 'Time (ms)'))
        spectrum_layout.addWidget(self.spectrum_data_viewer, 2)

        # Spectrum at bottom
        self.spectrum_panel = SpectrumPanel()
        spectrum_layout.addWidget(self.spectrum_panel, 1)

        self.view_tabs.addTab(spectrum_widget, "Spectrum Analysis")

        # Tab 3: Fold Map
        fold_widget = QWidget()
        fold_layout = QVBoxLayout(fold_widget)

        self.fold_viewer = SeismicSliceViewer("Fold Map", ('Inline', 'Crossline'))
        fold_layout.addWidget(self.fold_viewer)

        self.view_tabs.addTab(fold_widget, "Fold Map")

        # Tab 4: Gathers (for prestack data)
        gathers_widget = self._create_gathers_tab()
        self.view_tabs.addTab(gathers_widget, "Gathers")

        layout.addWidget(self.view_tabs)

        return widget

    def _create_gathers_tab(self) -> QWidget:
        """Create the gathers visualization tab for prestack data."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Header with prestack status
        self.gathers_status_label = QLabel("Loading...")
        self.gathers_status_label.setStyleSheet("font-weight: bold; color: #888888;")
        layout.addWidget(self.gathers_status_label)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Map views (offset and azimuth)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Map type selector
        map_selector = QHBoxLayout()
        map_label = QLabel("Display:")
        self.offset_radio = QRadioButton("Offset")
        self.offset_radio.setChecked(True)
        self.azimuth_radio = QRadioButton("Azimuth")
        self.trace_count_radio = QRadioButton("Trace Count")

        self.map_type_group = QButtonGroup()
        self.map_type_group.addButton(self.offset_radio, 0)
        self.map_type_group.addButton(self.azimuth_radio, 1)
        self.map_type_group.addButton(self.trace_count_radio, 2)
        self.map_type_group.buttonClicked.connect(self._on_gather_map_type_changed)

        map_selector.addWidget(map_label)
        map_selector.addWidget(self.offset_radio)
        map_selector.addWidget(self.azimuth_radio)
        map_selector.addWidget(self.trace_count_radio)
        map_selector.addStretch()
        left_layout.addLayout(map_selector)

        # Offset/Azimuth map viewer
        self.gather_map_viewer = SeismicSliceViewer("Bin Attributes", ('Inline', 'Crossline'))
        left_layout.addWidget(self.gather_map_viewer)

        splitter.addWidget(left_widget)

        # Right side: Gather info panel and CIG viewer (if available)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Location selection
        loc_group = QGroupBox("Bin Selection")
        loc_layout = QFormLayout(loc_group)

        # Inline/Crossline selectors for gather location
        self.gather_inline_spin = QSpinBox()
        self.gather_inline_spin.setRange(0, 1000)
        self.gather_inline_spin.valueChanged.connect(self._on_gather_location_changed)
        loc_layout.addRow("Inline:", self.gather_inline_spin)

        self.gather_crossline_spin = QSpinBox()
        self.gather_crossline_spin.setRange(0, 1000)
        self.gather_crossline_spin.valueChanged.connect(self._on_gather_location_changed)
        loc_layout.addRow("Crossline:", self.gather_crossline_spin)

        right_layout.addWidget(loc_group)

        # Bin info display
        info_group = QGroupBox("Bin Information")
        info_layout = QVBoxLayout(info_group)

        self.bin_info_label = QLabel("Select a bin to view information")
        self.bin_info_label.setWordWrap(True)
        self.bin_info_label.setStyleSheet("color: #cccccc; font-family: monospace;")
        info_layout.addWidget(self.bin_info_label)

        right_layout.addWidget(info_group)

        # Statistics summary
        stats_group = QGroupBox("Header Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.header_stats_label = QLabel("No prestack data loaded")
        self.header_stats_label.setWordWrap(True)
        self.header_stats_label.setStyleSheet("color: #888888;")
        stats_layout.addWidget(self.header_stats_label)

        right_layout.addWidget(stats_group)

        # CIG viewer (if CIG data available)
        cig_group = QGroupBox("Common Image Gather")
        cig_layout = QVBoxLayout(cig_group)

        self.cig_viewer = SeismicSliceViewer("CIG at Location", ('Offset Bin', 'Time (ms)'))
        cig_layout.addWidget(self.cig_viewer)

        self.cig_status_label = QLabel("CIG data not available")
        self.cig_status_label.setStyleSheet("color: #888888;")
        cig_layout.addWidget(self.cig_status_label)

        right_layout.addWidget(cig_group)

        splitter.addWidget(right_widget)
        splitter.setSizes([500, 300])

        layout.addWidget(splitter)

        return widget

    def _create_control_panel(self) -> QWidget:
        """Create control panel."""
        panel = QWidget()
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)

        # Slice selection
        slice_group = QGroupBox("Slice Selection")
        slice_layout = QFormLayout(slice_group)

        # Inline slider
        self.inline_slider = QSlider(Qt.Orientation.Horizontal)
        self.inline_slider.setRange(0, 100)
        self.inline_slider.setValue(50)
        self.inline_slider.valueChanged.connect(self._on_inline_changed)
        self.inline_label = QLabel("50")
        inline_row = QHBoxLayout()
        inline_row.addWidget(self.inline_slider)
        inline_row.addWidget(self.inline_label)
        slice_layout.addRow("Inline:", inline_row)

        # Crossline slider
        self.crossline_slider = QSlider(Qt.Orientation.Horizontal)
        self.crossline_slider.setRange(0, 100)
        self.crossline_slider.setValue(50)
        self.crossline_slider.valueChanged.connect(self._on_crossline_changed)
        self.crossline_label = QLabel("50")
        crossline_row = QHBoxLayout()
        crossline_row.addWidget(self.crossline_slider)
        crossline_row.addWidget(self.crossline_label)
        slice_layout.addRow("Crossline:", crossline_row)

        # Time slider
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 4000)
        self.time_slider.setValue(2000)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        self.time_label = QLabel("2000 ms")
        time_row = QHBoxLayout()
        time_row.addWidget(self.time_slider)
        time_row.addWidget(self.time_label)
        slice_layout.addRow("Time:", time_row)

        layout.addWidget(slice_group)

        # Display settings
        display_group = QGroupBox("Display Settings")
        display_layout = QFormLayout(display_group)

        # Colormap
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['seismic', 'grayscale', 'viridis'])
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        display_layout.addRow("Colormap:", self.colormap_combo)

        # Auto scale checkbox
        self.auto_scale_check = QCheckBox("Auto Scale")
        self.auto_scale_check.setChecked(True)
        self.auto_scale_check.stateChanged.connect(self._on_auto_scale_changed)
        display_layout.addRow("", self.auto_scale_check)

        # Clip percent (for auto mode)
        self.clip_spin = QSpinBox()
        self.clip_spin.setRange(90, 100)
        self.clip_spin.setValue(99)
        self.clip_spin.valueChanged.connect(self._update_all_views)
        display_layout.addRow("Clip %:", self.clip_spin)

        # Min value (scientific notation)
        self.scale_min_spin = QDoubleSpinBox()
        self.scale_min_spin.setRange(-1e15, 1e15)
        self.scale_min_spin.setDecimals(6)
        self.scale_min_spin.setValue(-1e-3)
        self.scale_min_spin.setEnabled(False)
        self.scale_min_spin.valueChanged.connect(self._update_all_views)
        display_layout.addRow("Scale Min:", self.scale_min_spin)

        # Max value (scientific notation)
        self.scale_max_spin = QDoubleSpinBox()
        self.scale_max_spin.setRange(-1e15, 1e15)
        self.scale_max_spin.setDecimals(6)
        self.scale_max_spin.setValue(1e-3)
        self.scale_max_spin.setEnabled(False)
        self.scale_max_spin.valueChanged.connect(self._update_all_views)
        display_layout.addRow("Scale Max:", self.scale_max_spin)

        # Auto-detect button
        self.auto_detect_btn = QPushButton("Detect Range")
        self.auto_detect_btn.setEnabled(False)
        self.auto_detect_btn.clicked.connect(self._detect_amplitude_range)
        display_layout.addRow("", self.auto_detect_btn)

        layout.addWidget(display_group)

        # Data info
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout(info_group)

        self.info_label = QLabel("No data loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #888888;")
        info_layout.addWidget(self.info_label)

        layout.addWidget(info_group)

        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        load_btn = QPushButton("Load Migration Output")
        load_btn.clicked.connect(self._load_data)
        actions_layout.addWidget(load_btn)

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_view)
        actions_layout.addWidget(reset_btn)

        layout.addWidget(actions_group)

        layout.addStretch()

        return panel

    def _load_data(self) -> None:
        """Load migration output data."""
        output_dir = self.controller.state.output.output_dir
        if not output_dir:
            return

        output_path = Path(output_dir)
        stack_path = output_path / "migrated_stack.zarr"
        fold_path = output_path / "fold.zarr"
        headers_path = output_path / "bin_headers.parquet"
        cig_path = output_path / "cig.zarr"

        if not stack_path.exists():
            self.info_label.setText("No migration output found.\nRun migration first.")
            return

        try:
            # Load stack data
            z = zarr.open(str(stack_path), mode='r')
            if isinstance(z, zarr.Array):
                self._data = np.array(z)
            else:
                # Handle group format
                key = list(z.keys())[0] if z.keys() else None
                if key:
                    self._data = np.array(z[key])

            # Load fold data
            if fold_path.exists():
                z_fold = zarr.open(str(fold_path), mode='r')
                if isinstance(z_fold, zarr.Array):
                    self._fold = np.array(z_fold)
                else:
                    key = list(z_fold.keys())[0] if z_fold.keys() else None
                    if key:
                        self._fold = np.array(z_fold[key])

            # Load prestack bin headers if available
            self._is_prestack = False
            self._bin_headers = None
            if headers_path.exists() and POLARS_AVAILABLE:
                try:
                    self._bin_headers = pl.read_parquet(str(headers_path))
                    self._is_prestack = True
                except Exception as e:
                    print(f"Warning: Could not load bin headers: {e}")

            # Load CIG data if available
            self._cig_data = None
            if cig_path.exists():
                try:
                    from pstm.pipeline.cig import load_cig_from_zarr
                    self._cig_data, _, self._cig_coords = load_cig_from_zarr(cig_path)
                except Exception as e:
                    print(f"Warning: Could not load CIG data: {e}")

            # Update sliders
            if self._data is not None:
                nx, ny, nt = self._data.shape
                self.inline_slider.setRange(0, nx - 1)
                self.inline_slider.setValue(nx // 2)
                self.crossline_slider.setRange(0, ny - 1)
                self.crossline_slider.setValue(ny // 2)

                # Get time range from output grid
                t_min = self.controller.state.output_grid.t_min_ms
                t_max = self.controller.state.output_grid.t_max_ms
                self.time_slider.setRange(int(t_min), int(t_max))
                self.time_slider.setValue(int((t_min + t_max) / 2))
                self._sample_rate_ms = self.controller.state.output_grid.dt_ms

                # Detect amplitude range
                valid_data = self._data[~np.isnan(self._data)]
                if valid_data.size > 0:
                    data_min = float(np.min(valid_data))
                    data_max = float(np.max(valid_data))
                    # Clamp to valid range
                    data_min = max(data_min, -1e15)
                    data_max = min(data_max, 1e15)
                else:
                    data_min, data_max = -1.0, 1.0

                # Set initial scale values
                self.scale_min_spin.setValue(data_min)
                self.scale_max_spin.setValue(data_max)

                # Update gather location spinboxes
                self.gather_inline_spin.setRange(0, nx - 1)
                self.gather_inline_spin.setValue(nx // 2)
                self.gather_crossline_spin.setRange(0, ny - 1)
                self.gather_crossline_spin.setValue(ny // 2)

                # Build info string
                info_text = (
                    f"Shape: {nx} x {ny} x {nt}\n"
                    f"Time: {t_min:.0f} - {t_max:.0f} ms\n"
                    f"Sample rate: {self._sample_rate_ms:.1f} ms\n"
                    f"Amplitude: {data_min:.2e} to {data_max:.2e}"
                )

                if self._is_prestack:
                    info_text += "\n\n✓ Prestack headers available"

                if self._cig_data is not None:
                    info_text += "\n✓ CIG gathers available"

                self.info_label.setText(info_text)

                # Update views
                self._update_all_views()

                # Update gathers tab
                self._update_gathers_tab()

        except Exception as e:
            self.info_label.setText(f"Error loading data:\n{e}")

    def _on_inline_changed(self, value: int) -> None:
        """Handle inline slider change."""
        self.inline_label.setText(str(value))
        self._update_inline_view()
        self._update_time_slice_view()

    def _on_crossline_changed(self, value: int) -> None:
        """Handle crossline slider change."""
        self.crossline_label.setText(str(value))
        self._update_crossline_view()
        self._update_time_slice_view()

    def _on_time_changed(self, value: int) -> None:
        """Handle time slider change."""
        self.time_label.setText(f"{value} ms")
        self._update_time_slice_view()

    def _on_colormap_changed(self, colormap: str) -> None:
        """Handle colormap change."""
        for viewer in [self.inline_viewer, self.crossline_viewer,
                       self.time_slice_viewer, self.spectrum_data_viewer]:
            viewer.set_colormap(colormap)

    def _on_auto_scale_changed(self, state: int) -> None:
        """Handle auto scale checkbox change."""
        is_auto = state == Qt.CheckState.Checked.value
        self.clip_spin.setEnabled(is_auto)
        self.scale_min_spin.setEnabled(not is_auto)
        self.scale_max_spin.setEnabled(not is_auto)
        self.auto_detect_btn.setEnabled(not is_auto)
        self._update_all_views()

    def _detect_amplitude_range(self) -> None:
        """Auto-detect amplitude range from data."""
        if self._data is None:
            return

        valid_data = self._data[~np.isnan(self._data)]
        if valid_data.size == 0:
            return

        data_min = float(np.min(valid_data))
        data_max = float(np.max(valid_data))

        # Clamp to valid range
        data_min = max(data_min, -1e15)
        data_max = min(data_max, 1e15)

        # Ensure min < max
        if data_min >= data_max:
            data_max = data_min + 1e-15

        self.scale_min_spin.setValue(data_min)
        self.scale_max_spin.setValue(data_max)

        # Update info label with detected range
        self.info_label.setText(
            self.info_label.text() + f"\nRange: {data_min:.2e} to {data_max:.2e}"
        )

    def _get_scale_params(self) -> tuple:
        """Get current scale parameters (vmin, vmax or None for auto)."""
        if self.auto_scale_check.isChecked():
            return None, None
        else:
            return self.scale_min_spin.value(), self.scale_max_spin.value()

    def _update_all_views(self) -> None:
        """Update all viewer panels."""
        self._update_inline_view()
        self._update_crossline_view()
        self._update_time_slice_view()
        self._update_fold_view()

    def _update_inline_view(self) -> None:
        """Update inline section view."""
        if self._data is None:
            return

        inline_idx = self.inline_slider.value()
        if inline_idx >= self._data.shape[0]:
            return

        # Extract inline slice (crossline x time)
        slice_data = self._data[inline_idx, :, :]

        # Create axes
        ny, nt = slice_data.shape
        y_axis = np.arange(ny)
        t_axis = np.linspace(
            self.controller.state.output_grid.t_min_ms,
            self.controller.state.output_grid.t_max_ms,
            nt
        )

        vmin, vmax = self._get_scale_params()
        self.inline_viewer.set_data(slice_data.T, self.clip_spin.value(), y_axis, t_axis, vmin, vmax)

    def _update_crossline_view(self) -> None:
        """Update crossline section view."""
        if self._data is None:
            return

        crossline_idx = self.crossline_slider.value()
        if crossline_idx >= self._data.shape[1]:
            return

        # Extract crossline slice (inline x time)
        slice_data = self._data[:, crossline_idx, :]

        # Create axes
        nx, nt = slice_data.shape
        x_axis = np.arange(nx)
        t_axis = np.linspace(
            self.controller.state.output_grid.t_min_ms,
            self.controller.state.output_grid.t_max_ms,
            nt
        )

        vmin, vmax = self._get_scale_params()
        self.crossline_viewer.set_data(slice_data.T, self.clip_spin.value(), x_axis, t_axis, vmin, vmax)

    def _update_time_slice_view(self) -> None:
        """Update time slice view."""
        if self._data is None:
            return

        time_ms = self.time_slider.value()
        t_min = self.controller.state.output_grid.t_min_ms
        t_max = self.controller.state.output_grid.t_max_ms
        nt = self._data.shape[2]

        # Convert time to index
        time_idx = int((time_ms - t_min) / (t_max - t_min) * (nt - 1))
        time_idx = max(0, min(time_idx, nt - 1))

        # Extract time slice (inline x crossline)
        slice_data = self._data[:, :, time_idx]

        # Create axes
        nx, ny = slice_data.shape
        x_axis = np.arange(nx)
        y_axis = np.arange(ny)

        vmin, vmax = self._get_scale_params()
        self.time_slice_viewer.set_data(slice_data, self.clip_spin.value(), x_axis, y_axis, vmin, vmax)

    def _update_fold_view(self) -> None:
        """Update fold map view."""
        if self._fold is None:
            return

        nx, ny = self._fold.shape
        x_axis = np.arange(nx)
        y_axis = np.arange(ny)

        self.fold_viewer.set_data(self._fold.astype(np.float32), 100, x_axis, y_axis)

    def _update_gathers_tab(self) -> None:
        """Update the gathers tab with prestack information."""
        if not self._is_prestack or self._bin_headers is None:
            self.gathers_status_label.setText(
                "No prestack data available.\n"
                "Run migration with 'Output Gathers' enabled to generate prestack headers."
            )
            self.gathers_status_label.setStyleSheet("color: #ff8888;")
            self.header_stats_label.setText("No prestack headers found")
            return

        # Update status
        self.gathers_status_label.setText("✓ Prestack migration data detected")
        self.gathers_status_label.setStyleSheet("color: #88ff88; font-weight: bold;")

        # Compute statistics from bin headers
        df = self._bin_headers
        n_bins = len(df)

        offset_min = df["offset_avg"].min()
        offset_max = df["offset_avg"].max()
        offset_mean = df["offset_avg"].mean()

        azimuth_min = df["azimuth_avg"].min()
        azimuth_max = df["azimuth_avg"].max()

        trace_count_total = df["trace_count"].sum()
        trace_count_mean = df["trace_count"].mean()
        trace_count_max = df["trace_count"].max()

        stats_text = (
            f"Bins with data: {n_bins:,}\n"
            f"Total traces: {trace_count_total:,}\n"
            f"Traces/bin: {trace_count_mean:.1f} (max: {trace_count_max})\n\n"
            f"Offset range: {offset_min:.0f} - {offset_max:.0f} m\n"
            f"Mean offset: {offset_mean:.0f} m\n\n"
            f"Azimuth range: {azimuth_min:.1f}° - {azimuth_max:.1f}°"
        )
        self.header_stats_label.setText(stats_text)
        self.header_stats_label.setStyleSheet("color: #cccccc;")

        # Update CIG status
        if self._cig_data is not None:
            n_offsets = self._cig_data.shape[3] if len(self._cig_data.shape) > 3 else 0
            self.cig_status_label.setText(f"✓ CIG available ({n_offsets} offset bins)")
            self.cig_status_label.setStyleSheet("color: #88ff88;")
        else:
            self.cig_status_label.setText(
                "CIG volume not available.\n"
                "Re-run migration with CIG output enabled."
            )
            self.cig_status_label.setStyleSheet("color: #888888;")

        # Update the map view
        self._update_gather_map_view()

        # Update bin info for current location
        self._on_gather_location_changed()

    def _on_gather_map_type_changed(self) -> None:
        """Handle change in gather map display type."""
        self._update_gather_map_view()

    def _update_gather_map_view(self) -> None:
        """Update the gather attribute map view."""
        if self._bin_headers is None or self._data is None:
            return

        nx, ny, _ = self._data.shape
        df = self._bin_headers

        # Determine which attribute to display
        if self.offset_radio.isChecked():
            attr_col = "offset_avg"
            title = "Average Offset (m)"
        elif self.azimuth_radio.isChecked():
            attr_col = "azimuth_avg"
            title = "Average Azimuth (°)"
        else:  # trace_count
            attr_col = "trace_count"
            title = "Trace Count"

        # Create 2D grid from sparse parquet data
        grid = np.zeros((nx, ny), dtype=np.float32)

        # Fill grid with values from bin headers
        ix_arr = df["ix"].to_numpy()
        iy_arr = df["iy"].to_numpy()
        val_arr = df[attr_col].to_numpy()

        # Ensure indices are within bounds
        valid_mask = (ix_arr < nx) & (iy_arr < ny)
        ix_valid = ix_arr[valid_mask]
        iy_valid = iy_arr[valid_mask]
        val_valid = val_arr[valid_mask]

        grid[ix_valid, iy_valid] = val_valid

        # Create axes
        x_axis = np.arange(nx)
        y_axis = np.arange(ny)

        # Use viridis for attribute maps
        self.gather_map_viewer.set_colormap('viridis')
        self.gather_map_viewer.set_data(grid, 100, x_axis, y_axis)
        self.gather_map_viewer.title = title

    def _on_gather_location_changed(self) -> None:
        """Handle change in gather location selection."""
        if self._bin_headers is None:
            return

        ix = self.gather_inline_spin.value()
        iy = self.gather_crossline_spin.value()

        # Find bin info at this location
        df = self._bin_headers
        bin_data = df.filter((pl.col("ix") == ix) & (pl.col("iy") == iy))

        if len(bin_data) == 0:
            self.bin_info_label.setText(
                f"Bin ({ix}, {iy}): No data\n\n"
                "This bin has no traces contributing to migration."
            )
            self.cig_viewer.clear()
            return

        # Extract values
        row = bin_data.row(0, named=True)
        x_coord = row["x"]
        y_coord = row["y"]
        trace_count = row["trace_count"]
        offset_avg = row["offset_avg"]
        azimuth_avg = row["azimuth_avg"]

        info_text = (
            f"Bin: ({ix}, {iy})\n"
            f"Coordinates: ({x_coord:.1f}, {y_coord:.1f})\n\n"
            f"Trace count: {trace_count}\n"
            f"Avg offset: {offset_avg:.1f} m\n"
            f"Avg azimuth: {azimuth_avg:.1f}°"
        )
        self.bin_info_label.setText(info_text)

        # Update CIG viewer if data available
        self._update_cig_view(ix, iy)

    def _update_cig_view(self, ix: int, iy: int) -> None:
        """Update CIG viewer for the selected bin location."""
        if self._cig_data is None:
            self.cig_viewer.clear()
            return

        # CIG shape: (nx, ny, nt, n_offset_bins)
        if ix >= self._cig_data.shape[0] or iy >= self._cig_data.shape[1]:
            self.cig_viewer.clear()
            return

        # Extract gather at this location: (nt, n_offset_bins)
        gather = self._cig_data[ix, iy, :, :]

        # Create axes
        nt, n_offsets = gather.shape
        t_axis = np.linspace(
            self.controller.state.output_grid.t_min_ms,
            self.controller.state.output_grid.t_max_ms,
            nt
        )
        offset_axis = np.arange(n_offsets)

        # Display with seismic colormap
        self.cig_viewer.set_colormap('seismic')
        self.cig_viewer.set_data(gather.T, 99, offset_axis, t_axis)

    def _reset_view(self) -> None:
        """Reset view to defaults."""
        if self._data is not None:
            nx, ny, nt = self._data.shape
            self.inline_slider.setValue(nx // 2)
            self.crossline_slider.setValue(ny // 2)
            self.time_slider.setValue(
                int((self.controller.state.output_grid.t_min_ms +
                     self.controller.state.output_grid.t_max_ms) / 2)
            )

    def on_enter(self) -> None:
        """Called when navigating to this step."""
        # Auto-load data if available
        output_dir = self.controller.state.output.output_dir
        if output_dir:
            output_path = Path(output_dir)
            stack_path = output_path / "migrated_stack.zarr"
            if stack_path.exists() and self._data is None:
                self._load_data()

    def on_leave(self) -> None:
        """Called when leaving this step."""
        pass

    def validate(self) -> bool:
        """Validate - always valid as it's view-only."""
        return True
