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


class SeismicSliceViewer(QWidget):
    """Single slice viewer panel using PyQtGraph."""

    def __init__(self, title: str, axis_labels: tuple = ('X', 'Y'), parent=None, invert_y: bool = False):
        super().__init__(parent)
        self.title = title
        self.axis_labels = axis_labels
        self._data = None
        self._clip_percent = 99
        self._colormap = 'seismic'
        self._invert_y = invert_y

        # Mouse interaction state
        self._initial_range = None
        self._is_panning = False
        self._pan_start = None

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

        # Invert Y axis if requested (for seismic time display: time 0 at top)
        if self._invert_y:
            self.plot_item.invertY(True)

        # Setup colormap
        self._apply_colormap('seismic')

        # Configure mouse
        self.view_box = self.plot_item.getViewBox()
        self.view_box.setMouseEnabled(x=True, y=True)
        self.view_box.setMouseMode(pg.ViewBox.RectMode)  # Left mouse = zoom rectangle

        # Install event filter for custom mouse handling
        # - Middle mouse: pan/drag
        # - Right click: reset to full view
        self.graphics_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.graphics_widget.viewport().installEventFilter(self)

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
        """Set display data."""
        if not PYQTGRAPH_AVAILABLE or data is None:
            return

        self._data = data
        self._clip_percent = clip_percent

        # Determine scale range
        if vmin is not None and vmax is not None:
            scale_min = vmin
            scale_max = vmax
        else:
            valid_data = data[~np.isnan(data)]
            if valid_data.size > 0:
                clip_val = np.percentile(np.abs(valid_data), clip_percent)
                clip_val = max(clip_val, 1e-15)
            else:
                clip_val = 1.0
            scale_min = -clip_val
            scale_max = clip_val

        # Clip and normalize to 0-1 range for colormap
        scale_range = scale_max - scale_min
        if scale_range < 1e-15:
            scale_range = 1e-15

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
            # Store initial range for reset (after a short delay to let view settle)
            self._store_initial_range(x_axis, y_axis)

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

    def _store_initial_range(self, x_axis: np.ndarray, y_axis: np.ndarray):
        """Store the initial view range for reset functionality."""
        if not PYQTGRAPH_AVAILABLE:
            return
        # Store the data extents
        self._initial_range = (
            (float(x_axis[0]), float(x_axis[-1])),
            (float(y_axis[0]), float(y_axis[-1]))
        )
        # Auto-range to fit data
        self.view_box.autoRange()

    def _on_mouse_clicked(self, event):
        """Handle mouse click events - right click resets view."""
        if not PYQTGRAPH_AVAILABLE:
            return
        # Check for right mouse button click
        if event.button() == Qt.MouseButton.RightButton:
            self.reset_view()

    def eventFilter(self, obj, event):
        """Event filter for middle mouse button panning."""
        if not PYQTGRAPH_AVAILABLE:
            return super().eventFilter(obj, event)

        from PyQt6.QtCore import QEvent

        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.MiddleButton:
                self._is_panning = True
                self._pan_start = event.pos()
                return True

        elif event.type() == QEvent.Type.MouseMove:
            if self._is_panning and self._pan_start is not None:
                # Calculate delta in view coordinates
                delta = event.pos() - self._pan_start
                self._pan_start = event.pos()

                # Get current view range
                vr = self.view_box.viewRange()
                x_range = vr[0][1] - vr[0][0]
                y_range = vr[1][1] - vr[1][0]

                # Get widget size for scaling
                widget_size = self.graphics_widget.size()

                # Convert pixel delta to data coordinates
                dx = -delta.x() * x_range / widget_size.width()
                dy = delta.y() * y_range / widget_size.height()

                # If Y is inverted, flip dy
                if self._invert_y:
                    dy = -dy

                # Pan the view
                self.view_box.translateBy(x=dx, y=dy)
                return True

        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.MiddleButton:
                self._is_panning = False
                self._pan_start = None
                return True

        return super().eventFilter(obj, event)

    def reset_view(self):
        """Reset view to show full data extent."""
        if not PYQTGRAPH_AVAILABLE:
            return

        if self._initial_range is not None:
            x_range, y_range = self._initial_range
            self.view_box.setRange(xRange=x_range, yRange=y_range, padding=0.02)
        else:
            self.view_box.autoRange()


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
            "View inline, crossline, and time slices."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888888; border: none; background: transparent;")
        header_layout.addWidget(desc)
        self.content_layout.addWidget(header_frame)

        # Initialize state
        self._data = None
        self._stack_data = None  # Full stack data (kept separate)
        self._fold = None
        self._sample_rate_ms = 2.0
        self._gathers_index = None
        self._gather_headers = None
        self._current_volume_id = "stack"  # "stack" or bin_id number

        # Main content area (takes most space)
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        # Compact control bar at top
        control_bar = self._create_control_bar()
        main_layout.addWidget(control_bar)

        # Tabs for different views
        self.view_tabs = QTabWidget()

        # Tab 1: Slice Views (main view)
        slice_widget = self._create_slice_views()
        self.view_tabs.addTab(slice_widget, "Slices")

        # Tab 2: Spectrum Analysis
        spectrum_widget = self._create_spectrum_tab()
        self.view_tabs.addTab(spectrum_widget, "Spectrum")

        # Tab 3: Fold Map
        fold_widget = self._create_fold_tab()
        self.view_tabs.addTab(fold_widget, "Fold")

        main_layout.addWidget(self.view_tabs, 1)  # Give tabs all remaining space

        self.content_layout.addWidget(main_widget, 1)

    def _create_control_bar(self) -> QWidget:
        """Create compact control bar with all settings."""
        bar = QWidget()
        bar.setMaximumHeight(90)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Row 1: Volume selector, Load button, Data info
        row1 = QHBoxLayout()
        row1.setSpacing(10)

        # Volume selector
        row1.addWidget(QLabel("Volume:"))
        self.volume_combo = QComboBox()
        self.volume_combo.setMinimumWidth(200)
        self.volume_combo.addItem("Full Stack", "stack")
        self.volume_combo.currentIndexChanged.connect(self._on_volume_changed)
        row1.addWidget(self.volume_combo)

        # Load button
        load_btn = QPushButton("Load Data")
        load_btn.setMaximumWidth(100)
        load_btn.clicked.connect(self._load_data)
        row1.addWidget(load_btn)

        # Data info label
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("color: #888888;")
        row1.addWidget(self.info_label)

        row1.addStretch()

        layout.addLayout(row1)

        # Row 2: Slice navigation
        row2 = QHBoxLayout()
        row2.setSpacing(15)

        # Inline
        row2.addWidget(QLabel("Inline:"))
        self.inline_slider = QSlider(Qt.Orientation.Horizontal)
        self.inline_slider.setRange(0, 100)
        self.inline_slider.setValue(50)
        self.inline_slider.setMaximumWidth(150)
        self.inline_slider.valueChanged.connect(self._on_inline_changed)
        row2.addWidget(self.inline_slider)
        self.inline_label = QLabel("50")
        self.inline_label.setMinimumWidth(35)
        row2.addWidget(self.inline_label)

        # Crossline
        row2.addWidget(QLabel("Crossline:"))
        self.crossline_slider = QSlider(Qt.Orientation.Horizontal)
        self.crossline_slider.setRange(0, 100)
        self.crossline_slider.setValue(50)
        self.crossline_slider.setMaximumWidth(150)
        self.crossline_slider.valueChanged.connect(self._on_crossline_changed)
        row2.addWidget(self.crossline_slider)
        self.crossline_label = QLabel("50")
        self.crossline_label.setMinimumWidth(35)
        row2.addWidget(self.crossline_label)

        # Time
        row2.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 4000)
        self.time_slider.setValue(2000)
        self.time_slider.setMaximumWidth(150)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        row2.addWidget(self.time_slider)
        self.time_label = QLabel("2000 ms")
        self.time_label.setMinimumWidth(55)
        row2.addWidget(self.time_label)

        row2.addStretch()

        layout.addLayout(row2)

        # Row 3: Display settings
        row3 = QHBoxLayout()
        row3.setSpacing(15)

        # Colormap
        row3.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['seismic', 'grayscale', 'viridis'])
        self.colormap_combo.setMaximumWidth(100)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        row3.addWidget(self.colormap_combo)

        # Auto scale
        self.auto_scale_check = QCheckBox("Auto Scale")
        self.auto_scale_check.setChecked(True)
        self.auto_scale_check.stateChanged.connect(self._on_auto_scale_changed)
        row3.addWidget(self.auto_scale_check)

        # Clip percent
        row3.addWidget(QLabel("Clip %:"))
        self.clip_spin = QSpinBox()
        self.clip_spin.setRange(90, 100)
        self.clip_spin.setValue(99)
        self.clip_spin.setMaximumWidth(60)
        self.clip_spin.valueChanged.connect(self._update_all_views)
        row3.addWidget(self.clip_spin)

        # Scale min
        row3.addWidget(QLabel("Min:"))
        self.scale_min_spin = QDoubleSpinBox()
        self.scale_min_spin.setRange(-1e15, 1e15)
        self.scale_min_spin.setDecimals(6)
        self.scale_min_spin.setValue(-1e-3)
        self.scale_min_spin.setMaximumWidth(100)
        self.scale_min_spin.setEnabled(False)
        self.scale_min_spin.valueChanged.connect(self._update_all_views)
        row3.addWidget(self.scale_min_spin)

        # Scale max
        row3.addWidget(QLabel("Max:"))
        self.scale_max_spin = QDoubleSpinBox()
        self.scale_max_spin.setRange(-1e15, 1e15)
        self.scale_max_spin.setDecimals(6)
        self.scale_max_spin.setValue(1e-3)
        self.scale_max_spin.setMaximumWidth(100)
        self.scale_max_spin.setEnabled(False)
        self.scale_max_spin.valueChanged.connect(self._update_all_views)
        row3.addWidget(self.scale_max_spin)

        # Reset view button
        reset_btn = QPushButton("Reset")
        reset_btn.setMaximumWidth(60)
        reset_btn.clicked.connect(self._reset_view)
        row3.addWidget(reset_btn)

        row3.addStretch()

        layout.addLayout(row3)

        return bar

    def _create_slice_views(self) -> QWidget:
        """Create the main slice views panel."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left side: Inline and Crossline (stacked vertically)
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        # Use invert_y=True for time axis to display time 0 at top (seismic convention)
        self.inline_viewer = SeismicSliceViewer("Inline Section", ('Crossline', 'Time (ms)'), invert_y=True)
        self.crossline_viewer = SeismicSliceViewer("Crossline Section", ('Inline', 'Time (ms)'), invert_y=True)

        left_splitter.addWidget(self.inline_viewer)
        left_splitter.addWidget(self.crossline_viewer)

        layout.addWidget(left_splitter, 2)

        # Right side: Time slice
        self.time_slice_viewer = SeismicSliceViewer("Time Slice", ('Inline', 'Crossline'))
        layout.addWidget(self.time_slice_viewer, 1)

        return widget

    def _create_spectrum_tab(self) -> QWidget:
        """Create spectrum analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Data viewer at top (time 0 at top)
        self.spectrum_data_viewer = SeismicSliceViewer("Data (click trace for spectrum)", ('Trace', 'Time (ms)'), invert_y=True)
        layout.addWidget(self.spectrum_data_viewer, 2)

        # Spectrum at bottom
        self.spectrum_panel = SpectrumPanel()
        layout.addWidget(self.spectrum_panel, 1)

        return widget

    def _create_fold_tab(self) -> QWidget:
        """Create fold map tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.fold_viewer = SeismicSliceViewer("Fold Map", ('Inline', 'Crossline'))
        layout.addWidget(self.fold_viewer)

        return widget

    def _load_data(self) -> None:
        """Load migration output data."""
        output_dir = self.controller.state.output.output_dir
        if not output_dir:
            self.info_label.setText("No output directory set")
            return

        output_path = Path(output_dir)
        stack_path = output_path / "migrated_stack.zarr"
        fold_path = output_path / "fold.zarr"
        gathers_index_path = output_path / "gathers_index.parquet"
        gather_headers_path = output_path / "gather_headers.parquet"

        # Check if any output exists (stack or gathers)
        has_stack = stack_path.exists()
        has_gathers = gathers_index_path.exists()

        if not has_stack and not has_gathers:
            self.info_label.setText("No migration output found. Run migration first.")
            return

        try:
            # Load stack data if available
            self._stack_data = None
            if has_stack:
                z = zarr.open(str(stack_path), mode='r')
                if isinstance(z, zarr.Array):
                    self._stack_data = np.array(z)
                else:
                    key = list(z.keys())[0] if z.keys() else None
                    if key:
                        self._stack_data = np.array(z[key])

            # Load fold data if available
            if fold_path.exists():
                z_fold = zarr.open(str(fold_path), mode='r')
                if isinstance(z_fold, zarr.Array):
                    self._fold = np.array(z_fold)
                else:
                    key = list(z_fold.keys())[0] if z_fold.keys() else None
                    if key:
                        self._fold = np.array(z_fold[key])

            # Load gathers index if available (offset-binned output)
            self._gathers_index = None
            self._gather_headers = None

            if has_gathers and POLARS_AVAILABLE:
                try:
                    self._gathers_index = pl.read_parquet(str(gathers_index_path))
                    if gather_headers_path.exists():
                        self._gather_headers = pl.read_parquet(str(gather_headers_path))
                except Exception as e:
                    print(f"Warning: Could not load gathers index: {e}")

            # Update volume selector
            self._update_volume_selector()

            # Set initial data - prefer stack if available, otherwise load first gather
            if self._stack_data is not None:
                self._data = self._stack_data
                self._current_volume_id = "stack"
            elif self._gathers_index is not None and len(self._gathers_index) > 0:
                # Load first gather as initial data
                first_row = self._gathers_index.row(0, named=True)
                bin_id = first_row["bin_id"]
                volume_path = first_row["volume_path"]
                gather_path = output_path / volume_path

                if gather_path.exists():
                    z = zarr.open(str(gather_path), mode='r')
                    if isinstance(z, zarr.Array):
                        self._data = np.array(z)
                    else:
                        key = list(z.keys())[0] if z.keys() else None
                        if key:
                            self._data = np.array(z[key])
                    self._current_volume_id = bin_id

                    # Select first gather in combo
                    self.volume_combo.setCurrentIndex(0)

            # Update sliders
            if self._data is not None:
                self._setup_sliders_for_data()

                # Build info string
                nx, ny, nt = self._data.shape
                t_min = self.controller.state.output_grid.t_min_ms
                t_max = self.controller.state.output_grid.t_max_ms

                valid_data = self._data[~np.isnan(self._data)]
                if valid_data.size > 0:
                    data_min = float(np.min(valid_data))
                    data_max = float(np.max(valid_data))
                    data_min = max(data_min, -1e15)
                    data_max = min(data_max, 1e15)
                else:
                    data_min, data_max = -1.0, 1.0

                self.scale_min_spin.setValue(data_min)
                self.scale_max_spin.setValue(data_max)

                n_gathers = len(self._gathers_index) if self._gathers_index is not None else 0
                stack_info = "Stack" if has_stack else "Gathers only"
                gather_info = f" | {n_gathers} offset bins" if n_gathers > 0 else ""

                self.info_label.setText(
                    f"{stack_info} | Shape: {nx}x{ny}x{nt} | "
                    f"Time: {t_min:.0f}-{t_max:.0f}ms | "
                    f"Amp: {data_min:.2e} to {data_max:.2e}"
                    f"{gather_info}"
                )

                # Update views
                self._update_all_views()

        except Exception as e:
            self.info_label.setText(f"Error: {e}")

    def _update_volume_selector(self) -> None:
        """Update the volume selector combo box."""
        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()

        # Add Full Stack only if it exists
        if self._stack_data is not None:
            self.volume_combo.addItem("Full Stack", "stack")

        # Add offset bins if available
        if self._gathers_index is not None:
            for row in self._gathers_index.iter_rows(named=True):
                bin_id = row["bin_id"]
                omin = row["offset_min"]
                omax = row["offset_max"]
                self.volume_combo.addItem(
                    f"Offset {omin:.0f}-{omax:.0f}m",
                    bin_id
                )

        self.volume_combo.blockSignals(False)

    def _setup_sliders_for_data(self) -> None:
        """Setup sliders based on current data dimensions."""
        if self._data is None:
            return

        nx, ny, nt = self._data.shape

        self.inline_slider.setRange(0, nx - 1)
        self.inline_slider.setValue(nx // 2)
        self.inline_label.setText(str(nx // 2))

        self.crossline_slider.setRange(0, ny - 1)
        self.crossline_slider.setValue(ny // 2)
        self.crossline_label.setText(str(ny // 2))

        t_min = self.controller.state.output_grid.t_min_ms
        t_max = self.controller.state.output_grid.t_max_ms
        self.time_slider.setRange(int(t_min), int(t_max))
        self.time_slider.setValue(int((t_min + t_max) / 2))
        self.time_label.setText(f"{int((t_min + t_max) / 2)} ms")

        self._sample_rate_ms = self.controller.state.output_grid.dt_ms

    def _on_volume_changed(self, index: int) -> None:
        """Handle volume selection change."""
        if index < 0:
            return

        volume_id = self.volume_combo.itemData(index)
        self._current_volume_id = volume_id

        if volume_id == "stack":
            # Use full stack
            self._data = self._stack_data
        else:
            # Load specific offset bin
            if self._gathers_index is None:
                return

            bin_row = self._gathers_index.filter(pl.col("bin_id") == volume_id).row(0, named=True)
            volume_path = bin_row["volume_path"]

            output_dir = self.controller.state.output.output_dir
            gather_path = Path(output_dir) / volume_path

            if gather_path.exists():
                try:
                    z = zarr.open(str(gather_path), mode='r')
                    if isinstance(z, zarr.Array):
                        self._data = np.array(z)
                    else:
                        key = list(z.keys())[0] if z.keys() else None
                        if key:
                            self._data = np.array(z[key])
                except Exception as e:
                    print(f"Error loading gather volume: {e}")
                    return

        # Update sliders and views
        if self._data is not None:
            self._setup_sliders_for_data()
            self._update_all_views()

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
                       self.time_slice_viewer, self.spectrum_data_viewer,
                       self.fold_viewer]:
            viewer.set_colormap(colormap)

    def _on_auto_scale_changed(self, state: int) -> None:
        """Handle auto scale checkbox change."""
        is_auto = state == Qt.CheckState.Checked.value
        self.clip_spin.setEnabled(is_auto)
        self.scale_min_spin.setEnabled(not is_auto)
        self.scale_max_spin.setEnabled(not is_auto)
        self._update_all_views()

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

        slice_data = self._data[inline_idx, :, :]

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

        slice_data = self._data[:, crossline_idx, :]

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

        time_idx = int((time_ms - t_min) / (t_max - t_min) * (nt - 1))
        time_idx = max(0, min(time_idx, nt - 1))

        slice_data = self._data[:, :, time_idx]

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
