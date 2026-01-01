#!/usr/bin/env python3
"""
PyQt6 Seismic Viewer for Zarr/Parquet data.
No matplotlib - uses QImage for fast rendering.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QGroupBox, QSlider, QStatusBar, QSplitter,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal, QSize
from PyQt6.QtGui import (
    QImage, QPainter, QColor, QPen, QWheelEvent, QMouseEvent,
    QKeyEvent, QResizeEvent, QPaintEvent, QFont
)

import zarr


# =============================================================================
# Color Palettes
# =============================================================================

def create_palette(name: str, n: int = 256) -> np.ndarray:
    """Create color palette as (n, 3) RGB array."""
    t = np.linspace(0, 1, n)

    if name == "gray":
        r = g = b = (t * 255).astype(np.uint8)
    elif name == "seismic":
        # Blue-White-Red
        r = np.where(t < 0.5, t * 2 * 255, 255).astype(np.uint8)
        g = np.where(t < 0.5, t * 2 * 255, (1 - t) * 2 * 255).astype(np.uint8)
        b = np.where(t < 0.5, 255, (1 - t) * 2 * 255).astype(np.uint8)
    elif name == "rwb":
        # Red-White-Blue (reversed seismic)
        r = np.where(t < 0.5, 255, (1 - t) * 2 * 255).astype(np.uint8)
        g = np.where(t < 0.5, t * 2 * 255, (1 - t) * 2 * 255).astype(np.uint8)
        b = np.where(t < 0.5, t * 2 * 255, 255).astype(np.uint8)
    elif name == "viridis":
        # Approximate viridis
        r = (np.clip(0.267 + 0.004*t + 1.2*t**2 - 0.8*t**3, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(0.004 + 1.0*t - 0.15*t**2, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(0.329 + 0.6*t - 0.6*t**2 - 0.2*t**3, 0, 1) * 255).astype(np.uint8)
    elif name == "bone":
        r = (np.clip(t * 0.75 + 0.25 * np.maximum(0, t - 0.75) * 4, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(t * 0.75 + 0.25 * np.clip((t - 0.25) * 4, 0, 1), 0, 1) * 255).astype(np.uint8)
        b = (np.clip(t * 0.75 + 0.25 * np.clip(t * 4, 0, 1), 0, 1) * 255).astype(np.uint8)
    else:  # default gray
        r = g = b = (t * 255).astype(np.uint8)

    return np.stack([r, g, b], axis=1)


PALETTES = {
    "Gray": create_palette("gray"),
    "Seismic (BWR)": create_palette("seismic"),
    "Seismic (RWB)": create_palette("rwb"),
    "Viridis": create_palette("viridis"),
    "Bone": create_palette("bone"),
}


# =============================================================================
# Seismic Canvas Widget
# =============================================================================

class SeismicCanvas(QWidget):
    """Widget for rendering seismic data using QImage."""

    slice_changed = pyqtSignal(int)  # Emitted when slice index changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Data
        self.data: Optional[np.ndarray] = None  # Current 2D slice
        self.image: Optional[QImage] = None

        # Display parameters
        self.palette = PALETTES["Gray"]
        self.gain = 1.0
        self.clip_percentile = 99.0

        # Zoom and pan
        self.zoom = 1.0
        self.pan_offset = QPoint(0, 0)
        self.dragging = False
        self.last_mouse_pos = QPoint()

        # Slice info
        self.slice_index = 0
        self.max_slice = 0
        self.slice_direction = "inline"  # inline, crossline, time
        self.axis_labels = ("X", "Y")

        # Mouse tracking for coordinates
        self.setMouseTracking(True)
        self.mouse_pos = QPoint()

    def set_data(self, data: np.ndarray, direction: str, index: int, max_idx: int,
                 axis_labels: Tuple[str, str] = ("X", "Y")):
        """Set 2D data slice to display."""
        self.data = data.astype(np.float32)
        self.slice_direction = direction
        self.slice_index = index
        self.max_slice = max_idx
        self.axis_labels = axis_labels
        self._update_image()
        self.update()

    def set_palette(self, name: str):
        """Set color palette by name."""
        if name in PALETTES:
            self.palette = PALETTES[name]
            self._update_image()
            self.update()

    def set_gain(self, gain: float):
        """Set display gain."""
        self.gain = gain
        self._update_image()
        self.update()

    def set_clip_percentile(self, pct: float):
        """Set clip percentile for normalization."""
        self.clip_percentile = pct
        self._update_image()
        self.update()

    def _update_image(self):
        """Convert data to QImage using current palette and gain."""
        if self.data is None:
            self.image = None
            return

        # Normalize data
        d = self.data * self.gain
        vmax = np.percentile(np.abs(d), self.clip_percentile)
        if vmax > 0:
            d = d / vmax  # Now in [-1, 1] range (mostly)
        d = np.clip(d, -1, 1)

        # Map to palette indices [0, 255]
        indices = ((d + 1) * 127.5).astype(np.uint8)

        # Apply palette
        h, w = indices.shape
        rgb = self.palette[indices.flatten()].reshape(h, w, 3)

        # Create QImage (need to add alpha channel for Format_RGBA8888)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255

        # Create QImage
        self.image = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        # Need to keep reference to data
        self._image_data = rgba

    def reset_view(self):
        """Reset zoom and pan to fit data."""
        self.zoom = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update()

    def fit_to_window(self):
        """Fit data to window."""
        if self.image is None:
            return
        w_ratio = self.width() / self.image.width()
        h_ratio = self.height() / self.image.height()
        self.zoom = min(w_ratio, h_ratio) * 0.95
        self.pan_offset = QPoint(0, 0)
        self.update()

    def paintEvent(self, event: QPaintEvent):
        """Paint the seismic image."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self.image is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                           "No data loaded\n\nDrop a .zarr folder or use File > Open")
            return

        # Calculate scaled size and position
        img_w = int(self.image.width() * self.zoom)
        img_h = int(self.image.height() * self.zoom)

        # Center image + pan offset
        x = (self.width() - img_w) // 2 + self.pan_offset.x()
        y = (self.height() - img_h) // 2 + self.pan_offset.y()

        # Draw image
        target_rect = QRect(x, y, img_w, img_h)
        painter.drawImage(target_rect, self.image)

        # Draw border
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(target_rect)

        # Draw info overlay
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Monospace", 10)
        painter.setFont(font)

        info = f"{self.slice_direction.capitalize()} {self.slice_index}/{self.max_slice}"
        info += f"  |  Zoom: {self.zoom:.1f}x  |  Gain: {self.gain:.1f}"
        painter.drawText(10, 20, info)

        # Draw axis labels
        painter.drawText(10, self.height() - 10, f"X: {self.axis_labels[0]}")
        painter.save()
        painter.translate(20, self.height() // 2)
        painter.rotate(-90)
        painter.drawText(0, 0, f"Y: {self.axis_labels[1]}")
        painter.restore()

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+wheel: zoom
            factor = 1.1 if delta > 0 else 0.9
            self.zoom = max(0.1, min(20.0, self.zoom * factor))
        else:
            # Plain wheel: change slice
            step = 1 if delta > 0 else -1
            new_idx = self.slice_index + step
            if 0 <= new_idx <= self.max_slice:
                self.slice_index = new_idx
                self.slice_changed.emit(new_idx)

        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning and coordinate display."""
        self.mouse_pos = event.pos()

        if self.dragging:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.pos()
            self.update()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard navigation."""
        key = event.key()

        if key in (Qt.Key.Key_Up, Qt.Key.Key_W):
            if self.slice_index < self.max_slice:
                self.slice_index += 1
                self.slice_changed.emit(self.slice_index)
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_S):
            if self.slice_index > 0:
                self.slice_index -= 1
                self.slice_changed.emit(self.slice_index)
        elif key == Qt.Key.Key_Home:
            self.slice_index = 0
            self.slice_changed.emit(self.slice_index)
        elif key == Qt.Key.Key_End:
            self.slice_index = self.max_slice
            self.slice_changed.emit(self.slice_index)
        elif key == Qt.Key.Key_F:
            self.fit_to_window()
        elif key == Qt.Key.Key_R:
            self.reset_view()
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.zoom = min(20.0, self.zoom * 1.2)
            self.update()
        elif key == Qt.Key.Key_Minus:
            self.zoom = max(0.1, self.zoom / 1.2)
            self.update()


# =============================================================================
# Main Window
# =============================================================================

class SeismicViewer(QMainWindow):
    """Main seismic viewer window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seismic Viewer")
        self.setMinimumSize(1000, 700)

        # Data
        self.cube: Optional[np.ndarray] = None
        self.cube_shape: Tuple[int, int, int] = (0, 0, 0)
        self.current_direction = "inline"
        self.current_index = 0

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self):
        """Setup the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel: controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(280)
        left_layout = QVBoxLayout(left_panel)

        # File group
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)

        self.open_btn = QPushButton("Open Zarr...")
        self.open_btn.clicked.connect(self.open_file)
        file_layout.addWidget(self.open_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        left_layout.addWidget(file_group)

        # Slice selection group
        slice_group = QGroupBox("Slice Selection")
        slice_layout = QVBoxLayout(slice_group)

        # Direction selector
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Inline", "Crossline", "Time Slice"])
        self.direction_combo.currentIndexChanged.connect(self.on_direction_changed)
        dir_layout.addWidget(self.direction_combo)
        slice_layout.addLayout(dir_layout)

        # Slice index
        idx_layout = QHBoxLayout()
        idx_layout.addWidget(QLabel("Index:"))
        self.slice_spin = QSpinBox()
        self.slice_spin.setRange(0, 0)
        self.slice_spin.valueChanged.connect(self.on_slice_changed)
        idx_layout.addWidget(self.slice_spin)
        slice_layout.addLayout(idx_layout)

        # Slice slider
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self.slice_spin.setValue)
        slice_layout.addWidget(self.slice_slider)

        # Step size
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step:"))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 100)
        self.step_spin.setValue(1)
        step_layout.addWidget(self.step_spin)
        slice_layout.addLayout(step_layout)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self.prev_slice)
        nav_layout.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.next_slice)
        nav_layout.addWidget(self.next_btn)
        slice_layout.addLayout(nav_layout)

        left_layout.addWidget(slice_group)

        # Display group
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        # Palette
        pal_layout = QHBoxLayout()
        pal_layout.addWidget(QLabel("Palette:"))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list(PALETTES.keys()))
        self.palette_combo.currentTextChanged.connect(self.on_palette_changed)
        pal_layout.addWidget(self.palette_combo)
        display_layout.addLayout(pal_layout)

        # Gain
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 100.0)
        self.gain_spin.setValue(1.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.valueChanged.connect(self.on_gain_changed)
        gain_layout.addWidget(self.gain_spin)
        display_layout.addLayout(gain_layout)

        # Gain slider
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(1, 500)  # 0.1 to 50.0
        self.gain_slider.setValue(10)
        self.gain_slider.valueChanged.connect(lambda v: self.gain_spin.setValue(v / 10.0))
        display_layout.addWidget(self.gain_slider)

        # Clip percentile
        clip_layout = QHBoxLayout()
        clip_layout.addWidget(QLabel("Clip %:"))
        self.clip_spin = QDoubleSpinBox()
        self.clip_spin.setRange(90.0, 100.0)
        self.clip_spin.setValue(99.0)
        self.clip_spin.setSingleStep(0.5)
        self.clip_spin.valueChanged.connect(self.on_clip_changed)
        clip_layout.addWidget(self.clip_spin)
        display_layout.addLayout(clip_layout)

        left_layout.addWidget(display_group)

        # Zoom group
        zoom_group = QGroupBox("View")
        zoom_layout = QVBoxLayout(zoom_group)

        zoom_btn_layout = QHBoxLayout()
        self.fit_btn = QPushButton("Fit (F)")
        self.fit_btn.clicked.connect(lambda: self.canvas.fit_to_window())
        zoom_btn_layout.addWidget(self.fit_btn)
        self.reset_btn = QPushButton("Reset (R)")
        self.reset_btn.clicked.connect(lambda: self.canvas.reset_view())
        zoom_btn_layout.addWidget(self.reset_btn)
        zoom_layout.addLayout(zoom_btn_layout)

        left_layout.addWidget(zoom_group)

        # Info group
        info_group = QGroupBox("Data Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("Shape: -\nRange: -")
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_group)

        left_layout.addStretch()

        # Canvas
        self.canvas = SeismicCanvas()
        self.canvas.slice_changed.connect(self.on_canvas_slice_changed)

        # Add to layout
        layout.addWidget(left_panel)
        layout.addWidget(self.canvas, 1)

        # Status bar
        self.statusBar().showMessage("Ready - Use Ctrl+Wheel to zoom, Wheel to change slice")

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        pass  # Handled in canvas keyPressEvent

    def open_file(self):
        """Open a zarr file."""
        path = QFileDialog.getExistingDirectory(
            self, "Open Zarr Directory",
            str(Path.home() / "SeismicData")
        )
        if not path:
            return

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
            else:
                # Just use the root as array
                data = np.asarray(z)

            if data.ndim != 3:
                raise ValueError(f"Expected 3D data, got {data.ndim}D")

            self.cube = data
            self.cube_shape = data.shape

            self.file_label.setText(f"{Path(path).name}\n{data.shape}")
            self.info_label.setText(
                f"Shape: {data.shape[0]} x {data.shape[1]} x {data.shape[2]}\n"
                f"Range: [{data.min():.4f}, {data.max():.4f}]\n"
                f"RMS: {np.sqrt(np.mean(data**2)):.4f}"
            )

            # Update controls
            self.on_direction_changed(self.direction_combo.currentIndex())

            self.statusBar().showMessage(f"Loaded: {path}")

        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
            self.file_label.setText(f"Error loading file:\n{e}")

    def on_direction_changed(self, index: int):
        """Handle direction change."""
        if self.cube is None:
            return

        directions = ["inline", "crossline", "time"]
        self.current_direction = directions[index]

        # Set max index based on direction
        if index == 0:  # Inline
            max_idx = self.cube_shape[0] - 1
        elif index == 1:  # Crossline
            max_idx = self.cube_shape[1] - 1
        else:  # Time
            max_idx = self.cube_shape[2] - 1

        self.slice_spin.setRange(0, max_idx)
        self.slice_slider.setRange(0, max_idx)

        # Reset to middle slice
        mid = max_idx // 2
        self.slice_spin.setValue(mid)

    def on_slice_changed(self, index: int):
        """Handle slice index change."""
        if self.cube is None:
            return

        self.current_index = index
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(index)
        self.slice_slider.blockSignals(False)

        self.update_display()

    def on_canvas_slice_changed(self, index: int):
        """Handle slice change from canvas (wheel)."""
        step = self.step_spin.value()
        current = self.slice_spin.value()

        if index > self.current_index:
            new_idx = min(current + step, self.slice_spin.maximum())
        else:
            new_idx = max(current - step, 0)

        self.slice_spin.setValue(new_idx)

    def prev_slice(self):
        """Go to previous slice."""
        step = self.step_spin.value()
        new_idx = max(0, self.slice_spin.value() - step)
        self.slice_spin.setValue(new_idx)

    def next_slice(self):
        """Go to next slice."""
        step = self.step_spin.value()
        new_idx = min(self.slice_spin.maximum(), self.slice_spin.value() + step)
        self.slice_spin.setValue(new_idx)

    def on_palette_changed(self, name: str):
        """Handle palette change."""
        self.canvas.set_palette(name)

    def on_gain_changed(self, value: float):
        """Handle gain change."""
        self.gain_slider.blockSignals(True)
        self.gain_slider.setValue(int(value * 10))
        self.gain_slider.blockSignals(False)
        self.canvas.set_gain(value)

    def on_clip_changed(self, value: float):
        """Handle clip percentile change."""
        self.canvas.set_clip_percentile(value)

    def update_display(self):
        """Update the canvas display."""
        if self.cube is None:
            return

        idx = self.current_index
        direction = self.current_direction

        if direction == "inline":
            data = self.cube[idx, :, :].T  # (crossline, time) -> show time vertical
            axis_labels = ("Crossline", "Time")
            max_idx = self.cube_shape[0] - 1
        elif direction == "crossline":
            data = self.cube[:, idx, :].T  # (inline, time) -> show time vertical
            axis_labels = ("Inline", "Time")
            max_idx = self.cube_shape[1] - 1
        else:  # time
            data = self.cube[:, :, idx].T  # (inline, crossline)
            axis_labels = ("Inline", "Crossline")
            max_idx = self.cube_shape[2] - 1

        self.canvas.set_data(data, direction, idx, max_idx, axis_labels)


# =============================================================================
# Main
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark theme
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(palette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    viewer = SeismicViewer()
    viewer.show()

    # Load file from command line if provided
    if len(sys.argv) > 1:
        viewer.load_zarr(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
