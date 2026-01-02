#!/usr/bin/env python3
"""
PyQt6 Seismic Viewer for Zarr/Parquet data.
No matplotlib - uses QImage for fast rendering.

Features:
- Proper axis visualization with tick marks and values
- Axis values update when changing slices
- Zoom works on axis coordinate system (like SeisProc)
- Dual volume support with synchronized slice/zoom/gain
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import json
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QGroupBox, QSlider, QSplitter, QCheckBox,
    QSizePolicy, QFrame, QDialog, QDialogButtonBox, QFormLayout,
    QMenuBar, QMenu
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QSize
from PyQt6.QtGui import (
    QImage, QPainter, QColor, QPen, QWheelEvent, QMouseEvent,
    QKeyEvent, QPaintEvent, QFont, QFontMetrics, QBrush
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
        r = (np.clip(0.267 + 0.004*t + 1.2*t**2 - 0.8*t**3, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(0.004 + 1.0*t - 0.15*t**2, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(0.329 + 0.6*t - 0.6*t**2 - 0.2*t**3, 0, 1) * 255).astype(np.uint8)
    elif name == "bone":
        r = (np.clip(t * 0.75 + 0.25 * np.maximum(0, t - 0.75) * 4, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(t * 0.75 + 0.25 * np.clip((t - 0.25) * 4, 0, 1), 0, 1) * 255).astype(np.uint8)
        b = (np.clip(t * 0.75 + 0.25 * np.clip(t * 4, 0, 1), 0, 1) * 255).astype(np.uint8)
    else:
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
# Axis Configuration
# =============================================================================

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


# =============================================================================
# Seismic Canvas Widget with Axes
# =============================================================================

class SeismicCanvas(QWidget):
    """Widget for rendering seismic data with proper axes."""

    # Signals
    slice_changed = pyqtSignal(int)
    view_changed = pyqtSignal(object)  # Emits ViewState
    cursor_moved = pyqtSignal(float, float, float)  # x, y, amplitude
    position_selected = pyqtSignal(float, float)  # x, y coordinates when clicked

    # Margins for axis labels
    LEFT_MARGIN = 70
    RIGHT_MARGIN = 20
    TOP_MARGIN = 30
    BOTTOM_MARGIN = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Data
        self.data: Optional[np.ndarray] = None
        self.image: Optional[QImage] = None
        self._image_data: Optional[np.ndarray] = None

        # Axis configuration
        self.x_axis = AxisConfig("X", 0, 100)
        self.y_axis = AxisConfig("Y", 0, 100)

        # Full data extents (for reset)
        self.data_x_min = 0.0
        self.data_x_max = 100.0
        self.data_y_min = 0.0
        self.data_y_max = 100.0

        # View state (in data coordinates)
        self.view = ViewState(0, 100, 0, 100)

        # Display parameters
        self.palette = PALETTES["Gray"]
        self.gain = 1.0
        self.clip_percentile = 99.0
        self.value_range = None  # (vmin, vmax) tuple for explicit range, None = use clip

        # Interaction
        self.dragging = False
        self.last_mouse_pos = QPointF()

        # Slice info
        self.slice_index = 0
        self.max_slice = 0
        self.slice_direction = "inline"
        self.slice_value = 0.0  # Actual coordinate value of current slice

        # Mouse tracking
        self.setMouseTracking(True)
        self.mouse_data_pos = QPointF(0, 0)

        # Selected position (for crosshair)
        self.selected_position: Optional[QPointF] = None
        self.show_crosshair = False

        # Font for axis labels
        self.axis_font = QFont("Arial", 9)
        self.title_font = QFont("Arial", 10, QFont.Weight.Bold)

    def set_data(self, data: np.ndarray, direction: str, index: int, max_idx: int,
                 x_axis: AxisConfig, y_axis: AxisConfig, slice_value: float = 0.0):
        """Set 2D data slice to display with axis configuration."""
        self.data = data.astype(np.float32)
        self.slice_direction = direction
        self.slice_index = index
        self.max_slice = max_idx
        self.slice_value = slice_value

        # Store axis config
        self.x_axis = x_axis
        self.y_axis = y_axis

        # Store full data extents
        self.data_x_min = x_axis.min_val
        self.data_x_max = x_axis.max_val
        self.data_y_min = y_axis.min_val
        self.data_y_max = y_axis.max_val

        # Initialize view to full data extent (important for coordinate conversion!)
        self.view = ViewState(
            self.data_x_min, self.data_x_max,
            self.data_y_min, self.data_y_max
        )

        self._update_image()
        self.update()

    def set_view(self, view: ViewState):
        """Set view state from external source (for synchronization)."""
        self.view = view.copy()
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
        self.value_range = None  # Clear explicit range when setting clip
        self._update_image()
        self.update()

    def set_value_range(self, vmin: float, vmax: float):
        """Set explicit value range for normalization (overrides clip_percentile)."""
        self.value_range = (vmin, vmax)
        self._update_image()
        self.update()

    def clear_value_range(self):
        """Clear explicit value range, revert to clip percentile."""
        self.value_range = None
        self._update_image()
        self.update()

    def set_crosshair_position(self, x: float, y: float):
        """Set crosshair position in data coordinates."""
        self.selected_position = QPointF(x, y)
        self.show_crosshair = True
        self.update()

    def clear_crosshair(self):
        """Clear the crosshair."""
        self.show_crosshair = False
        self.selected_position = None
        self.update()

    def _update_image(self):
        """Convert data to QImage using current palette and gain."""
        if self.data is None:
            self.image = None
            return

        d = self.data * self.gain

        if self.value_range is not None:
            # Use explicit min/max range
            vmin, vmax = self.value_range
            if vmax > vmin:
                d = (d - vmin) / (vmax - vmin)  # Normalize to 0-1
                d = d * 2 - 1  # Convert to -1 to 1 range
            else:
                d = np.zeros_like(d)
        else:
            # Use clip percentile
            vmax = np.percentile(np.abs(d), self.clip_percentile)
            if vmax > 0:
                d = d / vmax

        d = np.clip(d, -1, 1)

        indices = ((d + 1) * 127.5).astype(np.uint8)

        h, w = indices.shape
        rgb = self.palette[indices.flatten()].reshape(h, w, 3)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255

        self.image = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self._image_data = rgba

    def reset_view(self):
        """Reset view to show full data extent."""
        self.view = ViewState(
            self.data_x_min, self.data_x_max,
            self.data_y_min, self.data_y_max
        )
        self.view_changed.emit(self.view)
        self.update()

    def fit_to_window(self):
        """Fit data to window (same as reset for now)."""
        self.reset_view()

    def _get_plot_rect(self) -> QRectF:
        """Get the rectangle for the plot area (excluding margins)."""
        return QRectF(
            self.LEFT_MARGIN,
            self.TOP_MARGIN,
            self.width() - self.LEFT_MARGIN - self.RIGHT_MARGIN,
            self.height() - self.TOP_MARGIN - self.BOTTOM_MARGIN
        )

    def _data_to_screen(self, x_data: float, y_data: float) -> QPointF:
        """Convert data coordinates to screen coordinates."""
        rect = self._get_plot_rect()

        # Normalize to [0, 1] in view coordinates
        if self.view.x_range > 0:
            x_norm = (x_data - self.view.x_min) / self.view.x_range
        else:
            x_norm = 0.5

        if self.view.y_range > 0:
            y_norm = (y_data - self.view.y_min) / self.view.y_range
        else:
            y_norm = 0.5

        # Map to screen
        x_screen = rect.left() + x_norm * rect.width()
        y_screen = rect.top() + y_norm * rect.height()

        return QPointF(x_screen, y_screen)

    def _screen_to_data(self, x_screen: float, y_screen: float) -> QPointF:
        """Convert screen coordinates to data coordinates."""
        rect = self._get_plot_rect()

        # Normalize to [0, 1]
        if rect.width() > 0:
            x_norm = (x_screen - rect.left()) / rect.width()
        else:
            x_norm = 0.5

        if rect.height() > 0:
            y_norm = (y_screen - rect.top()) / rect.height()
        else:
            y_norm = 0.5

        # Map to data coordinates
        x_data = self.view.x_min + x_norm * self.view.x_range
        y_data = self.view.y_min + y_norm * self.view.y_range

        return QPointF(x_data, y_data)

    def _get_nice_ticks(self, vmin: float, vmax: float, n_ticks: int = 5) -> List[float]:
        """Generate nice tick values for axis."""
        if vmax <= vmin:
            return [vmin]

        range_val = vmax - vmin
        rough_step = range_val / n_ticks

        # Find nice step size
        magnitude = 10 ** np.floor(np.log10(rough_step))
        residual = rough_step / magnitude

        if residual <= 1.5:
            nice_step = 1.0 * magnitude
        elif residual <= 3:
            nice_step = 2.0 * magnitude
        elif residual <= 7:
            nice_step = 5.0 * magnitude
        else:
            nice_step = 10.0 * magnitude

        # Generate ticks
        tick_min = np.ceil(vmin / nice_step) * nice_step
        tick_max = np.floor(vmax / nice_step) * nice_step

        ticks = []
        tick = tick_min
        while tick <= tick_max + nice_step * 0.01:
            if vmin <= tick <= vmax:
                ticks.append(tick)
            tick += nice_step

        return ticks

    def paintEvent(self, event: QPaintEvent):
        """Paint the seismic image with axes."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        plot_rect = self._get_plot_rect()

        if self.image is None or self.data is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(plot_rect.toRect(), Qt.AlignmentFlag.AlignCenter,
                           "No data loaded\n\nUse File > Open to load .zarr")
            return

        # Draw image
        self._draw_image(painter, plot_rect)

        # Draw axes
        self._draw_axes(painter, plot_rect)

        # Draw title with slice info
        self._draw_title(painter)

        # Draw cursor info
        self._draw_cursor_info(painter)

        # Draw crosshair if enabled
        if self.show_crosshair and self.selected_position is not None:
            self._draw_crosshair(painter, plot_rect)

    def _draw_crosshair(self, painter: QPainter, plot_rect: QRectF):
        """Draw crosshair at selected position."""
        pos = self._data_to_screen(self.selected_position.x(), self.selected_position.y())

        # Check if within plot area
        if not (plot_rect.left() <= pos.x() <= plot_rect.right() and
                plot_rect.top() <= pos.y() <= plot_rect.bottom()):
            return

        # Draw crosshair lines
        pen = QPen(QColor(255, 255, 0), 2)  # Yellow crosshair
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)

        # Vertical line
        painter.drawLine(int(pos.x()), int(plot_rect.top()),
                        int(pos.x()), int(plot_rect.bottom()))
        # Horizontal line
        painter.drawLine(int(plot_rect.left()), int(pos.y()),
                        int(plot_rect.right()), int(pos.y()))

        # Draw small circle at intersection
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(pos, 5, 5)

    def _draw_image(self, painter: QPainter, plot_rect: QRectF):
        """Draw the seismic image within the plot area."""
        if self.image is None or self.data is None:
            return

        h, w = self.data.shape

        # Calculate which portion of the image to display based on view
        # Map view coordinates to image pixel coordinates
        x_scale = w / (self.data_x_max - self.data_x_min) if self.data_x_max > self.data_x_min else 1
        y_scale = h / (self.data_y_max - self.data_y_min) if self.data_y_max > self.data_y_min else 1

        src_x1 = (self.view.x_min - self.data_x_min) * x_scale
        src_x2 = (self.view.x_max - self.data_x_min) * x_scale
        src_y1 = (self.view.y_min - self.data_y_min) * y_scale
        src_y2 = (self.view.y_max - self.data_y_min) * y_scale

        # Clamp to image bounds
        src_x1 = max(0, min(w, src_x1))
        src_x2 = max(0, min(w, src_x2))
        src_y1 = max(0, min(h, src_y1))
        src_y2 = max(0, min(h, src_y2))

        src_rect = QRectF(src_x1, src_y1, src_x2 - src_x1, src_y2 - src_y1)

        # Draw
        painter.drawImage(plot_rect, self.image, src_rect)

        # Border
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(plot_rect)

    def _draw_axes(self, painter: QPainter, plot_rect: QRectF):
        """Draw axis ticks and labels."""
        painter.setFont(self.axis_font)
        fm = QFontMetrics(self.axis_font)

        painter.setPen(QPen(QColor(200, 200, 200), 1))

        # X-axis (bottom)
        x_ticks = self._get_nice_ticks(self.view.x_min, self.view.x_max, 6)
        for tick in x_ticks:
            pos = self._data_to_screen(tick, self.view.y_min)
            if plot_rect.left() <= pos.x() <= plot_rect.right():
                # Tick mark
                painter.drawLine(
                    int(pos.x()), int(plot_rect.bottom()),
                    int(pos.x()), int(plot_rect.bottom() + 5)
                )
                # Label
                label = f"{tick:.0f}" if abs(tick) >= 1 else f"{tick:.2f}"
                text_width = fm.horizontalAdvance(label)
                painter.drawText(
                    int(pos.x() - text_width / 2),
                    int(plot_rect.bottom() + 20),
                    label
                )

        # X-axis label
        painter.setFont(self.title_font)
        x_label = self.x_axis.label
        if self.x_axis.unit:
            x_label += f" ({self.x_axis.unit})"
        x_label_width = QFontMetrics(self.title_font).horizontalAdvance(x_label)
        painter.drawText(
            int(plot_rect.center().x() - x_label_width / 2),
            int(self.height() - 5),
            x_label
        )

        # Y-axis (left)
        painter.setFont(self.axis_font)
        y_ticks = self._get_nice_ticks(self.view.y_min, self.view.y_max, 6)
        for tick in y_ticks:
            pos = self._data_to_screen(self.view.x_min, tick)
            if plot_rect.top() <= pos.y() <= plot_rect.bottom():
                # Tick mark
                painter.drawLine(
                    int(plot_rect.left() - 5), int(pos.y()),
                    int(plot_rect.left()), int(pos.y())
                )
                # Label
                label = f"{tick:.0f}" if abs(tick) >= 1 else f"{tick:.2f}"
                text_width = fm.horizontalAdvance(label)
                painter.drawText(
                    int(plot_rect.left() - text_width - 8),
                    int(pos.y() + fm.height() / 3),
                    label
                )

        # Y-axis label (rotated)
        painter.setFont(self.title_font)
        y_label = self.y_axis.label
        if self.y_axis.unit:
            y_label += f" ({self.y_axis.unit})"

        painter.save()
        painter.translate(15, plot_rect.center().y())
        painter.rotate(-90)
        y_label_width = QFontMetrics(self.title_font).horizontalAdvance(y_label)
        painter.drawText(int(-y_label_width / 2), 0, y_label)
        painter.restore()

    def _draw_title(self, painter: QPainter):
        """Draw title with slice information."""
        painter.setFont(self.title_font)
        painter.setPen(QColor(255, 255, 255))

        title = f"{self.slice_direction.capitalize()}: {self.slice_value:.1f}"
        painter.drawText(self.LEFT_MARGIN, 20, title)

        # Zoom info on right
        zoom_x = self.data_x_max - self.data_x_min
        zoom_y = self.data_y_max - self.data_y_min
        if zoom_x > 0 and zoom_y > 0:
            zoom_level_x = zoom_x / self.view.x_range if self.view.x_range > 0 else 1
            zoom_level_y = zoom_y / self.view.y_range if self.view.y_range > 0 else 1
            zoom_info = f"Zoom: {min(zoom_level_x, zoom_level_y):.1f}x"
            painter.drawText(self.width() - 100, 20, zoom_info)

    def _draw_cursor_info(self, painter: QPainter):
        """Draw cursor position and amplitude."""
        plot_rect = self._get_plot_rect()

        painter.setFont(self.axis_font)
        painter.setPen(QColor(200, 200, 200))

        info = f"X: {self.mouse_data_pos.x():.1f}  Y: {self.mouse_data_pos.y():.1f}"

        # Get amplitude at cursor if within data
        if self.data is not None:
            h, w = self.data.shape
            x_scale = w / (self.data_x_max - self.data_x_min) if self.data_x_max > self.data_x_min else 1
            y_scale = h / (self.data_y_max - self.data_y_min) if self.data_y_max > self.data_y_min else 1

            ix = int((self.mouse_data_pos.x() - self.data_x_min) * x_scale)
            iy = int((self.mouse_data_pos.y() - self.data_y_min) * y_scale)

            if 0 <= ix < w and 0 <= iy < h:
                amp = self.data[iy, ix]
                info += f"  Amp: {amp:.4f}"

        painter.drawText(int(plot_rect.left()), int(plot_rect.top() - 5), info)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming and slice navigation."""
        delta = event.angleDelta().y()

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+wheel: zoom centered on cursor
            pos = event.position()
            data_pos = self._screen_to_data(pos.x(), pos.y())

            factor = 0.9 if delta > 0 else 1.1

            # Zoom around cursor position
            new_x_range = self.view.x_range * factor
            new_y_range = self.view.y_range * factor

            # Keep cursor position fixed in data coordinates
            x_ratio = (data_pos.x() - self.view.x_min) / self.view.x_range if self.view.x_range > 0 else 0.5
            y_ratio = (data_pos.y() - self.view.y_min) / self.view.y_range if self.view.y_range > 0 else 0.5

            new_x_min = data_pos.x() - x_ratio * new_x_range
            new_y_min = data_pos.y() - y_ratio * new_y_range

            self.view = ViewState(
                new_x_min, new_x_min + new_x_range,
                new_y_min, new_y_min + new_y_range
            )

            self.view_changed.emit(self.view)
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
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if this was a click (not a drag)
            pos = event.position()
            last = self.last_mouse_pos
            drag_distance = ((pos.x() - last.x())**2 + (pos.y() - last.y())**2)**0.5

            # Check if click is within plot area
            plot_rect = self._get_plot_rect()
            in_plot = (plot_rect.left() <= pos.x() <= plot_rect.right() and
                      plot_rect.top() <= pos.y() <= plot_rect.bottom())

            print(f"MouseRelease: pos=({pos.x():.0f},{pos.y():.0f}), drag={drag_distance:.1f}, in_plot={in_plot}")

            if drag_distance < 10 and in_plot:  # Click, not drag, within plot
                data_pos = self._screen_to_data(pos.x(), pos.y())
                print(f"  -> Emitting position_selected: ({data_pos.x():.1f}, {data_pos.y():.1f})")
                self.position_selected.emit(data_pos.x(), data_pos.y())

            self.dragging = False
            self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning and coordinate display."""
        pos = event.position()
        self.mouse_data_pos = self._screen_to_data(pos.x(), pos.y())

        if self.dragging:
            # Pan in data coordinates
            last_data = self._screen_to_data(self.last_mouse_pos.x(), self.last_mouse_pos.y())
            curr_data = self._screen_to_data(pos.x(), pos.y())

            dx = last_data.x() - curr_data.x()
            dy = last_data.y() - curr_data.y()

            self.view = ViewState(
                self.view.x_min + dx, self.view.x_max + dx,
                self.view.y_min + dy, self.view.y_max + dy
            )

            self.last_mouse_pos = pos
            self.view_changed.emit(self.view)

        # Emit cursor position
        if self.data is not None:
            h, w = self.data.shape
            x_scale = w / (self.data_x_max - self.data_x_min) if self.data_x_max > self.data_x_min else 1
            y_scale = h / (self.data_y_max - self.data_y_min) if self.data_y_max > self.data_y_min else 1

            ix = int((self.mouse_data_pos.x() - self.data_x_min) * x_scale)
            iy = int((self.mouse_data_pos.y() - self.data_y_min) * y_scale)

            if 0 <= ix < w and 0 <= iy < h:
                amp = float(self.data[iy, ix])
                self.cursor_moved.emit(self.mouse_data_pos.x(), self.mouse_data_pos.y(), amp)

        self.update()

    def enterEvent(self, event):
        """Change cursor when entering widget."""
        self.setCursor(Qt.CursorShape.CrossCursor)

    def leaveEvent(self, event):
        """Reset cursor when leaving widget."""
        self.setCursor(Qt.CursorShape.ArrowCursor)

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
            # Zoom in centered
            cx = (self.view.x_min + self.view.x_max) / 2
            cy = (self.view.y_min + self.view.y_max) / 2
            factor = 0.8
            new_x_range = self.view.x_range * factor
            new_y_range = self.view.y_range * factor
            self.view = ViewState(
                cx - new_x_range / 2, cx + new_x_range / 2,
                cy - new_y_range / 2, cy + new_y_range / 2
            )
            self.view_changed.emit(self.view)
            self.update()
        elif key == Qt.Key.Key_Minus:
            cx = (self.view.x_min + self.view.x_max) / 2
            cy = (self.view.y_min + self.view.y_max) / 2
            factor = 1.25
            new_x_range = self.view.x_range * factor
            new_y_range = self.view.y_range * factor
            self.view = ViewState(
                cx - new_x_range / 2, cx + new_x_range / 2,
                cy - new_y_range / 2, cy + new_y_range / 2
            )
            self.view_changed.emit(self.view)
            self.update()


# =============================================================================
# Volume Panel (contains canvas + controls for one volume)
# =============================================================================

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


# =============================================================================
# Gather Panel (for CIG visualization)
# =============================================================================

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


# =============================================================================
# Velocity Analysis Functions (Vectorized for Speed)
# =============================================================================

def create_super_gather(offset_bins: List[zarr.Array], il_center: int, xl_center: int,
                        il_half: int, xl_half: int, offset_values: np.ndarray,
                        offset_bin_size: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create super gather by stacking traces in spatial window and rebinning offsets.
    Vectorized implementation for speed.
    """
    if len(offset_bins) == 0:
        return None, None

    n_il, n_xl, n_time = offset_bins[0].shape

    # Clamp window to data bounds
    il_min = max(0, il_center - il_half)
    il_max = min(n_il - 1, il_center + il_half) + 1
    xl_min = max(0, xl_center - xl_half)
    xl_max = min(n_xl - 1, xl_center + xl_half) + 1

    # Determine new offset bins
    off_min = offset_values.min()
    off_max = offset_values.max()
    n_new_bins = max(1, int(np.ceil((off_max - off_min) / offset_bin_size)))
    new_offsets = np.arange(n_new_bins) * offset_bin_size + off_min + offset_bin_size / 2

    # Pre-compute bin assignments for all original offsets
    bin_indices = np.clip(((offset_values - off_min) / offset_bin_size).astype(int), 0, n_new_bins - 1)

    # Initialize super gather
    super_gather = np.zeros((n_new_bins, n_time), dtype=np.float64)
    fold = np.zeros(n_new_bins, dtype=np.int32)

    # Stack traces - read entire spatial window at once per offset bin
    for i, offset_zarr in enumerate(offset_bins):
        bin_idx = bin_indices[i]
        # Read entire spatial window at once (vectorized)
        window_data = np.asarray(offset_zarr[il_min:il_max, xl_min:xl_max, :])
        # Sum all traces in window
        super_gather[bin_idx, :] += window_data.sum(axis=(0, 1))
        fold[bin_idx] += window_data.shape[0] * window_data.shape[1]

    # Normalize by fold (vectorized)
    valid = fold > 0
    super_gather[valid, :] /= fold[valid, np.newaxis]

    return super_gather.astype(np.float32), new_offsets


def apply_velocity_mute(gather: np.ndarray, offsets: np.ndarray, t_coords: np.ndarray,
                        v_top: float = None, v_bottom: float = None) -> np.ndarray:
    """Apply velocity-based mute to gather. Fully vectorized implementation."""
    if gather is None:
        return None

    muted = gather.copy()
    n_offsets, n_time = gather.shape
    dt = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

    # Create time grid (1, n_time)
    time_grid = t_coords[np.newaxis, :]  # (1, n_time) in ms

    # Top mute: zero out samples before t = offset/v_top
    if v_top is not None and v_top > 0:
        t_top = np.abs(offsets[:, np.newaxis]) / v_top * 1000.0  # (n_offsets, 1) in ms
        mute_mask_top = time_grid < t_top
        muted[mute_mask_top] = 0

    # Bottom mute: zero out samples after t = offset/v_bottom
    if v_bottom is not None and v_bottom > 0:
        t_bottom = np.abs(offsets[:, np.newaxis]) / v_bottom * 1000.0  # (n_offsets, 1) in ms
        mute_mask_bottom = time_grid > t_bottom
        muted[mute_mask_bottom] = 0

    return muted


def apply_nmo_correction(gather: np.ndarray, offsets: np.ndarray, t_coords: np.ndarray,
                         velocity: float, inverse: bool = False) -> np.ndarray:
    """
    Apply NMO correction to gather. Fully vectorized implementation.

    Args:
        gather: (n_offsets, n_time) array
        offsets: Offset values in meters
        t_coords: Time values in ms
        velocity: NMO velocity in m/s
        inverse: If True, apply inverse NMO (stretch flat to hyperbola)
                 If False, apply forward NMO (flatten hyperbola)

    Returns:
        NMO corrected gather
    """
    if gather is None or velocity <= 0:
        return gather

    n_offsets, n_time = gather.shape
    dt = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

    # Output time grid (1, n_time) - where we place values
    time_grid = t_coords[np.newaxis, :]
    off_grid = offsets[:, np.newaxis]   # (n_offsets, 1)

    # Compute travel time correction term: (x/v)^2 in ms^2
    stretch_term_sq = (off_grid / velocity * 1000.0) ** 2

    if inverse:
        # Inverse NMO: stretch flat events back to hyperbola
        # At output time t, sample from input at t0 = sqrt(t^2 - (x/v)^2)
        t_sq = time_grid ** 2
        src_time = np.sqrt(np.maximum(0, t_sq - stretch_term_sq))
    else:
        # Forward NMO: flatten hyperbolic events
        # At output time t0, sample from input at t = sqrt(t0^2 + (x/v)^2)
        src_time = np.sqrt(time_grid**2 + stretch_term_sq)

    # Convert to indices and clip to valid range
    idx_src = np.clip((src_time / dt).astype(np.int32), 0, n_time - 1)

    # Fully vectorized gather using advanced indexing
    row_idx = np.arange(n_offsets)[:, np.newaxis]
    corrected = gather[row_idx, idx_src]

    return corrected.astype(np.float32)


def apply_bandpass_filter(data: np.ndarray, dt_ms: float,
                          f_low: float = 5.0, f_high: float = 80.0,
                          taper_width: float = 5.0) -> np.ndarray:
    """
    Apply bandpass filter using FFT. Vectorized for 2D arrays.

    Args:
        data: (n_traces, n_time) or (n_time,) array
        dt_ms: Sample interval in milliseconds
        f_low, f_high: Corner frequencies in Hz
        taper_width: Taper width in Hz for smooth rolloff
    """
    if data is None:
        return None

    was_1d = data.ndim == 1
    if was_1d:
        data = data[np.newaxis, :]

    n_traces, n_time = data.shape
    dt_s = dt_ms / 1000.0

    # FFT
    spectrum = np.fft.rfft(data, axis=1)
    freqs = np.fft.rfftfreq(n_time, dt_s)

    # Create bandpass filter with cosine taper
    filt = np.zeros_like(freqs)

    # Passband
    passband = (freqs >= f_low) & (freqs <= f_high)
    filt[passband] = 1.0

    # Low taper
    low_taper = (freqs >= f_low - taper_width) & (freqs < f_low)
    if np.any(low_taper):
        filt[low_taper] = 0.5 * (1 + np.cos(np.pi * (freqs[low_taper] - f_low) / taper_width))

    # High taper
    high_taper = (freqs > f_high) & (freqs <= f_high + taper_width)
    if np.any(high_taper):
        filt[high_taper] = 0.5 * (1 + np.cos(np.pi * (freqs[high_taper] - f_high) / taper_width))

    # Apply filter
    filtered = np.fft.irfft(spectrum * filt, n=n_time, axis=1)

    if was_1d:
        filtered = filtered[0]

    return filtered.astype(np.float32)


def apply_agc(data: np.ndarray, window_ms: float = 500.0, dt_ms: float = 2.0) -> np.ndarray:
    """
    Apply Automatic Gain Control. Fully vectorized implementation.

    Args:
        data: (n_traces, n_time) or (n_time,) array
        window_ms: AGC window length in milliseconds
        dt_ms: Sample interval in milliseconds
    """
    if data is None:
        return None

    was_1d = data.ndim == 1
    if was_1d:
        data = data[np.newaxis, :]

    n_traces, n_time = data.shape
    window_samples = max(1, int(window_ms / dt_ms))
    half_win = window_samples // 2

    # Compute envelope using sliding window RMS with cumsum
    data_sq = data ** 2

    # Pad data for edge handling
    padded = np.pad(data_sq, ((0, 0), (half_win, half_win)), mode='reflect')

    # Use cumsum for fast sliding window (fully vectorized)
    cumsum = np.zeros((n_traces, padded.shape[1] + 1))
    cumsum[:, 1:] = np.cumsum(padded, axis=1)

    # Compute RMS using vectorized slicing
    left_idx = np.arange(n_time)
    right_idx = left_idx + window_samples
    window_sums = cumsum[:, right_idx + 1] - cumsum[:, left_idx]
    rms = np.sqrt(window_sums / window_samples)

    # Apply gain (avoid division by zero)
    rms_max = rms.max()
    if rms_max > 0:
        rms = np.maximum(rms, rms_max * 1e-6)
        agc_data = data / rms
    else:
        agc_data = data.copy()

    if was_1d:
        agc_data = agc_data[0]

    return agc_data.astype(np.float32)


def compute_semblance_fast(gather: np.ndarray, offsets: np.ndarray, t_coords: np.ndarray,
                           v_min: float = 1500, v_max: float = 5000, v_step: float = 50,
                           window_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute semblance for velocity analysis. Fast vectorized implementation.
    """
    if gather is None:
        return None, None

    n_offsets, n_time = gather.shape
    dt = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

    velocities = np.arange(v_min, v_max + v_step, v_step, dtype=np.float32)
    n_vel = len(velocities)
    semblance = np.zeros((n_vel, n_time), dtype=np.float32)
    half_win = window_samples // 2

    # Pre-compute offset grid
    off_sq = (offsets[:, np.newaxis] / 1000.0) ** 2  # (n_offsets, 1), in km^2
    t0_sq = (t_coords[np.newaxis, :] / 1000.0) ** 2  # (1, n_time), in s^2

    for iv, vel in enumerate(velocities):
        # Vectorized NMO: t = sqrt(t0^2 + x^2/v^2)
        v_sq = (vel / 1000.0) ** 2  # km^2/s^2
        t_nmo = np.sqrt(t0_sq + off_sq / v_sq) * 1000.0  # Back to ms

        # Convert to indices
        idx_nmo = np.clip((t_nmo / dt).astype(np.int32), 0, n_time - 1)

        # Gather NMO-corrected traces using advanced indexing
        row_idx = np.arange(n_offsets)[:, np.newaxis]
        nmo_gather = gather[row_idx, idx_nmo]

        # Compute semblance using vectorized operations
        # Use cumsum for efficient window sums
        cumsum = np.zeros((n_offsets, n_time + 1))
        cumsum[:, 1:] = np.cumsum(nmo_gather, axis=1)
        cumsum_sq = np.zeros((n_offsets, n_time + 1))
        cumsum_sq[:, 1:] = np.cumsum(nmo_gather ** 2, axis=1)

        for it in range(half_win, n_time - half_win):
            left = it - half_win
            right = it + half_win + 1

            # Sum across traces for each time in window
            trace_sums = cumsum[:, right] - cumsum[:, left]  # (n_offsets,)
            stack_sum = trace_sums.sum()  # Stack amplitude

            trace_sq_sums = cumsum_sq[:, right] - cumsum_sq[:, left]
            energy = trace_sq_sums.sum()

            window_size = right - left
            numerator = stack_sum ** 2
            denominator = n_offsets * window_size * energy

            if denominator > 1e-10:
                semblance[iv, it] = numerator / denominator

    return semblance, velocities


def load_velocity_model(path: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Load velocity model from zarr or parquet file.

    Returns:
        Tuple of (velocity_data, metadata)
        - velocity_data: numpy array (can be 1D, 2D, or 3D)
        - metadata: dict with coordinate info (t_coords, il_coords, xl_coords)
    """
    path = Path(path)
    metadata = {}

    try:
        if path.suffix == '.parquet':
            import pandas as pd
            df = pd.read_parquet(path)
            # Check for different column formats
            if 'time_ms' in df.columns and 'velocity_ms' in df.columns:
                # 1D velocity function: (time, velocity)
                metadata['t_coords'] = df['time_ms'].values
                return df['velocity_ms'].values, metadata
            elif 'IL' in df.columns and 'XL' in df.columns:
                # 3D velocity: pivot to cube
                if 'time_ms' in df.columns:
                    metadata['t_coords'] = df['time_ms'].unique()
                return df, metadata  # Return DataFrame for 3D handling
            else:
                return df.values, metadata

        elif path.suffix == '.zarr' or path.is_dir():
            z = zarr.open(str(path), mode='r')

            # Try to get velocity array
            if isinstance(z, zarr.Array):
                vel_data = z
            elif hasattr(z, 'keys'):
                # It's a Group - look for velocity arrays
                if 'velocity' in z:
                    vel_data = z['velocity']
                elif 'vrms' in z:
                    vel_data = z['vrms']
                elif 'vint' in z:
                    vel_data = z['vint']
                else:
                    # Try first array found
                    for key in z.keys():
                        if isinstance(z[key], zarr.Array):
                            vel_data = z[key]
                            break
                    else:
                        vel_data = z
            else:
                vel_data = z

            # Load metadata/coordinates from attrs
            if hasattr(vel_data, 'attrs'):
                attrs = dict(vel_data.attrs)

                # Try various attribute names for time coordinates
                for t_key in ['t_coords', 't_axis_ms', 't', 'time']:
                    if t_key in attrs:
                        metadata['t_coords'] = np.array(attrs[t_key])
                        break

                # Try various attribute names for IL coordinates
                for il_key in ['il_coords', 'x_axis', 'x', 'inline']:
                    if il_key in attrs:
                        metadata['il_coords'] = np.array(attrs[il_key])
                        break

                # Try various attribute names for XL coordinates
                for xl_key in ['xl_coords', 'y_axis', 'y', 'crossline']:
                    if xl_key in attrs:
                        metadata['xl_coords'] = np.array(attrs[xl_key])
                        break

                if 'dt_ms' in attrs:
                    metadata['dt_ms'] = float(attrs['dt_ms'])

            # Check for coordinate arrays in zarr Group (only if it's a Group)
            if hasattr(z, 'keys'):
                if 'time' in z:
                    metadata['t_coords'] = np.asarray(z['time'])
                if 'inline' in z:
                    metadata['il_coords'] = np.asarray(z['inline'])
                if 'crossline' in z:
                    metadata['xl_coords'] = np.asarray(z['crossline'])

            return np.asarray(vel_data), metadata

        return None, None

    except Exception as e:
        print(f"Error loading velocity: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_velocity_function(vel_model: np.ndarray, il_idx: int, xl_idx: int,
                               t_coords: np.ndarray) -> np.ndarray:
    """
    Extract 1D velocity function at given IL/XL position.

    Args:
        vel_model: Velocity model (1D, 2D, or 3D)
        il_idx: Inline index
        xl_idx: Crossline index
        t_coords: Time coordinates

    Returns:
        1D velocity array (n_time,)
    """
    if vel_model is None:
        return None

    if vel_model.ndim == 1:
        # 1D velocity - same for all positions
        return vel_model
    elif vel_model.ndim == 2:
        # 2D velocity (IL, time) or (time, velocity pairs)
        if vel_model.shape[0] == len(t_coords):
            return vel_model[:, 0] if vel_model.shape[1] > 1 else vel_model.flatten()
        else:
            # Assume (IL, time)
            il_idx = min(il_idx, vel_model.shape[0] - 1)
            return vel_model[il_idx, :]
    elif vel_model.ndim == 3:
        # 3D velocity (IL, XL, time)
        il_idx = min(il_idx, vel_model.shape[0] - 1)
        xl_idx = min(xl_idx, vel_model.shape[1] - 1)
        return vel_model[il_idx, xl_idx, :]

    return None


def apply_nmo_with_velocity_model(gather: np.ndarray, offsets: np.ndarray,
                                   t_coords: np.ndarray, velocity_func: np.ndarray,
                                   inverse: bool = False) -> np.ndarray:
    """
    Apply NMO correction using time-varying velocity function.

    Args:
        gather: (n_offsets, n_time) array
        offsets: Offset values in meters
        t_coords: Time values in ms
        velocity_func: 1D array of velocities at each time sample (m/s)
        inverse: If True, apply inverse NMO

    Returns:
        NMO corrected gather
    """
    if gather is None or velocity_func is None:
        return gather

    n_offsets, n_time = gather.shape
    dt = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

    # Ensure velocity array matches time samples
    if len(velocity_func) != n_time:
        # Interpolate velocity to match time samples
        vel_t = np.linspace(0, t_coords[-1], len(velocity_func))
        velocity_func = np.interp(t_coords, vel_t, velocity_func)

    # Output time grid and offset grid
    time_grid = t_coords[np.newaxis, :]  # (1, n_time)
    off_grid = offsets[:, np.newaxis]    # (n_offsets, 1)
    vel_grid = velocity_func[np.newaxis, :]  # (1, n_time)

    # Compute travel time correction: (x/v(t))^2 in ms^2
    stretch_term_sq = (off_grid / vel_grid * 1000.0) ** 2

    if inverse:
        # Inverse NMO: at output time t, sample from t0 = sqrt(t^2 - (x/v)^2)
        t_sq = time_grid ** 2
        src_time = np.sqrt(np.maximum(0, t_sq - stretch_term_sq))
    else:
        # Forward NMO: at output time t0, sample from t = sqrt(t0^2 + (x/v)^2)
        src_time = np.sqrt(time_grid**2 + stretch_term_sq)

    # Convert to indices
    idx_src = np.clip((src_time / dt).astype(np.int32), 0, n_time - 1)

    # Vectorized gather
    row_idx = np.arange(n_offsets)[:, np.newaxis]
    corrected = gather[row_idx, idx_src]

    return corrected.astype(np.float32)


# =============================================================================
# Semblance Settings Dialog
# =============================================================================

class SemblanceSettingsDialog(QDialog):
    """Dialog for configuring semblance calculation and display parameters."""

    # Available colormaps
    COLORMAPS = ["Gray", "Seismic (BWR)", "Seismic (RWB)", "Viridis", "Bone"]

    def __init__(self, parent=None, settings: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Velocity Analysis Settings")
        self.setMinimumWidth(350)

        # Default settings
        self.settings = settings or {
            'v_min': 1500.0,
            'v_max': 5000.0,
            'v_step': 100.0,
            'window_samples': 5,
            # Gather display
            'gather_colormap': 'Seismic (BWR)',
            'gather_clip': 99.0,
            'gather_gain': 1.0,
            # Semblance/Velocity display
            'semblance_colormap': 'Viridis',
            'semblance_clip': 99.0,
            'velocity_colormap': 'Viridis',
            'velocity_clip': 99.0,
        }

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Velocity range
        vel_group = QGroupBox("Velocity Scan Range")
        vel_layout = QFormLayout(vel_group)

        self.v_min_spin = QSpinBox()
        self.v_min_spin.setRange(500, 5000)
        self.v_min_spin.setValue(int(self.settings.get('v_min', 1500)))
        self.v_min_spin.setSingleStep(100)
        self.v_min_spin.setSuffix(" m/s")
        vel_layout.addRow("V min:", self.v_min_spin)

        self.v_max_spin = QSpinBox()
        self.v_max_spin.setRange(1000, 10000)
        self.v_max_spin.setValue(int(self.settings.get('v_max', 5000)))
        self.v_max_spin.setSingleStep(100)
        self.v_max_spin.setSuffix(" m/s")
        vel_layout.addRow("V max:", self.v_max_spin)

        self.v_step_spin = QSpinBox()
        self.v_step_spin.setRange(10, 500)
        self.v_step_spin.setValue(int(self.settings.get('v_step', 100)))
        self.v_step_spin.setSingleStep(10)
        self.v_step_spin.setSuffix(" m/s")
        vel_layout.addRow("V step:", self.v_step_spin)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 21)
        self.window_spin.setValue(int(self.settings.get('window_samples', 5)))
        self.window_spin.setSingleStep(2)
        self.window_spin.setSuffix(" samples")
        self.window_spin.setToolTip("Semblance window size (odd values recommended)")
        vel_layout.addRow("Window size:", self.window_spin)

        layout.addWidget(vel_group)

        # Gather display settings
        gather_group = QGroupBox("Gather Display")
        gather_layout = QFormLayout(gather_group)

        self.gather_cmap_combo = QComboBox()
        self.gather_cmap_combo.addItems(self.COLORMAPS)
        current_cmap = self.settings.get('gather_colormap', 'Seismic (BWR)')
        if current_cmap in self.COLORMAPS:
            self.gather_cmap_combo.setCurrentText(current_cmap)
        gather_layout.addRow("Colormap:", self.gather_cmap_combo)

        self.gather_clip_spin = QDoubleSpinBox()
        self.gather_clip_spin.setRange(90.0, 100.0)
        self.gather_clip_spin.setValue(self.settings.get('gather_clip', 99.0))
        self.gather_clip_spin.setSingleStep(0.5)
        self.gather_clip_spin.setSuffix(" %")
        gather_layout.addRow("Clip percentile:", self.gather_clip_spin)

        self.gather_gain_spin = QDoubleSpinBox()
        self.gather_gain_spin.setRange(0.1, 100.0)
        self.gather_gain_spin.setValue(self.settings.get('gather_gain', 1.0))
        self.gather_gain_spin.setSingleStep(0.1)
        gather_layout.addRow("Gain:", self.gather_gain_spin)

        layout.addWidget(gather_group)

        # Semblance display settings
        sem_group = QGroupBox("Semblance Display")
        sem_layout = QFormLayout(sem_group)

        self.sem_cmap_combo = QComboBox()
        self.sem_cmap_combo.addItems(self.COLORMAPS)
        current_cmap = self.settings.get('semblance_colormap', 'Viridis')
        if current_cmap in self.COLORMAPS:
            self.sem_cmap_combo.setCurrentText(current_cmap)
        sem_layout.addRow("Colormap:", self.sem_cmap_combo)

        self.sem_clip_spin = QDoubleSpinBox()
        self.sem_clip_spin.setRange(90.0, 100.0)
        self.sem_clip_spin.setValue(self.settings.get('semblance_clip', 99.0))
        self.sem_clip_spin.setSingleStep(0.5)
        self.sem_clip_spin.setSuffix(" %")
        sem_layout.addRow("Clip percentile:", self.sem_clip_spin)

        layout.addWidget(sem_group)

        # Velocity model display settings
        vel_disp_group = QGroupBox("Velocity Model Display")
        vel_disp_layout = QFormLayout(vel_disp_group)

        self.vel_cmap_combo = QComboBox()
        self.vel_cmap_combo.addItems(self.COLORMAPS)
        current_cmap = self.settings.get('velocity_colormap', 'Viridis')
        if current_cmap in self.COLORMAPS:
            self.vel_cmap_combo.setCurrentText(current_cmap)
        vel_disp_layout.addRow("Colormap:", self.vel_cmap_combo)

        self.vel_clip_spin = QDoubleSpinBox()
        self.vel_clip_spin.setRange(90.0, 100.0)
        self.vel_clip_spin.setValue(self.settings.get('velocity_clip', 99.0))
        self.vel_clip_spin.setSingleStep(0.5)
        self.vel_clip_spin.setSuffix(" %")
        vel_disp_layout.addRow("Clip percentile:", self.vel_clip_spin)

        # Use min/max range option
        self.vel_use_minmax_check = QCheckBox("Use explicit min/max range")
        self.vel_use_minmax_check.setChecked(self.settings.get('velocity_use_minmax', False))
        self.vel_use_minmax_check.toggled.connect(self._on_vel_minmax_toggled)
        vel_disp_layout.addRow(self.vel_use_minmax_check)

        self.vel_min_spin = QSpinBox()
        self.vel_min_spin.setRange(500, 10000)
        self.vel_min_spin.setValue(int(self.settings.get('velocity_vmin', 1500)))
        self.vel_min_spin.setSingleStep(100)
        self.vel_min_spin.setSuffix(" m/s")
        self.vel_min_spin.setEnabled(self.settings.get('velocity_use_minmax', False))
        vel_disp_layout.addRow("V min:", self.vel_min_spin)

        self.vel_max_spin = QSpinBox()
        self.vel_max_spin.setRange(500, 10000)
        self.vel_max_spin.setValue(int(self.settings.get('velocity_vmax', 5000)))
        self.vel_max_spin.setSingleStep(100)
        self.vel_max_spin.setSuffix(" m/s")
        self.vel_max_spin.setEnabled(self.settings.get('velocity_use_minmax', False))
        vel_disp_layout.addRow("V max:", self.vel_max_spin)

        layout.addWidget(vel_disp_group)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_vel_minmax_toggled(self, checked: bool):
        """Enable/disable velocity min/max spinboxes based on checkbox."""
        self.vel_min_spin.setEnabled(checked)
        self.vel_max_spin.setEnabled(checked)
        self.vel_clip_spin.setEnabled(not checked)

    def get_settings(self) -> dict:
        """Return the current settings from the dialog."""
        return {
            'v_min': float(self.v_min_spin.value()),
            'v_max': float(self.v_max_spin.value()),
            'v_step': float(self.v_step_spin.value()),
            'window_samples': self.window_spin.value(),
            # Gather display
            'gather_colormap': self.gather_cmap_combo.currentText(),
            'gather_clip': self.gather_clip_spin.value(),
            'gather_gain': self.gather_gain_spin.value(),
            # Semblance display
            'semblance_colormap': self.sem_cmap_combo.currentText(),
            'semblance_clip': self.sem_clip_spin.value(),
            # Velocity display
            'velocity_colormap': self.vel_cmap_combo.currentText(),
            'velocity_clip': self.vel_clip_spin.value(),
            'velocity_use_minmax': self.vel_use_minmax_check.isChecked(),
            'velocity_vmin': float(self.vel_min_spin.value()),
            'velocity_vmax': float(self.vel_max_spin.value()),
        }


# =============================================================================
# Velocity Analysis Window
# =============================================================================

class VelocityAnalysisWindow(QMainWindow):
    """Separate window for velocity analysis with super gathers and semblance."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Velocity Analysis")
        self.setMinimumSize(1100, 750)

        # Data references
        self.offset_bins: List[zarr.Array] = []
        self.offset_values: np.ndarray = None
        self.t_coords: np.ndarray = None
        self.dt_ms = 2.0

        # Current position
        self.il_center = 0
        self.xl_center = 0

        # Super gather parameters
        self.il_half = 2  # 5x5 window
        self.xl_half = 2
        self.offset_bin_size = 50.0  # meters

        # Mute parameters
        self.v_top = 1500.0  # m/s
        self.v_bottom = 4000.0  # m/s
        self.top_mute_enabled = False
        self.bottom_mute_enabled = False

        # Processing parameters
        self.apply_bandpass = False
        self.f_low = 5.0
        self.f_high = 80.0
        self.apply_agc_flag = False
        self.agc_window = 500.0
        self.apply_inv_nmo = False
        self.nmo_velocity = 2500.0

        # Semblance parameters
        self.v_min = 1500.0
        self.v_max = 5000.0
        self.v_step = 100.0  # Larger step for speed
        self.semblance_window_samples = 5  # Window size for semblance calculation

        # Display settings
        self.gather_colormap = "Seismic (BWR)"
        self.gather_clip = 99.0
        self.gather_gain = 1.0
        self.semblance_colormap = "Viridis"
        self.semblance_clip = 99.0
        self.velocity_colormap = "Viridis"
        self.velocity_clip = 99.0
        self.velocity_use_minmax = False
        self.velocity_vmin = 1500.0
        self.velocity_vmax = 5000.0

        # Current data
        self.super_gather = None
        self.super_offsets = None
        self.processed_gather = None
        self.semblance = None
        self.velocities = None

        # Velocity model data
        self.vel_model = None
        self.vel_metadata = None
        self.vel_function = None  # 1D velocity at current position
        self.use_vel_model = False  # Use loaded model vs constant velocity

        self._setup_ui()

    def _setup_ui(self):
        # Create menubar
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Settings")

        # Semblance settings action
        semblance_settings_action = settings_menu.addAction("Semblance Parameters...")
        semblance_settings_action.triggered.connect(self._open_semblance_settings)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: Controls
        controls = QWidget()
        controls.setMaximumWidth(220)
        ctrl_layout = QVBoxLayout(controls)

        # Position display
        pos_group = QGroupBox("Position")
        pos_layout = QVBoxLayout(pos_group)
        self.pos_label = QLabel("IL=0, XL=0")
        self.pos_label.setStyleSheet("font-weight: bold;")
        pos_layout.addWidget(self.pos_label)
        ctrl_layout.addWidget(pos_group)

        # Data loading
        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout(data_group)

        self.load_gathers_btn = QPushButton("Load Gathers...")
        self.load_gathers_btn.clicked.connect(self._load_gathers)
        data_layout.addWidget(self.load_gathers_btn)

        self.gathers_label = QLabel("No gathers loaded")
        self.gathers_label.setStyleSheet("color: gray; font-size: 9px;")
        self.gathers_label.setWordWrap(True)
        data_layout.addWidget(self.gathers_label)

        ctrl_layout.addWidget(data_group)

        # Super gather parameters
        sg_group = QGroupBox("Super Gather")
        sg_layout = QVBoxLayout(sg_group)

        il_layout = QHBoxLayout()
        il_layout.addWidget(QLabel("IL window:"))
        self.il_spin = QSpinBox()
        self.il_spin.setRange(1, 20)
        self.il_spin.setValue(5)
        self.il_spin.valueChanged.connect(self._on_param_changed)
        il_layout.addWidget(self.il_spin)
        sg_layout.addLayout(il_layout)

        xl_layout = QHBoxLayout()
        xl_layout.addWidget(QLabel("XL window:"))
        self.xl_spin = QSpinBox()
        self.xl_spin.setRange(1, 20)
        self.xl_spin.setValue(5)
        self.xl_spin.valueChanged.connect(self._on_param_changed)
        xl_layout.addWidget(self.xl_spin)
        sg_layout.addLayout(xl_layout)

        off_layout = QHBoxLayout()
        off_layout.addWidget(QLabel("Offset bin (m):"))
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(10, 500)
        self.offset_spin.setValue(50)
        self.offset_spin.setSingleStep(10)
        self.offset_spin.valueChanged.connect(self._on_param_changed)
        off_layout.addWidget(self.offset_spin)
        sg_layout.addLayout(off_layout)

        ctrl_layout.addWidget(sg_group)

        # Mute parameters
        mute_group = QGroupBox("Velocity Mute")
        mute_layout = QVBoxLayout(mute_group)

        # Top mute with its own checkbox
        self.top_mute_check = QCheckBox("Top Mute")
        self.top_mute_check.toggled.connect(self._on_mute_changed)
        mute_layout.addWidget(self.top_mute_check)

        vtop_layout = QHBoxLayout()
        vtop_layout.addWidget(QLabel("V top (m/s):"))
        self.vtop_spin = QSpinBox()
        self.vtop_spin.setRange(500, 10000)
        self.vtop_spin.setValue(1500)
        self.vtop_spin.setSingleStep(100)
        self.vtop_spin.valueChanged.connect(self._on_mute_changed)
        vtop_layout.addWidget(self.vtop_spin)
        mute_layout.addLayout(vtop_layout)

        # Bottom mute with its own checkbox
        self.bottom_mute_check = QCheckBox("Bottom Mute")
        self.bottom_mute_check.toggled.connect(self._on_mute_changed)
        mute_layout.addWidget(self.bottom_mute_check)

        vbot_layout = QHBoxLayout()
        vbot_layout.addWidget(QLabel("V bottom (m/s):"))
        self.vbot_spin = QSpinBox()
        self.vbot_spin.setRange(500, 10000)
        self.vbot_spin.setValue(4000)
        self.vbot_spin.setSingleStep(100)
        self.vbot_spin.valueChanged.connect(self._on_mute_changed)
        vbot_layout.addWidget(self.vbot_spin)
        mute_layout.addLayout(vbot_layout)

        ctrl_layout.addWidget(mute_group)

        # Processing parameters
        proc_group = QGroupBox("Processing")
        proc_layout = QVBoxLayout(proc_group)

        # Inverse NMO
        self.inv_nmo_check = QCheckBox("Inverse NMO")
        self.inv_nmo_check.toggled.connect(self._on_processing_changed)
        proc_layout.addWidget(self.inv_nmo_check)

        vnmo_layout = QHBoxLayout()
        vnmo_layout.addWidget(QLabel("V NMO (m/s):"))
        self.vnmo_spin = QSpinBox()
        self.vnmo_spin.setRange(500, 8000)
        self.vnmo_spin.setValue(2500)
        self.vnmo_spin.setSingleStep(100)
        self.vnmo_spin.valueChanged.connect(self._on_processing_changed)
        vnmo_layout.addWidget(self.vnmo_spin)
        proc_layout.addLayout(vnmo_layout)

        # Bandpass filter
        self.bp_check = QCheckBox("Bandpass Filter")
        self.bp_check.toggled.connect(self._on_processing_changed)
        proc_layout.addWidget(self.bp_check)

        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("f:"))
        self.f_low_spin = QSpinBox()
        self.f_low_spin.setRange(1, 100)
        self.f_low_spin.setValue(5)
        self.f_low_spin.valueChanged.connect(self._on_processing_changed)
        freq_layout.addWidget(self.f_low_spin)
        freq_layout.addWidget(QLabel("-"))
        self.f_high_spin = QSpinBox()
        self.f_high_spin.setRange(10, 200)
        self.f_high_spin.setValue(80)
        self.f_high_spin.valueChanged.connect(self._on_processing_changed)
        freq_layout.addWidget(self.f_high_spin)
        freq_layout.addWidget(QLabel("Hz"))
        proc_layout.addLayout(freq_layout)

        # AGC
        self.agc_check = QCheckBox("AGC")
        self.agc_check.toggled.connect(self._on_processing_changed)
        proc_layout.addWidget(self.agc_check)

        agc_layout = QHBoxLayout()
        agc_layout.addWidget(QLabel("Window (ms):"))
        self.agc_spin = QSpinBox()
        self.agc_spin.setRange(50, 2000)
        self.agc_spin.setValue(500)
        self.agc_spin.setSingleStep(50)
        self.agc_spin.valueChanged.connect(self._on_processing_changed)
        agc_layout.addWidget(self.agc_spin)
        proc_layout.addLayout(agc_layout)

        ctrl_layout.addWidget(proc_group)

        # Velocity Model group
        vel_group = QGroupBox("Velocity Model")
        vel_layout = QVBoxLayout(vel_group)

        self.load_vel_btn = QPushButton("Load Velocity...")
        self.load_vel_btn.clicked.connect(self._load_velocity_model)
        vel_layout.addWidget(self.load_vel_btn)

        self.vel_file_label = QLabel("No velocity loaded")
        self.vel_file_label.setStyleSheet("color: gray; font-size: 9px;")
        self.vel_file_label.setWordWrap(True)
        vel_layout.addWidget(self.vel_file_label)

        self.use_vel_check = QCheckBox("Use loaded velocity")
        self.use_vel_check.setEnabled(False)
        self.use_vel_check.toggled.connect(self._on_vel_model_changed)
        vel_layout.addWidget(self.use_vel_check)

        ctrl_layout.addWidget(vel_group)

        # Semblance parameters
        sem_group = QGroupBox("Semblance")
        sem_layout = QVBoxLayout(sem_group)

        vmin_layout = QHBoxLayout()
        vmin_layout.addWidget(QLabel("V min (m/s):"))
        self.vmin_spin = QSpinBox()
        self.vmin_spin.setRange(500, 5000)
        self.vmin_spin.setValue(1500)
        self.vmin_spin.setSingleStep(100)
        vmin_layout.addWidget(self.vmin_spin)
        sem_layout.addLayout(vmin_layout)

        vmax_layout = QHBoxLayout()
        vmax_layout.addWidget(QLabel("V max (m/s):"))
        self.vmax_spin = QSpinBox()
        self.vmax_spin.setRange(1000, 10000)
        self.vmax_spin.setValue(5000)
        self.vmax_spin.setSingleStep(100)
        vmax_layout.addWidget(self.vmax_spin)
        sem_layout.addLayout(vmax_layout)

        self.compute_btn = QPushButton("Compute Semblance")
        self.compute_btn.clicked.connect(self._compute_semblance)
        sem_layout.addWidget(self.compute_btn)

        ctrl_layout.addWidget(sem_group)

        # Display controls
        disp_group = QGroupBox("Display")
        disp_layout = QVBoxLayout(disp_group)

        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 100.0)
        self.gain_spin.setValue(1.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.valueChanged.connect(self._update_display)
        gain_layout.addWidget(self.gain_spin)
        disp_layout.addLayout(gain_layout)

        ctrl_layout.addWidget(disp_group)
        ctrl_layout.addStretch()

        layout.addWidget(controls)

        # Right: Canvases
        canvas_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Super gather canvas
        sg_frame = QFrame()
        sg_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        sg_layout = QVBoxLayout(sg_frame)
        sg_layout.addWidget(QLabel("Super Gather"))
        self.gather_canvas = SeismicCanvas()
        sg_layout.addWidget(self.gather_canvas)
        canvas_splitter.addWidget(sg_frame)

        # Semblance canvas
        sem_frame = QFrame()
        sem_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        sem_layout_f = QVBoxLayout(sem_frame)
        sem_layout_f.addWidget(QLabel("Semblance"))
        self.semblance_canvas = SeismicCanvas()
        sem_layout_f.addWidget(self.semblance_canvas)
        canvas_splitter.addWidget(sem_frame)

        # Velocity model canvas
        vel_frame = QFrame()
        vel_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        vel_layout_f = QVBoxLayout(vel_frame)
        vel_layout_f.addWidget(QLabel("Velocity Model"))
        self.velocity_canvas = SeismicCanvas()
        vel_layout_f.addWidget(self.velocity_canvas)
        canvas_splitter.addWidget(vel_frame)

        layout.addWidget(canvas_splitter, 1)

        self.statusBar().showMessage("Ready")

    def set_gather_data(self, offset_bins: List[zarr.Array], offset_values: np.ndarray,
                        t_coords: np.ndarray):
        """Set the gather data source."""
        self.offset_bins = offset_bins
        self.offset_values = offset_values
        self.t_coords = t_coords
        self.gathers_label.setText(f"{len(offset_bins)} offset bins loaded")

    def _load_gathers(self):
        """Load gathers from folder with metadata."""
        path = QFileDialog.getExistingDirectory(
            self, "Open Common Offset Gathers Directory",
            str(Path.home() / "SeismicData")
        )
        if not path:
            return

        path = Path(path)
        metadata_path = path / 'gather_metadata.json'

        if not metadata_path.exists():
            self.statusBar().showMessage("No gather_metadata.json found in folder")
            return

        try:
            self.statusBar().showMessage("Loading gathers...")
            QApplication.processEvents()

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Load all offset bins
            self.offset_bins = []
            data_array_name = metadata.get('data_array', 'migrated_stack.zarr')

            for offset_info in metadata['offsets']:
                bin_path = path / offset_info['bin_name'] / data_array_name
                z = zarr.open(str(bin_path), mode='r')
                self.offset_bins.append(z)

            # Set coordinates
            self.offset_values = np.array([o['offset_m'] for o in metadata['offsets']])

            dims = metadata['dimensions']
            self.t_coords = np.linspace(
                metadata['coordinates']['time']['min'],
                metadata['coordinates']['time']['max'],
                dims['n_time']
            )
            self.dt_ms = self.t_coords[1] - self.t_coords[0] if len(self.t_coords) > 1 else 2.0

            n_off = len(self.offset_bins)
            self.gathers_label.setText(f"{path.name}\n{n_off} offsets")
            self.statusBar().showMessage(f"Loaded {n_off} offset bins")

            # Update display at center position
            self.il_center = dims['n_inline'] // 2
            self.xl_center = dims['n_crossline'] // 2
            self._compute_super_gather()
            self._update_display()

        except Exception as e:
            self.statusBar().showMessage(f"Error loading gathers: {e}")
            import traceback
            traceback.print_exc()

    def set_position(self, il: int, xl: int):
        """Set position and update display."""
        self.il_center = il
        self.xl_center = xl
        self.pos_label.setText(f"IL={il}, XL={xl}")
        self._compute_super_gather()
        self._update_velocity_function()
        self._update_display()
        self._update_velocity_display()

    def _on_param_changed(self):
        """Handle parameter changes."""
        self.il_half = self.il_spin.value() // 2
        self.xl_half = self.xl_spin.value() // 2
        self.offset_bin_size = self.offset_spin.value()
        self._compute_super_gather()
        self._update_display()

    def _on_mute_changed(self):
        """Handle mute parameter changes."""
        self.top_mute_enabled = self.top_mute_check.isChecked()
        self.bottom_mute_enabled = self.bottom_mute_check.isChecked()
        self.v_top = self.vtop_spin.value()
        self.v_bottom = self.vbot_spin.value()
        self._update_display()

    def _on_processing_changed(self):
        """Handle processing parameter changes."""
        self.apply_inv_nmo = self.inv_nmo_check.isChecked()
        self.nmo_velocity = self.vnmo_spin.value()
        self.apply_bandpass = self.bp_check.isChecked()
        self.f_low = self.f_low_spin.value()
        self.f_high = self.f_high_spin.value()
        self.apply_agc_flag = self.agc_check.isChecked()
        self.agc_window = self.agc_spin.value()
        self._update_display()

    def _compute_super_gather(self):
        """Compute super gather at current position."""
        if len(self.offset_bins) == 0:
            return

        self.statusBar().showMessage("Computing super gather...")
        QApplication.processEvents()

        self.super_gather, self.super_offsets = create_super_gather(
            self.offset_bins, self.il_center, self.xl_center,
            self.il_half, self.xl_half, self.offset_values,
            self.offset_bin_size
        )

        self.statusBar().showMessage(f"Super gather: {len(self.super_offsets)} offset bins")

    def _compute_semblance(self):
        """Compute semblance on processed gather."""
        if self.super_gather is None:
            return

        self.statusBar().showMessage("Computing semblance...")
        QApplication.processEvents()

        # Use processed gather if available, else apply processing
        if self.processed_gather is not None:
            gather = self.processed_gather
        else:
            gather = self.super_gather
            if self.top_mute_enabled or self.bottom_mute_enabled:
                v_top = self.v_top if self.top_mute_enabled else None
                v_bottom = self.v_bottom if self.bottom_mute_enabled else None
                gather = apply_velocity_mute(
                    gather, self.super_offsets, self.t_coords,
                    v_top, v_bottom
                )

        self.v_min = self.vmin_spin.value()
        self.v_max = self.vmax_spin.value()

        self.semblance, self.velocities = compute_semblance_fast(
            gather, self.super_offsets, self.t_coords,
            self.v_min, self.v_max, self.v_step,
            self.semblance_window_samples
        )

        self._update_semblance_display()
        self.statusBar().showMessage(
            f"Semblance computed (v={self.v_min}-{self.v_max}, step={self.v_step}, win={self.semblance_window_samples})"
        )

    def _open_semblance_settings(self):
        """Open the semblance settings dialog."""
        current_settings = {
            'v_min': self.v_min,
            'v_max': self.v_max,
            'v_step': self.v_step,
            'window_samples': self.semblance_window_samples,
            # Display settings
            'gather_colormap': self.gather_colormap,
            'gather_clip': self.gather_clip,
            'gather_gain': self.gather_gain,
            'semblance_colormap': self.semblance_colormap,
            'semblance_clip': self.semblance_clip,
            'velocity_colormap': self.velocity_colormap,
            'velocity_clip': self.velocity_clip,
            'velocity_use_minmax': self.velocity_use_minmax,
            'velocity_vmin': self.velocity_vmin,
            'velocity_vmax': self.velocity_vmax,
        }

        dialog = SemblanceSettingsDialog(self, current_settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_settings = dialog.get_settings()

            # Update semblance parameters
            self.v_min = new_settings['v_min']
            self.v_max = new_settings['v_max']
            self.v_step = new_settings['v_step']
            self.semblance_window_samples = new_settings['window_samples']

            # Update display settings
            self.gather_colormap = new_settings['gather_colormap']
            self.gather_clip = new_settings['gather_clip']
            self.gather_gain = new_settings['gather_gain']
            self.semblance_colormap = new_settings['semblance_colormap']
            self.semblance_clip = new_settings['semblance_clip']
            self.velocity_colormap = new_settings['velocity_colormap']
            self.velocity_clip = new_settings['velocity_clip']
            self.velocity_use_minmax = new_settings['velocity_use_minmax']
            self.velocity_vmin = new_settings['velocity_vmin']
            self.velocity_vmax = new_settings['velocity_vmax']

            # Update the spinboxes in the UI to reflect new values
            self.vmin_spin.setValue(int(self.v_min))
            self.vmax_spin.setValue(int(self.v_max))
            self.gain_spin.setValue(self.gather_gain)

            # Apply display settings to canvases
            self._apply_display_settings()

            # Refresh displays
            self._update_display()
            self._update_semblance_display()
            self._update_velocity_display()

            self.statusBar().showMessage("Settings updated")

    def _apply_display_settings(self):
        """Apply current display settings to all canvases."""
        # Gather canvas
        self.gather_canvas.set_palette(self.gather_colormap)
        self.gather_canvas.set_clip_percentile(self.gather_clip)
        self.gather_canvas.set_gain(self.gather_gain)

        # Semblance canvas
        self.semblance_canvas.set_palette(self.semblance_colormap)
        self.semblance_canvas.set_clip_percentile(self.semblance_clip)

        # Velocity canvas
        self.velocity_canvas.set_palette(self.velocity_colormap)
        if self.velocity_use_minmax:
            self.velocity_canvas.set_value_range(self.velocity_vmin, self.velocity_vmax)
        else:
            self.velocity_canvas.set_clip_percentile(self.velocity_clip)

    def _update_display(self):
        """Update the gather display with full processing chain."""
        if self.super_gather is None:
            return

        gather = self.super_gather.copy()
        dt_ms = self.t_coords[1] - self.t_coords[0] if len(self.t_coords) > 1 else 2.0

        # 1. Apply inverse NMO if enabled (before other processing)
        if self.apply_inv_nmo:
            if self.use_vel_model and self.vel_function is not None:
                # Use loaded velocity model
                gather = apply_nmo_with_velocity_model(
                    gather, self.super_offsets, self.t_coords,
                    self.vel_function, inverse=True
                )
            elif self.nmo_velocity > 0:
                # Use constant velocity
                gather = apply_nmo_correction(
                    gather, self.super_offsets, self.t_coords,
                    self.nmo_velocity, inverse=True
                )

        # 2. Apply bandpass filter if enabled
        if self.apply_bandpass:
            gather = apply_bandpass_filter(
                gather, dt_ms, self.f_low, self.f_high
            )

        # 3. Apply AGC if enabled
        if self.apply_agc_flag:
            gather = apply_agc(gather, self.agc_window, dt_ms)

        # 4. Apply velocity mute if enabled (after other processing)
        if self.top_mute_enabled or self.bottom_mute_enabled:
            v_top = self.v_top if self.top_mute_enabled else None
            v_bottom = self.v_bottom if self.bottom_mute_enabled else None
            gather = apply_velocity_mute(
                gather, self.super_offsets, self.t_coords,
                v_top, v_bottom
            )

        # Store processed gather for semblance
        self.processed_gather = gather

        # Display gather (offset x time)
        data = gather.T  # (time, offset)

        off_min = self.super_offsets[0] if len(self.super_offsets) > 0 else 0
        off_max = self.super_offsets[-1] if len(self.super_offsets) > 0 else 100
        t_min = self.t_coords[0] if len(self.t_coords) > 0 else 0
        t_max = self.t_coords[-1] if len(self.t_coords) > 0 else 2000

        x_axis = AxisConfig("Offset", off_min, off_max, "m")
        y_axis = AxisConfig("Time", t_min, t_max, "ms")

        self.gather_canvas.set_palette(self.gather_colormap)
        self.gather_canvas.set_clip_percentile(self.gather_clip)
        self.gather_canvas.set_gain(self.gain_spin.value())
        self.gather_canvas.set_data(data, "gather", 0, 0, x_axis, y_axis, 0)
        self.gather_canvas.slice_direction = f"Super Gather IL={self.il_center}, XL={self.xl_center}"

    def _update_semblance_display(self):
        """Update the semblance display."""
        if self.semblance is None:
            return

        # Display semblance (velocity x time)
        data = self.semblance.T  # (time, velocity)

        v_min = self.velocities[0] if len(self.velocities) > 0 else 1500
        v_max = self.velocities[-1] if len(self.velocities) > 0 else 5000
        t_min = self.t_coords[0] if len(self.t_coords) > 0 else 0
        t_max = self.t_coords[-1] if len(self.t_coords) > 0 else 2000

        x_axis = AxisConfig("Velocity", v_min, v_max, "m/s")
        y_axis = AxisConfig("Time", t_min, t_max, "ms")

        self.semblance_canvas.set_palette(self.semblance_colormap)
        self.semblance_canvas.set_clip_percentile(self.semblance_clip)
        self.semblance_canvas.set_data(data, "semblance", 0, 0, x_axis, y_axis, 0)
        self.semblance_canvas.slice_direction = "Semblance"

    def _load_velocity_model(self):
        """Load velocity model from file."""
        path = QFileDialog.getExistingDirectory(
            self, "Open Velocity Model (zarr directory)",
            str(Path.home() / "SeismicData")
        )
        if not path:
            # Try file dialog for parquet
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Velocity Model",
                str(Path.home() / "SeismicData"),
                "Velocity Files (*.parquet *.zarr);;All Files (*)"
            )

        if path:
            self.statusBar().showMessage(f"Loading velocity from {path}...")
            QApplication.processEvents()

            self.vel_model, self.vel_metadata = load_velocity_model(path)

            if self.vel_model is not None:
                shape_str = str(self.vel_model.shape)
                self.vel_file_label.setText(f"{Path(path).name}\n{shape_str}")
                self.use_vel_check.setEnabled(True)

                # Extract velocity at current position
                self._update_velocity_function()
                self._update_velocity_display()

                self.statusBar().showMessage(f"Velocity loaded: {shape_str}")
            else:
                self.vel_file_label.setText("Load failed")
                self.statusBar().showMessage("Failed to load velocity")

    def _on_vel_model_changed(self):
        """Handle velocity model checkbox change."""
        self.use_vel_model = self.use_vel_check.isChecked()
        self._update_display()

    def _update_velocity_function(self):
        """Extract 1D velocity function at current IL/XL position."""
        if self.vel_model is None:
            self.vel_function = None
            return

        self.vel_function = extract_velocity_function(
            self.vel_model, self.il_center, self.xl_center, self.t_coords
        )

    def _update_velocity_display(self):
        """Update the velocity model display."""
        if self.vel_model is None:
            return

        t_min = self.t_coords[0] if len(self.t_coords) > 0 else 0
        t_max = self.t_coords[-1] if len(self.t_coords) > 0 else 2000

        if self.vel_model.ndim == 1:
            # 1D velocity - display as single column
            vel_func = self.vel_model
            v_min = vel_func.min()
            v_max = vel_func.max()

            # Create 2D array for display (time x 1)
            data = vel_func[:, np.newaxis]

            x_axis = AxisConfig("V", v_min, v_max, "m/s")
            y_axis = AxisConfig("Time", t_min, t_max, "ms")

        elif self.vel_model.ndim == 2:
            # 2D velocity (IL, time) - display slice at current IL
            il_idx = min(self.il_center, self.vel_model.shape[0] - 1)
            vel_slice = self.vel_model[il_idx, :]
            v_min = self.vel_model.min()
            v_max = self.vel_model.max()

            # Create 2D for display
            data = vel_slice[:, np.newaxis]

            x_axis = AxisConfig("V", v_min, v_max, "m/s")
            y_axis = AxisConfig("Time", t_min, t_max, "ms")

        elif self.vel_model.ndim == 3:
            # 3D velocity (IL, XL, time) - display IL slice
            il_idx = min(self.il_center, self.vel_model.shape[0] - 1)
            vel_slice = self.vel_model[il_idx, :, :].T  # (time, XL)

            xl_min = 0
            xl_max = self.vel_model.shape[1] - 1

            x_axis = AxisConfig("XL", xl_min, xl_max)
            y_axis = AxisConfig("Time", t_min, t_max, "ms")

            data = vel_slice

        else:
            return

        self.velocity_canvas.set_palette(self.velocity_colormap)
        if self.velocity_use_minmax:
            self.velocity_canvas.set_value_range(self.velocity_vmin, self.velocity_vmax)
        else:
            self.velocity_canvas.set_clip_percentile(self.velocity_clip)
        self.velocity_canvas.set_data(data, "velocity", 0, 0, x_axis, y_axis, 0)
        self.velocity_canvas.slice_direction = f"Velocity @ IL={self.il_center}"

        # Draw crosshair at current XL position if 3D
        if self.vel_model.ndim == 3:
            xl_idx = min(self.xl_center, self.vel_model.shape[1] - 1)
            self.velocity_canvas.set_crosshair_position(xl_idx, t_max / 2)


# =============================================================================
# Main Window
# =============================================================================

class SeismicViewer(QMainWindow):
    """Main seismic viewer window with dual volume and Cube+CIG support."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seismic Viewer")
        self.setMinimumSize(1200, 800)

        self.view_mode = "single"  # "single", "dual", "cube_cig"
        self.sync_enabled = True

        # For Cube+CIG mode: track selected position
        self.cig_il_idx = 0
        self.cig_xl_idx = 0

        # Velocity Analysis window
        self.va_window = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(250)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(5)

        # Mode selection
        mode_group = QGroupBox("View Mode")
        mode_layout = QVBoxLayout(mode_group)

        mode_sel_layout = QHBoxLayout()
        mode_sel_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single Volume", "Dual Volume", "Velocity Analysis"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_sel_layout.addWidget(self.mode_combo)
        mode_layout.addLayout(mode_sel_layout)

        self.sync_check = QCheckBox("Synchronize Views")
        self.sync_check.setChecked(True)
        self.sync_check.toggled.connect(self._toggle_sync)
        mode_layout.addWidget(self.sync_check)

        left_layout.addWidget(mode_group)

        # Slice selection group
        slice_group = QGroupBox("Slice Selection")
        slice_layout = QVBoxLayout(slice_group)

        # Direction selector
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Inline", "Crossline", "Time Slice"])
        self.direction_combo.currentIndexChanged.connect(self._on_direction_changed)
        dir_layout.addWidget(self.direction_combo)
        slice_layout.addLayout(dir_layout)

        # Slice index
        idx_layout = QHBoxLayout()
        idx_layout.addWidget(QLabel("Index:"))
        self.slice_spin = QSpinBox()
        self.slice_spin.setRange(0, 0)
        self.slice_spin.valueChanged.connect(self._on_slice_changed)
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
        self.prev_btn.clicked.connect(self._prev_slice)
        nav_layout.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._next_slice)
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
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        pal_layout.addWidget(self.palette_combo)
        display_layout.addLayout(pal_layout)

        # Gain
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 100.0)
        self.gain_spin.setValue(1.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        gain_layout.addWidget(self.gain_spin)
        display_layout.addLayout(gain_layout)

        # Gain slider
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(1, 500)
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
        self.clip_spin.valueChanged.connect(self._on_clip_changed)
        clip_layout.addWidget(self.clip_spin)
        display_layout.addLayout(clip_layout)

        left_layout.addWidget(display_group)

        # View group
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)

        view_btn_layout = QHBoxLayout()
        self.fit_btn = QPushButton("Fit (F)")
        self.fit_btn.clicked.connect(self._fit_view)
        view_btn_layout.addWidget(self.fit_btn)
        self.reset_btn = QPushButton("Reset (R)")
        self.reset_btn.clicked.connect(self._reset_view)
        view_btn_layout.addWidget(self.reset_btn)
        view_layout.addLayout(view_btn_layout)

        left_layout.addWidget(view_group)

        # Info group
        info_group = QGroupBox("Cursor Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("X: -  Y: -  Amp: -")
        self.info_label.setStyleSheet("font-family: monospace;")
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_group)

        left_layout.addStretch()

        # Volume panels in splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.volume1 = VolumePanel("Volume 1")
        self.volume1.slice_changed.connect(self._on_volume_slice_changed)
        self.volume1.view_changed.connect(self._on_volume_view_changed)
        self.volume1.canvas.cursor_moved.connect(self._on_cursor_moved)
        self.volume1.canvas.position_selected.connect(self._on_position_selected)
        self.splitter.addWidget(self.volume1)

        self.volume2 = VolumePanel("Volume 2")
        self.volume2.slice_changed.connect(self._on_volume_slice_changed)
        self.volume2.view_changed.connect(self._on_volume_view_changed)
        self.volume2.canvas.cursor_moved.connect(self._on_cursor_moved)
        self.volume2.hide()  # Hidden by default
        self.splitter.addWidget(self.volume2)

        # Gather panel for CIG mode
        self.gather_panel = GatherPanel("CIG Gathers")
        self.gather_panel.canvas.cursor_moved.connect(self._on_cursor_moved)
        self.gather_panel.hide()  # Hidden by default
        self.splitter.addWidget(self.gather_panel)

        # Add to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.splitter, 1)

        # Status bar
        self.statusBar().showMessage("Ready - Ctrl+Wheel to zoom, Wheel to change slice, Drag to pan")

    def _on_mode_changed(self, index: int):
        """Handle view mode change."""
        from PyQt6.QtCore import QTimer

        modes = ["single", "dual", "velocity_analysis"]
        self.view_mode = modes[index]

        # Hide all secondary panels first
        self.volume2.hide()
        self.gather_panel.hide()
        self.volume1.canvas.show_crosshair = False
        self.volume1.canvas.update()

        if self.view_mode == "single":
            self.statusBar().showMessage("Ready - Ctrl+Wheel to zoom, Wheel to change slice, Drag to pan")
        elif self.view_mode == "dual":
            self.volume2.show()
            self.statusBar().showMessage("Dual Mode - Ctrl+Wheel to zoom, Wheel to change slice, Drag to pan")
        elif self.view_mode == "velocity_analysis":
            # Open VA window and show crosshair for position selection
            self._open_velocity_analysis()
            self.statusBar().showMessage(
                "Velocity Analysis Mode: Click on cube to update VA position | Ctrl+Wheel to zoom"
            )

        # Delay size adjustment to after widgets are shown
        QTimer.singleShot(50, self._adjust_splitter_sizes)

    def _adjust_splitter_sizes(self):
        """Adjust splitter sizes based on current mode."""
        w = self.splitter.width()
        if self.view_mode == "single":
            self.splitter.setSizes([w, 0, 0])
        elif self.view_mode == "dual":
            self.splitter.setSizes([w // 2, w // 2, 0])
        elif self.view_mode == "velocity_analysis":
            # Full width for cube, VA window is separate
            self.splitter.setSizes([w, 0, 0])

    def _on_position_selected(self, x: float, y: float):
        """Handle position selection on cube (for Velocity Analysis mode)."""
        if self.view_mode != "velocity_analysis":
            return

        if self.volume1.cube is None:
            return

        # Get current slice direction to determine which coordinates were clicked
        direction = self.volume1.current_direction

        if direction == "inline":
            # Viewing inline slice: X-axis=XL, Y-axis=Time
            # Current inline is slice_value, clicked crossline is x
            il_idx = int(round(self.volume1.slice_value))
            xl_idx = int(round(x))
            self.volume1.canvas.set_crosshair_position(x, y)
        elif direction == "crossline":
            # Viewing crossline slice: X-axis=IL, Y-axis=Time
            # Clicked inline is x, current crossline is slice_value
            il_idx = int(round(x))
            xl_idx = int(round(self.volume1.slice_value))
            self.volume1.canvas.set_crosshair_position(x, y)
        else:  # time slice
            # Viewing time slice: X-axis=IL, Y-axis=XL
            il_idx = int(round(x))
            xl_idx = int(round(y))
            self.volume1.canvas.set_crosshair_position(x, y)

        # Update VA window position
        if self.va_window is not None and self.va_window.isVisible():
            self.va_window.set_position(il_idx, xl_idx)
            self.statusBar().showMessage(
                f"VA Position: IL={il_idx}, XL={xl_idx}"
            )
        else:
            self.statusBar().showMessage(f"Clicked IL={il_idx}, XL={xl_idx} - VA window not open")

    def _toggle_sync(self, enabled: bool):
        """Toggle synchronization."""
        self.sync_enabled = enabled

    def _open_velocity_analysis(self):
        """Open Velocity Analysis window."""
        if self.va_window is None:
            self.va_window = VelocityAnalysisWindow(self)

        # Pass gather data if available from gather panel
        if self.gather_panel.gather_type == "common_offset_folder" and len(self.gather_panel.offset_bins) > 0:
            self.va_window.set_gather_data(
                self.gather_panel.offset_bins,
                self.gather_panel.offset_coords,
                self.gather_panel.t_coords
            )
            # Set initial position from cube center if available
            if self.volume1.cube is not None:
                il_center = self.volume1.cube_shape[0] // 2
                xl_center = self.volume1.cube_shape[1] // 2
                self.va_window.set_position(il_center, xl_center)
            else:
                self.va_window.set_position(
                    self.gather_panel.current_il_idx,
                    self.gather_panel.current_xl_idx
                )
        else:
            self.statusBar().showMessage("Load gathers in VA window or use gather panel first")

        self.va_window.show()
        self.va_window.raise_()

    def _on_direction_changed(self, index: int):
        """Handle direction change."""
        directions = ["inline", "crossline", "time"]
        direction = directions[index]

        self.volume1.set_direction(direction)
        if self.view_mode == "dual":
            self.volume2.set_direction(direction)

        # Update max slice
        if self.volume1.cube is not None:
            if index == 0:
                max_idx = self.volume1.cube_shape[0] - 1
            elif index == 1:
                max_idx = self.volume1.cube_shape[1] - 1
            else:
                max_idx = self.volume1.cube_shape[2] - 1

            self.slice_spin.setRange(0, max_idx)
            self.slice_slider.setRange(0, max_idx)
            self.slice_spin.setValue(max_idx // 2)

    def _on_slice_changed(self, index: int):
        """Handle slice change from controls."""
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(index)
        self.slice_slider.blockSignals(False)

        self.volume1.set_slice_index(index)
        if self.view_mode == "dual" and self.sync_enabled:
            self.volume2.set_slice_index(index)

    def _on_volume_slice_changed(self, index: int):
        """Handle slice change from volume panel."""
        step = self.step_spin.value()
        current = self.slice_spin.value()

        if index > self.volume1.current_index:
            new_idx = min(current + step, self.slice_spin.maximum())
        else:
            new_idx = max(current - step, 0)

        self.slice_spin.setValue(new_idx)

    def _on_volume_view_changed(self, view: ViewState):
        """Handle view change from volume panel."""
        if self.sync_enabled and self.view_mode == "dual":
            sender = self.sender()
            if sender == self.volume1:
                self.volume2.set_view(view)
            else:
                self.volume1.set_view(view)

    def _on_cursor_moved(self, x: float, y: float, amp: float):
        """Handle cursor movement."""
        self.info_label.setText(f"X: {x:.1f}  Y: {y:.1f}  Amp: {amp:.4f}")

    def _prev_slice(self):
        """Go to previous slice."""
        step = self.step_spin.value()
        new_idx = max(0, self.slice_spin.value() - step)
        self.slice_spin.setValue(new_idx)

    def _next_slice(self):
        """Go to next slice."""
        step = self.step_spin.value()
        new_idx = min(self.slice_spin.maximum(), self.slice_spin.value() + step)
        self.slice_spin.setValue(new_idx)

    def _on_palette_changed(self, name: str):
        """Handle palette change."""
        self.volume1.set_palette(name)
        if self.view_mode == "dual":
            self.volume2.set_palette(name)
        elif self.view_mode == "cube_cig":
            self.gather_panel.set_palette(name)

    def _on_gain_changed(self, value: float):
        """Handle gain change."""
        self.gain_slider.blockSignals(True)
        self.gain_slider.setValue(int(value * 10))
        self.gain_slider.blockSignals(False)

        self.volume1.set_gain(value)
        if self.view_mode == "dual" and self.sync_enabled:
            self.volume2.set_gain(value)
        elif self.view_mode == "cube_cig" and self.sync_enabled:
            self.gather_panel.set_gain(value)

    def _on_clip_changed(self, value: float):
        """Handle clip percentile change."""
        self.volume1.set_clip_percentile(value)
        if self.view_mode == "dual" and self.sync_enabled:
            self.volume2.set_clip_percentile(value)
        elif self.view_mode == "cube_cig" and self.sync_enabled:
            self.gather_panel.set_clip_percentile(value)

    def _fit_view(self):
        """Fit view to data."""
        self.volume1.reset_view()
        if self.view_mode == "dual":
            self.volume2.reset_view()
        elif self.view_mode == "cube_cig":
            self.gather_panel.reset_view()

    def _reset_view(self):
        """Reset view."""
        self.volume1.reset_view()
        if self.view_mode == "dual":
            self.volume2.reset_view()
        elif self.view_mode == "cube_cig":
            self.gather_panel.reset_view()

    def load_zarr(self, path: str, volume: int = 1):
        """Load zarr data into specified volume."""
        if volume == 1:
            self.volume1.load_zarr(path)
            # Update controls after loading
            if self.volume1.cube is not None:
                direction_idx = self.direction_combo.currentIndex()
                if direction_idx == 0:
                    max_idx = self.volume1.cube_shape[0] - 1
                elif direction_idx == 1:
                    max_idx = self.volume1.cube_shape[1] - 1
                else:
                    max_idx = self.volume1.cube_shape[2] - 1

                self.slice_spin.setRange(0, max_idx)
                self.slice_slider.setRange(0, max_idx)
                self.slice_spin.setValue(max_idx // 2)
        else:
            self.volume2.load_zarr(path)

    def load_gathers(self, path: str, gather_type: str = "common_offset"):
        """Load gather data for CIG mode."""
        self.gather_panel.load_zarr(path)
        if gather_type == "common_offset":
            self.gather_panel.type_combo.setCurrentIndex(1)
        else:
            self.gather_panel.type_combo.setCurrentIndex(0)


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Seismic Viewer with Cube+CIG support')
    parser.add_argument('cube', nargs='?', help='Path to seismic cube (zarr)')
    parser.add_argument('second', nargs='?', help='Path to second volume or gathers (zarr)')
    parser.add_argument('--cig', action='store_true',
                       help='Enable Cube+CIG mode (second file is gathers)')
    parser.add_argument('--gather-type', choices=['cig', 'common_offset'], default='common_offset',
                       help='Type of gather organization (default: common_offset)')
    args = parser.parse_args()

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

    # Load files from command line
    if args.cube:
        viewer.load_zarr(args.cube, volume=1)

    if args.second:
        if args.cig:
            # Cube+CIG mode
            viewer.mode_combo.setCurrentIndex(2)  # "Cube + CIG"
            viewer.load_gathers(args.second, args.gather_type)
        else:
            # Dual volume mode
            viewer.mode_combo.setCurrentIndex(1)  # "Dual Volume"
            viewer.load_zarr(args.second, volume=2)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
