"""Seismic canvas widget for rendering seismic data with axes."""

from typing import Optional, List, Tuple
import time
import numpy as np

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QImage, QPainter, QColor, QPen, QBrush, QWheelEvent, QMouseEvent,
    QKeyEvent, QPaintEvent, QFont, QFontMetrics
)

from ..core import AxisConfig, ViewState, PALETTES


class SeismicCanvas(QWidget):
    """Widget for rendering seismic data with proper axes."""

    # Signals
    slice_changed = pyqtSignal(int)
    view_changed = pyqtSignal(object)  # Emits ViewState
    cursor_moved = pyqtSignal(float, float, float)  # x, y, amplitude
    position_selected = pyqtSignal(float, float)  # x, y coordinates when clicked

    # Edit mode signals
    pick_added = pyqtSignal(float, float)        # time_ms, velocity when pick added
    pick_moved = pyqtSignal(int, float, float)   # index, new_time, new_velocity
    pick_removed = pyqtSignal(int)               # index of removed pick
    pick_drag_started = pyqtSignal(int)          # index of pick being dragged
    pick_drag_ended = pyqtSignal(int)            # index of pick after drag
    pick_drag_update = pyqtSignal(int, float, float)  # index, time, velocity during drag
    preview_velocity_changed = pyqtSignal(float, float)  # time, velocity for live preview
    preview_ended = pyqtSignal()                 # emitted when mouse leaves canvas

    # Margins for axis labels
    LEFT_MARGIN = 70
    RIGHT_MARGIN = 20
    RIGHT_MARGIN_COLORBAR = 80  # Extra margin when colorbar is shown
    TOP_MARGIN = 30
    BOTTOM_MARGIN = 50
    COLORBAR_WIDTH = 20
    COLORBAR_PADDING = 10

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

        # Velocity overlay (for semblance display)
        self.velocity_overlay: Optional[np.ndarray] = None  # 1D velocity array
        self.velocity_t_coords: Optional[np.ndarray] = None  # Corresponding time coordinates

        # Font for axis labels
        self.axis_font = QFont("Arial", 9)
        self.title_font = QFont("Arial", 10, QFont.Weight.Bold)

        # Edit mode for velocity picking
        self.edit_mode = False
        self.picks: List[Tuple[float, float]] = []  # List of (time, velocity) picks
        self.selected_pick_index: Optional[int] = None
        self.hovered_pick_index: Optional[int] = None
        self.dragging_pick = False
        self.pick_radius = 8  # Radius of pick circles in pixels

        # Preview throttling
        self._last_preview_time = 0.0
        self._preview_interval = 0.1  # 100ms = 10 FPS max for preview updates
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._emit_preview_update)
        self._pending_preview_pos: Optional[Tuple[float, float]] = None

        # Snap to maximum option
        self.snap_to_max = False
        self.snap_radius = 50  # Search radius in pixels for max semblance

        # Colorbar
        self.show_colorbar = False
        self._colorbar_vmin = 0.0
        self._colorbar_vmax = 1.0

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

        # Initialize view to full data extent
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

    def set_velocity_overlay(self, velocity: np.ndarray, t_coords: np.ndarray):
        """Set velocity function to overlay on the canvas (for semblance display)."""
        self.velocity_overlay = velocity
        self.velocity_t_coords = t_coords
        self.update()

    def clear_velocity_overlay(self):
        """Clear the velocity overlay."""
        self.velocity_overlay = None
        self.velocity_t_coords = None
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
        right_margin = self.RIGHT_MARGIN_COLORBAR if self.show_colorbar else self.RIGHT_MARGIN
        return QRectF(
            self.LEFT_MARGIN,
            self.TOP_MARGIN,
            self.width() - self.LEFT_MARGIN - right_margin,
            self.height() - self.TOP_MARGIN - self.BOTTOM_MARGIN
        )

    def set_colorbar(self, show: bool, vmin: float = 0.0, vmax: float = 1.0):
        """Enable/disable colorbar and set value range."""
        self.show_colorbar = show
        self._colorbar_vmin = vmin
        self._colorbar_vmax = vmax
        self.update()

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

        # Draw velocity overlay if present
        if self.velocity_overlay is not None:
            self._draw_velocity_overlay(painter, plot_rect)

        # Draw velocity picks if in edit mode or picks exist
        if self.edit_mode or self.picks:
            self._draw_picks(painter, plot_rect)

        # Draw edit mode indicator
        if self.edit_mode:
            self._draw_edit_mode_indicator(painter)

        # Draw colorbar if enabled
        if self.show_colorbar:
            self._draw_colorbar(painter, plot_rect)

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

    def _draw_velocity_overlay(self, painter: QPainter, plot_rect: QRectF):
        """Draw velocity curve overlay on the canvas."""
        if self.velocity_overlay is None or self.velocity_t_coords is None:
            return

        # Set pen for velocity curve - bright green, thick line
        pen = QPen(QColor(0, 255, 100), 3)
        painter.setPen(pen)

        # Build list of screen points
        points = []
        for i in range(len(self.velocity_overlay)):
            if i >= len(self.velocity_t_coords):
                break

            vel = self.velocity_overlay[i]
            t = self.velocity_t_coords[i]

            # Skip NaN values
            if np.isnan(vel):
                continue

            # Convert to screen coordinates (velocity is X, time is Y)
            pos = self._data_to_screen(vel, t)

            # Check if within plot area
            if (plot_rect.left() <= pos.x() <= plot_rect.right() and
                plot_rect.top() <= pos.y() <= plot_rect.bottom()):
                points.append(pos)

        # Draw line segments between points
        if len(points) > 1:
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                painter.drawLine(int(p1.x()), int(p1.y()), int(p2.x()), int(p2.y()))

            # Draw circles at velocity pick points (every 10th point)
            painter.setPen(QPen(QColor(0, 255, 100), 2))
            painter.setBrush(QColor(0, 255, 100))
            for i, p in enumerate(points):
                if i % 10 == 0:
                    painter.drawEllipse(p, 4, 4)

    def _draw_image(self, painter: QPainter, plot_rect: QRectF):
        """Draw the seismic image within the plot area."""
        if self.image is None or self.data is None:
            return

        h, w = self.data.shape

        # Calculate which portion of the image to display based on view
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
        """Handle mouse press for panning or pick editing."""
        pos = event.position()
        self.last_mouse_pos = pos

        # Check if click is within plot area
        plot_rect = self._get_plot_rect()
        in_plot = (plot_rect.left() <= pos.x() <= plot_rect.right() and
                  plot_rect.top() <= pos.y() <= plot_rect.bottom())

        if not in_plot:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.edit_mode:
                # Check if clicking on existing pick
                pick_idx = self._find_pick_at_screen(pos.x(), pos.y())

                if pick_idx is not None:
                    # Start dragging this pick
                    self.dragging_pick = True
                    self.selected_pick_index = pick_idx
                    self.pick_drag_started.emit(pick_idx)
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    # Add new pick at this location
                    data_pos = self._screen_to_data(pos.x(), pos.y())
                    velocity = data_pos.x()
                    time_ms = data_pos.y()

                    # Snap to maximum if enabled
                    if self.snap_to_max and self.data is not None:
                        velocity = self._find_max_velocity_near(time_ms, velocity)

                    self.pick_added.emit(time_ms, velocity)
            else:
                # Normal panning mode
                self.dragging = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

        elif event.button() == Qt.MouseButton.RightButton and self.edit_mode:
            # Delete pick with right click
            pick_idx = self._find_pick_at_screen(pos.x(), pos.y())
            if pick_idx is not None:
                self.pick_removed.emit(pick_idx)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            last = self.last_mouse_pos
            drag_distance = ((pos.x() - last.x())**2 + (pos.y() - last.y())**2)**0.5

            if self.dragging_pick:
                # End pick drag
                if self.selected_pick_index is not None:
                    self.pick_drag_ended.emit(self.selected_pick_index)
                self.dragging_pick = False
                self._update_cursor_for_mode()

            elif self.dragging:
                # End panning
                self.dragging = False
                self._update_cursor_for_mode()

                # Check if this was a click (not a drag) within plot
                plot_rect = self._get_plot_rect()
                in_plot = (plot_rect.left() <= pos.x() <= plot_rect.right() and
                          plot_rect.top() <= pos.y() <= plot_rect.bottom())

                if drag_distance < 10 and in_plot and not self.edit_mode:
                    data_pos = self._screen_to_data(pos.x(), pos.y())
                    self.position_selected.emit(data_pos.x(), data_pos.y())

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning, picking, and coordinate display."""
        pos = event.position()
        self.mouse_data_pos = self._screen_to_data(pos.x(), pos.y())

        plot_rect = self._get_plot_rect()
        in_plot = (plot_rect.left() <= pos.x() <= plot_rect.right() and
                  plot_rect.top() <= pos.y() <= plot_rect.bottom())

        if self.dragging_pick and self.selected_pick_index is not None and in_plot:
            # Move the selected pick
            data_pos = self._screen_to_data(pos.x(), pos.y())
            velocity = data_pos.x()
            time_ms = data_pos.y()

            # Snap to maximum if enabled
            if self.snap_to_max and self.data is not None:
                velocity = self._find_max_velocity_near(time_ms, velocity)

            self.pick_moved.emit(self.selected_pick_index, time_ms, velocity)
            # Emit drag update for live stack preview (throttled in receiver)
            self._schedule_preview_update(time_ms, velocity)
            self.pick_drag_update.emit(self.selected_pick_index, time_ms, velocity)

        elif self.dragging:
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

        elif self.edit_mode and in_plot:
            # Update hovered pick
            old_hovered = self.hovered_pick_index
            self.hovered_pick_index = self._find_pick_at_screen(pos.x(), pos.y())

            if self.hovered_pick_index != old_hovered:
                self._update_cursor_for_mode()

            # Note: preview_velocity_changed is now only emitted during pick dragging
            # (see dragging_pick block above) for live stack updates

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
        self.preview_ended.emit()

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

        # Edit mode shortcuts
        elif key == Qt.Key.Key_Delete and self.edit_mode:
            # Delete selected pick
            if self.selected_pick_index is not None:
                self.pick_removed.emit(self.selected_pick_index)
                self.selected_pick_index = None
                self.update()

        elif key == Qt.Key.Key_Escape:
            # Clear selection
            self.selected_pick_index = None
            self.update()

    # ==================== Edit Mode Methods ====================

    def set_edit_mode(self, enabled: bool):
        """Enable or disable edit mode for velocity picking."""
        self.edit_mode = enabled
        self.selected_pick_index = None
        self.hovered_pick_index = None
        self.dragging_pick = False
        self._update_cursor_for_mode()
        self.update()

    def set_picks(self, picks: List[Tuple[float, float]]):
        """Set the velocity picks to display. Each pick is (time_ms, velocity)."""
        self.picks = list(picks)
        self.update()

    def clear_picks(self):
        """Clear all picks."""
        self.picks = []
        self.selected_pick_index = None
        self.hovered_pick_index = None
        self.update()

    def set_snap_to_max(self, enabled: bool):
        """Enable/disable snap to maximum semblance."""
        self.snap_to_max = enabled

    def _update_cursor_for_mode(self):
        """Update cursor based on current mode and hover state."""
        if self.edit_mode:
            if self.hovered_pick_index is not None:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def _find_pick_at_screen(self, screen_x: float, screen_y: float) -> Optional[int]:
        """Find index of pick at given screen coordinates, or None."""
        if not self.picks:
            return None

        for i, (time_ms, velocity) in enumerate(self.picks):
            # Convert pick to screen coordinates (velocity=X, time=Y)
            pos = self._data_to_screen(velocity, time_ms)

            # Check distance
            dist = np.sqrt((screen_x - pos.x())**2 + (screen_y - pos.y())**2)
            if dist <= self.pick_radius + 2:  # Small tolerance
                return i

        return None

    def _find_max_velocity_near(self, time_ms: float, velocity: float) -> float:
        """Find velocity with maximum semblance near the given position."""
        if self.data is None:
            return velocity

        h, w = self.data.shape

        # Convert to data indices
        x_scale = w / (self.data_x_max - self.data_x_min) if self.data_x_max > self.data_x_min else 1
        y_scale = h / (self.data_y_max - self.data_y_min) if self.data_y_max > self.data_y_min else 1

        ix = int((velocity - self.data_x_min) * x_scale)
        iy = int((time_ms - self.data_y_min) * y_scale)

        # Search radius in data indices
        search_x = int(self.snap_radius / (self._get_plot_rect().width() / w)) if w > 0 else 5
        search_y = int(self.snap_radius / (self._get_plot_rect().height() / h)) if h > 0 else 5

        # Keep time fixed, search for max velocity
        best_ix = ix
        best_val = -np.inf

        for dx in range(-search_x, search_x + 1):
            test_ix = ix + dx
            if 0 <= test_ix < w and 0 <= iy < h:
                val = self.data[iy, test_ix]
                if val > best_val:
                    best_val = val
                    best_ix = test_ix

        # Convert back to velocity
        return self.data_x_min + best_ix / x_scale

    def _schedule_preview_update(self, time_ms: float, velocity: float):
        """Schedule a throttled preview update."""
        self._pending_preview_pos = (time_ms, velocity)

        current_time = time.time()
        if current_time - self._last_preview_time >= self._preview_interval:
            # Enough time has passed, emit immediately
            self._emit_preview_update()
        elif not self._preview_timer.isActive():
            # Schedule for later
            remaining = int((self._preview_interval - (current_time - self._last_preview_time)) * 1000)
            self._preview_timer.start(max(10, remaining))

    def _emit_preview_update(self):
        """Emit the preview velocity update."""
        if self._pending_preview_pos is not None:
            time_ms, velocity = self._pending_preview_pos
            self._pending_preview_pos = None
            self._last_preview_time = time.time()
            self.preview_velocity_changed.emit(time_ms, velocity)

    def _draw_picks(self, painter: QPainter, plot_rect: QRectF):
        """Draw velocity picks on the canvas."""
        if not self.picks:
            return

        for i, (time_ms, velocity) in enumerate(self.picks):
            # Convert to screen coordinates (velocity=X, time=Y)
            pos = self._data_to_screen(velocity, time_ms)

            # Check if within plot area
            if not (plot_rect.left() <= pos.x() <= plot_rect.right() and
                   plot_rect.top() <= pos.y() <= plot_rect.bottom()):
                continue

            # Determine pick appearance
            if i == self.selected_pick_index:
                # Selected pick
                pen = QPen(QColor(255, 255, 0), 3)  # Yellow, thick
                brush = QBrush(QColor(255, 255, 0, 180))
                radius = self.pick_radius + 2
            elif i == self.hovered_pick_index:
                # Hovered pick
                pen = QPen(QColor(255, 200, 0), 2)  # Orange
                brush = QBrush(QColor(255, 200, 0, 150))
                radius = self.pick_radius + 1
            else:
                # Normal pick
                pen = QPen(QColor(255, 100, 100), 2)  # Light red
                brush = QBrush(QColor(255, 100, 100, 120))
                radius = self.pick_radius

            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawEllipse(pos, radius, radius)

        # Draw connecting line through picks (sorted by time)
        if len(self.picks) >= 2:
            sorted_picks = sorted(self.picks, key=lambda p: p[0])  # Sort by time

            pen = QPen(QColor(255, 100, 100), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            for i in range(len(sorted_picks) - 1):
                t1, v1 = sorted_picks[i]
                t2, v2 = sorted_picks[i + 1]

                pos1 = self._data_to_screen(v1, t1)
                pos2 = self._data_to_screen(v2, t2)

                # Only draw if both points are visible
                if (plot_rect.left() <= pos1.x() <= plot_rect.right() and
                    plot_rect.top() <= pos1.y() <= plot_rect.bottom() and
                    plot_rect.left() <= pos2.x() <= plot_rect.right() and
                    plot_rect.top() <= pos2.y() <= plot_rect.bottom()):
                    painter.drawLine(pos1, pos2)

    def _draw_edit_mode_indicator(self, painter: QPainter):
        """Draw visual indicator that edit mode is active."""
        # Draw a colored border around the widget
        pen = QPen(QColor(255, 100, 100), 3)  # Red border
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        rect = self.rect()
        painter.drawRect(rect.adjusted(1, 1, -2, -2))

        # Draw "EDIT MODE" label in corner
        painter.setPen(QColor(255, 100, 100))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))

        label = "EDIT MODE"
        if self.snap_to_max:
            label += " [SNAP]"

        painter.drawText(self.LEFT_MARGIN + 5, 20, label)

    def _draw_colorbar(self, painter: QPainter, plot_rect: QRectF):
        """Draw colorbar with value labels."""
        # Colorbar position (right of plot)
        cb_left = plot_rect.right() + self.COLORBAR_PADDING
        cb_top = plot_rect.top()
        cb_height = plot_rect.height()

        # Draw colorbar gradient
        n_steps = int(cb_height)
        for i in range(n_steps):
            # Map i to palette index (0 at bottom = low values, n_steps at top = high values)
            # We want low values at bottom, high at top
            t = 1.0 - (i / n_steps)  # Reverse so top is high
            palette_idx = int(t * 255)
            palette_idx = max(0, min(255, palette_idx))

            # Get color from palette (use center of palette for positive data)
            # For positive-only data, we use the upper half of palette (0.5 to 1.0 range)
            color_idx = 128 + palette_idx // 2  # Map to upper half
            color_idx = max(0, min(255, color_idx))

            r, g, b = self.palette[color_idx]
            painter.setPen(QPen(QColor(r, g, b)))
            painter.drawLine(
                int(cb_left), int(cb_top + i),
                int(cb_left + self.COLORBAR_WIDTH), int(cb_top + i)
            )

        # Draw border
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawRect(int(cb_left), int(cb_top),
                        self.COLORBAR_WIDTH, int(cb_height))

        # Draw value labels
        painter.setFont(self.axis_font)
        painter.setPen(QColor(200, 200, 200))
        fm = QFontMetrics(self.axis_font)

        # Top label (max)
        max_label = f"{self._colorbar_vmax:.2f}"
        painter.drawText(int(cb_left), int(cb_top - 3), max_label)

        # Bottom label (min)
        min_label = f"{self._colorbar_vmin:.2f}"
        painter.drawText(int(cb_left), int(cb_top + cb_height + fm.height()), min_label)

        # Middle label
        mid_val = (self._colorbar_vmin + self._colorbar_vmax) / 2
        mid_label = f"{mid_val:.2f}"
        painter.drawText(int(cb_left), int(cb_top + cb_height / 2 + fm.height() / 3), mid_label)
