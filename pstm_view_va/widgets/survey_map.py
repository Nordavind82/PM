"""Survey map widget showing fold and velocity grid locations."""

from typing import Optional, List, Tuple
import numpy as np

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QImage, QPainter, QColor, QPen, QBrush, QWheelEvent, QMouseEvent,
    QPaintEvent, QFont, QFontMetrics
)

from ..core import PALETTES


class SurveyMapWidget(QWidget):
    """Widget for displaying survey fold map and velocity grid locations."""

    # Signal emitted when user clicks on a location
    location_selected = pyqtSignal(int, int)  # il, xl

    # Margins
    LEFT_MARGIN = 50
    RIGHT_MARGIN = 20
    TOP_MARGIN = 30
    BOTTOM_MARGIN = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        # Fold map data
        self.fold_map: Optional[np.ndarray] = None
        self.fold_image: Optional[QImage] = None
        self._fold_image_data: Optional[np.ndarray] = None

        # Coordinate ranges
        self.il_min = 0
        self.il_max = 100
        self.xl_min = 0
        self.xl_max = 100

        # Velocity grid locations (list of (il, xl) tuples)
        self.grid_locations: List[Tuple[int, int]] = []

        # Current position marker
        self.current_il: Optional[int] = None
        self.current_xl: Optional[int] = None

        # Visited/edited locations
        self.visited_locations: set = set()

        # Display settings
        self.palette = PALETTES.get("Viridis", PALETTES["Gray"])
        self.show_grid = True
        self.show_current = True

        # Fonts
        self.axis_font = QFont("Arial", 9)
        self.title_font = QFont("Arial", 10, QFont.Weight.Bold)

        # Mouse position
        self.mouse_il = 0
        self.mouse_xl = 0

    def set_fold_map(self, fold: np.ndarray, il_coords: np.ndarray, xl_coords: np.ndarray):
        """Set the fold map data.

        Args:
            fold: 2D array of fold values (il x xl)
            il_coords: 1D array of inline coordinates
            xl_coords: 1D array of crossline coordinates
        """
        self.fold_map = fold
        self.il_min = int(il_coords[0]) if len(il_coords) > 0 else 0
        self.il_max = int(il_coords[-1]) if len(il_coords) > 0 else 100
        self.xl_min = int(xl_coords[0]) if len(xl_coords) > 0 else 0
        self.xl_max = int(xl_coords[-1]) if len(xl_coords) > 0 else 100

        self._update_fold_image()
        self.update()

    def set_grid_locations(self, locations: List[Tuple[int, int]]):
        """Set velocity grid locations to display."""
        self.grid_locations = list(locations)
        self.update()

    def set_current_position(self, il: int, xl: int):
        """Set current position marker."""
        self.current_il = il
        self.current_xl = xl
        self.update()

    def mark_visited(self, il: int, xl: int):
        """Mark a location as visited/edited."""
        self.visited_locations.add((il, xl))
        self.update()

    def clear_visited(self):
        """Clear all visited markers."""
        self.visited_locations.clear()
        self.update()

    def set_palette(self, name: str):
        """Set color palette by name."""
        if name in PALETTES:
            self.palette = PALETTES[name]
            self._update_fold_image()
            self.update()

    def _update_fold_image(self):
        """Convert fold data to QImage."""
        if self.fold_map is None:
            self.fold_image = None
            return

        fold = self.fold_map.astype(np.float32)

        # Normalize to 0-1
        vmin = np.min(fold)
        vmax = np.max(fold)
        if vmax > vmin:
            fold = (fold - vmin) / (vmax - vmin)
        else:
            fold = np.zeros_like(fold)

        # Map to palette (use upper half for positive data)
        indices = (fold * 127 + 128).astype(np.uint8)

        h, w = indices.shape
        rgb = self.palette[indices.flatten()].reshape(h, w, 3)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255

        self.fold_image = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self._fold_image_data = rgba

    def _get_plot_rect(self) -> QRectF:
        """Get the rectangle for the plot area."""
        return QRectF(
            self.LEFT_MARGIN,
            self.TOP_MARGIN,
            self.width() - self.LEFT_MARGIN - self.RIGHT_MARGIN,
            self.height() - self.TOP_MARGIN - self.BOTTOM_MARGIN
        )

    def _il_xl_to_screen(self, il: float, xl: float) -> QPointF:
        """Convert IL/XL coordinates to screen coordinates."""
        rect = self._get_plot_rect()

        # Normalize
        il_range = self.il_max - self.il_min
        xl_range = self.xl_max - self.xl_min

        if il_range > 0:
            x_norm = (il - self.il_min) / il_range
        else:
            x_norm = 0.5

        if xl_range > 0:
            y_norm = (xl - self.xl_min) / xl_range
        else:
            y_norm = 0.5

        x_screen = rect.left() + x_norm * rect.width()
        y_screen = rect.top() + y_norm * rect.height()

        return QPointF(x_screen, y_screen)

    def _screen_to_il_xl(self, x: float, y: float) -> Tuple[int, int]:
        """Convert screen coordinates to IL/XL."""
        rect = self._get_plot_rect()

        if rect.width() > 0:
            x_norm = (x - rect.left()) / rect.width()
        else:
            x_norm = 0.5

        if rect.height() > 0:
            y_norm = (y - rect.top()) / rect.height()
        else:
            y_norm = 0.5

        il = int(self.il_min + x_norm * (self.il_max - self.il_min))
        xl = int(self.xl_min + y_norm * (self.xl_max - self.xl_min))

        return il, xl

    def paintEvent(self, event: QPaintEvent):
        """Paint the survey map."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        plot_rect = self._get_plot_rect()

        # Draw fold image
        if self.fold_image is not None:
            painter.drawImage(plot_rect.toRect(), self.fold_image)
        else:
            # No data placeholder
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(plot_rect.toRect(), Qt.AlignmentFlag.AlignCenter,
                           "No fold data\n\nLoad gathers to show fold map")

        # Draw border
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(plot_rect.toRect())

        # Draw velocity grid locations
        if self.show_grid and self.grid_locations:
            self._draw_grid_locations(painter, plot_rect)

        # Draw current position
        if self.show_current and self.current_il is not None:
            self._draw_current_position(painter, plot_rect)

        # Draw axes
        self._draw_axes(painter, plot_rect)

        # Draw title
        self._draw_title(painter)

        # Draw cursor info
        self._draw_cursor_info(painter, plot_rect)

    def _draw_grid_locations(self, painter: QPainter, plot_rect: QRectF):
        """Draw velocity grid location markers."""
        for il, xl in self.grid_locations:
            pos = self._il_xl_to_screen(il, xl)

            if not (plot_rect.left() <= pos.x() <= plot_rect.right() and
                   plot_rect.top() <= pos.y() <= plot_rect.bottom()):
                continue

            # Check if visited
            if (il, xl) in self.visited_locations:
                # Visited - green
                painter.setPen(QPen(QColor(0, 200, 0), 2))
                painter.setBrush(QBrush(QColor(0, 200, 0, 100)))
            else:
                # Not visited - white outline
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)

            painter.drawEllipse(pos, 4, 4)

    def _draw_current_position(self, painter: QPainter, plot_rect: QRectF):
        """Draw current position marker."""
        pos = self._il_xl_to_screen(self.current_il, self.current_xl)

        if not (plot_rect.left() <= pos.x() <= plot_rect.right() and
               plot_rect.top() <= pos.y() <= plot_rect.bottom()):
            return

        # Draw crosshair
        pen = QPen(QColor(255, 255, 0), 2)
        painter.setPen(pen)

        # Vertical line
        painter.drawLine(int(pos.x()), int(plot_rect.top()),
                        int(pos.x()), int(plot_rect.bottom()))
        # Horizontal line
        painter.drawLine(int(plot_rect.left()), int(pos.y()),
                        int(plot_rect.right()), int(pos.y()))

        # Draw circle
        painter.setBrush(QBrush(QColor(255, 255, 0, 150)))
        painter.drawEllipse(pos, 6, 6)

    def _draw_axes(self, painter: QPainter, plot_rect: QRectF):
        """Draw axis labels."""
        painter.setFont(self.axis_font)
        painter.setPen(QColor(200, 200, 200))
        fm = QFontMetrics(self.axis_font)

        # X-axis (Inline) - bottom
        x_ticks = self._get_nice_ticks(self.il_min, self.il_max, 5)
        for tick in x_ticks:
            pos = self._il_xl_to_screen(tick, self.xl_min)
            if plot_rect.left() <= pos.x() <= plot_rect.right():
                painter.drawLine(int(pos.x()), int(plot_rect.bottom()),
                               int(pos.x()), int(plot_rect.bottom() + 5))
                label = f"{int(tick)}"
                text_width = fm.horizontalAdvance(label)
                painter.drawText(int(pos.x() - text_width / 2),
                               int(plot_rect.bottom() + 18), label)

        # X-axis label
        painter.setFont(self.title_font)
        il_label = "Inline"
        label_width = QFontMetrics(self.title_font).horizontalAdvance(il_label)
        painter.drawText(int(plot_rect.center().x() - label_width / 2),
                        int(self.height() - 5), il_label)

        # Y-axis (Crossline) - left
        painter.setFont(self.axis_font)
        y_ticks = self._get_nice_ticks(self.xl_min, self.xl_max, 5)
        for tick in y_ticks:
            pos = self._il_xl_to_screen(self.il_min, tick)
            if plot_rect.top() <= pos.y() <= plot_rect.bottom():
                painter.drawLine(int(plot_rect.left() - 5), int(pos.y()),
                               int(plot_rect.left()), int(pos.y()))
                label = f"{int(tick)}"
                text_width = fm.horizontalAdvance(label)
                painter.drawText(int(plot_rect.left() - text_width - 8),
                               int(pos.y() + fm.height() / 3), label)

        # Y-axis label (rotated)
        painter.setFont(self.title_font)
        xl_label = "Crossline"
        painter.save()
        painter.translate(12, plot_rect.center().y())
        painter.rotate(-90)
        label_width = QFontMetrics(self.title_font).horizontalAdvance(xl_label)
        painter.drawText(int(-label_width / 2), 0, xl_label)
        painter.restore()

    def _draw_title(self, painter: QPainter):
        """Draw title."""
        painter.setFont(self.title_font)
        painter.setPen(QColor(255, 255, 255))

        title = "Survey Map"
        if self.fold_map is not None:
            max_fold = int(np.max(self.fold_map))
            title += f" (max fold: {max_fold})"

        if self.grid_locations:
            n_grid = len(self.grid_locations)
            n_visited = len(self.visited_locations)
            title += f" | Grid: {n_visited}/{n_grid}"

        painter.drawText(self.LEFT_MARGIN, 20, title)

    def _draw_cursor_info(self, painter: QPainter, plot_rect: QRectF):
        """Draw cursor position info."""
        painter.setFont(self.axis_font)
        painter.setPen(QColor(200, 200, 200))

        info = f"IL: {self.mouse_il}  XL: {self.mouse_xl}"

        if self.fold_map is not None:
            # Get fold at cursor
            il_range = self.il_max - self.il_min
            xl_range = self.xl_max - self.xl_min
            h, w = self.fold_map.shape

            if il_range > 0 and xl_range > 0:
                ix = int((self.mouse_il - self.il_min) / il_range * (w - 1))
                iy = int((self.mouse_xl - self.xl_min) / xl_range * (h - 1))
                if 0 <= ix < w and 0 <= iy < h:
                    fold_val = self.fold_map[iy, ix]
                    info += f"  Fold: {int(fold_val)}"

        painter.drawText(int(plot_rect.right() - 150), int(plot_rect.top() - 5), info)

    def _get_nice_ticks(self, vmin: float, vmax: float, n_ticks: int = 5) -> List[float]:
        """Generate nice tick values."""
        if vmax <= vmin:
            return [vmin]

        range_val = vmax - vmin
        rough_step = range_val / n_ticks

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

        tick_min = np.ceil(vmin / nice_step) * nice_step
        tick_max = np.floor(vmax / nice_step) * nice_step

        ticks = []
        tick = tick_min
        while tick <= tick_max + nice_step * 0.01:
            if vmin <= tick <= vmax:
                ticks.append(tick)
            tick += nice_step

        return ticks

    def mouseMoveEvent(self, event: QMouseEvent):
        """Track mouse position."""
        pos = event.position()
        self.mouse_il, self.mouse_xl = self._screen_to_il_xl(pos.x(), pos.y())
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse click to select location."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            plot_rect = self._get_plot_rect()

            if (plot_rect.left() <= pos.x() <= plot_rect.right() and
               plot_rect.top() <= pos.y() <= plot_rect.bottom()):

                il, xl = self._screen_to_il_xl(pos.x(), pos.y())

                # Snap to nearest grid location if available
                if self.grid_locations:
                    best_dist = float('inf')
                    best_loc = (il, xl)
                    for grid_il, grid_xl in self.grid_locations:
                        dist = (grid_il - il)**2 + (grid_xl - xl)**2
                        if dist < best_dist:
                            best_dist = dist
                            best_loc = (grid_il, grid_xl)

                    # Only snap if close enough (within 20 units)
                    if best_dist < 400:  # 20^2
                        il, xl = best_loc

                self.location_selected.emit(il, xl)

    def enterEvent(self, event):
        """Change cursor when entering."""
        self.setCursor(Qt.CursorShape.CrossCursor)

    def leaveEvent(self, event):
        """Reset cursor when leaving."""
        self.setCursor(Qt.CursorShape.ArrowCursor)
