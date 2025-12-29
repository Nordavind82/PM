"""
Step 3: Output Grid - Define output grid by corner points and bin size.

IMPORTANT: This step must be completed BEFORE velocity configuration,
as velocity will be interpolated to the output grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QDoubleSpinBox, QFrame, QGridLayout,
    QComboBox, QSplitter, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt

from pstm.gui.steps.base import WizardStepWidget
from pstm.gui.theme import COLORS

if TYPE_CHECKING:
    from pstm.gui.models import ProjectModel


class CornerPointInput(QFrame):
    """Widget for inputting a corner point coordinate."""
    
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("cornerPointFrame")
        self.setStyleSheet(f"""
            QFrame#cornerPointFrame {{
                background-color: {COLORS['background_alt']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 5px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)
        
        # Label
        title = QLabel(label)
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        # X coordinate
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-1e9, 1e9)
        self.x_spin.setDecimals(2)
        self.x_spin.setSuffix(" m")
        self.x_spin.setMinimumWidth(120)
        x_layout.addWidget(self.x_spin)
        layout.addLayout(x_layout)
        
        # Y coordinate
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-1e9, 1e9)
        self.y_spin.setDecimals(2)
        self.y_spin.setSuffix(" m")
        self.y_spin.setMinimumWidth(120)
        y_layout.addWidget(self.y_spin)
        layout.addLayout(y_layout)
    
    def get_point(self) -> tuple[float, float]:
        return (self.x_spin.value(), self.y_spin.value())
    
    def set_point(self, x: float, y: float) -> None:
        self.x_spin.setValue(x)
        self.y_spin.setValue(y)


class OutputGridStep(WizardStepWidget):
    """Step 3: Output Grid configuration."""
    
    @property
    def title(self) -> str:
        return "Output Grid"
    
    def _setup_ui(self) -> None:
        """Set up the UI."""
        # Header
        header = self.create_header(
            "Step 3: Output Grid Definition",
            "Define the output grid using corner point coordinates and bin sizes. "
            "The output bin size is INDEPENDENT of input data spacing."
        )
        self.content_layout.addWidget(header)
        
        # Info box
        info = self.create_info_box(
            "Output grid must be defined before velocity configuration. "
            "Velocity will be interpolated to this grid.",
            "info"
        )
        self.content_layout.addWidget(info)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        self._create_method_section(left_layout)
        self._create_azimuth_section(left_layout)
        self._create_corners_section(left_layout)
        self._create_binsize_section(left_layout)
        self._create_time_section(left_layout)
        left_layout.addStretch()
        
        splitter.addWidget(left_widget)
        
        # Right side - Preview and Statistics
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        self._create_preview_section(right_layout)
        self._create_stats_section(right_layout)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 500])
        
        self.content_layout.addWidget(splitter, 1)
    
    def _create_method_section(self, parent_layout: QVBoxLayout) -> None:
        """Create grid definition method selection."""
        group = QGroupBox("Grid Definition Method")
        layout = QVBoxLayout(group)
        
        self.method_group = QButtonGroup(self)
        
        self.corner_radio = QRadioButton("Corner Points (4 corners)")
        self.corner_radio.setChecked(True)
        self.method_group.addButton(self.corner_radio, 0)
        layout.addWidget(self.corner_radio)
        
        self.bounds_radio = QRadioButton("Bounding Box (min/max)")
        self.method_group.addButton(self.bounds_radio, 1)
        layout.addWidget(self.bounds_radio)
        
        self.method_group.idToggled.connect(self._on_method_changed)

        parent_layout.addWidget(group)

    def _create_azimuth_section(self, parent_layout: QVBoxLayout) -> None:
        """Create grid azimuth/rotation control section."""
        group = QGroupBox("Grid Azimuth / Rotation")
        layout = QVBoxLayout(group)

        desc = QLabel(
            "Grid azimuth defines the rotation of the output grid relative to North (Y-axis). "
            "Auto-detect calculates the acquisition azimuth from midpoint distribution."
        )
        desc.setWordWrap(True)
        desc.setObjectName("sectionDescription")
        layout.addWidget(desc)

        # Azimuth control row
        azimuth_layout = QHBoxLayout()

        azimuth_layout.addWidget(QLabel("Azimuth:"))

        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(0, 180)
        self.azimuth_spin.setValue(0)
        self.azimuth_spin.setDecimals(1)
        self.azimuth_spin.setSuffix("°")
        self.azimuth_spin.setMinimumWidth(100)
        self.azimuth_spin.setToolTip("Grid rotation from North (Y-axis), clockwise positive")
        self.azimuth_spin.valueChanged.connect(self._on_azimuth_changed)
        azimuth_layout.addWidget(self.azimuth_spin)

        self.auto_azimuth_btn = QPushButton("Auto-Detect")
        self.auto_azimuth_btn.setToolTip("Calculate acquisition azimuth from midpoint distribution (PCA)")
        self.auto_azimuth_btn.clicked.connect(self._auto_detect_azimuth)
        azimuth_layout.addWidget(self.auto_azimuth_btn)

        self.azimuth_info_label = QLabel("")
        self.azimuth_info_label.setStyleSheet("color: #888; font-size: 11px;")
        azimuth_layout.addWidget(self.azimuth_info_label)

        azimuth_layout.addStretch()
        layout.addLayout(azimuth_layout)

        parent_layout.addWidget(group)

    def _create_corners_section(self, parent_layout: QVBoxLayout) -> None:
        """Create corner points input section."""
        group = QGroupBox("Corner Point Coordinates")
        layout = QVBoxLayout(group)
        
        desc = QLabel(
            "Define the output grid by specifying 4 corner points. "
            "Grid orientation is determined by the corner arrangement."
        )
        desc.setWordWrap(True)
        desc.setObjectName("sectionDescription")
        layout.addWidget(desc)
        
        # Corner points in a grid layout
        corners_layout = QGridLayout()
        corners_layout.setSpacing(10)
        
        # NW - Corner 4       NE - Corner 3
        # SW - Corner 1       SE - Corner 2
        
        self.corner4 = CornerPointInput("Corner 4 (NW)")
        corners_layout.addWidget(self.corner4, 0, 0)
        
        self.corner3 = CornerPointInput("Corner 3 (NE)")
        corners_layout.addWidget(self.corner3, 0, 1)
        
        self.corner1 = CornerPointInput("Corner 1 (SW/Origin)")
        corners_layout.addWidget(self.corner1, 1, 0)
        
        self.corner2 = CornerPointInput("Corner 2 (SE)")
        corners_layout.addWidget(self.corner2, 1, 1)
        
        layout.addLayout(corners_layout)
        
        # Quick actions
        btn_layout = QHBoxLayout()

        copy_survey_btn = QPushButton("Copy from Survey Extent")
        copy_survey_btn.setToolTip("Use source/receiver extent from Survey step")
        copy_survey_btn.clicked.connect(self._copy_from_survey)
        btn_layout.addWidget(copy_survey_btn)

        copy_midpoints_btn = QPushButton("Copy from Midpoints")
        copy_midpoints_btn.setToolTip("Use calculated midpoint (CDP) extent from headers")
        copy_midpoints_btn.clicked.connect(self._copy_from_midpoints)
        btn_layout.addWidget(copy_midpoints_btn)

        load_sps_btn = QPushButton("Load from SPS")
        btn_layout.addWidget(load_sps_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Connect change signals
        for corner in [self.corner1, self.corner2, self.corner3, self.corner4]:
            corner.x_spin.valueChanged.connect(self._on_grid_changed)
            corner.y_spin.valueChanged.connect(self._on_grid_changed)
        
        self.corners_group = group
        parent_layout.addWidget(group)
    
    def _create_binsize_section(self, parent_layout: QVBoxLayout) -> None:
        """Create bin size input section."""
        group = QGroupBox("Output Bin Size")
        layout = QVBoxLayout(group)

        info = QLabel("Note: Output bin size is INDEPENDENT of input data spacing")
        info.setObjectName("sectionDescription")
        layout.addWidget(info)

        form = QFormLayout()
        form.setSpacing(10)

        # Inline bin size
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 1000)
        self.dx_spin.setValue(25.0)
        self.dx_spin.setDecimals(1)
        self.dx_spin.setSuffix(" m")
        self.dx_spin.valueChanged.connect(self._on_grid_changed)
        form.addRow("Inline (X) Bin Size:", self.dx_spin)

        # Crossline bin size
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.1, 1000)
        self.dy_spin.setValue(25.0)
        self.dy_spin.setDecimals(1)
        self.dy_spin.setSuffix(" m")
        self.dy_spin.valueChanged.connect(self._on_grid_changed)
        form.addRow("Crossline (Y) Bin Size:", self.dy_spin)

        # Time sample interval
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.1, 100)
        self.dt_spin.setValue(2.0)
        self.dt_spin.setDecimals(1)
        self.dt_spin.setSuffix(" ms")
        self.dt_spin.valueChanged.connect(self._on_grid_changed)
        form.addRow("Time Sample Interval:", self.dt_spin)

        layout.addLayout(form)

        # Auto-calculate and quick presets row
        btn_layout = QHBoxLayout()

        # Auto-calculate button
        self.auto_bin_btn = QPushButton("Auto-Calculate")
        self.auto_bin_btn.setToolTip(
            "Automatically calculate optimal bin size from midpoint distribution"
        )
        self.auto_bin_btn.clicked.connect(self._auto_calculate_bin_size)
        btn_layout.addWidget(self.auto_bin_btn)

        # Auto-calc info label
        self.bin_calc_info_label = QLabel("")
        self.bin_calc_info_label.setStyleSheet("color: #888; font-size: 11px;")
        btn_layout.addWidget(self.bin_calc_info_label)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Quick presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Quick Presets:"))

        for size in [12.5, 25, 50]:
            btn = QPushButton(f"{size}m × {size}m")
            btn.clicked.connect(lambda checked, s=size: self._set_bin_preset(s))
            preset_layout.addWidget(btn)

        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        parent_layout.addWidget(group)
    
    def _create_time_section(self, parent_layout: QVBoxLayout) -> None:
        """Create time range input section."""
        group = QGroupBox("Time Range")
        layout = QFormLayout(group)
        
        # Start time
        self.t_min_spin = QDoubleSpinBox()
        self.t_min_spin.setRange(0, 50000)
        self.t_min_spin.setValue(0)
        self.t_min_spin.setDecimals(1)
        self.t_min_spin.setSuffix(" ms")
        self.t_min_spin.valueChanged.connect(self._on_grid_changed)
        layout.addRow("Start Time:", self.t_min_spin)
        
        # End time
        self.t_max_spin = QDoubleSpinBox()
        self.t_max_spin.setRange(0, 50000)
        self.t_max_spin.setValue(4000)
        self.t_max_spin.setDecimals(1)
        self.t_max_spin.setSuffix(" ms")
        self.t_max_spin.valueChanged.connect(self._on_grid_changed)
        layout.addRow("End Time:", self.t_max_spin)
        
        parent_layout.addWidget(group)
    
    def _create_preview_section(self, parent_layout: QVBoxLayout) -> None:
        """Create grid preview section."""
        group = QGroupBox("Grid Preview")
        layout = QVBoxLayout(group)
        
        # Matplotlib canvas placeholder
        self.preview_frame = QFrame()
        self.preview_frame.setMinimumSize(350, 350)
        self.preview_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['background_alt']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        
        preview_layout = QVBoxLayout(self.preview_frame)
        self.preview_label = QLabel("Configure grid to see preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(self.preview_frame, 1)
        
        # View controls
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QPushButton("Snap to Survey"))
        btn_layout.addWidget(QPushButton("Rotate Grid..."))
        btn_layout.addWidget(QPushButton("Reset"))
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        parent_layout.addWidget(group, 1)
    
    def _create_stats_section(self, parent_layout: QVBoxLayout) -> None:
        """Create grid statistics section."""
        group = QGroupBox("Grid Statistics")
        layout = QVBoxLayout(group)

        form = QFormLayout()

        self.inline_extent_label = QLabel("--")
        form.addRow("Inline extent:", self.inline_extent_label)

        self.crossline_extent_label = QLabel("--")
        form.addRow("Crossline extent:", self.crossline_extent_label)

        self.time_extent_label = QLabel("--")
        form.addRow("Time extent:", self.time_extent_label)

        self.rotation_label = QLabel("--")
        form.addRow("Grid rotation:", self.rotation_label)

        self.total_points_label = QLabel("--")
        form.addRow("Total grid points:", self.total_points_label)

        self.estimated_size_label = QLabel("--")
        form.addRow("Estimated output size:", self.estimated_size_label)

        layout.addLayout(form)

        # Coverage analysis section
        coverage_group = QGroupBox("Coverage Analysis")
        coverage_layout = QVBoxLayout(coverage_group)

        coverage_form = QFormLayout()

        self.coverage_inside_label = QLabel("--")
        coverage_form.addRow("Points inside grid:", self.coverage_inside_label)

        self.coverage_outside_label = QLabel("--")
        coverage_form.addRow("Points outside grid:", self.coverage_outside_label)

        self.coverage_ratio_label = QLabel("--")
        coverage_form.addRow("Coverage ratio:", self.coverage_ratio_label)

        self.outlier_warning_label = QLabel("")
        self.outlier_warning_label.setStyleSheet("color: #FFA500; font-weight: bold;")
        self.outlier_warning_label.setWordWrap(True)
        coverage_form.addRow("", self.outlier_warning_label)

        coverage_layout.addLayout(coverage_form)

        # Analyze coverage button
        analyze_btn_layout = QHBoxLayout()
        self.analyze_coverage_btn = QPushButton("Analyze Coverage")
        self.analyze_coverage_btn.setToolTip(
            "Analyze how many midpoints fall inside/outside the defined grid"
        )
        self.analyze_coverage_btn.clicked.connect(self._analyze_coverage)
        analyze_btn_layout.addWidget(self.analyze_coverage_btn)

        self.extend_grid_btn = QPushButton("Extend to Include All")
        self.extend_grid_btn.setToolTip(
            "Extend grid boundaries to include all data points"
        )
        self.extend_grid_btn.clicked.connect(self._extend_grid_to_include_all)
        self.extend_grid_btn.setEnabled(False)
        analyze_btn_layout.addWidget(self.extend_grid_btn)

        analyze_btn_layout.addStretch()
        coverage_layout.addLayout(analyze_btn_layout)

        layout.addWidget(coverage_group)

        parent_layout.addWidget(group)
    
    def _on_method_changed(self, button_id: int, checked: bool) -> None:
        """Handle grid definition method change."""
        if not checked:
            return
        # Toggle corner point inputs based on method
        is_corner_method = button_id == 0
        self.corners_group.setEnabled(is_corner_method)

    def _auto_detect_azimuth(self) -> None:
        """Auto-detect acquisition azimuth from midpoint distribution."""
        azimuth = self.controller.compute_acquisition_azimuth()
        if azimuth is not None:
            self.azimuth_spin.setValue(azimuth)
            self.azimuth_info_label.setText(f"(detected: {azimuth:.1f}°)")
        else:
            self._show_warning(
                "Could not calculate acquisition azimuth. "
                "Please ensure header data is loaded with valid coordinates."
            )

    def _on_azimuth_changed(self) -> None:
        """Handle azimuth value change - update corners if extent was set."""
        # If corners are currently set to a valid grid, recalculate with new azimuth
        c1 = self.corner1.get_point()
        c2 = self.corner2.get_point()
        if c1 != (0, 0) or c2 != (0, 0):
            # Corners are set - just update the display/preview
            self._on_grid_changed()

    def _apply_azimuth_to_corners(self, use_midpoints: bool = True) -> bool:
        """Apply current azimuth to compute rotated corners from survey data.

        Args:
            use_midpoints: If True, use midpoint extent; otherwise use source/receiver extent

        Returns:
            True if successful, False otherwise
        """
        azimuth = self.azimuth_spin.value()
        corners = self.controller.compute_rotated_extent(azimuth, use_midpoints=use_midpoints)

        if corners is None:
            return False

        c1, c2, c3, c4 = corners
        self.corner1.set_point(c1[0], c1[1])
        self.corner2.set_point(c2[0], c2[1])
        self.corner3.set_point(c3[0], c3[1])
        self.corner4.set_point(c4[0], c4[1])

        return True

    def _copy_from_survey(self) -> None:
        """Copy grid corners from survey extent with rotation."""
        geom = self.controller.state.survey

        # Check if survey data is valid
        if geom is None:
            self._show_warning("Survey data not available. Please complete the Survey step first.")
            return

        if geom.x_max <= geom.x_min or geom.y_max <= geom.y_min:
            self._show_warning(
                "Survey extent not analyzed. Please run geometry analysis in the Survey step first."
            )
            return

        # Auto-detect azimuth if not set
        if self.azimuth_spin.value() == 0:
            azimuth = self.controller.compute_acquisition_azimuth()
            if azimuth is not None:
                self.azimuth_spin.setValue(azimuth)
                self.azimuth_info_label.setText(f"(detected: {azimuth:.1f}°)")

        # Compute rotated corners using source/receiver extent
        if not self._apply_azimuth_to_corners(use_midpoints=False):
            # Fallback to axis-aligned if rotation fails
            self.corner1.set_point(geom.x_min, geom.y_min)
            self.corner2.set_point(geom.x_max, geom.y_min)
            self.corner3.set_point(geom.x_max, geom.y_max)
            self.corner4.set_point(geom.x_min, geom.y_max)

        self._on_grid_changed()

    def _copy_from_midpoints(self) -> None:
        """Copy grid corners from calculated midpoint extent with rotation."""
        try:
            # Get midpoint extent from controller
            stats = self.controller.get_header_statistics()
            if stats is None:
                self._show_warning(
                    "Header data not loaded. Please load input data first."
                )
                return

            mx_min, mx_max = stats.midpoint_x_range
            my_min, my_max = stats.midpoint_y_range

            if mx_max <= mx_min or my_max <= my_min:
                self._show_warning("Invalid midpoint extent in header data.")
                return

            # Auto-detect azimuth if not set
            if self.azimuth_spin.value() == 0:
                azimuth = self.controller.compute_acquisition_azimuth()
                if azimuth is not None:
                    self.azimuth_spin.setValue(azimuth)
                    self.azimuth_info_label.setText(f"(detected: {azimuth:.1f}°)")

            # Compute rotated corners using midpoint extent
            if not self._apply_azimuth_to_corners(use_midpoints=True):
                # Fallback to axis-aligned if rotation fails
                self.corner1.set_point(mx_min, my_min)
                self.corner2.set_point(mx_max, my_min)
                self.corner3.set_point(mx_max, my_max)
                self.corner4.set_point(mx_min, my_max)

            self._on_grid_changed()

        except Exception as e:
            self._show_warning(f"Failed to get midpoint extent: {e}")

    def _show_warning(self, message: str) -> None:
        """Show a warning message box."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Warning", message)
    
    def _set_bin_preset(self, size: float) -> None:
        """Set bin size preset."""
        self.dx_spin.setValue(size)
        self.dy_spin.setValue(size)

    def _auto_calculate_bin_size(self) -> None:
        """Auto-calculate optimal bin size from midpoint distribution."""
        result = self.controller.auto_calculate_bin_size(method="ensemble")

        if result is None:
            self._show_warning(
                "Could not calculate bin size. "
                "Please ensure header data is loaded with valid coordinates."
            )
            return

        dx, dy, details = result

        # Update spinboxes
        self.dx_spin.setValue(dx)
        self.dy_spin.setValue(dy)

        # Show info
        confidence = details.get("confidence", 0) * 100
        fold = details.get("fold_estimate", 0)
        method = details.get("method", "unknown")

        self.bin_calc_info_label.setText(
            f"Detected: {dx:.1f}m × {dy:.1f}m (conf: {confidence:.0f}%, fold: {fold:.1f})"
        )

        # Show warnings if any
        warnings = details.get("warnings", [])
        if warnings:
            self._show_warning("\n".join(warnings))

        self._on_grid_changed()

    def _analyze_coverage(self) -> None:
        """Analyze midpoint coverage relative to current grid."""
        result = self.controller.analyze_grid_coverage()

        if result is None:
            self._show_warning(
                "Could not analyze coverage. "
                "Please ensure header data and grid are properly configured."
            )
            return

        classification = result.get("classification", {})

        n_inside = classification.get("n_inside", 0)
        n_outside = classification.get("n_outside", 0)
        n_total = classification.get("n_total", 1)
        inside_ratio = classification.get("inside_ratio", 1.0)
        suggested_buffer = classification.get("suggested_buffer_m", 0)

        # Update labels
        self.coverage_inside_label.setText(f"{n_inside:,}")
        self.coverage_outside_label.setText(f"{n_outside:,}")
        self.coverage_ratio_label.setText(f"{inside_ratio*100:.1f}%")

        # Store for extend button
        self._last_coverage_result = result
        self._suggested_buffer = suggested_buffer

        # Show warning if significant outliers
        if n_outside > 0 and inside_ratio < 0.99:
            warning = result.get("recommendation_reason", "")
            self.outlier_warning_label.setText(warning)
            self.extend_grid_btn.setEnabled(True)
        else:
            self.outlier_warning_label.setText("")
            self.extend_grid_btn.setEnabled(False)

    def _extend_grid_to_include_all(self) -> None:
        """Extend grid boundaries to include all data points."""
        if not hasattr(self, "_suggested_buffer") or self._suggested_buffer <= 0:
            return

        buffer = self._suggested_buffer

        # Get current corners
        c1 = list(self.corner1.get_point())
        c2 = list(self.corner2.get_point())
        c3 = list(self.corner3.get_point())
        c4 = list(self.corner4.get_point())

        # Compute center
        center_x = (c1[0] + c2[0] + c3[0] + c4[0]) / 4
        center_y = (c1[1] + c2[1] + c3[1] + c4[1]) / 4

        # Extend each corner outward from center
        for corner, (cx, cy) in [(c1, c1), (c2, c2), (c3, c3), (c4, c4)]:
            dx = corner[0] - center_x
            dy = corner[1] - center_y
            length = (dx**2 + dy**2) ** 0.5
            if length > 0:
                corner[0] += (dx / length) * buffer
                corner[1] += (dy / length) * buffer

        # Update corner widgets
        self.corner1.set_point(c1[0], c1[1])
        self.corner2.set_point(c2[0], c2[1])
        self.corner3.set_point(c3[0], c3[1])
        self.corner4.set_point(c4[0], c4[1])

        self._on_grid_changed()

        # Re-analyze to update display
        self._analyze_coverage()
    
    def _on_grid_changed(self) -> None:
        """Handle grid parameter change."""
        self._update_statistics()
        self._update_preview()
    
    def _update_statistics(self) -> None:
        """Update grid statistics display."""
        import numpy as np
        
        # Get corner points
        c1 = self.corner1.get_point()
        c2 = self.corner2.get_point()
        c3 = self.corner3.get_point()
        c4 = self.corner4.get_point()
        
        # Calculate inline and crossline lengths
        inline_length = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
        crossline_length = np.sqrt((c4[0] - c1[0])**2 + (c4[1] - c1[1])**2)
        
        dx = self.dx_spin.value()
        dy = self.dy_spin.value()
        dt = self.dt_spin.value()
        t_min = self.t_min_spin.value()
        t_max = self.t_max_spin.value()
        
        nx = max(1, int(inline_length / dx) + 1)
        ny = max(1, int(crossline_length / dy) + 1)
        nt = max(1, int((t_max - t_min) / dt) + 1)
        
        self.inline_extent_label.setText(f"{inline_length:.1f} m → {nx} bins")
        self.crossline_extent_label.setText(f"{crossline_length:.1f} m → {ny} bins")
        self.time_extent_label.setText(f"{t_max - t_min:.1f} ms → {nt} samples")
        
        # Rotation
        angle_rad = np.arctan2(c2[0] - c1[0], c2[1] - c1[1])
        rotation = np.degrees(angle_rad)
        self.rotation_label.setText(f"{rotation:.1f}°")
        
        # Total points
        total = nx * ny * nt
        self.total_points_label.setText(f"{nx} × {ny} × {nt} = {total:,}")
        
        # Size estimate
        size_gb_f32 = total * 4 / (1024**3)
        size_gb_f64 = total * 8 / (1024**3)
        self.estimated_size_label.setText(
            f"{size_gb_f32:.2f} GB (float32) / {size_gb_f64:.2f} GB (float64)"
        )
    
    def _update_preview(self) -> None:
        """Update grid preview visualization."""
        try:
            import matplotlib
            matplotlib.use('QtAgg')
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            import numpy as np
            
            # Clear previous
            for i in reversed(range(self.preview_frame.layout().count())):
                widget = self.preview_frame.layout().itemAt(i).widget()
                if widget:
                    widget.deleteLater()
            
            # Get corners
            c1 = self.corner1.get_point()
            c2 = self.corner2.get_point()
            c3 = self.corner3.get_point()
            c4 = self.corner4.get_point()
            
            # Skip if no valid grid
            if c1 == c2 == c3 == c4:
                self.preview_label = QLabel("Configure grid corners")
                self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.preview_frame.layout().addWidget(self.preview_label)
                return
            
            # Create figure
            fig = Figure(figsize=(4, 4), dpi=100)
            canvas = FigureCanvasQTAgg(fig)
            
            ax = fig.add_subplot(111)
            
            # Plot output grid polygon
            corners_x = [c1[0], c2[0], c3[0], c4[0], c1[0]]
            corners_y = [c1[1], c2[1], c3[1], c4[1], c1[1]]
            
            ax.fill(corners_x, corners_y, alpha=0.3, color='blue', label='Output Grid')
            ax.plot(corners_x, corners_y, 'b-', linewidth=2)
            
            # Mark corners
            for i, (cx, cy) in enumerate([(c1[0], c1[1]), (c2[0], c2[1]), 
                                           (c3[0], c3[1]), (c4[0], c4[1])]):
                ax.plot(cx, cy, 'bo', markersize=8)
                ax.annotate(f'C{i+1}', (cx, cy), textcoords="offset points",
                           xytext=(5, 5), fontsize=8)
            
            # Plot survey extent if available
            geom = self.controller.state.survey
            if geom.x_max > geom.x_min:
                survey_x = [geom.x_min, geom.x_max, geom.x_max, geom.x_min, geom.x_min]
                survey_y = [geom.y_min, geom.y_min, geom.y_max, geom.y_max, geom.y_min]
                ax.plot(survey_x, survey_y, 'g--', linewidth=1, label='Survey Extent')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Output Grid')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            self.preview_frame.layout().addWidget(canvas)
            
        except ImportError:
            pass
    
    def on_enter(self) -> None:
        """Called when navigating to this step."""
        self.load_from_model()

    def load_from_model(self) -> None:
        """Load data from project model."""
        cfg = self.controller.state.output_grid

        # Load corner points from state (uses corners.c1_x, c1_y, etc.)
        self.corner1.set_point(cfg.corners.c1_x, cfg.corners.c1_y)
        self.corner2.set_point(cfg.corners.c2_x, cfg.corners.c2_y)
        self.corner3.set_point(cfg.corners.c3_x, cfg.corners.c3_y)
        self.corner4.set_point(cfg.corners.c4_x, cfg.corners.c4_y)

        self.dx_spin.setValue(cfg.dx)
        self.dy_spin.setValue(cfg.dy)
        self.dt_spin.setValue(cfg.dt_ms)

        self.t_min_spin.setValue(cfg.t_min_ms)
        self.t_max_spin.setValue(cfg.t_max_ms)

        self._update_statistics()
        self._update_preview()

    def on_leave(self) -> None:
        """Save state when leaving this step."""
        self.save_to_model()

    def save_to_model(self) -> None:
        """Save data to project model."""
        import logging
        debug_logger = logging.getLogger("pstm.migration.debug")

        cfg = self.controller.state.output_grid

        c1 = self.corner1.get_point()
        c2 = self.corner2.get_point()
        c3 = self.corner3.get_point()
        c4 = self.corner4.get_point()

        debug_logger.info(f"OUTPUT_GRID save_to_model: c1={c1}, c2={c2}, c3={c3}, c4={c4}")
        debug_logger.info(f"OUTPUT_GRID save_to_model: dx={self.dx_spin.value()}, dy={self.dy_spin.value()}")

        # Save corner points to state (uses corners.c1_x, c1_y, etc.)
        cfg.corners.c1_x, cfg.corners.c1_y = c1
        cfg.corners.c2_x, cfg.corners.c2_y = c2
        cfg.corners.c3_x, cfg.corners.c3_y = c3
        cfg.corners.c4_x, cfg.corners.c4_y = c4

        cfg.dx = self.dx_spin.value()
        cfg.dy = self.dy_spin.value()
        cfg.dt_ms = self.dt_spin.value()

        cfg.t_min_ms = self.t_min_spin.value()
        cfg.t_max_ms = self.t_max_spin.value()

        debug_logger.info(f"OUTPUT_GRID after save: nx={cfg.nx}, ny={cfg.ny}, nt={cfg.nt}")
    
    def validate(self) -> bool:
        """Validate step data."""
        self._validation_errors = []

        # Check corners form a valid quadrilateral
        c1 = self.corner1.get_point()
        c2 = self.corner2.get_point()
        c3 = self.corner3.get_point()
        c4 = self.corner4.get_point()

        if c1 == c2 == c3 == c4 == (0, 0):
            self._validation_errors.append("Please define output grid corners")

        if self.dx_spin.value() <= 0:
            self._validation_errors.append("Inline bin size must be positive")

        if self.dy_spin.value() <= 0:
            self._validation_errors.append("Crossline bin size must be positive")

        if self.t_max_spin.value() <= self.t_min_spin.value():
            self._validation_errors.append("End time must be greater than start time")

        if self._validation_errors:
            self.show_validation_errors()
            return False

        return True

    def refresh_from_state(self) -> None:
        """Refresh UI from loaded state."""
        self.load_from_model()
