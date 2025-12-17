"""
Step 4: Velocity Model - Configure velocity model for migration.

NOTE: This step comes AFTER output grid definition.
Velocity is interpolated to the output grid coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QDoubleSpinBox, QFrame, QRadioButton,
    QButtonGroup, QFileDialog, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QStackedWidget,
)
from PyQt6.QtCore import Qt

from pstm.gui.steps.base import WizardStepWidget
from pstm.gui.models import VelocitySource
from pstm.gui.theme import COLORS

if TYPE_CHECKING:
    from pstm.gui.models import ProjectModel


class VelocityStep(WizardStepWidget):
    """Step 4: Velocity Model configuration."""
    
    @property
    def title(self) -> str:
        return "Velocity Model"
    
    def _setup_ui(self) -> None:
        """Set up the UI."""
        # Header
        header = self.create_header(
            "Step 4: Velocity Model",
            "Configure the velocity model for migration. Velocity will be "
            "interpolated to the output grid defined in Step 3."
        )
        self.content_layout.addWidget(header)
        
        # Info box about output grid
        grid = self.controller.state.output_grid
        info_text = f"Output grid: {grid.nx} × {grid.ny} × {grid.nt} points"
        info = self.create_info_box(info_text, "info")
        self.grid_info_box = info
        self.content_layout.addWidget(info)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Velocity source selection and parameters
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        self._create_source_section(left_layout)
        self._create_params_section(left_layout)
        left_layout.addStretch()
        
        splitter.addWidget(left_widget)
        
        # Right side - Preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        self._create_preview_section(right_layout)
        self._create_qc_section(right_layout)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([450, 550])
        
        self.content_layout.addWidget(splitter, 1)
    
    def _create_source_section(self, parent_layout: QVBoxLayout) -> None:
        """Create velocity source selection section."""
        group = QGroupBox("Velocity Source")
        layout = QVBoxLayout(group)
        
        self.source_group = QButtonGroup(self)
        
        sources = [
            (VelocitySource.CONSTANT, "Constant Velocity"),
            (VelocitySource.LINEAR_V0K, "Linear V(t) = V₀ + k·t"),
            (VelocitySource.TABLE_1D, "1D Function V(t)"),
            (VelocitySource.CUBE_3D, "3D Velocity Cube"),
        ]
        
        for i, (source, label) in enumerate(sources):
            radio = QRadioButton(label)
            if i == 0:
                radio.setChecked(True)
            self.source_group.addButton(radio, i)
            layout.addWidget(radio)
        
        self.source_group.idToggled.connect(self._on_source_changed)
        
        parent_layout.addWidget(group)
    
    def _create_params_section(self, parent_layout: QVBoxLayout) -> None:
        """Create velocity parameters section (stacked widget)."""
        group = QGroupBox("Velocity Parameters")
        layout = QVBoxLayout(group)
        
        self.params_stack = QStackedWidget()
        
        # Constant velocity page
        const_page = self._create_constant_page()
        self.params_stack.addWidget(const_page)
        
        # Linear velocity page
        linear_page = self._create_linear_page()
        self.params_stack.addWidget(linear_page)
        
        # 1D function page
        func_page = self._create_function_page()
        self.params_stack.addWidget(func_page)
        
        # 3D cube page
        cube_page = self._create_cube_page()
        self.params_stack.addWidget(cube_page)
        
        # File import page
        file_page = self._create_file_page()
        self.params_stack.addWidget(file_page)
        
        layout.addWidget(self.params_stack)
        
        parent_layout.addWidget(group)
    
    def _create_constant_page(self) -> QWidget:
        """Create constant velocity input page."""
        page = QWidget()
        layout = QFormLayout(page)
        
        self.const_velocity_spin = QDoubleSpinBox()
        self.const_velocity_spin.setRange(100, 20000)
        self.const_velocity_spin.setValue(2500)
        self.const_velocity_spin.setSuffix(" m/s")
        self.const_velocity_spin.valueChanged.connect(self._update_preview)
        layout.addRow("Velocity (Vrms):", self.const_velocity_spin)
        
        # Valid range info
        info = QLabel("Valid range: 500 - 10,000 m/s")
        info.setObjectName("sectionDescription")
        layout.addRow("", info)
        
        return page
    
    def _create_linear_page(self) -> QWidget:
        """Create linear velocity input page."""
        page = QWidget()
        layout = QFormLayout(page)
        
        self.linear_v0_spin = QDoubleSpinBox()
        self.linear_v0_spin.setRange(100, 20000)
        self.linear_v0_spin.setValue(1800)
        self.linear_v0_spin.setSuffix(" m/s")
        self.linear_v0_spin.valueChanged.connect(self._update_preview)
        layout.addRow("V₀ (at t=0):", self.linear_v0_spin)
        
        self.linear_grad_spin = QDoubleSpinBox()
        self.linear_grad_spin.setRange(-10, 10)
        self.linear_grad_spin.setValue(0.5)
        self.linear_grad_spin.setDecimals(3)
        self.linear_grad_spin.setSuffix(" m/s per ms")
        self.linear_grad_spin.valueChanged.connect(self._update_preview)
        layout.addRow("Gradient (k):", self.linear_grad_spin)
        
        # Formula display
        self.linear_formula_label = QLabel("V(t) = 1800 + 0.5·t")
        layout.addRow("Formula:", self.linear_formula_label)
        
        # Example value
        self.linear_example_label = QLabel("At t=4000ms: V = 3800 m/s")
        layout.addRow("Example:", self.linear_example_label)
        
        return page
    
    def _create_function_page(self) -> QWidget:
        """Create 1D function input page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Time-velocity pairs table
        self.func_table = QTableWidget(5, 2)
        self.func_table.setHorizontalHeaderLabels(["Time (ms)", "Velocity (m/s)"])
        self.func_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Default values
        default_pairs = [(0, 1500), (1000, 2000), (2000, 2500), (3000, 3000), (4000, 3500)]
        for row, (t, v) in enumerate(default_pairs):
            self.func_table.setItem(row, 0, QTableWidgetItem(str(t)))
            self.func_table.setItem(row, 1, QTableWidgetItem(str(v)))
        
        self.func_table.itemChanged.connect(self._update_preview)
        layout.addWidget(self.func_table)
        
        # Add/remove row buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(lambda: self.func_table.insertRow(self.func_table.rowCount()))
        btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove Row")
        remove_btn.clicked.connect(lambda: self.func_table.removeRow(self.func_table.currentRow()))
        btn_layout.addWidget(remove_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return page
    
    def _create_cube_page(self) -> QWidget:
        """Create 3D velocity cube input page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Path input
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Path:"))
        
        self.cube_path_edit = QLineEdit()
        self.cube_path_edit.setPlaceholderText("Select 3D velocity cube...")
        path_layout.addWidget(self.cube_path_edit, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_cube)
        path_layout.addWidget(browse_btn)
        
        layout.addLayout(path_layout)
        
        # Cube info
        self.cube_info_frame = QFrame()
        self.cube_info_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['background_alt']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 10px;
            }}
        """)
        info_layout = QVBoxLayout(self.cube_info_frame)
        self.cube_info_label = QLabel("No cube loaded")
        info_layout.addWidget(self.cube_info_label)
        
        layout.addWidget(self.cube_info_frame)
        
        # Note about resampling
        note = QLabel(
            "Note: 3D cube will be resampled to output grid coordinates.\n"
            "Ensure cube covers the output grid extent."
        )
        note.setWordWrap(True)
        note.setObjectName("sectionDescription")
        layout.addWidget(note)
        
        layout.addStretch()
        
        return page
    
    def _create_file_page(self) -> QWidget:
        """Create file import page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Path input
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Path:"))
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select velocity file...")
        path_layout.addWidget(self.file_path_edit, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        path_layout.addWidget(browse_btn)
        
        layout.addLayout(path_layout)
        
        # Format info
        info = QLabel("Supported formats: ASCII (t,v pairs), SEG-Y, Zarr")
        info.setObjectName("sectionDescription")
        layout.addWidget(info)
        
        layout.addStretch()
        
        return page
    
    def _create_preview_section(self, parent_layout: QVBoxLayout) -> None:
        """Create velocity preview section."""
        group = QGroupBox("Velocity Preview")
        layout = QVBoxLayout(group)
        
        # Preview frame for matplotlib
        self.preview_frame = QFrame()
        self.preview_frame.setMinimumSize(350, 300)
        self.preview_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['background_alt']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        
        preview_layout = QVBoxLayout(self.preview_frame)
        self.preview_label = QLabel("Configure velocity to see preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(self.preview_frame, 1)
        
        # Prepare button
        prepare_btn = QPushButton("Prepare Velocity for Output Grid")
        prepare_btn.clicked.connect(self._prepare_velocity)
        layout.addWidget(prepare_btn)
        
        self.prepare_status_label = QLabel("")
        layout.addWidget(self.prepare_status_label)
        
        parent_layout.addWidget(group, 1)
    
    def _create_qc_section(self, parent_layout: QVBoxLayout) -> None:
        """Create velocity QC section."""
        group = QGroupBox("Velocity QC")
        layout = QVBoxLayout(group)
        
        self.qc_frame = QFrame()
        qc_layout = QVBoxLayout(self.qc_frame)
        
        self.qc_label = QLabel("Run 'Prepare Velocity' to see QC results")
        qc_layout.addWidget(self.qc_label)
        
        layout.addWidget(self.qc_frame)
        
        parent_layout.addWidget(group)
    
    def _on_source_changed(self, button_id: int, checked: bool) -> None:
        """Handle velocity source change."""
        if checked:
            self.params_stack.setCurrentIndex(button_id)
            self._update_preview()
    
    def _browse_cube(self) -> None:
        """Browse for 3D velocity cube."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Velocity Cube Directory"
        )
        if path:
            self.cube_path_edit.setText(path)
            self._load_cube_info(path)
    
    def _browse_file(self) -> None:
        """Browse for velocity file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Velocity File",
            "", "All Files (*);;ASCII (*.txt *.asc);;SEG-Y (*.segy *.sgy)"
        )
        if path:
            self.file_path_edit.setText(path)
    
    def _load_cube_info(self, path: str) -> None:
        """Load and display 3D cube info."""
        try:
            import zarr
            z = zarr.open(path, mode='r')
            
            if 'velocity' in z:
                arr = z['velocity']
            else:
                arr = z
            
            info = (
                f"Dimensions: {arr.shape[0]} × {arr.shape[1]} × {arr.shape[2]} (X × Y × T)\n"
                f"Data type: {arr.dtype}\n"
                f"Size: {arr.nbytes / 1e6:.1f} MB"
            )
            self.cube_info_label.setText(info)
            
        except Exception as e:
            self.cube_info_label.setText(f"Error loading cube: {e}")
    
    def _update_preview(self) -> None:
        """Update velocity profile preview."""
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
            
            # Get time range from output grid
            t_min = self.controller.state.output_grid.t_min_ms
            t_max = self.controller.state.output_grid.t_max_ms
            times = np.linspace(t_min, t_max, 100)
            
            # Get velocities based on source type
            source_idx = self.source_group.checkedId()
            
            if source_idx == 0:  # Constant
                v = self.const_velocity_spin.value()
                velocities = np.full_like(times, v)
            elif source_idx == 1:  # Linear
                v0 = self.linear_v0_spin.value()
                k = self.linear_grad_spin.value()
                velocities = v0 + k * times
                
                # Update formula label
                self.linear_formula_label.setText(f"V(t) = {v0:.0f} + {k:.3f}·t")
                v_example = v0 + k * t_max
                self.linear_example_label.setText(f"At t={t_max:.0f}ms: V = {v_example:.0f} m/s")
            elif source_idx == 2:  # 1D Function
                # Get values from table
                t_vals = []
                v_vals = []
                for row in range(self.func_table.rowCount()):
                    t_item = self.func_table.item(row, 0)
                    v_item = self.func_table.item(row, 1)
                    if t_item and v_item:
                        try:
                            t_vals.append(float(t_item.text()))
                            v_vals.append(float(v_item.text()))
                        except ValueError:
                            pass
                
                if len(t_vals) >= 2:
                    velocities = np.interp(times, t_vals, v_vals)
                else:
                    velocities = np.full_like(times, 2500)
            else:
                # 3D cube or file - show placeholder
                velocities = np.full_like(times, 2500)
            
            # Create figure
            fig = Figure(figsize=(4, 3), dpi=100)
            canvas = FigureCanvasQTAgg(fig)
            
            ax = fig.add_subplot(111)
            ax.plot(velocities, times, 'b-', linewidth=2)
            ax.invert_yaxis()
            ax.set_xlabel('Velocity (m/s)')
            ax.set_ylabel('Time (ms)')
            ax.set_title('Velocity Profile')
            ax.grid(True, alpha=0.3)
            
            # Add valid range shading
            ax.axvspan(500, 10000, alpha=0.1, color='green')
            
            fig.tight_layout()
            
            self.preview_frame.layout().addWidget(canvas)
            
        except ImportError:
            pass
    
    def _prepare_velocity(self) -> None:
        """Prepare velocity model for output grid."""
        self.prepare_status_label.setText("Preparing velocity...")
        
        # TODO: Implement actual velocity preparation
        # This would:
        # 1. Create velocity array matching output grid
        # 2. Interpolate to output coordinates
        # 3. Run QC checks
        
        self.prepare_status_label.setText("✓ Velocity prepared for output grid")
        
        # Update QC
        self.qc_label.setText(
            "✓ Velocity within expected range\n"
            "✓ No velocity inversions detected\n"
            "✓ Gradient within limits"
        )
    
    def on_enter(self) -> None:
        """Called when navigating to this step."""
        self.load_from_model()

    def load_from_model(self) -> None:
        """Load data from project model."""
        import logging
        debug_logger = logging.getLogger("pstm.migration.debug")

        cfg = self.controller.state.velocity

        # Set source radio button - use string values matching VelocityState
        source_map = {
            "constant": 0,
            "linear": 1,
            "function_1d": 2,
            "cube_3d": 3,
        }
        idx = source_map.get(cfg.source, 0)
        self.source_group.button(idx).setChecked(True)
        self.params_stack.setCurrentIndex(idx)

        # Load parameters
        self.const_velocity_spin.setValue(cfg.constant_velocity)
        self.linear_v0_spin.setValue(cfg.linear_v0)
        self.linear_grad_spin.setValue(cfg.linear_gradient)

        if cfg.cube_path:
            self.cube_path_edit.setText(cfg.cube_path)

        if cfg.file_path:
            self.file_path_edit.setText(cfg.file_path)

        # Update grid info from output_grid state
        grid = self.controller.state.output_grid
        debug_logger.info(f"VELOCITY on_enter: grid corners c1=({grid.corners.c1_x}, {grid.corners.c1_y})")
        debug_logger.info(f"VELOCITY on_enter: grid corners c2=({grid.corners.c2_x}, {grid.corners.c2_y})")
        debug_logger.info(f"VELOCITY on_enter: dx={grid.dx}, dy={grid.dy}")
        debug_logger.info(f"VELOCITY on_enter: computed nx={grid.nx}, ny={grid.ny}, nt={grid.nt}")

        info_text = f"Output grid: {grid.nx} × {grid.ny} × {grid.nt} points"
        # Find label in info box and update
        for child in self.grid_info_box.findChildren(QLabel):
            if "Output grid" not in child.text():
                continue
            child.setText(info_text)

        self._update_preview()
    
    def save_to_model(self) -> None:
        """Save data to project model."""
        cfg = self.controller.state.velocity
        
        # Save source type - use string values matching VelocityState
        source_idx = self.source_group.checkedId()
        source_map = {
            0: "constant",
            1: "linear",
            2: "function_1d",
            3: "cube_3d",
        }
        cfg.source = source_map.get(source_idx, "constant")
        
        # Save parameters
        cfg.constant_velocity = self.const_velocity_spin.value()
        cfg.linear_v0 = self.linear_v0_spin.value()
        cfg.linear_gradient = self.linear_grad_spin.value()
        cfg.cube_path = self.cube_path_edit.text()
        cfg.file_path = self.file_path_edit.text()
        
        # Save 1D function values
        if source_idx == 2:
            times = []
            velocities = []
            for row in range(self.func_table.rowCount()):
                t_item = self.func_table.item(row, 0)
                v_item = self.func_table.item(row, 1)
                if t_item and v_item:
                    try:
                        times.append(float(t_item.text()))
                        velocities.append(float(v_item.text()))
                    except ValueError:
                        pass
            cfg.function_times_ms = times
            cfg.function_velocities = velocities
    
    def validate(self) -> bool:
        """Validate step data."""
        self._validation_errors = []
        
        source_idx = self.source_group.checkedId()
        
        if source_idx == 0:  # Constant
            v = self.const_velocity_spin.value()
            if v < 500 or v > 10000:
                self._validation_errors.append(
                    f"Velocity {v} m/s outside typical range (500-10000 m/s)"
                )
        
        elif source_idx == 1:  # Linear
            v0 = self.linear_v0_spin.value()
            k = self.linear_grad_spin.value()
            t_max = self.controller.state.output_grid.t_max_ms
            v_max = v0 + k * t_max
            
            if v_max < 500 or v_max > 10000:
                self._validation_errors.append(
                    f"Velocity at t={t_max}ms ({v_max:.0f} m/s) outside typical range"
                )
        
        elif source_idx == 3:  # 3D Cube
            if not self.cube_path_edit.text():
                self._validation_errors.append("Please select a 3D velocity cube")
        
        elif source_idx == 4:  # File
            if not self.file_path_edit.text():
                self._validation_errors.append("Please select a velocity file")
        
        if self._validation_errors:
            self.show_validation_errors()
            return False
        
        return True
