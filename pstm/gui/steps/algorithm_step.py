"""
Step 6: Algorithm Parameters

Configure migration algorithm parameters: aperture, interpolation, amplitude corrections.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QGroupBox,
    QGridLayout, QFormLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath

from pstm.gui.steps.base import WizardStepWidget
from pstm.gui.state import StepStatus


class AlgorithmStep(WizardStepWidget):
    """Step 6: Algorithm - Migration parameters."""
    
    @property
    def title(self) -> str:
        return "Algorithm"
    
    def _setup_ui(self) -> None:
        # Main splitter for left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Main parameters
        left_widget = QFrame()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)

        self._create_kernel_type_section(left_layout)
        self._create_aperture_section(left_layout)
        self._create_interpolation_section(left_layout)
        self._create_amplitude_section(left_layout)
        self._create_time_variant_section(left_layout)
        left_layout.addStretch()
        
        splitter.addWidget(left_widget)
        
        # Right panel - Visualization and info
        right_widget = QFrame()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        self._create_aperture_diagram(right_layout)
        self._create_summary_section(right_layout)
        right_layout.addStretch()
        
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 400])
        
        self.content_layout.addWidget(splitter)

    def _create_kernel_type_section(self, parent_layout: QVBoxLayout) -> None:
        """Create migration kernel type selection section."""
        frame, layout = self.create_section("Migration Kernel Type")

        info = self.create_info_box(
            "Select traveltime computation method. Curved ray accounts for velocity gradients. "
            "VTI anisotropic uses Alkhalifah-Tsvankin eta parameter for non-hyperbolic moveout.",
            "info"
        )
        layout.addWidget(info)

        form = QFormLayout()
        form.setSpacing(10)

        # Kernel type selection
        self.kernel_type = QComboBox()
        self.kernel_type.addItems([
            "Straight Ray (Isotropic)",
            "Curved Ray (V(z) gradient)",
            "Anisotropic VTI (Eta parameter)",
        ])
        self.kernel_type.setCurrentIndex(0)
        self.kernel_type.currentIndexChanged.connect(self._on_kernel_type_changed)
        form.addRow("Kernel Type:", self.kernel_type)

        layout.addLayout(form)

        # Info label about parameter configuration
        self.kernel_params_info = QLabel(
            "Configure gradient and eta parameters in the Velocity step."
        )
        self.kernel_params_info.setStyleSheet("color: #888888; font-size: 11px;")
        self.kernel_params_info.setVisible(False)
        layout.addWidget(self.kernel_params_info)

        parent_layout.addWidget(frame)

    def _on_kernel_type_changed(self, index: int) -> None:
        """Handle kernel type selection change."""
        # 0 = Straight Ray, 1 = Curved Ray, 2 = VTI Anisotropic
        # Show info label for non-isotropic kernels
        self.kernel_params_info.setVisible(index > 0)
        self._on_param_changed()

    def _create_aperture_section(self, parent_layout: QVBoxLayout) -> None:
        """Create aperture configuration section."""
        frame, layout = self.create_section("Migration Aperture")
        
        info = self.create_info_box(
            "Aperture controls which input traces contribute to each output point. "
            "Time-dependent aperture adapts to structure dip.",
            "info"
        )
        layout.addWidget(info)
        
        form = QFormLayout()
        form.setSpacing(10)
        
        # Max dip
        self.max_dip_spin = self.create_double_spinbox(0, 89, 1, "°")
        self.max_dip_spin.setValue(45.0)
        self.max_dip_spin.valueChanged.connect(self._on_param_changed)
        form.addRow("Maximum Dip:", self.max_dip_spin)
        
        # Min aperture
        self.min_aperture_spin = self.create_double_spinbox(0, 50000, 0, "m")
        self.min_aperture_spin.setValue(500.0)
        self.min_aperture_spin.valueChanged.connect(self._on_param_changed)
        form.addRow("Minimum Aperture:", self.min_aperture_spin)
        
        # Max aperture
        self.max_aperture_spin = self.create_double_spinbox(0, 50000, 0, "m")
        self.max_aperture_spin.setValue(5000.0)
        self.max_aperture_spin.valueChanged.connect(self._on_param_changed)
        form.addRow("Maximum Aperture:", self.max_aperture_spin)
        
        layout.addLayout(form)
        
        # Taper settings
        taper_group = QGroupBox("Aperture Taper")
        taper_layout = QFormLayout(taper_group)
        
        self.taper_type = QComboBox()
        self.taper_type.addItems(["Cosine", "Linear", "Gaussian"])
        self.taper_type.setCurrentText("Cosine")
        taper_layout.addRow("Taper Type:", self.taper_type)
        
        self.taper_fraction = self.create_double_spinbox(0, 0.5, 2)
        self.taper_fraction.setValue(0.1)
        taper_layout.addRow("Taper Fraction:", self.taper_fraction)
        
        layout.addWidget(taper_group)
        
        parent_layout.addWidget(frame)
    
    def _create_interpolation_section(self, parent_layout: QVBoxLayout) -> None:
        """Create interpolation method section."""
        frame, layout = self.create_section("Trace Interpolation")
        
        form = QFormLayout()
        form.setSpacing(10)
        
        # Method selection
        self.interp_method = QComboBox()
        self.interp_method.addItems([
            "nearest - Nearest neighbor (fastest)",
            "linear - Linear (default, good balance)",
            "cubic - Catmull-Rom spline (smooth)",
            "sinc4 - 4-point windowed sinc",
            "sinc8 - 8-point sinc (recommended)",
            "sinc16 - 16-point sinc (highest quality)",
            "lanczos3 - 3-lobe Lanczos",
            "lanczos5 - 5-lobe Lanczos (sharpest)",
        ])
        self.interp_method.setCurrentIndex(1)  # linear default
        form.addRow("Method:", self.interp_method)
        
        layout.addLayout(form)
        
        # Comparison button
        btn_row = QHBoxLayout()
        compare_btn = QPushButton("Compare Methods...")
        compare_btn.clicked.connect(self._show_method_comparison)
        btn_row.addWidget(compare_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        
        parent_layout.addWidget(frame)
    
    def _create_amplitude_section(self, parent_layout: QVBoxLayout) -> None:
        """Create amplitude handling section."""
        frame, layout = self.create_section("Amplitude Corrections")
        
        # Checkboxes
        self.apply_spreading = QCheckBox("Apply geometrical spreading correction")
        self.apply_spreading.setChecked(True)
        self.apply_spreading.setToolTip("Compensate for amplitude decay due to wavefront spreading")
        layout.addWidget(self.apply_spreading)
        
        self.apply_obliquity = QCheckBox("Apply obliquity (cosine) factor")
        self.apply_obliquity.setChecked(True)
        self.apply_obliquity.setToolTip("Weight by cosine of incidence angle")
        layout.addWidget(self.apply_obliquity)
        
        self.wavelet_stretch = QCheckBox("Apply wavelet stretch correction (for AVO)")
        self.wavelet_stretch.setChecked(False)
        self.wavelet_stretch.setToolTip("Correct for NMO stretch at far offsets")
        layout.addWidget(self.wavelet_stretch)
        
        # Anti-aliasing
        aa_group = QGroupBox("Anti-Aliasing")
        aa_layout = QFormLayout(aa_group)
        
        self.enable_aa = QCheckBox("Enable anti-aliasing")
        self.enable_aa.setChecked(False)
        aa_layout.addRow(self.enable_aa)
        
        self.aa_method = QComboBox()
        self.aa_method.addItems(["Triangle filter", "Offset sectoring", "Dip filter"])
        self.aa_method.setEnabled(False)
        aa_layout.addRow("Method:", self.aa_method)
        
        self.enable_aa.toggled.connect(self.aa_method.setEnabled)
        
        layout.addWidget(aa_group)

        parent_layout.addWidget(frame)

    def _create_time_variant_section(self, parent_layout: QVBoxLayout) -> None:
        """Create time-variant sampling section."""
        frame, layout = self.create_section("Time-Variant Sampling")

        info = self.create_info_box(
            "Reduce computation by using coarser sampling at depth where high "
            "frequencies are attenuated. Define frequency vs time to optimize sampling.",
            "info"
        )
        layout.addWidget(info)

        # Enable checkbox
        self.tv_enabled = QCheckBox("Enable time-variant sampling")
        self.tv_enabled.setChecked(False)
        self.tv_enabled.setToolTip("Use adaptive time sampling based on frequency decay")
        self.tv_enabled.toggled.connect(self._on_tv_toggled)
        layout.addWidget(self.tv_enabled)

        # Container for TV controls (disabled when TV is off)
        self.tv_container = QFrame()
        tv_layout = QVBoxLayout(self.tv_container)
        tv_layout.setContentsMargins(0, 10, 0, 0)

        # Frequency-Time table
        table_label = QLabel("Frequency vs Time Table (edit values):")
        tv_layout.addWidget(table_label)

        self.freq_table = QTableWidget()
        self.freq_table.setColumnCount(2)
        self.freq_table.setHorizontalHeaderLabels(["Time (ms)", "Max Freq (Hz)"])
        self.freq_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.freq_table.setMaximumHeight(150)
        self.freq_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.freq_table.itemChanged.connect(self._on_freq_table_changed)
        tv_layout.addWidget(self.freq_table)

        # Add/Remove buttons
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self._add_freq_row)
        btn_row.addWidget(add_btn)

        remove_btn = QPushButton("Remove Row")
        remove_btn.clicked.connect(self._remove_freq_row)
        btn_row.addWidget(remove_btn)

        reset_btn = QPushButton("Reset Default")
        reset_btn.clicked.connect(self._reset_freq_table)
        btn_row.addWidget(reset_btn)

        btn_row.addStretch()
        tv_layout.addLayout(btn_row)

        # Downsample factor limits
        factor_layout = QFormLayout()

        self.min_factor_spin = QSpinBox()
        self.min_factor_spin.setRange(1, 8)
        self.min_factor_spin.setValue(1)
        self.min_factor_spin.setToolTip("Minimum downsample factor (1 = no downsampling)")
        factor_layout.addRow("Min Factor:", self.min_factor_spin)

        self.max_factor_spin = QSpinBox()
        self.max_factor_spin.setRange(1, 16)
        self.max_factor_spin.setValue(8)
        self.max_factor_spin.setToolTip("Maximum downsample factor")
        factor_layout.addRow("Max Factor:", self.max_factor_spin)

        tv_layout.addLayout(factor_layout)

        # Speedup estimate
        self.tv_speedup_label = QLabel("Expected speedup: ~1.0x")
        self.tv_speedup_label.setStyleSheet("color: #88cc88; font-weight: bold;")
        tv_layout.addWidget(self.tv_speedup_label)

        layout.addWidget(self.tv_container)
        self.tv_container.setEnabled(False)

        # Initialize table with default values
        self._init_freq_table()

        parent_layout.addWidget(frame)

    def _init_freq_table(self) -> None:
        """Initialize frequency table with default values."""
        default_data = [
            (0.0, 80.0),
            (1000.0, 50.0),
            (2500.0, 30.0),
            (5000.0, 20.0),
        ]
        self._set_freq_table_data(default_data)

    def _set_freq_table_data(self, data: list) -> None:
        """Set frequency table data."""
        self.freq_table.blockSignals(True)
        self.freq_table.setRowCount(len(data))
        for i, (t, f) in enumerate(data):
            self.freq_table.setItem(i, 0, QTableWidgetItem(f"{t:.1f}"))
            self.freq_table.setItem(i, 1, QTableWidgetItem(f"{f:.1f}"))
        self.freq_table.blockSignals(False)
        self._update_tv_estimate()

    def _get_freq_table_data(self) -> list:
        """Get frequency table data as list of tuples."""
        data = []
        for row in range(self.freq_table.rowCount()):
            try:
                t_item = self.freq_table.item(row, 0)
                f_item = self.freq_table.item(row, 1)
                if t_item and f_item:
                    t = float(t_item.text())
                    f = float(f_item.text())
                    data.append((t, f))
            except ValueError:
                pass
        return sorted(data, key=lambda x: x[0])

    def _on_tv_toggled(self, checked: bool) -> None:
        """Handle time-variant checkbox toggle."""
        self.tv_container.setEnabled(checked)
        self._update_tv_estimate()
        self._on_param_changed()

    def _on_freq_table_changed(self) -> None:
        """Handle frequency table value change."""
        self._update_tv_estimate()
        self._on_param_changed()

    def _add_freq_row(self) -> None:
        """Add row to frequency table."""
        current_data = self._get_freq_table_data()
        if current_data:
            last_t = current_data[-1][0]
            last_f = current_data[-1][1]
            new_t = last_t + 1000.0
            new_f = max(10.0, last_f - 10.0)
        else:
            new_t = 0.0
            new_f = 80.0

        row = self.freq_table.rowCount()
        self.freq_table.insertRow(row)
        self.freq_table.setItem(row, 0, QTableWidgetItem(f"{new_t:.1f}"))
        self.freq_table.setItem(row, 1, QTableWidgetItem(f"{new_f:.1f}"))
        self._update_tv_estimate()

    def _remove_freq_row(self) -> None:
        """Remove selected row from frequency table."""
        current_row = self.freq_table.currentRow()
        if current_row >= 0 and self.freq_table.rowCount() > 2:
            self.freq_table.removeRow(current_row)
            self._update_tv_estimate()

    def _reset_freq_table(self) -> None:
        """Reset frequency table to defaults."""
        self._init_freq_table()

    def _update_tv_estimate(self) -> None:
        """Update time-variant speedup estimate."""
        if not self.tv_enabled.isChecked():
            self.tv_speedup_label.setText("Time-variant sampling disabled")
            return

        data = self._get_freq_table_data()
        if len(data) < 2:
            self.tv_speedup_label.setText("Need at least 2 points")
            return

        try:
            from pstm.algorithm.time_variant import (
                FrequencyTimeTable, estimate_speedup,
            )

            times = [p[0] for p in data]
            freqs = [p[1] for p in data]
            freq_table = FrequencyTimeTable(times_ms=times, frequencies_hz=freqs)

            # Estimate using output grid time range
            t_min = self.controller.state.output_grid.t_min_ms
            t_max = self.controller.state.output_grid.t_max_ms
            dt = self.controller.state.output_grid.dt_ms

            speedup = estimate_speedup(t_min, t_max, dt, freq_table)

            self.tv_speedup_label.setText(f"Expected speedup: ~{speedup:.1f}x")
            if speedup > 1.5:
                self.tv_speedup_label.setStyleSheet("color: #88cc88; font-weight: bold;")
            else:
                self.tv_speedup_label.setStyleSheet("color: #cccc88; font-weight: bold;")

        except Exception as e:
            self.tv_speedup_label.setText(f"Error: {e}")

    def _create_aperture_diagram(self, parent_layout: QVBoxLayout) -> None:
        """Create aperture visualization."""
        frame, layout = self.create_section("Aperture Diagram")
        
        self.diagram_frame = QFrame()
        self.diagram_frame.setMinimumHeight(250)
        self.diagram_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        diagram_layout = QVBoxLayout(self.diagram_frame)
        self.diagram_label = QLabel("Aperture cone visualization")
        self.diagram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diagram_label.setStyleSheet("color: #666666; border: none; background: transparent;")
        diagram_layout.addWidget(self.diagram_label)
        
        layout.addWidget(self.diagram_frame)
        
        parent_layout.addWidget(frame)
    
    def _create_summary_section(self, parent_layout: QVBoxLayout) -> None:
        """Create parameter summary section."""
        frame, layout = self.create_section("Parameter Summary")
        
        self.summary_label = QLabel(self._get_summary_text())
        self.summary_label.setStyleSheet("""
            QLabel {
                font-family: monospace;
                color: #cccccc;
                border: none;
                background: transparent;
            }
        """)
        layout.addWidget(self.summary_label)
        
        parent_layout.addWidget(frame)
    
    def _get_summary_text(self) -> str:
        """Generate parameter summary."""
        tv_status = "Yes" if self.tv_enabled.isChecked() else "No"
        kernel_idx = self.kernel_type.currentIndex()
        kernel_names = ["Straight Ray", "Curved Ray", "VTI Anisotropic"]
        kernel_name = kernel_names[kernel_idx]

        summary = (
            f"Kernel Type:    {kernel_name}\n"
            f"Max Dip:        {self.max_dip_spin.value():.0f}°\n"
            f"Aperture:       {self.min_aperture_spin.value():.0f} - {self.max_aperture_spin.value():.0f} m\n"
            f"Taper:          {self.taper_type.currentText()} ({self.taper_fraction.value():.0%})\n"
            f"Interpolation:  {self.interp_method.currentText().split(' - ')[0]}\n"
            f"Spreading:      {'Yes' if self.apply_spreading.isChecked() else 'No'}\n"
            f"Obliquity:      {'Yes' if self.apply_obliquity.isChecked() else 'No'}\n"
            f"Anti-aliasing:  {'Yes' if self.enable_aa.isChecked() else 'No'}\n"
            f"Time-Variant:   {tv_status}"
        )
        return summary
    
    def _on_param_changed(self) -> None:
        """Update summary when parameters change."""
        self.summary_label.setText(self._get_summary_text())
        self._update_diagram()
    
    def _update_diagram(self) -> None:
        """Update aperture diagram."""
        # Placeholder - would draw aperture cone
        max_dip = self.max_dip_spin.value()
        self.diagram_label.setText(
            f"Aperture cone: {max_dip:.0f}° max dip\n"
            f"(Visualization placeholder)"
        )
    
    def _show_method_comparison(self) -> None:
        """Show interpolation method comparison dialog."""
        from PyQt6.QtWidgets import QMessageBox
        
        comparison = """
Interpolation Method Comparison:

Method      Points  Quality   Speed      Use Case
------------------------------------------------------
nearest       1     Low       Fastest    Quick preview
linear        2     Medium    Fast       Default, balanced
cubic         4     Good      Medium     Smooth results
sinc4         4     Good      Medium     Standard quality
sinc8         8     High      Medium     RECOMMENDED
sinc16       16     Highest   Slow       Maximum accuracy
lanczos3      6     High      Medium     Sharp transitions
lanczos5     10     Highest   Slow       Sharpest, ringing risk

Recommendation:
• For production: sinc8 or lanczos3
• For speed: linear
• For highest quality: sinc16 or lanczos5
        """
        
        QMessageBox.information(self, "Interpolation Methods", comparison)
    
    def on_enter(self) -> None:
        """Load state into UI."""
        state = self.controller.state.algorithm

        # Load kernel type
        kernel_type_map = {"straight_ray": 0, "curved_ray": 1, "anisotropic_vti": 2}
        self.kernel_type.setCurrentIndex(kernel_type_map.get(state.kernel_type, 0))
        self._on_kernel_type_changed(self.kernel_type.currentIndex())

        self.max_dip_spin.setValue(state.max_dip_degrees)
        self.min_aperture_spin.setValue(state.min_aperture_m)
        self.max_aperture_spin.setValue(state.max_aperture_m)
        self.taper_type.setCurrentText(state.taper_type.title())
        self.taper_fraction.setValue(state.taper_fraction)

        # Find interpolation method
        for i in range(self.interp_method.count()):
            if state.interpolation_method in self.interp_method.itemText(i):
                self.interp_method.setCurrentIndex(i)
                break

        self.apply_spreading.setChecked(state.apply_spreading)
        self.apply_obliquity.setChecked(state.apply_obliquity)
        self.enable_aa.setChecked(state.enable_antialiasing)

        # Load time-variant state
        tv = state.time_variant
        self.tv_enabled.setChecked(tv.enabled)
        self._set_freq_table_data(tv.frequency_table)
        self.min_factor_spin.setValue(tv.min_downsample_factor)
        self.max_factor_spin.setValue(tv.max_downsample_factor)
        self.tv_container.setEnabled(tv.enabled)

        self._on_param_changed()
    
    def on_leave(self) -> None:
        """Save UI to state."""
        state = self.controller.state.algorithm

        # Save kernel type
        kernel_type_map = {0: "straight_ray", 1: "curved_ray", 2: "anisotropic_vti"}
        state.kernel_type = kernel_type_map.get(self.kernel_type.currentIndex(), "straight_ray")

        state.max_dip_degrees = self.max_dip_spin.value()
        state.min_aperture_m = self.min_aperture_spin.value()
        state.max_aperture_m = self.max_aperture_spin.value()
        state.taper_type = self.taper_type.currentText().lower()
        state.taper_fraction = self.taper_fraction.value()
        state.interpolation_method = self.interp_method.currentText().split(" - ")[0]
        state.apply_spreading = self.apply_spreading.isChecked()
        state.apply_obliquity = self.apply_obliquity.isChecked()
        state.enable_antialiasing = self.enable_aa.isChecked()

        # Save time-variant state
        tv = state.time_variant
        tv.enabled = self.tv_enabled.isChecked()
        tv.frequency_table = self._get_freq_table_data()
        tv.min_downsample_factor = self.min_factor_spin.value()
        tv.max_downsample_factor = self.max_factor_spin.value()

        self.controller.state.step_status["algorithm"] = StepStatus.COMPLETE
        self.controller.notify_change()
    
    def validate(self) -> bool:
        """Validate parameters."""
        if self.min_aperture_spin.value() > self.max_aperture_spin.value():
            return False
        if self.max_dip_spin.value() <= 0:
            return False
        return True

    def refresh_from_state(self) -> None:
        """Refresh UI from loaded state."""
        state = self.controller.state.algorithm

        # Refresh kernel type
        kernel_type_map = {"straight_ray": 0, "curved_ray": 1, "anisotropic_vti": 2}
        self.kernel_type.setCurrentIndex(kernel_type_map.get(state.kernel_type, 0))
        self._on_kernel_type_changed(self.kernel_type.currentIndex())

        self.max_dip_spin.setValue(state.max_dip_degrees)
        self.min_aperture_spin.setValue(state.min_aperture_m)
        self.max_aperture_spin.setValue(state.max_aperture_m)
        self.taper_type.setCurrentText(state.taper_type.title())
        self.taper_fraction.setValue(state.taper_fraction)

        for i in range(self.interp_method.count()):
            if state.interpolation_method in self.interp_method.itemText(i):
                self.interp_method.setCurrentIndex(i)
                break

        self.apply_spreading.setChecked(state.apply_spreading)
        self.apply_obliquity.setChecked(state.apply_obliquity)
        self.enable_aa.setChecked(state.enable_antialiasing)

        # Refresh time-variant state
        tv = state.time_variant
        self.tv_enabled.setChecked(tv.enabled)
        self._set_freq_table_data(tv.frequency_table)
        self.min_factor_spin.setValue(tv.min_downsample_factor)
        self.max_factor_spin.setValue(tv.max_downsample_factor)
        self.tv_container.setEnabled(tv.enabled)
