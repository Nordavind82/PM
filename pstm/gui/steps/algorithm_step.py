"""
Step 6: Algorithm Parameters

Configure migration algorithm parameters: aperture, interpolation, amplitude corrections.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QGroupBox,
    QGridLayout, QFormLayout, QSplitter,
)
from PyQt6.QtCore import Qt

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
        
        self._create_aperture_section(left_layout)
        self._create_interpolation_section(left_layout)
        self._create_amplitude_section(left_layout)
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
        return (
            f"Max Dip:        {self.max_dip_spin.value():.0f}°\n"
            f"Aperture:       {self.min_aperture_spin.value():.0f} - {self.max_aperture_spin.value():.0f} m\n"
            f"Taper:          {self.taper_type.currentText()} ({self.taper_fraction.value():.0%})\n"
            f"Interpolation:  {self.interp_method.currentText().split(' - ')[0]}\n"
            f"Spreading:      {'Yes' if self.apply_spreading.isChecked() else 'No'}\n"
            f"Obliquity:      {'Yes' if self.apply_obliquity.isChecked() else 'No'}\n"
            f"Anti-aliasing:  {'Yes' if self.enable_aa.isChecked() else 'No'}"
        )
    
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
        
        self._on_param_changed()
    
    def on_leave(self) -> None:
        """Save UI to state."""
        state = self.controller.state.algorithm
        
        state.max_dip_degrees = self.max_dip_spin.value()
        state.min_aperture_m = self.min_aperture_spin.value()
        state.max_aperture_m = self.max_aperture_spin.value()
        state.taper_type = self.taper_type.currentText().lower()
        state.taper_fraction = self.taper_fraction.value()
        state.interpolation_method = self.interp_method.currentText().split(" - ")[0]
        state.apply_spreading = self.apply_spreading.isChecked()
        state.apply_obliquity = self.apply_obliquity.isChecked()
        state.enable_antialiasing = self.enable_aa.isChecked()
        
        self.controller.state.step_status["algorithm"] = StepStatus.COMPLETE
        self.controller.notify_change()
    
    def validate(self) -> bool:
        """Validate parameters."""
        if self.min_aperture_spin.value() > self.max_aperture_spin.value():
            return False
        if self.max_dip_spin.value() <= 0:
            return False
        return True
