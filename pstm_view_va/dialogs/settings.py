"""Settings dialogs for velocity analysis."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QDialogButtonBox, QLabel, QPushButton, QFileDialog, QLineEdit
)
from pathlib import Path


class VelocityGridDialog(QDialog):
    """Dialog for configuring velocity output grid and loading velocity model."""

    def __init__(self, parent=None, current_settings: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Load Velocity Model")
        self.setMinimumWidth(400)

        self.velocity_path = None
        self.settings = current_settings or {}

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # File selection
        file_group = QGroupBox("Velocity File")
        file_layout = QVBoxLayout(file_group)

        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select velocity file...")
        self.path_edit.setReadOnly(True)
        path_layout.addWidget(self.path_edit)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_file)
        path_layout.addWidget(self.browse_btn)
        file_layout.addLayout(path_layout)

        self.file_info_label = QLabel("")
        self.file_info_label.setStyleSheet("color: gray; font-size: 10px;")
        file_layout.addWidget(self.file_info_label)

        layout.addWidget(file_group)

        # Output grid configuration
        grid_group = QGroupBox("Output Grid")
        grid_layout = QFormLayout(grid_group)

        # Inline range
        il_layout = QHBoxLayout()
        self.il_start_spin = QSpinBox()
        self.il_start_spin.setRange(0, 10000)
        self.il_start_spin.setValue(self.settings.get('il_start', 0))
        il_layout.addWidget(self.il_start_spin)
        il_layout.addWidget(QLabel("to"))
        self.il_end_spin = QSpinBox()
        self.il_end_spin.setRange(0, 10000)
        self.il_end_spin.setValue(self.settings.get('il_end', 100))
        il_layout.addWidget(self.il_end_spin)
        il_layout.addWidget(QLabel("step"))
        self.il_step_spin = QSpinBox()
        self.il_step_spin.setRange(1, 100)
        self.il_step_spin.setValue(self.settings.get('il_step', 10))
        il_layout.addWidget(self.il_step_spin)
        grid_layout.addRow("Inline:", il_layout)

        # Crossline range
        xl_layout = QHBoxLayout()
        self.xl_start_spin = QSpinBox()
        self.xl_start_spin.setRange(0, 10000)
        self.xl_start_spin.setValue(self.settings.get('xl_start', 0))
        xl_layout.addWidget(self.xl_start_spin)
        xl_layout.addWidget(QLabel("to"))
        self.xl_end_spin = QSpinBox()
        self.xl_end_spin.setRange(0, 10000)
        self.xl_end_spin.setValue(self.settings.get('xl_end', 100))
        xl_layout.addWidget(self.xl_end_spin)
        xl_layout.addWidget(QLabel("step"))
        self.xl_step_spin = QSpinBox()
        self.xl_step_spin.setRange(1, 100)
        self.xl_step_spin.setValue(self.settings.get('xl_step', 10))
        xl_layout.addWidget(self.xl_step_spin)
        grid_layout.addRow("Crossline:", xl_layout)

        # Output sample rate
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["10 ms", "100 ms", "1000 ms"])
        current_rate = self.settings.get('output_dt_ms', 100)
        if current_rate == 10:
            self.sample_rate_combo.setCurrentIndex(0)
        elif current_rate == 1000:
            self.sample_rate_combo.setCurrentIndex(2)
        else:
            self.sample_rate_combo.setCurrentIndex(1)  # Default 100ms
        self.sample_rate_combo.setToolTip("Output velocity sample rate")
        grid_layout.addRow("Sample rate:", self.sample_rate_combo)

        # Grid info
        self.grid_info_label = QLabel("")
        self.grid_info_label.setStyleSheet("color: #4a9eff; font-size: 10px;")
        grid_layout.addRow("Grid points:", self.grid_info_label)

        # Connect signals to update grid info
        for spin in [self.il_start_spin, self.il_end_spin, self.il_step_spin,
                     self.xl_start_spin, self.xl_end_spin, self.xl_step_spin]:
            spin.valueChanged.connect(self._update_grid_info)

        self._update_grid_info()

        layout.addWidget(grid_group)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_file(self):
        """Browse for velocity file."""
        # Try directory first (zarr)
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
            self.velocity_path = path
            self.path_edit.setText(path)
            self._load_file_info(path)

    def _load_file_info(self, path: str):
        """Load and display file info."""
        try:
            from ..io import load_velocity_model
            vel_model, metadata = load_velocity_model(path)
            if vel_model is not None:
                shape_str = f"Shape: {vel_model.shape}"
                self.file_info_label.setText(shape_str)
            else:
                self.file_info_label.setText("Failed to load")
        except Exception as e:
            self.file_info_label.setText(f"Error: {str(e)[:50]}")

    def _update_grid_info(self):
        """Update grid info label."""
        il_start = self.il_start_spin.value()
        il_end = self.il_end_spin.value()
        il_step = self.il_step_spin.value()
        xl_start = self.xl_start_spin.value()
        xl_end = self.xl_end_spin.value()
        xl_step = self.xl_step_spin.value()

        n_il = max(1, (il_end - il_start) // il_step + 1)
        n_xl = max(1, (xl_end - xl_start) // xl_step + 1)
        total = n_il * n_xl

        self.grid_info_label.setText(f"{n_il} IL x {n_xl} XL = {total} locations")

    def get_settings(self) -> dict:
        """Return the grid settings."""
        # Parse sample rate from combo box
        rate_text = self.sample_rate_combo.currentText()
        if "10" in rate_text:
            output_dt_ms = 10
        elif "1000" in rate_text:
            output_dt_ms = 1000
        else:
            output_dt_ms = 100

        return {
            'velocity_path': self.velocity_path,
            'il_start': self.il_start_spin.value(),
            'il_end': self.il_end_spin.value(),
            'il_step': self.il_step_spin.value(),
            'xl_start': self.xl_start_spin.value(),
            'xl_end': self.xl_end_spin.value(),
            'xl_step': self.xl_step_spin.value(),
            'output_dt_ms': output_dt_ms,
        }


class SemblanceSettingsDialog(QDialog):
    """Dialog for configuring semblance calculation and display parameters."""

    # Colormaps for seismic amplitude data (centered at zero, +/- values)
    SEISMIC_COLORMAPS = ["Gray", "Seismic (BWR)", "Seismic (RWB)", "Coolwarm", "Bone"]

    # Colormaps for positive-only data (semblance, velocity, attributes)
    POSITIVE_COLORMAPS = ["Hot", "Plasma", "Inferno", "Magma", "Viridis", "Turbo", "Jet", "Cividis"]

    # All colormaps for backwards compatibility
    COLORMAPS = SEISMIC_COLORMAPS + POSITIVE_COLORMAPS

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
            # Inverse NMO for semblance (uses initial loaded velocity only)
            'use_inverse_nmo': False,
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

        # Inverse NMO for semblance calculation
        inv_nmo_group = QGroupBox("Inverse NMO (for Semblance)")
        inv_nmo_layout = QFormLayout(inv_nmo_group)

        self.inv_nmo_check = QCheckBox("Apply Inverse NMO before semblance")
        self.inv_nmo_check.setChecked(self.settings.get('use_inverse_nmo', False))
        self.inv_nmo_check.setToolTip(
            "Remove residual moveout before semblance calculation.\n"
            "Uses the initially loaded velocity model (not edited picks)."
        )
        inv_nmo_layout.addRow(self.inv_nmo_check)

        self.inv_nmo_info_label = QLabel("Uses initially loaded velocity model only")
        self.inv_nmo_info_label.setStyleSheet("color: #4a9eff; font-size: 10px;")
        inv_nmo_layout.addRow(self.inv_nmo_info_label)

        layout.addWidget(inv_nmo_group)

        # Gather display settings
        gather_group = QGroupBox("Gather Display")
        gather_layout = QFormLayout(gather_group)

        self.gather_cmap_combo = QComboBox()
        # Use seismic colormaps for gathers (amplitude data with +/- values)
        self.gather_cmap_combo.addItems(self.SEISMIC_COLORMAPS)
        current_cmap = self.settings.get('gather_colormap', 'Seismic (BWR)')
        if current_cmap in self.SEISMIC_COLORMAPS:
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
        # Use positive colormaps for semblance (0-1 range, high contrast needed)
        self.sem_cmap_combo.addItems(self.POSITIVE_COLORMAPS)
        current_cmap = self.settings.get('semblance_colormap', 'Hot')
        if current_cmap in self.POSITIVE_COLORMAPS:
            self.sem_cmap_combo.setCurrentText(current_cmap)
        elif current_cmap == 'Viridis':  # Handle old default
            self.sem_cmap_combo.setCurrentText('Viridis')
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
        # Use positive colormaps for velocity (always positive values)
        self.vel_cmap_combo.addItems(self.POSITIVE_COLORMAPS)
        current_cmap = self.settings.get('velocity_colormap', 'Turbo')
        if current_cmap in self.POSITIVE_COLORMAPS:
            self.vel_cmap_combo.setCurrentText(current_cmap)
        elif current_cmap == 'Viridis':  # Handle old default
            self.vel_cmap_combo.setCurrentText('Viridis')
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
            # Inverse NMO for semblance (uses initial velocity only)
            'use_inverse_nmo': self.inv_nmo_check.isChecked(),
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
