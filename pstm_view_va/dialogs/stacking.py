"""Stacking dialog for configuring stack parameters."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QDialogButtonBox, QLabel, QPushButton, QLineEdit, QRadioButton,
    QButtonGroup
)
from PyQt6.QtCore import Qt


class StackingDialog(QDialog):
    """Dialog for configuring stacking parameters."""

    def __init__(self, parent=None, current_settings: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Create Stack")
        self.setMinimumWidth(450)

        self.settings = current_settings or {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Data selection
        data_group = QGroupBox("Data to Stack")
        data_layout = QVBoxLayout(data_group)

        self.data_btn_group = QButtonGroup(self)

        self.inline_radio = QRadioButton("Current inline (all crosslines)")
        self.inline_radio.setChecked(True)
        self.data_btn_group.addButton(self.inline_radio, 0)
        data_layout.addWidget(self.inline_radio)

        self.crossline_radio = QRadioButton("Current crossline (all inlines)")
        self.data_btn_group.addButton(self.crossline_radio, 1)
        data_layout.addWidget(self.crossline_radio)

        self.full_volume_radio = QRadioButton("Full volume (all IL/XL)")
        self.data_btn_group.addButton(self.full_volume_radio, 2)
        data_layout.addWidget(self.full_volume_radio)

        # Current position info
        current_il = self.settings.get('current_il', 0)
        current_xl = self.settings.get('current_xl', 0)
        self.position_label = QLabel(f"Current position: IL={current_il}, XL={current_xl}")
        self.position_label.setStyleSheet("color: gray; font-size: 10px;")
        data_layout.addWidget(self.position_label)

        layout.addWidget(data_group)

        # Velocity selection
        vel_group = QGroupBox("Velocity Model")
        vel_layout = QVBoxLayout(vel_group)

        self.vel_btn_group = QButtonGroup(self)

        self.initial_vel_radio = QRadioButton("Initial velocity (loaded from file)")
        self.initial_vel_radio.setChecked(True)
        self.vel_btn_group.addButton(self.initial_vel_radio, 0)
        vel_layout.addWidget(self.initial_vel_radio)

        self.edited_vel_radio = QRadioButton("Edited velocity (current picks)")
        self.vel_btn_group.addButton(self.edited_vel_radio, 1)
        vel_layout.addWidget(self.edited_vel_radio)

        layout.addWidget(vel_group)

        # Processing parameters (from active panel)
        proc_group = QGroupBox("Processing Parameters")
        proc_layout = QFormLayout(proc_group)

        # Mute settings
        self.top_mute_check = QCheckBox()
        self.top_mute_check.setChecked(self.settings.get('top_mute_enabled', False))
        self.vtop_spin = QSpinBox()
        self.vtop_spin.setRange(500, 10000)
        self.vtop_spin.setValue(int(self.settings.get('v_top', 1500)))
        self.vtop_spin.setSuffix(" m/s")
        mute_top_layout = QHBoxLayout()
        mute_top_layout.addWidget(self.top_mute_check)
        mute_top_layout.addWidget(self.vtop_spin)
        proc_layout.addRow("Top mute:", mute_top_layout)

        self.bottom_mute_check = QCheckBox()
        self.bottom_mute_check.setChecked(self.settings.get('bottom_mute_enabled', False))
        self.vbot_spin = QSpinBox()
        self.vbot_spin.setRange(500, 10000)
        self.vbot_spin.setValue(int(self.settings.get('v_bottom', 5000)))
        self.vbot_spin.setSuffix(" m/s")
        mute_bot_layout = QHBoxLayout()
        mute_bot_layout.addWidget(self.bottom_mute_check)
        mute_bot_layout.addWidget(self.vbot_spin)
        proc_layout.addRow("Bottom mute:", mute_bot_layout)

        # Stretch mute
        self.stretch_spin = QSpinBox()
        self.stretch_spin.setRange(0, 100)
        self.stretch_spin.setValue(int(self.settings.get('stretch_percent', 30)))
        self.stretch_spin.setSuffix(" %")
        proc_layout.addRow("Stretch mute:", self.stretch_spin)

        # Bandpass filter
        self.bp_check = QCheckBox("Apply bandpass filter")
        self.bp_check.setChecked(self.settings.get('apply_bandpass', False))
        proc_layout.addRow("", self.bp_check)

        bp_layout = QHBoxLayout()
        self.f_low_spin = QDoubleSpinBox()
        self.f_low_spin.setRange(0, 100)
        self.f_low_spin.setValue(self.settings.get('f_low', 5))
        self.f_low_spin.setSuffix(" Hz")
        bp_layout.addWidget(self.f_low_spin)
        bp_layout.addWidget(QLabel("-"))
        self.f_high_spin = QDoubleSpinBox()
        self.f_high_spin.setRange(0, 200)
        self.f_high_spin.setValue(self.settings.get('f_high', 80))
        self.f_high_spin.setSuffix(" Hz")
        bp_layout.addWidget(self.f_high_spin)
        proc_layout.addRow("Frequency:", bp_layout)

        # AGC
        self.agc_check = QCheckBox("Apply AGC")
        self.agc_check.setChecked(self.settings.get('apply_agc', False))
        proc_layout.addRow("", self.agc_check)

        self.agc_spin = QSpinBox()
        self.agc_spin.setRange(50, 1000)
        self.agc_spin.setValue(int(self.settings.get('agc_window', 250)))
        self.agc_spin.setSuffix(" ms")
        proc_layout.addRow("AGC window:", self.agc_spin)

        layout.addWidget(proc_group)

        # Output naming
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        self.name_btn_group = QButtonGroup(self)

        self.name_initial_radio = QRadioButton("stack_initial")
        self.name_btn_group.addButton(self.name_initial_radio, 0)
        output_layout.addWidget(self.name_initial_radio)

        self.name_current_radio = QRadioButton("stack_edited")
        self.name_btn_group.addButton(self.name_current_radio, 1)
        output_layout.addWidget(self.name_current_radio)

        custom_layout = QHBoxLayout()
        self.name_custom_radio = QRadioButton("Custom:")
        self.name_custom_radio.setChecked(True)
        self.name_btn_group.addButton(self.name_custom_radio, 2)
        custom_layout.addWidget(self.name_custom_radio)

        self.custom_name_edit = QLineEdit()
        self.custom_name_edit.setText(self.settings.get('default_name', 'stack'))
        self.custom_name_edit.setPlaceholderText("Enter stack name...")
        custom_layout.addWidget(self.custom_name_edit)
        output_layout.addLayout(custom_layout)

        layout.addWidget(output_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_settings(self) -> dict:
        """Return configured settings."""
        # Determine data scope
        if self.inline_radio.isChecked():
            data_scope = 'inline'
        elif self.crossline_radio.isChecked():
            data_scope = 'crossline'
        else:
            data_scope = 'full'

        # Determine velocity type
        use_initial_velocity = self.initial_vel_radio.isChecked()

        # Determine output name
        if self.name_initial_radio.isChecked():
            output_name = 'stack_initial'
        elif self.name_current_radio.isChecked():
            output_name = 'stack_edited'
        else:
            output_name = self.custom_name_edit.text() or 'stack'

        return {
            'data_scope': data_scope,
            'use_initial_velocity': use_initial_velocity,
            'output_name': output_name,
            'top_mute_enabled': self.top_mute_check.isChecked(),
            'v_top': self.vtop_spin.value(),
            'bottom_mute_enabled': self.bottom_mute_check.isChecked(),
            'v_bottom': self.vbot_spin.value(),
            'stretch_percent': self.stretch_spin.value(),
            'apply_bandpass': self.bp_check.isChecked(),
            'f_low': self.f_low_spin.value(),
            'f_high': self.f_high_spin.value(),
            'apply_agc': self.agc_check.isChecked(),
            'agc_window': self.agc_spin.value(),
        }
