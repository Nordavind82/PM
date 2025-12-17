"""
Step 1: Input Data

Load and validate input seismic dataset (traces + headers).
Includes header column mapping for PSTM requirements.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
)
from PyQt6.QtCore import Qt

from pstm.gui.steps.base import WizardStepWidget
from pstm.gui.state import StepStatus


# Column name lookup patterns for auto-detection
# Maps PSTM field -> list of possible column names (case-insensitive)
COLUMN_PATTERNS = {
    "source_x": [
        "source_x", "src_x", "srcx", "sx", "sou_x", "shot_x", "shotx",
        "source_easting", "src_easting", "shot_easting",
    ],
    "source_y": [
        "source_y", "src_y", "srcy", "sy", "sou_y", "shot_y", "shoty",
        "source_northing", "src_northing", "shot_northing",
    ],
    "receiver_x": [
        "receiver_x", "rec_x", "recx", "rx", "gx", "group_x", "groupx",
        "receiver_easting", "rec_easting", "geophone_x",
    ],
    "receiver_y": [
        "receiver_y", "rec_y", "recy", "ry", "gy", "group_y", "groupy",
        "receiver_northing", "rec_northing", "geophone_y",
    ],
    "midpoint_x": [
        "cdp_x", "cdpx", "cmp_x", "cmpx", "midpoint_x", "mx", "bin_x",
        "cdp_easting", "midpoint_easting",
    ],
    "midpoint_y": [
        "cdp_y", "cdpy", "cmp_y", "cmpy", "midpoint_y", "my", "bin_y",
        "cdp_northing", "midpoint_northing",
    ],
    "offset": [
        "offset", "off", "distance", "src_rec_distance", "sr_distance",
    ],
    "azimuth": [
        "azimuth", "azi", "az", "sr_azim", "src_rec_azimuth", "sr_azimuth",
        "source_receiver_azimuth",
    ],
    "trace_index": [
        "trace_index", "trace_idx", "traceidx", "trace_id", "traceid",
        "trace_no", "traceno", "trace_number", "trace_seq",
        "trace_sequence_file", "trace_sequence_line",
    ],
    "scalar_coord": [
        "scalar_coord", "scalco", "coord_scalar", "scalar",
        "coordinate_scalar", "scal_coord",
    ],
}


def auto_detect_column(columns: list[str], field: str) -> str | None:
    """
    Auto-detect column name for a PSTM field.

    Args:
        columns: Available column names
        field: PSTM field name (e.g., 'source_x')

    Returns:
        Detected column name or None
    """
    patterns = COLUMN_PATTERNS.get(field, [])
    columns_lower = {c.lower(): c for c in columns}

    for pattern in patterns:
        if pattern.lower() in columns_lower:
            return columns_lower[pattern.lower()]

    return None


class InputDataStep(WizardStepWidget):
    """Step 1: Input Data - Load seismic dataset and map headers."""

    @property
    def title(self) -> str:
        return "Input Data"

    def _setup_ui(self) -> None:
        # Dataset Section
        dataset_frame, dataset_layout = self.create_section("Seismic Dataset")

        desc = QLabel(
            "Select a dataset directory containing traces (*.zarr) and headers (*.parquet).\n"
            "Standard format: traces.zarr + headers.parquet in the same directory."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888888; border: none; background: transparent;")
        dataset_layout.addWidget(desc)

        # Path selection
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Dataset:"))
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("/path/to/dataset/directory")
        path_row.addWidget(self.dataset_path)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_dataset)
        path_row.addWidget(self.browse_btn)
        dataset_layout.addLayout(path_row)

        # Dataset info/preview
        self.dataset_info = QFrame()
        self.dataset_info.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        info_layout = QVBoxLayout(self.dataset_info)
        self.info_label = QLabel("No dataset loaded")
        self.info_label.setStyleSheet("color: #888888; border: none; background: transparent;")
        info_layout.addWidget(self.info_label)
        dataset_layout.addWidget(self.dataset_info)

        # Load button
        load_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Dataset")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }
        """)
        self.load_btn.clicked.connect(self._load_data)
        load_row.addWidget(self.load_btn)
        load_row.addStretch()
        dataset_layout.addLayout(load_row)

        self.content_layout.addWidget(dataset_frame)

        # Header Mapping Section
        mapping_frame, mapping_layout = self.create_section("Header Column Mapping")

        mapping_desc = QLabel(
            "Map required PSTM fields to columns in your header file.\n"
            "Required fields are marked with *. Auto-detection runs after loading."
        )
        mapping_desc.setWordWrap(True)
        mapping_desc.setStyleSheet("color: #888888; border: none; background: transparent;")
        mapping_layout.addWidget(mapping_desc)

        # Required fields group
        required_group = QGroupBox("Required Fields")
        required_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #ff6b6b;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        required_layout = QGridLayout(required_group)
        required_layout.setSpacing(8)

        self.col_combos = {}
        required_fields = [
            ("Source X *", "source_x", "X coordinate of source/shot point"),
            ("Source Y *", "source_y", "Y coordinate of source/shot point"),
            ("Receiver X *", "receiver_x", "X coordinate of receiver/geophone"),
            ("Receiver Y *", "receiver_y", "Y coordinate of receiver/geophone"),
            ("Trace Index *", "trace_index", "Unique trace identifier (0-based index)"),
        ]

        for i, (label_text, key, tooltip) in enumerate(required_fields):
            row, col = divmod(i, 2)

            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #ff6b6b; border: none; background: transparent;")
            lbl.setToolTip(tooltip)
            required_layout.addWidget(lbl, row, col * 2)

            combo = QComboBox()
            combo.setEditable(False)
            combo.setMinimumWidth(180)
            combo.setToolTip(tooltip)
            combo.addItem("-- Not Set --", "")
            self.col_combos[key] = combo
            required_layout.addWidget(combo, row, col * 2 + 1)

        mapping_layout.addWidget(required_group)

        # Optional fields group
        optional_group = QGroupBox("Optional Fields")
        optional_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #4ecdc4;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        optional_layout = QGridLayout(optional_group)
        optional_layout.setSpacing(8)

        optional_fields = [
            ("Midpoint X", "midpoint_x", "X coordinate of CDP/CMP (computed if not set)"),
            ("Midpoint Y", "midpoint_y", "Y coordinate of CDP/CMP (computed if not set)"),
            ("Offset", "offset", "Source-receiver distance (computed if not set)"),
            ("Azimuth", "azimuth", "Source-receiver azimuth in degrees"),
        ]

        for i, (label_text, key, tooltip) in enumerate(optional_fields):
            row, col = divmod(i, 2)

            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #4ecdc4; border: none; background: transparent;")
            lbl.setToolTip(tooltip)
            optional_layout.addWidget(lbl, row, col * 2)

            combo = QComboBox()
            combo.setEditable(False)
            combo.setMinimumWidth(180)
            combo.setToolTip(tooltip)
            combo.addItem("-- Not Set --", "")
            self.col_combos[key] = combo
            optional_layout.addWidget(combo, row, col * 2 + 1)

        mapping_layout.addWidget(optional_group)

        # Auto-detect button row
        btn_row = QHBoxLayout()
        self.auto_detect_btn = QPushButton("Auto-Detect Columns")
        self.auto_detect_btn.clicked.connect(self._auto_detect_columns)
        self.auto_detect_btn.setEnabled(False)
        btn_row.addWidget(self.auto_detect_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_mappings)
        btn_row.addWidget(self.clear_btn)

        btn_row.addStretch()
        mapping_layout.addLayout(btn_row)

        # Validation status
        self.validation_frame = QFrame()
        self.validation_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        val_layout = QVBoxLayout(self.validation_frame)
        self.validation_label = QLabel("Load a dataset to configure column mapping")
        self.validation_label.setStyleSheet("color: #888888; border: none; background: transparent;")
        val_layout.addWidget(self.validation_label)
        mapping_layout.addWidget(self.validation_frame)

        self.content_layout.addWidget(mapping_frame)
        self.content_layout.addStretch()

        # Store available columns
        self._available_columns: list[str] = []

    def _browse_dataset(self) -> None:
        """Browse for dataset directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if path:
            self.dataset_path.setText(path)
            self._preview_dataset(path)

    def _preview_dataset(self, path: str) -> None:
        """Preview dataset contents."""
        base = Path(path)
        if not base.exists():
            self.info_label.setText("Directory does not exist")
            self.info_label.setStyleSheet("color: #f44336; border: none; background: transparent;")
            return

        # Find files
        zarrs = list(base.glob("*.zarr"))
        parquets = list(base.glob("*.parquet"))

        info_lines = [f"Directory: {base.name}"]

        if zarrs:
            info_lines.append(f"Traces: {', '.join(z.name for z in zarrs)}")
        else:
            info_lines.append("Traces: No .zarr files found")

        if parquets:
            info_lines.append(f"Headers: {', '.join(p.name for p in parquets)}")
        else:
            info_lines.append("Headers: No .parquet files found")

        if zarrs and parquets:
            self.info_label.setStyleSheet("color: #4caf50; border: none; background: transparent;")
        else:
            self.info_label.setStyleSheet("color: #ff9800; border: none; background: transparent;")

        self.info_label.setText("\n".join(info_lines))

    def _load_data(self) -> None:
        """Load dataset and populate column combos."""
        state = self.controller.state.input_data
        state.dataset_path = self.dataset_path.text()

        # Load data
        success, message = self.controller.load_input_data()

        if success:
            # Show scalar info if detected
            scalar_info = ""
            if state.coord_scalar_value != 1:
                scalar_info = f"\n⚠ Coordinate scalar detected: {state.coord_scalar_value} (will be applied)"
            self.info_label.setText(f"✓ {message}{scalar_info}")
            self.info_label.setStyleSheet("color: #4caf50; border: none; background: transparent;")

            # Get available columns from headers
            if self.controller.headers_df is not None:
                self._available_columns = sorted(self.controller.headers_df.columns.tolist())
                self._populate_column_combos()
                self._auto_detect_columns()
                self.auto_detect_btn.setEnabled(True)

            self._update_validation()
        else:
            self.info_label.setText(f"✗ {message}")
            self.info_label.setStyleSheet("color: #f44336; border: none; background: transparent;")
            self.validation_label.setText(f"✗ Failed to load: {message}")
            self.validation_label.setStyleSheet("color: #f44336; border: none; background: transparent;")

    def _populate_column_combos(self) -> None:
        """Populate all column combos with available columns."""
        for key, combo in self.col_combos.items():
            current = combo.currentData()
            combo.clear()
            combo.addItem("-- Not Set --", "")
            for col in self._available_columns:
                combo.addItem(col, col)

            # Restore previous selection if valid
            if current and current in self._available_columns:
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        # Connect change signals for validation
        for combo in self.col_combos.values():
            combo.currentIndexChanged.connect(self._on_mapping_changed)

    def _auto_detect_columns(self) -> None:
        """Auto-detect column mappings based on naming patterns."""
        if not self._available_columns:
            return

        detected_count = 0
        for key, combo in self.col_combos.items():
            detected = auto_detect_column(self._available_columns, key)
            if detected:
                idx = combo.findData(detected)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    detected_count += 1

        self._update_validation()

        # Show detection result
        if detected_count > 0:
            self.validation_label.setText(
                f"Auto-detected {detected_count} column(s). Please verify mappings."
            )

    def _clear_mappings(self) -> None:
        """Clear all column mappings."""
        for combo in self.col_combos.values():
            combo.setCurrentIndex(0)
        self._update_validation()

    def _on_mapping_changed(self) -> None:
        """Handle column mapping change."""
        self._save_mappings_to_state()
        self._update_validation()

    def _save_mappings_to_state(self) -> None:
        """Save current mappings to state."""
        state = self.controller.state.input_data

        state.col_source_x = self.col_combos["source_x"].currentData() or ""
        state.col_source_y = self.col_combos["source_y"].currentData() or ""
        state.col_receiver_x = self.col_combos["receiver_x"].currentData() or ""
        state.col_receiver_y = self.col_combos["receiver_y"].currentData() or ""
        state.col_midpoint_x = self.col_combos["midpoint_x"].currentData() or ""
        state.col_midpoint_y = self.col_combos["midpoint_y"].currentData() or ""
        state.col_offset = self.col_combos["offset"].currentData() or ""
        state.col_azimuth = self.col_combos["azimuth"].currentData() or ""

        # Store trace index column name
        if not hasattr(state, 'col_trace_index'):
            # Add dynamically if not in dataclass
            pass
        state.col_trace_index = self.col_combos["trace_index"].currentData() or ""

    def _update_validation(self) -> None:
        """Update validation status display."""
        required_fields = ["source_x", "source_y", "receiver_x", "receiver_y", "trace_index"]
        missing = []
        mapped = []

        for field in required_fields:
            combo = self.col_combos[field]
            if combo.currentData():
                mapped.append(field)
            else:
                missing.append(field)

        if not missing:
            self.validation_label.setText(
                f"✓ All required fields mapped ({len(mapped)}/5)"
            )
            self.validation_label.setStyleSheet("color: #4caf50; border: none; background: transparent;")
            self.controller.state.step_status["input"] = StepStatus.COMPLETE
        else:
            missing_names = [f.replace("_", " ").title() for f in missing]
            self.validation_label.setText(
                f"✗ Missing required fields: {', '.join(missing_names)}"
            )
            self.validation_label.setStyleSheet("color: #f44336; border: none; background: transparent;")
            self.controller.state.step_status["input"] = StepStatus.IN_PROGRESS

    def on_enter(self) -> None:
        """Called when navigating to this step."""
        state = self.controller.state.input_data

        if state.dataset_path:
            self.dataset_path.setText(state.dataset_path)
            self._preview_dataset(state.dataset_path)

        # If data is already loaded, populate combos
        if state.is_loaded and self.controller.headers_df is not None:
            self._available_columns = sorted(self.controller.headers_df.columns.tolist())
            self._populate_column_combos()
            self._restore_mappings_from_state()
            self.auto_detect_btn.setEnabled(True)
            self._update_validation()

    def _restore_mappings_from_state(self) -> None:
        """Restore column mappings from state."""
        state = self.controller.state.input_data

        mappings = {
            "source_x": state.col_source_x,
            "source_y": state.col_source_y,
            "receiver_x": state.col_receiver_x,
            "receiver_y": state.col_receiver_y,
            "midpoint_x": state.col_midpoint_x,
            "midpoint_y": state.col_midpoint_y,
            "offset": state.col_offset,
            "azimuth": state.col_azimuth,
            "trace_index": getattr(state, 'col_trace_index', ''),
        }

        for key, value in mappings.items():
            if value and key in self.col_combos:
                combo = self.col_combos[key]
                idx = combo.findData(value)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def on_leave(self) -> None:
        """Called when leaving this step."""
        self._save_mappings_to_state()

    def validate(self) -> bool:
        """Validate this step is complete."""
        state = self.controller.state.input_data
        if not state.is_loaded:
            return False

        # Check required fields are mapped
        required = [
            state.col_source_x,
            state.col_source_y,
            state.col_receiver_x,
            state.col_receiver_y,
            getattr(state, 'col_trace_index', ''),
        ]
        return all(required)

    def refresh_from_state(self) -> None:
        """Refresh UI from loaded state."""
        state = self.controller.state.input_data

        self.dataset_path.setText(state.dataset_path or "")
        if state.dataset_path:
            self._preview_dataset(state.dataset_path)

        if state.is_loaded and self.controller.headers_df is not None:
            self._available_columns = sorted(self.controller.headers_df.columns.tolist())
            self._populate_column_combos()
            self._restore_mappings_from_state()
            self.auto_detect_btn.setEnabled(True)
            self._update_validation()
