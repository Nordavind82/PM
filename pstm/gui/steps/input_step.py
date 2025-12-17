"""
Step 1: Input Data

Load and validate input seismic traces and headers.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QFrame, QGridLayout,
)
from PyQt6.QtCore import Qt

from pstm.gui.steps.base import WizardStepWidget
from pstm.gui.state import StepStatus


class InputDataStep(WizardStepWidget):
    """Step 1: Input Data - Load traces and headers."""
    
    @property
    def title(self) -> str:
        return "Input Data"
    
    def _setup_ui(self) -> None:
        # Trace Data Section
        trace_frame, trace_layout = self.create_section("Trace Data")
        
        # Format selection
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.trace_format = QComboBox()
        self.trace_format.addItems(["Zarr", "SEG-Y", "NumPy"])
        self.trace_format.setCurrentText("Zarr")
        format_row.addWidget(self.trace_format)
        format_row.addStretch()
        trace_layout.addLayout(format_row)
        
        # Path selection
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path:"))
        self.trace_path = QLineEdit()
        self.trace_path.setPlaceholderText("/path/to/traces.zarr")
        path_row.addWidget(self.trace_path)
        self.trace_browse = QPushButton("Browse...")
        self.trace_browse.clicked.connect(self._browse_traces)
        path_row.addWidget(self.trace_browse)
        trace_layout.addLayout(path_row)
        
        # Preview box
        self.trace_preview = QFrame()
        self.trace_preview.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        preview_layout = QVBoxLayout(self.trace_preview)
        self.trace_info = QLabel("No data loaded")
        self.trace_info.setStyleSheet("color: #888888; border: none; background: transparent;")
        preview_layout.addWidget(self.trace_info)
        trace_layout.addWidget(self.trace_preview)
        
        self.content_layout.addWidget(trace_frame)
        
        # Header Data Section
        header_frame, header_layout = self.create_section("Header Data")
        
        # Format
        hformat_row = QHBoxLayout()
        hformat_row.addWidget(QLabel("Format:"))
        self.header_format = QComboBox()
        self.header_format.addItems(["Parquet", "CSV", "SEG-Y embedded"])
        self.header_format.setCurrentText("Parquet")
        hformat_row.addWidget(self.header_format)
        hformat_row.addStretch()
        header_layout.addLayout(hformat_row)
        
        # Path
        hpath_row = QHBoxLayout()
        hpath_row.addWidget(QLabel("Path:"))
        self.header_path = QLineEdit()
        self.header_path.setPlaceholderText("/path/to/headers.parquet")
        hpath_row.addWidget(self.header_path)
        self.header_browse = QPushButton("Browse...")
        self.header_browse.clicked.connect(self._browse_headers)
        hpath_row.addWidget(self.header_browse)
        header_layout.addLayout(hpath_row)
        
        self.content_layout.addWidget(header_frame)
        
        # Column Mapping Section
        mapping_frame, mapping_layout = self.create_section("Column Mapping")
        
        grid = QGridLayout()
        grid.setSpacing(10)
        
        self.col_combos = {}
        columns = [
            ("Source X:", "source_x"),
            ("Source Y:", "source_y"),
            ("Receiver X:", "receiver_x"),
            ("Receiver Y:", "receiver_y"),
            ("Midpoint X:", "midpoint_x"),
            ("Midpoint Y:", "midpoint_y"),
            ("Offset:", "offset"),
            ("Azimuth:", "azimuth"),
        ]
        
        for i, (label, key) in enumerate(columns):
            row, col = divmod(i, 2)
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #cccccc; border: none; background: transparent;")
            grid.addWidget(lbl, row, col * 2)
            
            combo = QComboBox()
            combo.setEditable(True)
            combo.setMinimumWidth(150)
            self.col_combos[key] = combo
            grid.addWidget(combo, row, col * 2 + 1)
        
        mapping_layout.addLayout(grid)
        
        # Auto-detect button
        btn_row = QHBoxLayout()
        self.auto_detect_btn = QPushButton("Auto-Detect Columns")
        self.auto_detect_btn.clicked.connect(self._auto_detect_columns)
        btn_row.addWidget(self.auto_detect_btn)
        btn_row.addStretch()
        mapping_layout.addLayout(btn_row)
        
        self.content_layout.addWidget(mapping_frame)
        
        # Validation Section
        valid_frame, valid_layout = self.create_section("Validation Status")
        
        self.validation_label = QLabel("Load data to validate")
        self.validation_label.setStyleSheet("color: #888888; border: none; background: transparent;")
        valid_layout.addWidget(self.validation_label)
        
        # Load button
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Data")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
            }
        """)
        self.load_btn.clicked.connect(self._load_data)
        btn_row.addWidget(self.load_btn)
        btn_row.addStretch()
        valid_layout.addLayout(btn_row)
        
        self.content_layout.addWidget(valid_frame)
        self.content_layout.addStretch()
    
    def _browse_traces(self) -> None:
        fmt = self.trace_format.currentText().lower()
        if fmt == "zarr":
            path = QFileDialog.getExistingDirectory(self, "Select Zarr Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Trace File")
        
        if path:
            self.trace_path.setText(path)
    
    def _browse_headers(self) -> None:
        fmt = self.header_format.currentText().lower()
        if fmt == "parquet":
            filter_str = "Parquet Files (*.parquet);;All Files (*)"
        elif fmt == "csv":
            filter_str = "CSV Files (*.csv);;All Files (*)"
        else:
            filter_str = "All Files (*)"
        
        path, _ = QFileDialog.getOpenFileName(self, "Select Header File", "", filter_str)
        
        if path:
            self.header_path.setText(path)
    
    def _auto_detect_columns(self) -> None:
        mapping = self.controller.auto_detect_columns()
        
        for key, combo in self.col_combos.items():
            if key in mapping and mapping[key]:
                idx = combo.findText(mapping[key])
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                else:
                    combo.setCurrentText(mapping[key])
    
    def _load_data(self) -> None:
        state = self.controller.state.input_data
        
        # Update state from UI
        state.traces_path = self.trace_path.text()
        state.traces_format = self.trace_format.currentText().lower()
        state.headers_path = self.header_path.text()
        state.headers_format = self.header_format.currentText().lower()
        
        # Column mapping
        state.col_source_x = self.col_combos["source_x"].currentText()
        state.col_source_y = self.col_combos["source_y"].currentText()
        state.col_receiver_x = self.col_combos["receiver_x"].currentText()
        state.col_receiver_y = self.col_combos["receiver_y"].currentText()
        state.col_midpoint_x = self.col_combos["midpoint_x"].currentText()
        state.col_midpoint_y = self.col_combos["midpoint_y"].currentText()
        state.col_offset = self.col_combos["offset"].currentText()
        state.col_azimuth = self.col_combos["azimuth"].currentText()
        
        # Load data
        success, message = self.controller.load_input_data()
        
        if success:
            self.trace_info.setText(f"✓ {message}")
            self.trace_info.setStyleSheet("color: #4caf50; border: none; background: transparent;")
            self.validation_label.setText(f"✓ Data loaded successfully\n{message}")
            self.validation_label.setStyleSheet("color: #4caf50; border: none; background: transparent;")
            
            # Update column combos with available columns
            if self.controller.headers_df is not None:
                columns = list(self.controller.headers_df.columns)
                for combo in self.col_combos.values():
                    current = combo.currentText()
                    combo.clear()
                    combo.addItems(columns)
                    if current in columns:
                        combo.setCurrentText(current)
                
                # Auto-detect if not already set
                self._auto_detect_columns()
        else:
            self.trace_info.setText(f"✗ {message}")
            self.trace_info.setStyleSheet("color: #f44336; border: none; background: transparent;")
            self.validation_label.setText(f"✗ Failed to load: {message}")
            self.validation_label.setStyleSheet("color: #f44336; border: none; background: transparent;")
    
    def on_enter(self) -> None:
        # Load state into UI
        state = self.controller.state.input_data
        
        if state.traces_path:
            self.trace_path.setText(state.traces_path)
        if state.headers_path:
            self.header_path.setText(state.headers_path)
        
        # Set column mappings
        defaults = {
            "source_x": state.col_source_x,
            "source_y": state.col_source_y,
            "receiver_x": state.col_receiver_x,
            "receiver_y": state.col_receiver_y,
            "midpoint_x": state.col_midpoint_x,
            "midpoint_y": state.col_midpoint_y,
            "offset": state.col_offset,
            "azimuth": state.col_azimuth,
        }
        
        for key, value in defaults.items():
            if value:
                self.col_combos[key].setCurrentText(value)
    
    def on_leave(self) -> None:
        # Save state from UI
        state = self.controller.state.input_data
        state.traces_path = self.trace_path.text()
        state.headers_path = self.header_path.text()
        state.col_source_x = self.col_combos["source_x"].currentText()
        state.col_source_y = self.col_combos["source_y"].currentText()
        state.col_receiver_x = self.col_combos["receiver_x"].currentText()
        state.col_receiver_y = self.col_combos["receiver_y"].currentText()
        state.col_midpoint_x = self.col_combos["midpoint_x"].currentText()
        state.col_midpoint_y = self.col_combos["midpoint_y"].currentText()
        state.col_offset = self.col_combos["offset"].currentText()
        state.col_azimuth = self.col_combos["azimuth"].currentText()
    
    def validate(self) -> bool:
        state = self.controller.state.input_data
        return state.is_loaded
