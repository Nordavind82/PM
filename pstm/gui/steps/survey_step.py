"""
Step 2: Survey Geometry

Analyze and display survey geometry from loaded headers.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QGridLayout, QSizePolicy,
)
from PyQt6.QtCore import Qt

from pstm.gui.steps.base import WizardStepWidget
from pstm.gui.state import StepStatus


class SurveyStep(WizardStepWidget):
    """Step 2: Survey Geometry - Analyze input geometry."""
    
    @property
    def title(self) -> str:
        return "Survey Geometry"
    
    def _setup_ui(self) -> None:
        # Survey Extent Section
        extent_frame, extent_layout = self.create_section("Survey Extent")
        
        grid = QGridLayout()
        grid.setSpacing(10)
        
        labels = [
            ("X Range:", "x_range"),
            ("Y Range:", "y_range"),
            ("Offset Range:", "offset_range"),
            ("Mean Offset:", "offset_mean"),
            ("Number of Shots:", "n_shots"),
            ("Number of Traces:", "n_traces"),
        ]
        
        self.stat_labels = {}
        for i, (text, key) in enumerate(labels):
            row, col = divmod(i, 2)
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #888888; border: none; background: transparent;")
            grid.addWidget(lbl, row, col * 2)
            
            val = QLabel("--")
            val.setStyleSheet("color: #ffffff; border: none; background: transparent;")
            self.stat_labels[key] = val
            grid.addWidget(val, row, col * 2 + 1)
        
        extent_layout.addLayout(grid)
        
        # Analyze button
        btn_row = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze Survey")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
            }
        """)
        self.analyze_btn.clicked.connect(self._analyze_survey)
        btn_row.addWidget(self.analyze_btn)
        btn_row.addStretch()
        extent_layout.addLayout(btn_row)
        
        self.content_layout.addWidget(extent_frame)
        
        # Fold Analysis Section
        fold_frame, fold_layout = self.create_section("Fold Analysis")
        
        fold_grid = QGridLayout()
        fold_grid.setSpacing(10)
        
        fold_labels = [
            ("Maximum Fold:", "max_fold"),
            ("Mean Fold:", "mean_fold"),
        ]
        
        for i, (text, key) in enumerate(fold_labels):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #888888; border: none; background: transparent;")
            fold_grid.addWidget(lbl, i, 0)
            
            val = QLabel("--")
            val.setStyleSheet("color: #ffffff; border: none; background: transparent;")
            self.stat_labels[key] = val
            fold_grid.addWidget(val, i, 1)
        
        fold_layout.addLayout(fold_grid)
        
        # Fold map placeholder
        self.fold_map_frame = QFrame()
        self.fold_map_frame.setMinimumHeight(300)
        self.fold_map_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        fold_map_layout = QVBoxLayout(self.fold_map_frame)
        self.fold_map_label = QLabel("Click 'Compute Fold Map' to generate")
        self.fold_map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fold_map_label.setStyleSheet("color: #666666; border: none; background: transparent;")
        fold_map_layout.addWidget(self.fold_map_label)
        fold_layout.addWidget(self.fold_map_frame)
        
        # Compute fold button
        fold_btn_row = QHBoxLayout()
        self.fold_btn = QPushButton("Compute Fold Map")
        self.fold_btn.clicked.connect(self._compute_fold)
        fold_btn_row.addWidget(self.fold_btn)
        
        self.bin_size_label = QLabel("Bin size: 25m")
        self.bin_size_label.setStyleSheet("color: #888888; border: none; background: transparent;")
        fold_btn_row.addWidget(self.bin_size_label)
        fold_btn_row.addStretch()
        fold_layout.addLayout(fold_btn_row)
        
        self.content_layout.addWidget(fold_frame)
        
        # Survey Map Section
        map_frame, map_layout = self.create_section("Survey Map")
        
        self.map_frame = QFrame()
        self.map_frame.setMinimumHeight(400)
        self.map_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        inner_layout = QVBoxLayout(self.map_frame)
        self.map_label = QLabel("Survey map will be displayed here after analysis")
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_label.setStyleSheet("color: #666666; border: none; background: transparent;")
        inner_layout.addWidget(self.map_label)
        map_layout.addWidget(self.map_frame)
        
        self.content_layout.addWidget(map_frame)
        self.content_layout.addStretch()
    
    def _analyze_survey(self) -> None:
        success, message = self.controller.analyze_survey_geometry()
        
        if success:
            state = self.controller.state.survey
            inp = self.controller.state.input_data
            
            self.stat_labels["x_range"].setText(f"{state.x_min:.1f} - {state.x_max:.1f} m")
            self.stat_labels["y_range"].setText(f"{state.y_min:.1f} - {state.y_max:.1f} m")
            self.stat_labels["offset_range"].setText(f"{state.offset_min:.1f} - {state.offset_max:.1f} m")
            self.stat_labels["offset_mean"].setText(f"{state.offset_mean:.1f} m")
            self.stat_labels["n_shots"].setText(f"{state.n_shots:,}")
            self.stat_labels["n_traces"].setText(f"{inp.n_traces:,}")
            
            self.map_label.setText("✓ Survey analyzed")
            self.map_label.setStyleSheet("color: #4caf50; border: none; background: transparent;")
        else:
            self.map_label.setText(f"✗ {message}")
            self.map_label.setStyleSheet("color: #f44336; border: none; background: transparent;")
    
    def _compute_fold(self) -> None:
        self.fold_map_label.setText("Computing fold map...")
        
        # This would integrate with matplotlib for actual visualization
        fold_map, message = self.controller.compute_fold_map(bin_size=25.0)
        
        if fold_map is not None:
            state = self.controller.state.survey
            self.stat_labels["max_fold"].setText(f"{state.max_fold}")
            self.stat_labels["mean_fold"].setText(f"{state.mean_fold:.1f}")
            self.fold_map_label.setText(f"✓ Fold map computed: {message}")
            self.fold_map_label.setStyleSheet("color: #4caf50; border: none; background: transparent;")
        else:
            self.fold_map_label.setText(f"✗ {message}")
            self.fold_map_label.setStyleSheet("color: #f44336; border: none; background: transparent;")
    
    def on_enter(self) -> None:
        # Auto-analyze if data is loaded but survey not analyzed
        if self.controller.state.input_data.is_loaded:
            state = self.controller.state.survey
            if state.x_max == 0 and state.y_max == 0:
                self._analyze_survey()
            else:
                # Refresh display
                inp = self.controller.state.input_data
                self.stat_labels["x_range"].setText(f"{state.x_min:.1f} - {state.x_max:.1f} m")
                self.stat_labels["y_range"].setText(f"{state.y_min:.1f} - {state.y_max:.1f} m")
                self.stat_labels["offset_range"].setText(f"{state.offset_min:.1f} - {state.offset_max:.1f} m")
                self.stat_labels["offset_mean"].setText(f"{state.offset_mean:.1f} m")
                self.stat_labels["n_shots"].setText(f"{state.n_shots:,}")
                self.stat_labels["n_traces"].setText(f"{inp.n_traces:,}")
    
    def validate(self) -> bool:
        state = self.controller.state.survey
        return state.x_max > state.x_min and state.y_max > state.y_min
