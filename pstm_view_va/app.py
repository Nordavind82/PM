"""Main SeismicViewer application."""

import sys
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QSlider, QSplitter, QCheckBox, QMenuBar, QMenu,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QColor, QAction

from .core import PALETTES, ViewState, PSTMProject, add_recent_project
from .widgets import VolumePanel, GatherPanel, SurveyMapWidget
from .windows import VelocityAnalysisWindow
from .dialogs import StartupDialog


class SeismicViewer(QMainWindow):
    """Main seismic viewer window with dual volume and Velocity Analysis support."""

    def __init__(self, project: Optional[PSTMProject] = None):
        super().__init__()
        self.setWindowTitle("Seismic Viewer")
        self.setMinimumSize(1200, 800)

        self.view_mode = "single"  # "single", "dual", "velocity_analysis"
        self.sync_enabled = True

        # Project
        self.project: Optional[PSTMProject] = project

        # For CIG mode: track selected position
        self.cig_il_idx = 0
        self.cig_xl_idx = 0

        # Velocity Analysis window
        self.va_window = None

        self._setup_menubar()
        self._setup_ui()

        # Restore state from project or QSettings
        if self.project:
            self._restore_from_project()
            self._update_title()
        else:
            self._restore_state()

    def _setup_ui(self):
        """Setup the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(250)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(5)

        # Mode selection
        mode_group = QGroupBox("View Mode")
        mode_layout = QVBoxLayout(mode_group)

        mode_sel_layout = QHBoxLayout()
        mode_sel_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single Volume", "Dual Volume", "Velocity Analysis"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_sel_layout.addWidget(self.mode_combo)
        mode_layout.addLayout(mode_sel_layout)

        self.sync_check = QCheckBox("Synchronize Views")
        self.sync_check.setChecked(True)
        self.sync_check.toggled.connect(self._toggle_sync)
        mode_layout.addWidget(self.sync_check)

        left_layout.addWidget(mode_group)

        # Slice selection group
        slice_group = QGroupBox("Slice Selection")
        slice_layout = QVBoxLayout(slice_group)

        # Direction selector
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Inline", "Crossline", "Time Slice"])
        self.direction_combo.currentIndexChanged.connect(self._on_direction_changed)
        dir_layout.addWidget(self.direction_combo)
        slice_layout.addLayout(dir_layout)

        # Slice index
        idx_layout = QHBoxLayout()
        idx_layout.addWidget(QLabel("Index:"))
        self.slice_spin = QSpinBox()
        self.slice_spin.setRange(0, 0)
        self.slice_spin.valueChanged.connect(self._on_slice_changed)
        idx_layout.addWidget(self.slice_spin)
        slice_layout.addLayout(idx_layout)

        # Slice slider
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self.slice_spin.setValue)
        slice_layout.addWidget(self.slice_slider)

        # Step size
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step:"))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 100)
        self.step_spin.setValue(1)
        step_layout.addWidget(self.step_spin)
        slice_layout.addLayout(step_layout)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self._prev_slice)
        nav_layout.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._next_slice)
        nav_layout.addWidget(self.next_btn)
        slice_layout.addLayout(nav_layout)

        left_layout.addWidget(slice_group)

        # Display group
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        # Palette
        pal_layout = QHBoxLayout()
        pal_layout.addWidget(QLabel("Palette:"))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list(PALETTES.keys()))
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        pal_layout.addWidget(self.palette_combo)
        display_layout.addLayout(pal_layout)

        # Gain
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 100.0)
        self.gain_spin.setValue(1.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        gain_layout.addWidget(self.gain_spin)
        display_layout.addLayout(gain_layout)

        # Gain slider
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(1, 500)
        self.gain_slider.setValue(10)
        self.gain_slider.valueChanged.connect(lambda v: self.gain_spin.setValue(v / 10.0))
        display_layout.addWidget(self.gain_slider)

        # Clip percentile
        clip_layout = QHBoxLayout()
        clip_layout.addWidget(QLabel("Clip %:"))
        self.clip_spin = QDoubleSpinBox()
        self.clip_spin.setRange(90.0, 100.0)
        self.clip_spin.setValue(99.0)
        self.clip_spin.setSingleStep(0.5)
        self.clip_spin.valueChanged.connect(self._on_clip_changed)
        clip_layout.addWidget(self.clip_spin)
        display_layout.addLayout(clip_layout)

        left_layout.addWidget(display_group)

        # View group
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)

        view_btn_layout = QHBoxLayout()
        self.fit_btn = QPushButton("Fit (F)")
        self.fit_btn.clicked.connect(self._fit_view)
        view_btn_layout.addWidget(self.fit_btn)
        self.reset_btn = QPushButton("Reset (R)")
        self.reset_btn.clicked.connect(self._reset_view)
        view_btn_layout.addWidget(self.reset_btn)
        view_layout.addLayout(view_btn_layout)

        left_layout.addWidget(view_group)

        # Info group
        info_group = QGroupBox("Cursor Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("X: -  Y: -  Amp: -")
        self.info_label.setStyleSheet("font-family: monospace;")
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_group)

        # Survey map
        map_group = QGroupBox("Survey Map")
        map_layout = QVBoxLayout(map_group)
        self.survey_map = SurveyMapWidget()
        self.survey_map.setMinimumHeight(180)
        self.survey_map.location_selected.connect(self._on_map_location_selected)
        map_layout.addWidget(self.survey_map)
        left_layout.addWidget(map_group)

        left_layout.addStretch()

        # Volume panels in splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.volume1 = VolumePanel("Volume 1")
        self.volume1.slice_changed.connect(self._on_volume_slice_changed)
        self.volume1.view_changed.connect(self._on_volume_view_changed)
        self.volume1.canvas.cursor_moved.connect(self._on_cursor_moved)
        self.volume1.canvas.position_selected.connect(self._on_position_selected)
        self.splitter.addWidget(self.volume1)

        self.volume2 = VolumePanel("Volume 2")
        self.volume2.slice_changed.connect(self._on_volume_slice_changed)
        self.volume2.view_changed.connect(self._on_volume_view_changed)
        self.volume2.canvas.cursor_moved.connect(self._on_cursor_moved)
        self.volume2.hide()  # Hidden by default
        self.splitter.addWidget(self.volume2)

        # Gather panel for CIG mode
        self.gather_panel = GatherPanel("CIG Gathers")
        self.gather_panel.canvas.cursor_moved.connect(self._on_cursor_moved)
        self.gather_panel.hide()  # Hidden by default
        self.splitter.addWidget(self.gather_panel)

        # Add to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.splitter, 1)

        # Status bar
        self.statusBar().showMessage("Ready - Ctrl+Wheel to zoom, Wheel to change slice, Drag to pan")

    def _setup_menubar(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        new_action = QAction("New Project...", self)
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)

        open_action = QAction("Open Project...", self)
        open_action.triggered.connect(self._open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("Save Project", self)
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # Data loading actions
        load_cube_action = QAction("Load Cube...", self)
        load_cube_action.triggered.connect(self._load_cube_dialog)
        file_menu.addAction(load_cube_action)

        load_gathers_action = QAction("Load Gathers...", self)
        load_gathers_action.triggered.connect(self._load_gathers_dialog)
        file_menu.addAction(load_gathers_action)

    def _update_title(self):
        """Update window title with project name."""
        if self.project:
            self.setWindowTitle(f"Seismic Viewer - {self.project.name}")
        else:
            self.setWindowTitle("Seismic Viewer")

    def _new_project(self):
        """Create a new project."""
        path = QFileDialog.getSaveFileName(
            self, "Create New Project",
            str(Path.home() / "SeismicData" / "new_project.pstm"),
            "PSTM Project (*.pstm)"
        )[0]

        if not path:
            return

        try:
            self.project = PSTMProject.create(Path(path))
            add_recent_project(str(self.project.path))
            self._update_title()
            self.statusBar().showMessage(f"Created project: {self.project.name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{e}")

    def _open_project(self):
        """Open an existing project."""
        path = QFileDialog.getExistingDirectory(
            self, "Open PSTM Project",
            str(Path.home() / "SeismicData")
        )

        if not path:
            return

        if not PSTMProject.is_project(Path(path)):
            QMessageBox.warning(
                self, "Invalid Project",
                f"'{Path(path).name}' is not a valid PSTM project."
            )
            return

        try:
            self.project = PSTMProject.open(Path(path))
            add_recent_project(str(self.project.path))
            self._restore_from_project()
            self._update_title()
            self.statusBar().showMessage(f"Opened project: {self.project.name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{e}")

    def _save_project(self):
        """Save current project."""
        if not self.project:
            QMessageBox.information(
                self, "No Project",
                "No project is open. Use File > New Project to create one."
            )
            return

        try:
            self._save_to_project()
            self.project.save()
            self.statusBar().showMessage(f"Project saved: {self.project.name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{e}")

    def _load_cube_dialog(self):
        """Load seismic cube via dialog."""
        path = QFileDialog.getExistingDirectory(
            self, "Open Seismic Cube (Zarr)",
            str(Path.home() / "SeismicData")
        )
        if path:
            self.load_zarr(path, volume=1)
            if self.project:
                self.project.set_cube_path(path)

    def _load_gathers_dialog(self):
        """Load gathers via dialog."""
        path = QFileDialog.getExistingDirectory(
            self, "Open Gathers Directory",
            str(Path.home() / "SeismicData")
        )
        if path:
            self.gather_panel.load_zarr(path)
            if self.project:
                self.project.set_gathers_path(path)
            # Update survey map with fold data
            self._update_survey_map_fold()

    def _save_to_project(self):
        """Save current state to project."""
        if not self.project:
            return

        # Main viewer state
        state = {
            "view_mode": self.view_mode,
            "direction": ["inline", "crossline", "time"][self.direction_combo.currentIndex()],
            "slice_index": self.slice_spin.value(),
            "palette": self.palette_combo.currentText(),
            "gain": self.gain_spin.value(),
            "clip": self.clip_spin.value()
        }
        self.project.set_main_viewer_state(state)

        # Data sources
        if self.volume1.file_path:
            self.project.set_cube_path(self.volume1.file_path)
        if self.gather_panel.file_path:
            self.project.set_gathers_path(self.gather_panel.file_path)
        # Also check VA window's gathers path (if gathers loaded directly in VA)
        elif self.va_window and self.va_window.gathers_path:
            self.project.set_gathers_path(self.va_window.gathers_path)

        # VA state
        if self.va_window:
            self._save_va_to_project()

    def _save_va_to_project(self):
        """Save VA window state to project."""
        if not self.project or not self.va_window:
            return

        va = self.va_window
        state = {
            "position": {"il": va.il_center, "xl": va.xl_center},
            "super_gather": {
                "il_window": va.il_spin.value(),
                "xl_window": va.xl_spin.value(),
                "offset_bin": va.offset_spin.value()
            },
            "mute": {
                "top_enabled": va.top_mute_check.isChecked(),
                "v_top": va.vtop_spin.value(),
                "bottom_enabled": va.bottom_mute_check.isChecked(),
                "v_bottom": va.vbot_spin.value()
            },
            "processing": {
                "bandpass": va.bp_check.isChecked(),
                "f_low": va.f_low_spin.value(),
                "f_high": va.f_high_spin.value(),
                "agc": va.agc_check.isChecked(),
                "agc_window": va.agc_spin.value()
            },
            "nmo": {
                "use_inverse": va.use_inverse_nmo,
                "forward_applied": va.nmo_applied,
                "stretch_percent": va.stretch_spin.value()
            },
            "semblance": {
                "v_min": va.v_min,
                "v_max": va.v_max,
                "v_step": va.v_step,
                "window_samples": va.semblance_window_samples
            },
            "display": {
                "gather_colormap": va.gather_colormap,
                "gather_clip": va.gather_clip,
                "gather_gain": va.gather_gain,
                "semblance_colormap": va.semblance_colormap,
                "semblance_clip": va.semblance_clip
            }
        }
        self.project.set_va_state(state)

    def _restore_from_project(self):
        """Restore state from project."""
        if not self.project:
            return

        # Load data sources
        cube_path = self.project.get_cube_path()
        if cube_path and Path(cube_path).exists():
            self.load_zarr(cube_path, volume=1)

        gathers_path = self.project.get_gathers_path()
        if gathers_path and Path(gathers_path).exists():
            self.gather_panel.load_zarr(gathers_path)
            # Update survey map with fold data
            self._update_survey_map_fold()

        # Restore main viewer state
        state = self.project.get_main_viewer_state()
        if state:
            # Palette
            palette = state.get("palette", "Gray")
            idx = self.palette_combo.findText(palette)
            if idx >= 0:
                self.palette_combo.setCurrentIndex(idx)

            # Display settings
            self.gain_spin.setValue(float(state.get("gain", 1.0)))
            self.clip_spin.setValue(float(state.get("clip", 99.0)))

            # Direction
            direction = state.get("direction", "inline")
            directions = ["inline", "crossline", "time"]
            if direction in directions:
                self.direction_combo.setCurrentIndex(directions.index(direction))

            # Slice index
            slice_idx = state.get("slice_index", 0)
            if self.volume1.cube is not None:
                max_idx = self._get_max_slice_index()
                self.slice_spin.setRange(0, max_idx)
                self.slice_slider.setRange(0, max_idx)
                self.slice_spin.setValue(min(slice_idx, max_idx))

            # View mode (set last to trigger mode change, which will restore VA state)
            view_mode = state.get("view_mode", "single")
            modes = ["single", "dual", "velocity_analysis"]
            if view_mode in modes:
                self.mode_combo.setCurrentIndex(modes.index(view_mode))

        self.statusBar().showMessage("Project loaded")

    def _on_mode_changed(self, index: int):
        """Handle view mode change."""
        from PyQt6.QtCore import QTimer

        modes = ["single", "dual", "velocity_analysis"]
        self.view_mode = modes[index]

        # Hide all secondary panels first
        self.volume2.hide()
        self.gather_panel.hide()
        self.volume1.canvas.show_crosshair = False
        self.volume1.canvas.update()

        if self.view_mode == "single":
            self.statusBar().showMessage("Ready - Ctrl+Wheel to zoom, Wheel to change slice, Drag to pan")
        elif self.view_mode == "dual":
            self.volume2.show()
            self.statusBar().showMessage("Dual Mode - Ctrl+Wheel to zoom, Wheel to change slice, Drag to pan")
        elif self.view_mode == "velocity_analysis":
            # Open VA window and show crosshair for position selection
            self._open_velocity_analysis()
            self.statusBar().showMessage(
                "Velocity Analysis Mode: Click on cube to update VA position | Ctrl+Wheel to zoom"
            )

        # Delay size adjustment to after widgets are shown
        QTimer.singleShot(50, self._adjust_splitter_sizes)

    def _adjust_splitter_sizes(self):
        """Adjust splitter sizes based on current mode."""
        w = self.splitter.width()
        if self.view_mode == "single":
            self.splitter.setSizes([w, 0, 0])
        elif self.view_mode == "dual":
            self.splitter.setSizes([w // 2, w // 2, 0])
        elif self.view_mode == "velocity_analysis":
            # Full width for cube, VA window is separate
            self.splitter.setSizes([w, 0, 0])

    def _on_position_selected(self, x: float, y: float):
        """Handle position selection on cube (for Velocity Analysis mode)."""
        if self.view_mode != "velocity_analysis":
            return

        if self.volume1.cube is None:
            return

        # Get current slice direction to determine which coordinates were clicked
        direction = self.volume1.current_direction

        if direction == "inline":
            # Viewing inline slice: X-axis=XL, Y-axis=Time
            # Current inline is slice_value, clicked crossline is x
            il_idx = int(round(self.volume1.slice_value))
            xl_idx = int(round(x))
            self.volume1.canvas.set_crosshair_position(x, y)
        elif direction == "crossline":
            # Viewing crossline slice: X-axis=IL, Y-axis=Time
            # Clicked inline is x, current crossline is slice_value
            il_idx = int(round(x))
            xl_idx = int(round(self.volume1.slice_value))
            self.volume1.canvas.set_crosshair_position(x, y)
        else:  # time slice
            # Viewing time slice: X-axis=IL, Y-axis=XL
            il_idx = int(round(x))
            xl_idx = int(round(y))
            self.volume1.canvas.set_crosshair_position(x, y)

        # Update VA window position
        if self.va_window is not None and self.va_window.isVisible():
            self.va_window.set_position(il_idx, xl_idx)
            self.statusBar().showMessage(
                f"VA Position: IL={il_idx}, XL={xl_idx}"
            )
        else:
            self.statusBar().showMessage(f"Clicked IL={il_idx}, XL={xl_idx} - VA window not open")

    def _toggle_sync(self, enabled: bool):
        """Toggle synchronization."""
        self.sync_enabled = enabled

    def _on_map_location_selected(self, il: int, xl: int):
        """Handle location selection on survey map."""
        # Update VA window position if open
        if self.va_window is not None and self.va_window.isVisible():
            # Set position with auto_compute=True to recalculate semblance
            self.va_window.set_position(il, xl, auto_compute=True)

            # Update survey map current position
            self.survey_map.set_current_position(il, xl)

            self.statusBar().showMessage(f"Map selection: IL={il}, XL={xl}")

    def _on_va_velocity_loaded(self, grid_locations: list):
        """Handle velocity loaded signal from VA window."""
        self.survey_map.set_grid_locations(grid_locations)
        print(f"[DEBUG] VA velocity loaded: {len(grid_locations)} grid locations")

    def _on_va_position_changed(self, il: int, xl: int):
        """Handle position changed signal from VA window."""
        self.survey_map.set_current_position(il, xl)

    def _update_survey_map_fold(self):
        """Update survey map with fold data from gathers."""
        if not hasattr(self, 'survey_map'):
            return

        # Check if gather panel has data to compute fold from
        if self.gather_panel.gather_type == "common_offset_folder" and len(self.gather_panel.offset_bins) > 0:
            import numpy as np

            try:
                # Get IL/XL coordinates from gather panel
                il_coords = self.gather_panel.il_coords
                xl_coords = self.gather_panel.xl_coords

                if il_coords is None or xl_coords is None:
                    print("[DEBUG] No IL/XL coordinates in gather panel")
                    return

                n_il = len(il_coords)
                n_xl = len(xl_coords)

                # Compute fold as number of offset bins with non-zero data at each location
                fold = np.zeros((n_il, n_xl), dtype=np.float32)

                for offset_bin in self.gather_panel.offset_bins:
                    # Sum non-zero samples for each IL/XL
                    data = offset_bin[:]  # Shape: (n_il, n_xl, n_samples)
                    # Fold = count of offsets with data
                    has_data = np.any(data != 0, axis=2)
                    fold += has_data.astype(np.float32)

                self.survey_map.set_fold_map(fold, il_coords, xl_coords)
                print(f"[DEBUG] Survey map fold computed: {n_il}x{n_xl}, IL={il_coords[0]}-{il_coords[-1]}, XL={xl_coords[0]}-{xl_coords[-1]}, max fold={int(np.max(fold))}")

            except Exception as e:
                print(f"[DEBUG] Failed to compute fold map: {e}")
                import traceback
                traceback.print_exc()

    def _update_survey_map_grid(self):
        """Update survey map with velocity grid locations."""
        if not hasattr(self, 'survey_map'):
            return

        if self.va_window is not None and self.va_window.vel_grid.has_velocity():
            locations = self.va_window.vel_grid.grid_locations
            self.survey_map.set_grid_locations(locations)
            print(f"[DEBUG] Survey map grid updated: {len(locations)} locations")

    def _open_velocity_analysis(self):
        """Open Velocity Analysis window."""
        is_new_window = self.va_window is None
        if is_new_window:
            self.va_window = VelocityAnalysisWindow(self, project=self.project)
            # Connect VA window signals to update survey map
            self.va_window.velocity_loaded.connect(self._on_va_velocity_loaded)
            self.va_window.position_changed.connect(self._on_va_position_changed)

        # Pass gather data if available from gather panel
        if self.gather_panel.gather_type == "common_offset_folder" and len(self.gather_panel.offset_bins) > 0:
            self.va_window.set_gather_data(
                self.gather_panel.offset_bins,
                self.gather_panel.offset_coords,
                self.gather_panel.t_coords
            )

        # Always restore VA state for a new window (handles both gather panel and direct load)
        if is_new_window:
            self._restore_va_state()

        # Set initial position from cube center if no gathers loaded and position not restored
        if len(self.va_window.offset_bins) == 0:
            self.statusBar().showMessage("Load gathers in VA window or use gather panel first")
        elif self.va_window.il_center == 0 and self.va_window.xl_center == 0:
            # Position wasn't restored, set to center
            if self.volume1.cube is not None:
                il_center = self.volume1.cube_shape[0] // 2
                xl_center = self.volume1.cube_shape[1] // 2
                self.va_window.set_position(il_center, xl_center)
            elif len(self.gather_panel.offset_bins) > 0:
                self.va_window.set_position(
                    self.gather_panel.current_il_idx,
                    self.gather_panel.current_xl_idx
                )

        self.va_window.show()
        self.va_window.raise_()

    def _on_direction_changed(self, index: int):
        """Handle direction change."""
        directions = ["inline", "crossline", "time"]
        direction = directions[index]

        self.volume1.set_direction(direction)
        if self.view_mode == "dual":
            self.volume2.set_direction(direction)

        # Update max slice
        if self.volume1.cube is not None:
            if index == 0:
                max_idx = self.volume1.cube_shape[0] - 1
            elif index == 1:
                max_idx = self.volume1.cube_shape[1] - 1
            else:
                max_idx = self.volume1.cube_shape[2] - 1

            self.slice_spin.setRange(0, max_idx)
            self.slice_slider.setRange(0, max_idx)
            self.slice_spin.setValue(max_idx // 2)

    def _on_slice_changed(self, index: int):
        """Handle slice change from controls."""
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(index)
        self.slice_slider.blockSignals(False)

        self.volume1.set_slice_index(index)
        if self.view_mode == "dual" and self.sync_enabled:
            self.volume2.set_slice_index(index)

    def _on_volume_slice_changed(self, index: int):
        """Handle slice change from volume panel."""
        step = self.step_spin.value()
        current = self.slice_spin.value()

        if index > self.volume1.current_index:
            new_idx = min(current + step, self.slice_spin.maximum())
        else:
            new_idx = max(current - step, 0)

        self.slice_spin.setValue(new_idx)

    def _on_volume_view_changed(self, view: ViewState):
        """Handle view change from volume panel."""
        if self.sync_enabled and self.view_mode == "dual":
            sender = self.sender()
            if sender == self.volume1:
                self.volume2.set_view(view)
            else:
                self.volume1.set_view(view)

    def _on_cursor_moved(self, x: float, y: float, amp: float):
        """Handle cursor movement."""
        self.info_label.setText(f"X: {x:.1f}  Y: {y:.1f}  Amp: {amp:.4f}")

    def _prev_slice(self):
        """Go to previous slice."""
        step = self.step_spin.value()
        new_idx = max(0, self.slice_spin.value() - step)
        self.slice_spin.setValue(new_idx)

    def _next_slice(self):
        """Go to next slice."""
        step = self.step_spin.value()
        new_idx = min(self.slice_spin.maximum(), self.slice_spin.value() + step)
        self.slice_spin.setValue(new_idx)

    def _on_palette_changed(self, name: str):
        """Handle palette change."""
        self.volume1.set_palette(name)
        if self.view_mode == "dual":
            self.volume2.set_palette(name)
        elif self.view_mode == "cube_cig":
            self.gather_panel.set_palette(name)

    def _on_gain_changed(self, value: float):
        """Handle gain change."""
        self.gain_slider.blockSignals(True)
        self.gain_slider.setValue(int(value * 10))
        self.gain_slider.blockSignals(False)

        self.volume1.set_gain(value)
        if self.view_mode == "dual" and self.sync_enabled:
            self.volume2.set_gain(value)
        elif self.view_mode == "cube_cig" and self.sync_enabled:
            self.gather_panel.set_gain(value)

    def _on_clip_changed(self, value: float):
        """Handle clip percentile change."""
        self.volume1.set_clip_percentile(value)
        if self.view_mode == "dual" and self.sync_enabled:
            self.volume2.set_clip_percentile(value)
        elif self.view_mode == "cube_cig" and self.sync_enabled:
            self.gather_panel.set_clip_percentile(value)

    def _fit_view(self):
        """Fit view to data."""
        self.volume1.reset_view()
        if self.view_mode == "dual":
            self.volume2.reset_view()
        elif self.view_mode == "cube_cig":
            self.gather_panel.reset_view()

    def _reset_view(self):
        """Reset view."""
        self.volume1.reset_view()
        if self.view_mode == "dual":
            self.volume2.reset_view()
        elif self.view_mode == "cube_cig":
            self.gather_panel.reset_view()

    # =========================================================================
    # State Persistence
    # =========================================================================

    def _save_state(self):
        """Save application state for recovery on next launch."""
        settings = QSettings("PSTM", "SeismicViewerVA")

        print(f"[DEBUG] _save_state called, settings file: {settings.fileName()}")

        # File paths - get from panels
        vol1_path = self.volume1.file_path if hasattr(self.volume1, 'file_path') and self.volume1.file_path else ""
        vol2_path = self.volume2.file_path if hasattr(self.volume2, 'file_path') and self.volume2.file_path else ""
        gather_path = self.gather_panel.file_path if hasattr(self.gather_panel, 'file_path') and self.gather_panel.file_path else ""

        print(f"[DEBUG] Saving paths: vol1={vol1_path}, gather={gather_path}")

        settings.setValue("vol1_path", vol1_path)
        settings.setValue("vol2_path", vol2_path)
        settings.setValue("gather_path", gather_path)

        # View mode
        settings.setValue("view_mode", self.view_mode)
        settings.setValue("sync_enabled", self.sync_enabled)

        # Navigation
        settings.setValue("direction", self.direction_combo.currentIndex())
        settings.setValue("slice_index", self.slice_spin.value())
        settings.setValue("step_size", self.step_spin.value())

        # Display settings
        settings.setValue("palette", self.palette_combo.currentText())
        settings.setValue("gain", self.gain_spin.value())
        settings.setValue("clip", self.clip_spin.value())

        # Volume 1 position
        if self.volume1.cube is not None:
            settings.setValue("vol1_direction", self.volume1.current_direction)
            settings.setValue("vol1_index", self.volume1.current_index)

        # Gather position
        if self.gather_panel.gather_data is not None or len(self.gather_panel.offset_bins) > 0:
            settings.setValue("gather_il", self.gather_panel.current_il_idx)
            settings.setValue("gather_xl", self.gather_panel.current_xl_idx)
            settings.setValue("gather_type", self.gather_panel.gather_type)

        # VA window state (if exists)
        if self.va_window is not None:
            self._save_va_state(settings)

        # Ensure settings are written to disk
        settings.sync()
        print("[DEBUG] Settings saved and synced")

    def _save_va_state(self, settings: QSettings):
        """Save Velocity Analysis window state."""
        va = self.va_window
        settings.beginGroup("VA")

        # Gathers path (for restoring data)
        gathers_path = va.gathers_path if hasattr(va, 'gathers_path') and va.gathers_path else ""
        settings.setValue("gathers_path", gathers_path)
        print(f"[DEBUG] Saving VA gathers_path: {gathers_path}")

        # Velocity model path (for restoring velocity)
        velocity_path = va.velocity_path if hasattr(va, 'velocity_path') and va.velocity_path else ""
        settings.setValue("velocity_path", velocity_path)

        # Position
        settings.setValue("il_center", va.il_center)
        settings.setValue("xl_center", va.xl_center)

        # Super gather params
        settings.setValue("il_window", va.il_spin.value())
        settings.setValue("xl_window", va.xl_spin.value())
        settings.setValue("offset_bin", va.offset_spin.value())

        # Mute params
        settings.setValue("top_mute", va.top_mute_check.isChecked())
        settings.setValue("v_top", va.vtop_spin.value())
        settings.setValue("bottom_mute", va.bottom_mute_check.isChecked())
        settings.setValue("v_bottom", va.vbot_spin.value())

        # Processing params (use attributes, not widgets that don't exist)
        settings.setValue("use_inverse_nmo", va.use_inverse_nmo)
        settings.setValue("bandpass", va.bp_check.isChecked())
        settings.setValue("f_low", va.f_low_spin.value())
        settings.setValue("f_high", va.f_high_spin.value())
        settings.setValue("agc", va.agc_check.isChecked())
        settings.setValue("agc_window", va.agc_spin.value())

        # Forward NMO state
        settings.setValue("nmo_applied", va.nmo_applied)
        settings.setValue("stretch_percent", va.stretch_spin.value())

        # Semblance params
        settings.setValue("v_min", va.v_min)
        settings.setValue("v_max", va.v_max)
        settings.setValue("v_step", va.v_step)
        settings.setValue("semblance_window", va.semblance_window_samples)

        # Display settings
        settings.setValue("gather_colormap", va.gather_colormap)
        settings.setValue("gather_clip", va.gather_clip)
        settings.setValue("gather_gain", va.gather_gain)
        settings.setValue("semblance_colormap", va.semblance_colormap)
        settings.setValue("semblance_clip", va.semblance_clip)

        settings.endGroup()

    def _restore_state(self):
        """Restore application state from previous session."""
        settings = QSettings("PSTM", "SeismicViewerVA")

        print(f"[DEBUG] _restore_state called, settings file: {settings.fileName()}")
        print(f"[DEBUG] All keys: {settings.allKeys()}")

        # Check if we have saved state
        if not settings.contains("vol1_path"):
            print("[DEBUG] No saved state found (vol1_path key missing)")
            return

        print("[DEBUG] Found saved state, restoring...")

        try:
            # Restore display settings first (before loading data)
            palette = settings.value("palette", "Gray")
            idx = self.palette_combo.findText(palette)
            if idx >= 0:
                self.palette_combo.setCurrentIndex(idx)

            gain = settings.value("gain", 1.0)
            self.gain_spin.setValue(float(gain) if gain else 1.0)

            clip = settings.value("clip", 99.0)
            self.clip_spin.setValue(float(clip) if clip else 99.0)

            step = settings.value("step_size", 1)
            self.step_spin.setValue(int(step) if step else 1)

            # Load volume 1
            vol1_path = settings.value("vol1_path", "")
            if vol1_path and Path(vol1_path).exists():
                print(f"[DEBUG] Loading volume 1: {vol1_path}")
                self.volume1.load_zarr(vol1_path)
                # Restore position after load
                direction = settings.value("vol1_direction", "inline")
                index = settings.value("vol1_index", 0)
                self.volume1.set_direction(direction)
                self.volume1.set_slice_index(int(index) if index else 0)

            # Load gathers
            gather_path = settings.value("gather_path", "")
            if gather_path and Path(gather_path).exists():
                print(f"[DEBUG] Loading gathers: {gather_path}")
                self.gather_panel.load_zarr(gather_path)
                # Restore position
                il = settings.value("gather_il", 0)
                xl = settings.value("gather_xl", 0)
                self.gather_panel.set_position(int(il) if il else 0, int(xl) if xl else 0)

            # Restore navigation
            direction_idx = settings.value("direction", 0)
            self.direction_combo.setCurrentIndex(int(direction_idx) if direction_idx else 0)

            slice_idx = settings.value("slice_index", 0)
            if self.volume1.cube is not None:
                max_idx = self._get_max_slice_index()
                idx = min(int(slice_idx) if slice_idx else 0, max_idx)
                self.slice_spin.setRange(0, max_idx)
                self.slice_slider.setRange(0, max_idx)
                self.slice_spin.setValue(idx)

            # Restore view mode
            view_mode = settings.value("view_mode", "single")
            modes = ["single", "dual", "velocity_analysis"]
            if view_mode in modes:
                idx = modes.index(view_mode)
                self.mode_combo.setCurrentIndex(idx)

            # Restore sync
            sync = settings.value("sync_enabled", True)
            self.sync_check.setChecked(sync == True or sync == "true")

            self.statusBar().showMessage("Session restored")
            print("[DEBUG] Session restored successfully")

        except Exception as e:
            self.statusBar().showMessage(f"Failed to restore session: {e}")
            print(f"[DEBUG] Failed to restore: {e}")
            import traceback
            traceback.print_exc()

    def _restore_va_state(self):
        """Restore Velocity Analysis window state."""
        va = self.va_window

        # Use project-based restore if project is available
        if self.project:
            self._restore_va_state_from_project()
            return

        # Fall back to QSettings-based restore
        settings = QSettings("PSTM", "SeismicViewerVA")

        if not settings.childGroups() or "VA" not in settings.childGroups():
            print("[DEBUG] No VA state to restore")
            return

        print("[DEBUG] Restoring VA state from QSettings...")
        settings.beginGroup("VA")

        # First load gathers if path exists (needed before setting position)
        gathers_path = settings.value("gathers_path", "")
        if gathers_path and Path(gathers_path).exists():
            print(f"[DEBUG] Loading VA gathers from: {gathers_path}")
            va.load_gathers_from_path(gathers_path, set_center_position=False)

        # Position
        il = settings.value("il_center", va.il_center)
        xl = settings.value("xl_center", va.xl_center)
        va.il_center = int(il) if il else va.il_center
        va.xl_center = int(xl) if xl else va.xl_center

        # Super gather params - set both widgets and internal attributes
        il_win = settings.value("il_window", 5)
        va.il_spin.setValue(int(il_win) if il_win else 5)
        va.il_half = va.il_spin.value() // 2
        xl_win = settings.value("xl_window", 5)
        va.xl_spin.setValue(int(xl_win) if xl_win else 5)
        va.xl_half = va.xl_spin.value() // 2
        off_bin = settings.value("offset_bin", 50)
        va.offset_spin.setValue(int(off_bin) if off_bin else 50)
        va.offset_bin_size = float(va.offset_spin.value())

        # Mute params - set both widgets and internal attributes
        top_mute = settings.value("top_mute", False)
        va.top_mute_check.setChecked(top_mute == True or top_mute == "true")
        va.top_mute_enabled = va.top_mute_check.isChecked()
        v_top = settings.value("v_top", 1500)
        va.vtop_spin.setValue(int(v_top) if v_top else 1500)
        va.v_top = float(va.vtop_spin.value())
        bottom_mute = settings.value("bottom_mute", False)
        va.bottom_mute_check.setChecked(bottom_mute == True or bottom_mute == "true")
        va.bottom_mute_enabled = va.bottom_mute_check.isChecked()
        v_bottom = settings.value("v_bottom", 4000)
        va.vbot_spin.setValue(int(v_bottom) if v_bottom else 4000)
        va.v_bottom = float(va.vbot_spin.value())

        # Processing params - set both widgets and internal attributes
        use_inv_nmo = settings.value("use_inverse_nmo", False)
        va.use_inverse_nmo = use_inv_nmo == True or use_inv_nmo == "true"

        bandpass = settings.value("bandpass", False)
        va.bp_check.setChecked(bandpass == True or bandpass == "true")
        va.apply_bandpass = va.bp_check.isChecked()
        f_low = settings.value("f_low", 5)
        va.f_low_spin.setValue(int(f_low) if f_low else 5)
        va.f_low = va.f_low_spin.value()
        f_high = settings.value("f_high", 80)
        va.f_high_spin.setValue(int(f_high) if f_high else 80)
        va.f_high = va.f_high_spin.value()

        agc = settings.value("agc", False)
        va.agc_check.setChecked(agc == True or agc == "true")
        va.apply_agc_flag = va.agc_check.isChecked()
        agc_window = settings.value("agc_window", 500)
        va.agc_spin.setValue(int(agc_window) if agc_window else 500)
        va.agc_window = va.agc_spin.value()

        # Forward NMO state
        nmo_applied = settings.value("nmo_applied", False)
        va.nmo_applied = nmo_applied == True or nmo_applied == "true"
        stretch_pct = settings.value("stretch_percent", 30)
        va.stretch_spin.setValue(int(stretch_pct) if stretch_pct else 30)
        va.stretch_percent = va.stretch_spin.value()
        # Update button text based on NMO state
        if va.nmo_applied:
            va.apply_nmo_btn.setText("Remove NMO")
            va.nmo_applied_label.setText(f"NMO applied (stretch mute: {va.stretch_percent}%)")

        # Semblance params
        v_min = settings.value("v_min", 1500.0)
        va.v_min = float(v_min) if v_min else 1500.0
        v_max = settings.value("v_max", 5000.0)
        va.v_max = float(v_max) if v_max else 5000.0
        v_step = settings.value("v_step", 100.0)
        va.v_step = float(v_step) if v_step else 100.0
        sem_win = settings.value("semblance_window", 5)
        va.semblance_window_samples = int(sem_win) if sem_win else 5

        # Display settings
        gather_cmap = settings.value("gather_colormap", "Seismic (BWR)")
        va.gather_colormap = gather_cmap if gather_cmap else "Seismic (BWR)"
        gather_clip = settings.value("gather_clip", 99.0)
        va.gather_clip = float(gather_clip) if gather_clip else 99.0
        gather_gain = settings.value("gather_gain", 1.0)
        va.gather_gain = float(gather_gain) if gather_gain else 1.0
        va.gain_spin.setValue(va.gather_gain)
        sem_cmap = settings.value("semblance_colormap", "Viridis")
        va.semblance_colormap = sem_cmap if sem_cmap else "Viridis"
        sem_clip = settings.value("semblance_clip", 99.0)
        va.semblance_clip = float(sem_clip) if sem_clip else 99.0

        settings.endGroup()

        # Update position display and compute if data was loaded
        va.pos_label.setText(f"IL={va.il_center}, XL={va.xl_center}")
        print(f"[DEBUG] VA state restored: IL={va.il_center}, XL={va.xl_center}")

        # If gathers were loaded, update displays with restored position
        if len(va.offset_bins) > 0:
            va._compute_super_gather()
            va._update_display()
            va._compute_semblance()
            print("[DEBUG] VA display updated with restored position")

    def _restore_va_state_from_project(self):
        """Restore Velocity Analysis window state from project."""
        if not self.project or not self.va_window:
            return

        va = self.va_window
        print("[DEBUG] Restoring VA state from project...")

        # First load gathers from project
        gathers_path = self.project.get_gathers_path()
        print(f"[DEBUG] Project gathers_path: {gathers_path}")
        if gathers_path and Path(gathers_path).exists():
            print(f"[DEBUG] Loading VA gathers from project: {gathers_path}")
            va.load_gathers_from_path(gathers_path, set_center_position=False)
        else:
            print(f"[DEBUG] Gathers path not found or doesn't exist")

        # Load velocities from project
        if self.project.has_initial_velocity() or self.project.has_edited_velocity():
            print("[DEBUG] Loading velocities from project...")
            va.load_velocities_from_project()
            # Update survey map with grid locations
            self._update_survey_map_grid()

        # Get VA state from project
        state = self.project.get_va_state()
        if not state:
            print("[DEBUG] No VA state in project")
            return

        # Position
        pos = state.get("position", {})
        va.il_center = pos.get("il", va.il_center)
        va.xl_center = pos.get("xl", va.xl_center)

        # Super gather params
        sg = state.get("super_gather", {})
        va.il_spin.setValue(sg.get("il_window", 5))
        va.il_half = va.il_spin.value() // 2
        va.xl_spin.setValue(sg.get("xl_window", 5))
        va.xl_half = va.xl_spin.value() // 2
        va.offset_spin.setValue(sg.get("offset_bin", 50))
        va.offset_bin_size = float(va.offset_spin.value())

        # Mute params
        mute = state.get("mute", {})
        va.top_mute_check.setChecked(mute.get("top_enabled", False))
        va.top_mute_enabled = va.top_mute_check.isChecked()
        va.vtop_spin.setValue(mute.get("v_top", 1500))
        va.v_top = float(va.vtop_spin.value())
        va.bottom_mute_check.setChecked(mute.get("bottom_enabled", False))
        va.bottom_mute_enabled = va.bottom_mute_check.isChecked()
        va.vbot_spin.setValue(mute.get("v_bottom", 4000))
        va.v_bottom = float(va.vbot_spin.value())

        # Processing params
        proc = state.get("processing", {})
        va.bp_check.setChecked(proc.get("bandpass", False))
        va.apply_bandpass = va.bp_check.isChecked()
        va.f_low_spin.setValue(proc.get("f_low", 5))
        va.f_low = va.f_low_spin.value()
        va.f_high_spin.setValue(proc.get("f_high", 80))
        va.f_high = va.f_high_spin.value()
        va.agc_check.setChecked(proc.get("agc", False))
        va.apply_agc_flag = va.agc_check.isChecked()
        va.agc_spin.setValue(proc.get("agc_window", 500))
        va.agc_window = va.agc_spin.value()

        # NMO params
        nmo = state.get("nmo", {})
        va.use_inverse_nmo = nmo.get("use_inverse", False)
        va.nmo_applied = nmo.get("forward_applied", False)
        va.stretch_spin.setValue(nmo.get("stretch_percent", 30))
        va.stretch_percent = va.stretch_spin.value()
        if va.nmo_applied:
            va.apply_nmo_btn.setText("Remove NMO")
            va.nmo_applied_label.setText(f"NMO applied (stretch mute: {va.stretch_percent}%)")

        # Semblance params
        sem = state.get("semblance", {})
        va.v_min = sem.get("v_min", 1500.0)
        va.v_max = sem.get("v_max", 5000.0)
        va.v_step = sem.get("v_step", 100.0)
        va.semblance_window_samples = sem.get("window_samples", 5)

        # Display settings
        disp = state.get("display", {})
        va.gather_colormap = disp.get("gather_colormap", "Seismic (BWR)")
        va.gather_clip = disp.get("gather_clip", 99.0)
        va.gather_gain = disp.get("gather_gain", 1.0)
        va.gain_spin.setValue(va.gather_gain)
        va.semblance_colormap = disp.get("semblance_colormap", "Viridis")
        va.semblance_clip = disp.get("semblance_clip", 99.0)

        # Update position display
        va.pos_label.setText(f"IL={va.il_center}, XL={va.xl_center}")
        print(f"[DEBUG] VA state restored from project: IL={va.il_center}, XL={va.xl_center}")

        # Load picks for restored position
        va._load_picks_for_position()

        # If gathers were loaded, update displays
        if len(va.offset_bins) > 0:
            va._compute_super_gather()
            va._update_display()
            va._compute_semblance()
            print("[DEBUG] VA display updated with restored position")

    def _get_max_slice_index(self) -> int:
        """Get max slice index for current direction."""
        if self.volume1.cube is None:
            return 0
        direction_idx = self.direction_combo.currentIndex()
        if direction_idx == 0:
            return self.volume1.cube_shape[0] - 1
        elif direction_idx == 1:
            return self.volume1.cube_shape[1] - 1
        else:
            return self.volume1.cube_shape[2] - 1

    def closeEvent(self, event):
        """Handle window close - save state before exit."""
        print("[DEBUG] closeEvent called")

        # Save to project if open, otherwise use QSettings
        if self.project:
            try:
                self._save_to_project()
                self.project.save()
                print(f"[DEBUG] Project saved: {self.project.name}")
            except Exception as e:
                print(f"[DEBUG] Failed to save project: {e}")
        else:
            self._save_state()

        # Close VA window if open
        if self.va_window is not None:
            self.va_window.close()

        event.accept()
        print("[DEBUG] closeEvent finished")

    def load_zarr(self, path: str, volume: int = 1):
        """Load zarr data into specified volume."""
        if volume == 1:
            self.volume1.load_zarr(path)
            # Update controls after loading
            if self.volume1.cube is not None:
                direction_idx = self.direction_combo.currentIndex()
                if direction_idx == 0:
                    max_idx = self.volume1.cube_shape[0] - 1
                elif direction_idx == 1:
                    max_idx = self.volume1.cube_shape[1] - 1
                else:
                    max_idx = self.volume1.cube_shape[2] - 1

                self.slice_spin.setRange(0, max_idx)
                self.slice_slider.setRange(0, max_idx)
                self.slice_spin.setValue(max_idx // 2)
        else:
            self.volume2.load_zarr(path)

    def load_gathers(self, path: str, gather_type: str = "common_offset"):
        """Load gather data for CIG mode."""
        self.gather_panel.load_zarr(path)
        if gather_type == "common_offset":
            self.gather_panel.type_combo.setCurrentIndex(1)
        else:
            self.gather_panel.type_combo.setCurrentIndex(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Seismic Viewer with Velocity Analysis')
    parser.add_argument('cube', nargs='?', help='Path to seismic cube (zarr)')
    parser.add_argument('second', nargs='?', help='Path to second volume or gathers (zarr)')
    parser.add_argument('--cig', action='store_true',
                       help='Enable CIG mode (second file is gathers)')
    parser.add_argument('--gather-type', choices=['cig', 'common_offset'], default='common_offset',
                       help='Type of gather organization (default: common_offset)')
    parser.add_argument('--project', '-p', help='Path to PSTM project to open')
    parser.add_argument('--no-startup', action='store_true',
                       help='Skip startup dialog (for command-line usage)')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setOrganizationName("PSTM")
    app.setApplicationName("SeismicViewerVA")
    app.setStyle('Fusion')

    # Dark theme
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(palette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    project = None

    # Handle project from command line
    if args.project:
        try:
            project = PSTMProject.open(Path(args.project))
            add_recent_project(str(project.path))
        except Exception as e:
            print(f"Error opening project: {e}")
            sys.exit(1)

    # Show startup dialog if no command-line args and not skipped
    elif not args.cube and not args.no_startup:
        startup = StartupDialog()
        if startup.exec() != startup.DialogCode.Accepted:
            sys.exit(0)
        project = startup.get_project()

    # Create viewer with project
    viewer = SeismicViewer(project=project)
    viewer.show()

    # Load files from command line (if no project)
    if not project:
        if args.cube:
            viewer.load_zarr(args.cube, volume=1)

        if args.second:
            if args.cig:
                # CIG mode
                viewer.mode_combo.setCurrentIndex(2)  # "Velocity Analysis"
                viewer.load_gathers(args.second, args.gather_type)
            else:
                # Dual volume mode
                viewer.mode_combo.setCurrentIndex(1)  # "Dual Volume"
                viewer.load_zarr(args.second, volume=2)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
