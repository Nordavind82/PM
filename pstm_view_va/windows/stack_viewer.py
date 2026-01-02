"""Stack viewer window for comparing multiple stacks."""

from typing import List, Optional, Tuple, Dict
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QListWidget, QListWidgetItem,
    QSplitter, QGroupBox, QSlider, QSpinBox, QComboBox, QCheckBox,
    QApplication, QStatusBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QWheelEvent

import zarr

from ..core import AxisConfig
from ..widgets import SeismicCanvas


class StackViewerWindow(QMainWindow):
    """Window for viewing and comparing stacks."""

    def __init__(self, parent=None, project_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Stack Viewer")
        self.setMinimumSize(900, 700)

        # Project path for loading/saving stacks
        self.project_path = Path(project_path) if project_path else None

        # Loaded stacks: list of (name, data, il_coords, xl_coords, t_coords, metadata)
        self.stacks: List[Dict] = []
        self.current_stack_idx = 0

        # Display settings
        self.colormap = "Seismic (BWR)"
        self.clip_percentile = 99.0
        self.gain = 1.0

        # Flicker mode for comparison
        self.flicker_mode = False
        self.flicker_idx_a = 0
        self.flicker_idx_b = 1

        self._setup_ui()

        # Load existing stacks from project
        if self.project_path:
            self._load_project_stacks()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # Left panel - stack list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(250)

        # Stack list
        list_group = QGroupBox("Loaded Stacks")
        list_layout = QVBoxLayout(list_group)

        self.stack_list = QListWidget()
        self.stack_list.currentRowChanged.connect(self._on_stack_selected)
        list_layout.addWidget(self.stack_list)

        # Load/Remove buttons
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Stack...")
        self.load_btn.clicked.connect(self._load_stack)
        btn_layout.addWidget(self.load_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_stack)
        self.remove_btn.setEnabled(False)
        btn_layout.addWidget(self.remove_btn)
        list_layout.addLayout(btn_layout)

        left_layout.addWidget(list_group)

        # Display controls
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        # Colormap
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["Seismic (BWR)", "Gray", "Viridis", "Hot"])
        self.cmap_combo.currentTextChanged.connect(self._on_colormap_changed)
        cmap_layout.addWidget(self.cmap_combo)
        display_layout.addLayout(cmap_layout)

        # Gain
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(1, 100)
        self.gain_spin.setValue(1)
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        gain_layout.addWidget(self.gain_spin)
        display_layout.addLayout(gain_layout)

        left_layout.addWidget(display_group)

        # Comparison controls
        compare_group = QGroupBox("Compare (Mouse Wheel)")
        compare_layout = QVBoxLayout(compare_group)

        self.flicker_check = QCheckBox("Flicker mode")
        self.flicker_check.toggled.connect(self._on_flicker_toggled)
        compare_layout.addWidget(self.flicker_check)

        flicker_a_layout = QHBoxLayout()
        flicker_a_layout.addWidget(QLabel("Stack A:"))
        self.flicker_a_combo = QComboBox()
        self.flicker_a_combo.currentIndexChanged.connect(self._on_flicker_a_changed)
        flicker_a_layout.addWidget(self.flicker_a_combo)
        compare_layout.addLayout(flicker_a_layout)

        flicker_b_layout = QHBoxLayout()
        flicker_b_layout.addWidget(QLabel("Stack B:"))
        self.flicker_b_combo = QComboBox()
        self.flicker_b_combo.currentIndexChanged.connect(self._on_flicker_b_changed)
        flicker_b_layout.addWidget(self.flicker_b_combo)
        compare_layout.addLayout(flicker_b_layout)

        self.current_flicker_label = QLabel("Showing: -")
        self.current_flicker_label.setStyleSheet("font-weight: bold;")
        compare_layout.addWidget(self.current_flicker_label)

        left_layout.addWidget(compare_group)

        # Stack info
        info_group = QGroupBox("Stack Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("No stack loaded")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Right panel - canvas
        self.canvas = SeismicCanvas()
        self.canvas.setMinimumWidth(600)
        main_layout.addWidget(self.canvas, 1)

        # Status bar
        self.setStatusBar(QStatusBar())

        # Install wheel event filter for flicker
        self.canvas.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Handle wheel events on canvas for flicker mode."""
        if obj == self.canvas and self.flicker_mode and len(self.stacks) >= 2:
            if event.type() == event.Type.Wheel:
                delta = event.angleDelta().y()
                if delta > 0:
                    self._show_flicker_stack(self.flicker_idx_a)
                else:
                    self._show_flicker_stack(self.flicker_idx_b)
                return True
        return super().eventFilter(obj, event)

    def add_stack(self, name: str, data: np.ndarray,
                  coords: dict, metadata: dict = None):
        """
        Add a stack to the viewer.

        Args:
            name: Display name for the stack
            data: 2D array (n_traces, n_time) or 3D array
            coords: dict with 'il_coords', 'xl_coords', 't_coords'
            metadata: Optional metadata dict
        """
        stack_info = {
            'name': name,
            'data': data,
            'il_coords': coords.get('il_coords'),
            'xl_coords': coords.get('xl_coords'),
            't_coords': coords.get('t_coords'),
            'metadata': metadata or {}
        }
        self.stacks.append(stack_info)

        # Add to list widget
        item = QListWidgetItem(name)
        self.stack_list.addItem(item)

        # Update flicker combos
        self._update_flicker_combos()

        # Select if first stack
        if len(self.stacks) == 1:
            self.stack_list.setCurrentRow(0)

        self.remove_btn.setEnabled(True)
        self.statusBar().showMessage(f"Added stack: {name}")

    def set_live_preview(self, name: str, data: np.ndarray, coords: dict):
        """
        Set or update the live preview stack.

        This is used for real-time stack updates during velocity analysis.
        The live preview stack is always displayed and updated in-place.

        Args:
            name: Display name for the preview
            data: 2D array (n_traces, n_time)
            coords: dict with 'il_coords', 'xl_coords', 't_coords'
        """
        # Look for existing live preview stack
        live_idx = None
        for i, stack_info in enumerate(self.stacks):
            if stack_info.get('is_live_preview', False):
                live_idx = i
                break

        if live_idx is not None:
            # Update existing live preview
            self.stacks[live_idx]['name'] = name
            self.stacks[live_idx]['data'] = data
            self.stacks[live_idx]['il_coords'] = coords.get('il_coords')
            self.stacks[live_idx]['xl_coords'] = coords.get('xl_coords')
            self.stacks[live_idx]['t_coords'] = coords.get('t_coords')

            # Update list widget text
            self.stack_list.item(live_idx).setText(f"[LIVE] {name}")

            # If this is the current stack, update display
            if self.current_stack_idx == live_idx:
                self._update_display()
        else:
            # Add new live preview stack
            stack_info = {
                'name': name,
                'data': data,
                'il_coords': coords.get('il_coords'),
                'xl_coords': coords.get('xl_coords'),
                't_coords': coords.get('t_coords'),
                'metadata': {},
                'is_live_preview': True
            }
            self.stacks.insert(0, stack_info)  # Insert at beginning

            # Add to list widget at beginning
            item = QListWidgetItem(f"[LIVE] {name}")
            self.stack_list.insertItem(0, item)

            # Update flicker combos
            self._update_flicker_combos()

            # Select the live preview
            self.stack_list.setCurrentRow(0)

            self.remove_btn.setEnabled(True)

        self.statusBar().showMessage(f"Live preview: {name}")

    def _load_stack(self):
        """Load a stack from file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Stack",
            str(Path.home() / "SeismicData"),
            "Zarr (*.zarr);;All Files (*)"
        )

        if not path:
            return

        try:
            data = None
            attrs = {}

            # Try opening as a direct array first (zarr v3 compatible)
            try:
                z = zarr.open_array(path, mode='r')
                data = np.asarray(z)
                attrs = dict(z.attrs) if hasattr(z, 'attrs') else {}
            except Exception:
                # Fall back to opening as group
                z = zarr.open(path, mode='r')

                if isinstance(z, zarr.Array):
                    data = np.asarray(z)
                    attrs = dict(z.attrs) if hasattr(z, 'attrs') else {}
                elif hasattr(z, 'keys'):
                    # It's a group - try to find data array
                    for key in ['data', 'stack']:
                        if key in z:
                            arr = z[key]
                            data = np.asarray(arr)
                            attrs = dict(arr.attrs) if hasattr(arr, 'attrs') else {}
                            break
                    else:
                        # Try first array found
                        for key in z.keys():
                            arr = z[key]
                            if hasattr(arr, 'shape'):
                                data = np.asarray(arr)
                                attrs = dict(arr.attrs) if hasattr(arr, 'attrs') else {}
                                break

            if data is None:
                self.statusBar().showMessage("No data array found in file")
                return

            # Extract coordinates
            coords = {}
            for key in ['t_coords', 't_axis_ms', 'time']:
                if key in attrs:
                    coords['t_coords'] = np.array(attrs[key])
                    break

            for key in ['il_coords', 'inline', 'x_axis']:
                if key in attrs:
                    coords['il_coords'] = np.array(attrs[key])
                    break

            for key in ['xl_coords', 'crossline', 'y_axis']:
                if key in attrs:
                    coords['xl_coords'] = np.array(attrs[key])
                    break

            # Default coordinates if not found
            if 't_coords' not in coords:
                n_time = data.shape[-1]
                coords['t_coords'] = np.arange(n_time) * 2.0  # Assume 2ms sampling

            name = Path(path).stem
            self.add_stack(name, data, coords, {'path': path})

        except Exception as e:
            self.statusBar().showMessage(f"Error loading: {e}")

    def _remove_stack(self):
        """Remove selected stack."""
        idx = self.stack_list.currentRow()
        if idx >= 0 and idx < len(self.stacks):
            self.stacks.pop(idx)
            self.stack_list.takeItem(idx)
            self._update_flicker_combos()

            if len(self.stacks) == 0:
                self.remove_btn.setEnabled(False)
                self.canvas.set_data(None, "", 0, 0, None, None, 0)
                self.info_label.setText("No stack loaded")

    def _on_stack_selected(self, idx: int):
        """Handle stack selection."""
        if idx < 0 or idx >= len(self.stacks):
            return

        self.current_stack_idx = idx
        self._update_display()

    def _update_display(self):
        """Update canvas with current stack."""
        if self.current_stack_idx >= len(self.stacks):
            return

        stack_info = self.stacks[self.current_stack_idx]
        data = stack_info['data']
        t_coords = stack_info['t_coords']
        il_coords = stack_info['il_coords']
        xl_coords = stack_info['xl_coords']

        # Handle different data dimensions
        if data.ndim == 3:
            # 3D: show inline section (first inline)
            display_data = data[0, :, :].T  # (n_time, n_xl)
            x_coords = xl_coords if xl_coords is not None else np.arange(data.shape[1])
            x_label = "Crossline"
        elif data.ndim == 2:
            display_data = data.T  # (n_time, n_traces)
            if xl_coords is not None:
                x_coords = xl_coords
                x_label = "Crossline"
            elif il_coords is not None:
                x_coords = il_coords
                x_label = "Inline"
            else:
                x_coords = np.arange(data.shape[0])
                x_label = "Trace"
        else:
            return

        t_min = t_coords[0] if t_coords is not None else 0
        t_max = t_coords[-1] if t_coords is not None else display_data.shape[0] * 2
        x_min = x_coords[0]
        x_max = x_coords[-1]

        x_axis = AxisConfig(x_label, x_min, x_max, "")
        y_axis = AxisConfig("Time", t_min, t_max, "ms")

        self.canvas.set_palette(self.colormap)
        self.canvas.set_clip_percentile(self.clip_percentile)
        self.canvas.set_gain(self.gain)
        self.canvas.set_data(display_data, "stack", 0, 0, x_axis, y_axis, 0)
        self.canvas.slice_direction = f"Stack: {stack_info['name']}"

        # Update info
        shape_str = ' x '.join(str(s) for s in data.shape)
        self.info_label.setText(
            f"Name: {stack_info['name']}\n"
            f"Shape: {shape_str}\n"
            f"Time: {t_min:.0f} - {t_max:.0f} ms"
        )

    def _update_flicker_combos(self):
        """Update flicker combo boxes with stack names."""
        self.flicker_a_combo.clear()
        self.flicker_b_combo.clear()

        for stack_info in self.stacks:
            self.flicker_a_combo.addItem(stack_info['name'])
            self.flicker_b_combo.addItem(stack_info['name'])

        if len(self.stacks) >= 2:
            self.flicker_a_combo.setCurrentIndex(0)
            self.flicker_b_combo.setCurrentIndex(1)

    def _on_colormap_changed(self, cmap: str):
        """Handle colormap change."""
        self.colormap = cmap
        self._update_display()

    def _on_gain_changed(self, value: int):
        """Handle gain change."""
        self.gain = float(value)
        self._update_display()

    def _on_flicker_toggled(self, enabled: bool):
        """Toggle flicker mode."""
        self.flicker_mode = enabled
        if enabled and len(self.stacks) >= 2:
            self.current_flicker_label.setText(f"Showing: {self.stacks[self.flicker_idx_a]['name']}")
        else:
            self.current_flicker_label.setText("Showing: -")

    def _on_flicker_a_changed(self, idx: int):
        """Handle flicker A selection."""
        self.flicker_idx_a = idx

    def _on_flicker_b_changed(self, idx: int):
        """Handle flicker B selection."""
        self.flicker_idx_b = idx

    def _show_flicker_stack(self, idx: int):
        """Show specific stack in flicker mode."""
        if idx < 0 or idx >= len(self.stacks):
            return

        self.current_stack_idx = idx
        self._update_display()
        self.current_flicker_label.setText(f"Showing: {self.stacks[idx]['name']}")

    def _load_project_stacks(self):
        """Load all stacks from the project's stacks folder."""
        if not self.project_path:
            return

        stacks_dir = self.project_path / "stacks"
        if not stacks_dir.exists():
            print(f"[StackViewer] No stacks directory found at {stacks_dir}")
            return

        # Find all zarr directories/files in stacks folder
        stack_paths = list(stacks_dir.glob("*.zarr"))
        if not stack_paths:
            print(f"[StackViewer] No .zarr stacks found in {stacks_dir}")
            return

        print(f"[StackViewer] Found {len(stack_paths)} stacks in project")

        for stack_path in sorted(stack_paths):
            try:
                data = None
                attrs = {}

                # Try opening as a direct array first (zarr v3 compatible)
                try:
                    z = zarr.open_array(str(stack_path), mode='r')
                    data = np.asarray(z)
                    attrs = dict(z.attrs) if hasattr(z, 'attrs') else {}
                except Exception:
                    # Fall back to opening as group
                    z = zarr.open(str(stack_path), mode='r')

                    if isinstance(z, zarr.Array):
                        data = np.asarray(z)
                        attrs = dict(z.attrs) if hasattr(z, 'attrs') else {}
                    elif hasattr(z, 'keys'):
                        # It's a group - try to find data array
                        for key in ['data', 'stack']:
                            if key in z:
                                arr = z[key]
                                data = np.asarray(arr)
                                attrs = dict(arr.attrs) if hasattr(arr, 'attrs') else {}
                                break
                        else:
                            # Try first array found
                            for key in z.keys():
                                arr = z[key]
                                if hasattr(arr, 'shape'):
                                    data = np.asarray(arr)
                                    attrs = dict(arr.attrs) if hasattr(arr, 'attrs') else {}
                                    break

                if data is None:
                    print(f"[StackViewer] No data array found in {stack_path.name}")
                    continue

                # Extract coordinates from attributes
                coords = {}
                for key in ['t_coords', 't_axis_ms', 'time']:
                    if key in attrs:
                        coords['t_coords'] = np.array(attrs[key])
                        break

                for key in ['il_coords', 'inline', 'x_axis']:
                    if key in attrs:
                        coords['il_coords'] = np.array(attrs[key])
                        break

                for key in ['xl_coords', 'crossline', 'y_axis']:
                    if key in attrs:
                        coords['xl_coords'] = np.array(attrs[key])
                        break

                # Default time coordinates if not found
                if 't_coords' not in coords:
                    n_time = data.shape[-1]
                    coords['t_coords'] = np.arange(n_time) * 2.0  # Assume 2ms sampling

                name = stack_path.stem
                self.add_stack(name, data, coords, {'path': str(stack_path)})
                print(f"[StackViewer] Loaded: {name} - shape {data.shape}")

            except Exception as e:
                print(f"[StackViewer] Error loading {stack_path.name}: {e}")
