"""Velocity Analysis Window for super gathers and semblance."""

import json
from typing import List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import numpy as np
import zarr

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QSplitter, QFrame, QFileDialog, QDialog,
    QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPen, QColor, QKeySequence, QShortcut

from ..core import AxisConfig
from ..widgets import SeismicCanvas
from ..dialogs import SemblanceSettingsDialog, VelocityGridDialog, StackingDialog
from ..processing import (
    create_super_gather, apply_velocity_mute,
    apply_nmo_correction, apply_nmo_with_velocity_model,
    apply_bandpass_filter, apply_agc,
    compute_semblance_fast,
    compute_inline_stack, compute_crossline_stack, compute_stack,
    GatherCache, LiveStackUpdater
)
from ..io import load_velocity_model, extract_velocity_function

if TYPE_CHECKING:
    from ..core import PSTMProject


class VelocityOutputGrid:
    """Manages the output velocity grid for velocity analysis."""

    def __init__(self):
        self.il_coords: np.ndarray = np.array([])  # IL grid coordinates
        self.xl_coords: np.ndarray = np.array([])  # XL grid coordinates
        self.t_coords: np.ndarray = np.array([])   # Time coordinates
        self.velocities: np.ndarray = None         # 3D velocity array (n_il, n_xl, n_time)
        self.grid_locations: List[Tuple[int, int]] = []  # List of (il, xl) tuples
        self.current_idx = 0  # Current index in grid_locations

    def setup_grid(self, il_start: int, il_end: int, il_step: int,
                   xl_start: int, xl_end: int, xl_step: int,
                   output_dt_ms: float, t_min: float = 0.0, t_max: float = 4000.0):
        """Setup the output grid.

        Args:
            il_start, il_end, il_step: Inline range and step
            xl_start, xl_end, xl_step: Crossline range and step
            output_dt_ms: Output sample rate in ms (10, 100, or 1000)
            t_min, t_max: Time range in ms for output velocity samples
        """
        self.il_coords = np.arange(il_start, il_end + 1, il_step)
        self.xl_coords = np.arange(xl_start, xl_end + 1, xl_step)

        # Compute time coordinates from sample rate
        self.t_coords = np.arange(t_min, t_max + output_dt_ms, output_dt_ms)

        # Initialize velocity array
        self.velocities = np.full(
            (len(self.il_coords), len(self.xl_coords), len(self.t_coords)),
            np.nan, dtype=np.float32
        )

        # Build list of grid locations
        self.grid_locations = []
        for il in self.il_coords:
            for xl in self.xl_coords:
                self.grid_locations.append((int(il), int(xl)))

        self.current_idx = 0

    def load_velocity(self, vel_model: np.ndarray, metadata: dict):
        """Load and resample velocity model onto the grid."""
        if vel_model is None:
            return False

        n_il = len(self.il_coords)
        n_xl = len(self.xl_coords)
        n_t = len(self.t_coords)

        # Get source time coordinates from metadata or infer from model shape
        source_t_coords = None
        if metadata and 't_coords' in metadata:
            source_t_coords = np.array(metadata['t_coords'])
        elif metadata and 'dt_ms' in metadata:
            # Infer from dt and model shape
            n_src_t = vel_model.shape[-1]  # Time is last axis
            dt_ms = metadata['dt_ms']
            source_t_coords = np.arange(n_src_t) * dt_ms
        else:
            # Fallback: assume 2ms sampling (common for seismic)
            n_src_t = vel_model.shape[-1]
            source_t_coords = np.arange(n_src_t) * 2.0

        # Get IL/XL coordinates from metadata for proper indexing
        source_il_coords = metadata.get('il_coords') if metadata else None
        source_xl_coords = metadata.get('xl_coords') if metadata else None

        # Resample velocity onto our grid using interpolation
        for i, il in enumerate(self.il_coords):
            for j, xl in enumerate(self.xl_coords):
                vel_func = extract_velocity_function(
                    vel_model, int(il), int(xl), source_t_coords,
                    il_coords=source_il_coords, xl_coords=source_xl_coords
                )
                if vel_func is not None and len(vel_func) > 0:
                    # Interpolate from source sampling to output sampling
                    vel_resampled = np.interp(
                        self.t_coords,      # Output times (e.g., 0, 100, 200... ms)
                        source_t_coords[:len(vel_func)],  # Source times (e.g., 0, 2, 4... ms)
                        vel_func            # Velocity values at source times
                    )
                    self.velocities[i, j, :] = vel_resampled

        return True

    def get_current_location(self) -> Tuple[int, int]:
        """Get current grid location."""
        if len(self.grid_locations) == 0:
            return (0, 0)
        return self.grid_locations[self.current_idx]

    def next_location(self) -> Tuple[int, int]:
        """Move to next grid location."""
        if len(self.grid_locations) == 0:
            return (0, 0)
        self.current_idx = (self.current_idx + 1) % len(self.grid_locations)
        return self.grid_locations[self.current_idx]

    def prev_location(self) -> Tuple[int, int]:
        """Move to previous grid location."""
        if len(self.grid_locations) == 0:
            return (0, 0)
        self.current_idx = (self.current_idx - 1) % len(self.grid_locations)
        return self.grid_locations[self.current_idx]

    def is_on_grid(self, il: int, xl: int) -> bool:
        """Check if position is on the output grid."""
        return (il, xl) in self.grid_locations

    def get_grid_index(self, il: int, xl: int) -> Tuple[int, int]:
        """Get grid indices for given IL/XL position."""
        il_idx = np.argmin(np.abs(self.il_coords - il))
        xl_idx = np.argmin(np.abs(self.xl_coords - xl))
        return int(il_idx), int(xl_idx)

    def get_velocity_at(self, il: int, xl: int) -> Optional[np.ndarray]:
        """Get velocity function at given position."""
        if self.velocities is None:
            return None

        il_idx, xl_idx = self.get_grid_index(il, xl)

        # Check if we're close enough to a grid point
        if abs(self.il_coords[il_idx] - il) > 1 or abs(self.xl_coords[xl_idx] - xl) > 1:
            return None

        vel = self.velocities[il_idx, xl_idx, :]
        if np.all(np.isnan(vel)):
            return None
        return vel

    def has_velocity(self) -> bool:
        """Check if any velocity data is loaded."""
        return self.velocities is not None and not np.all(np.isnan(self.velocities))

    def get_velocity_as_picks(self, il: int, xl: int, max_picks: int = 20) -> List[Tuple[float, float]]:
        """Extract velocity function as sparse picks for editing.

        Args:
            il, xl: Grid location
            max_picks: Maximum number of picks to return (samples at key times)

        Returns:
            List of (time_ms, velocity) tuples
        """
        vel_func = self.get_velocity_at(il, xl)
        if vel_func is None or len(vel_func) == 0:
            return []

        # Sample at evenly spaced intervals
        n_samples = len(self.t_coords)
        step = max(1, n_samples // max_picks)
        indices = list(range(0, n_samples, step))

        # Always include last point
        if indices[-1] != n_samples - 1:
            indices.append(n_samples - 1)

        picks = []
        for idx in indices:
            t = self.t_coords[idx]
            v = vel_func[idx]
            if not np.isnan(v):
                picks.append((float(t), float(v)))

        return picks

    def set_velocity_from_picks(self, il: int, xl: int, picks: List[Tuple[float, float]]) -> bool:
        """Update velocity function from edited picks using interpolation.

        Args:
            il, xl: Grid location
            picks: List of (time_ms, velocity) tuples

        Returns:
            True if successful
        """
        if self.velocities is None or len(picks) == 0:
            return False

        il_idx, xl_idx = self.get_grid_index(il, xl)

        # Sort picks by time
        picks_sorted = sorted(picks, key=lambda p: p[0])

        # Extract times and velocities
        pick_times = np.array([p[0] for p in picks_sorted])
        pick_vels = np.array([p[1] for p in picks_sorted])

        # Interpolate to full time array
        vel_interp = np.interp(self.t_coords, pick_times, pick_vels)

        # Update grid
        self.velocities[il_idx, xl_idx, :] = vel_interp.astype(np.float32)

        return True

    def save_to_zarr(self, path: str) -> bool:
        """Save velocity grid to zarr file.

        Args:
            path: Output path for zarr store

        Returns:
            True if successful
        """
        if self.velocities is None:
            return False

        try:
            # Create zarr group (v3 compatible)
            import shutil
            from pathlib import Path
            path_obj = Path(path)
            if path_obj.exists():
                shutil.rmtree(path_obj)
            root = zarr.open_group(str(path), mode='w')

            # Save velocity array (zarr v3 API)
            z = root.create_array('velocities', shape=self.velocities.shape,
                                  dtype=self.velocities.dtype,
                                  chunks=(1, 1, self.velocities.shape[2]))
            z[:] = self.velocities

            # Save coordinates
            for name, arr in [('il_coords', self.il_coords),
                              ('xl_coords', self.xl_coords),
                              ('t_coords', self.t_coords)]:
                z = root.create_array(name, shape=arr.shape, dtype=arr.dtype)
                z[:] = arr

            # Save metadata
            root.attrs['type'] = 'velocity_grid'
            root.attrs['n_il'] = len(self.il_coords)
            root.attrs['n_xl'] = len(self.xl_coords)
            root.attrs['n_time'] = len(self.t_coords)
            root.attrs['dt_ms'] = float(self.t_coords[1] - self.t_coords[0]) if len(self.t_coords) > 1 else 100.0

            return True
        except Exception as e:
            print(f"Error saving velocity grid: {e}")
            return False


class VelocityAnalysisWindow(QMainWindow):
    """Separate window for velocity analysis with super gathers and semblance."""

    # Signals for main window to update survey map
    velocity_loaded = pyqtSignal(list)  # Emitted with grid_locations list
    position_changed = pyqtSignal(int, int)  # Emitted with (il, xl)

    def __init__(self, parent=None, project: Optional['PSTMProject'] = None):
        super().__init__(parent)
        self.setWindowTitle("Velocity Analysis")
        self.setMinimumSize(1100, 750)

        # Project reference for auto-save
        self.project: Optional['PSTMProject'] = project

        # Data references
        self.offset_bins: List[zarr.Array] = []
        self.offset_values: np.ndarray = None
        self.t_coords: np.ndarray = None
        self.dt_ms = 2.0

        # Current position
        self.il_center = 0
        self.xl_center = 0

        # Super gather parameters
        self.il_half = 2  # 5x5 window
        self.xl_half = 2
        self.offset_bin_size = 50.0  # meters

        # Mute parameters
        self.v_top = 1500.0  # m/s
        self.v_bottom = 4000.0  # m/s
        self.top_mute_enabled = False
        self.bottom_mute_enabled = False

        # Processing parameters
        self.apply_bandpass = False
        self.f_low = 5.0
        self.f_high = 80.0
        self.apply_agc_flag = False
        self.agc_window = 500.0

        # Inverse NMO for semblance (uses initial velocity only)
        self.use_inverse_nmo = False

        # Forward NMO application (user control)
        self.nmo_applied = False
        self.stretch_percent = 30

        # Semblance parameters
        self.v_min = 1500.0
        self.v_max = 5000.0
        self.v_step = 100.0
        self.semblance_window_samples = 5

        # Display settings
        self.gather_colormap = "Seismic (BWR)"
        self.gather_clip = 99.0
        self.gather_gain = 1.0
        self.semblance_colormap = "Hot"  # High contrast for positive-only data
        self.semblance_clip = 99.0

        # Current data
        self.super_gather = None
        self.super_offsets = None
        self.processed_gather = None
        self.semblance = None
        self.velocities = None

        # Velocity output grid
        self.vel_grid = VelocityOutputGrid()
        self.initial_vel_grid: Optional[VelocityOutputGrid] = None  # Store initial velocity for NMO
        self.vel_model = None
        self.vel_metadata = None
        self.velocity_path: Optional[str] = None  # Path for saving edited velocity
        self.gathers_path: Optional[str] = None  # Path to loaded gathers for state recovery

        # Velocity picks - edits the loaded velocity directly
        self.current_picks: List[Tuple[float, float]] = []  # (time_ms, velocity) tuples
        self.edit_mode = False
        self._undo_stack: List[List[Tuple[float, float]]] = []
        self._redo_stack: List[List[Tuple[float, float]]] = []
        self._max_undo = 50

        # Preview state for live NMO
        self._preview_velocity: Optional[float] = None
        self._preview_time: Optional[float] = None

        # Live stack preview
        self._gather_cache = GatherCache()
        self._live_stack_updater = LiveStackUpdater(self._gather_cache)
        self._live_stack_direction: Optional[str] = None  # 'inline' or 'crossline'
        self._dragging_pick_index: Optional[int] = None  # Index of pick being dragged

        self._setup_ui()

    def _setup_ui(self):
        # Create menubar
        menubar = self.menuBar()

        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        semblance_settings_action = settings_menu.addAction("Semblance Parameters...")
        semblance_settings_action.triggered.connect(self._open_semblance_settings)

        # Velocity menu
        velocity_menu = menubar.addMenu("Velocity")
        load_velocity_action = velocity_menu.addAction("Load Velocity Model...")
        load_velocity_action.triggered.connect(self._open_velocity_dialog)

        self.save_velocity_action = velocity_menu.addAction("Save Edited Grid...")
        self.save_velocity_action.triggered.connect(self._save_velocity_grid)
        self.save_velocity_action.setEnabled(False)

        velocity_menu.addSeparator()

        self.vel_info_action = velocity_menu.addAction("No velocity loaded")
        self.vel_info_action.setEnabled(False)

        # Stacking menu
        stacking_menu = menubar.addMenu("Stacking")
        create_stack_action = stacking_menu.addAction("Create Stack...")
        create_stack_action.triggered.connect(self._open_stacking_dialog)

        stacking_menu.addSeparator()

        open_viewer_action = stacking_menu.addAction("Open Stack Viewer")
        open_viewer_action.triggered.connect(self._open_stack_viewer)

        # Stack viewer window reference
        self.stack_viewer = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: Controls
        controls = QWidget()
        controls.setMaximumWidth(220)
        ctrl_layout = QVBoxLayout(controls)

        # Position display with navigation
        pos_group = QGroupBox("Position")
        pos_layout = QVBoxLayout(pos_group)

        self.pos_label = QLabel("IL=0, XL=0")
        self.pos_label.setStyleSheet("font-weight: bold;")
        pos_layout.addWidget(self.pos_label)

        # Grid navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_grid_btn = QPushButton("< Prev")
        self.prev_grid_btn.clicked.connect(self._prev_grid_location)
        self.prev_grid_btn.setEnabled(False)
        self.prev_grid_btn.setToolTip("Go to previous velocity grid location")
        nav_layout.addWidget(self.prev_grid_btn)

        self.next_grid_btn = QPushButton("Next >")
        self.next_grid_btn.clicked.connect(self._next_grid_location)
        self.next_grid_btn.setEnabled(False)
        self.next_grid_btn.setToolTip("Go to next velocity grid location")
        nav_layout.addWidget(self.next_grid_btn)
        pos_layout.addLayout(nav_layout)

        self.grid_pos_label = QLabel("")
        self.grid_pos_label.setStyleSheet("color: #4a9eff; font-size: 10px;")
        pos_layout.addWidget(self.grid_pos_label)

        ctrl_layout.addWidget(pos_group)

        # Data loading
        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout(data_group)

        self.load_gathers_btn = QPushButton("Load Gathers...")
        self.load_gathers_btn.clicked.connect(self._load_gathers)
        data_layout.addWidget(self.load_gathers_btn)

        self.gathers_label = QLabel("No gathers loaded")
        self.gathers_label.setStyleSheet("color: gray; font-size: 9px;")
        self.gathers_label.setWordWrap(True)
        data_layout.addWidget(self.gathers_label)

        ctrl_layout.addWidget(data_group)

        # Super gather parameters
        sg_group = QGroupBox("Super Gather")
        sg_layout = QVBoxLayout(sg_group)

        il_layout = QHBoxLayout()
        il_layout.addWidget(QLabel("IL window:"))
        self.il_spin = QSpinBox()
        self.il_spin.setRange(1, 20)
        self.il_spin.setValue(5)
        self.il_spin.valueChanged.connect(self._on_param_changed)
        il_layout.addWidget(self.il_spin)
        sg_layout.addLayout(il_layout)

        xl_layout = QHBoxLayout()
        xl_layout.addWidget(QLabel("XL window:"))
        self.xl_spin = QSpinBox()
        self.xl_spin.setRange(1, 20)
        self.xl_spin.setValue(5)
        self.xl_spin.valueChanged.connect(self._on_param_changed)
        xl_layout.addWidget(self.xl_spin)
        sg_layout.addLayout(xl_layout)

        off_layout = QHBoxLayout()
        off_layout.addWidget(QLabel("Offset bin (m):"))
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(10, 500)
        self.offset_spin.setValue(50)
        self.offset_spin.setSingleStep(10)
        self.offset_spin.valueChanged.connect(self._on_param_changed)
        off_layout.addWidget(self.offset_spin)
        sg_layout.addLayout(off_layout)

        ctrl_layout.addWidget(sg_group)

        # Mute parameters
        mute_group = QGroupBox("Velocity Mute")
        mute_layout = QVBoxLayout(mute_group)

        self.top_mute_check = QCheckBox("Top Mute")
        self.top_mute_check.toggled.connect(self._on_mute_changed)
        mute_layout.addWidget(self.top_mute_check)

        vtop_layout = QHBoxLayout()
        vtop_layout.addWidget(QLabel("V top (m/s):"))
        self.vtop_spin = QSpinBox()
        self.vtop_spin.setRange(500, 10000)
        self.vtop_spin.setValue(1500)
        self.vtop_spin.setSingleStep(100)
        self.vtop_spin.valueChanged.connect(self._on_mute_changed)
        vtop_layout.addWidget(self.vtop_spin)
        mute_layout.addLayout(vtop_layout)

        self.bottom_mute_check = QCheckBox("Bottom Mute")
        self.bottom_mute_check.toggled.connect(self._on_mute_changed)
        mute_layout.addWidget(self.bottom_mute_check)

        vbot_layout = QHBoxLayout()
        vbot_layout.addWidget(QLabel("V bottom (m/s):"))
        self.vbot_spin = QSpinBox()
        self.vbot_spin.setRange(500, 10000)
        self.vbot_spin.setValue(4000)
        self.vbot_spin.setSingleStep(100)
        self.vbot_spin.valueChanged.connect(self._on_mute_changed)
        vbot_layout.addWidget(self.vbot_spin)
        mute_layout.addLayout(vbot_layout)

        ctrl_layout.addWidget(mute_group)

        # Processing parameters
        proc_group = QGroupBox("Processing")
        proc_layout = QVBoxLayout(proc_group)

        self.bp_check = QCheckBox("Bandpass Filter")
        self.bp_check.toggled.connect(self._on_processing_changed)
        proc_layout.addWidget(self.bp_check)

        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("f:"))
        self.f_low_spin = QSpinBox()
        self.f_low_spin.setRange(1, 100)
        self.f_low_spin.setValue(5)
        self.f_low_spin.valueChanged.connect(self._on_processing_changed)
        freq_layout.addWidget(self.f_low_spin)
        freq_layout.addWidget(QLabel("-"))
        self.f_high_spin = QSpinBox()
        self.f_high_spin.setRange(10, 200)
        self.f_high_spin.setValue(80)
        self.f_high_spin.valueChanged.connect(self._on_processing_changed)
        freq_layout.addWidget(self.f_high_spin)
        freq_layout.addWidget(QLabel("Hz"))
        proc_layout.addLayout(freq_layout)

        self.agc_check = QCheckBox("AGC")
        self.agc_check.toggled.connect(self._on_processing_changed)
        proc_layout.addWidget(self.agc_check)

        agc_layout = QHBoxLayout()
        agc_layout.addWidget(QLabel("Window (ms):"))
        self.agc_spin = QSpinBox()
        self.agc_spin.setRange(50, 2000)
        self.agc_spin.setValue(500)
        self.agc_spin.setSingleStep(50)
        self.agc_spin.valueChanged.connect(self._on_processing_changed)
        agc_layout.addWidget(self.agc_spin)
        proc_layout.addLayout(agc_layout)

        ctrl_layout.addWidget(proc_group)

        # NMO Correction group (forward NMO with loaded velocity)
        nmo_group = QGroupBox("NMO Correction")
        nmo_layout = QVBoxLayout(nmo_group)

        self.apply_nmo_btn = QPushButton("Apply NMO")
        self.apply_nmo_btn.setToolTip("Apply forward NMO using loaded velocity model")
        self.apply_nmo_btn.clicked.connect(self._apply_forward_nmo)
        self.apply_nmo_btn.setEnabled(False)  # Enabled when velocity loads
        nmo_layout.addWidget(self.apply_nmo_btn)

        stretch_layout = QHBoxLayout()
        stretch_layout.addWidget(QLabel("Stretch mute %:"))
        self.stretch_spin = QSpinBox()
        self.stretch_spin.setRange(0, 100)
        self.stretch_spin.setValue(30)
        self.stretch_spin.setSingleStep(5)
        self.stretch_spin.setToolTip("Mute samples with NMO stretch exceeding this percentage")
        stretch_layout.addWidget(self.stretch_spin)
        nmo_layout.addLayout(stretch_layout)

        self.nmo_applied_label = QLabel("")
        self.nmo_applied_label.setStyleSheet("color: #4a9eff; font-size: 10px;")
        nmo_layout.addWidget(self.nmo_applied_label)

        ctrl_layout.addWidget(nmo_group)

        # Semblance controls (only compute button, velocities moved to Settings)
        sem_group = QGroupBox("Semblance")
        sem_layout = QVBoxLayout(sem_group)

        self.compute_btn = QPushButton("Compute Semblance")
        self.compute_btn.clicked.connect(self._compute_semblance)
        sem_layout.addWidget(self.compute_btn)

        ctrl_layout.addWidget(sem_group)

        # Velocity Picking / Edit Mode controls
        edit_group = QGroupBox("Velocity Picking")
        edit_layout = QVBoxLayout(edit_group)

        self.edit_mode_btn = QPushButton("Edit Mode: OFF")
        self.edit_mode_btn.setCheckable(True)
        self.edit_mode_btn.setToolTip("Toggle edit mode (E)\nClick to add picks, drag to move, right-click to delete")
        self.edit_mode_btn.toggled.connect(self._toggle_edit_mode)
        edit_layout.addWidget(self.edit_mode_btn)

        self.snap_check = QCheckBox("Snap to max")
        self.snap_check.setToolTip("Snap picks to maximum semblance (S)")
        self.snap_check.toggled.connect(self._toggle_snap_mode)
        edit_layout.addWidget(self.snap_check)

        # Undo/Redo buttons
        undo_redo_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip("Undo last pick action (Ctrl+Z)")
        self.undo_btn.clicked.connect(self._undo)
        undo_redo_layout.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setEnabled(False)
        self.redo_btn.setToolTip("Redo last undone action (Ctrl+Shift+Z)")
        self.redo_btn.clicked.connect(self._redo)
        undo_redo_layout.addWidget(self.redo_btn)
        edit_layout.addLayout(undo_redo_layout)

        self.picks_label = QLabel("No picks")
        self.picks_label.setStyleSheet("color: #4a9eff; font-size: 10px;")
        edit_layout.addWidget(self.picks_label)

        ctrl_layout.addWidget(edit_group)

        # Display controls
        disp_group = QGroupBox("Display")
        disp_layout = QVBoxLayout(disp_group)

        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 100.0)
        self.gain_spin.setValue(1.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.valueChanged.connect(self._update_display)
        gain_layout.addWidget(self.gain_spin)
        disp_layout.addLayout(gain_layout)

        ctrl_layout.addWidget(disp_group)
        ctrl_layout.addStretch()

        layout.addWidget(controls)

        # Right: Canvases
        canvas_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Super gather canvas
        sg_frame = QFrame()
        sg_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        sg_layout_f = QVBoxLayout(sg_frame)
        sg_layout_f.addWidget(QLabel("Super Gather"))
        self.gather_canvas = SeismicCanvas()
        sg_layout_f.addWidget(self.gather_canvas)
        canvas_splitter.addWidget(sg_frame)

        # Semblance canvas (with velocity overlay)
        sem_frame = QFrame()
        sem_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        sem_layout_f = QVBoxLayout(sem_frame)
        sem_layout_f.addWidget(QLabel("Semblance"))
        self.semblance_canvas = SeismicCanvas()
        sem_layout_f.addWidget(self.semblance_canvas)
        canvas_splitter.addWidget(sem_frame)

        # Connect semblance canvas signals for velocity picking
        self.semblance_canvas.pick_added.connect(self._on_pick_added)
        self.semblance_canvas.pick_moved.connect(self._on_pick_moved)
        self.semblance_canvas.pick_removed.connect(self._on_pick_removed)
        self.semblance_canvas.pick_drag_started.connect(self._on_pick_drag_started)
        self.semblance_canvas.pick_drag_ended.connect(self._on_pick_drag_ended)
        self.semblance_canvas.pick_drag_update.connect(self._on_pick_drag_update)
        self.semblance_canvas.preview_velocity_changed.connect(self._on_preview_velocity_changed)
        self.semblance_canvas.preview_ended.connect(self._on_preview_ended)

        layout.addWidget(canvas_splitter, 1)

        # Setup keyboard shortcuts
        self._setup_shortcuts()

        self.statusBar().showMessage("Ready")

    def set_gather_data(self, offset_bins: List[zarr.Array], offset_values: np.ndarray,
                        t_coords: np.ndarray):
        """Set the gather data source."""
        self.offset_bins = offset_bins
        self.offset_values = offset_values
        self.t_coords = t_coords
        self.gathers_label.setText(f"{len(offset_bins)} offset bins loaded")

    def _load_gathers(self):
        """Load gathers from folder with metadata (via file dialog)."""
        path = QFileDialog.getExistingDirectory(
            self, "Open Common Offset Gathers Directory",
            str(Path.home() / "SeismicData")
        )
        if not path:
            return

        self.load_gathers_from_path(path)

    def load_gathers_from_path(self, path_str: str, set_center_position: bool = True):
        """Load gathers from a given path.

        Args:
            path_str: Path to the gathers folder
            set_center_position: If True, set position to center of survey
        """
        path = Path(path_str)
        metadata_path = path / 'gather_metadata.json'

        if not metadata_path.exists():
            self.statusBar().showMessage("No gather_metadata.json found in folder")
            return False

        try:
            self.statusBar().showMessage("Loading gathers...")
            QApplication.processEvents()

            with open(metadata_path) as f:
                metadata = json.load(f)

            self.offset_bins = []
            data_array_name = metadata.get('data_array', 'migrated_stack.zarr')

            for offset_info in metadata['offsets']:
                bin_path = path / offset_info['bin_name'] / data_array_name
                z = zarr.open(str(bin_path), mode='r')
                self.offset_bins.append(z)

            self.offset_values = np.array([o['offset_m'] for o in metadata['offsets']])

            dims = metadata['dimensions']
            self.t_coords = np.linspace(
                metadata['coordinates']['time']['min'],
                metadata['coordinates']['time']['max'],
                dims['n_time']
            )
            self.dt_ms = self.t_coords[1] - self.t_coords[0] if len(self.t_coords) > 1 else 2.0

            n_off = len(self.offset_bins)
            self.gathers_label.setText(f"{path.name}\n{n_off} offsets")
            self.statusBar().showMessage(f"Loaded {n_off} offset bins")

            # Store path for state recovery
            self.gathers_path = str(path)

            if set_center_position:
                self.il_center = dims['n_inline'] // 2
                self.xl_center = dims['n_crossline'] // 2

            self._compute_super_gather()
            self._update_display()
            return True

        except Exception as e:
            self.statusBar().showMessage(f"Error loading gathers: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _open_velocity_dialog(self):
        """Open velocity load dialog with grid configuration."""
        # Pre-fill with data dimensions if available
        current_settings = {}
        if len(self.offset_bins) > 0:
            dims = self.offset_bins[0].shape
            current_settings = {
                'il_start': 0,
                'il_end': dims[0] - 1,
                'il_step': max(1, dims[0] // 10),
                'xl_start': 0,
                'xl_end': dims[1] - 1,
                'xl_step': max(1, dims[1] // 10),
                'output_dt_ms': 100,
            }

        dialog = VelocityGridDialog(self, current_settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()

            if settings['velocity_path'] is None:
                self.statusBar().showMessage("No velocity file selected")
                return

            # Get time range from loaded data
            t_min = self.t_coords[0] if len(self.t_coords) > 0 else 0.0
            t_max = self.t_coords[-1] if len(self.t_coords) > 0 else 4000.0

            # Setup the output grid with output sample rate
            self.vel_grid.setup_grid(
                settings['il_start'], settings['il_end'], settings['il_step'],
                settings['xl_start'], settings['xl_end'], settings['xl_step'],
                settings['output_dt_ms'], t_min, t_max
            )

            # Load velocity model
            self.statusBar().showMessage("Loading velocity model...")
            QApplication.processEvents()

            self.vel_model, self.vel_metadata = load_velocity_model(settings['velocity_path'])
            self.velocity_path = settings['velocity_path']  # Store for saving

            if self.vel_model is not None:
                # Resample onto grid
                self.vel_grid.load_velocity(self.vel_model, self.vel_metadata)

                # Store initial velocity for NMO (deep copy)
                self.initial_vel_grid = VelocityOutputGrid()
                self.initial_vel_grid.il_coords = self.vel_grid.il_coords.copy()
                self.initial_vel_grid.xl_coords = self.vel_grid.xl_coords.copy()
                self.initial_vel_grid.t_coords = self.vel_grid.t_coords.copy()
                self.initial_vel_grid.velocities = self.vel_grid.velocities.copy()
                self.initial_vel_grid.grid_locations = list(self.vel_grid.grid_locations)

                # Save initial velocity to project
                if self.project:
                    self._save_initial_velocity_to_project()
                    # Also save grid config to project
                    self.project.set_grid_config(
                        settings['il_start'], settings['il_end'], settings['il_step'],
                        settings['xl_start'], settings['xl_end'], settings['xl_step'],
                        settings['output_dt_ms'], t_min, t_max
                    )

                # Load picks from grid for current location
                self._load_picks_from_grid()

                # Enable navigation
                self.prev_grid_btn.setEnabled(True)
                self.next_grid_btn.setEnabled(True)

                # Enable Apply NMO button and Save action
                self.apply_nmo_btn.setEnabled(True)
                self.save_velocity_action.setEnabled(True)

                # Auto-enable inverse NMO for semblance calculation
                self.use_inverse_nmo = True

                # Update menu info
                n_locs = len(self.vel_grid.grid_locations)
                self.vel_info_action.setText(f"Grid: {n_locs} locations")
                self._update_grid_label()

                self.statusBar().showMessage(
                    f"Velocity loaded: {n_locs} grid locations (Inverse NMO enabled)"
                )

                # Move to first grid location and compute semblance
                il, xl = self.vel_grid.get_current_location()
                self.set_position(il, xl)
                self._compute_semblance()

                # Emit signal for main window to update survey map
                self.velocity_loaded.emit(list(self.vel_grid.grid_locations))
            else:
                self.statusBar().showMessage("Failed to load velocity")

    def _save_velocity_grid(self):
        """Save the edited velocity grid to a zarr file."""
        if not self.vel_grid.has_velocity():
            self.statusBar().showMessage("No velocity grid to save")
            return

        # Get save path
        default_path = str(Path.home() / "SeismicData" / "velocity_edited.zarr")
        if self.velocity_path:
            # Suggest a modified name based on original
            orig_path = Path(self.velocity_path)
            default_path = str(orig_path.parent / f"{orig_path.stem}_edited.zarr")

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Velocity Grid",
            default_path,
            "Zarr Directory (*)"
        )

        if not path:
            return

        # Ensure .zarr extension
        if not path.endswith('.zarr'):
            path += '.zarr'

        self.statusBar().showMessage(f"Saving velocity grid to {path}...")
        QApplication.processEvents()

        if self.vel_grid.save_to_zarr(path):
            self.statusBar().showMessage(f"Velocity grid saved to {path}")
        else:
            self.statusBar().showMessage("Failed to save velocity grid")

    def _prev_grid_location(self):
        """Navigate to previous grid location."""
        il, xl = self.vel_grid.prev_location()
        self.set_position(il, xl)
        self._update_grid_label()
        self._compute_semblance()

    def _next_grid_location(self):
        """Navigate to next grid location."""
        il, xl = self.vel_grid.next_location()
        self.set_position(il, xl)
        self._update_grid_label()
        self._compute_semblance()

    def _update_grid_label(self):
        """Update grid position label."""
        if len(self.vel_grid.grid_locations) > 0:
            idx = self.vel_grid.current_idx
            total = len(self.vel_grid.grid_locations)
            self.grid_pos_label.setText(f"Grid: {idx + 1}/{total}")
        else:
            self.grid_pos_label.setText("")

    def set_position(self, il: int, xl: int, auto_compute: bool = False):
        """Set position and update display.

        Args:
            il: Inline position
            xl: Crossline position
            auto_compute: If True, automatically compute semblance after setting position
        """
        self.il_center = il
        self.xl_center = xl
        self.pos_label.setText(f"IL={il}, XL={xl}")
        self._compute_super_gather()
        self._update_display()

        # Load picks for new position
        self._load_picks_for_position()

        if auto_compute:
            self._compute_semblance()

        # Emit signal for main window to update survey map
        self.position_changed.emit(il, xl)

    def _on_param_changed(self):
        """Handle parameter changes."""
        self.il_half = self.il_spin.value() // 2
        self.xl_half = self.xl_spin.value() // 2
        self.offset_bin_size = self.offset_spin.value()
        self._compute_super_gather()
        self._update_display()

    def _on_mute_changed(self):
        """Handle mute parameter changes."""
        self.top_mute_enabled = self.top_mute_check.isChecked()
        self.bottom_mute_enabled = self.bottom_mute_check.isChecked()
        self.v_top = self.vtop_spin.value()
        self.v_bottom = self.vbot_spin.value()
        self._update_display()
        self._compute_semblance()

    def _on_processing_changed(self):
        """Handle processing parameter changes."""
        self.apply_bandpass = self.bp_check.isChecked()
        self.f_low = self.f_low_spin.value()
        self.f_high = self.f_high_spin.value()
        self.apply_agc_flag = self.agc_check.isChecked()
        self.agc_window = self.agc_spin.value()
        self._update_display()
        self._compute_semblance()

    def _apply_forward_nmo(self):
        """Toggle forward NMO application on gather display."""
        if not self.vel_grid.has_velocity():
            self.statusBar().showMessage("No velocity model loaded")
            return

        # Toggle NMO application
        self.nmo_applied = not self.nmo_applied
        self.stretch_percent = self.stretch_spin.value()

        if self.nmo_applied:
            self.apply_nmo_btn.setText("Remove NMO")
            self.nmo_applied_label.setText(f"NMO applied (stretch mute: {self.stretch_percent}%)")
            self.statusBar().showMessage(
                f"Forward NMO applied with {self.stretch_percent}% stretch mute"
            )
        else:
            self.apply_nmo_btn.setText("Apply NMO")
            self.nmo_applied_label.setText("")
            self.statusBar().showMessage("NMO removed from gather display")

        self._update_display()

    def _compute_super_gather(self):
        """Compute super gather at current position."""
        if len(self.offset_bins) == 0:
            return

        self.statusBar().showMessage("Computing super gather...")
        QApplication.processEvents()

        self.super_gather, self.super_offsets = create_super_gather(
            self.offset_bins, self.il_center, self.xl_center,
            self.il_half, self.xl_half, self.offset_values,
            self.offset_bin_size
        )

        self.statusBar().showMessage(f"Super gather: {len(self.super_offsets)} offset bins")

    def _compute_semblance(self):
        """Compute semblance on processed gather."""
        if self.super_gather is None:
            return

        self.statusBar().showMessage("Computing semblance...")
        QApplication.processEvents()

        gather = self.super_gather.copy()

        # Apply inverse NMO before semblance using INITIAL velocity only
        if self.use_inverse_nmo and self.initial_vel_grid is not None:
            vel_func = self.initial_vel_grid.get_velocity_at(self.il_center, self.xl_center)
            if vel_func is not None:
                gather = apply_nmo_with_velocity_model(
                    gather, self.super_offsets, self.t_coords,
                    vel_func, inverse=True,
                    vel_t_coords=self.initial_vel_grid.t_coords
                )

        # Apply velocity mute if enabled
        if self.top_mute_enabled or self.bottom_mute_enabled:
            v_top = self.v_top if self.top_mute_enabled else None
            v_bottom = self.v_bottom if self.bottom_mute_enabled else None
            gather = apply_velocity_mute(
                gather, self.super_offsets, self.t_coords,
                v_top, v_bottom
            )

        self.semblance, self.velocities = compute_semblance_fast(
            gather, self.super_offsets, self.t_coords,
            self.v_min, self.v_max, self.v_step,
            self.semblance_window_samples
        )

        self._update_semblance_display()
        inv_nmo_str = " (with inverse NMO)" if self.use_inverse_nmo else ""
        self.statusBar().showMessage(
            f"Semblance computed{inv_nmo_str} (v={self.v_min}-{self.v_max}, step={self.v_step})"
        )

    def _open_semblance_settings(self):
        """Open the semblance settings dialog."""
        current_settings = {
            'v_min': self.v_min,
            'v_max': self.v_max,
            'v_step': self.v_step,
            'window_samples': self.semblance_window_samples,
            'use_inverse_nmo': self.use_inverse_nmo,
            'gather_colormap': self.gather_colormap,
            'gather_clip': self.gather_clip,
            'gather_gain': self.gather_gain,
            'semblance_colormap': self.semblance_colormap,
            'semblance_clip': self.semblance_clip,
            'velocity_colormap': 'Viridis',
            'velocity_clip': 99.0,
            'velocity_use_minmax': False,
            'velocity_vmin': 1500.0,
            'velocity_vmax': 5000.0,
        }

        dialog = SemblanceSettingsDialog(self, current_settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_settings = dialog.get_settings()

            self.v_min = new_settings['v_min']
            self.v_max = new_settings['v_max']
            self.v_step = new_settings['v_step']
            self.semblance_window_samples = new_settings['window_samples']

            # Inverse NMO setting (uses initial velocity only, no fallback)
            self.use_inverse_nmo = new_settings['use_inverse_nmo']

            self.gather_colormap = new_settings['gather_colormap']
            self.gather_clip = new_settings['gather_clip']
            self.gather_gain = new_settings['gather_gain']
            self.semblance_colormap = new_settings['semblance_colormap']
            self.semblance_clip = new_settings['semblance_clip']

            self.gain_spin.setValue(self.gather_gain)

            self._update_display()
            self._compute_semblance()  # Recompute with new inverse NMO setting

            self.statusBar().showMessage("Settings updated")

    def _update_display(self):
        """Update the gather display with full processing chain."""
        if self.super_gather is None:
            return

        gather = self.super_gather.copy()
        dt_ms = self.t_coords[1] - self.t_coords[0] if len(self.t_coords) > 1 else 2.0

        # Get INITIAL velocity function for inverse NMO
        initial_vel_func = None
        if self.initial_vel_grid is not None:
            initial_vel_func = self.initial_vel_grid.get_velocity_at(self.il_center, self.xl_center)

        # Get EDITED velocity function for forward NMO (so user sees effect of picks)
        edited_vel_func = None
        if self.vel_grid.has_velocity():
            edited_vel_func = self.vel_grid.get_velocity_at(self.il_center, self.xl_center)

        # 1. Apply inverse NMO using INITIAL velocity only
        if self.use_inverse_nmo and initial_vel_func is not None:
            gather = apply_nmo_with_velocity_model(
                gather, self.super_offsets, self.t_coords,
                initial_vel_func, inverse=True,
                vel_t_coords=self.initial_vel_grid.t_coords
            )

        # 2. Apply forward NMO using EDITED velocity (so picks affect display)
        if self.nmo_applied and edited_vel_func is not None:
            gather = apply_nmo_with_velocity_model(
                gather, self.super_offsets, self.t_coords,
                edited_vel_func, inverse=False, stretch_mute_percent=self.stretch_percent,
                vel_t_coords=self.vel_grid.t_coords
            )

        # 3. Apply bandpass filter if enabled
        if self.apply_bandpass:
            gather = apply_bandpass_filter(
                gather, dt_ms, self.f_low, self.f_high
            )

        # 3. Apply AGC if enabled
        if self.apply_agc_flag:
            gather = apply_agc(gather, self.agc_window, dt_ms)

        # 4. Apply velocity mute if enabled
        if self.top_mute_enabled or self.bottom_mute_enabled:
            v_top = self.v_top if self.top_mute_enabled else None
            v_bottom = self.v_bottom if self.bottom_mute_enabled else None
            gather = apply_velocity_mute(
                gather, self.super_offsets, self.t_coords,
                v_top, v_bottom
            )

        self.processed_gather = gather

        # Display gather
        data = gather.T

        off_min = self.super_offsets[0] if len(self.super_offsets) > 0 else 0
        off_max = self.super_offsets[-1] if len(self.super_offsets) > 0 else 100
        t_min = self.t_coords[0] if len(self.t_coords) > 0 else 0
        t_max = self.t_coords[-1] if len(self.t_coords) > 0 else 2000

        x_axis = AxisConfig("Offset", off_min, off_max, "m")
        y_axis = AxisConfig("Time", t_min, t_max, "ms")

        self.gather_canvas.set_palette(self.gather_colormap)
        self.gather_canvas.set_clip_percentile(self.gather_clip)
        self.gather_canvas.set_gain(self.gain_spin.value())
        self.gather_canvas.set_data(data, "gather", 0, 0, x_axis, y_axis, 0)
        self.gather_canvas.slice_direction = f"Super Gather IL={self.il_center}, XL={self.xl_center}"

    def _update_semblance_display(self):
        """Update the semblance display with velocity overlay if on grid."""
        if self.semblance is None:
            return

        data = self.semblance.T

        v_min = self.velocities[0] if len(self.velocities) > 0 else 1500
        v_max = self.velocities[-1] if len(self.velocities) > 0 else 5000
        t_min = self.t_coords[0] if len(self.t_coords) > 0 else 0
        t_max = self.t_coords[-1] if len(self.t_coords) > 0 else 2000

        x_axis = AxisConfig("Velocity", v_min, v_max, "m/s")
        y_axis = AxisConfig("Time", t_min, t_max, "ms")

        self.semblance_canvas.set_palette(self.semblance_colormap)
        self.semblance_canvas.set_clip_percentile(self.semblance_clip)
        self.semblance_canvas.set_data(data, "semblance", 0, 0, x_axis, y_axis, 0)
        self.semblance_canvas.slice_direction = "Semblance"

        # Enable colorbar with actual semblance range
        sem_min = float(np.min(self.semblance))
        sem_max = float(np.max(self.semblance))
        self.semblance_canvas.set_colorbar(True, sem_min, sem_max)

        # Draw velocity curve overlay if at a grid location
        if self.vel_grid.has_velocity():
            vel_func = self.vel_grid.get_velocity_at(self.il_center, self.xl_center)
            if vel_func is not None:
                # Use vel_grid.t_coords since vel_func is sampled at those times
                self.semblance_canvas.set_velocity_overlay(vel_func, self.vel_grid.t_coords)
            else:
                self.semblance_canvas.clear_velocity_overlay()
        else:
            self.semblance_canvas.clear_velocity_overlay()

        # Update picks display
        self._update_picks_display()

    # ==================== Velocity Picking Methods ====================

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for velocity picking."""
        # Edit mode toggle
        edit_shortcut = QShortcut(QKeySequence("E"), self)
        edit_shortcut.activated.connect(self._toggle_edit_mode_shortcut)

        # Snap toggle
        snap_shortcut = QShortcut(QKeySequence("S"), self)
        snap_shortcut.activated.connect(self._toggle_snap_shortcut)

        # Undo/Redo
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self._undo)

        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.activated.connect(self._redo)

    def _toggle_edit_mode_shortcut(self):
        """Toggle edit mode via keyboard shortcut."""
        self.edit_mode_btn.toggle()

    def _toggle_snap_shortcut(self):
        """Toggle snap mode via keyboard shortcut."""
        if self.edit_mode:
            self.snap_check.toggle()

    def _toggle_edit_mode(self, enabled: bool):
        """Enable or disable velocity picking edit mode."""
        self.edit_mode = enabled
        self.semblance_canvas.set_edit_mode(enabled)

        if enabled:
            self.edit_mode_btn.setText("Edit Mode: ON")
            self.edit_mode_btn.setStyleSheet("background-color: #664444;")
            self.statusBar().showMessage(
                "Edit Mode: Click to add picks, drag to move, right-click to delete, E to exit"
            )
        else:
            self.edit_mode_btn.setText("Edit Mode: OFF")
            self.edit_mode_btn.setStyleSheet("")
            self.statusBar().showMessage("Edit mode disabled")

        self._update_picks_display()

    def _toggle_snap_mode(self, enabled: bool):
        """Enable or disable snap to maximum."""
        self.semblance_canvas.set_snap_to_max(enabled)

    def _load_picks_from_grid(self):
        """Load picks from vel_grid for current IL/XL position."""
        if not self.vel_grid.has_velocity():
            self.current_picks = []
            return

        # Extract velocity as sparse picks
        self.current_picks = self.vel_grid.get_velocity_as_picks(
            self.il_center, self.xl_center, max_picks=20
        )
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_picks_display()

    def _load_picks_for_position(self):
        """Load picks for current IL/XL position (from grid)."""
        self._load_picks_from_grid()

    def _save_picks_to_grid(self):
        """Save current picks back to vel_grid."""
        if not self.vel_grid.has_velocity() or len(self.current_picks) == 0:
            return False

        # Update grid with picks
        success = self.vel_grid.set_velocity_from_picks(
            self.il_center, self.xl_center, self.current_picks
        )

        if success:
            # Update displays to show new velocity
            self._update_semblance_display()
            self._update_display()

            # Auto-save to project if available
            self._save_edited_velocity_to_project()

        return success

    def _push_undo_state(self):
        """Push current picks to undo stack."""
        state = list(self.current_picks)  # Copy list
        self._undo_stack.append(state)
        self._redo_stack.clear()

        # Limit stack size
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

        self._update_undo_buttons()

    def _update_undo_buttons(self):
        """Update undo/redo button states."""
        self.undo_btn.setEnabled(len(self._undo_stack) > 0)
        self.redo_btn.setEnabled(len(self._redo_stack) > 0)

    def _update_picks_display(self):
        """Update the picks displayed on the semblance canvas."""
        if self.current_picks:
            self.semblance_canvas.set_picks(self.current_picks)
            self.picks_label.setText(f"{len(self.current_picks)} picks")
        else:
            self.semblance_canvas.clear_picks()
            self.picks_label.setText("No picks")

        self._update_undo_buttons()

    def _on_picks_changed(self):
        """Callback when picks are changed - save to grid and update displays."""
        self._save_picks_to_grid()
        self._update_picks_display()

    def _on_pick_added(self, time_ms: float, velocity: float):
        """Handle adding a new pick."""
        if not self.vel_grid.has_velocity():
            self.statusBar().showMessage("No velocity loaded - cannot add picks")
            return

        # Save state for undo
        self._push_undo_state()

        # Add pick maintaining sort order
        self.current_picks.append((time_ms, velocity))
        self.current_picks.sort(key=lambda p: p[0])

        # Save to grid and update displays
        self._on_picks_changed()
        self.statusBar().showMessage(f"Pick added: t={time_ms:.0f}ms, v={velocity:.0f}m/s")

    def _on_pick_moved(self, index: int, time_ms: float, velocity: float):
        """Handle moving a pick - update grid and gather display in real-time."""
        if 0 <= index < len(self.current_picks):
            self.current_picks[index] = (time_ms, velocity)
            self.current_picks.sort(key=lambda p: p[0])

            # Save to grid and update displays so user sees NMO effect immediately
            if self.vel_grid.has_velocity() and len(self.current_picks) > 0:
                self.vel_grid.set_velocity_from_picks(
                    self.il_center, self.xl_center, self.current_picks
                )
                self._update_display()  # Update gather with new velocity
                self._update_semblance_display()  # Update velocity overlay

    def _on_pick_removed(self, index: int):
        """Handle removing a pick."""
        if 0 <= index < len(self.current_picks):
            # Save state for undo
            self._push_undo_state()

            removed = self.current_picks.pop(index)
            self._on_picks_changed()
            self.statusBar().showMessage(f"Pick removed: t={removed[0]:.0f}ms")

    def _on_pick_drag_started(self, index: int):
        """Handle start of pick drag - save undo state and track drag."""
        self._push_undo_state()
        self._dragging_pick_index = index

    def _on_pick_drag_ended(self, index: int):
        """Handle end of pick drag - save picks to grid and update stack."""
        self._dragging_pick_index = None
        self._save_picks_to_grid()

        # Update stack with final picks using spatial interpolation
        if (self.stack_viewer is not None and self.stack_viewer.isVisible()
                and self.offset_bins is not None and len(self.offset_bins) > 0):
            self._update_stack_with_current_picks()

        self.statusBar().showMessage("Pick moved and saved to grid")

    def _on_pick_drag_update(self, index: int, time_ms: float, velocity: float):
        """Handle pick drag update - update live stack with spatial interpolation."""
        # Only update stack during dragging when stack viewer is open
        if (self.stack_viewer is None or not self.stack_viewer.isVisible()
                or self.offset_bins is None or len(self.offset_bins) == 0):
            return

        self._preview_time = time_ms
        self._preview_velocity = velocity

        # Use spatial velocity interpolation for live stack
        self._update_live_stack_with_modified_pick(index, time_ms, velocity)

        self.statusBar().showMessage(f"Dragging pick {index}: t={time_ms:.0f}ms, v={velocity:.0f}m/s")

    def _on_preview_velocity_changed(self, time_ms: float, velocity: float):
        """Handle live preview velocity update - only used for gather display now."""
        self._preview_time = time_ms
        self._preview_velocity = velocity
        # Stack updates are now handled by _on_pick_drag_update during dragging

    def _on_preview_ended(self):
        """Handle mouse leaving semblance - revert to current picks."""
        self._preview_time = None
        self._preview_velocity = None
        self._dragging_pick_index = None

        # Revert stack to current picks if stack viewer is open and data is loaded
        if (self.stack_viewer is not None and self.stack_viewer.isVisible()
                and self.offset_bins is not None and len(self.offset_bins) > 0):
            self._update_stack_with_current_picks()

    def _ensure_gather_cache(self, direction: str) -> bool:
        """Ensure gather cache is initialized for the given direction."""
        if direction == 'inline':
            if self._gather_cache.is_valid_for(self.il_center, self.xl_center, 'inline'):
                return True
            # Build cache for current inline
            settings = self._get_current_processing_settings()
            return self._gather_cache.cache_inline(
                self.offset_bins, self.offset_values, self.t_coords,
                self.il_center, self.initial_vel_grid, settings
            )
        else:  # crossline
            if self._gather_cache.is_valid_for(self.il_center, self.xl_center, 'crossline'):
                return True
            # Build cache for current crossline
            settings = self._get_current_processing_settings()
            return self._gather_cache.cache_crossline(
                self.offset_bins, self.offset_values, self.t_coords,
                self.xl_center, self.initial_vel_grid, settings
            )

    def _get_current_processing_settings(self) -> dict:
        """Get current processing settings for stacking."""
        return {
            'stretch_percent': self.stretch_percent,
            'top_mute_enabled': self.top_mute_enabled,
            'v_top': self.v_top,
            'bottom_mute_enabled': self.bottom_mute_enabled,
            'v_bottom': self.v_bottom,
            'apply_bandpass': self.apply_bandpass,
            'f_low': self.f_low,
            'f_high': self.f_high,
            'apply_agc': self.apply_agc_flag,
            'agc_window': self.agc_window,
        }

    def _update_live_stack_with_modified_pick(self, pick_index: int, new_time: float, new_velocity: float):
        """Update stack viewer with live preview stack using spatial velocity interpolation."""
        if self.stack_viewer is None or not self.stack_viewer.isVisible():
            return
        if self.offset_bins is None or len(self.offset_bins) == 0:
            return

        # Use inline direction by default (can be made configurable)
        direction = self._live_stack_direction or 'inline'

        # Ensure cache is ready
        if not self._ensure_gather_cache(direction):
            return

        # Get velocity time coords
        vel_t_coords = self.vel_grid.t_coords if self.vel_grid.has_velocity() else self.t_coords

        # Compute stack with modified pick using spatial interpolation
        stack = self._live_stack_updater.compute_stack_with_modified_pick(
            self.vel_grid,
            self.il_center, self.xl_center,
            self.current_picks, pick_index,
            new_time, new_velocity,
            vel_t_coords, self.stretch_percent
        )

        if stack is not None:
            # Update stack viewer with preview
            self._send_stack_to_viewer(stack, direction, is_preview=True)

    def _update_stack_with_current_picks(self):
        """Update stack viewer with stack based on current picks using spatial interpolation."""
        if self.stack_viewer is None or not self.stack_viewer.isVisible():
            return

        direction = self._live_stack_direction or 'inline'

        if not self._ensure_gather_cache(direction):
            return

        # Get velocity time coords
        vel_t_coords = self.vel_grid.t_coords if self.vel_grid.has_velocity() else self.t_coords

        if len(self.current_picks) < 1:
            return  # Need at least 1 pick

        # Create velocity function from current picks
        if len(self.current_picks) == 1:
            velocity_func = np.full(len(vel_t_coords), self.current_picks[0][1], dtype=np.float32)
        else:
            pick_times = np.array([p[0] for p in self.current_picks])
            pick_vels = np.array([p[1] for p in self.current_picks])
            velocity_func = np.interp(vel_t_coords, pick_times, pick_vels).astype(np.float32)

        # Use spatial interpolation with current velocity at this location
        stack = self._live_stack_updater.compute_stack_with_spatial_velocities(
            self.vel_grid,
            self.il_center, self.xl_center,
            velocity_func, vel_t_coords, self.stretch_percent
        )

        if stack is not None:
            self._send_stack_to_viewer(stack, direction, is_preview=False)

    def _send_stack_to_viewer(self, stack: np.ndarray, direction: str, is_preview: bool = False):
        """Send computed stack to stack viewer."""
        if self.stack_viewer is None:
            return

        # Build coordinates
        coords = {'t_coords': self.t_coords}
        if direction == 'inline':
            coords['xl_coords'] = self._gather_cache.trace_coords
            coords['il_coords'] = np.array([self.il_center])
            name = f"Live IL{self.il_center}" if is_preview else f"IL{self.il_center}"
        else:
            coords['il_coords'] = self._gather_cache.trace_coords
            coords['xl_coords'] = np.array([self.xl_center])
            name = f"Live XL{self.xl_center}" if is_preview else f"XL{self.xl_center}"

        # Update or add stack to viewer
        self.stack_viewer.set_live_preview(name, stack, coords)

    def _undo(self):
        """Undo last pick action."""
        if not self._undo_stack:
            return

        # Save current state to redo
        self._redo_stack.append(list(self.current_picks))

        # Restore previous state
        self.current_picks = self._undo_stack.pop()
        self._on_picks_changed()
        self.statusBar().showMessage("Undo")

    def _redo(self):
        """Redo last undone action."""
        if not self._redo_stack:
            return

        # Save current state to undo
        self._undo_stack.append(list(self.current_picks))

        # Restore redo state
        self.current_picks = self._redo_stack.pop()
        self._on_picks_changed()
        self.statusBar().showMessage("Redo")

    # ==================== Stacking Methods ====================

    def _open_stacking_dialog(self):
        """Open dialog to configure and create stack."""
        if not self.vel_grid.has_velocity():
            self.statusBar().showMessage("Load velocity first to create stack")
            return

        if len(self.offset_bins) == 0:
            self.statusBar().showMessage("No gathers loaded")
            return

        # Get current settings from panel
        current_settings = {
            'current_il': self.il_center,
            'current_xl': self.xl_center,
            'top_mute_enabled': self.top_mute_enabled,
            'v_top': self.v_top,
            'bottom_mute_enabled': self.bottom_mute_enabled,
            'v_bottom': self.v_bottom,
            'stretch_percent': self.stretch_percent,
            'apply_bandpass': self.apply_bandpass,
            'f_low': self.f_low,
            'f_high': self.f_high,
            'apply_agc': self.apply_agc_flag,
            'agc_window': self.agc_window,
            'default_name': f'stack_IL{self.il_center}',
        }

        dialog = StackingDialog(self, current_settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self._create_stack(settings)

    def _create_stack(self, settings: dict):
        """Create stack with given settings."""
        self.statusBar().showMessage("Creating stack...")
        QApplication.processEvents()

        # Choose velocity grid
        if settings['use_initial_velocity']:
            vel_grid = self.initial_vel_grid
            vel_type = "initial"
        else:
            vel_grid = self.vel_grid
            vel_type = "edited"

        if vel_grid is None or not vel_grid.has_velocity():
            self.statusBar().showMessage("No velocity available")
            return

        # Determine QC directory for saving velocity QC images
        qc_dir = None
        if self.project:
            qc_dir = Path(self.project.path) / "qc" / "velocity_interp"
            qc_dir.mkdir(parents=True, exist_ok=True)

        output_name = settings['output_name']

        # Determine scope and compute stack
        scope = settings['data_scope']
        stack_data = None
        coords = {}

        try:
            if scope == 'inline':
                # Stack current inline (all crosslines)
                # Sequence: Inverse NMO (initial) -> Processing -> Forward NMO (selected) -> Stack
                stack_data, xl_coords = compute_inline_stack(
                    self.offset_bins, self.offset_values, self.t_coords,
                    vel_grid, self.il_center, settings,
                    progress_callback=self._stack_progress,
                    qc_dir=qc_dir,
                    stack_name=output_name,
                    initial_velocity_grid=self.initial_vel_grid
                )
                coords['xl_coords'] = xl_coords
                coords['t_coords'] = self.t_coords
                coords['il_coords'] = np.array([self.il_center])

            elif scope == 'crossline':
                # Stack current crossline (all inlines)
                # Sequence: Inverse NMO (initial) -> Processing -> Forward NMO (selected) -> Stack
                stack_data, il_coords = compute_crossline_stack(
                    self.offset_bins, self.offset_values, self.t_coords,
                    vel_grid, self.xl_center, settings,
                    progress_callback=self._stack_progress,
                    qc_dir=qc_dir,
                    stack_name=output_name,
                    initial_velocity_grid=self.initial_vel_grid
                )
                coords['il_coords'] = il_coords
                coords['t_coords'] = self.t_coords
                coords['xl_coords'] = np.array([self.xl_center])

            elif scope == 'full':
                # Full volume stack
                n_il, n_xl, _ = self.offset_bins[0].shape
                il_range = (0, n_il - 1, 1)
                xl_range = (0, n_xl - 1, 1)

                stack_data, il_coords, xl_coords = compute_stack(
                    self.offset_bins, self.offset_values, self.t_coords,
                    vel_grid, il_range, xl_range, settings,
                    progress_callback=self._stack_progress
                )
                coords['il_coords'] = il_coords
                coords['xl_coords'] = xl_coords
                coords['t_coords'] = self.t_coords

        except Exception as e:
            self.statusBar().showMessage(f"Stacking failed: {e}")
            import traceback
            traceback.print_exc()
            return

        if stack_data is None:
            self.statusBar().showMessage("Stacking produced no data")
            return

        # Save stack to zarr
        self._save_and_show_stack(output_name, stack_data, coords, vel_type, settings)

    def _stack_progress(self, current: int, total: int, message: str):
        """Progress callback for stacking."""
        self.statusBar().showMessage(f"{message} ({current}/{total})")
        QApplication.processEvents()

    def _save_and_show_stack(self, name: str, data: np.ndarray, coords: dict,
                              vel_type: str, settings: dict):
        """Save stack to zarr and open in viewer."""
        # Determine save path
        if self.project:
            save_dir = Path(self.project.path) / "stacks"
        else:
            save_dir = Path.home() / "SeismicData" / "stacks"

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{name}.zarr"

        # Save to zarr
        try:
            # Use zarr.open_array for zarr v3 compatibility
            if data.ndim == 3:
                chunks = (min(64, data.shape[0]), min(64, data.shape[1]), min(256, data.shape[2]))
            elif data.ndim == 2:
                chunks = (min(64, data.shape[0]), min(256, data.shape[1]))
            else:
                chunks = None

            z = zarr.open_array(str(save_path), mode='w', shape=data.shape,
                               dtype=data.dtype, chunks=chunks)
            z[:] = data

            # Save coordinates as attributes
            if coords.get('t_coords') is not None:
                z.attrs['t_coords'] = coords['t_coords'].tolist()
            if coords.get('il_coords') is not None:
                z.attrs['il_coords'] = coords['il_coords'].tolist()
            if coords.get('xl_coords') is not None:
                z.attrs['xl_coords'] = coords['xl_coords'].tolist()

            # Save metadata
            z.attrs['velocity_type'] = vel_type
            z.attrs['settings'] = settings

            self.statusBar().showMessage(f"Stack saved: {save_path}")

        except Exception as e:
            self.statusBar().showMessage(f"Failed to save stack: {e}")
            return

        # Open in stack viewer
        self._open_stack_viewer()
        if self.stack_viewer:
            self.stack_viewer.add_stack(name, data, coords, {'path': str(save_path)})

    def _open_stack_viewer(self):
        """Open or focus the stack viewer window."""
        from .stack_viewer import StackViewerWindow

        if self.stack_viewer is None or not self.stack_viewer.isVisible():
            # Pass project path so stack viewer can auto-load existing stacks
            project_path = None
            if self.project:
                project_path = str(self.project.path)
            self.stack_viewer = StackViewerWindow(self, project_path=project_path)
            self.stack_viewer.show()
        else:
            self.stack_viewer.raise_()
            self.stack_viewer.activateWindow()

    # ==================== Project Integration Methods ====================

    def set_project(self, project: Optional['PSTMProject']):
        """Set/update the project reference."""
        self.project = project

    def _save_initial_velocity_to_project(self):
        """Save the initial velocity to the project folder."""
        if not self.project or self.initial_vel_grid is None:
            return

        metadata = {
            'il_coords': self.initial_vel_grid.il_coords,
            'xl_coords': self.initial_vel_grid.xl_coords,
            't_coords': self.initial_vel_grid.t_coords,
            'dt_ms': float(self.initial_vel_grid.t_coords[1] - self.initial_vel_grid.t_coords[0]) if len(self.initial_vel_grid.t_coords) > 1 else 100.0
        }

        self.project.save_initial_velocity(
            self.initial_vel_grid.velocities,
            metadata,
            source_path=self.velocity_path
        )
        print(f"[DEBUG] Saved initial velocity to project: {self.project.path}")

    def _save_edited_velocity_to_project(self):
        """Auto-save edited velocity to project after picks change."""
        if not self.project or not self.vel_grid.has_velocity():
            return

        success = self.project.save_edited_velocity(self.vel_grid)
        if success:
            print(f"[DEBUG] Auto-saved edited velocity to project")

    def load_velocities_from_project(self):
        """Load velocities from project folder.

        Returns:
            True if velocities were loaded successfully
        """
        if not self.project:
            return False

        # Get grid config from project
        grid_config = self.project.get_grid_config()
        if not grid_config:
            print("[DEBUG] No grid config in project")
            return False

        # Get time range from loaded data or defaults
        t_min = grid_config.get('t_min', 0.0)
        t_max = grid_config.get('t_max', 4000.0)

        # Setup output grid from project config
        self.vel_grid.setup_grid(
            grid_config['il_start'], grid_config['il_end'], grid_config['il_step'],
            grid_config['xl_start'], grid_config['xl_end'], grid_config['xl_step'],
            grid_config['output_dt_ms'], t_min, t_max
        )

        # Load initial velocity
        if self.project.has_initial_velocity():
            vel_data, metadata = self.project.load_initial_velocity()
            if vel_data is not None:
                # Setup initial_vel_grid
                self.initial_vel_grid = VelocityOutputGrid()
                self.initial_vel_grid.il_coords = self.vel_grid.il_coords.copy()
                self.initial_vel_grid.xl_coords = self.vel_grid.xl_coords.copy()
                self.initial_vel_grid.t_coords = self.vel_grid.t_coords.copy()
                self.initial_vel_grid.velocities = vel_data.copy()
                self.initial_vel_grid.grid_locations = list(self.vel_grid.grid_locations)

                # Copy to vel_grid as base
                self.vel_grid.velocities = vel_data.copy()

                print(f"[DEBUG] Loaded initial velocity from project")

        # Load edited velocity if exists (overwrites initial)
        if self.project.has_edited_velocity():
            success = self.project.load_edited_velocity(self.vel_grid)
            if success:
                print(f"[DEBUG] Loaded edited velocity from project")

        # Enable controls if velocity loaded
        if self.vel_grid.has_velocity():
            self.prev_grid_btn.setEnabled(True)
            self.next_grid_btn.setEnabled(True)
            self.apply_nmo_btn.setEnabled(True)
            self.save_velocity_action.setEnabled(True)
            self.use_inverse_nmo = True

            n_locs = len(self.vel_grid.grid_locations)
            self.vel_info_action.setText(f"Grid: {n_locs} locations")
            self._update_grid_label()

            # Emit signal for main window to update survey map
            self.velocity_loaded.emit(list(self.vel_grid.grid_locations))

            return True

        return False
