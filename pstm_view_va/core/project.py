"""PSTM Project management - handles project folder, state, and velocities."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import zarr


class PSTMProject:
    """Manages project folder, state, and velocities.

    Project structure:
        my_project.pstm/
        ├── project.json              # Main project metadata and state
        ├── velocity_initial.zarr/    # Initial velocity (copy from source)
        ├── velocity_edited.zarr/     # Edited velocity with user picks
        └── grid_config.json          # Velocity grid configuration
    """

    VERSION = "1.0"
    PROJECT_EXT = ".pstm"

    def __init__(self, path: Optional[Path] = None):
        self.path: Optional[Path] = Path(path) if path else None
        self.config: Dict[str, Any] = self._default_config()
        self.grid_config: Dict[str, Any] = {}
        self._modified = False

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default project configuration."""
        return {
            "version": PSTMProject.VERSION,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "data_sources": {
                "gathers_path": "",
                "cube_path": ""
            },
            "velocity": {
                "initial_source": "",
                "has_edits": False
            },
            "va_state": {
                "position": {"il": 0, "xl": 0},
                "super_gather": {"il_window": 5, "xl_window": 5, "offset_bin": 50},
                "mute": {"top_enabled": False, "v_top": 1500, "bottom_enabled": False, "v_bottom": 4000},
                "processing": {"bandpass": False, "f_low": 5, "f_high": 80, "agc": False, "agc_window": 500},
                "nmo": {"use_inverse": False, "forward_applied": False, "stretch_percent": 30},
                "semblance": {"v_min": 1500, "v_max": 5000, "v_step": 100, "window_samples": 5},
                "display": {
                    "gather_colormap": "Seismic (BWR)", "gather_clip": 99.0, "gather_gain": 1.0,
                    "semblance_colormap": "Viridis", "semblance_clip": 99.0
                }
            },
            "main_viewer_state": {
                "view_mode": "velocity_analysis",
                "direction": "inline",
                "slice_index": 0,
                "palette": "Gray",
                "gain": 1.0,
                "clip": 99.0
            }
        }

    @classmethod
    def create(cls, path: Path, name: Optional[str] = None) -> 'PSTMProject':
        """Create a new project folder.

        Args:
            path: Parent directory for the project
            name: Project name (without extension). If None, uses path's name.

        Returns:
            New PSTMProject instance
        """
        path = Path(path)

        # Add extension if not present
        if not path.suffix == cls.PROJECT_EXT:
            if name:
                path = path / f"{name}{cls.PROJECT_EXT}"
            elif not path.name.endswith(cls.PROJECT_EXT):
                path = Path(str(path) + cls.PROJECT_EXT)

        # Create project folder
        path.mkdir(parents=True, exist_ok=True)

        project = cls(path)
        project.save()

        return project

    @classmethod
    def open(cls, path: Path) -> 'PSTMProject':
        """Open an existing project.

        Args:
            path: Path to project folder (.pstm directory)

        Returns:
            PSTMProject instance

        Raises:
            FileNotFoundError: If project.json doesn't exist
            ValueError: If project format is invalid
        """
        path = Path(path)
        project_file = path / "project.json"

        if not project_file.exists():
            raise FileNotFoundError(f"No project.json found in {path}")

        project = cls(path)

        # Load project config
        with open(project_file, 'r') as f:
            project.config = json.load(f)

        # Load grid config if exists
        grid_file = path / "grid_config.json"
        if grid_file.exists():
            with open(grid_file, 'r') as f:
                project.grid_config = json.load(f)

        return project

    @classmethod
    def is_project(cls, path: Path) -> bool:
        """Check if a path is a valid PSTM project."""
        path = Path(path)
        return path.is_dir() and (path / "project.json").exists()

    def save(self):
        """Save project.json and grid_config.json."""
        if self.path is None:
            raise ValueError("Project path not set")

        # Update modified timestamp
        self.config["modified"] = datetime.now().isoformat()

        # Save project config
        project_file = self.path / "project.json"
        with open(project_file, 'w') as f:
            json.dump(self.config, f, indent=2)

        # Save grid config if set
        if self.grid_config:
            grid_file = self.path / "grid_config.json"
            with open(grid_file, 'w') as f:
                json.dump(self.grid_config, f, indent=2)

        self._modified = False

    @property
    def name(self) -> str:
        """Get project name (folder name without extension)."""
        if self.path:
            return self.path.stem
        return "Untitled"

    @property
    def is_modified(self) -> bool:
        """Check if project has unsaved changes."""
        return self._modified

    def mark_modified(self):
        """Mark project as having unsaved changes."""
        self._modified = True

    # =========================================================================
    # Data Source Management
    # =========================================================================

    def set_gathers_path(self, path: str):
        """Set path to gathers data."""
        self.config["data_sources"]["gathers_path"] = str(path)
        self.mark_modified()

    def get_gathers_path(self) -> Optional[str]:
        """Get path to gathers data."""
        path = self.config["data_sources"].get("gathers_path", "")
        return path if path else None

    def set_cube_path(self, path: str):
        """Set path to seismic cube."""
        self.config["data_sources"]["cube_path"] = str(path)
        self.mark_modified()

    def get_cube_path(self) -> Optional[str]:
        """Get path to seismic cube."""
        path = self.config["data_sources"].get("cube_path", "")
        return path if path else None

    # =========================================================================
    # Velocity Management
    # =========================================================================

    def _initial_velocity_path(self) -> Path:
        """Get path to initial velocity zarr."""
        return self.path / "velocity_initial.zarr"

    def _edited_velocity_path(self) -> Path:
        """Get path to edited velocity zarr."""
        return self.path / "velocity_edited.zarr"

    def has_initial_velocity(self) -> bool:
        """Check if project has initial velocity."""
        return self._initial_velocity_path().exists()

    def has_edited_velocity(self) -> bool:
        """Check if project has edited velocity."""
        return self._edited_velocity_path().exists()

    def save_initial_velocity(self, vel_data: np.ndarray, metadata: Dict[str, Any],
                               source_path: Optional[str] = None):
        """Save initial velocity to project.

        Args:
            vel_data: 3D velocity array (il, xl, time)
            metadata: Velocity metadata (coordinates, dt_ms, etc.)
            source_path: Original source path (for reference)
        """
        vel_path = self._initial_velocity_path()

        # Create zarr group (v3 compatible)
        if vel_path.exists():
            shutil.rmtree(vel_path)
        root = zarr.open_group(str(vel_path), mode='w')

        # Save velocity array (zarr v3 API)
        z = root.create_array('velocities', shape=vel_data.shape, dtype=vel_data.dtype,
                              chunks=(1, 1, vel_data.shape[2]))
        z[:] = vel_data

        # Save coordinates if available
        if 'il_coords' in metadata:
            arr = np.array(metadata['il_coords'])
            z = root.create_array('il_coords', shape=arr.shape, dtype=arr.dtype)
            z[:] = arr
        if 'xl_coords' in metadata:
            arr = np.array(metadata['xl_coords'])
            z = root.create_array('xl_coords', shape=arr.shape, dtype=arr.dtype)
            z[:] = arr
        if 't_coords' in metadata:
            arr = np.array(metadata['t_coords'])
            z = root.create_array('t_coords', shape=arr.shape, dtype=arr.dtype)
            z[:] = arr

        # Save metadata as attributes
        root.attrs['type'] = 'velocity_initial'
        root.attrs['source_path'] = source_path or ''
        for key, value in metadata.items():
            if key not in ['il_coords', 'xl_coords', 't_coords']:
                try:
                    root.attrs[key] = value
                except TypeError:
                    root.attrs[key] = str(value)

        # Update project config
        self.config["velocity"]["initial_source"] = source_path or ""
        self.mark_modified()

    def load_initial_velocity(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Load initial velocity from project.

        Returns:
            Tuple of (velocity_array, metadata) or (None, {}) if not found
        """
        vel_path = self._initial_velocity_path()
        if not vel_path.exists():
            return None, {}

        try:
            z = zarr.open(str(vel_path), mode='r')
            vel_data = np.array(z['velocities'])

            metadata = dict(z.attrs)

            # Load coordinates
            if 'il_coords' in z:
                metadata['il_coords'] = np.array(z['il_coords'])
            if 'xl_coords' in z:
                metadata['xl_coords'] = np.array(z['xl_coords'])
            if 't_coords' in z:
                metadata['t_coords'] = np.array(z['t_coords'])

            return vel_data, metadata
        except Exception as e:
            print(f"Error loading initial velocity: {e}")
            return None, {}

    def save_edited_velocity(self, vel_grid) -> bool:
        """Save edited velocity grid to project.

        Args:
            vel_grid: VelocityOutputGrid instance

        Returns:
            True if successful
        """
        if vel_grid is None or vel_grid.velocities is None:
            return False

        vel_path = self._edited_velocity_path()

        try:
            # Create zarr group (v3 compatible)
            if vel_path.exists():
                shutil.rmtree(vel_path)
            root = zarr.open_group(str(vel_path), mode='w')

            # Save velocity array (zarr v3 API)
            z = root.create_array('velocities', shape=vel_grid.velocities.shape,
                                  dtype=vel_grid.velocities.dtype,
                                  chunks=(1, 1, vel_grid.velocities.shape[2]))
            z[:] = vel_grid.velocities

            # Save coordinates
            for name, arr in [('il_coords', vel_grid.il_coords),
                              ('xl_coords', vel_grid.xl_coords),
                              ('t_coords', vel_grid.t_coords)]:
                z = root.create_array(name, shape=arr.shape, dtype=arr.dtype)
                z[:] = arr

            # Save metadata
            root.attrs['type'] = 'velocity_edited'
            root.attrs['n_il'] = len(vel_grid.il_coords)
            root.attrs['n_xl'] = len(vel_grid.xl_coords)
            root.attrs['n_time'] = len(vel_grid.t_coords)
            if len(vel_grid.t_coords) > 1:
                root.attrs['dt_ms'] = float(vel_grid.t_coords[1] - vel_grid.t_coords[0])

            # Update project config
            self.config["velocity"]["has_edits"] = True
            self.mark_modified()

            return True
        except Exception as e:
            print(f"Error saving edited velocity: {e}")
            return False

    def load_edited_velocity(self, vel_grid) -> bool:
        """Load edited velocity into a VelocityOutputGrid.

        Args:
            vel_grid: VelocityOutputGrid instance to populate

        Returns:
            True if successful
        """
        vel_path = self._edited_velocity_path()
        if not vel_path.exists():
            return False

        try:
            z = zarr.open(str(vel_path), mode='r')

            vel_grid.velocities = np.array(z['velocities'])
            vel_grid.il_coords = np.array(z['il_coords'])
            vel_grid.xl_coords = np.array(z['xl_coords'])
            vel_grid.t_coords = np.array(z['t_coords'])

            # Rebuild grid locations
            vel_grid.grid_locations = []
            for il in vel_grid.il_coords:
                for xl in vel_grid.xl_coords:
                    vel_grid.grid_locations.append((int(il), int(xl)))
            vel_grid.current_idx = 0

            return True
        except Exception as e:
            print(f"Error loading edited velocity: {e}")
            return False

    def set_grid_config(self, il_start: int, il_end: int, il_step: int,
                        xl_start: int, xl_end: int, xl_step: int,
                        output_dt_ms: float, t_min: float = 0.0, t_max: float = 4000.0):
        """Set velocity grid configuration."""
        self.grid_config = {
            "il_start": il_start, "il_end": il_end, "il_step": il_step,
            "xl_start": xl_start, "xl_end": xl_end, "xl_step": xl_step,
            "output_dt_ms": output_dt_ms,
            "t_min": t_min, "t_max": t_max
        }
        self.mark_modified()

    def get_grid_config(self) -> Dict[str, Any]:
        """Get velocity grid configuration."""
        return self.grid_config.copy()

    # =========================================================================
    # VA State Management
    # =========================================================================

    def get_va_state(self) -> Dict[str, Any]:
        """Get VA window state."""
        return self.config.get("va_state", {}).copy()

    def set_va_state(self, state: Dict[str, Any]):
        """Set VA window state."""
        self.config["va_state"] = state
        self.mark_modified()

    def update_va_position(self, il: int, xl: int):
        """Update VA position in state."""
        if "va_state" not in self.config:
            self.config["va_state"] = {}
        if "position" not in self.config["va_state"]:
            self.config["va_state"]["position"] = {}
        self.config["va_state"]["position"]["il"] = il
        self.config["va_state"]["position"]["xl"] = xl
        self.mark_modified()

    # =========================================================================
    # Main Viewer State Management
    # =========================================================================

    def get_main_viewer_state(self) -> Dict[str, Any]:
        """Get main viewer state."""
        return self.config.get("main_viewer_state", {}).copy()

    def set_main_viewer_state(self, state: Dict[str, Any]):
        """Set main viewer state."""
        self.config["main_viewer_state"] = state
        self.mark_modified()


def get_recent_projects(max_count: int = 10) -> List[Tuple[str, str]]:
    """Get list of recent projects from QSettings.

    Returns:
        List of (path, modified_date) tuples
    """
    from PyQt6.QtCore import QSettings
    settings = QSettings("PSTM", "SeismicViewerVA")

    recent = settings.value("recent_projects", [])
    if not isinstance(recent, list):
        recent = []

    # Filter to existing projects and get modified dates
    result = []
    for path_str in recent[:max_count]:
        path = Path(path_str)
        if PSTMProject.is_project(path):
            try:
                # Get modified date from project.json
                project_file = path / "project.json"
                with open(project_file, 'r') as f:
                    config = json.load(f)
                modified = config.get("modified", "")
                result.append((str(path), modified))
            except:
                result.append((str(path), ""))

    return result


def add_recent_project(path: str, max_count: int = 10):
    """Add a project to recent projects list."""
    from PyQt6.QtCore import QSettings
    settings = QSettings("PSTM", "SeismicViewerVA")

    recent = settings.value("recent_projects", [])
    if not isinstance(recent, list):
        recent = []

    # Remove if already in list
    path_str = str(path)
    if path_str in recent:
        recent.remove(path_str)

    # Add to front
    recent.insert(0, path_str)

    # Trim to max count
    recent = recent[:max_count]

    settings.setValue("recent_projects", recent)
    settings.sync()
