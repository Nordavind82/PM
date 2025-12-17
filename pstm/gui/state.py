"""
Wizard State Management

Central state management for the PSTM wizard, connecting GUI to backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class StepStatus(Enum):
    """Status of a wizard step."""
    NOT_VISITED = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()
    ERROR = auto()
    WARNING = auto()


@dataclass
class CornerPoints:
    """Output grid corner point coordinates."""
    c1_x: float = 0.0  # Origin (SW)
    c1_y: float = 0.0
    c2_x: float = 1000.0  # SE
    c2_y: float = 0.0
    c3_x: float = 1000.0  # NE
    c3_y: float = 1000.0
    c4_x: float = 0.0  # NW
    c4_y: float = 1000.0
    
    def as_array(self) -> NDArray:
        """Return corners as (4, 2) array."""
        return np.array([
            [self.c1_x, self.c1_y],
            [self.c2_x, self.c2_y],
            [self.c3_x, self.c3_y],
            [self.c4_x, self.c4_y],
        ])
    
    @classmethod
    def from_bounds(cls, x_min: float, x_max: float, y_min: float, y_max: float) -> "CornerPoints":
        """Create from bounding box."""
        return cls(
            c1_x=x_min, c1_y=y_min,
            c2_x=x_max, c2_y=y_min,
            c3_x=x_max, c3_y=y_max,
            c4_x=x_min, c4_y=y_max,
        )


@dataclass
class OutputGridState:
    """Output grid configuration state."""
    corners: CornerPoints = field(default_factory=CornerPoints)
    dx: float = 25.0  # Inline bin size
    dy: float = 25.0  # Crossline bin size
    dt_ms: float = 2.0  # Time sample interval
    t_min_ms: float = 0.0
    t_max_ms: float = 4000.0
    
    @property
    def nx(self) -> int:
        """Number of inline bins."""
        pts = self.corners.as_array()
        inline_length = np.linalg.norm(pts[1] - pts[0])
        return max(1, int(np.ceil(inline_length / self.dx)))
    
    @property
    def ny(self) -> int:
        """Number of crossline bins."""
        pts = self.corners.as_array()
        xline_length = np.linalg.norm(pts[3] - pts[0])
        return max(1, int(np.ceil(xline_length / self.dy)))
    
    @property
    def nt(self) -> int:
        """Number of time samples."""
        return max(1, int(np.ceil((self.t_max_ms - self.t_min_ms) / self.dt_ms)) + 1)
    
    @property
    def total_points(self) -> int:
        """Total output grid points."""
        return self.nx * self.ny * self.nt
    
    @property
    def estimated_size_gb(self) -> float:
        """Estimated output size in GB (float32)."""
        return self.total_points * 4 / (1024**3)


@dataclass
class VelocityState:
    """Velocity model configuration state."""
    source: str = "constant"  # constant, linear, function_1d, cube_3d, file
    constant_velocity: float = 2500.0
    linear_v0: float = 1800.0
    linear_gradient: float = 0.5  # m/s per ms
    function_1d_times: list[float] = field(default_factory=list)
    function_1d_values: list[float] = field(default_factory=list)
    cube_path: str = ""
    file_path: str = ""
    is_prepared: bool = False


@dataclass
class OffsetRange:
    """Single offset range."""
    min_offset: float | None = None
    max_offset: float | None = None


@dataclass
class AzimuthSector:
    """Azimuth sector definition."""
    offset_min: float | None = None
    offset_max: float | None = None
    azimuth_min: float = 0.0
    azimuth_max: float = 360.0
    active: bool = True


@dataclass
class DataSelectionState:
    """Data selection/filtering state - NO validation, user takes responsibility."""
    mode: str = "all"  # all, offset, azimuth, ovt, custom
    
    # Offset mode
    offset_ranges: list[OffsetRange] = field(default_factory=list)
    offset_include_mode: bool = True
    include_negative_offsets: bool = True
    
    # Azimuth mode
    azimuth_sectors: list[AzimuthSector] = field(default_factory=list)
    azimuth_convention: str = "receiver_relative"
    
    # OVT mode - signed offset components
    offset_x_min: float | None = None
    offset_x_max: float | None = None
    offset_y_min: float | None = None
    offset_y_max: float | None = None
    ovt_tile_size_x: float = 500.0
    ovt_tile_size_y: float = 500.0
    ovt_selected_tiles: list[tuple[int, int]] = field(default_factory=list)
    
    # Custom expression mode
    custom_expression: str = ""
    
    # Statistics
    total_traces: int = 0
    selected_traces: int = 0


@dataclass 
class AlgorithmState:
    """Migration algorithm parameters."""
    max_aperture_m: float = 5000.0
    min_aperture_m: float = 500.0
    max_dip_degrees: float = 45.0
    taper_type: str = "cosine"
    taper_fraction: float = 0.1
    interpolation_method: str = "linear"
    apply_spreading: bool = True
    apply_obliquity: bool = True
    enable_antialiasing: bool = False
    trace_weighting: str = "none"
    phase_rotation: float = 0.0
    mute_above_ms: float = 0.0
    mute_below_ms: float = 10000.0


@dataclass
class ExecutionState:
    """Execution configuration state."""
    backend: str = "auto"
    max_memory_gb: float = 8.0
    n_threads: int = 0
    auto_tile_size: bool = True
    tile_nx: int = 64  # Default manual tile size (when auto is disabled)
    tile_ny: int = 64  # Default manual tile size (when auto is disabled)
    tile_ordering: str = "snake"
    enable_checkpoint: bool = True
    checkpoint_interval_tiles: int = 10
    checkpoint_interval_seconds: float = 300.0
    resume_from_checkpoint: bool = True
    estimated_time_minutes: float = 0.0
    estimated_tiles: int = 0


@dataclass
class InputDataState:
    """Input data state."""
    traces_path: str = ""
    traces_format: str = "zarr"
    headers_path: str = ""
    headers_format: str = "parquet"
    col_source_x: str = "SOU_X"
    col_source_y: str = "SOU_Y"
    col_receiver_x: str = "REC_X"
    col_receiver_y: str = "REC_Y"
    col_midpoint_x: str = "CDP_X"
    col_midpoint_y: str = "CDP_Y"
    col_offset: str = "OFFSET"
    col_azimuth: str = "AZIMUTH"
    n_traces: int = 0
    n_samples: int = 0
    sample_rate_ms: float = 0.0
    is_loaded: bool = False


@dataclass
class SurveyState:
    """Survey geometry analysis state."""
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    offset_min: float = 0.0
    offset_max: float = 0.0
    offset_mean: float = 0.0
    n_shots: int = 0
    fold_computed: bool = False
    max_fold: int = 0
    mean_fold: float = 0.0
    spatial_index_built: bool = False


@dataclass
class HeaderStatistics:
    """Statistics computed from trace headers."""
    n_traces: int = 0
    source_x_range: tuple[float, float] = (0.0, 0.0)
    source_y_range: tuple[float, float] = (0.0, 0.0)
    receiver_x_range: tuple[float, float] = (0.0, 0.0)
    receiver_y_range: tuple[float, float] = (0.0, 0.0)
    midpoint_x_range: tuple[float, float] = (0.0, 0.0)
    midpoint_y_range: tuple[float, float] = (0.0, 0.0)


@dataclass
class OutputState:
    """Output configuration state."""
    output_dir: str = ""
    project_name: str = "migration_output"
    output_stacked_image: bool = True
    output_fold_map: bool = True
    output_cig: bool = False
    output_segy: bool = False
    output_qc_report: bool = True
    cig_n_offset_bins: int = 20
    cig_min_offset: float = 0.0
    cig_max_offset: float = 5000.0
    output_format: str = "zarr"
    output_dtype: str = "float32"


@dataclass
class WizardState:
    """Complete wizard state."""
    
    input_data: InputDataState = field(default_factory=InputDataState)
    survey: SurveyState = field(default_factory=SurveyState)
    output_grid: OutputGridState = field(default_factory=OutputGridState)
    velocity: VelocityState = field(default_factory=VelocityState)
    data_selection: DataSelectionState = field(default_factory=DataSelectionState)
    algorithm: AlgorithmState = field(default_factory=AlgorithmState)
    execution: ExecutionState = field(default_factory=ExecutionState)
    output: OutputState = field(default_factory=OutputState)
    
    step_status: dict[str, StepStatus] = field(default_factory=lambda: {
        "input": StepStatus.NOT_VISITED,
        "survey": StepStatus.NOT_VISITED,
        "output_grid": StepStatus.NOT_VISITED,
        "velocity": StepStatus.NOT_VISITED,
        "data_selection": StepStatus.NOT_VISITED,
        "algorithm": StepStatus.NOT_VISITED,
        "execution": StepStatus.NOT_VISITED,
    })
    
    current_step: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "input_data": asdict(self.input_data),
            "survey": asdict(self.survey),
            "output_grid": {
                **{k: v for k, v in asdict(self.output_grid).items() if k != "corners"},
                "corners": asdict(self.output_grid.corners),
            },
            "velocity": asdict(self.velocity),
            "data_selection": asdict(self.data_selection),
            "algorithm": asdict(self.algorithm),
            "execution": asdict(self.execution),
            "output": asdict(self.output),
            "current_step": self.current_step,
        }
    
    def save(self, path: Path | str) -> None:
        """Save state to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path | str) -> "WizardState":
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        state = cls()
        
        if "input_data" in data:
            state.input_data = InputDataState(**data["input_data"])
        if "survey" in data:
            state.survey = SurveyState(**data["survey"])
        if "output_grid" in data:
            og = data["output_grid"]
            corners = CornerPoints(**og.pop("corners", {}))
            state.output_grid = OutputGridState(corners=corners, **og)
        if "velocity" in data:
            state.velocity = VelocityState(**data["velocity"])
        if "data_selection" in data:
            ds = data["data_selection"]
            if "offset_ranges" in ds:
                ds["offset_ranges"] = [OffsetRange(**r) for r in ds["offset_ranges"]]
            if "azimuth_sectors" in ds:
                ds["azimuth_sectors"] = [AzimuthSector(**s) for s in ds["azimuth_sectors"]]
            if "ovt_selected_tiles" in ds:
                ds["ovt_selected_tiles"] = [tuple(t) for t in ds["ovt_selected_tiles"]]
            state.data_selection = DataSelectionState(**ds)
        if "algorithm" in data:
            state.algorithm = AlgorithmState(**data["algorithm"])
        if "execution" in data:
            state.execution = ExecutionState(**data["execution"])
        if "output" in data:
            state.output = OutputState(**data["output"])
        if "current_step" in data:
            state.current_step = data["current_step"]
        
        return state


class WizardController:
    """Controller for wizard state and operations."""
    
    def __init__(self):
        self.state = WizardState()
        self._change_callbacks: list[Callable[[], None]] = []
        self._headers_df = None
        self._traces_array = None
        self._velocity_model = None
    
    def add_change_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for state changes."""
        self._change_callbacks.append(callback)
    
    def notify_change(self) -> None:
        """Notify all callbacks of state change."""
        for cb in self._change_callbacks:
            try:
                cb()
            except Exception as e:
                print(f"Callback error: {e}")
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.state = WizardState()
        self._headers_df = None
        self._traces_array = None
        self._velocity_model = None
        self.notify_change()
    
    @property
    def headers_df(self):
        """Get loaded headers DataFrame."""
        return self._headers_df
    
    def load_input_data(self) -> tuple[bool, str]:
        """Load and validate input data."""
        try:
            import zarr
            import pandas as pd
            
            state = self.state.input_data
            
            if state.headers_format == "parquet":
                self._headers_df = pd.read_parquet(state.headers_path)
            elif state.headers_format == "csv":
                self._headers_df = pd.read_csv(state.headers_path)
            else:
                return False, f"Unsupported header format: {state.headers_format}"
            
            if state.traces_format == "zarr":
                traces = zarr.open(state.traces_path, mode='r')
                if isinstance(traces, zarr.Array):
                    state.n_traces, state.n_samples = traces.shape
                else:
                    if 'data' in traces:
                        state.n_traces, state.n_samples = traces['data'].shape
                    else:
                        return False, "Could not find trace data in Zarr group"
            
            if len(self._headers_df) != state.n_traces:
                return False, f"Header count ({len(self._headers_df)}) doesn't match trace count ({state.n_traces})"
            
            state.is_loaded = True
            self.state.step_status["input"] = StepStatus.COMPLETE
            self.notify_change()
            
            return True, f"Loaded {state.n_traces} traces × {state.n_samples} samples"
            
        except Exception as e:
            self.state.step_status["input"] = StepStatus.ERROR
            return False, str(e)
    
    def auto_detect_columns(self) -> dict[str, str | None]:
        """Auto-detect header column mapping."""
        if self._headers_df is None:
            return {}
        
        columns = list(self._headers_df.columns)
        mapping = {}
        
        patterns = {
            "source_x": ["SOU_X", "SRCX", "SX", "SOURCE_X", "SHOT_X"],
            "source_y": ["SOU_Y", "SRCY", "SY", "SOURCE_Y", "SHOT_Y"],
            "receiver_x": ["REC_X", "RECX", "RX", "RECEIVER_X", "GX"],
            "receiver_y": ["REC_Y", "RECY", "RY", "RECEIVER_Y", "GY"],
            "midpoint_x": ["CDP_X", "CDPX", "MX", "CMP_X", "MIDPOINT_X"],
            "midpoint_y": ["CDP_Y", "CDPY", "MY", "CMP_Y", "MIDPOINT_Y"],
            "offset": ["OFFSET", "OFF", "DISTANCE"],
            "azimuth": ["AZIMUTH", "AZI", "AZ"],
        }
        
        for field, names in patterns.items():
            for name in names:
                matches = [c for c in columns if c.upper() == name.upper()]
                if matches:
                    mapping[field] = matches[0]
                    break
        
        return mapping
    
    def analyze_survey_geometry(self) -> tuple[bool, str]:
        """Analyze survey geometry from loaded headers."""
        if self._headers_df is None:
            return False, "No header data loaded"
        
        try:
            df = self._headers_df
            state = self.state.survey
            inp = self.state.input_data
            
            sx = df[inp.col_source_x].values
            sy = df[inp.col_source_y].values
            rx = df[inp.col_receiver_x].values
            ry = df[inp.col_receiver_y].values
            
            state.x_min = min(sx.min(), rx.min())
            state.x_max = max(sx.max(), rx.max())
            state.y_min = min(sy.min(), ry.min())
            state.y_max = max(sy.max(), ry.max())
            
            if inp.col_offset in df.columns:
                offsets = df[inp.col_offset].values
            else:
                offsets = np.sqrt((rx - sx)**2 + (ry - sy)**2)
            
            state.offset_min = float(offsets.min())
            state.offset_max = float(offsets.max())
            state.offset_mean = float(offsets.mean())
            
            if "SHOT_ID" in df.columns:
                state.n_shots = df["SHOT_ID"].nunique()
            elif "FFID" in df.columns:
                state.n_shots = df["FFID"].nunique()
            
            self.state.step_status["survey"] = StepStatus.COMPLETE
            self.notify_change()
            
            return True, "Survey geometry analyzed"
            
        except Exception as e:
            self.state.step_status["survey"] = StepStatus.ERROR
            return False, str(e)
    
    def set_grid_from_survey(self) -> None:
        """Set output grid to match survey extent."""
        survey = self.state.survey
        self.state.output_grid.corners = CornerPoints.from_bounds(
            survey.x_min, survey.x_max,
            survey.y_min, survey.y_max
        )
        self.state.step_status["output_grid"] = StepStatus.IN_PROGRESS
        self.notify_change()

    def get_header_statistics(self) -> "HeaderStatistics | None":
        """Get statistics from loaded headers including midpoint extent.

        Returns:
            HeaderStatistics object or None if headers not loaded.
        """
        if self._headers_df is None:
            return None

        try:
            df = self._headers_df
            inp = self.state.input_data

            # Get source/receiver coordinates
            sx = df[inp.col_source_x].values
            sy = df[inp.col_source_y].values
            rx = df[inp.col_receiver_x].values
            ry = df[inp.col_receiver_y].values

            # Compute midpoints
            mx_col = inp.col_midpoint_x
            my_col = inp.col_midpoint_y

            if mx_col in df.columns and my_col in df.columns:
                mx = df[mx_col].values
                my = df[my_col].values
            else:
                # Compute from source/receiver
                mx = (sx + rx) / 2
                my = (sy + ry) / 2

            return HeaderStatistics(
                n_traces=len(df),
                source_x_range=(float(sx.min()), float(sx.max())),
                source_y_range=(float(sy.min()), float(sy.max())),
                receiver_x_range=(float(rx.min()), float(rx.max())),
                receiver_y_range=(float(ry.min()), float(ry.max())),
                midpoint_x_range=(float(mx.min()), float(mx.max())),
                midpoint_y_range=(float(my.min()), float(my.max())),
            )

        except Exception:
            return None
    
    def compute_selection_mask(self) -> NDArray[np.bool_] | None:
        """Compute trace selection mask based on current criteria."""
        if self._headers_df is None:
            return None
        
        df = self._headers_df
        inp = self.state.input_data
        sel = self.state.data_selection
        n = len(df)
        
        mask = np.ones(n, dtype=bool)
        
        if sel.mode == "all":
            pass
            
        elif sel.mode == "offset":
            if inp.col_offset in df.columns:
                offsets = df[inp.col_offset].values
            else:
                sx = df[inp.col_source_x].values
                rx = df[inp.col_receiver_x].values
                sy = df[inp.col_source_y].values
                ry = df[inp.col_receiver_y].values
                offsets = np.sqrt((rx - sx)**2 + (ry - sy)**2)
            
            if sel.offset_ranges:
                range_mask = np.zeros(n, dtype=bool)
                for r in sel.offset_ranges:
                    r_mask = np.ones(n, dtype=bool)
                    if r.min_offset is not None:
                        r_mask &= (offsets >= r.min_offset)
                    if r.max_offset is not None:
                        r_mask &= (offsets <= r.max_offset)
                    range_mask |= r_mask
                
                if sel.offset_include_mode:
                    mask &= range_mask
                else:
                    mask &= ~range_mask
                    
        elif sel.mode == "ovt":
            sx = df[inp.col_source_x].values
            rx = df[inp.col_receiver_x].values
            sy = df[inp.col_source_y].values
            ry = df[inp.col_receiver_y].values
            
            offset_x = rx - sx
            offset_y = ry - sy
            
            if sel.offset_x_min is not None:
                mask &= (offset_x >= sel.offset_x_min)
            if sel.offset_x_max is not None:
                mask &= (offset_x <= sel.offset_x_max)
            if sel.offset_y_min is not None:
                mask &= (offset_y >= sel.offset_y_min)
            if sel.offset_y_max is not None:
                mask &= (offset_y <= sel.offset_y_max)
                
        elif sel.mode == "custom" and sel.custom_expression:
            try:
                sx = df[inp.col_source_x].values
                rx = df[inp.col_receiver_x].values
                sy = df[inp.col_source_y].values
                ry = df[inp.col_receiver_y].values
                
                context = {
                    "sx": sx, "sy": sy, "rx": rx, "ry": ry,
                    "mx": (sx + rx) / 2, "my": (sy + ry) / 2,
                    "offset": np.sqrt((rx - sx)**2 + (ry - sy)**2),
                    "offset_x": rx - sx, "offset_y": ry - sy,
                    "azimuth": np.degrees(np.arctan2(rx - sx, ry - sy)) % 360,
                    "np": np, "abs": np.abs, "sqrt": np.sqrt,
                    "sin": np.sin, "cos": np.cos,
                }
                
                mask = eval(sel.custom_expression, {"__builtins__": {}}, context)
                mask = np.asarray(mask, dtype=bool)
                
            except Exception as e:
                print(f"Expression error: {e}")
        
        sel.total_traces = n
        sel.selected_traces = int(mask.sum())
        self.notify_change()
        
        return mask
    
    def compute_fold_map(self, bin_size: float = 25.0) -> tuple[NDArray | None, str]:
        """Compute fold map from loaded headers.
        
        Args:
            bin_size: Bin size in meters for fold computation
            
        Returns:
            Tuple of (fold_map array or None, message string)
        """
        if self._headers_df is None:
            return None, "No header data loaded"
        
        try:
            df = self._headers_df
            inp = self.state.input_data
            survey = self.state.survey
            
            # Get midpoint columns
            mx_col = inp.col_midpoint_x
            my_col = inp.col_midpoint_y
            
            if mx_col not in df.columns or my_col not in df.columns:
                # Compute from source/receiver if midpoint not available
                sx = df[inp.col_source_x].values
                sy = df[inp.col_source_y].values
                rx = df[inp.col_receiver_x].values
                ry = df[inp.col_receiver_y].values
                mx = (sx + rx) / 2
                my = (sy + ry) / 2
            else:
                mx = df[mx_col].values
                my = df[my_col].values
            
            # Compute fold using histogram
            x_bins = np.arange(survey.x_min, survey.x_max + bin_size, bin_size)
            y_bins = np.arange(survey.y_min, survey.y_max + bin_size, bin_size)
            
            fold_map, _, _ = np.histogram2d(mx, my, bins=[x_bins, y_bins])
            
            # Update survey state
            survey.max_fold = int(fold_map.max())
            survey.mean_fold = float(fold_map[fold_map > 0].mean()) if fold_map.max() > 0 else 0.0
            survey.fold_computed = True
            
            self.state.step_status["survey"] = StepStatus.COMPLETE
            self.notify_change()
            
            return fold_map, f"Max fold: {survey.max_fold}, Mean: {survey.mean_fold:.1f}"
            
        except Exception as e:
            return None, str(e)
    
    def build_migration_config(self):
        """Build MigrationConfig from wizard state."""
        import logging
        debug_logger = logging.getLogger("pstm.migration.debug")

        from pstm.config.models import (
            MigrationConfig, InputConfig, OutputConfig, OutputGridConfig,
            VelocityConfig, VelocitySource, AlgorithmConfig, TilingConfig,
            CheckpointConfig, OutputProductsConfig, OutputFormat,
            ApertureConfig, AmplitudeConfig, InterpolationMethod,
            ExecutionConfig, ResourceConfig, ComputeBackend,
        )

        state = self.state

        debug_logger.info(f"BUILD_CONFIG: state.execution.backend = '{state.execution.backend}'")
        og = state.output_grid
        
        pts = og.corners.as_array()
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        
        output_grid = OutputGridConfig(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            t_min_ms=og.t_min_ms, t_max_ms=og.t_max_ms,
            dx=og.dx, dy=og.dy, dt_ms=og.dt_ms,
        )
        
        vel = state.velocity
        if vel.source == "constant":
            velocity_config = VelocityConfig(
                source=VelocitySource.CONSTANT,
                constant_velocity=vel.constant_velocity,
            )
        elif vel.source == "linear":
            velocity_config = VelocityConfig(
                source=VelocitySource.LINEAR_V0K,
                v0=vel.linear_v0,
                k=vel.linear_gradient,
            )
        elif vel.source == "function_1d":
            velocity_config = VelocityConfig(
                source=VelocitySource.TABLE_1D,
                times_ms=vel.function_1d_times,
                velocities=vel.function_1d_values,
            )
        else:
            velocity_config = VelocityConfig(
                source=VelocitySource.CUBE_3D,
                cube_path=vel.cube_path,
            )
        
        algo = state.algorithm

        # Map interpolation method string to enum
        interp_map = {
            "linear": InterpolationMethod.LINEAR,
            "sinc8": InterpolationMethod.SINC8,
            "nearest": InterpolationMethod.NEAREST,
            "kaiser": InterpolationMethod.KAISER,
        }
        interpolation = interp_map.get(algo.interpolation_method, InterpolationMethod.LINEAR)

        # Build nested configs
        aperture_config = ApertureConfig(
            max_aperture_m=algo.max_aperture_m,
            min_aperture_m=algo.min_aperture_m,
            max_dip_degrees=algo.max_dip_degrees,
            taper_fraction=algo.taper_fraction,
        )

        amplitude_config = AmplitudeConfig(
            geometrical_spreading=algo.apply_spreading,
            obliquity_factor=algo.apply_obliquity,
        )

        algorithm_config = AlgorithmConfig(
            interpolation=interpolation,
            aperture=aperture_config,
            amplitude=amplitude_config,
        )
        
        # Validate required fields - fail fast, no fallbacks
        if not state.input_data.traces_path:
            raise ValueError("Input traces path is required")
        if not state.input_data.headers_path:
            raise ValueError("Input headers path is required")
        if not state.output.output_dir:
            raise ValueError("Output directory is required")

        # Build output products config
        products = OutputProductsConfig(
            stacked_image=state.output.output_stacked_image,
            fold_volume=state.output.output_fold_map,
            common_image_gathers=state.output.output_cig,
        )

        # Determine output format
        output_format = OutputFormat.ZARR if state.output.output_format == "zarr" else OutputFormat.SEGY

        # Parse backend string to enum - fail fast if invalid
        from pstm.config.backends import parse_backend
        backend_str = state.execution.backend
        backend = parse_backend(backend_str)

        debug_logger.info(f"BUILD_CONFIG: backend_str = '{backend_str}'")
        debug_logger.info(f"BUILD_CONFIG: backend_enum = {backend} ({backend.value})")

        # Build execution config with tiling, checkpointing, and resources
        execution_config = ExecutionConfig(
            resources=ResourceConfig(
                backend=backend,
                max_memory_gb=state.execution.max_memory_gb,
                num_workers=state.execution.n_threads if state.execution.n_threads > 0 else None,
            ),
            tiling=TilingConfig(
                auto_tile_size=state.execution.auto_tile_size,
                tile_nx=state.execution.tile_nx,
                tile_ny=state.execution.tile_ny,
            ),
            checkpoint=CheckpointConfig(
                enabled=state.execution.enable_checkpoint,
                interval_tiles=state.execution.checkpoint_interval_tiles,
            ),
        )

        config = MigrationConfig(
            name=state.output.project_name,
            input=InputConfig(
                traces_path=state.input_data.traces_path,
                headers_path=state.input_data.headers_path,
            ),
            output=OutputConfig(
                output_dir=state.output.output_dir,
                grid=output_grid,
                products=products,
                format=output_format,
            ),
            velocity=velocity_config,
            algorithm=algorithm_config,
            execution=execution_config,
        )

        return config
