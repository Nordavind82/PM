"""
PSTM Application Settings - Centralized Configuration

This module contains all configurable parameters for the PSTM application.
Users can modify settings via:
1. Settings file (~/.pstm/settings.toml or custom path)
2. Environment variables (PSTM_*)
3. Wizard UI initialization
4. Programmatic access via Settings singleton

All hardcoded values throughout the codebase should reference this module.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

# Try to import tomllib (Python 3.11+) or tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

try:
    import tomli_w
except ImportError:
    tomli_w = None


# =============================================================================
# Settings Data Classes - Organized by Domain
# =============================================================================


@dataclass
class GridSettings:
    """Default grid parameters."""
    
    # Spatial grid defaults
    dx_m: float = 25.0  # Default inline spacing (meters)
    dy_m: float = 25.0  # Default crossline spacing (meters)
    
    # Temporal grid defaults
    dt_ms: float = 2.0  # Default sample interval (milliseconds)
    t_start_ms: float = 0.0  # Default start time (milliseconds)
    
    # Grid size limits
    max_inline_bins: int = 10000  # Maximum inline bins before warning
    max_crossline_bins: int = 10000  # Maximum crossline bins before warning
    max_time_samples: int = 10000  # Maximum time samples before warning
    
    # Memory estimation
    bytes_per_sample: int = 8  # float64
    max_output_size_gb: float = 100.0  # Warning threshold for output size


@dataclass
class VelocitySettings:
    """Velocity model parameters."""
    
    # Valid velocity range (m/s)
    min_velocity_ms: float = 500.0  # Minimum valid velocity
    max_velocity_ms: float = 10000.0  # Maximum valid velocity
    
    # Default velocity range for QC (m/s)
    qc_min_velocity_ms: float = 1000.0  # QC warning below this
    qc_max_velocity_ms: float = 8000.0  # QC warning above this
    
    # Velocity gradient limits (m/s per second)
    max_gradient_ms_per_s: float = 5000.0  # Maximum velocity gradient
    
    # Inversion detection threshold (m/s)
    inversion_threshold_ms: float = -50.0  # Negative = inversion
    
    # Default constant velocity (m/s)
    default_constant_velocity_ms: float = 2000.0
    
    # Linear velocity default gradient (m/s per second)
    default_linear_gradient: float = 0.5


@dataclass
class ApertureSettings:
    """Migration aperture parameters."""
    
    # Aperture range (meters)
    min_aperture_m: float = 500.0  # Minimum aperture
    max_aperture_m: float = 5000.0  # Maximum aperture
    default_aperture_m: float = 3000.0  # Default aperture
    
    # Dip limit (degrees)
    max_dip_degrees: float = 45.0  # Maximum dip angle
    
    # Taper parameters
    taper_fraction: float = 0.1  # Edge taper as fraction of aperture
    taper_type: Literal["linear", "cosine", "hann"] = "cosine"


@dataclass
class KernelSettings:
    """Compute kernel parameters."""
    
    # Interpolation - available: nearest, linear, cubic, sinc4, sinc8, sinc16, lanczos3, lanczos5
    default_interpolation: Literal[
        "nearest", "linear", "cubic", 
        "sinc4", "sinc8", "sinc16",
        "lanczos3", "lanczos5"
    ] = "linear"
    sinc_half_width: int = 8  # Sinc interpolation kernel half-width (deprecated, use method name)
    
    # Anti-aliasing
    enable_antialiasing: bool = True
    antialiasing_filter_length: int = 8
    
    # Chunk sizes for different backends
    numpy_chunk_traces: int = 100
    numba_chunk_traces: int = 500
    mlx_chunk_traces: int = 1000
    mlx_time_batch: int = 50
    
    # Numerical stability
    epsilon: float = 1e-10  # Small value to avoid division by zero
    
    # Spreading correction
    apply_spreading_correction: bool = True
    spreading_power: float = 1.0  # v*t spreading correction power


@dataclass
class TilingSettings:
    """Tile planning parameters."""
    
    # Memory management
    max_memory_gb: float = 8.0  # Maximum memory per tile
    memory_safety_factor: float = 0.8  # Use 80% of available memory
    
    # Tile dimensions
    min_tile_size: int = 8  # Minimum tile dimension
    max_tile_size: int = 512  # Maximum tile dimension
    default_tile_size: int = 64  # Default tile dimension (when auto is disabled)
    
    # Ordering
    default_ordering: Literal["row_major", "column_major", "snake", "hilbert"] = "snake"
    
    # Overlap for aperture
    tile_overlap_factor: float = 1.5  # Overlap as fraction of aperture


@dataclass
class CheckpointSettings:
    """Checkpoint and recovery parameters."""
    
    # Checkpoint frequency
    checkpoint_interval_tiles: int = 10  # Checkpoint every N tiles
    checkpoint_interval_seconds: float = 300.0  # Or every N seconds
    
    # File management
    checkpoint_dir_name: str = ".pstm_checkpoint"
    keep_checkpoint_on_success: bool = False
    max_checkpoint_history: int = 3  # Keep last N checkpoints


@dataclass
class IOSettings:
    """I/O and data handling parameters."""
    
    # Trace reading
    trace_chunk_size: int = 1000  # Traces per read chunk
    prefetch_chunks: int = 2  # Number of chunks to prefetch
    
    # Memory mapping
    memmap_threshold_mb: float = 100.0  # Use memmap above this size
    memmap_mode: Literal["r", "r+", "w+", "c"] = "r+"
    
    # Compression
    zarr_compressor: str = "blosc"
    zarr_compression_level: int = 3
    
    # Buffer sizes
    double_buffer_size_mb: float = 512.0
    
    # NaN/Inf handling
    max_nan_fraction: float = 0.01  # 1% max NaN allowed
    max_inf_fraction: float = 0.0  # No Inf allowed
    replace_nan_value: float = 0.0
    replace_inf_value: float = 0.0


@dataclass 
class QCSettings:
    """Quality control parameters."""
    
    # Geometry QC
    fold_bin_size_m: float = 25.0  # Bin size for fold calculation
    min_fold_warning: int = 1  # Warn if fold below this
    
    # Offset/azimuth analysis
    offset_bin_size_m: float = 100.0  # Offset bin size
    azimuth_bin_size_deg: float = 10.0  # Azimuth bin size
    
    # Diffractor focusing verification
    focus_tolerance_xy_m: float = 50.0  # XY position tolerance
    focus_tolerance_t_ms: float = 20.0  # Time tolerance
    
    # Reflector depth verification  
    depth_tolerance_ms: float = 10.0  # Depth (time) tolerance
    
    # Output QC thresholds
    max_zero_fraction: float = 0.9  # Warn if >90% zeros
    amplitude_clip_percentile: float = 99.0  # For display clipping


@dataclass
class CIGSettings:
    """Common Image Gather parameters."""
    
    # Offset binning
    min_offset_m: float = 0.0
    max_offset_m: float = 5000.0
    n_offset_bins: int = 20
    
    # Semblance analysis
    semblance_window_ms: float = 100.0  # Time window for semblance
    n_velocities: int = 100  # Number of velocity scans
    velocity_range_fraction: float = 0.2  # ±20% of reference velocity
    
    # Flatness threshold
    flatness_threshold: float = 0.5  # Minimum flatness score


@dataclass
class ProfilingSettings:
    """Performance profiling parameters."""
    
    # Memory tracking
    memory_sample_interval_s: float = 1.0  # Memory sampling interval
    track_peak_memory: bool = True
    
    # Timing
    enable_detailed_timing: bool = False  # Per-trace timing (slow)
    timer_precision: Literal["ns", "us", "ms", "s"] = "ms"
    
    # Reporting
    report_interval_tiles: int = 10  # Progress report frequency


@dataclass
class UISettings:
    """User interface parameters."""
    
    # Console output
    use_colors: bool = True
    show_progress_bar: bool = True
    progress_update_hz: float = 2.0  # Progress updates per second
    
    # Wizard defaults
    wizard_theme: Literal["dark", "light"] = "dark"
    show_advanced_options: bool = False
    
    # Validation feedback
    validate_on_change: bool = True
    show_warnings: bool = True


@dataclass
class UnitSettings:
    """Unit conversion parameters."""
    
    # Coordinate units
    default_coordinate_unit: Literal["m", "ft"] = "m"
    
    # Conversion factors
    feet_per_meter: float = 3.28084
    meters_per_foot: float = 0.3048
    ms_per_second: float = 1000.0
    
    # Display precision
    coordinate_decimals: int = 2
    time_decimals: int = 1
    velocity_decimals: int = 0


@dataclass
class ApplicationSettings:
    """Root settings container with all subsections."""
    
    grid: GridSettings = field(default_factory=GridSettings)
    velocity: VelocitySettings = field(default_factory=VelocitySettings)
    aperture: ApertureSettings = field(default_factory=ApertureSettings)
    kernel: KernelSettings = field(default_factory=KernelSettings)
    tiling: TilingSettings = field(default_factory=TilingSettings)
    checkpoint: CheckpointSettings = field(default_factory=CheckpointSettings)
    io: IOSettings = field(default_factory=IOSettings)
    qc: QCSettings = field(default_factory=QCSettings)
    cig: CIGSettings = field(default_factory=CIGSettings)
    profiling: ProfilingSettings = field(default_factory=ProfilingSettings)
    ui: UISettings = field(default_factory=UISettings)
    units: UnitSettings = field(default_factory=UnitSettings)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApplicationSettings":
        """Create from nested dictionary."""
        return cls(
            grid=GridSettings(**data.get("grid", {})),
            velocity=VelocitySettings(**data.get("velocity", {})),
            aperture=ApertureSettings(**data.get("aperture", {})),
            kernel=KernelSettings(**data.get("kernel", {})),
            tiling=TilingSettings(**data.get("tiling", {})),
            checkpoint=CheckpointSettings(**data.get("checkpoint", {})),
            io=IOSettings(**data.get("io", {})),
            qc=QCSettings(**data.get("qc", {})),
            cig=CIGSettings(**data.get("cig", {})),
            profiling=ProfilingSettings(**data.get("profiling", {})),
            ui=UISettings(**data.get("ui", {})),
            units=UnitSettings(**data.get("units", {})),
        )


# =============================================================================
# Settings Manager - Singleton for Global Access
# =============================================================================


class SettingsManager:
    """
    Singleton manager for application settings.
    
    Usage:
        from pstm.settings import get_settings, settings
        
        # Access current settings
        s = get_settings()
        print(s.grid.dx_m)
        
        # Or use module-level shortcut
        print(settings.grid.dx_m)
        
        # Modify settings
        s.grid.dx_m = 50.0
        
        # Save to file
        save_settings(s, "my_settings.toml")
        
        # Load from file
        load_settings("my_settings.toml")
    """
    
    _instance: "SettingsManager | None" = None
    _settings: ApplicationSettings
    _settings_path: Path | None = None
    
    def __new__(cls) -> "SettingsManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._settings = ApplicationSettings()
            cls._instance._settings_path = None
        return cls._instance
    
    @property
    def settings(self) -> ApplicationSettings:
        """Get current settings."""
        return self._settings
    
    @property
    def path(self) -> Path | None:
        """Get path of loaded settings file."""
        return self._settings_path
    
    def reset(self) -> None:
        """Reset to default settings."""
        self._settings = ApplicationSettings()
        self._settings_path = None
    
    def update(self, **kwargs) -> None:
        """Update settings from keyword arguments."""
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like "grid.dx_m"
                parts = key.split(".")
                obj = self._settings
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            elif hasattr(self._settings, key):
                setattr(self._settings, key, value)
    
    def load_from_file(self, path: Path | str) -> None:
        """Load settings from TOML or JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Settings file not found: {path}")
        
        with open(path, "rb") as f:
            if path.suffix == ".toml":
                if tomllib is None:
                    raise ImportError("tomllib/tomli required for TOML support")
                data = tomllib.load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        self._settings = ApplicationSettings.from_dict(data)
        self._settings_path = path
    
    def save_to_file(self, path: Path | str) -> None:
        """Save settings to TOML or JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self._settings.to_dict()
        
        if path.suffix == ".toml":
            if tomli_w is None:
                # Fallback to manual TOML generation
                with open(path, "w") as f:
                    f.write(self._to_toml_string(data))
            else:
                with open(path, "wb") as f:
                    tomli_w.dump(data, f)
        elif path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        self._settings_path = path
    
    def _to_toml_string(self, data: dict, prefix: str = "") -> str:
        """Convert dict to TOML string (simple fallback)."""
        lines = []
        
        # First pass: simple values
        for key, value in data.items():
            if not isinstance(value, dict):
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                elif isinstance(value, bool):
                    lines.append(f'{key} = {str(value).lower()}')
                elif isinstance(value, (int, float)):
                    lines.append(f'{key} = {value}')
        
        # Second pass: nested sections
        for key, value in data.items():
            if isinstance(value, dict):
                section = f"{prefix}{key}" if prefix else key
                lines.append(f"\n[{section}]")
                for k, v in value.items():
                    if isinstance(v, str):
                        lines.append(f'{k} = "{v}"')
                    elif isinstance(v, bool):
                        lines.append(f'{k} = {str(v).lower()}')
                    elif isinstance(v, (int, float)):
                        lines.append(f'{k} = {v}')
        
        return "\n".join(lines)
    
    def load_from_env(self) -> None:
        """Load settings from environment variables (PSTM_*)."""
        for key, value in os.environ.items():
            if key.startswith("PSTM_"):
                # Convert PSTM_GRID_DX_M to grid.dx_m
                setting_key = key[5:].lower().replace("__", ".")
                
                # Try to convert value to appropriate type
                try:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    elif "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                
                try:
                    self.update(**{setting_key: value})
                except AttributeError:
                    pass  # Ignore unknown settings
    
    def get_default_path(self) -> Path:
        """Get default settings file path."""
        # Check environment variable first
        if "PSTM_SETTINGS_PATH" in os.environ:
            return Path(os.environ["PSTM_SETTINGS_PATH"])
        
        # Default to user config directory
        home = Path.home()
        return home / ".pstm" / "settings.toml"
    
    def auto_load(self) -> bool:
        """
        Automatically load settings from default locations.
        
        Search order:
        1. PSTM_SETTINGS_PATH environment variable
        2. ./pstm_settings.toml (current directory)
        3. ~/.pstm/settings.toml (user config)
        
        Returns:
            True if settings were loaded, False if using defaults
        """
        # Check environment variable
        if "PSTM_SETTINGS_PATH" in os.environ:
            path = Path(os.environ["PSTM_SETTINGS_PATH"])
            if path.exists():
                self.load_from_file(path)
                return True
        
        # Check current directory
        local_path = Path("pstm_settings.toml")
        if local_path.exists():
            self.load_from_file(local_path)
            return True
        
        # Check user config
        user_path = self.get_default_path()
        if user_path.exists():
            self.load_from_file(user_path)
            return True
        
        # Load from environment variables
        self.load_from_env()
        
        return False


# =============================================================================
# Module-Level Convenience Functions and Singleton Access
# =============================================================================


_manager = SettingsManager()


def get_settings() -> ApplicationSettings:
    """Get current application settings."""
    return _manager.settings


def get_settings_manager() -> SettingsManager:
    """Get the settings manager instance."""
    return _manager


def load_settings(path: Path | str) -> ApplicationSettings:
    """Load settings from file."""
    _manager.load_from_file(path)
    return _manager.settings


def save_settings(path: Path | str | None = None) -> Path:
    """
    Save current settings to file.
    
    Args:
        path: Output path. If None, uses default path.
        
    Returns:
        Path where settings were saved
    """
    if path is None:
        path = _manager.get_default_path()
    _manager.save_to_file(path)
    return Path(path)


def reset_settings() -> ApplicationSettings:
    """Reset to default settings."""
    _manager.reset()
    return _manager.settings


# Singleton shortcut for direct attribute access
settings = _manager.settings


# =============================================================================
# Settings Documentation Generator
# =============================================================================


def generate_settings_documentation() -> str:
    """Generate markdown documentation for all settings."""
    lines = [
        "# PSTM Application Settings Reference",
        "",
        "This document describes all configurable parameters in PSTM.",
        "",
    ]
    
    default_settings = ApplicationSettings()
    
    for section_name in [
        "grid", "velocity", "aperture", "kernel", "tiling",
        "checkpoint", "io", "qc", "cig", "profiling", "ui", "units"
    ]:
        section = getattr(default_settings, section_name)
        section_class = type(section)
        
        lines.append(f"## {section_name.title()} Settings")
        lines.append("")
        lines.append(f"*{section_class.__doc__ or 'No description'}*")
        lines.append("")
        lines.append("| Parameter | Default | Type | Description |")
        lines.append("|-----------|---------|------|-------------|")
        
        for field_name, field_value in asdict(section).items():
            field_type = type(field_value).__name__
            # Get field annotation if available
            if hasattr(section_class, "__annotations__"):
                ann = section_class.__annotations__.get(field_name, field_type)
                if hasattr(ann, "__origin__"):
                    field_type = str(ann)
            lines.append(f"| `{field_name}` | `{field_value}` | {field_type} | |")
        
        lines.append("")
    
    return "\n".join(lines)


def generate_default_settings_file(path: Path | str, format: str = "toml") -> None:
    """Generate a default settings file with comments."""
    path = Path(path)
    
    if format == "toml":
        content = generate_toml_with_comments()
    else:
        content = json.dumps(ApplicationSettings().to_dict(), indent=2)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def generate_toml_with_comments() -> str:
    """Generate TOML file with descriptive comments."""
    return '''# PSTM Application Settings
# Generated default configuration - modify as needed

# =============================================================================
# Grid Settings - Default spatial and temporal grid parameters
# =============================================================================
[grid]
# Spatial grid spacing (meters)
dx_m = 25.0
dy_m = 25.0

# Temporal grid (milliseconds)
dt_ms = 2.0
t_start_ms = 0.0

# Grid size limits (for warnings)
max_inline_bins = 10000
max_crossline_bins = 10000
max_time_samples = 10000

# Memory estimation
bytes_per_sample = 8
max_output_size_gb = 100.0

# =============================================================================
# Velocity Settings - Velocity model validation and defaults
# =============================================================================
[velocity]
# Valid velocity range (m/s)
min_velocity_ms = 500.0
max_velocity_ms = 10000.0

# QC velocity range (m/s) - warnings outside this range
qc_min_velocity_ms = 1000.0
qc_max_velocity_ms = 8000.0

# Gradient limits
max_gradient_ms_per_s = 5000.0
inversion_threshold_ms = -50.0

# Defaults for constant/linear velocity
default_constant_velocity_ms = 2000.0
default_linear_gradient = 0.5

# =============================================================================
# Aperture Settings - Migration aperture parameters
# =============================================================================
[aperture]
# Aperture range (meters)
min_aperture_m = 500.0
max_aperture_m = 5000.0
default_aperture_m = 3000.0

# Dip and taper
max_dip_degrees = 45.0
taper_fraction = 0.1
taper_type = "cosine"  # Options: linear, cosine, hann

# =============================================================================
# Kernel Settings - Compute kernel parameters
# =============================================================================
[kernel]
# Interpolation
default_interpolation = "linear"  # Options: linear, sinc, cubic
sinc_half_width = 8

# Anti-aliasing
enable_antialiasing = true
antialiasing_filter_length = 8

# Backend-specific chunk sizes
numpy_chunk_traces = 100
numba_chunk_traces = 500
mlx_chunk_traces = 1000
mlx_time_batch = 50

# Numerical stability
epsilon = 1e-10

# Amplitude correction
apply_spreading_correction = true
spreading_power = 1.0

# =============================================================================
# Tiling Settings - Tile planning for memory management
# =============================================================================
[tiling]
# Memory management
max_memory_gb = 8.0
memory_safety_factor = 0.8

# Tile dimensions
min_tile_size = 10
max_tile_size = 500
default_tile_size = 100

# Tile ordering
default_ordering = "snake"  # Options: row_major, column_major, snake, hilbert
tile_overlap_factor = 1.5

# =============================================================================
# Checkpoint Settings - Recovery and checkpointing
# =============================================================================
[checkpoint]
checkpoint_interval_tiles = 10
checkpoint_interval_seconds = 300.0
checkpoint_dir_name = ".pstm_checkpoint"
keep_checkpoint_on_success = false
max_checkpoint_history = 3

# =============================================================================
# I/O Settings - Data handling parameters
# =============================================================================
[io]
# Trace reading
trace_chunk_size = 1000
prefetch_chunks = 2

# Memory mapping
memmap_threshold_mb = 100.0
memmap_mode = "r+"

# Compression
zarr_compressor = "blosc"
zarr_compression_level = 3

# Buffers
double_buffer_size_mb = 512.0

# NaN/Inf handling
max_nan_fraction = 0.01
max_inf_fraction = 0.0
replace_nan_value = 0.0
replace_inf_value = 0.0

# =============================================================================
# QC Settings - Quality control parameters
# =============================================================================
[qc]
# Geometry QC
fold_bin_size_m = 25.0
min_fold_warning = 1

# Offset/azimuth binning
offset_bin_size_m = 100.0
azimuth_bin_size_deg = 10.0

# Verification tolerances
focus_tolerance_xy_m = 50.0
focus_tolerance_t_ms = 20.0
depth_tolerance_ms = 10.0

# Output QC
max_zero_fraction = 0.9
amplitude_clip_percentile = 99.0

# =============================================================================
# CIG Settings - Common Image Gather analysis
# =============================================================================
[cig]
# Offset binning
min_offset_m = 0.0
max_offset_m = 5000.0
n_offset_bins = 20

# Semblance analysis
semblance_window_ms = 100.0
n_velocities = 100
velocity_range_fraction = 0.2

# Flatness threshold
flatness_threshold = 0.5

# =============================================================================
# Profiling Settings - Performance monitoring
# =============================================================================
[profiling]
memory_sample_interval_s = 1.0
track_peak_memory = true
enable_detailed_timing = false
timer_precision = "ms"
report_interval_tiles = 10

# =============================================================================
# UI Settings - User interface configuration
# =============================================================================
[ui]
use_colors = true
show_progress_bar = true
progress_update_hz = 2.0
wizard_theme = "dark"
show_advanced_options = false
validate_on_change = true
show_warnings = true

# =============================================================================
# Unit Settings - Unit handling and conversion
# =============================================================================
[units]
default_coordinate_unit = "m"
feet_per_meter = 3.28084
meters_per_foot = 0.3048
ms_per_second = 1000.0
coordinate_decimals = 2
time_decimals = 1
velocity_decimals = 0
'''


# =============================================================================
# Auto-load settings on module import
# =============================================================================

# Try to auto-load settings (silent if not found)
try:
    _manager.auto_load()
except Exception:
    pass  # Use defaults if auto-load fails
