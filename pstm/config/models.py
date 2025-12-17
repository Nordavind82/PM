"""
Pydantic configuration models for PSTM.

This module defines all configuration dataclasses used throughout the migration pipeline.
All models are validated, serializable, and support JSON/YAML export.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# =============================================================================
# Enumerations
# =============================================================================


class CoordinateUnit(str, Enum):
    """Coordinate system units."""
    METERS = "meters"
    FEET = "feet"


class VelocityType(str, Enum):
    """Type of velocity model."""
    VRMS = "vrms"
    VINT = "vint"
    VAVG = "vavg"


class VelocitySource(str, Enum):
    """Source of velocity model."""
    CONSTANT = "constant"
    LINEAR_V0K = "linear_v0k"
    TABLE_1D = "table_1d"
    CUBE_3D = "cube_3d"


class InterpolationMethod(str, Enum):
    """Trace interpolation method."""
    NEAREST = "nearest"
    LINEAR = "linear"
    SINC8 = "sinc8"
    KAISER = "kaiser"


class AntiAliasingMethod(str, Enum):
    """Anti-aliasing filter method."""
    NONE = "none"
    TRIANGLE = "triangle"
    OFFSET_SECTORING = "offset_sectoring"
    DIP_FILTER = "dip_filter"


class TaperType(str, Enum):
    """Aperture taper function type."""
    COSINE = "cosine"
    LINEAR = "linear"
    GAUSSIAN = "gaussian"


class ComputeBackend(str, Enum):
    """Compute kernel backend."""
    AUTO = "auto"
    NUMPY = "numpy"
    NUMBA_CPU = "numba_cpu"
    MLX_METAL = "mlx_metal"
    METAL_CPP = "metal_cpp"
    METAL_COMPILED = "metal_compiled"  # PyObjC + compiled .metallib (fastest on Apple Silicon)


class OutputFormat(str, Enum):
    """Output file format."""
    ZARR = "zarr"
    SEGY = "segy"
    NUMPY = "numpy"


class SpatialIndexType(str, Enum):
    """Spatial index algorithm."""
    KDTREE = "kdtree"
    BALLTREE = "balltree"


# =============================================================================
# Type Aliases
# =============================================================================

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Percentage = Annotated[float, Field(ge=0, le=100)]
Fraction = Annotated[float, Field(ge=0, le=1)]


# =============================================================================
# Base Configuration
# =============================================================================


class BaseConfig(BaseModel):
    """Base configuration with common settings."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=False,
    )


# =============================================================================
# Input Configuration
# =============================================================================


class ColumnMapping(BaseConfig):
    """Mapping of header column names to internal names."""

    source_x: str = Field(default="SOU_X", description="Source X coordinate column")
    source_y: str = Field(default="SOU_Y", description="Source Y coordinate column")
    receiver_x: str = Field(default="REC_X", description="Receiver X coordinate column")
    receiver_y: str = Field(default="REC_Y", description="Receiver Y coordinate column")
    cdp_x: str | None = Field(default="CDP_X", description="CDP X coordinate column")
    cdp_y: str | None = Field(default="CDP_Y", description="CDP Y coordinate column")
    offset: str | None = Field(default="OFFSET", description="Offset column")
    azimuth: str | None = Field(default="AZIMUTH", description="Azimuth column")
    shot_id: str = Field(default="FFID", description="Shot/FFID column")
    trace_in_shot: str = Field(default="CHAN", description="Trace/channel in shot column")
    trace_index: str = Field(default="trace_idx", description="Global trace index column")
    coord_scalar: str | None = Field(
        default=None, description="Coordinate scalar column (SEG-Y style)"
    )


class InputConfig(BaseConfig):
    """Configuration for input data."""

    # Paths
    traces_path: Path = Field(description="Path to Zarr trace data")
    headers_path: Path = Field(description="Path to Parquet headers")

    # Column mapping
    columns: ColumnMapping = Field(default_factory=ColumnMapping)

    # Coordinate system
    coordinate_unit: CoordinateUnit = Field(
        default=CoordinateUnit.METERS,
        description="Coordinate system units",
    )
    apply_coord_scalar: bool = Field(
        default=False,
        description="Apply coordinate scalar from headers",
    )
    crs: str | None = Field(
        default=None,
        description="Coordinate reference system (e.g., 'EPSG:32640')",
    )

    # Data properties (auto-detected if None)
    sample_rate_ms: PositiveFloat | None = Field(
        default=None,
        description="Sample rate in milliseconds (auto-detected if None)",
    )
    num_samples: PositiveInt | None = Field(
        default=None,
        description="Number of samples per trace (auto-detected if None)",
    )
    num_traces: PositiveInt | None = Field(
        default=None,
        description="Number of traces (auto-detected if None)",
    )
    start_time_ms: float | None = Field(
        default=None,
        description="Recording start time in ms (auto-detected if None)",
    )
    transposed: bool = Field(
        default=False,
        description="True if traces stored as (n_samples, n_traces) instead of (n_traces, n_samples)",
    )

    @field_validator("traces_path", "headers_path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Ensure path is absolute."""
        return v.resolve() if v else v


class GeometryConfig(BaseConfig):
    """Configuration for geometry analysis and spatial indexing."""

    # Spatial index
    index_type: SpatialIndexType = Field(
        default=SpatialIndexType.KDTREE,
        description="Spatial index algorithm",
    )
    index_key: Literal["midpoint", "cdp", "source"] = Field(
        default="midpoint",
        description="Coordinate type for spatial indexing",
    )

    # Precomputed index (optional)
    index_path: Path | None = Field(
        default=None,
        description="Path to precomputed spatial index",
    )

    # Offset binning for anti-aliasing
    offset_bin_size: PositiveFloat | None = Field(
        default=None,
        description="Offset bin size for sectoring (meters)",
    )

    # Survey bounds override (optional)
    x_min: float | None = Field(default=None, description="Override survey X minimum")
    x_max: float | None = Field(default=None, description="Override survey X maximum")
    y_min: float | None = Field(default=None, description="Override survey Y minimum")
    y_max: float | None = Field(default=None, description="Override survey Y maximum")


# =============================================================================
# Velocity Configuration
# =============================================================================


class VelocityConfig(BaseConfig):
    """Configuration for velocity model."""

    # Source type
    source: VelocitySource = Field(
        default=VelocitySource.CUBE_3D,
        description="Velocity model source type",
    )
    velocity_type: VelocityType = Field(
        default=VelocityType.VRMS,
        description="Type of velocity (Vrms, Vint, Vavg)",
    )

    # For CONSTANT source
    constant_velocity: PositiveFloat | None = Field(
        default=None,
        description="Constant velocity value (m/s)",
    )

    # For LINEAR_V0K source: V(t) = v0 + k * t
    v0: PositiveFloat | None = Field(
        default=None,
        description="Velocity at t=0 (m/s)",
    )
    k: float | None = Field(
        default=None,
        description="Velocity gradient (m/s per second)",
    )

    # For TABLE_1D source
    velocity_table: list[tuple[float, float]] | None = Field(
        default=None,
        description="List of (time_ms, velocity_m_s) pairs",
    )

    # For CUBE_3D source
    velocity_path: Path | None = Field(
        default=None,
        description="Path to 3D velocity cube (Zarr or SEG-Y)",
    )

    # Interpolation
    interpolation: InterpolationMethod = Field(
        default=InterpolationMethod.LINEAR,
        description="Velocity interpolation method",
    )
    precompute_to_output_grid: bool = Field(
        default=True,
        description="Pre-interpolate velocity to output grid",
    )

    # Validation bounds
    min_velocity: PositiveFloat = Field(
        default=1000.0,
        description="Minimum valid velocity (m/s)",
    )
    max_velocity: PositiveFloat = Field(
        default=8000.0,
        description="Maximum valid velocity (m/s)",
    )

    @model_validator(mode="after")
    def validate_velocity_source(self) -> "VelocityConfig":
        """Validate that required fields are present for each source type."""
        if self.source == VelocitySource.CONSTANT:
            if self.constant_velocity is None:
                raise ValueError("constant_velocity required for CONSTANT source")
        elif self.source == VelocitySource.LINEAR_V0K:
            if self.v0 is None or self.k is None:
                raise ValueError("v0 and k required for LINEAR_V0K source")
        elif self.source == VelocitySource.TABLE_1D:
            if not self.velocity_table:
                raise ValueError("velocity_table required for TABLE_1D source")
        elif self.source == VelocitySource.CUBE_3D:
            if self.velocity_path is None:
                raise ValueError("velocity_path required for CUBE_3D source")
        return self


# =============================================================================
# Algorithm Configuration
# =============================================================================


class ApertureConfig(BaseConfig):
    """Configuration for migration aperture."""

    # Aperture mode
    time_dependent: bool = Field(
        default=True,
        description="Use time-dependent aperture",
    )

    # Aperture parameters
    max_dip_degrees: float = Field(
        default=45.0,
        ge=0,
        le=89,
        description="Maximum dip angle for aperture calculation",
    )
    min_aperture_m: PositiveFloat = Field(
        default=500.0,
        description="Minimum aperture radius (meters)",
    )
    max_aperture_m: PositiveFloat = Field(
        default=10000.0,
        description="Maximum aperture radius (meters)",
    )

    # Taper
    taper_enabled: bool = Field(
        default=True,
        description="Enable aperture edge taper",
    )
    taper_type: TaperType = Field(
        default=TaperType.COSINE,
        description="Taper function type",
    )
    taper_fraction: Fraction = Field(
        default=0.1,
        description="Taper width as fraction of aperture",
    )

    @model_validator(mode="after")
    def validate_aperture_range(self) -> "ApertureConfig":
        """Ensure min_aperture <= max_aperture."""
        if self.min_aperture_m > self.max_aperture_m:
            raise ValueError("min_aperture_m must be <= max_aperture_m")
        return self


class AntiAliasingConfig(BaseConfig):
    """Configuration for anti-aliasing."""

    enabled: bool = Field(
        default=True,
        description="Enable anti-aliasing",
    )
    method: AntiAliasingMethod = Field(
        default=AntiAliasingMethod.TRIANGLE,
        description="Anti-aliasing method",
    )

    # Filter bank parameters
    num_filters: PositiveInt = Field(
        default=32,
        description="Number of filters in filter bank",
    )
    min_frequency_hz: PositiveFloat = Field(
        default=5.0,
        description="Minimum frequency (Hz)",
    )
    max_frequency_hz: PositiveFloat = Field(
        default=80.0,
        description="Maximum frequency (Hz)",
    )


class TimeVariantConfig(BaseConfig):
    """Configuration for time-variant sampling.

    Allows coarser sampling at deeper times where high frequencies
    are naturally attenuated, providing significant performance gains.
    """

    enabled: bool = Field(
        default=False,
        description="Enable time-variant sampling",
    )
    frequency_table: list[tuple[float, float]] = Field(
        default_factory=lambda: [
            (0.0, 80.0),
            (1000.0, 50.0),
            (2500.0, 30.0),
            (5000.0, 20.0),
        ],
        description="List of (time_ms, max_freq_hz) pairs defining frequency vs time",
    )
    min_downsample_factor: PositiveInt = Field(
        default=1,
        description="Minimum downsample factor (1 = no downsampling)",
    )
    max_downsample_factor: PositiveInt = Field(
        default=8,
        description="Maximum downsample factor (power of 2)",
    )

    @field_validator("frequency_table")
    @classmethod
    def validate_frequency_table(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Validate frequency table entries."""
        if len(v) < 2:
            raise ValueError("Frequency table must have at least 2 entries")

        # Sort by time
        v = sorted(v, key=lambda x: x[0])

        # Check times are unique and increasing
        times = [t for t, f in v]
        if len(times) != len(set(times)):
            raise ValueError("Times in frequency table must be unique")

        # Check frequencies are positive
        for t, f in v:
            if f <= 0:
                raise ValueError(f"Frequencies must be positive, got {f} at t={t}")

        return v

    @field_validator("max_downsample_factor")
    @classmethod
    def validate_max_downsample(cls, v: int) -> int:
        """Ensure max downsample is power of 2."""
        if v & (v - 1) != 0:
            raise ValueError(f"max_downsample_factor must be power of 2, got {v}")
        return v

    # Offset sectoring parameters
    offset_sectors: list[tuple[float, float]] | None = Field(
        default=None,
        description="Manual offset sector boundaries (min, max) pairs",
    )
    auto_offset_sectors: PositiveInt | None = Field(
        default=None,
        description="Number of automatic offset sectors",
    )


class AmplitudeConfig(BaseConfig):
    """Configuration for amplitude handling."""

    # Weight functions
    geometrical_spreading: bool = Field(
        default=True,
        description="Apply geometrical spreading correction",
    )
    obliquity_factor: bool = Field(
        default=True,
        description="Apply obliquity (cosine) factor",
    )
    wavelet_stretch_correction: bool = Field(
        default=False,
        description="Apply wavelet stretch correction (for AVO)",
    )

    # Normalization
    trace_normalization: bool = Field(
        default=False,
        description="Normalize traces before migration",
    )
    normalization_window_ms: PositiveFloat | None = Field(
        default=None,
        description="RMS normalization window (ms)",
    )


class MuteConfig(BaseConfig):
    """Configuration for muting."""

    # Inner mute (stretched wavelets)
    inner_mute_enabled: bool = Field(
        default=False,
        description="Enable inner mute",
    )
    inner_mute_stretch_factor: PositiveFloat = Field(
        default=1.5,
        description="Maximum allowed stretch factor",
    )

    # Outer mute
    outer_mute_enabled: bool = Field(
        default=False,
        description="Enable outer mute",
    )
    outer_mute_velocity: PositiveFloat | None = Field(
        default=None,
        description="Outer mute velocity (m/s)",
    )


class AlgorithmConfig(BaseConfig):
    """Configuration for migration algorithm parameters."""

    # Core algorithm
    interpolation: InterpolationMethod = Field(
        default=InterpolationMethod.LINEAR,
        description="Trace sample interpolation method",
    )

    # Sub-configurations
    aperture: ApertureConfig = Field(default_factory=ApertureConfig)
    anti_aliasing: AntiAliasingConfig = Field(default_factory=AntiAliasingConfig)
    amplitude: AmplitudeConfig = Field(default_factory=AmplitudeConfig)
    mute: MuteConfig = Field(default_factory=MuteConfig)
    time_variant: TimeVariantConfig = Field(default_factory=TimeVariantConfig)


# =============================================================================
# Output Configuration
# =============================================================================

# OutputGridConfig is now imported from output_grid.py for enhanced functionality
# (supports corner-point definition for rotated grids)
from pstm.config.output_grid import OutputGridConfig


class OutputProductsConfig(BaseConfig):
    """Configuration for output products."""

    stacked_image: bool = Field(
        default=True,
        description="Output stacked migration image",
    )
    fold_volume: bool = Field(
        default=True,
        description="Output fold (hit count) volume",
    )
    common_image_gathers: bool = Field(
        default=False,
        description="Output common image gathers",
    )
    cig_offset_bins: list[float] | None = Field(
        default=None,
        description="Offset bin centers for CIGs (meters)",
    )
    # Offset/azimuth gather output bins
    output_gathers: bool = Field(
        default=False,
        description="Output separate migrated volumes per offset/azimuth bin",
    )
    gather_offset_ranges: list[tuple[float, float]] | None = Field(
        default=None,
        description="List of (min, max) offset ranges defining output gather bins (meters)",
    )
    semblance_panels: bool = Field(
        default=False,
        description="Output semblance panels at key locations",
    )
    illumination_map: bool = Field(
        default=False,
        description="Output illumination/coverage map",
    )


class OutputConfig(BaseConfig):
    """Configuration for output data."""

    # Output directory
    output_dir: Path = Field(description="Output directory path")

    # Grid definition
    grid: OutputGridConfig

    # Products
    products: OutputProductsConfig = Field(default_factory=OutputProductsConfig)

    # Format
    format: OutputFormat = Field(
        default=OutputFormat.ZARR,
        description="Output file format",
    )
    compression: str = Field(
        default="blosc",
        description="Compression codec for Zarr",
    )
    compression_level: int = Field(
        default=3,
        ge=1,
        le=9,
        description="Compression level",
    )

    # Chunking
    chunk_x: PositiveInt = Field(
        default=64,
        description="Chunk size in X",
    )
    chunk_y: PositiveInt = Field(
        default=64,
        description="Chunk size in Y",
    )
    chunk_t: PositiveInt | None = Field(
        default=None,
        description="Chunk size in T (None = full time axis)",
    )

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Ensure output directory is absolute."""
        return v.resolve()


# =============================================================================
# Execution Configuration
# =============================================================================


class ResourceConfig(BaseConfig):
    """Configuration for compute resources."""

    # Memory
    max_memory_gb: PositiveFloat = Field(
        default=32.0,
        description="Maximum memory usage (GB)",
    )
    memory_safety_factor: Fraction = Field(
        default=0.75,
        description="Fraction of max_memory to actually use",
    )

    # CPU
    num_workers: PositiveInt | None = Field(
        default=None,
        description="Number of parallel workers (None = auto)",
    )

    # Compute backend
    backend: ComputeBackend = Field(
        default=ComputeBackend.AUTO,
        description="Compute kernel backend",
    )


class TilingConfig(BaseConfig):
    """Configuration for output tiling."""

    # Tile sizing
    auto_tile_size: bool = Field(
        default=True,
        description="Automatically determine tile size",
    )
    tile_nx: PositiveInt | None = Field(
        default=None,
        description="Manual tile size in X",
    )
    tile_ny: PositiveInt | None = Field(
        default=None,
        description="Manual tile size in Y",
    )

    # Tile ordering
    ordering: Literal["row_major", "column_major", "snake", "hilbert"] = Field(
        default="snake",
        description="Tile processing order",
    )


class CheckpointConfig(BaseConfig):
    """Configuration for checkpointing."""

    enabled: bool = Field(
        default=True,
        description="Enable checkpointing",
    )
    interval_tiles: PositiveInt = Field(
        default=100,
        description="Checkpoint every N tiles",
    )
    checkpoint_dir: Path | None = Field(
        default=None,
        description="Checkpoint directory (default: output_dir/.checkpoint)",
    )


class ExecutionConfig(BaseConfig):
    """Configuration for execution parameters."""

    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # Verbosity
    verbose: bool = Field(
        default=True,
        description="Enable verbose output",
    )
    log_file: Path | None = Field(
        default=None,
        description="Log file path",
    )

    # Dry run
    dry_run: bool = Field(
        default=False,
        description="Validate configuration without running",
    )


# =============================================================================
# Root Configuration
# =============================================================================

# Import DataSelectionConfig
from pstm.config.data_selection import DataSelectionConfig


class MigrationConfig(BaseConfig):
    """
    Root configuration model for 3D PSTM.

    This is the main configuration object that combines all sub-configurations.
    It can be serialized to/from JSON or YAML for persistence.
    
    WORKFLOW ORDER (Important for GUI wizard):
    1. Input Data      - Load traces & headers
    2. Survey Geometry - Analyze geometry, build spatial index  
    3. Output Grid     - Define output by corner points + bin size
    4. Velocity Model  - Configure velocity (interpolated to OUTPUT grid)
    5. Data Selection  - Filter traces by offset/azimuth/OVT
    6. Algorithm       - Migration parameters
    7. Execution       - Run migration
    8. Results         - QC and export
    
    Key design decisions:
    - Velocity is prepared AFTER output grid to ensure correct interpolation
    - Data selection has NO validation - user takes full responsibility
    - Output bin size is INDEPENDENT of input data spacing
    """

    # Sub-configurations (ordered by workflow)
    input: InputConfig
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    output: OutputConfig  # Output grid defined BEFORE velocity
    velocity: VelocityConfig  # Velocity interpolated TO output grid
    data_selection: DataSelectionConfig = Field(
        default_factory=DataSelectionConfig.use_all,
        description="Data selection/filtering configuration",
    )
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    # Metadata
    name: str = Field(
        default="unnamed_migration",
        description="Project/job name",
    )
    description: str | None = Field(
        default=None,
        description="Project description",
    )

    @model_validator(mode="after")
    def validate_coordinate_consistency(self) -> "MigrationConfig":
        """Validate coordinate system consistency across configs."""
        # Check that output grid is within reasonable bounds of input survey
        # (This would require loading data, so we just do basic checks here)
        return self

    def to_json(self, path: Path | str, indent: int = 2) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.write_text(self.model_dump_json(indent=indent))

    @classmethod
    def from_json(cls, path: Path | str) -> "MigrationConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        return cls.model_validate_json(path.read_text())

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of key configuration parameters."""
        summary = {
            "name": self.name,
            "input_traces": str(self.input.traces_path),
            "output_grid_shape": self.output.grid.shape,
            "output_size_gb": f"{self.output.grid.size_gb:.2f}",
            "velocity_source": self.velocity.source.value,
            "compute_backend": self.execution.resources.backend.value,
            "anti_aliasing": self.algorithm.anti_aliasing.method.value,
            "max_dip": f"{self.algorithm.aperture.max_dip_degrees}°",
        }
        
        # Add data selection info
        if self.data_selection.mode.value != "all":
            summary["data_selection"] = self.data_selection.get_summary()
        
        return summary


# =============================================================================
# Convenience factory functions
# =============================================================================


def create_minimal_config(
    traces_path: str | Path,
    headers_path: str | Path,
    output_dir: str | Path,
    velocity: float | Path,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    t_range_ms: tuple[float, float],
    dx: float | None = None,
    dy: float | None = None,
    dt_ms: float | None = None,
) -> MigrationConfig:
    """
    Create a minimal migration configuration with sensible defaults.

    Args:
        traces_path: Path to Zarr trace data
        headers_path: Path to Parquet headers
        output_dir: Output directory
        velocity: Constant velocity (m/s) or path to velocity cube
        x_range: (x_min, x_max) tuple
        y_range: (y_min, y_max) tuple
        t_range_ms: (t_min, t_max) tuple in milliseconds
        dx: X spacing (default from settings)
        dy: Y spacing (default from settings)
        dt_ms: Time spacing (default from settings)

    Returns:
        Configured MigrationConfig instance
    """
    # Get defaults from settings
    from pstm.settings import get_settings
    s = get_settings()
    
    if dx is None:
        dx = s.grid.dx_m
    if dy is None:
        dy = s.grid.dy_m
    if dt_ms is None:
        dt_ms = s.grid.dt_ms
    
    # Determine velocity config
    if isinstance(velocity, (int, float)):
        velocity_config = VelocityConfig(
            source=VelocitySource.CONSTANT,
            constant_velocity=float(velocity),
        )
    else:
        velocity_config = VelocityConfig(
            source=VelocitySource.CUBE_3D,
            velocity_path=Path(velocity),
        )

    return MigrationConfig(
        input=InputConfig(
            traces_path=Path(traces_path),
            headers_path=Path(headers_path),
        ),
        output=OutputConfig(  # Output defined before velocity
            output_dir=Path(output_dir),
            grid=OutputGridConfig.from_bounding_box(
                x_min=x_range[0],
                x_max=x_range[1],
                y_min=y_range[0],
                y_max=y_range[1],
                dx=dx,
                dy=dy,
                t_min_ms=t_range_ms[0],
                t_max_ms=t_range_ms[1],
                dt_ms=dt_ms,
            ),
        ),
        velocity=velocity_config,  # Velocity after output grid
    )


def create_config_with_corners(
    traces_path: str | Path,
    headers_path: str | Path,
    output_dir: str | Path,
    velocity: float | Path,
    corners: tuple[
        tuple[float, float],  # C1 (origin)
        tuple[float, float],  # C2
        tuple[float, float],  # C3
        tuple[float, float],  # C4
    ],
    t_range_ms: tuple[float, float],
    dx: float | None = None,
    dy: float | None = None,
    dt_ms: float | None = None,
) -> MigrationConfig:
    """
    Create migration configuration with corner-point grid definition.
    
    This allows defining rotated output grids.

    Args:
        traces_path: Path to Zarr trace data
        headers_path: Path to Parquet headers
        output_dir: Output directory
        velocity: Constant velocity (m/s) or path to velocity cube
        corners: Four corner points ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
        t_range_ms: (t_min, t_max) tuple in milliseconds
        dx: Inline bin size (default from settings)
        dy: Crossline bin size (default from settings)
        dt_ms: Time spacing (default from settings)

    Returns:
        Configured MigrationConfig instance
    """
    from pstm.settings import get_settings
    s = get_settings()
    
    if dx is None:
        dx = s.grid.dx_m
    if dy is None:
        dy = s.grid.dy_m
    if dt_ms is None:
        dt_ms = s.grid.dt_ms
    
    # Determine velocity config
    if isinstance(velocity, (int, float)):
        velocity_config = VelocityConfig(
            source=VelocitySource.CONSTANT,
            constant_velocity=float(velocity),
        )
    else:
        velocity_config = VelocityConfig(
            source=VelocitySource.CUBE_3D,
            velocity_path=Path(velocity),
        )

    return MigrationConfig(
        input=InputConfig(
            traces_path=Path(traces_path),
            headers_path=Path(headers_path),
        ),
        output=OutputConfig(
            output_dir=Path(output_dir),
            grid=OutputGridConfig.from_corners(
                corner1=corners[0],
                corner2=corners[1],
                corner3=corners[2],
                corner4=corners[3],
                dx=dx,
                dy=dy,
                t_min_ms=t_range_ms[0],
                t_max_ms=t_range_ms[1],
                dt_ms=dt_ms,
            ),
        ),
        velocity=velocity_config,
    )
