#!/usr/bin/env python3
"""
Run PSTM migration using presorted common offset gathers.

This script:
1. Loads velocity from SEG-Y file with custom byte positions
2. Processes common offset gathers from presorted data
3. Uses time-variant sampling for efficiency (coarser at depth)
4. Provides progress tracking

Usage:
    python run_pstm_common_offset.py [--dry-run] [--resume]
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add pstm to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_file_logging(output_dir: Path, offset_bin: int | None = None) -> Path:
    """
    Configure logging to save all debug output to a file.

    Args:
        output_dir: Base output directory
        offset_bin: Optional offset bin number for log file name

    Returns:
        Path to the log file
    """
    # Create log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if offset_bin is not None:
        log_filename = f"pstm_migration_bin_{offset_bin:02d}_{timestamp}.log"
    else:
        log_filename = f"pstm_migration_{timestamp}.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_filename

    # Create file handler with detailed format
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Configure root logger to capture everything
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # CRITICAL: Configure the pstm.migration.debug logger BEFORE executor runs
    # This logger is used for memory debugging in metal_compiled.py and executor.py
    debug_logger = logging.getLogger("pstm.migration.debug")
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.addHandler(file_handler)
    debug_logger.propagate = True  # Also propagate to root

    # Configure pstm logger hierarchy
    pstm_logger = logging.getLogger("pstm")
    pstm_logger.setLevel(logging.DEBUG)
    pstm_logger.addHandler(file_handler)
    pstm_logger.propagate = True

    # Also add to pstm.pstm (double prefix issue workaround)
    pstm_pstm_logger = logging.getLogger("pstm.pstm")
    pstm_pstm_logger.setLevel(logging.DEBUG)
    pstm_pstm_logger.addHandler(file_handler)
    pstm_pstm_logger.propagate = True

    print(f"Logging configured: {log_path}")
    print(f"  - Root logger handlers: {len(root_logger.handlers)}")
    print(f"  - pstm.migration.debug handlers: {len(debug_logger.handlers)}")

    return log_path

import numpy as np
import segyio
import zarr
from typing import Callable

from pstm.config.models import (
    MigrationConfig,
    InputConfig,
    VelocityConfig,
    AlgorithmConfig,
    OutputConfig,
    ExecutionConfig,
    OutputGridConfig,
    ApertureConfig,
    AmplitudeConfig,
    ResourceConfig,
    TilingConfig,
    CheckpointConfig,
    VelocitySource,
    InterpolationMethod,
    ComputeBackend,
    OutputFormat,
    TimeVariantConfig,
    AntiAliasingConfig,
    AntiAliasingMethod,
    ColumnMapping,
)
from pstm.pipeline.executor import run_migration, ProgressInfo

# =============================================================================
# Configuration Constants
# =============================================================================

# Input data paths
COMMON_OFFSET_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
VELOCITY_SEGY_PATH = Path("/Users/olegadamovich/SeismicData/vels.sgy")

# Output directory
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset")

# Velocity SEG-Y byte positions
INLINE_BYTE = 237  # 4-byte integer
XLINE_BYTE = 21   # 4-byte integer (typically right after inline)

# Grid parameters (input = output)
# Inline direction: C1->C2, 511 inlines, 25m spacing
# Crossline direction: C1->C4, 427 crosslines, 12.5m spacing
DX = 25.0   # Inline bin size (m)
DY = 12.5   # Crossline bin size (m)
DT_MS = 2.0 # Time sample interval (ms)

# Time range
T_MIN_MS = 0.0
T_MAX_MS = 2000.0  # 1600 samples * 2ms

# Grid corners (rotated grid from inline/crossline geometry)
# C1 (IL=1, XL=1) - Origin
# C2 (IL=511, XL=1) - Inline end
# C3 (IL=511, XL=427) - Far corner
# C4 (IL=1, XL=427) - Crossline end
# Corners computed to give exactly:
#   - Inline extent: 12,750m → 511 points with 25m spacing
#   - Crossline extent: 5,325m → 427 points with 12.5m spacing
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),  # Origin (IL=1, XL=1)
    'c2': (627094.02, 5106803.16),  # Inline end (IL=511, XL=1)
    'c3': (631143.35, 5110261.43),  # Far corner (IL=511, XL=427)
    'c4': (622862.92, 5119956.77),  # Crossline end (IL=1, XL=427)
}

# Algorithm parameters
MAX_APERTURE_M = 2000.0
MIN_APERTURE_M = 500.0
MAX_DIP_DEGREES = 65.0

# Tile size (reduced from 96 to 64 for better memory management)
TILE_NX = 512
TILE_NY = 512

# Time-variant sampling configuration
# Format: [(time_ms, max_freq_hz), ...]
# 120 Hz at shallow (up to 800 ms)
# 80 Hz at 1200 ms
# 50 Hz at deeper times
TIME_VARIANT_TABLE = [
    (0.0, 120.0),      # At surface: preserve up to 120 Hz
    (800.0, 120.0),    # Up to 800ms: preserve 120 Hz
    (1200.0, 80.0),    # At 1200ms: 80 Hz sufficient
    (2000.0, 50.0),    # At 2000ms and deeper: 50 Hz
    (3200.0, 50.0),    # Bottom of data
]


# =============================================================================
# Velocity Loading from SEG-Y
# =============================================================================

def load_segy_velocity_scattered(
    segy_path: Path,
    inline_byte: int = 237,
    xline_byte: int = 21,
    target_nx: int = 50,
    target_ny: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load velocity model from SEG-Y file and grid to regular coordinates.

    Handles scattered velocity data by interpolating to a regular grid.

    Args:
        segy_path: Path to velocity SEG-Y file
        inline_byte: Byte position for inline number (4-byte integer)
        xline_byte: Byte position for crossline number (4-byte integer)
        target_nx: Target grid size in X
        target_ny: Target grid size in Y

    Returns:
        velocity_cube: 3D array (nx, ny, nt)
        x_axis: Regular X coordinates
        y_axis: Regular Y coordinates
        t_axis_ms: Time axis in milliseconds
    """
    from scipy.interpolate import griddata

    print(f"Loading velocity from: {segy_path}")

    # Open in non-strict mode to handle various formats
    with segyio.open(str(segy_path), 'r', strict=False) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)
        t_axis_ms = f.samples.astype(np.float64)

        print(f"  Traces: {n_traces}, Samples: {n_samples}")
        print(f"  Time range: {t_axis_ms[0]:.0f} - {t_axis_ms[-1]:.0f} ms")

        # Get coordinate scalar
        h0 = f.header[0]
        scalar = h0[segyio.TraceField.SourceGroupScalar]
        if scalar < 0:
            scale = 1.0 / abs(scalar)
        elif scalar > 0:
            scale = float(scalar)
        else:
            scale = 1.0
        print(f"  Coordinate scalar: {scalar} (scale factor: {scale})")

        # Extract coordinates and velocity data
        x_coords = np.zeros(n_traces, dtype=np.float64)
        y_coords = np.zeros(n_traces, dtype=np.float64)
        velocity_data = np.zeros((n_traces, n_samples), dtype=np.float32)

        for i in range(n_traces):
            h = f.header[i]
            x_coords[i] = h[segyio.TraceField.CDP_X] * scale
            y_coords[i] = h[segyio.TraceField.CDP_Y] * scale
            velocity_data[i, :] = f.trace[i]

        print(f"  X range: {x_coords.min():.2f} - {x_coords.max():.2f}")
        print(f"  Y range: {y_coords.min():.2f} - {y_coords.max():.2f}")
        print(f"  Velocity range: {velocity_data.min():.0f} - {velocity_data.max():.0f} m/s")

    # Create regular grid
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_axis = np.linspace(x_min, x_max, target_nx)
    y_axis = np.linspace(y_min, y_max, target_ny)

    print(f"  Interpolating to {target_nx} x {target_ny} grid...")

    # Interpolate each time slice
    xi, yi = np.meshgrid(x_axis, y_axis, indexing='ij')
    points = np.column_stack([x_coords, y_coords])

    velocity_cube = np.zeros((target_nx, target_ny, n_samples), dtype=np.float32)

    for t in range(n_samples):
        velocity_cube[:, :, t] = griddata(
            points,
            velocity_data[:, t],
            (xi, yi),
            method='linear',
            fill_value=np.nanmean(velocity_data[:, t])
        )

    print(f"  Gridded velocity cube shape: {velocity_cube.shape}")
    print(f"  Gridded velocity range: {np.nanmin(velocity_cube):.0f} - {np.nanmax(velocity_cube):.0f} m/s")

    return velocity_cube, x_axis, y_axis, t_axis_ms


def load_segy_velocity_custom_bytes(
    segy_path: Path,
    inline_byte: int = 237,
    xline_byte: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load velocity model from SEG-Y file with custom byte positions.

    Falls back to scattered point interpolation if standard loading fails.

    Args:
        segy_path: Path to velocity SEG-Y file
        inline_byte: Byte position for inline number (4-byte integer)
        xline_byte: Byte position for crossline number (4-byte integer)

    Returns:
        velocity_cube: 3D array (nx, ny, nt)
        x_coords: X coordinates
        y_coords: Y coordinates
        t_axis_ms: Time axis in milliseconds
    """
    print(f"Loading velocity from: {segy_path}")
    print(f"  Inline byte: {inline_byte}, Xline byte: {xline_byte}")

    # First try standard loading with specified byte positions
    try:
        with segyio.open(
            str(segy_path),
            'r',
            iline=inline_byte,
            xline=xline_byte,
        ) as f:
            n_inlines = len(f.ilines)
            n_xlines = len(f.xlines)
            n_samples = len(f.samples)

            print(f"  Velocity cube dimensions: {n_inlines} IL x {n_xlines} XL x {n_samples} samples")
            print(f"  Inlines: {f.ilines[0]} - {f.ilines[-1]}")
            print(f"  Xlines: {f.xlines[0]} - {f.xlines[-1]}")
            print(f"  Time range: {f.samples[0]} - {f.samples[-1]} ms")

            # Coordinate scaling
            h0 = f.header[0]
            scalar = h0[segyio.TraceField.SourceGroupScalar]
            if scalar < 0:
                scale = 1.0 / abs(scalar)
            elif scalar > 0:
                scale = float(scalar)
            else:
                scale = 1.0
            print(f"  Coordinate scalar: {scalar} (scale factor: {scale})")

            # Extract coordinates from trace headers
            x_coords = np.zeros(n_inlines, dtype=np.float64)
            y_coords = np.zeros(n_xlines, dtype=np.float64)

            # Get X coords from first trace of each inline
            for i, il in enumerate(f.ilines):
                h = f.header[i * n_xlines]
                x_coords[i] = h[segyio.TraceField.CDP_X] * scale

            # Get Y coords from first crossline trace of first inline
            for j, xl in enumerate(f.xlines):
                h = f.header[j]
                y_coords[j] = h[segyio.TraceField.CDP_Y] * scale

            # Check if coordinates are valid
            if x_coords.max() == x_coords.min():
                print("  WARNING: X coordinates are constant, using inline indices as proxy")
                x_coords = f.ilines.astype(np.float64) * DX

            if y_coords.max() == y_coords.min():
                print("  WARNING: Y coordinates are constant, using xline indices as proxy")
                y_coords = f.xlines.astype(np.float64) * DY

            print(f"  X range: {x_coords.min():.2f} - {x_coords.max():.2f}")
            print(f"  Y range: {y_coords.min():.2f} - {y_coords.max():.2f}")

            # Load velocity cube
            print("  Loading velocity data...")
            velocity_cube = np.zeros((n_inlines, n_xlines, n_samples), dtype=np.float32)

            for i in range(n_inlines):
                for j in range(n_xlines):
                    trace_idx = i * n_xlines + j
                    velocity_cube[i, j, :] = f.trace[trace_idx]

            print(f"  Velocity range: {velocity_cube.min():.0f} - {velocity_cube.max():.0f} m/s")

            return (
                velocity_cube,
                x_coords,
                y_coords,
                f.samples.astype(np.float64)
            )

    except Exception as e:
        print(f"  Standard loading failed: {e}")
        print("  Falling back to scattered point interpolation...")
        return load_segy_velocity_scattered(segy_path)


def convert_velocity_to_zarr(
    velocity_cube: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    t_axis_ms: np.ndarray,
    output_path: Path,
) -> Path:
    """Convert velocity cube to Zarr format with required metadata."""
    print(f"Converting velocity to Zarr: {output_path}")

    store = zarr.storage.LocalStore(str(output_path))
    arr = zarr.create_array(
        store=store,
        shape=velocity_cube.shape,
        dtype=np.float32,
        chunks=(
            min(64, velocity_cube.shape[0]),
            min(64, velocity_cube.shape[1]),
            velocity_cube.shape[2]
        ),
        overwrite=True
    )
    arr[:] = velocity_cube

    # Required attributes for CubeVelocityModel
    arr.attrs['x_axis'] = x_axis.tolist()
    arr.attrs['y_axis'] = y_axis.tolist()
    arr.attrs['t_axis_ms'] = t_axis_ms.tolist()
    arr.attrs['units'] = 'm/s'

    print(f"  Saved velocity cube with shape: {velocity_cube.shape}")

    return output_path


# =============================================================================
# Survey Geometry Detection
# =============================================================================

def get_survey_extent(headers_path: Path) -> dict:
    """Get survey extent from headers parquet file."""
    import polars as pl

    df = pl.read_parquet(headers_path)

    # Calculate midpoints with scalar correction
    scalar = df['scalar_coord'].max()
    if scalar is None or scalar == 0:
        scalar = -100  # Default SEG-Y scalar

    scale = 1.0 / abs(scalar) if scalar < 0 else scalar

    mx = (df['source_x'].cast(pl.Float64) + df['receiver_x'].cast(pl.Float64)) / 2 * scale
    my = (df['source_y'].cast(pl.Float64) + df['receiver_y'].cast(pl.Float64)) / 2 * scale

    return {
        'x_min': mx.min(),
        'x_max': mx.max(),
        'y_min': my.min(),
        'y_max': my.max(),
        'n_traces': len(df),
        'offset_min': df['offset'].min(),
        'offset_max': df['offset'].max(),
    }


# =============================================================================
# Progress Tracking
# =============================================================================

class ProgressTracker:
    """Track and display migration progress."""

    def __init__(self):
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 2.0  # Update every 2 seconds

    def __call__(self, info: ProgressInfo):
        """Progress callback - called after each tile."""
        current_time = time.time()

        # Rate-limit updates
        if current_time - self.last_update < self.update_interval:
            if info.current_tile < info.total_tiles:
                return

        self.last_update = current_time
        elapsed = current_time - self.start_time

        # Calculate progress
        if info.total_tiles > 0:
            progress = info.current_tile / info.total_tiles
        else:
            progress = 0

        # Estimate remaining time
        if progress > 0:
            eta = (elapsed / progress) * (1 - progress)
            eta_str = self._format_time(eta)
        else:
            eta_str = "calculating..."

        # Calculate processing rate
        if info.traces_processed > 0 and elapsed > 0:
            rate = info.traces_processed / elapsed
            rate_str = f"{rate/1e6:.2f}M traces/s"
        else:
            rate_str = "..."

        # Display progress bar
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Print progress line
        print(f"\r[{bar}] {progress*100:5.1f}% | "
              f"Tile {info.current_tile}/{info.total_tiles} | "
              f"{rate_str} | "
              f"ETA: {eta_str}    ", end='', flush=True)

        # Print newline on completion
        if info.current_tile >= info.total_tiles:
            print()

    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


# =============================================================================
# Configuration Building
# =============================================================================

def build_migration_config(
    traces_path: Path,
    headers_path: Path,
    velocity_zarr_path: Path,
    output_dir: Path,
    survey_extent: dict,
) -> MigrationConfig:
    """Build migration configuration for common offset data."""

    # Input configuration
    input_config = InputConfig(
        traces_path=traces_path,
        headers_path=headers_path,
        columns=ColumnMapping(
            source_x="source_x",
            source_y="source_y",
            receiver_x="receiver_x",
            receiver_y="receiver_y",
            offset="offset",
            azimuth="sr_azim",
            trace_index="bin_trace_idx",
            coord_scalar="scalar_coord",
        ),
        apply_coord_scalar=True,
        sample_rate_ms=DT_MS,
        transposed=True,  # Data is stored as (n_samples, n_traces)
    )

    # Velocity configuration (3D cube from SEG-Y)
    velocity_config = VelocityConfig(
        source=VelocitySource.CUBE_3D,
        velocity_path=velocity_zarr_path,
        precompute_to_output_grid=True,
    )

    # Algorithm configuration with time-variant sampling
    aperture_config = ApertureConfig(
        max_dip_degrees=MAX_DIP_DEGREES,
        min_aperture_m=MIN_APERTURE_M,
        max_aperture_m=MAX_APERTURE_M,
        taper_fraction=0.1,
    )

    amplitude_config = AmplitudeConfig(
        geometrical_spreading=False,
        obliquity_factor=False,
    )

    # Time-variant sampling for efficiency
    time_variant_config = TimeVariantConfig(
        enabled=True,
        frequency_table=TIME_VARIANT_TABLE,
        min_downsample_factor=1,
        max_downsample_factor=8,
    )

    anti_aliasing_config = AntiAliasingConfig(
        enabled=False,
        method=AntiAliasingMethod.NONE,
    )

    algorithm_config = AlgorithmConfig(
        interpolation=InterpolationMethod.LINEAR,
        aperture=aperture_config,
        amplitude=amplitude_config,
        time_variant=time_variant_config,
        anti_aliasing=anti_aliasing_config,
    )

    # Output grid configuration - corner-point definition for rotated grid
    # 511 inlines (dx=25m) x 427 crosslines (dy=12.5m)
    output_grid = OutputGridConfig.from_corners(
        corner1=GRID_CORNERS['c1'],  # Origin (IL=1, XL=1)
        corner2=GRID_CORNERS['c2'],  # Inline end (IL=511, XL=1)
        corner3=GRID_CORNERS['c3'],  # Far corner (IL=511, XL=427)
        corner4=GRID_CORNERS['c4'],  # Crossline end (IL=1, XL=427)
        t_min_ms=T_MIN_MS,
        t_max_ms=T_MAX_MS,
        dx=DX,
        dy=DY,
        dt_ms=DT_MS,
    )

    # Output configuration
    output_config = OutputConfig(
        output_dir=output_dir,
        grid=output_grid,
        format=OutputFormat.ZARR,
    )

    # Execution configuration with fixed tile size
    # Use METAL_COMPILED for 3D velocity support (METAL_CPP only supports 1D)
    resource_config = ResourceConfig(
        backend=ComputeBackend.METAL_COMPILED,
        max_memory_gb=32.0,
    )

    tiling_config = TilingConfig(
        auto_tile_size=False,
        tile_nx=TILE_NX,
        tile_ny=TILE_NY,
        ordering='snake',
    )

    checkpoint_config = CheckpointConfig(
        enabled=True,
        interval_tiles=10,
    )

    execution_config = ExecutionConfig(
        resources=resource_config,
        tiling=tiling_config,
        checkpoint=checkpoint_config,
    )

    return MigrationConfig(
        name="PSTM_Common_Offset",
        input=input_config,
        velocity=velocity_config,
        algorithm=algorithm_config,
        output=output_config,
        execution=execution_config,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run PSTM with presorted common offset gathers"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running migration",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available (default: True)",
    )
    parser.add_argument(
        "--offset-bin",
        type=int,
        default=None,
        help="Process only a specific offset bin (0-39)",
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        default=True,
        help="Use combined all_headers.parquet (default: True)",
    )

    args = parser.parse_args()

    # Setup file logging early to capture all debug messages
    log_path = setup_file_logging(OUTPUT_DIR, args.offset_bin)

    print("=" * 70)
    print("PSTM Migration with Presorted Common Offset Gathers")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_path}")
    print()

    # Log startup info
    logging.info("=" * 70)
    logging.info("PSTM Migration Started")
    logging.info(f"Offset bin: {args.offset_bin}")
    logging.info(f"Resume: {args.resume}")
    logging.info(f"Dry run: {args.dry_run}")
    logging.info("=" * 70)

    # Step 1: Load velocity from SEG-Y
    print("[1] Loading Velocity Model")
    print("-" * 40)

    if not VELOCITY_SEGY_PATH.exists():
        print(f"ERROR: Velocity file not found: {VELOCITY_SEGY_PATH}")
        return 1

    try:
        velocity_cube, x_axis, y_axis, t_axis_ms = load_segy_velocity_custom_bytes(
            VELOCITY_SEGY_PATH,
            inline_byte=INLINE_BYTE,
            xline_byte=XLINE_BYTE,
        )
    except Exception as e:
        print(f"ERROR: Failed to load velocity: {e}")
        return 1

    # Convert velocity to Zarr
    velocity_zarr_path = OUTPUT_DIR / "velocity_pstm.zarr"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_velocity_to_zarr(velocity_cube, x_axis, y_axis, t_axis_ms, velocity_zarr_path)
    print()

    # Step 2: Determine input data paths
    print("[2] Input Data Configuration")
    print("-" * 40)

    if args.offset_bin is not None:
        # Process single offset bin
        bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{args.offset_bin:02d}"
        traces_path = bin_dir / "traces.zarr"
        headers_path = bin_dir / "headers.parquet"
        output_suffix = f"_bin_{args.offset_bin:02d}"
    else:
        # Use all data - need to determine which traces/headers to use
        # For now, we'll use bin 0 as reference and the combined headers
        headers_path = COMMON_OFFSET_DIR / "all_headers.parquet"

        # For traces, we need to process each bin or use a combined approach
        # Here we'll use offset_bin_00 as a start - modify as needed
        traces_path = COMMON_OFFSET_DIR / "offset_bin_00" / "traces.zarr"
        output_suffix = "_all"

    if not headers_path.exists():
        print(f"ERROR: Headers file not found: {headers_path}")
        return 1

    print(f"  Headers: {headers_path}")
    print(f"  Traces: {traces_path}")

    # Get survey extent
    survey_extent = get_survey_extent(headers_path)
    print(f"  Total traces: {survey_extent['n_traces']:,}")
    print(f"  X range: {survey_extent['x_min']:.2f} - {survey_extent['x_max']:.2f} m")
    print(f"  Y range: {survey_extent['y_min']:.2f} - {survey_extent['y_max']:.2f} m")
    print(f"  Offset range: {survey_extent['offset_min']:.0f} - {survey_extent['offset_max']:.0f} m")
    print()

    # Step 3: Build configuration
    print("[3] Migration Configuration")
    print("-" * 40)

    output_dir = OUTPUT_DIR / f"migration{output_suffix}"

    config = build_migration_config(
        traces_path=traces_path,
        headers_path=headers_path,
        velocity_zarr_path=velocity_zarr_path,
        output_dir=output_dir,
        survey_extent=survey_extent,
    )

    # Print configuration summary
    grid = config.output.grid
    print(f"  Output grid shape: {grid.nx} x {grid.ny} x {grid.nt}")
    print(f"  Output grid size: {grid.size_gb:.2f} GB")
    print(f"  Bin sizes: dx={DX}m, dy={DY}m, dt={DT_MS}ms")
    print(f"  Tile size: {TILE_NX} x {TILE_NY}")
    print(f"  Max aperture: {MAX_APERTURE_M}m")
    print(f"  Time-variant sampling: ENABLED")
    for t, f in TIME_VARIANT_TABLE:
        print(f"    {t:.0f} ms: {f:.0f} Hz")
    print(f"  Backend: {config.execution.resources.backend.value}")
    print(f"  Output: {output_dir}")
    print()

    if args.dry_run:
        print("[DRY RUN] Configuration complete. Use --no-dry-run to execute.")
        return 0

    # Step 4: Run migration
    print("[4] Starting Migration")
    print("-" * 40)

    progress_tracker = ProgressTracker()

    try:
        success = run_migration(
            config,
            resume=args.resume,
            progress_callback=progress_tracker
        )
    except KeyboardInterrupt:
        print("\nMigration interrupted by user. Progress saved to checkpoint.")
        return 1

    if success:
        print()
        print("=" * 70)
        print("Migration Completed Successfully!")
        print("=" * 70)
        print(f"Output: {output_dir}")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\nMigration failed! Check logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
