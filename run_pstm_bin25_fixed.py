#!/usr/bin/env python3
"""
Run PSTM Migration for Bin 25 with FIXED amplitude settings.

This script enables geometrical_spreading and obliquity_factor
which are critical for proper Kirchhoff migration.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

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
from pstm.pipeline.executor import run_migration

# =============================================================================
# Configuration
# =============================================================================

BIN_NUM = 25
COMMON_OFFSET_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")
VELOCITY_PATH = COMMON_OFFSET_DIR / "velocity_pstm.zarr"
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/migration_bin_25_fixed")

# Grid parameters
DX = 25.0
DY = 12.5
DT_MS = 2.0
T_MIN_MS = 0.0
T_MAX_MS = 2000.0

# Grid corners (rotated grid)
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}

# Algorithm parameters
MAX_APERTURE_M = 2000.0
MIN_APERTURE_M = 500.0
MAX_DIP_DEGREES = 65.0

# Tile size
TILE_NX = 128
TILE_NY = 128


def main():
    print("=" * 70)
    print("PSTM MIGRATION - BIN 25 WITH FIXED AMPLITUDE SETTINGS")
    print("=" * 70)

    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{BIN_NUM:02d}"
    traces_path = bin_dir / "traces.zarr"
    headers_path = bin_dir / "headers.parquet"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Input config
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
        transposed=True,
    )

    # Velocity config
    velocity_config = VelocityConfig(
        source=VelocitySource.CUBE_3D,
        velocity_path=VELOCITY_PATH,
        precompute_to_output_grid=True,
    )

    # Aperture config
    aperture_config = ApertureConfig(
        max_dip_degrees=MAX_DIP_DEGREES,
        min_aperture_m=MIN_APERTURE_M,
        max_aperture_m=MAX_APERTURE_M,
        taper_fraction=0.1,
    )

    # FIXED: Enable amplitude corrections
    amplitude_config = AmplitudeConfig(
        geometrical_spreading=True,   # ENABLED - critical for proper weighting
        obliquity_factor=True,        # ENABLED - critical for proper weighting
    )

    print("\n*** AMPLITUDE CORRECTIONS ENABLED ***")
    print(f"  geometrical_spreading = True (1/v*t weighting)")
    print(f"  obliquity_factor = True (t0/t weighting)")

    # Other configs
    time_variant_config = TimeVariantConfig(enabled=False)
    anti_aliasing_config = AntiAliasingConfig(enabled=False, method=AntiAliasingMethod.NONE)

    algorithm_config = AlgorithmConfig(
        interpolation=InterpolationMethod.LINEAR,
        aperture=aperture_config,
        amplitude=amplitude_config,
        time_variant=time_variant_config,
        anti_aliasing=anti_aliasing_config,
    )

    # Output grid
    output_grid = OutputGridConfig.from_corners(
        corner1=GRID_CORNERS['c1'],
        corner2=GRID_CORNERS['c2'],
        corner3=GRID_CORNERS['c3'],
        corner4=GRID_CORNERS['c4'],
        t_min_ms=T_MIN_MS,
        t_max_ms=T_MAX_MS,
        dx=DX,
        dy=DY,
        dt_ms=DT_MS,
    )

    output_config = OutputConfig(
        output_dir=OUTPUT_DIR,
        grid=output_grid,
        format=OutputFormat.ZARR,
    )

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

    checkpoint_config = CheckpointConfig(enabled=False)

    execution_config = ExecutionConfig(
        resources=resource_config,
        tiling=tiling_config,
        checkpoint=checkpoint_config,
    )

    # Create migration config
    config = MigrationConfig(
        name=f"pstm_bin_{BIN_NUM:02d}_fixed",
        input=input_config,
        velocity=velocity_config,
        algorithm=algorithm_config,
        output=output_config,
        execution=execution_config,
    )

    # Run migration
    print(f"\nStarting migration for bin {BIN_NUM}...")
    print(f"Output: {OUTPUT_DIR}")

    start_time = time.time()
    success = run_migration(config)
    elapsed = time.time() - start_time

    if success:
        print(f"\nMigration completed in {elapsed:.1f} seconds")
    else:
        print(f"\nMigration FAILED after {elapsed:.1f} seconds")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
