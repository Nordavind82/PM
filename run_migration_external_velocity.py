#!/usr/bin/env python3
"""
Run PSTM migration with external velocity model.

This script demonstrates loading velocity from SEG-Y, converting to Zarr,
and running migration with the external velocity cube.

Usage:
    python run_migration_external_velocity.py --velocity <segy_file> [options]

Example:
    python run_migration_external_velocity.py --velocity v2000_k500.sgy
"""

import sys
from pathlib import Path
import argparse

# Add pstm to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import segyio
import zarr
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
)
from pstm.pipeline.executor import run_migration


def load_segy_velocity(segy_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load velocity model from SEG-Y file.

    Returns:
        velocity_cube: 3D array (nx, ny, nt)
        x_axis: X coordinates
        y_axis: Y coordinates
        t_axis_ms: Time axis in milliseconds
    """
    with segyio.open(str(segy_path), 'r') as f:
        n_inlines = len(f.ilines)
        n_xlines = len(f.xlines)
        n_samples = len(f.samples)

        # Coordinate scaling
        h0 = f.header[0]
        scalar = h0[segyio.TraceField.SourceGroupScalar]
        scale = 1.0 / abs(scalar) if scalar < 0 else (scalar if scalar > 0 else 1.0)

        # Extract coordinates
        x_coords = []
        y_coords = []

        for i in range(n_inlines):
            h = f.header[i * n_xlines]
            x_coords.append(h[segyio.TraceField.CDP_X] * scale)

        for j in range(n_xlines):
            h = f.header[j]
            y_coords.append(h[segyio.TraceField.CDP_Y] * scale)

        # Load traces
        velocity_cube = np.zeros((n_inlines, n_xlines, n_samples), dtype=np.float32)
        for i in range(n_inlines):
            for j in range(n_xlines):
                trace_idx = i * n_xlines + j
                velocity_cube[i, j, :] = f.trace[trace_idx]

        return (
            velocity_cube,
            np.array(x_coords),
            np.array(y_coords),
            f.samples.astype(np.float64)
        )


def convert_to_zarr(velocity_cube: np.ndarray, x_axis: np.ndarray,
                     y_axis: np.ndarray, t_axis_ms: np.ndarray,
                     output_path: Path) -> Path:
    """Convert velocity cube to Zarr format with required metadata."""

    store = zarr.storage.LocalStore(str(output_path))
    arr = zarr.create_array(
        store=store,
        shape=velocity_cube.shape,
        dtype=np.float32,
        chunks=(min(64, velocity_cube.shape[0]),
                min(64, velocity_cube.shape[1]),
                velocity_cube.shape[2]),
        overwrite=True
    )
    arr[:] = velocity_cube

    # Required attributes for CubeVelocityModel
    arr.attrs['x_axis'] = x_axis.tolist()
    arr.attrs['y_axis'] = y_axis.tolist()
    arr.attrs['t_axis_ms'] = t_axis_ms.tolist()
    arr.attrs['units'] = 'm/s'

    return output_path


def build_config(velocity_zarr_path: Path, output_suffix: str = "") -> MigrationConfig:
    """Build migration config with external velocity model."""

    # Input configuration
    input_config = InputConfig(
        traces_path="/Users/olegadamovich/SeismicData/processing/processed_scd_xsd_data_new_20251221_225219_20251221_232357/output/traces.zarr",
        headers_path="/Users/olegadamovich/SeismicData/processing/processed_scd_xsd_data_new_20251221_225219_20251221_232357/output/headers.parquet",
        apply_coord_scalar=True,
        sample_rate_ms=2.0,
    )

    # Velocity configuration - EXTERNAL CUBE
    velocity_config = VelocityConfig(
        source=VelocitySource.CUBE_3D,
        velocity_path=velocity_zarr_path,
        precompute_to_output_grid=True,
    )

    # Algorithm configuration
    aperture_config = ApertureConfig(
        max_dip_degrees=65.0,
        min_aperture_m=500.0,
        max_aperture_m=5000.0,
        taper_fraction=0.1,
    )

    amplitude_config = AmplitudeConfig(
        geometrical_spreading=False,
        obliquity_factor=False,
    )

    algorithm_config = AlgorithmConfig(
        interpolation=InterpolationMethod.LINEAR,
        aperture=aperture_config,
        amplitude=amplitude_config,
    )

    # Output grid configuration
    output_grid = OutputGridConfig(
        x_min=617443.56,
        x_max=632512.02,
        dx=12.5,
        y_min=5106192.26,
        y_max=5120569.31,
        dy=25.0,
        t_min_ms=0.0,
        t_max_ms=2000.0,
        dt_ms=2.0,
    )

    # Output configuration
    output_dir = f"/Users/olegadamovich/SeismicData/PSTM_external_vel{output_suffix}"
    output_config = OutputConfig(
        output_dir=output_dir,
        grid=output_grid,
        format=OutputFormat.ZARR,
    )

    # Execution configuration
    resource_config = ResourceConfig(
        backend=ComputeBackend.METAL_CPP,
        max_memory_gb=32.0,
    )

    tiling_config = TilingConfig(
        auto_tile_size=True,
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
        input=input_config,
        velocity=velocity_config,
        algorithm=algorithm_config,
        output=output_config,
        execution=execution_config,
    )


def main():
    parser = argparse.ArgumentParser(description="Run PSTM with external velocity model")
    parser.add_argument(
        "--velocity", "-v",
        help="Velocity SEG-Y file (in velocity_models/segy directory) or full path",
        default="v2000_k500.sgy"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running migration"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available"
    )

    args = parser.parse_args()

    # Find velocity file
    vel_dir = Path("/Users/olegadamovich/SeismicData/velocity_models/segy")
    if Path(args.velocity).exists():
        segy_path = Path(args.velocity)
    else:
        segy_path = vel_dir / args.velocity
        if not segy_path.exists():
            print(f"Error: Velocity file not found: {segy_path}")
            print(f"\nAvailable velocity models:")
            for f in sorted(vel_dir.glob("*.sgy")):
                print(f"  {f.name}")
            return 1

    print("=" * 60)
    print("PSTM Migration with External Velocity Model")
    print("=" * 60)

    # Step 1: Load SEG-Y velocity
    print(f"\n[1] Loading velocity from: {segy_path.name}")
    velocity_cube, x_axis, y_axis, t_axis_ms = load_segy_velocity(segy_path)
    print(f"    Shape: {velocity_cube.shape}")
    print(f"    Velocity range: {velocity_cube.min():.0f} - {velocity_cube.max():.0f} m/s")

    # Step 2: Convert to Zarr
    zarr_path = Path("/Users/olegadamovich/SeismicData/velocity_models") / f"{segy_path.stem}_pstm.zarr"
    print(f"\n[2] Converting to Zarr: {zarr_path.name}")
    convert_to_zarr(velocity_cube, x_axis, y_axis, t_axis_ms, zarr_path)

    # Step 3: Build config
    output_suffix = f"_{segy_path.stem}"
    print(f"\n[3] Building migration configuration")
    config = build_config(zarr_path, output_suffix)

    print(f"\n    Velocity source: {config.velocity.source.value}")
    print(f"    Velocity path: {config.velocity.velocity_path}")
    print(f"    Output dir: {config.output.output_dir}")
    print(f"    Backend: {config.execution.resources.backend.value}")

    if args.dry_run:
        print("\n[Dry run] Configuration complete. Use --no-dry-run to execute.")
        return 0

    # Step 4: Run migration
    print(f"\n[4] Starting migration...")
    print("=" * 60)

    success = run_migration(config, resume=args.resume)

    if success:
        print("\n" + "=" * 60)
        print("Migration completed successfully!")
        print(f"Output: {config.output.output_dir}")
    else:
        print("\nMigration failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
