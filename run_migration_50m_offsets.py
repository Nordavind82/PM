#!/usr/bin/env python3
"""
Run PSTM migration with 50m offset bin output.

This script runs the migration with CIG (Common Image Gather) output
enabled, producing separate images for each 50m offset bin.
"""

import sys
from pathlib import Path

# Add pstm to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from pstm.config.models import (
    MigrationConfig,
    InputConfig,
    ColumnMapping,
    VelocityConfig,
    AlgorithmConfig,
    OutputConfig,
    ExecutionConfig,
    OutputGridConfig,
    OutputProductsConfig,
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


def build_config() -> MigrationConfig:
    """Build migration config for 50m offset bins."""

    # Column mapping for this dataset
    column_mapping = ColumnMapping(
        source_x="source_x",
        source_y="source_y",
        receiver_x="receiver_x",
        receiver_y="receiver_y",
        offset="offset",
        azimuth="sr_azim",
        trace_index="trace_index",
        coord_scalar="scalar_coord",  # Required for apply_coord_scalar=True
    )

    # Input configuration - using AGC 500ms + Bandpass 8-80Hz processed data
    input_config = InputConfig(
        traces_path="/Users/olegadamovich/SeismicData/processing/processed_agc500_bp8_80/traces.zarr",
        headers_path="/Users/olegadamovich/SeismicData/processing/processed_agc500_bp8_80/headers.parquet",
        columns=column_mapping,
        apply_coord_scalar=True,
        sample_rate_ms=2.0,
        transposed=True,  # Data stored as (samples, traces) not (traces, samples)
    )

    # Velocity configuration (constant 2500 m/s)
    velocity_config = VelocityConfig(
        source=VelocitySource.CONSTANT,
        constant_velocity=2500.0,
    )

    # Algorithm configuration
    # Reduced aperture to 1000m to prevent OOM with dense data (was 5000m, then 2500m)
    aperture_config = ApertureConfig(
        max_dip_degrees=65.0,
        min_aperture_m=300.0,
        max_aperture_m=1000.0,  # Reduced from 2500m - dense tiles were loading 6M traces
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

    # Output grid configuration - from original config
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

    # CIG offset bins - 50m bins from 0 to 2000m
    # Create bin edges: [0, 50, 100, 150, ..., 2000]
    offset_bins = list(np.arange(0, 2050, 50).astype(float))

    # Output products with CIG enabled
    products_config = OutputProductsConfig(
        stacked_image=True,
        fold_volume=True,
        common_image_gathers=True,  # Enable CIG output
        cig_offset_bins=offset_bins,  # 50m offset bins
    )

    # Output configuration
    output_config = OutputConfig(
        output_dir="/Users/olegadamovich/SeismicData/PSTM_offset_bins",
        grid=output_grid,
        products=products_config,
        format=OutputFormat.ZARR,
    )

    # Execution configuration with nested configs
    resource_config = ResourceConfig(
        backend=ComputeBackend.METAL_CPP,
        max_memory_gb=24.0,  # Reduced from 32 to leave headroom
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
    print("Building PSTM configuration for 50m offset bins...")
    config = build_config()

    # Print summary
    n_bins = len(config.output.products.cig_offset_bins) - 1 if config.output.products.cig_offset_bins else 0
    offset_min = config.output.products.cig_offset_bins[0] if config.output.products.cig_offset_bins else 0
    offset_max = config.output.products.cig_offset_bins[-1] if config.output.products.cig_offset_bins else 0

    print("\nMigration Configuration:")
    print(f"  Input: {config.input.traces_path}")
    print(f"  Output: {config.output.output_dir}")
    print(f"  CIG enabled: {config.output.products.common_image_gathers}")
    print(f"  Offset bins: {n_bins} (edges: {len(config.output.products.cig_offset_bins)})")
    print(f"  Offset range: {offset_min}-{offset_max}m")
    print(f"  Backend: {config.execution.resources.backend}")
    print()

    # Run migration
    print("Starting migration...")
    print("=" * 60)
    success = run_migration(config, resume=True)

    if success:
        print("\n" + "=" * 60)
        print("Migration completed successfully!")
        print(f"Output saved to: {config.output.output_dir}")
    else:
        print("\nMigration failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
