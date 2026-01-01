#!/usr/bin/env python3
"""
Compare Metal kernel output through two processing paths:
1. Direct kernel call (my diagnostic approach)
2. Executor-style processing (using the same code as executor.py)

This identifies exactly where the amplitude discrepancy occurs.
"""

import numpy as np
import zarr
import polars as pl
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import OutputGridConfig, TilingConfig, VelocitySource, VelocityConfig
from pstm.pipeline.tile_planner import TilePlanner
from pstm.kernels.metal_compiled import CompiledMetalKernel
from pstm.kernels.base import TraceBlock, OutputTile, VelocitySlice, KernelConfig, create_trace_block
from pstm.data.velocity_model import create_velocity_model, VelocityManager
from pstm.data.zarr_reader import ZarrTraceReader
from pstm.data.parquet_headers import ParquetHeaderManager as HeaderManager
from pstm.data.spatial_index import SpatialIndex

# Paths
DATA_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")
BIN_DIR = DATA_DIR / "offset_bin_01"
VELOCITY_PATH = DATA_DIR / "velocity_pstm.zarr"

# Grid params
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Executor vs Direct Kernel Comparison")
    print("=" * 70)

    # Create grid
    grid = OutputGridConfig.from_corners(
        corner1=GRID_CORNERS['c1'],
        corner2=GRID_CORNERS['c2'],
        corner3=GRID_CORNERS['c3'],
        corner4=GRID_CORNERS['c4'],
        t_min_ms=0.0,
        t_max_ms=2000.0,
        dx=25.0,
        dy=12.5,
        dt_ms=2.0,
    )

    print(f"Grid: {grid.nx} x {grid.ny} x {grid.nt}")

    # Get coordinates
    coords = grid.get_output_coordinates()
    X_grid = coords['X']
    Y_grid = coords['Y']

    # Create tile plan
    tiling_config = TilingConfig(auto_tile_size=False, tile_nx=128, tile_ny=128)
    planner = TilePlanner(grid, tiling_config)
    tile_plan = planner.plan()

    # Find the partial tile (first X, last Y)
    partial_tile = None
    full_tile = None
    for t in tile_plan.tiles:
        if t.x_start == 0 and t.y_start == 384:
            partial_tile = t
        if t.x_start == 0 and t.y_start == 256:
            full_tile = t

    print(f"\nFull tile: y=[{full_tile.y_start}:{full_tile.y_end}] (ny={full_tile.ny})")
    print(f"Partial tile: y=[{partial_tile.y_start}:{partial_tile.y_end}] (ny={partial_tile.ny})")

    # Load velocity model
    print("\nLoading velocity model...")
    velocity_config = VelocityConfig(
        source=VelocitySource.CUBE_3D,
        velocity_path=VELOCITY_PATH,
        precompute_to_output_grid=True,
    )
    vel_model = create_velocity_model(velocity_config)

    # Create 1D axes for velocity manager
    t_axis_ms = np.arange(0, grid.nt * grid.dt_ms, grid.dt_ms)
    x_axis_1d = np.linspace(X_grid.min(), X_grid.max(), grid.nx)
    y_axis_1d = np.linspace(Y_grid.min(), Y_grid.max(), grid.ny)

    vel_manager = VelocityManager(
        vel_model,
        output_x_axis=x_axis_1d,
        output_y_axis=y_axis_1d,
        output_t_axis_ms=t_axis_ms,
        output_x_grid=X_grid,
        output_y_grid=Y_grid,
        precompute=True,
    )
    print(f"Velocity precomputed: {vel_manager.memory_usage_gb:.2f} GB")

    # Open data using executor's approach
    print("\nOpening data...")

    # Load headers and traces
    headers_path = BIN_DIR / "headers.parquet"
    traces_path = BIN_DIR / "traces.zarr"

    df = pl.read_parquet(headers_path)
    z = zarr.open_array(str(traces_path), mode='r')

    print(f"  Traces: {len(df)}, Samples: {z.shape[0]}")
    print(f"  Zarr shape: {z.shape}")

    # Compute midpoint coordinates
    source_x = df['source_x'].to_numpy().astype(np.float64)
    source_y = df['source_y'].to_numpy().astype(np.float64)
    receiver_x = df['receiver_x'].to_numpy().astype(np.float64)
    receiver_y = df['receiver_y'].to_numpy().astype(np.float64)
    scalar = df['scalar_coord'].to_numpy()
    scale = np.where(scalar < 0, 1.0 / np.abs(scalar), scalar).astype(np.float64)

    source_x = source_x * scale
    source_y = source_y * scale
    receiver_x = receiver_x * scale
    receiver_y = receiver_y * scale
    midpoint_x = (source_x + receiver_x) / 2.0
    midpoint_y = (source_y + receiver_y) / 2.0

    sample_rate_ms = 2.0  # Default sample rate

    # Build spatial index
    print("Building spatial index...")
    trace_indices = np.arange(len(df), dtype=np.int64)
    spatial_index = SpatialIndex.build(trace_indices, midpoint_x, midpoint_y)

    # Create kernel config (same as run script)
    kernel_config = KernelConfig(
        max_aperture_m=2000.0,
        min_aperture_m=500.0,
        taper_fraction=0.1,
        max_dip_degrees=65.0,
        apply_spreading=True,
        apply_obliquity=True,
        aa_enabled=False,
    )

    # Create kernel
    kernel = CompiledMetalKernel(use_simd=True)
    kernel.initialize(kernel_config)

    # Process FULL tile
    tile = full_tile
    print(f"\n{'='*60}")
    print(f"PROCESSING FULL TILE: y=[{tile.y_start}:{tile.y_end}]")
    print(f"{'='*60}")

    # Extract tile coordinates (executor-style)
    tile_X = X_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]
    tile_Y = Y_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]

    # Query traces (executor-style)
    aperture = kernel_config.max_aperture_m
    query_indices = spatial_index.query_rectangle(
        tile_X.min() - aperture, tile_X.max() + aperture,
        tile_Y.min() - aperture, tile_Y.max() + aperture
    )
    print(f"  Query returned {len(query_indices)} traces")

    # Load traces (zarr is stored as [n_samples, n_traces], need to transpose)
    trace_data = np.array(z[:, query_indices]).T  # (n_traces, n_samples)
    print(f"  Trace data shape: {trace_data.shape}")

    # Get geometry for selected traces
    src_x_sel = source_x[query_indices]
    src_y_sel = source_y[query_indices]
    rec_x_sel = receiver_x[query_indices]
    rec_y_sel = receiver_y[query_indices]
    print(f"  Geometry: {len(query_indices)} traces")

    # Create TraceBlock (executor-style using create_trace_block)
    traces_full = create_trace_block(
        amplitudes=trace_data,
        source_x=src_x_sel,
        source_y=src_y_sel,
        receiver_x=rec_x_sel,
        receiver_y=rec_y_sel,
        sample_rate_ms=sample_rate_ms,
        start_time_ms=0.0,
    )
    print(f"  TraceBlock: {traces_full.n_traces} traces, {traces_full.n_samples} samples")
    print(f"  Amplitudes shape: {traces_full.amplitudes.shape}")
    print(f"  Amplitudes RMS: {np.sqrt(np.mean(traces_full.amplitudes**2)):.6f}")

    # Create OutputTile (executor-style)
    output_full = OutputTile(
        image=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.float64),
        fold=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.int32),
        x_axis=np.linspace(tile.x_min, tile.x_max, tile.nx),
        y_axis=np.linspace(tile.y_min, tile.y_max, tile.ny),
        t_axis_ms=np.arange(grid.t_min_ms, grid.t_max_ms + grid.dt_ms / 2, grid.dt_ms),
        x_grid=tile_X,
        y_grid=tile_Y,
    )
    print(f"  OutputTile: {output_full.nx} x {output_full.ny} x {output_full.nt}")
    print(f"  y_axis: [{output_full.y_axis[0]:.2f}, {output_full.y_axis[-1]:.2f}]")
    print(f"  y_axis spacing: {output_full.y_axis[1] - output_full.y_axis[0]:.4f}")

    # Get velocity slice (executor-style)
    velocity_full = vel_manager.get_velocity_slice_for_tile(
        tile.x_start, tile.x_end,
        tile.y_start, tile.y_end,
    )
    print(f"  Velocity shape: {velocity_full.vrms.shape}")

    # Run kernel
    t0 = time.time()
    metrics_full = kernel.migrate_tile(traces_full, output_full, velocity_full, kernel_config)
    print(f"  Kernel completed in {time.time()-t0:.3f}s")

    # Calculate stats
    t_range = (300, 500)
    full_rms = np.sqrt(np.mean(output_full.image[:, :, t_range[0]:t_range[1]]**2))
    full_fold_mean = output_full.fold[:, :, t_range[0]:t_range[1]].mean()
    print(f"  Full tile RMS: {full_rms:.8f}, mean fold: {full_fold_mean:.1f}")

    # Save last row stats
    full_last_row_rms = np.sqrt(np.mean(output_full.image[:, -1, t_range[0]:t_range[1]]**2))
    full_last_row_fold = output_full.fold[:, -1, t_range[0]:t_range[1]].mean()

    # Process PARTIAL tile
    tile = partial_tile
    print(f"\n{'='*60}")
    print(f"PROCESSING PARTIAL TILE: y=[{tile.y_start}:{tile.y_end}]")
    print(f"{'='*60}")

    # Extract tile coordinates (executor-style)
    tile_X = X_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]
    tile_Y = Y_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]

    # Query traces (executor-style)
    query_indices = spatial_index.query_rectangle(
        tile_X.min() - aperture, tile_X.max() + aperture,
        tile_Y.min() - aperture, tile_Y.max() + aperture
    )
    print(f"  Query returned {len(query_indices)} traces")

    # Load traces (zarr is stored as [n_samples, n_traces], need to transpose)
    trace_data = np.array(z[:, query_indices]).T  # (n_traces, n_samples)
    print(f"  Trace data shape: {trace_data.shape}")

    # Get geometry for selected traces
    src_x_sel = source_x[query_indices]
    src_y_sel = source_y[query_indices]
    rec_x_sel = receiver_x[query_indices]
    rec_y_sel = receiver_y[query_indices]
    print(f"  Geometry: {len(query_indices)} traces")

    # Create TraceBlock (executor-style using create_trace_block)
    traces_partial = create_trace_block(
        amplitudes=trace_data,
        source_x=src_x_sel,
        source_y=src_y_sel,
        receiver_x=rec_x_sel,
        receiver_y=rec_y_sel,
        sample_rate_ms=sample_rate_ms,
        start_time_ms=0.0,
    )
    print(f"  TraceBlock: {traces_partial.n_traces} traces, {traces_partial.n_samples} samples")
    print(f"  Amplitudes shape: {traces_partial.amplitudes.shape}")
    print(f"  Amplitudes RMS: {np.sqrt(np.mean(traces_partial.amplitudes**2)):.6f}")

    # Create OutputTile (executor-style)
    output_partial = OutputTile(
        image=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.float64),
        fold=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.int32),
        x_axis=np.linspace(tile.x_min, tile.x_max, tile.nx),
        y_axis=np.linspace(tile.y_min, tile.y_max, tile.ny),
        t_axis_ms=np.arange(grid.t_min_ms, grid.t_max_ms + grid.dt_ms / 2, grid.dt_ms),
        x_grid=tile_X,
        y_grid=tile_Y,
    )
    print(f"  OutputTile: {output_partial.nx} x {output_partial.ny} x {output_partial.nt}")
    print(f"  y_axis: [{output_partial.y_axis[0]:.2f}, {output_partial.y_axis[-1]:.2f}]")
    print(f"  y_axis spacing: {output_partial.y_axis[1] - output_partial.y_axis[0]:.4f}")

    # Get velocity slice (executor-style)
    velocity_partial = vel_manager.get_velocity_slice_for_tile(
        tile.x_start, tile.x_end,
        tile.y_start, tile.y_end,
    )
    print(f"  Velocity shape: {velocity_partial.vrms.shape}")

    # Run kernel
    t0 = time.time()
    metrics_partial = kernel.migrate_tile(traces_partial, output_partial, velocity_partial, kernel_config)
    print(f"  Kernel completed in {time.time()-t0:.3f}s")

    # Calculate stats
    partial_rms = np.sqrt(np.mean(output_partial.image[:, :, t_range[0]:t_range[1]]**2))
    partial_fold_mean = output_partial.fold[:, :, t_range[0]:t_range[1]].mean()
    print(f"  Partial tile RMS: {partial_rms:.8f}, mean fold: {partial_fold_mean:.1f}")

    # Save first row stats
    partial_first_row_rms = np.sqrt(np.mean(output_partial.image[:, 0, t_range[0]:t_range[1]]**2))
    partial_first_row_fold = output_partial.fold[:, 0, t_range[0]:t_range[1]].mean()

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"Full tile (last row Y=383):    RMS={full_last_row_rms:.8f}, fold={full_last_row_fold:.1f}")
    print(f"Partial tile (first row Y=384): RMS={partial_first_row_rms:.8f}, fold={partial_first_row_fold:.1f}")
    print(f"Boundary RMS ratio: {partial_first_row_rms/full_last_row_rms:.4f}")
    print(f"Boundary fold ratio: {partial_first_row_fold/full_last_row_fold:.4f}")
    print()
    print(f"Full tile overall:    RMS={full_rms:.8f}, fold={full_fold_mean:.1f}")
    print(f"Partial tile overall: RMS={partial_rms:.8f}, fold={partial_fold_mean:.1f}")
    print(f"Overall RMS ratio: {partial_rms/full_rms:.4f}")

    # Compare with actual migration output
    print(f"\n{'='*60}")
    print("COMPARISON WITH ACTUAL MIGRATION OUTPUT")
    print(f"{'='*60}")

    actual_stack_path = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/migration_bin_01/migrated_stack.zarr")
    actual_fold_path = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/migration_bin_01/fold.zarr")

    if actual_stack_path.exists():
        z_stack = zarr.open_array(str(actual_stack_path), mode='r')
        z_fold = zarr.open_array(str(actual_fold_path), mode='r')

        actual_full_row = np.array(z_stack[:128, 383, t_range[0]:t_range[1]])
        actual_partial_row = np.array(z_stack[:128, 384, t_range[0]:t_range[1]])
        actual_full_fold = np.array(z_fold[:128, 383, t_range[0]:t_range[1]])
        actual_partial_fold = np.array(z_fold[:128, 384, t_range[0]:t_range[1]])

        actual_full_rms = np.sqrt(np.mean(actual_full_row**2))
        actual_partial_rms = np.sqrt(np.mean(actual_partial_row**2))

        print(f"Actual migration (Y=383): RMS={actual_full_rms:.8f}, fold={actual_full_fold.mean():.1f}")
        print(f"Actual migration (Y=384): RMS={actual_partial_rms:.8f}, fold={actual_partial_fold.mean():.1f}")
        print(f"Actual boundary ratio: {actual_partial_rms/actual_full_rms:.4f}")
        print()
        print(f"This diagnostic (Y=383): RMS={full_last_row_rms:.8f}, fold={full_last_row_fold:.1f}")
        print(f"This diagnostic (Y=384): RMS={partial_first_row_rms:.8f}, fold={partial_first_row_fold:.1f}")
        print(f"This diagnostic ratio: {partial_first_row_rms/full_last_row_rms:.4f}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
