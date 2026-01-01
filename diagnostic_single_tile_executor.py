#!/usr/bin/env python3
"""
Run executor-style migration for a SINGLE tile and compare with diagnostic.

This isolates whether the issue is per-tile or in the accumulation.
"""

import numpy as np
import zarr
import polars as pl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import OutputGridConfig, TilingConfig, VelocitySource, VelocityConfig
from pstm.pipeline.tile_planner import TilePlanner
from pstm.kernels.metal_compiled import CompiledMetalKernel
from pstm.kernels.base import OutputTile, KernelConfig, create_trace_block
from pstm.data.velocity_model import create_velocity_model, VelocityManager
from pstm.data.spatial_index import SpatialIndex
from pstm.data.zarr_reader import ZarrTraceReader
from pstm.data.trace_cache import LRUTraceCache
from pstm.data.parquet_headers import ParquetHeaderManager
from pstm.config.models import ColumnMapping

# Configuration
DATA_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")
BIN_NUM = 25
BIN_DIR = DATA_DIR / f"offset_bin_{BIN_NUM:02d}"
VELOCITY_PATH = DATA_DIR / "velocity_pstm.zarr"

GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}


def run_diagnostic_method(tile, grid, X_grid, Y_grid, kernel, kernel_config, vel_manager, spatial_index,
                          source_x, source_y, receiver_x, receiver_y, z_zarr, sample_rate_ms):
    """Run tile processing using diagnostic-style direct loading."""
    # Extract tile coordinates
    tile_X = X_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]
    tile_Y = Y_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]

    # Query traces
    aperture = kernel_config.max_aperture_m
    query_indices = spatial_index.query_rectangle(
        tile_X.min() - aperture, tile_X.max() + aperture,
        tile_Y.min() - aperture, tile_Y.max() + aperture
    )

    # Load traces - DIAGNOSTIC METHOD: direct zarr access
    trace_data = np.array(z_zarr[:, query_indices]).T  # (n_traces, n_samples)

    # Get geometry
    src_x_sel = source_x[query_indices]
    src_y_sel = source_y[query_indices]
    rec_x_sel = receiver_x[query_indices]
    rec_y_sel = receiver_y[query_indices]

    # Create TraceBlock
    traces = create_trace_block(
        amplitudes=trace_data,
        source_x=src_x_sel,
        source_y=src_y_sel,
        receiver_x=rec_x_sel,
        receiver_y=rec_y_sel,
        sample_rate_ms=sample_rate_ms,
        start_time_ms=0.0,
    )

    # Create OutputTile
    output_tile = OutputTile(
        image=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.float64),
        fold=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.int32),
        x_axis=np.linspace(tile.x_min, tile.x_max, tile.nx),
        y_axis=np.linspace(tile.y_min, tile.y_max, tile.ny),
        t_axis_ms=np.arange(grid.t_min_ms, grid.t_max_ms + grid.dt_ms / 2, grid.dt_ms),
        x_grid=tile_X,
        y_grid=tile_Y,
    )

    # Get velocity
    velocity = vel_manager.get_velocity_slice_for_tile(
        tile.x_start, tile.x_end,
        tile.y_start, tile.y_end,
    )

    # Run kernel
    metrics = kernel.migrate_tile(traces, output_tile, velocity, kernel_config)

    return output_tile, query_indices


def run_executor_method(tile, grid, X_grid, Y_grid, kernel, kernel_config, vel_manager, spatial_index,
                        trace_reader, trace_cache, header_manager, sample_rate_ms):
    """Run tile processing using executor-style loading."""
    # Extract tile coordinates
    tile_X = X_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]
    tile_Y = Y_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]

    # Query traces
    aperture = kernel_config.max_aperture_m
    query_indices = spatial_index.query_rectangle(
        tile_X.min() - aperture, tile_X.max() + aperture,
        tile_Y.min() - aperture, tile_Y.max() + aperture
    )

    # Load traces - EXECUTOR METHOD: via cache and reader
    trace_data = trace_cache.get_traces(query_indices, trace_reader)

    # Get geometry - EXECUTOR METHOD: via header manager
    geometry = header_manager.get_geometry_for_indices(query_indices)

    # Create TraceBlock
    traces = create_trace_block(
        amplitudes=trace_data,
        source_x=geometry.source_x,
        source_y=geometry.source_y,
        receiver_x=geometry.receiver_x,
        receiver_y=geometry.receiver_y,
        sample_rate_ms=sample_rate_ms,
        start_time_ms=0.0,
    )

    # Create OutputTile - same as diagnostic
    output_tile = OutputTile(
        image=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.float64),
        fold=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.int32),
        x_axis=np.linspace(tile.x_min, tile.x_max, tile.nx),
        y_axis=np.linspace(tile.y_min, tile.y_max, tile.ny),
        t_axis_ms=np.arange(grid.t_min_ms, grid.t_max_ms + grid.dt_ms / 2, grid.dt_ms),
        x_grid=tile_X,
        y_grid=tile_Y,
    )

    # Get velocity - same as diagnostic
    velocity = vel_manager.get_velocity_slice_for_tile(
        tile.x_start, tile.x_end,
        tile.y_start, tile.y_end,
    )

    # Run kernel
    metrics = kernel.migrate_tile(traces, output_tile, velocity, kernel_config)

    return output_tile, query_indices


def main():
    print("=" * 70)
    print("SINGLE TILE COMPARISON: Diagnostic vs Executor Method")
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

    coords = grid.get_output_coordinates()
    X_grid = coords['X']
    Y_grid = coords['Y']

    # Create tile plan and pick a test tile (tile 8: y=[256:384], full tile)
    tiling_config = TilingConfig(auto_tile_size=False, tile_nx=128, tile_ny=128)
    planner = TilePlanner(grid, tiling_config)
    tile_plan = planner.plan()

    # Find tile 15 (PARTIAL Y tile at y=384:427)
    test_tile = None
    for t in tile_plan.tiles:
        if t.x_start == 0 and t.y_start == 384:
            test_tile = t
            break

    if test_tile is None:
        print("ERROR: Could not find test tile")
        return

    print(f"\nTest tile: {test_tile.tile_id} (PARTIAL Y TILE)")
    print(f"  x=[{test_tile.x_start}:{test_tile.x_end}], y=[{test_tile.y_start}:{test_tile.y_end}]")
    print(f"  nx={test_tile.nx}, ny={test_tile.ny}")
    print(f"  is_partial_y: {test_tile.ny < 128}")

    # Load velocity model
    print("\nLoading velocity model...")
    velocity_config = VelocityConfig(
        source=VelocitySource.CUBE_3D,
        velocity_path=VELOCITY_PATH,
        precompute_to_output_grid=True,
    )
    vel_model = create_velocity_model(velocity_config)

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

    # Load input data for DIAGNOSTIC method
    print("\nLoading input data...")
    headers_path = BIN_DIR / "headers.parquet"
    traces_path = BIN_DIR / "traces.zarr"

    df = pl.read_parquet(headers_path)
    z_zarr = zarr.open_array(str(traces_path), mode='r')

    print(f"  Traces: {len(df)}, Samples: {z_zarr.shape[0]}")

    # Check zarr shape
    is_transposed = z_zarr.shape[0] < z_zarr.shape[1]
    n_traces = z_zarr.shape[1] if is_transposed else z_zarr.shape[0]
    n_samples = z_zarr.shape[0] if is_transposed else z_zarr.shape[1]
    print(f"  Zarr shape: {z_zarr.shape}, transposed={is_transposed}")

    # Compute coordinates for diagnostic
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

    sample_rate_ms = 2.0

    # Build spatial index
    print("Building spatial index...")
    trace_indices = np.arange(len(df), dtype=np.int64)
    spatial_index = SpatialIndex.build(trace_indices, midpoint_x, midpoint_y)

    # Setup executor-style readers
    print("Setting up executor-style readers...")
    trace_reader = ZarrTraceReader(
        traces_path,
        transposed=is_transposed,
        n_traces=n_traces,
        n_samples=n_samples,
        sample_rate_ms=sample_rate_ms,
    )
    trace_reader.open()

    trace_cache = LRUTraceCache(max_size_mb=500.0)

    column_mapping = ColumnMapping(
        source_x="source_x",
        source_y="source_y",
        receiver_x="receiver_x",
        receiver_y="receiver_y",
        offset="offset",
        azimuth="sr_azim",
        trace_index="bin_trace_idx",
        coord_scalar="scalar_coord",
    )
    header_manager = ParquetHeaderManager(
        headers_path,
        column_mapping=column_mapping,
        apply_scalar=True,
    )
    header_manager.open()

    # Create kernel
    kernel_config = KernelConfig(
        max_aperture_m=2000.0,
        min_aperture_m=500.0,
        taper_fraction=0.1,
        max_dip_degrees=65.0,
        apply_spreading=True,
        apply_obliquity=True,
        aa_enabled=False,
    )

    kernel = CompiledMetalKernel(use_simd=True)
    kernel.initialize(kernel_config)

    # Run DIAGNOSTIC method
    print("\n--- Running DIAGNOSTIC method ---")
    diag_output, diag_indices = run_diagnostic_method(
        test_tile, grid, X_grid, Y_grid, kernel, kernel_config, vel_manager, spatial_index,
        source_x, source_y, receiver_x, receiver_y, z_zarr, sample_rate_ms
    )
    print(f"  Traces used: {len(diag_indices)}")
    diag_rms = np.sqrt(np.mean(diag_output.image[:, :, 300:500]**2))
    diag_fold_mean = diag_output.fold[:, :, 300:500].mean()
    print(f"  Output RMS: {diag_rms:.6f}")
    print(f"  Mean fold: {diag_fold_mean:.1f}")

    # Run EXECUTOR method
    print("\n--- Running EXECUTOR method ---")
    exec_output, exec_indices = run_executor_method(
        test_tile, grid, X_grid, Y_grid, kernel, kernel_config, vel_manager, spatial_index,
        trace_reader, trace_cache, header_manager, sample_rate_ms
    )
    print(f"  Traces used: {len(exec_indices)}")
    exec_rms = np.sqrt(np.mean(exec_output.image[:, :, 300:500]**2))
    exec_fold_mean = exec_output.fold[:, :, 300:500].mean()
    print(f"  Output RMS: {exec_rms:.6f}")
    print(f"  Mean fold: {exec_fold_mean:.1f}")

    # Compare
    print("\n--- Comparison ---")
    print(f"RMS ratio (exec/diag): {exec_rms/diag_rms:.4f}")
    print(f"Fold ratio (exec/diag): {exec_fold_mean/diag_fold_mean:.4f}")

    # Check if fold is identical
    fold_match = np.allclose(diag_output.fold, exec_output.fold)
    print(f"Fold arrays match: {fold_match}")

    # Check image difference
    image_diff = np.abs(diag_output.image - exec_output.image)
    print(f"Max image difference: {image_diff.max():.6f}")
    print(f"Mean image difference: {image_diff.mean():.6f}")

    if image_diff.max() < 1e-6:
        print("\nOUTPUTS ARE IDENTICAL!")
    else:
        print("\nOUTPUTS DIFFER!")

        # Check specific sample
        print("\nSample values at (x=64, y=64, t=400):")
        ix, iy, it = 64, 64, 200
        print(f"  Diagnostic: {diag_output.image[ix, iy, it]:.8f}")
        print(f"  Executor:   {exec_output.image[ix, iy, it]:.8f}")
        print(f"  Fold diag:  {diag_output.fold[ix, iy, it]}")
        print(f"  Fold exec:  {exec_output.fold[ix, iy, it]}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
