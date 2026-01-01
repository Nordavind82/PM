#!/usr/bin/env python3
"""
Run Metal kernel directly on real data for partial vs full tiles.

This isolates whether the bug is in:
1. The kernel itself (if outputs differ for same input region)
2. The data preparation (if inputs differ unexpectedly)
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
from pstm.kernels.base import TraceBlock, OutputTile, VelocitySlice, KernelConfig
from pstm.data.velocity_model import create_velocity_model, VelocityManager

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


def load_traces(trace_indices):
    """Load traces for given indices."""
    headers_path = BIN_DIR / "headers.parquet"
    traces_path = BIN_DIR / "traces.zarr"

    df = pl.read_parquet(headers_path)
    z = zarr.open_array(str(traces_path), mode='r')

    print(f"  Zarr shape: {z.shape}")  # Debug

    # Get data for selected traces
    sub_df = df[trace_indices]

    source_x = sub_df['source_x'].to_numpy().astype(np.float64)
    source_y = sub_df['source_y'].to_numpy().astype(np.float64)
    receiver_x = sub_df['receiver_x'].to_numpy().astype(np.float64)
    receiver_y = sub_df['receiver_y'].to_numpy().astype(np.float64)
    offset = sub_df['offset'].to_numpy().astype(np.float64)

    # Apply scalar
    scalar = sub_df['scalar_coord'].to_numpy()
    scale_factor = np.where(scalar < 0, 1.0 / np.abs(scalar), scalar).astype(np.float64)

    source_x = source_x * scale_factor
    source_y = source_y * scale_factor
    receiver_x = receiver_x * scale_factor
    receiver_y = receiver_y * scale_factor

    midpoint_x = (source_x + receiver_x) / 2.0
    midpoint_y = (source_y + receiver_y) / 2.0

    # Load amplitudes - zarr is (n_samples, n_traces)
    # TraceBlock expects (n_traces, n_samples), so transpose
    amplitudes = np.array(z[:, trace_indices]).T  # (n_selected_traces, n_samples)
    print(f"  Loaded amplitudes: {amplitudes.shape}")

    return TraceBlock(
        amplitudes=amplitudes,
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        midpoint_x=midpoint_x,
        midpoint_y=midpoint_y,
        offset=offset,
        sample_rate_ms=2.0,
        start_time_ms=0.0,
    )


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Real Data Kernel Test")
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

    # Find tiles
    full_tile = None
    partial_tile = None
    for t in tile_plan.tiles:
        if t.x_start == 0 and t.y_start == 256:
            full_tile = t
        if t.x_start == 0 and t.y_start == 384:
            partial_tile = t

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

    # Create 1D axes for velocity manager (even though we have 2D grids)
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

    # Get velocity for each tile
    velocity_full = vel_manager.get_velocity_slice_for_tile(
        full_tile.x_start, full_tile.x_end,
        full_tile.y_start, full_tile.y_end,
    )
    velocity_partial = vel_manager.get_velocity_slice_for_tile(
        partial_tile.x_start, partial_tile.x_end,
        partial_tile.y_start, partial_tile.y_end,
    )

    print(f"\nVelocity full shape: {velocity_full.vrms.shape}")
    print(f"Velocity partial shape: {velocity_partial.vrms.shape}")

    # Load traces that are common to both tiles (within aperture of boundary region)
    print("\nLoading traces...")
    headers = pl.read_parquet(BIN_DIR / "headers.parquet")

    # Compute midpoints
    source_x = headers['source_x'].to_numpy().astype(np.float64)
    source_y = headers['source_y'].to_numpy().astype(np.float64)
    receiver_x = headers['receiver_x'].to_numpy().astype(np.float64)
    receiver_y = headers['receiver_y'].to_numpy().astype(np.float64)
    scalar = headers['scalar_coord'].to_numpy()
    scale = np.where(scalar < 0, 1.0 / np.abs(scalar), scalar).astype(np.float64)

    midpoint_x = (source_x * scale + receiver_x * scale) / 2.0
    midpoint_y = (source_y * scale + receiver_y * scale) / 2.0

    # Get tile coordinate ranges
    full_X = X_grid[full_tile.x_start:full_tile.x_end, full_tile.y_start:full_tile.y_end]
    full_Y = Y_grid[full_tile.x_start:full_tile.x_end, full_tile.y_start:full_tile.y_end]

    partial_X = X_grid[partial_tile.x_start:partial_tile.x_end, partial_tile.y_start:partial_tile.y_end]
    partial_Y = Y_grid[partial_tile.x_start:partial_tile.x_end, partial_tile.y_start:partial_tile.y_end]

    aperture = 2000.0

    # Query traces for full tile
    full_mask = (
        (midpoint_x >= full_X.min() - aperture) &
        (midpoint_x <= full_X.max() + aperture) &
        (midpoint_y >= full_Y.min() - aperture) &
        (midpoint_y <= full_Y.max() + aperture)
    )
    full_indices = np.where(full_mask)[0]

    # Query traces for partial tile
    partial_mask = (
        (midpoint_x >= partial_X.min() - aperture) &
        (midpoint_x <= partial_X.max() + aperture) &
        (midpoint_y >= partial_Y.min() - aperture) &
        (midpoint_y <= partial_Y.max() + aperture)
    )
    partial_indices = np.where(partial_mask)[0]

    print(f"Full tile traces: {len(full_indices)}")
    print(f"Partial tile traces: {len(partial_indices)}")

    # Load traces for each tile separately (they may have different traces)
    print("\nLoading traces for full tile...")
    traces_full = load_traces(full_indices)
    print(f"Full tile: {traces_full.n_traces} traces, {traces_full.n_samples} samples")

    print("\nLoading traces for partial tile...")
    traces_partial = load_traces(partial_indices)
    print(f"Partial tile: {traces_partial.n_traces} traces, {traces_partial.n_samples} samples")

    # Create kernel
    config = KernelConfig(
        max_aperture_m=2000.0,
        min_aperture_m=500.0,
        taper_fraction=0.1,
        max_dip_degrees=65.0,
        apply_spreading=True,
        apply_obliquity=True,
        aa_enabled=False,
    )

    kernel = CompiledMetalKernel(use_simd=True)
    kernel.initialize(config)

    # Create output tiles
    full_output = OutputTile(
        image=np.zeros((full_tile.nx, full_tile.ny, grid.nt), dtype=np.float64),
        fold=np.zeros((full_tile.nx, full_tile.ny, grid.nt), dtype=np.int32),
        x_axis=np.linspace(full_tile.x_min, full_tile.x_max, full_tile.nx),
        y_axis=np.linspace(full_tile.y_min, full_tile.y_max, full_tile.ny),
        t_axis_ms=np.arange(0, grid.nt * grid.dt_ms, grid.dt_ms),
        x_grid=full_X,
        y_grid=full_Y,
    )

    partial_output = OutputTile(
        image=np.zeros((partial_tile.nx, partial_tile.ny, grid.nt), dtype=np.float64),
        fold=np.zeros((partial_tile.nx, partial_tile.ny, grid.nt), dtype=np.int32),
        x_axis=np.linspace(partial_tile.x_min, partial_tile.x_max, partial_tile.nx),
        y_axis=np.linspace(partial_tile.y_min, partial_tile.y_max, partial_tile.ny),
        t_axis_ms=np.arange(0, grid.nt * grid.dt_ms, grid.dt_ms),
        x_grid=partial_X,
        y_grid=partial_Y,
    )

    # Run kernel on full tile
    print("\n--- Running kernel on full tile ---")
    t0 = time.time()
    metrics_full = kernel.migrate_tile(traces_full, full_output, velocity_full, config)
    print(f"  Completed in {time.time()-t0:.3f}s")

    # Run kernel on partial tile
    print("\n--- Running kernel on partial tile ---")
    t0 = time.time()
    metrics_partial = kernel.migrate_tile(traces_partial, partial_output, velocity_partial, config)
    print(f"  Completed in {time.time()-t0:.3f}s")

    # Compare outputs
    print("\n--- Comparison ---")

    # RMS for overlapping Y region
    # Full tile's last 43 Y points should correspond to partial tile's first 43
    # But actually the coordinates are different!
    # Full tile covers Y=256:384, partial tile covers Y=384:427
    # There's no direct overlap - they're adjacent

    # Instead, compare boundary points
    # Last row of full tile (Y=383, which is local y=127)
    # First row of partial tile (Y=384, which is local y=0)

    t_range = (300, 500)

    # Full tile stats
    full_rms = np.sqrt(np.mean(full_output.image[:, :, t_range[0]:t_range[1]]**2))
    full_fold_mean = full_output.fold[:, :, t_range[0]:t_range[1]].mean()

    # Partial tile stats
    partial_rms = np.sqrt(np.mean(partial_output.image[:, :, t_range[0]:t_range[1]]**2))
    partial_fold_mean = partial_output.fold[:, :, t_range[0]:t_range[1]].mean()

    print(f"Full tile RMS: {full_rms:.8f}, mean fold: {full_fold_mean:.1f}")
    print(f"Partial tile RMS: {partial_rms:.8f}, mean fold: {partial_fold_mean:.1f}")
    print(f"RMS ratio (partial/full): {partial_rms/full_rms:.4f}")
    print(f"Fold ratio (partial/full): {partial_fold_mean/full_fold_mean:.4f}")

    # Boundary row comparison
    print("\n--- Boundary Row Comparison ---")
    # Last row of full tile (local y=127, corresponds to global y=383)
    full_last_row = full_output.image[:, 127, :]
    full_last_fold = full_output.fold[:, 127, :]

    # First row of partial tile (local y=0, corresponds to global y=384)
    partial_first_row = partial_output.image[:, 0, :]
    partial_first_fold = partial_output.fold[:, 0, :]

    # RMS of each row
    full_boundary_rms = np.sqrt(np.mean(full_last_row[:, t_range[0]:t_range[1]]**2))
    partial_boundary_rms = np.sqrt(np.mean(partial_first_row[:, t_range[0]:t_range[1]]**2))

    full_boundary_fold = full_last_fold[:, t_range[0]:t_range[1]].mean()
    partial_boundary_fold = partial_first_fold[:, t_range[0]:t_range[1]].mean()

    print(f"Full tile last row (Y=383): RMS={full_boundary_rms:.8f}, fold={full_boundary_fold:.1f}")
    print(f"Partial tile first row (Y=384): RMS={partial_boundary_rms:.8f}, fold={partial_boundary_fold:.1f}")
    print(f"RMS ratio at boundary: {partial_boundary_rms/full_boundary_rms:.4f}")

    # Per-point comparison at boundary
    print("\n--- Per-Point Comparison (boundary) ---")
    for ix in [32, 64, 96]:
        for it in [350, 400, 450]:
            val_full = full_output.image[ix, 127, it]
            val_partial = partial_output.image[ix, 0, it]
            fold_full = full_output.fold[ix, 127, it]
            fold_partial = partial_output.fold[ix, 0, it]
            print(f"  ({ix}, t={it*2}ms): full={val_full:.6f} partial={val_partial:.6f} "
                  f"fold={fold_full}/{fold_partial}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
