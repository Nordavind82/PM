#!/usr/bin/env python3
"""
Tile-by-tile migration diagnostic for offset bin 25.

Records parameters for each tile and analyzes output separately to identify
the root cause of partial tile amplitude drop.
"""

import numpy as np
import zarr
import polars as pl
from pathlib import Path
import sys
import time
import json
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import OutputGridConfig, TilingConfig, VelocitySource, VelocityConfig
from pstm.pipeline.tile_planner import TilePlanner
from pstm.kernels.metal_compiled import CompiledMetalKernel
from pstm.kernels.base import OutputTile, KernelConfig, create_trace_block
from pstm.data.velocity_model import create_velocity_model, VelocityManager
from pstm.data.spatial_index import SpatialIndex

# Configuration
DATA_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")
BIN_NUM = 25
BIN_DIR = DATA_DIR / f"offset_bin_{BIN_NUM:02d}"
VELOCITY_PATH = DATA_DIR / "velocity_pstm.zarr"
OUTPUT_DIR = Path(f"/tmp/pstm_fresh_diagnostic_bin{BIN_NUM:02d}")

GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}


@dataclass
class TileParams:
    """Parameters recorded for each tile."""
    tile_id: int
    ix: int
    iy: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    nx: int
    ny: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    dx: float
    dy: float
    n_traces: int
    trace_rms: float
    output_rms: float
    output_fold_mean: float
    kernel_time_s: float
    is_partial_x: bool
    is_partial_y: bool


@dataclass
class TileOutput:
    """Output statistics for a tile."""
    tile_id: int
    rms_by_row: list  # RMS for each Y row in the tile
    fold_by_row: list  # Mean fold for each Y row
    boundary_rms: float  # RMS at boundary row (first or last depending on tile position)
    boundary_fold: float


def main():
    print("=" * 70)
    print(f"TILE-BY-TILE DIAGNOSTIC: Offset Bin {BIN_NUM}")
    print("=" * 70)

    # Check if bin exists
    if not BIN_DIR.exists():
        print(f"ERROR: Bin directory not found: {BIN_DIR}")
        print(f"Available bins:")
        for p in sorted(DATA_DIR.glob("offset_bin_*")):
            print(f"  {p.name}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    print(f"Total tiles: {tile_plan.n_tiles}")

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
    print(f"Velocity precomputed: {vel_manager.memory_usage_gb:.2f} GB")

    # Load input data
    print(f"\nLoading input data from {BIN_DIR}...")
    headers_path = BIN_DIR / "headers.parquet"
    traces_path = BIN_DIR / "traces.zarr"

    df = pl.read_parquet(headers_path)
    z = zarr.open_array(str(traces_path), mode='r')

    print(f"  Traces: {len(df)}, Samples: {z.shape[0]}")

    # Compute coordinates
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

    # Allocate global output arrays
    global_image = np.zeros((grid.nx, grid.ny, grid.nt), dtype=np.float64)
    global_fold = np.zeros((grid.nx, grid.ny, grid.nt), dtype=np.int32)

    # Process each tile
    tile_params_list = []
    tile_outputs_list = []
    t_range = (300, 500)  # Time samples for RMS calculation

    print(f"\n{'='*70}")
    print("PROCESSING TILES")
    print(f"{'='*70}")

    for tile in tile_plan.tiles:
        print(f"\n--- Tile {tile.tile_id}: x=[{tile.x_start}:{tile.x_end}], y=[{tile.y_start}:{tile.y_end}] ---")

        # Extract tile coordinates
        tile_X = X_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]
        tile_Y = Y_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]

        # Query traces
        aperture = kernel_config.max_aperture_m
        query_indices = spatial_index.query_rectangle(
            tile_X.min() - aperture, tile_X.max() + aperture,
            tile_Y.min() - aperture, tile_Y.max() + aperture
        )

        if len(query_indices) == 0:
            print(f"  No traces found, skipping")
            continue

        print(f"  Traces: {len(query_indices)}")

        # Load traces
        trace_data = np.array(z[:, query_indices]).T  # (n_traces, n_samples)

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

        trace_rms = np.sqrt(np.mean(traces.amplitudes**2))

        # Create OutputTile
        x_axis = np.linspace(tile.x_min, tile.x_max, tile.nx)
        y_axis = np.linspace(tile.y_min, tile.y_max, tile.ny)
        t_axis = np.arange(grid.t_min_ms, grid.t_max_ms + grid.dt_ms / 2, grid.dt_ms)

        dx = x_axis[1] - x_axis[0] if tile.nx > 1 else 25.0
        dy = y_axis[1] - y_axis[0] if tile.ny > 1 else 12.5

        output_tile = OutputTile(
            image=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.float64),
            fold=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.int32),
            x_axis=x_axis,
            y_axis=y_axis,
            t_axis_ms=t_axis,
            x_grid=tile_X,
            y_grid=tile_Y,
        )

        # Get velocity
        velocity = vel_manager.get_velocity_slice_for_tile(
            tile.x_start, tile.x_end,
            tile.y_start, tile.y_end,
        )

        # Run kernel
        t0 = time.time()
        metrics = kernel.migrate_tile(traces, output_tile, velocity, kernel_config)
        kernel_time = time.time() - t0

        # Calculate output statistics
        output_rms = np.sqrt(np.mean(output_tile.image[:, :, t_range[0]:t_range[1]]**2))
        output_fold_mean = output_tile.fold[:, :, t_range[0]:t_range[1]].mean()

        # Check if partial tile
        is_partial_x = tile.nx < 128
        is_partial_y = tile.ny < 128

        print(f"  nx={tile.nx}, ny={tile.ny}, dx={dx:.4f}, dy={dy:.4f}")
        print(f"  Trace RMS: {trace_rms:.6f}")
        print(f"  Output RMS: {output_rms:.8f}, Mean fold: {output_fold_mean:.1f}")
        print(f"  Kernel time: {kernel_time:.3f}s")
        print(f"  Partial: X={is_partial_x}, Y={is_partial_y}")

        # Record parameters
        params = TileParams(
            tile_id=tile.tile_id,
            ix=tile.ix,
            iy=tile.iy,
            x_start=tile.x_start,
            x_end=tile.x_end,
            y_start=tile.y_start,
            y_end=tile.y_end,
            nx=tile.nx,
            ny=tile.ny,
            x_min=tile.x_min,
            x_max=tile.x_max,
            y_min=tile.y_min,
            y_max=tile.y_max,
            dx=dx,
            dy=dy,
            n_traces=len(query_indices),
            trace_rms=trace_rms,
            output_rms=output_rms,
            output_fold_mean=output_fold_mean,
            kernel_time_s=kernel_time,
            is_partial_x=is_partial_x,
            is_partial_y=is_partial_y,
        )
        tile_params_list.append(params)

        # Calculate per-row statistics
        rms_by_row = []
        fold_by_row = []
        for iy_local in range(tile.ny):
            row_rms = np.sqrt(np.mean(output_tile.image[:, iy_local, t_range[0]:t_range[1]]**2))
            row_fold = output_tile.fold[:, iy_local, t_range[0]:t_range[1]].mean()
            rms_by_row.append(float(row_rms))
            fold_by_row.append(float(row_fold))

        # Boundary stats
        if tile.iy == 0:
            # First Y tile row - check last row
            boundary_rms = rms_by_row[-1]
            boundary_fold = fold_by_row[-1]
        else:
            # Other tiles - check first row
            boundary_rms = rms_by_row[0]
            boundary_fold = fold_by_row[0]

        tile_output = TileOutput(
            tile_id=tile.tile_id,
            rms_by_row=rms_by_row,
            fold_by_row=fold_by_row,
            boundary_rms=boundary_rms,
            boundary_fold=boundary_fold,
        )
        tile_outputs_list.append(tile_output)

        # Accumulate to global arrays
        global_image[tile.x_start:tile.x_end, tile.y_start:tile.y_end, :] += output_tile.image
        global_fold[tile.x_start:tile.x_end, tile.y_start:tile.y_end, :] += output_tile.fold

    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    # Save tile parameters as JSON (convert numpy types to Python types)
    params_file = OUTPUT_DIR / "tile_params.json"

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(params_file, 'w') as f:
        json.dump([convert_types(asdict(p)) for p in tile_params_list], f, indent=2)
    print(f"Saved tile parameters to {params_file}")

    # Save tile outputs as JSON
    outputs_file = OUTPUT_DIR / "tile_outputs.json"
    with open(outputs_file, 'w') as f:
        json.dump([convert_types(asdict(o)) for o in tile_outputs_list], f, indent=2)
    print(f"Saved tile outputs to {outputs_file}")

    # Save global arrays
    stack_path = OUTPUT_DIR / "migrated_stack.zarr"
    fold_path = OUTPUT_DIR / "fold.zarr"

    z_stack = zarr.open(str(stack_path), mode='w', shape=global_image.shape, dtype=np.float32)
    z_stack[:] = global_image.astype(np.float32)

    z_fold = zarr.open(str(fold_path), mode='w', shape=global_fold.shape, dtype=np.int32)
    z_fold[:] = global_fold

    print(f"Saved stack to {stack_path}")
    print(f"Saved fold to {fold_path}")

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    # Group tiles by type
    full_tiles = [p for p in tile_params_list if not p.is_partial_x and not p.is_partial_y]
    partial_y_tiles = [p for p in tile_params_list if not p.is_partial_x and p.is_partial_y]
    partial_x_tiles = [p for p in tile_params_list if p.is_partial_x and not p.is_partial_y]
    partial_xy_tiles = [p for p in tile_params_list if p.is_partial_x and p.is_partial_y]

    print(f"\nTile counts:")
    print(f"  Full tiles (128x128): {len(full_tiles)}")
    print(f"  Partial Y tiles (128x{partial_y_tiles[0].ny if partial_y_tiles else '?'}): {len(partial_y_tiles)}")
    print(f"  Partial X tiles ({partial_x_tiles[0].nx if partial_x_tiles else '?'}x128): {len(partial_x_tiles)}")
    print(f"  Partial XY tiles: {len(partial_xy_tiles)}")

    # Compare average RMS
    if full_tiles and partial_y_tiles:
        avg_full_rms = np.mean([p.output_rms for p in full_tiles])
        avg_partial_y_rms = np.mean([p.output_rms for p in partial_y_tiles])
        print(f"\nAverage output RMS:")
        print(f"  Full tiles: {avg_full_rms:.8f}")
        print(f"  Partial Y tiles: {avg_partial_y_rms:.8f}")
        print(f"  Ratio (partial_y/full): {avg_partial_y_rms/avg_full_rms:.4f}")

        avg_full_fold = np.mean([p.output_fold_mean for p in full_tiles])
        avg_partial_y_fold = np.mean([p.output_fold_mean for p in partial_y_tiles])
        print(f"\nAverage fold:")
        print(f"  Full tiles: {avg_full_fold:.1f}")
        print(f"  Partial Y tiles: {avg_partial_y_fold:.1f}")
        print(f"  Ratio: {avg_partial_y_fold/avg_full_fold:.4f}")

        # RMS per trace
        avg_full_rms_per_trace = np.mean([p.output_rms / p.n_traces for p in full_tiles])
        avg_partial_y_rms_per_trace = np.mean([p.output_rms / p.n_traces for p in partial_y_tiles])
        print(f"\nRMS per trace:")
        print(f"  Full tiles: {avg_full_rms_per_trace:.12f}")
        print(f"  Partial Y tiles: {avg_partial_y_rms_per_trace:.12f}")
        print(f"  Ratio: {avg_partial_y_rms_per_trace/avg_full_rms_per_trace:.4f}")

    # Boundary analysis
    print(f"\n{'='*70}")
    print("BOUNDARY ANALYSIS")
    print(f"{'='*70}")

    # Find adjacent full and partial Y tiles at same X position
    for ix in range(4):  # 4 X tile columns
        full_at_ix = [p for p in full_tiles if p.ix == ix and p.iy == 2]  # iy=2 is y=[256:384]
        partial_at_ix = [p for p in partial_y_tiles if p.ix == ix]  # iy=3 is y=[384:427]

        if full_at_ix and partial_at_ix:
            full_p = full_at_ix[0]
            partial_p = partial_at_ix[0]

            # Get outputs for these tiles
            full_out = next(o for o in tile_outputs_list if o.tile_id == full_p.tile_id)
            partial_out = next(o for o in tile_outputs_list if o.tile_id == partial_p.tile_id)

            print(f"\nBoundary at ix={ix}:")
            print(f"  Full tile {full_p.tile_id} last row (Y=383): RMS={full_out.rms_by_row[-1]:.8f}, fold={full_out.fold_by_row[-1]:.1f}")
            print(f"  Partial tile {partial_p.tile_id} first row (Y=384): RMS={partial_out.rms_by_row[0]:.8f}, fold={partial_out.fold_by_row[0]:.1f}")
            if full_out.rms_by_row[-1] > 0:
                print(f"  Boundary ratio: {partial_out.rms_by_row[0]/full_out.rms_by_row[-1]:.4f}")

    # Check global output at boundary
    print(f"\n{'='*70}")
    print("GLOBAL OUTPUT CHECK")
    print(f"{'='*70}")

    # Check Y=383 vs Y=384 in global output
    y383_rms = np.sqrt(np.mean(global_image[:128, 383, t_range[0]:t_range[1]]**2))
    y384_rms = np.sqrt(np.mean(global_image[:128, 384, t_range[0]:t_range[1]]**2))
    y383_fold = global_fold[:128, 383, t_range[0]:t_range[1]].mean()
    y384_fold = global_fold[:128, 384, t_range[0]:t_range[1]].mean()

    print(f"Global output (first X column, x=[0:128]):")
    print(f"  Y=383: RMS={y383_rms:.8f}, fold={y383_fold:.1f}")
    print(f"  Y=384: RMS={y384_rms:.8f}, fold={y384_fold:.1f}")
    print(f"  Boundary ratio: {y384_rms/y383_rms:.4f}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
