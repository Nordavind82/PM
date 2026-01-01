#!/usr/bin/env python3
"""
Compare trace selection between full and partial tiles.

This checks if the spatial query returns different traces for the partial tile.
"""

import numpy as np
import polars as pl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import OutputGridConfig, TilingConfig
from pstm.pipeline.tile_planner import TilePlanner
from pstm.data.spatial_index import SpatialIndex

# Paths
DATA_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")
BIN_DIR = DATA_DIR / "offset_bin_01"

# Grid params
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}

APERTURE_RADIUS = 2000.0  # Max aperture in meters


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Trace Selection Analysis")
    print("=" * 70)

    # Load headers
    headers_path = BIN_DIR / "headers.parquet"
    print(f"Loading headers from {headers_path}...")
    df = pl.read_parquet(headers_path)
    print(f"Total traces: {len(df)}")

    # Compute midpoint coordinates from source and receiver
    source_x = df['source_x'].to_numpy().astype(np.float64)
    source_y = df['source_y'].to_numpy().astype(np.float64)
    receiver_x = df['receiver_x'].to_numpy().astype(np.float64)
    receiver_y = df['receiver_y'].to_numpy().astype(np.float64)

    # Apply coordinate scalar (typically -100 = divide by 100)
    scalar = df['scalar_coord'].to_numpy()
    scale_factor = np.where(scalar < 0, 1.0 / np.abs(scalar), scalar).astype(np.float64)

    source_x = source_x * scale_factor
    source_y = source_y * scale_factor
    receiver_x = receiver_x * scale_factor
    receiver_y = receiver_y * scale_factor

    midpoint_x = (source_x + receiver_x) / 2.0
    midpoint_y = (source_y + receiver_y) / 2.0

    print(f"Midpoint X range: [{midpoint_x.min():.1f}, {midpoint_x.max():.1f}]")
    print(f"Midpoint Y range: [{midpoint_y.min():.1f}, {midpoint_y.max():.1f}]")

    # Create spatial index
    print("\nBuilding spatial index...")
    trace_indices = np.arange(len(df), dtype=np.int64)
    index = SpatialIndex.build(trace_indices, midpoint_x, midpoint_y)

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

    # Get coordinate grids
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

    print(f"\nFull tile: x=[{full_tile.x_start}:{full_tile.x_end}], y=[{full_tile.y_start}:{full_tile.y_end}]")
    print(f"Partial tile: x=[{partial_tile.x_start}:{partial_tile.x_end}], y=[{partial_tile.y_start}:{partial_tile.y_end}]")

    # Query traces for each tile
    print("\n--- Querying traces for each tile ---")

    # For full tile
    full_x_coords = X_grid[full_tile.x_start:full_tile.x_end, full_tile.y_start:full_tile.y_end]
    full_y_coords = Y_grid[full_tile.x_start:full_tile.x_end, full_tile.y_start:full_tile.y_end]

    full_x_min = full_x_coords.min() - APERTURE_RADIUS
    full_x_max = full_x_coords.max() + APERTURE_RADIUS
    full_y_min = full_y_coords.min() - APERTURE_RADIUS
    full_y_max = full_y_coords.max() + APERTURE_RADIUS

    full_trace_indices = index.query_rectangle(full_x_min, full_x_max, full_y_min, full_y_max)

    print(f"Full tile query bounds:")
    print(f"  X: [{full_x_min:.1f}, {full_x_max:.1f}]")
    print(f"  Y: [{full_y_min:.1f}, {full_y_max:.1f}]")
    print(f"  Traces found: {len(full_trace_indices)}")

    # For partial tile
    partial_x_coords = X_grid[partial_tile.x_start:partial_tile.x_end, partial_tile.y_start:partial_tile.y_end]
    partial_y_coords = Y_grid[partial_tile.x_start:partial_tile.x_end, partial_tile.y_start:partial_tile.y_end]

    partial_x_min = partial_x_coords.min() - APERTURE_RADIUS
    partial_x_max = partial_x_coords.max() + APERTURE_RADIUS
    partial_y_min = partial_y_coords.min() - APERTURE_RADIUS
    partial_y_max = partial_y_coords.max() + APERTURE_RADIUS

    partial_trace_indices = index.query_rectangle(partial_x_min, partial_x_max, partial_y_min, partial_y_max)

    print(f"\nPartial tile query bounds:")
    print(f"  X: [{partial_x_min:.1f}, {partial_x_max:.1f}]")
    print(f"  Y: [{partial_y_min:.1f}, {partial_y_max:.1f}]")
    print(f"  Traces found: {len(partial_trace_indices)}")

    # Check overlap
    full_set = set(full_trace_indices)
    partial_set = set(partial_trace_indices)

    overlap = len(full_set & partial_set)
    only_full = len(full_set - partial_set)
    only_partial = len(partial_set - full_set)

    print(f"\n--- Trace Overlap Analysis ---")
    print(f"Full tile traces: {len(full_set)}")
    print(f"Partial tile traces: {len(partial_set)}")
    print(f"Overlapping traces: {overlap}")
    print(f"Only in full: {only_full}")
    print(f"Only in partial: {only_partial}")

    # Check if partial tile is a subset (or mostly overlap)
    if len(partial_set) > 0:
        overlap_pct = 100 * overlap / len(partial_set)
        print(f"\nPartial tile trace overlap with full: {overlap_pct:.1f}%")

    # Now check specific output points near the boundary
    print("\n--- Traces per Output Point (at tile boundary) ---")

    # Test points at the Y boundary
    y_before = 383  # Last Y of full tile
    y_after = 384   # First Y of partial tile

    for ix in [32, 64, 96]:
        # Get coordinates
        x_before = X_grid[ix, y_before]
        y_coord_before = Y_grid[ix, y_before]

        x_after = X_grid[ix, y_after]
        y_coord_after = Y_grid[ix, y_after]

        # Query traces within aperture of each point
        traces_before = index.query_rectangle(
            x_before - APERTURE_RADIUS, x_before + APERTURE_RADIUS,
            y_coord_before - APERTURE_RADIUS, y_coord_before + APERTURE_RADIUS
        )
        traces_after = index.query_rectangle(
            x_after - APERTURE_RADIUS, x_after + APERTURE_RADIUS,
            y_coord_after - APERTURE_RADIUS, y_coord_after + APERTURE_RADIUS
        )

        set_before = set(traces_before)
        set_after = set(traces_after)

        overlap_point = len(set_before & set_after)
        pct_same = 100 * overlap_point / len(set_before) if len(set_before) > 0 else 0

        print(f"  ix={ix}: before={len(set_before)} traces, after={len(set_after)} traces, "
              f"overlap={overlap_point}, same%={pct_same:.1f}%")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
