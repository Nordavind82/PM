#!/usr/bin/env python3
"""
Check if the output tile coordinates match the expected grid coordinates.

This diagnoses if there's a mismatch between what coordinates the kernel uses
vs what the grid defines.
"""

import numpy as np
import zarr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import OutputGridConfig, TilingConfig
from pstm.pipeline.tile_planner import TilePlanner

# Paths
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m")
BIN_DIR = OUTPUT_DIR / "migration_bin_01"

# Grid params
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Coordinate System Verification")
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

    print(f"Grid dimensions: ({grid.nx}, {grid.ny}, {grid.nt})")

    # Get coordinate grids
    coords = grid.get_output_coordinates()
    X_grid = coords['X']  # (nx, ny)
    Y_grid = coords['Y']  # (nx, ny)

    print(f"X_grid shape: {X_grid.shape}")
    print(f"Y_grid shape: {Y_grid.shape}")

    # Create tile plan
    tiling_config = TilingConfig(auto_tile_size=False, tile_nx=128, tile_ny=128)
    planner = TilePlanner(grid, tiling_config)
    tile_plan = planner.plan()

    # Find full and partial Y tiles
    full_tile = None
    partial_tile = None
    for t in tile_plan.tiles:
        if t.x_start == 0 and t.y_start == 256:
            full_tile = t
        if t.x_start == 0 and t.y_start == 384:
            partial_tile = t

    print(f"\nFull tile: y=[{full_tile.y_start}:{full_tile.y_end}] (ny={full_tile.ny})")
    print(f"Partial tile: y=[{partial_tile.y_start}:{partial_tile.y_end}] (ny={partial_tile.ny})")

    # Extract coordinate grids for each tile (as executor does)
    full_X = X_grid[full_tile.x_start:full_tile.x_end, full_tile.y_start:full_tile.y_end]
    full_Y = Y_grid[full_tile.x_start:full_tile.x_end, full_tile.y_start:full_tile.y_end]

    partial_X = X_grid[partial_tile.x_start:partial_tile.x_end, partial_tile.y_start:partial_tile.y_end]
    partial_Y = Y_grid[partial_tile.x_start:partial_tile.x_end, partial_tile.y_start:partial_tile.y_end]

    print(f"\nFull tile coords: X shape={full_X.shape}, Y shape={full_Y.shape}")
    print(f"Partial tile coords: X shape={partial_X.shape}, Y shape={partial_Y.shape}")

    # Create 1D axes as executor does
    # This is where the bug might be!
    full_x_axis = np.linspace(full_tile.x_min, full_tile.x_max, full_tile.nx)
    full_y_axis = np.linspace(full_tile.y_min, full_tile.y_max, full_tile.ny)

    partial_x_axis = np.linspace(partial_tile.x_min, partial_tile.x_max, partial_tile.nx)
    partial_y_axis = np.linspace(partial_tile.y_min, partial_tile.y_max, partial_tile.ny)

    # Compute dx, dy as executor does
    full_dx = full_x_axis[1] - full_x_axis[0] if full_tile.nx > 1 else 25.0
    full_dy = full_y_axis[1] - full_y_axis[0] if full_tile.ny > 1 else 12.5

    partial_dx = partial_x_axis[1] - partial_x_axis[0] if partial_tile.nx > 1 else 25.0
    partial_dy = partial_y_axis[1] - partial_y_axis[0] if partial_tile.ny > 1 else 12.5

    print("\n--- 1D Axis Analysis (as executor creates them) ---")
    print(f"Full tile: dx={full_dx:.4f}, dy={full_dy:.4f}")
    print(f"Partial tile: dx={partial_dx:.4f}, dy={partial_dy:.4f}")
    print(f"Expected: dx=25.0, dy=12.5")

    # Check if 2D grids match 1D axes (they shouldn't for rotated grids!)
    print("\n--- Checking if 2D grids match 1D axes ---")

    # For full tile, compare first row of 2D grid with 1D x_axis
    x_axis_spacing = full_x_axis[1] - full_x_axis[0]
    x_grid_row_spacing = full_X[1, 0] - full_X[0, 0]
    print(f"Full tile X: 1D axis spacing={x_axis_spacing:.4f}, 2D grid row spacing={x_grid_row_spacing:.4f}")

    y_axis_spacing = full_y_axis[1] - full_y_axis[0]
    y_grid_col_spacing = full_Y[0, 1] - full_Y[0, 0]
    print(f"Full tile Y: 1D axis spacing={y_axis_spacing:.4f}, 2D grid col spacing={y_grid_col_spacing:.4f}")

    # For partial tile
    px_axis_spacing = partial_x_axis[1] - partial_x_axis[0]
    px_grid_row_spacing = partial_X[1, 0] - partial_X[0, 0]
    print(f"Partial tile X: 1D axis spacing={px_axis_spacing:.4f}, 2D grid row spacing={px_grid_row_spacing:.4f}")

    py_axis_spacing = partial_y_axis[1] - partial_y_axis[0]
    py_grid_col_spacing = partial_Y[0, 1] - partial_Y[0, 0]
    print(f"Partial tile Y: 1D axis spacing={py_axis_spacing:.4f}, 2D grid col spacing={py_grid_col_spacing:.4f}")

    # Check actual coordinate values
    print("\n--- Actual Coordinate Values ---")

    # Check coordinates at specific points
    test_points = [(0, 0), (64, 64), (127, 127), (64, 0)]
    print("Full tile (x_idx, y_idx) -> (X, Y):")
    for ix, iy in test_points:
        if ix < full_tile.nx and iy < full_tile.ny:
            print(f"  ({ix}, {iy}): ({full_X[ix, iy]:.2f}, {full_Y[ix, iy]:.2f})")

    print("Partial tile (x_idx, y_idx) -> (X, Y):")
    for ix, iy in [(0, 0), (64, 21), (127, 42), (64, 0)]:
        if ix < partial_tile.nx and iy < partial_tile.ny:
            print(f"  ({ix}, {iy}): ({partial_X[ix, iy]:.2f}, {partial_Y[ix, iy]:.2f})")

    # Verify coordinate continuity at tile boundary
    print("\n--- Coordinate Continuity at Tile Boundary ---")
    # Last Y of full tile should be adjacent to first Y of partial tile

    # Global indices
    last_full_y = full_tile.y_end - 1  # 383
    first_partial_y = partial_tile.y_start  # 384

    # Check from global grid
    print(f"From global grid:")
    print(f"  Y={last_full_y}: coord=({X_grid[64, last_full_y]:.2f}, {Y_grid[64, last_full_y]:.2f})")
    print(f"  Y={first_partial_y}: coord=({X_grid[64, first_partial_y]:.2f}, {Y_grid[64, first_partial_y]:.2f})")

    spacing = np.sqrt((X_grid[64, first_partial_y] - X_grid[64, last_full_y])**2 +
                      (Y_grid[64, first_partial_y] - Y_grid[64, last_full_y])**2)
    print(f"  Spacing: {spacing:.2f}m (expected ~12.5m)")

    # Check from tile-local grids
    print(f"From tile-local grids:")
    last_local = full_tile.ny - 1  # 127
    print(f"  Full tile (64, {last_local}): ({full_X[64, last_local]:.2f}, {full_Y[64, last_local]:.2f})")
    print(f"  Partial tile (64, 0): ({partial_X[64, 0]:.2f}, {partial_Y[64, 0]:.2f})")

    spacing_local = np.sqrt((partial_X[64, 0] - full_X[64, last_local])**2 +
                            (partial_Y[64, 0] - full_Y[64, last_local])**2)
    print(f"  Spacing: {spacing_local:.2f}m (expected ~12.5m)")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
