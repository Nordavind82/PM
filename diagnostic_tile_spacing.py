#!/usr/bin/env python3
"""
Diagnostic: Check dx/dy spacing for each tile.

This checks if tile spacing is computed correctly for full vs partial tiles.
"""

import numpy as np
from pathlib import Path

# Simulate tile generation like the executor does
def analyze_tile_spacing():
    """Analyze tile spacing for full and partial tiles."""

    print("=" * 70)
    print("TILE SPACING DIAGNOSTIC")
    print("=" * 70)

    # Grid parameters (from run_pstm_all_offsets.py)
    nx = 511  # Total X points
    ny = 427  # Total Y points
    nt = 1001

    # Grid spacing from config
    dx_grid = 25.0  # meters
    dy_grid = 12.5  # meters

    # X range: 0 to (nx-1)*dx = 510*25 = 12750
    # Y range: 0 to (ny-1)*dy = 426*12.5 = 5325
    x_min_global = 0.0
    y_min_global = 0.0

    # Create global coordinate arrays like OutputGridConfig does
    x_coords_global = np.arange(nx) * dx_grid + x_min_global
    y_coords_global = np.arange(ny) * dy_grid + y_min_global

    print(f"\nGlobal grid: {nx} x {ny} x {nt}")
    print(f"Global dx={dx_grid}, dy={dy_grid}")
    print(f"X range: {x_coords_global[0]} to {x_coords_global[-1]}")
    print(f"Y range: {y_coords_global[0]} to {y_coords_global[-1]}")

    # Tile settings (128x128)
    tile_nx = 128
    tile_ny = 128

    n_tiles_x = (nx + tile_nx - 1) // tile_nx  # 4
    n_tiles_y = (ny + tile_ny - 1) // tile_ny  # 4

    print(f"\nTile size: {tile_nx} x {tile_ny}")
    print(f"Number of tiles: {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y}")

    print("\n" + "=" * 70)
    print("TILE-BY-TILE SPACING ANALYSIS")
    print("=" * 70)

    for iy in range(n_tiles_y):
        for ix in range(n_tiles_x):
            # Grid indices (same as tile_planner.py)
            x_start = ix * tile_nx
            x_end = min((ix + 1) * tile_nx, nx)
            y_start = iy * tile_ny
            y_end = min((iy + 1) * tile_ny, ny)

            tile_nx_actual = x_end - x_start
            tile_ny_actual = y_end - y_start

            # Tile bounds (same as tile_planner.py lines 354-357)
            x_min = x_coords_global[x_start]
            x_max = x_coords_global[x_end - 1]
            y_min = y_coords_global[y_start]
            y_max = y_coords_global[y_end - 1]

            # Create tile axes using linspace (same as executor.py line 1643-1644)
            x_axis = np.linspace(x_min, x_max, tile_nx_actual)
            y_axis = np.linspace(y_min, y_max, tile_ny_actual)

            # Compute dx/dy as the kernel does (metal_compiled.py lines 693-695)
            dx_computed = x_axis[1] - x_axis[0] if tile_nx_actual > 1 else 25.0
            dy_computed = y_axis[1] - y_axis[0] if tile_ny_actual > 1 else 25.0

            is_partial_x = tile_nx_actual < tile_nx
            is_partial_y = tile_ny_actual < tile_ny

            marker = ""
            if is_partial_x or is_partial_y:
                marker = " *** PARTIAL TILE ***"

            dy_error = abs(dy_computed - dy_grid) / dy_grid * 100

            print(f"\nTile ({ix},{iy}): indices [{x_start}:{x_end}, {y_start}:{y_end}]")
            print(f"  Size: {tile_nx_actual} x {tile_ny_actual}{marker}")
            print(f"  X bounds: [{x_min:.1f}, {x_max:.1f}], range = {x_max - x_min:.1f}")
            print(f"  Y bounds: [{y_min:.1f}, {y_max:.1f}], range = {y_max - y_min:.1f}")
            print(f"  dx_computed = {dx_computed:.4f} (expected {dx_grid}, error = {abs(dx_computed - dx_grid) / dx_grid * 100:.2f}%)")
            print(f"  dy_computed = {dy_computed:.4f} (expected {dy_grid}, error = {dy_error:.2f}%)")

            if dy_error > 1.0:
                print(f"  !!! WARNING: dy error > 1% !!!")

    # Specific analysis for the boundary tiles
    print("\n" + "=" * 70)
    print("BOUNDARY TILE ANALYSIS (Y=384 boundary)")
    print("=" * 70)

    # Tile at iy=2 (full tile ending at Y=384)
    iy = 2
    y_start = iy * tile_ny  # 256
    y_end = min((iy + 1) * tile_ny, ny)  # 384
    tile_ny_2 = y_end - y_start  # 128
    y_min_2 = y_coords_global[y_start]
    y_max_2 = y_coords_global[y_end - 1]
    y_axis_2 = np.linspace(y_min_2, y_max_2, tile_ny_2)
    dy_2 = y_axis_2[1] - y_axis_2[0]

    print(f"\nTile iy=2 (FULL - ends at Y boundary):")
    print(f"  y_start={y_start}, y_end={y_end}, ny={tile_ny_2}")
    print(f"  y_min={y_min_2:.1f}, y_max={y_max_2:.1f}")
    print(f"  y_axis[383] in global = {y_coords_global[383]:.1f}")
    print(f"  dy_computed = {dy_2:.4f}")

    # Tile at iy=3 (partial tile starting at Y=384)
    iy = 3
    y_start = iy * tile_ny  # 384
    y_end = min((iy + 1) * tile_ny, ny)  # 427
    tile_ny_3 = y_end - y_start  # 43
    y_min_3 = y_coords_global[y_start]
    y_max_3 = y_coords_global[y_end - 1]
    y_axis_3 = np.linspace(y_min_3, y_max_3, tile_ny_3)
    dy_3 = y_axis_3[1] - y_axis_3[0]

    print(f"\nTile iy=3 (PARTIAL - starts at Y boundary):")
    print(f"  y_start={y_start}, y_end={y_end}, ny={tile_ny_3}")
    print(f"  y_min={y_min_3:.1f}, y_max={y_max_3:.1f}")
    print(f"  y_axis[384] in global = {y_coords_global[384]:.1f}")
    print(f"  dy_computed = {dy_3:.4f}")

    # Compare
    print(f"\nDY COMPARISON:")
    print(f"  Full tile (iy=2): dy = {dy_2:.4f}")
    print(f"  Partial tile (iy=3): dy = {dy_3:.4f}")
    print(f"  Difference: {abs(dy_3 - dy_2):.4f} ({abs(dy_3 - dy_2) / dy_2 * 100:.2f}%)")

    # Check what happens at the boundary
    print(f"\n  Y coordinate at index 383 (last in full tile): {y_coords_global[383]:.1f}")
    print(f"  Y coordinate at index 384 (first in partial tile): {y_coords_global[384]:.1f}")
    print(f"  Gap between tiles: {y_coords_global[384] - y_coords_global[383]:.4f} (should be {dy_grid})")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    analyze_tile_spacing()
