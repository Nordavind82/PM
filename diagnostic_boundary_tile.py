#!/usr/bin/env python3
"""
Diagnostic script to isolate the partial tile amplitude bug.

Tests the Metal kernel directly on:
1. The last partial Y tile (y_start=384, ny=43)
2. The preceding full tile (y_start=256, ny=128)

Compares Metal output with expected Python computation.
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

# Grid params (matching run_pstm_all_offsets.py)
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}
DX = 25.0
DY = 12.5
DT_MS = 2.0
T_MIN_MS = 0.0
T_MAX_MS = 2000.0


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Partial Tile Boundary Analysis")
    print("=" * 70)

    # Create output grid
    grid = OutputGridConfig.from_corners(
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

    print(f"\nGrid dimensions: ({grid.nx}, {grid.ny}, {grid.nt})")

    # Get coordinate grids
    coords = grid.get_output_coordinates()
    X_full = coords['X']  # (nx, ny)
    Y_full = coords['Y']  # (nx, ny)

    print(f"Full X grid shape: {X_full.shape}")
    print(f"Full Y grid shape: {Y_full.shape}")

    # Create tile planner
    tiling_config = TilingConfig(auto_tile_size=False, tile_nx=128, tile_ny=128)
    planner = TilePlanner(grid, tiling_config)
    tile_plan = planner.plan()
    tiles = tile_plan.tiles

    print(f"\nTotal tiles: {len(tiles)}")

    # Find tiles near Y boundary
    y_boundary_tiles = []
    for t in tiles:
        if t.y_start >= 256:
            y_boundary_tiles.append(t)

    # Sort by y_start
    y_boundary_tiles.sort(key=lambda t: (t.y_start, t.x_start))

    # Get last two rows of tiles (y=256:384 and y=384:427)
    print("\n--- Tiles near Y boundary ---")
    for t in y_boundary_tiles[:8]:  # First 8 tiles after y=256
        print(f"  Tile: x=[{t.x_start}:{t.x_end}] ({t.nx}), y=[{t.y_start}:{t.y_end}] ({t.ny})")

    # Load migration output for bin 01
    stack_path = BIN_DIR / "migrated_stack.zarr"
    if not stack_path.exists():
        print(f"\nERROR: {stack_path} not found")
        return

    z = zarr.open_array(str(stack_path), mode='r')
    stack = np.array(z[:])
    print(f"\nLoaded stack shape: {stack.shape}")

    # Analyze amplitude profile across Y for a few X positions
    t_idx = 400  # 800ms
    x_positions = [64, 256, 450]

    print(f"\n--- Y amplitude profile at t={t_idx*DT_MS}ms ---")
    for x_idx in x_positions:
        y_profile = np.abs(stack[x_idx, :, t_idx])

        # Compute RMS in windows
        window = 10
        n_windows = len(y_profile) // window

        for w in range(n_windows):
            y_start = w * window
            y_end = min((w + 1) * window, len(y_profile))
            rms = np.sqrt(np.mean(y_profile[y_start:y_end]**2))

            # Mark boundaries
            marker = ""
            if y_start in [256, 384]:
                marker = " <-- TILE BOUNDARY"
            elif y_start == 380:
                marker = " <-- just before boundary"
            elif y_start == 390:
                marker = " <-- just after boundary"

            if rms > 0.001:  # Only print non-zero
                pass  # Print all for debugging

    # Detailed analysis near Y=384 boundary
    print("\n--- Detailed analysis at Y=384 boundary ---")

    # Compare amplitudes just before and just after boundary
    y_before = 383  # Last pixel of full tile
    y_after = 384   # First pixel of partial tile

    for x_idx in [64, 256, 450]:
        amp_before = stack[x_idx, y_before, t_idx]
        amp_after = stack[x_idx, y_after, t_idx]
        ratio = amp_after / amp_before if amp_before != 0 else float('nan')
        print(f"  X={x_idx}: Y={y_before} amp={amp_before:.6f}, Y={y_after} amp={amp_after:.6f}, ratio={ratio:.3f}")

    # Check fold
    fold_path = BIN_DIR / "fold.zarr"
    if fold_path.exists():
        z_fold = zarr.open_array(str(fold_path), mode='r')
        fold = np.array(z_fold[:])
        print(f"\n--- Fold analysis at Y=384 boundary (t={t_idx*DT_MS}ms) ---")
        for x_idx in [64, 256, 450]:
            fold_before = fold[x_idx, y_before, t_idx]
            fold_after = fold[x_idx, y_after, t_idx]
            print(f"  X={x_idx}: Y={y_before} fold={fold_before}, Y={y_after} fold={fold_after}")

    # Verify coordinate grids at boundary
    print("\n--- Coordinate verification at Y=384 boundary ---")
    for x_idx in [64, 256, 450]:
        coord_before_x = X_full[x_idx, y_before]
        coord_before_y = Y_full[x_idx, y_before]
        coord_after_x = X_full[x_idx, y_after]
        coord_after_y = Y_full[x_idx, y_after]

        # Expected spacing
        expected_dy = DY
        actual_dy = np.sqrt((coord_after_x - coord_before_x)**2 + (coord_after_y - coord_before_y)**2)

        print(f"  X={x_idx}:")
        print(f"    Y={y_before}: coord=({coord_before_x:.2f}, {coord_before_y:.2f})")
        print(f"    Y={y_after}:  coord=({coord_after_x:.2f}, {coord_after_y:.2f})")
        print(f"    Spacing: expected={expected_dy:.2f}m, actual={actual_dy:.2f}m")

    # Now check the tile extraction
    print("\n--- Tile coordinate extraction check ---")

    # Tile at y_start=256 (full tile, ny=128)
    tile_256_X = X_full[0:128, 256:384]
    tile_256_Y = Y_full[0:128, 256:384]

    # Tile at y_start=384 (partial tile, ny=43)
    tile_384_X = X_full[0:128, 384:427]
    tile_384_Y = Y_full[0:128, 384:427]

    print(f"  Full tile (256:384): shape {tile_256_X.shape}")
    print(f"  Partial tile (384:427): shape {tile_384_X.shape}")

    # Check coordinates at overlap boundary
    # Last point of full tile at y=383 (relative y=127)
    # First point of partial tile at y=384 (relative y=0)
    print("\n  Coordinate continuity check (X=64):")
    last_of_full_x = tile_256_X[64, 127]
    last_of_full_y = tile_256_Y[64, 127]
    first_of_partial_x = tile_384_X[64, 0]
    first_of_partial_y = tile_384_Y[64, 0]

    print(f"    Last of full tile (y=383 -> local y=127): ({last_of_full_x:.2f}, {last_of_full_y:.2f})")
    print(f"    First of partial tile (y=384 -> local y=0): ({first_of_partial_x:.2f}, {first_of_partial_y:.2f})")

    spacing = np.sqrt((first_of_partial_x - last_of_full_x)**2 + (first_of_partial_y - last_of_full_y)**2)
    print(f"    Spacing between them: {spacing:.2f}m (expected {DY:.2f}m)")

    # Check flattening order
    print("\n--- Flattening order check ---")
    flat_256_X = tile_256_X.flatten()
    flat_256_Y = tile_256_Y.flatten()
    flat_384_X = tile_384_X.flatten()
    flat_384_Y = tile_384_Y.flatten()

    print(f"  Full tile flattened: {flat_256_X.shape}")
    print(f"  Partial tile flattened: {flat_384_X.shape}")

    # Check indexing: for ix=64, iy=0 in partial tile
    # Index should be 64 * 43 + 0 = 2752
    ix, iy = 64, 0
    expected_idx = ix * 43 + iy
    coord_x = flat_384_X[expected_idx]
    coord_y = flat_384_Y[expected_idx]
    direct_x = tile_384_X[ix, iy]
    direct_y = tile_384_Y[ix, iy]

    print(f"\n  Index check for partial tile (ix={ix}, iy={iy}):")
    print(f"    Expected flat index: {expected_idx}")
    print(f"    From flattened array: ({coord_x:.2f}, {coord_y:.2f})")
    print(f"    Direct 2D access: ({direct_x:.2f}, {direct_y:.2f})")
    print(f"    Match: {np.allclose([coord_x, coord_y], [direct_x, direct_y])}")

    # Same for full tile (ix=64, iy=0)
    expected_idx_full = ix * 128 + iy
    coord_x_full = flat_256_X[expected_idx_full]
    coord_y_full = flat_256_Y[expected_idx_full]
    direct_x_full = tile_256_X[ix, iy]
    direct_y_full = tile_256_Y[ix, iy]

    print(f"\n  Index check for full tile (ix={ix}, iy={iy}):")
    print(f"    Expected flat index: {expected_idx_full}")
    print(f"    From flattened array: ({coord_x_full:.2f}, {coord_y_full:.2f})")
    print(f"    Direct 2D access: ({direct_x_full:.2f}, {direct_y_full:.2f})")
    print(f"    Match: {np.allclose([coord_x_full, coord_y_full], [direct_x_full, direct_y_full])}")

    # RMS amplitude analysis across tiles
    print("\n--- RMS Amplitude Analysis Across Y Tiles ---")

    # Compute RMS for each Y tile band
    y_ranges = [(0, 128), (128, 256), (256, 384), (384, 427)]
    t_range = (300, 500)  # 600-1000ms

    for y_start, y_end in y_ranges:
        region = stack[:, y_start:y_end, t_range[0]:t_range[1]]
        rms = np.sqrt(np.mean(region**2))
        abs_mean = np.mean(np.abs(region))
        print(f"  Y=[{y_start}:{y_end}] ({y_end-y_start} pixels): RMS={rms:.6f}, |mean|={abs_mean:.6f}")

    # Ratio between last two tiles
    full_region = stack[:, 256:384, t_range[0]:t_range[1]]
    partial_region = stack[:, 384:427, t_range[0]:t_range[1]]

    full_rms = np.sqrt(np.mean(full_region**2))
    partial_rms = np.sqrt(np.mean(partial_region**2))
    ratio = partial_rms / full_rms

    print(f"\n  Ratio (partial/full): {ratio:.3f}")
    print(f"  Expected if bug: ~0.4-0.5 | Expected if correct: ~1.0")

    # Also compare adjacent X tiles (control test)
    print("\n--- RMS Amplitude Analysis Across X Tiles (Control) ---")
    x_ranges = [(0, 128), (128, 256), (256, 384), (384, 511)]

    for x_start, x_end in x_ranges:
        region = stack[x_start:x_end, :, t_range[0]:t_range[1]]
        rms = np.sqrt(np.mean(region**2))
        print(f"  X=[{x_start}:{x_end}] ({x_end-x_start} pixels): RMS={rms:.6f}")

    # Ratio between last two X tiles
    full_x_region = stack[256:384, :, t_range[0]:t_range[1]]
    partial_x_region = stack[384:511, :, t_range[0]:t_range[1]]

    full_x_rms = np.sqrt(np.mean(full_x_region**2))
    partial_x_rms = np.sqrt(np.mean(partial_x_region**2))
    x_ratio = partial_x_rms / full_x_rms

    print(f"\n  Ratio (partial/full for X): {x_ratio:.3f}")
    print(f"  If Y-only bug, this should be ~1.0")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
