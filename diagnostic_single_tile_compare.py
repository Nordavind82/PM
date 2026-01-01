#!/usr/bin/env python3
"""
Compare Metal kernel output with actual migration output for partial tiles.

Loads the actual migration output and runs the kernel directly on a partial tile
using the same input data to identify where the amplitude discrepancy occurs.
"""

import numpy as np
import zarr
from pathlib import Path
import sys
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import OutputGridConfig, TilingConfig
from pstm.pipeline.tile_planner import TilePlanner

# Paths
DATA_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m")
BIN_DIR = OUTPUT_DIR / "migration_bin_01"
BIN_INPUT = DATA_DIR / "offset_bin_01"

# Grid params
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Single Tile Amplitude Analysis")
    print("=" * 70)

    # Load migration output
    stack_path = BIN_DIR / "migrated_stack.zarr"
    fold_path = BIN_DIR / "fold.zarr"

    print(f"Loading migration output from {BIN_DIR}...")
    z_stack = zarr.open_array(str(stack_path), mode='r')
    z_fold = zarr.open_array(str(fold_path), mode='r')

    stack = np.array(z_stack[:])
    fold = np.array(z_fold[:])

    print(f"Stack shape: {stack.shape}")
    print(f"Fold shape: {fold.shape}")

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

    # Find the partial Y tile and the preceding full Y tile (at same X)
    full_tile = None
    partial_tile = None
    for t in tile_plan.tiles:
        if t.x_start == 0 and t.y_start == 256:  # Full Y tile
            full_tile = t
        if t.x_start == 0 and t.y_start == 384:  # Partial Y tile
            partial_tile = t

    print(f"\nFull tile: x=[{full_tile.x_start}:{full_tile.x_end}], y=[{full_tile.y_start}:{full_tile.y_end}]")
    print(f"Partial tile: x=[{partial_tile.x_start}:{partial_tile.x_end}], y=[{partial_tile.y_start}:{partial_tile.y_end}]")

    # Extract regions from migration output
    full_region = stack[full_tile.x_start:full_tile.x_end, full_tile.y_start:full_tile.y_end, :]
    partial_region = stack[partial_tile.x_start:partial_tile.x_end, partial_tile.y_start:partial_tile.y_end, :]

    full_fold_region = fold[full_tile.x_start:full_tile.x_end, full_tile.y_start:full_tile.y_end, :]
    partial_fold_region = fold[partial_tile.x_start:partial_tile.x_end, partial_tile.y_start:partial_tile.y_end, :]

    print(f"\nFull tile region shape: {full_region.shape}")
    print(f"Partial tile region shape: {partial_region.shape}")

    # Time range for analysis
    t_start, t_end = 300, 500  # 600-1000ms

    # Compute RMS for each tile
    full_rms = np.sqrt(np.mean(full_region[:, :, t_start:t_end]**2))
    partial_rms = np.sqrt(np.mean(partial_region[:, :, t_start:t_end]**2))

    print(f"\n--- RMS Amplitude (t={t_start*2}ms to {t_end*2}ms) ---")
    print(f"Full tile RMS: {full_rms:.6f}")
    print(f"Partial tile RMS: {partial_rms:.6f}")
    print(f"Ratio (partial/full): {partial_rms/full_rms:.4f}")

    # Check fold
    full_fold_mean = full_fold_region[:, :, t_start:t_end].mean()
    partial_fold_mean = partial_fold_region[:, :, t_start:t_end].mean()

    print(f"\n--- Fold Analysis ---")
    print(f"Full tile mean fold: {full_fold_mean:.1f}")
    print(f"Partial tile mean fold: {partial_fold_mean:.1f}")
    print(f"Fold ratio: {partial_fold_mean/full_fold_mean:.4f}")

    # Per-output-point amplitude ratio
    # For corresponding points across the tile boundary
    print("\n--- Per-Point Comparison (at Y boundary) ---")

    y_before = 383  # Last Y of full tile (in global coords)
    y_after = 384   # First Y of partial tile (in global coords)

    # Convert to tile-local indices
    y_local_before = y_before - full_tile.y_start  # 383 - 256 = 127
    y_local_after = y_after - partial_tile.y_start  # 384 - 384 = 0

    for ix in [32, 64, 96]:
        for it in [350, 400, 450]:  # Three time samples
            val_before = stack[ix, y_before, it]
            val_after = stack[ix, y_after, it]
            fold_before = fold[ix, y_before, it]
            fold_after = fold[ix, y_after, it]

            # Compute ratio
            if val_before != 0:
                amp_ratio = val_after / val_before
            else:
                amp_ratio = float('nan')

            print(f"  ({ix}, Y={y_before}/{y_after}, t={it*2}ms): "
                  f"before={val_before:.6f} after={val_after:.6f} ratio={amp_ratio:.3f} "
                  f"fold={fold_before}/{fold_after}")

    # Check if amplitude/fold ratio is consistent
    print("\n--- Amplitude/Fold Ratio Check ---")
    # If the issue is with fold, then amplitude/fold should be constant

    for ix in [32, 64, 96]:
        for it in [350, 400, 450]:
            val_before = stack[ix, y_before, it]
            val_after = stack[ix, y_after, it]
            fold_before = fold[ix, y_before, it]
            fold_after = fold[ix, y_after, it]

            if fold_before > 0 and fold_after > 0:
                amp_per_fold_before = val_before / fold_before
                amp_per_fold_after = val_after / fold_after
                ratio = amp_per_fold_after / amp_per_fold_before if amp_per_fold_before != 0 else float('nan')
                print(f"  ({ix}, t={it*2}ms): amp/fold before={amp_per_fold_before:.8f}, "
                      f"after={amp_per_fold_after:.8f}, ratio={ratio:.4f}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
