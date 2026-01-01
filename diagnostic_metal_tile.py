#!/usr/bin/env python3
"""
Diagnostic to trace Metal kernel execution for partial vs full tiles.

Compares buffer sizes, params, and outputs between:
1. Full tile (y=256:384, ny=128)
2. Partial tile (y=384:427, ny=43)
"""

import numpy as np
import zarr
from pathlib import Path
import sys
import ctypes

sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import (
    OutputGridConfig, TilingConfig,
    AmplitudeConfig, ApertureConfig
)
from pstm.pipeline.tile_planner import TilePlanner
from pstm.kernels.metal_compiled import CompiledMetalKernel, MigrationParams

# Test constants
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


def create_params_struct(
    dx, dy, dt_ms, t_start_ms,
    max_aperture, min_aperture, taper_fraction, max_dip_deg,
    apply_spreading, apply_obliquity, apply_aa, aa_dominant_freq,
    n_traces, n_samples, nx, ny, nt, use_3d_velocity
):
    """Create and fill a MigrationParams struct."""
    params = MigrationParams()
    params.dx = dx
    params.dy = dy
    params.dt_ms = dt_ms
    params.t_start_ms = t_start_ms
    params.max_aperture = max_aperture
    params.min_aperture = min_aperture
    params.taper_fraction = taper_fraction
    params.max_dip_deg = max_dip_deg
    params.apply_spreading = apply_spreading
    params.apply_obliquity = apply_obliquity
    params.apply_aa = apply_aa
    params.aa_dominant_freq = aa_dominant_freq
    params.n_traces = n_traces
    params.n_samples = n_samples
    params.nx = nx
    params.ny = ny
    params.nt = nt
    params.use_3d_velocity = use_3d_velocity
    return params


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Metal Kernel Tile Execution Analysis")
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

    print(f"Grid dimensions: ({grid.nx}, {grid.ny}, {grid.nt})")

    # Get coordinate grids
    coords = grid.get_output_coordinates()
    X_full = coords['X']  # (511, 427)
    Y_full = coords['Y']  # (511, 427)

    # Create tile plan
    tiling_config = TilingConfig(auto_tile_size=False, tile_nx=128, tile_ny=128)
    planner = TilePlanner(grid, tiling_config)
    tile_plan = planner.plan()

    # Find a full Y tile and the partial Y tile (same X position)
    full_tile = None
    partial_tile = None
    for t in tile_plan.tiles:
        if t.x_start == 0 and t.y_start == 256:  # Full Y tile
            full_tile = t
        if t.x_start == 0 and t.y_start == 384:  # Partial Y tile
            partial_tile = t

    if not full_tile or not partial_tile:
        print("ERROR: Could not find tiles")
        return

    print(f"\nFull tile: x=[{full_tile.x_start}:{full_tile.x_end}], y=[{full_tile.y_start}:{full_tile.y_end}]")
    print(f"           nx={full_tile.nx}, ny={full_tile.ny}")
    print(f"Partial tile: x=[{partial_tile.x_start}:{partial_tile.x_end}], y=[{partial_tile.y_start}:{partial_tile.y_end}]")
    print(f"              nx={partial_tile.nx}, ny={partial_tile.ny}")

    # Extract coordinates for each tile
    full_X = X_full[full_tile.x_start:full_tile.x_end, full_tile.y_start:full_tile.y_end]
    full_Y = Y_full[full_tile.x_start:full_tile.x_end, full_tile.y_start:full_tile.y_end]

    partial_X = X_full[partial_tile.x_start:partial_tile.x_end, partial_tile.y_start:partial_tile.y_end]
    partial_Y = Y_full[partial_tile.x_start:partial_tile.x_end, partial_tile.y_start:partial_tile.y_end]

    print(f"\nFull tile coords shape: X={full_X.shape}, Y={full_Y.shape}")
    print(f"Partial tile coords shape: X={partial_X.shape}, Y={partial_Y.shape}")

    # Flatten for Metal
    full_x_coords = full_X.flatten().astype(np.float32)
    full_y_coords = full_Y.flatten().astype(np.float32)
    partial_x_coords = partial_X.flatten().astype(np.float32)
    partial_y_coords = partial_Y.flatten().astype(np.float32)

    print(f"\nFlattened coords: full={full_x_coords.shape}, partial={partial_x_coords.shape}")

    # Check coordinate values at specific indices
    print("\n--- Coordinate Values Check ---")

    # Full tile: (ix=64, iy=64) -> index = 64*128+64 = 8256
    full_idx = 64 * 128 + 64
    print(f"Full tile (ix=64, iy=64): index={full_idx}")
    print(f"  coords=({full_x_coords[full_idx]:.2f}, {full_y_coords[full_idx]:.2f})")

    # Partial tile: (ix=64, iy=21) -> index = 64*43+21 = 2773
    partial_idx = 64 * 43 + 21
    print(f"Partial tile (ix=64, iy=21): index={partial_idx}")
    print(f"  coords=({partial_x_coords[partial_idx]:.2f}, {partial_y_coords[partial_idx]:.2f})")

    # Also check direct 2D access
    print(f"  direct 2D: ({partial_X[64, 21]:.2f}, {partial_Y[64, 21]:.2f})")

    # Create MigrationParams for each tile
    print("\n--- MigrationParams Comparison ---")

    # Dummy values for non-geometry params
    n_traces = 1000
    n_samples = 1001
    nt = 1001

    full_params = create_params_struct(
        dx=DX, dy=DY, dt_ms=DT_MS, t_start_ms=0.0,
        max_aperture=2000.0, min_aperture=500.0, taper_fraction=0.1, max_dip_deg=65.0,
        apply_spreading=1, apply_obliquity=1, apply_aa=0, aa_dominant_freq=30.0,
        n_traces=n_traces, n_samples=n_samples,
        nx=full_tile.nx, ny=full_tile.ny, nt=nt,
        use_3d_velocity=1
    )

    partial_params = create_params_struct(
        dx=DX, dy=DY, dt_ms=DT_MS, t_start_ms=0.0,
        max_aperture=2000.0, min_aperture=500.0, taper_fraction=0.1, max_dip_deg=65.0,
        apply_spreading=1, apply_obliquity=1, apply_aa=0, aa_dominant_freq=30.0,
        n_traces=n_traces, n_samples=n_samples,
        nx=partial_tile.nx, ny=partial_tile.ny, nt=nt,
        use_3d_velocity=1
    )

    print(f"Full tile params: nx={full_params.nx}, ny={full_params.ny}, nt={full_params.nt}")
    print(f"Partial tile params: nx={partial_params.nx}, ny={partial_params.ny}, nt={partial_params.nt}")

    # Buffer sizes
    print("\n--- Buffer Sizes ---")

    full_image_size = full_params.nx * full_params.ny * full_params.nt * 4  # float32
    partial_image_size = partial_params.nx * partial_params.ny * partial_params.nt * 4

    full_coords_size = full_params.nx * full_params.ny * 4 * 2  # x and y, float32
    partial_coords_size = partial_params.nx * partial_params.ny * 4 * 2

    full_vel_size = full_params.nx * full_params.ny * full_params.nt * 4  # 3D velocity
    partial_vel_size = partial_params.nx * partial_params.ny * partial_params.nt * 4

    print(f"Full tile:")
    print(f"  Image buffer: {full_image_size / 1024**2:.2f} MB")
    print(f"  Coords buffer: {full_coords_size / 1024**2:.2f} MB")
    print(f"  Velocity buffer: {full_vel_size / 1024**2:.2f} MB")

    print(f"Partial tile:")
    print(f"  Image buffer: {partial_image_size / 1024**2:.2f} MB")
    print(f"  Coords buffer: {partial_coords_size / 1024**2:.2f} MB")
    print(f"  Velocity buffer: {partial_vel_size / 1024**2:.2f} MB")

    print(f"\nRatio (partial/full):")
    print(f"  Image: {partial_image_size / full_image_size:.3f}")
    print(f"  Coords: {partial_coords_size / full_coords_size:.3f}")
    print(f"  Velocity: {partial_vel_size / full_vel_size:.3f}")

    # Index calculation verification
    print("\n--- Index Calculation Verification ---")

    # For partial tile, check that all valid indices are covered
    max_coord_idx = partial_params.nx * partial_params.ny - 1
    max_vel_idx = partial_params.nx * partial_params.ny * partial_params.nt - 1
    max_img_idx = max_vel_idx

    print(f"Partial tile index ranges:")
    print(f"  coord_idx: 0 to {max_coord_idx} (for {partial_params.nx * partial_params.ny} elements)")
    print(f"  vel_idx/img_idx: 0 to {max_vel_idx} (for {partial_params.nx * partial_params.ny * partial_params.nt} elements)")

    # Metal indexing formula verification
    # coord_idx = ix * ny + iy
    # For max valid (ix, iy) = (nx-1, ny-1):
    test_coord_idx = (partial_params.nx - 1) * partial_params.ny + (partial_params.ny - 1)
    print(f"  Max coord index by formula: {test_coord_idx} (should be {max_coord_idx})")

    # vel_idx = ix * ny * nt + iy * nt + it
    test_vel_idx = (partial_params.nx - 1) * partial_params.ny * partial_params.nt + (partial_params.ny - 1) * partial_params.nt + (partial_params.nt - 1)
    print(f"  Max vel index by formula: {test_vel_idx} (should be {max_vel_idx})")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
