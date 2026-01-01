#!/usr/bin/env python3
"""
Regenerate velocity cube with correct coverage for rotated output grid.

This script fixes the velocity cube coverage issue that causes diagonal
artifacts in time slices when the velocity cube doesn't fully cover
the rotated output grid.

The original velocity cube only covered the axis-aligned bounding box
of corners c1-c2 (for X) and c1-c4 (for Y), but the rotated grid extends
beyond these bounds at corners c2 (Y min) and c3 (X max).
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from scipy.interpolate import RegularGridInterpolator


# Grid corners for the rotated output grid
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),  # Origin (IL=1, XL=1)
    'c2': (627094.02, 5106803.16),  # Inline end (IL=511, XL=1)
    'c3': (631143.35, 5110261.43),  # Far corner (IL=511, XL=427)
    'c4': (622862.92, 5119956.77),  # Crossline end (IL=1, XL=427)
}


def compute_required_bounds(margin_m: float = 100.0) -> dict:
    """
    Compute the required bounding box for the velocity cube.

    Args:
        margin_m: Extra margin in meters around the grid bounds

    Returns:
        Dictionary with x_min, x_max, y_min, y_max
    """
    all_x = [c[0] for c in GRID_CORNERS.values()]
    all_y = [c[1] for c in GRID_CORNERS.values()]

    return {
        'x_min': min(all_x) - margin_m,
        'x_max': max(all_x) + margin_m,
        'y_min': min(all_y) - margin_m,
        'y_max': max(all_y) + margin_m,
    }


def regenerate_velocity_cube(
    source_path: Path,
    output_path: Path,
    nx: int = 511,
    ny: int = 427,
    margin_m: float = 100.0,
) -> None:
    """
    Regenerate velocity cube with correct coverage.

    Args:
        source_path: Path to original velocity cube (zarr)
        output_path: Path to output velocity cube (zarr)
        nx: Number of X grid points
        ny: Number of Y grid points
        margin_m: Extra margin around grid bounds
    """
    print("=" * 70)
    print("VELOCITY CUBE REGENERATION")
    print("=" * 70)

    # Load source velocity
    print(f"\nLoading source velocity: {source_path}")
    source = zarr.open(str(source_path), mode='r')

    source_data = np.asarray(source)
    source_x = np.array(source.attrs['x_axis'])
    source_y = np.array(source.attrs['y_axis'])
    source_t = np.array(source.attrs['t_axis_ms'])

    print(f"  Shape: {source_data.shape}")
    print(f"  X range: [{source_x[0]:.1f}, {source_x[-1]:.1f}]")
    print(f"  Y range: [{source_y[0]:.1f}, {source_y[-1]:.1f}]")
    print(f"  T range: [{source_t[0]:.1f}, {source_t[-1]:.1f}] ms")
    print(f"  Velocity range: [{source_data.min():.0f}, {source_data.max():.0f}] m/s")

    # Compute required bounds
    bounds = compute_required_bounds(margin_m)
    print(f"\nRequired bounds (with {margin_m}m margin):")
    print(f"  X: [{bounds['x_min']:.1f}, {bounds['x_max']:.1f}]")
    print(f"  Y: [{bounds['y_min']:.1f}, {bounds['y_max']:.1f}]")

    # Check if expansion is needed
    expand_x_min = bounds['x_min'] < source_x[0]
    expand_x_max = bounds['x_max'] > source_x[-1]
    expand_y_min = bounds['y_min'] < source_y[0]
    expand_y_max = bounds['y_max'] > source_y[-1]

    if not any([expand_x_min, expand_x_max, expand_y_min, expand_y_max]):
        print("\nSource velocity already covers required bounds!")
        print("No regeneration needed.")
        return

    print("\nExpansion needed:")
    if expand_x_min:
        print(f"  X min: {source_x[0]:.1f} -> {bounds['x_min']:.1f} ({source_x[0] - bounds['x_min']:.1f}m)")
    if expand_x_max:
        print(f"  X max: {source_x[-1]:.1f} -> {bounds['x_max']:.1f} ({bounds['x_max'] - source_x[-1]:.1f}m)")
    if expand_y_min:
        print(f"  Y min: {source_y[0]:.1f} -> {bounds['y_min']:.1f} ({source_y[0] - bounds['y_min']:.1f}m)")
    if expand_y_max:
        print(f"  Y max: {source_y[-1]:.1f} -> {bounds['y_max']:.1f} ({bounds['y_max'] - source_y[-1]:.1f}m)")

    # Create new axis arrays
    new_x = np.linspace(bounds['x_min'], bounds['x_max'], nx)
    new_y = np.linspace(bounds['y_min'], bounds['y_max'], ny)
    nt = len(source_t)

    print(f"\nNew grid dimensions: {nx} x {ny} x {nt}")
    print(f"  X spacing: {(bounds['x_max'] - bounds['x_min']) / (nx - 1):.2f}m")
    print(f"  Y spacing: {(bounds['y_max'] - bounds['y_min']) / (ny - 1):.2f}m")

    # Create interpolator for source data
    # Use nearest-neighbor for extrapolation by clamping coordinates
    interp = RegularGridInterpolator(
        (source_x, source_y, source_t),
        source_data,
        method='linear',
        bounds_error=False,
        fill_value=None,  # Will clamp instead
    )

    # Create new velocity cube with clamped interpolation
    print("\nInterpolating velocity to new grid...")
    new_data = np.empty((nx, ny, nt), dtype=np.float32)

    # Process in chunks to show progress
    chunk_size = 50
    total_chunks = (nx + chunk_size - 1) // chunk_size

    for i_chunk, i_start in enumerate(range(0, nx, chunk_size)):
        i_end = min(i_start + chunk_size, nx)
        chunk_x = new_x[i_start:i_end]

        # Create meshgrid for this chunk
        xx, yy, tt = np.meshgrid(chunk_x, new_y, source_t, indexing='ij')
        points = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])

        # Clamp coordinates to source bounds
        points[:, 0] = np.clip(points[:, 0], source_x[0], source_x[-1])
        points[:, 1] = np.clip(points[:, 1], source_y[0], source_y[-1])

        # Interpolate
        chunk_vel = interp(points).reshape((i_end - i_start, ny, nt))
        new_data[i_start:i_end, :, :] = chunk_vel

        progress = (i_chunk + 1) / total_chunks * 100
        print(f"  Progress: {progress:.0f}%", end='\r')

    print(f"  Progress: 100%")
    print(f"  New velocity range: [{new_data.min():.0f}, {new_data.max():.0f}] m/s")

    # Save to zarr
    print(f"\nSaving to: {output_path}")
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    zarr_out = zarr.open(
        str(output_path),
        mode='w',
        shape=new_data.shape,
        dtype=np.float32,
        chunks=(64, 64, 128),
    )
    zarr_out[:] = new_data

    # Store metadata
    zarr_out.attrs['x_axis'] = new_x.tolist()
    zarr_out.attrs['y_axis'] = new_y.tolist()
    zarr_out.attrs['t_axis_ms'] = source_t.tolist()
    zarr_out.attrs['description'] = (
        'Velocity cube regenerated with full rotated grid coverage. '
        f'Expanded from source {source_path.name} with margin={margin_m}m.'
    )
    zarr_out.attrs['source_file'] = str(source_path)

    print("\nVelocity cube regenerated successfully!")
    print("=" * 70)

    # Verify coverage
    print("\nVerifying new velocity cube coverage...")
    verify_velocity_coverage(output_path)


def verify_velocity_coverage(velocity_path: Path) -> None:
    """Verify that velocity cube covers the required grid."""
    vel = zarr.open(str(velocity_path), mode='r')
    x_axis = np.array(vel.attrs['x_axis'])
    y_axis = np.array(vel.attrs['y_axis'])

    bounds = compute_required_bounds(margin_m=0)

    print(f"\nVelocity bounds: X [{x_axis[0]:.1f}, {x_axis[-1]:.1f}], Y [{y_axis[0]:.1f}, {y_axis[-1]:.1f}]")
    print(f"Required bounds: X [{bounds['x_min']:.1f}, {bounds['x_max']:.1f}], Y [{bounds['y_min']:.1f}, {bounds['y_max']:.1f}]")

    x_ok = (x_axis[0] <= bounds['x_min']) and (x_axis[-1] >= bounds['x_max'])
    y_ok = (y_axis[0] <= bounds['y_min']) and (y_axis[-1] >= bounds['y_max'])

    if x_ok and y_ok:
        print("\nCOVERAGE CHECK: PASSED")
    else:
        print("\nCOVERAGE CHECK: FAILED")
        if not x_ok:
            print(f"  X coverage insufficient")
        if not y_ok:
            print(f"  Y coverage insufficient")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate velocity cube with correct coverage for rotated grid"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm.zarr"),
        help="Source velocity cube path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm_full.zarr"),
        help="Output velocity cube path",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=511,
        help="Number of X grid points",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=427,
        help="Number of Y grid points",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=100.0,
        help="Margin in meters around grid bounds",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing velocity coverage",
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_velocity_coverage(args.source)
    else:
        regenerate_velocity_cube(
            source_path=args.source,
            output_path=args.output,
            nx=args.nx,
            ny=args.ny,
            margin_m=args.margin,
        )


if __name__ == "__main__":
    main()
