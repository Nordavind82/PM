#!/usr/bin/env python3
"""
Regenerate velocity cube with SMOOTH extrapolation for rotated output grid.

Version 2: Uses vectorized gradient-based extrapolation instead of clamping
to avoid discontinuity artifacts at the velocity boundary.
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}


def compute_required_bounds(margin_m: float = 100.0) -> dict:
    all_x = [c[0] for c in GRID_CORNERS.values()]
    all_y = [c[1] for c in GRID_CORNERS.values()]
    return {
        'x_min': min(all_x) - margin_m,
        'x_max': max(all_x) + margin_m,
        'y_min': min(all_y) - margin_m,
        'y_max': max(all_y) + margin_m,
    }


def extrapolate_velocity_vectorized(
    source_data: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_t: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
    gradient_damping: float = 0.3,
) -> np.ndarray:
    """
    Vectorized velocity extrapolation using edge gradients.

    Uses scipy.ndimage.map_coordinates for efficient interpolation,
    then applies gradient-based correction for extrapolated regions.
    """
    nx_new, ny_new, nt = len(new_x), len(new_y), len(source_t)
    nx_old, ny_old = len(source_x), len(source_y)

    print(f"  Source shape: ({nx_old}, {ny_old}, {nt})")
    print(f"  Target shape: ({nx_new}, {ny_new}, {nt})")

    # Create meshgrid for new coordinates
    XX, YY = np.meshgrid(new_x, new_y, indexing='ij')

    # Compute normalized coordinates (0 to nx_old-1, 0 to ny_old-1)
    # for the source grid
    x_norm = (XX - source_x[0]) / (source_x[-1] - source_x[0]) * (nx_old - 1)
    y_norm = (YY - source_y[0]) / (source_y[-1] - source_y[0]) * (ny_old - 1)

    # Clamp for interpolation
    x_clamped = np.clip(x_norm, 0, nx_old - 1)
    y_clamped = np.clip(y_norm, 0, ny_old - 1)

    # Compute distances from clamped positions (in grid units)
    dx = x_norm - x_clamped
    dy = y_norm - y_clamped

    # Create masks for extrapolated regions
    extrap_x_low = x_norm < 0
    extrap_x_high = x_norm > nx_old - 1
    extrap_y_low = y_norm < 0
    extrap_y_high = y_norm > ny_old - 1
    needs_extrap = extrap_x_low | extrap_x_high | extrap_y_low | extrap_y_high

    print(f"  Points needing extrapolation: {np.sum(needs_extrap)} ({100*np.sum(needs_extrap)/XX.size:.1f}%)")

    # Allocate output
    result = np.zeros((nx_new, ny_new, nt), dtype=np.float32)

    # Process each time slice
    from scipy.ndimage import map_coordinates

    for it in range(nt):
        if it % 100 == 0:
            print(f"  Time slice {it}/{nt}...", end='\r')

        vel_slice = source_data[:, :, it].astype(np.float64)

        # Use map_coordinates with clamped coords for base interpolation
        coords = np.array([x_clamped, y_clamped])
        vel_interp = map_coordinates(vel_slice, coords, order=1, mode='nearest')

        # Apply gradient correction for extrapolated regions
        if np.any(needs_extrap):
            # Compute gradients at edges
            grad_x = np.gradient(vel_slice, axis=0)
            grad_y = np.gradient(vel_slice, axis=1)

            # Get edge gradients for extrapolated points
            x_idx = np.clip(x_clamped.astype(int), 0, nx_old - 2)
            y_idx = np.clip(y_clamped.astype(int), 0, ny_old - 2)

            # Sample gradients at clamped positions
            grad_x_at_clamp = map_coordinates(grad_x, coords, order=1, mode='nearest')
            grad_y_at_clamp = map_coordinates(grad_y, coords, order=1, mode='nearest')

            # Grid spacing in source coordinates
            dx_src = source_x[1] - source_x[0]
            dy_src = source_y[1] - source_y[0]

            # Apply gradient correction (dx, dy are in grid units)
            correction = gradient_damping * (
                grad_x_at_clamp * dx * dx_src +
                grad_y_at_clamp * dy * dy_src
            )

            vel_interp = vel_interp + correction

        result[:, :, it] = vel_interp.astype(np.float32)

    print(f"  Time slice {nt}/{nt}... done")

    # Clamp to reasonable range
    result = np.clip(result, 1500, 6000)

    return result


def regenerate_velocity_smooth(
    source_path: Path,
    output_path: Path,
    nx: int = 511,
    ny: int = 427,
    margin_m: float = 100.0,
    smooth_sigma: float = 2.0,
    gradient_damping: float = 0.3,
) -> None:
    """Regenerate velocity cube with smooth extrapolation."""
    print("=" * 70)
    print("VELOCITY CUBE REGENERATION (V2 - Smooth Extrapolation)")
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
    print(f"  Velocity range: [{source_data.min():.0f}, {source_data.max():.0f}] m/s")

    # Compute required bounds
    bounds = compute_required_bounds(margin_m)
    print(f"\nRequired bounds (with {margin_m}m margin):")
    print(f"  X: [{bounds['x_min']:.1f}, {bounds['x_max']:.1f}]")
    print(f"  Y: [{bounds['y_min']:.1f}, {bounds['y_max']:.1f}]")

    # Create new axis arrays
    new_x = np.linspace(bounds['x_min'], bounds['x_max'], nx)
    new_y = np.linspace(bounds['y_min'], bounds['y_max'], ny)

    print(f"\nExtrapolating velocity (gradient damping={gradient_damping})...")
    new_data = extrapolate_velocity_vectorized(
        source_data, source_x, source_y, source_t,
        new_x, new_y,
        gradient_damping=gradient_damping,
    )

    # Apply smoothing to blend extrapolated regions
    if smooth_sigma > 0:
        print(f"\nApplying Gaussian smoothing (sigma={smooth_sigma})...")
        for it in range(len(source_t)):
            if it % 100 == 0:
                print(f"  Smoothing time slice {it}/{len(source_t)}...", end='\r')
            new_data[:, :, it] = gaussian_filter(new_data[:, :, it], sigma=smooth_sigma)
        print(f"  Smoothing time slice {len(source_t)}/{len(source_t)}... done")

    print(f"\nNew velocity range: [{new_data.min():.0f}, {new_data.max():.0f}] m/s")

    # Save
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

    zarr_out.attrs['x_axis'] = new_x.tolist()
    zarr_out.attrs['y_axis'] = new_y.tolist()
    zarr_out.attrs['t_axis_ms'] = source_t.tolist()
    zarr_out.attrs['description'] = (
        f'Velocity cube with smooth gradient extrapolation. '
        f'gradient_damping={gradient_damping}, smooth_sigma={smooth_sigma}'
    )
    zarr_out.attrs['source_file'] = str(source_path)

    print("\n" + "=" * 70)
    print("DONE - Velocity cube regenerated with smooth extrapolation")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate velocity cube with smooth extrapolation"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm.zarr"),
        help="Source velocity cube path (original)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm_smooth.zarr"),
        help="Output velocity cube path",
    )
    parser.add_argument("--nx", type=int, default=511)
    parser.add_argument("--ny", type=int, default=427)
    parser.add_argument("--margin", type=float, default=100.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.0,
                        help="Gaussian smoothing sigma (0 to disable)")
    parser.add_argument("--gradient-damping", type=float, default=0.3,
                        help="Damping factor for gradient extrapolation (0-1)")

    args = parser.parse_args()

    regenerate_velocity_smooth(
        source_path=args.source,
        output_path=args.output,
        nx=args.nx,
        ny=args.ny,
        margin_m=args.margin,
        smooth_sigma=args.smooth_sigma,
        gradient_damping=args.gradient_damping,
    )


if __name__ == "__main__":
    main()
