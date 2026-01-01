#!/usr/bin/env python3
"""
Diagnostic: Analyze AA weight behavior at Y=384 boundary.

Simulate the AA weight calculation for points near the boundary to identify
why there's an amplitude drop in partial Y tiles.
"""

import numpy as np
import zarr
from pathlib import Path

# Parameters matching migration config
DX = 25.0
DY = 12.5
DT_MS = 2.0
DOMINANT_FREQ = 30.0  # Hz


def compute_aa_weight_py(ox, oy, mx, my, velocity, t_travel, dx, dy, dominant_freq):
    """
    Python implementation of Metal's compute_aa_weight function.

    Returns:
        AA weight (0 to 1)
    """
    if t_travel < 0.001:
        return 1.0

    # Compute local dip in x and y directions
    denom = velocity * velocity * t_travel * 0.5
    dip_x = (mx - ox) / denom
    dip_y = (my - oy) / denom

    # sin(theta) = |dt/dx| * v / 2, clamped to [0, 1]
    sin_theta_x = min(abs(dip_x) * velocity * 0.5, 1.0)
    sin_theta_y = min(abs(dip_y) * velocity * 0.5, 1.0)

    # Combined sin(theta) using max of both directions
    sin_theta = max(sin_theta_x, sin_theta_y)

    # Avoid division by zero for near-zero dip
    if sin_theta < 0.01:
        return 1.0

    # Maximum unaliased frequency
    grid_spacing = max(dx, dy)
    f_max = velocity / (4.0 * grid_spacing * sin_theta)

    # Clamp to Nyquist
    f_max = min(f_max, 500.0)

    # Triangle filter response: W(f) = max(0, 1 - f/f_max)
    aa_weight = max(0.0, 1.0 - dominant_freq / f_max)

    return aa_weight


def analyze_aa_at_boundary():
    """Analyze AA weights near Y=384 boundary."""

    print("=" * 70)
    print("AA WEIGHT BOUNDARY ANALYSIS")
    print("=" * 70)

    # Load migrated data to get actual amplitudes
    base_dir = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/migration_bin_25")
    stack_path = base_dir / "migrated_stack.zarr"
    fold_path = base_dir / "fold.zarr"

    if not stack_path.exists():
        print(f"Migration output not found: {stack_path}")
        print("Run migration first!")
        return

    z_stack = zarr.open_array(str(stack_path), mode='r')
    z_fold = zarr.open_array(str(fold_path), mode='r')

    print(f"\nLoaded stack shape: {z_stack.shape}")

    # Analyze boundary region
    data = np.array(z_stack[:])
    fold = np.array(z_fold[:])

    # Analyze RMS by Y in the boundary region
    t_range = slice(300, 500)  # 600-1000 ms

    print("\n--- Boundary Region Analysis ---")
    print("Y index | Y coord (m) | RMS amplitude | Mean fold | Ratio to Y=383")
    print("-" * 70)

    y383_rms = np.sqrt(np.mean(data[:, 383, t_range]**2))

    for iy in range(380, 390):
        if iy >= data.shape[1]:
            continue

        rms = np.sqrt(np.mean(data[:, iy, t_range]**2))
        mean_fold = np.mean(fold[:, iy, t_range])
        y_coord = iy * DY
        ratio = rms / y383_rms if y383_rms > 0 else 0

        marker = ""
        if iy == 383:
            marker = " <- Last in full tile (iy=2)"
        elif iy == 384:
            marker = " <- First in partial tile (iy=3)"

        print(f"  {iy:3d}   |  {y_coord:6.1f}    |  {rms:.8f} | {mean_fold:8.1f} | {ratio:.4f}{marker}")

    # Simulate AA calculation at boundary
    print("\n" + "=" * 70)
    print("SIMULATED AA WEIGHT CALCULATION")
    print("=" * 70)

    # Typical parameters for the survey
    velocity = 3000.0  # m/s (typical RMS velocity)
    t_out_s = 0.8  # 800 ms output time

    # Simulate a trace at midpoint near Y=383 output point
    ox = 6400.0  # Middle of X range

    print("\nFor output point at X=6400, t=800ms, velocity=3000 m/s:")
    print(f"Grid spacing: dx={DX}, dy={DY}")
    print(f"Dominant frequency: {DOMINANT_FREQ} Hz")

    print("\nY index | Y coord | Midpoint offset | t_travel (s) | AA weight")
    print("-" * 70)

    for iy in [382, 383, 384, 385]:
        oy = iy * DY

        # Simulate a trace with midpoint 200m away
        mx = ox + 150  # Midpoint slightly offset in X
        my = oy + 50   # Midpoint slightly offset in Y

        # Compute DSR traveltime (simplified - assume straight ray for demo)
        half_offset = 100  # Half offset distance
        t_half_sq = (t_out_s / 2) ** 2
        inv_v_sq = 1.0 / (velocity ** 2)
        ds = np.sqrt((ox - mx - half_offset)**2 + (oy - my)**2)
        dr = np.sqrt((ox - mx + half_offset)**2 + (oy - my)**2)
        t_travel = np.sqrt(t_half_sq + ds**2 * inv_v_sq) + np.sqrt(t_half_sq + dr**2 * inv_v_sq)

        midpoint_offset = np.sqrt((mx - ox)**2 + (my - oy)**2)

        aa_weight = compute_aa_weight_py(ox, oy, mx, my, velocity, t_travel, DX, DY, DOMINANT_FREQ)

        marker = ""
        if iy == 383:
            marker = " <- Last in full tile"
        elif iy == 384:
            marker = " <- First in partial tile"

        print(f"  {iy:3d}   | {oy:6.1f} |     {midpoint_offset:6.1f}     |   {t_travel:.4f}   |   {aa_weight:.4f}{marker}")

    # The AA weight should be the same for adjacent Y indices if:
    # - Same velocity
    # - Same output time
    # - Same trace geometry (relative positions)

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The AA weight depends on:
  - Local dip (mx-ox, my-oy)
  - Velocity
  - Traveltime
  - Grid spacing (dx, dy)

All these should be consistent across tiles since:
  - Grid spacing is the same (just verified: dy=12.5 for all tiles)
  - Same velocity model
  - Same trace geometries (relative positions)

If there's still an amplitude drop, the issue might be:
  1. Different set of traces being loaded for partial tiles
  2. Spatial query returning fewer traces
  3. Some other tile-boundary effect
""")

    # Check fold difference more carefully
    print("\n--- Fold Analysis at Boundary ---")
    for iy in [383, 384]:
        fold_slice = fold[:, iy, t_range]
        print(f"Y={iy}: fold min={fold_slice.min()}, max={fold_slice.max()}, "
              f"mean={fold_slice.mean():.1f}, std={fold_slice.std():.1f}")

    fold_383 = fold[:, 383, t_range].mean()
    fold_384 = fold[:, 384, t_range].mean()
    if fold_383 > 0:
        fold_ratio = fold_384 / fold_383
        print(f"\nFold ratio (Y=384 / Y=383): {fold_ratio:.4f}")
        if fold_ratio < 0.95:
            print("WARNING: Fold drops at boundary - fewer traces contributing!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_aa_at_boundary()
