#!/usr/bin/env python3
"""
Diagnostic: Analyze AA weight distribution for traces in different tiles.

This checks if the AA weight behavior differs between full and partial tiles
due to differences in trace geometry (offset, dip).
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pyarrow.parquet as pq
from pstm.data.spatial_index import SpatialIndex

# Constants
VELOCITY = 3000.0  # m/s typical
T_OUTPUT_S = 0.8  # 800 ms output time
DX = 25.0
DY = 12.5
DOMINANT_FREQ = 30.0  # Hz

DATA_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m/offset_bin_25")


def compute_aa_weight(ox, oy, mx, my, velocity, t_travel, dx, dy, dominant_freq):
    """
    Compute AA weight (vectorized for arrays).
    """
    # Avoid division issues for small traveltimes
    t_safe = np.maximum(t_travel, 0.001)

    # Compute local dip
    denom = velocity * velocity * t_safe * 0.5
    dip_x = (mx - ox) / denom
    dip_y = (my - oy) / denom

    # sin(theta)
    sin_theta_x = np.minimum(np.abs(dip_x) * velocity * 0.5, 1.0)
    sin_theta_y = np.minimum(np.abs(dip_y) * velocity * 0.5, 1.0)
    sin_theta = np.maximum(sin_theta_x, sin_theta_y)

    # For very small dip, weight = 1
    weight = np.ones_like(sin_theta)

    # Compute f_max for significant dips
    mask = sin_theta >= 0.01
    grid_spacing = max(dx, dy)
    f_max = velocity / (4.0 * grid_spacing * sin_theta[mask])
    f_max = np.minimum(f_max, 500.0)

    weight[mask] = np.maximum(0.0, 1.0 - dominant_freq / f_max)

    return weight


def main():
    print("=" * 70)
    print("AA WEIGHT TRACE ANALYSIS")
    print("=" * 70)

    # Load headers
    parquet_file = DATA_DIR / "headers.parquet"
    print(f"\nLoading parquet: {parquet_file}")
    table = pq.read_table(parquet_file)

    # Compute midpoint from source/receiver
    source_x_all = np.array(table['source_x'].to_numpy(), dtype=np.float64)
    source_y_all = np.array(table['source_y'].to_numpy(), dtype=np.float64)
    receiver_x_all = np.array(table['receiver_x'].to_numpy(), dtype=np.float64)
    receiver_y_all = np.array(table['receiver_y'].to_numpy(), dtype=np.float64)

    midpoint_x = (source_x_all + receiver_x_all) / 2.0
    midpoint_y = (source_y_all + receiver_y_all) / 2.0
    n_traces = len(midpoint_x)

    # Build spatial index
    all_trace_indices = np.arange(n_traces, dtype=np.int64)
    spatial_idx = SpatialIndex.build(all_trace_indices, midpoint_x, midpoint_y)
    print(f"Indexed {n_traces:,} traces")

    # Define test regions
    # Full tile region (iy=2): Y = 256*12.5 to 383*12.5 = 3200 to 4787.5
    # Partial tile region (iy=3): Y = 384*12.5 to 426*12.5 = 4800 to 5325

    regions = [
        ("Full tile (iy=2)", 3200.0, 4787.5),
        ("Partial tile (iy=3)", 4800.0, 5325.0),
    ]

    # X center (middle of survey)
    x_center = 6400.0

    # Output time
    t0_half = T_OUTPUT_S / 2.0
    t0_half_sq = t0_half ** 2
    inv_v_sq = 1.0 / (VELOCITY ** 2)

    aperture = 2000.0

    print("\nAnalyzing AA weights for traces in different Y regions...")
    print(f"X center: {x_center} m, aperture: {aperture} m, t_output: {T_OUTPUT_S*1000} ms")
    print(f"dx={DX}, dy={DY}, dominant_freq={DOMINANT_FREQ} Hz")

    for region_name, y_min, y_max in regions:
        print(f"\n{'='*60}")
        print(f"Region: {region_name} (Y = {y_min} to {y_max})")
        print(f"{'='*60}")

        # Output points to analyze
        y_points = [y_min + 12.5, (y_min + y_max) / 2, y_max - 12.5]

        for oy in y_points:
            ox = x_center

            # Query traces in aperture
            trace_indices = spatial_idx.query_radius(ox, oy, aperture)

            if len(trace_indices) == 0:
                print(f"\n  Output point ({ox:.0f}, {oy:.1f}): No traces in aperture")
                continue

            # Get trace geometry
            mx = midpoint_x[trace_indices]
            my = midpoint_y[trace_indices]

            source_x = source_x_all[trace_indices]
            source_y = source_y_all[trace_indices]
            receiver_x = receiver_x_all[trace_indices]
            receiver_y = receiver_y_all[trace_indices]

            # Compute distances
            ds = np.sqrt((ox - source_x)**2 + (oy - source_y)**2)
            dr = np.sqrt((ox - receiver_x)**2 + (oy - receiver_y)**2)

            # DSR traveltime
            t_travel = np.sqrt(t0_half_sq + ds**2 * inv_v_sq) + np.sqrt(t0_half_sq + dr**2 * inv_v_sq)

            # Compute AA weights
            aa_weights = compute_aa_weight(ox, oy, mx, my, VELOCITY, t_travel, DX, DY, DOMINANT_FREQ)

            # Midpoint distance
            dm = np.sqrt((ox - mx)**2 + (oy - my)**2)

            print(f"\n  Output point ({ox:.0f}, {oy:.1f}):")
            print(f"    Traces in aperture: {len(trace_indices):,}")
            print(f"    Midpoint distances: min={dm.min():.0f}m, max={dm.max():.0f}m, mean={dm.mean():.0f}m")
            print(f"    AA weights: min={aa_weights.min():.4f}, max={aa_weights.max():.4f}, mean={aa_weights.mean():.4f}")
            print(f"    AA weight > 0.5: {(aa_weights > 0.5).sum():,} ({100*(aa_weights > 0.5).mean():.1f}%)")
            print(f"    AA weight == 0: {(aa_weights == 0).sum():,} ({100*(aa_weights == 0).mean():.1f}%)")

    # Cross-comparison at the boundary
    print("\n" + "=" * 70)
    print("CROSS-BOUNDARY COMPARISON")
    print("=" * 70)

    y_vals = [4787.5, 4800.0]  # Last point in full tile, first in partial tile

    for oy in y_vals:
        ox = x_center
        trace_indices = spatial_idx.query_radius(ox, oy, aperture)

        if len(trace_indices) == 0:
            continue

        mx = midpoint_x[trace_indices]
        my = midpoint_y[trace_indices]
        source_x = source_x_all[trace_indices]
        source_y = source_y_all[trace_indices]
        receiver_x = receiver_x_all[trace_indices]
        receiver_y = receiver_y_all[trace_indices]

        ds = np.sqrt((ox - source_x)**2 + (oy - source_y)**2)
        dr = np.sqrt((ox - receiver_x)**2 + (oy - receiver_y)**2)
        t_travel = np.sqrt(t0_half_sq + ds**2 * inv_v_sq) + np.sqrt(t0_half_sq + dr**2 * inv_v_sq)

        aa_weights = compute_aa_weight(ox, oy, mx, my, VELOCITY, t_travel, DX, DY, DOMINANT_FREQ)

        tile_label = "Full tile" if oy < 4800 else "Partial tile"
        print(f"\nY = {oy:.1f} ({tile_label}):")
        print(f"  Traces: {len(trace_indices):,}")
        print(f"  Mean AA weight: {aa_weights.mean():.4f}")
        print(f"  Median AA weight: {np.median(aa_weights):.4f}")
        print(f"  Sum of weighted traces: {aa_weights.sum():.1f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
