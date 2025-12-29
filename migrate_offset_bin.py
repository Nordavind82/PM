#!/usr/bin/env python3
"""
Common Offset PSTM Migration for a single offset bin.
Uses picked velocity function V(t) = 1783 + 0.30*t + 0.00131*t²
"""

import numpy as np
import zarr
import pandas as pd
from pathlib import Path
from numba import njit, prange
import time
import sys

# Parameters
INPUT_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_offset_bins/migrated")

# Velocity function coefficients: V(t) = V0 + k1*t + k2*t²
V0 = 1783.0      # m/s at t=0
K1 = 0.30        # m/s per ms
K2 = 0.00131     # m/s per ms²

# Migration parameters
APERTURE = 1500.0        # meters
MAX_STRETCH = 2.0        # stretch mute factor
DT_MS = 2.0              # sample rate
N_SAMPLES_OUT = 1001     # output samples (0-2000ms)

@njit
def velocity_at_time(t_ms):
    """Time-varying velocity in m/s."""
    return V0 + K1 * t_ms + K2 * t_ms * t_ms

@njit(parallel=True, fastmath=True)
def migrate_common_offset(traces, midx, midy, offsets,
                          target_x, target_y, n_il, n_xl,
                          dt_ms, n_samples_out, n_samples_in,
                          max_stretch, aperture):
    """
    Kirchhoff PSTM for common offset data.

    traces: (n_samples, n_traces) input traces
    midx, midy: midpoint coordinates for each trace
    offsets: offset for each trace
    target_x, target_y: 1D arrays of output grid coordinates
    """
    n_traces = traces.shape[1]
    output = np.zeros((n_samples_out, n_il, n_xl), dtype=np.float32)
    fold = np.zeros((n_samples_out, n_il, n_xl), dtype=np.int32)

    # Process each output location
    for il_idx in prange(n_il):
        tx = target_x[il_idx]

        for xl_idx in range(n_xl):
            ty = target_y[xl_idx]

            # Find traces within aperture
            for tr_idx in range(n_traces):
                dist = np.sqrt((midx[tr_idx] - tx)**2 + (midy[tr_idx] - ty)**2)

                if dist > aperture:
                    continue

                h = offsets[tr_idx] / 2.0  # half offset

                # Process each output time
                for t_out_idx in range(n_samples_out):
                    t_out_ms = t_out_idx * dt_ms

                    if t_out_ms < 1.0:
                        continue

                    v = velocity_at_time(t_out_ms)

                    # Travel time calculation (diffraction hyperbola)
                    # t² = t0² + (2*dist/v)² for zero offset at distance dist
                    # Plus offset contribution
                    t0_sq = t_out_ms * t_out_ms
                    offset_term = (2.0 * h / v) ** 2
                    dist_term = (2.0 * dist / v) ** 2

                    t_in_sq = t0_sq + offset_term + dist_term
                    t_in_ms = np.sqrt(t_in_sq)

                    # Stretch mute
                    if t_in_ms > max_stretch * t_out_ms:
                        continue

                    # Get input sample
                    t_in_sample = t_in_ms / dt_ms
                    if t_in_sample < 0 or t_in_sample >= n_samples_in - 1:
                        continue

                    # Linear interpolation
                    idx_lo = int(t_in_sample)
                    idx_hi = idx_lo + 1
                    frac = t_in_sample - idx_lo

                    amp = traces[idx_lo, tr_idx] * (1.0 - frac) + traces[idx_hi, tr_idx] * frac

                    # Obliquity factor (simple cosine weighting)
                    obliquity = t_out_ms / t_in_ms if t_in_ms > 0 else 1.0

                    output[t_out_idx, il_idx, xl_idx] += amp * obliquity
                    fold[t_out_idx, il_idx, xl_idx] += 1

    # Normalize by fold
    for il_idx in prange(n_il):
        for xl_idx in range(n_xl):
            for t_idx in range(n_samples_out):
                if fold[t_idx, il_idx, xl_idx] > 0:
                    output[t_idx, il_idx, xl_idx] /= fold[t_idx, il_idx, xl_idx]

    return output, fold

def main():
    bin_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print("=" * 60)
    print(f"Common Offset PSTM Migration - Bin {bin_idx}")
    print("=" * 60)

    bin_dir = INPUT_DIR / f"offset_bin_{bin_idx:02d}"

    # Load data
    print("Loading data...")
    t0 = time.time()
    traces = zarr.open(bin_dir / "traces.zarr", mode='r')[:]
    headers = pd.read_parquet(bin_dir / "headers.parquet")
    print(f"  {traces.shape[1]:,} traces, {traces.shape[0]} samples in {time.time()-t0:.1f}s")

    # Get coordinates (convert from cm to m)
    midx = ((headers['source_x'].values + headers['receiver_x'].values) / 2.0) / 100.0
    midy = ((headers['source_y'].values + headers['receiver_y'].values) / 2.0) / 100.0
    offsets = headers['offset'].values

    print(f"  Offset range: {offsets.min():.1f} - {offsets.max():.1f} m")
    print(f"  Midpoint X: {midx.min():.1f} - {midx.max():.1f} m")
    print(f"  Midpoint Y: {midy.min():.1f} - {midy.max():.1f} m")

    # Define output grid based on data extent
    x_min, x_max = midx.min(), midx.max()
    y_min, y_max = midy.min(), midy.max()

    # Use 25m grid spacing
    grid_spacing = 25.0
    target_x = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    target_y = np.arange(y_min, y_max + grid_spacing, grid_spacing)

    n_il = len(target_x)
    n_xl = len(target_y)

    print(f"\nOutput grid:")
    print(f"  {n_il} inlines x {n_xl} crosslines = {n_il * n_xl:,} traces")
    print(f"  Grid spacing: {grid_spacing}m")
    print(f"  X range: {target_x[0]:.1f} - {target_x[-1]:.1f} m")
    print(f"  Y range: {target_y[0]:.1f} - {target_y[-1]:.1f} m")

    print(f"\nMigration parameters:")
    print(f"  Velocity: V(t) = {V0} + {K1}*t + {K2}*t²")
    print(f"  Aperture: {APERTURE}m")
    print(f"  Stretch mute: {MAX_STRETCH}x")
    print(f"  Output: {N_SAMPLES_OUT} samples at {DT_MS}ms")

    # Run migration
    print("\nMigrating...")
    t0 = time.time()

    migrated, fold = migrate_common_offset(
        traces.astype(np.float32),
        midx.astype(np.float64),
        midy.astype(np.float64),
        offsets.astype(np.float64),
        target_x.astype(np.float64),
        target_y.astype(np.float64),
        n_il, n_xl,
        DT_MS, N_SAMPLES_OUT, traces.shape[0],
        MAX_STRETCH, APERTURE
    )

    elapsed = time.time() - t0
    print(f"  Migration complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"offset_bin_{bin_idx:02d}"
    output_path.mkdir(exist_ok=True)

    print("\nSaving output...")
    np.savez_compressed(
        output_path / "migrated.npz",
        data=migrated,
        fold=fold,
        target_x=target_x,
        target_y=target_y,
        dt_ms=DT_MS,
        velocity_coeffs=np.array([V0, K1, K2]),
        aperture=APERTURE,
        offset_range=np.array([offsets.min(), offsets.max()])
    )

    # Quick stats
    print(f"\nOutput statistics:")
    print(f"  Shape: {migrated.shape}")
    print(f"  Non-zero samples: {(migrated != 0).sum():,} / {migrated.size:,}")
    print(f"  Max fold: {fold.max()}")
    print(f"  Data range: {migrated.min():.4f} to {migrated.max():.4f}")

    print(f"\nOutput saved to: {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()
