#!/usr/bin/env python3
"""
Minimal test to compare Metal kernel output for full vs partial tiles.

Creates synthetic input data and runs the kernel on:
1. Full tile (128x128)
2. Partial tile (128x43)

Compares output for overlapping points.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pstm.kernels.metal_compiled import CompiledMetalKernel, MigrationParams
from pstm.kernels.base import OutputTile, VelocitySlice, TraceBlock, KernelConfig


def run_test():
    print("=" * 70)
    print("DIAGNOSTIC: Kernel Output Comparison")
    print("=" * 70)

    # Kernel config
    config = KernelConfig(
        max_aperture_m=3000.0,
        min_aperture_m=500.0,
        taper_fraction=0.1,
        max_dip_degrees=65.0,
        apply_spreading=True,
        apply_obliquity=True,
        aa_enabled=False,
    )

    # Create kernel
    kernel = CompiledMetalKernel(use_simd=True)
    kernel.initialize(config)

    # Fixed parameters for both tests
    n_traces = 100
    n_samples = 501
    nt = 501
    sample_rate_ms = 4.0
    start_time_ms = 0.0

    # Create synthetic trace data (same for both tiles)
    np.random.seed(42)
    amplitudes = np.random.randn(n_samples, n_traces).astype(np.float32) * 0.01

    # Place traces in a small region
    center_x, center_y = 625000.0, 5115000.0
    source_x = np.random.uniform(center_x - 1000, center_x + 1000, n_traces).astype(np.float32)
    source_y = np.random.uniform(center_y - 1000, center_y + 1000, n_traces).astype(np.float32)
    receiver_x = source_x + np.random.uniform(-500, 500, n_traces).astype(np.float32)
    receiver_y = source_y + np.random.uniform(-500, 500, n_traces).astype(np.float32)
    midpoint_x = (source_x + receiver_x) / 2
    midpoint_y = (source_y + receiver_y) / 2
    offset = np.sqrt((receiver_x - source_x)**2 + (receiver_y - source_y)**2)

    traces = TraceBlock(
        amplitudes=amplitudes,
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        midpoint_x=midpoint_x,
        midpoint_y=midpoint_y,
        offset=offset,
        sample_rate_ms=sample_rate_ms,
        start_time_ms=start_time_ms,
    )

    # Define output grids
    # Full tile: 128 x 128
    nx_full = 128
    ny_full = 128

    # Partial tile: 128 x 43
    nx_partial = 128
    ny_partial = 43

    # Create velocity (3D - varies slightly with position)
    velocity_val = 3000.0  # m/s base velocity
    # Create 3D velocity that varies slightly with position
    # For full tile: shape (128, 128, nt)
    # For partial tile: shape (128, 43, nt)
    vrms_full = np.full((nx_full, ny_full, nt), velocity_val, dtype=np.float64)
    # Add small lateral variation
    for i in range(nx_full):
        for j in range(ny_full):
            vrms_full[i, j, :] += (i + j) * 0.5  # Small variation

    vrms_partial = vrms_full[:, :ny_partial, :]  # Extract partial region

    velocity_full = VelocitySlice(vrms=vrms_full, is_1d=False)
    velocity_partial = VelocitySlice(vrms=vrms_partial, is_1d=False)

    # Create coordinate grids centered on traces
    x_range = (center_x - 500, center_x + 500)
    y_range = (center_y - 500, center_y + 500)

    # Full tile
    x_axis_full = np.linspace(x_range[0], x_range[1], nx_full)
    y_axis_full = np.linspace(y_range[0], y_range[1], ny_full)
    t_axis = np.arange(0, nt * sample_rate_ms, sample_rate_ms)

    # Create 2D coordinate grids
    X_full, Y_full = np.meshgrid(x_axis_full, y_axis_full, indexing='ij')

    output_full = OutputTile(
        image=np.zeros((nx_full, ny_full, nt), dtype=np.float64),
        fold=np.zeros((nx_full, ny_full, nt), dtype=np.int32),
        x_axis=x_axis_full,
        y_axis=y_axis_full,
        t_axis_ms=t_axis,
        x_grid=X_full,
        y_grid=Y_full,
    )

    # Partial tile - same X range but truncated Y
    x_axis_partial = x_axis_full  # Same
    y_axis_partial = np.linspace(y_range[0], y_range[0] + (y_range[1] - y_range[0]) * ny_partial / ny_full, ny_partial)
    X_partial, Y_partial = np.meshgrid(x_axis_partial, y_axis_partial, indexing='ij')

    output_partial = OutputTile(
        image=np.zeros((nx_partial, ny_partial, nt), dtype=np.float64),
        fold=np.zeros((nx_partial, ny_partial, nt), dtype=np.int32),
        x_axis=x_axis_partial,
        y_axis=y_axis_partial,
        t_axis_ms=t_axis,
        x_grid=X_partial,
        y_grid=Y_partial,
    )

    print(f"\nFull tile shape: ({nx_full}, {ny_full}, {nt})")
    print(f"Partial tile shape: ({nx_partial}, {ny_partial}, {nt})")
    print(f"Traces: {n_traces}")
    print(f"Velocity: {velocity_val} m/s (constant)")

    # Run kernel on full tile
    print("\n--- Running kernel on full tile ---")
    metrics_full = kernel.migrate_tile(traces, output_full, velocity_full, config)
    print(f"  Completed in {metrics_full.compute_time_s:.3f}s")

    # Run kernel on partial tile
    print("\n--- Running kernel on partial tile ---")
    metrics_partial = kernel.migrate_tile(traces, output_partial, velocity_partial, config)
    print(f"  Completed in {metrics_partial.compute_time_s:.3f}s")

    # Compare outputs for overlapping region
    # The partial tile covers y indices 0:43 which should match full tile's 0:43
    print("\n--- Comparing outputs ---")

    # Get overlapping regions
    full_overlap = output_full.image[:, :ny_partial, :]
    partial_all = output_partial.image

    # Check if coordinates match
    coord_match = np.allclose(X_full[:, :ny_partial], X_partial) and np.allclose(Y_full[:, :ny_partial], Y_partial)
    print(f"Coordinates match: {coord_match}")

    # Compare amplitudes
    diff = full_overlap - partial_all
    max_diff = np.abs(diff).max()
    rms_diff = np.sqrt(np.mean(diff**2))

    full_rms = np.sqrt(np.mean(full_overlap**2))
    partial_rms = np.sqrt(np.mean(partial_all**2))

    print(f"\nFull tile RMS (first 43 Y): {full_rms:.8f}")
    print(f"Partial tile RMS: {partial_rms:.8f}")
    print(f"Ratio: {partial_rms / full_rms if full_rms > 0 else 'N/A':.4f}")
    print(f"Max difference: {max_diff:.8f}")
    print(f"RMS difference: {rms_diff:.8f}")

    # Compare fold
    fold_full_overlap = output_full.fold[:, :ny_partial, :]
    fold_partial = output_partial.fold

    fold_match = np.array_equal(fold_full_overlap, fold_partial)
    print(f"Fold matches: {fold_match}")

    if not fold_match:
        diff_count = np.sum(fold_full_overlap != fold_partial)
        print(f"  Different fold values: {diff_count} out of {fold_full_overlap.size}")

    # Check specific points
    print("\n--- Specific point comparison ---")
    test_points = [(64, 21, 200), (10, 10, 150), (100, 30, 250)]
    for ix, iy, it in test_points:
        if iy < ny_partial:
            full_val = output_full.image[ix, iy, it]
            partial_val = output_partial.image[ix, iy, it]
            full_fold = output_full.fold[ix, iy, it]
            partial_fold = output_partial.fold[ix, iy, it]
            print(f"  ({ix},{iy},{it}): full={full_val:.6f} partial={partial_val:.6f} fold_full={full_fold} fold_partial={partial_fold}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
