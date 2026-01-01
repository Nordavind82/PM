#!/usr/bin/env python3
"""
Compare diagnostic tile-by-tile output with executor output for bin 25.

This directly compares the migrated_stack.zarr files.
"""

import numpy as np
import zarr
from pathlib import Path

# Paths
DIAGNOSTIC_DIR = Path("/tmp/pstm_diagnostic_bin25")
EXECUTOR_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/migration_bin_25")


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Output Comparison")
    print("=" * 70)

    # Load diagnostic output
    diag_path = DIAGNOSTIC_DIR / "migrated_stack.zarr"
    exec_path = EXECUTOR_DIR / "migrated_stack.zarr"

    print(f"\nDiagnostic path: {diag_path}")
    print(f"Executor path: {exec_path}")

    diag_z = zarr.open_array(str(diag_path), mode='r')
    exec_z = zarr.open_array(str(exec_path), mode='r')

    print(f"\nDiagnostic shape: {diag_z.shape}")
    print(f"Executor shape: {exec_z.shape}")

    # Also load fold
    diag_fold_path = DIAGNOSTIC_DIR / "fold.zarr"
    exec_fold_path = EXECUTOR_DIR / "fold.zarr"

    diag_fold = zarr.open_array(str(diag_fold_path), mode='r')
    exec_fold = zarr.open_array(str(exec_fold_path), mode='r')

    print(f"\nDiagnostic fold shape: {diag_fold.shape}")
    print(f"Executor fold shape: {exec_fold.shape}")

    # Compare shapes
    if diag_z.shape != exec_z.shape:
        print("\nWARNING: Shapes differ!")

    # Load data into memory for comparison
    print("\nLoading data...")
    diag_data = np.array(diag_z[:])
    exec_data = np.array(exec_z[:])
    diag_fold_data = np.array(diag_fold[:])
    exec_fold_data = np.array(exec_fold[:])

    print(f"Diagnostic dtype: {diag_data.dtype}")
    print(f"Executor dtype: {exec_data.dtype}")

    # Global statistics
    print("\n--- Global Statistics ---")
    print(f"Diagnostic: min={diag_data.min():.6f}, max={diag_data.max():.6f}, RMS={np.sqrt(np.mean(diag_data**2)):.6f}")
    print(f"Executor:   min={exec_data.min():.6f}, max={exec_data.max():.6f}, RMS={np.sqrt(np.mean(exec_data**2)):.6f}")

    # Ratio of total energy
    diag_energy = np.sum(diag_data**2)
    exec_energy = np.sum(exec_data**2)
    print(f"\nEnergy ratio (exec/diag): {exec_energy/diag_energy:.4f}")

    # Fold comparison
    print("\n--- Fold Comparison ---")
    fold_match = np.allclose(diag_fold_data, exec_fold_data)
    print(f"Fold arrays match: {fold_match}")
    if not fold_match:
        diff_fold = np.abs(diag_fold_data - exec_fold_data)
        print(f"  Max fold diff: {diff_fold.max()}")
        print(f"  Mean fold diff: {diff_fold.mean():.2f}")
        print(f"  Locations with diff > 0: {np.count_nonzero(diff_fold)}")
    else:
        print("  Fold values are IDENTICAL")

    # Check if there's a simple scaling relationship
    print("\n--- Scaling Analysis ---")

    # Find non-zero locations
    mask = (diag_fold_data > 0) & (diag_data != 0)
    if np.any(mask):
        # Calculate ratio at non-zero locations
        ratio = exec_data[mask] / diag_data[mask]
        ratio = ratio[np.isfinite(ratio)]

        if len(ratio) > 0:
            print(f"Ratio (exec/diag) at non-zero points:")
            print(f"  Mean ratio: {np.mean(ratio):.6f}")
            print(f"  Median ratio: {np.median(ratio):.6f}")
            print(f"  Std ratio: {np.std(ratio):.6f}")
            print(f"  Min ratio: {np.min(ratio):.6f}")
            print(f"  Max ratio: {np.max(ratio):.6f}")

            # If ratio is consistent, outputs are just scaled
            if np.std(ratio) < 0.01 * np.abs(np.mean(ratio)):
                print(f"\nOUTPUTS ARE SCALED by factor of {np.mean(ratio):.4f}")
            else:
                print("\nRatio is NOT consistent - not a simple scaling")

    # Y boundary analysis (Y=383 vs Y=384)
    print("\n--- Boundary Analysis (Y=383 vs Y=384) ---")
    t_range = slice(300, 500)

    # Y=383 (last row of full tile)
    diag_y383 = diag_data[:, 383, t_range]
    exec_y383 = exec_data[:, 383, t_range]
    diag_y383_rms = np.sqrt(np.mean(diag_y383**2))
    exec_y383_rms = np.sqrt(np.mean(exec_y383**2))

    # Y=384 (first row of partial tile)
    diag_y384 = diag_data[:, 384, t_range]
    exec_y384 = exec_data[:, 384, t_range]
    diag_y384_rms = np.sqrt(np.mean(diag_y384**2))
    exec_y384_rms = np.sqrt(np.mean(exec_y384**2))

    print(f"Y=383 (full tile boundary):")
    print(f"  Diagnostic RMS: {diag_y383_rms:.8f}")
    print(f"  Executor RMS:   {exec_y383_rms:.8f}")
    print(f"  Ratio (exec/diag): {exec_y383_rms/diag_y383_rms:.4f}")

    print(f"\nY=384 (partial tile boundary):")
    print(f"  Diagnostic RMS: {diag_y384_rms:.8f}")
    print(f"  Executor RMS:   {exec_y384_rms:.8f}")
    print(f"  Ratio (exec/diag): {exec_y384_rms/diag_y384_rms:.4f}")

    # Boundary ratio in each dataset
    diag_boundary_ratio = diag_y384_rms / diag_y383_rms
    exec_boundary_ratio = exec_y384_rms / exec_y383_rms

    print(f"\nBoundary ratio (Y=384/Y=383):")
    print(f"  Diagnostic: {diag_boundary_ratio:.4f}")
    print(f"  Executor:   {exec_boundary_ratio:.4f}")

    # Sample-by-sample comparison at specific locations
    print("\n--- Sample Comparison at (x=64, t=400) ---")
    t_idx = 200  # t=400ms at 2ms sample
    for iy in [380, 381, 382, 383, 384, 385, 386, 387]:
        diag_val = diag_data[64, iy, t_idx]
        exec_val = exec_data[64, iy, t_idx]
        diag_fold_val = diag_fold_data[64, iy, t_idx]
        exec_fold_val = exec_fold_data[64, iy, t_idx]
        ratio = exec_val / diag_val if diag_val != 0 else float('nan')
        print(f"  Y={iy}: diag={diag_val:12.6f}, exec={exec_val:12.6f}, ratio={ratio:.4f}, fold_diag={diag_fold_val}, fold_exec={exec_fold_val}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
