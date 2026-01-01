#!/usr/bin/env python3
"""
Compare fresh diagnostic and executor outputs.
"""

import numpy as np
import zarr
from pathlib import Path

DIAGNOSTIC_DIR = Path("/tmp/pstm_fresh_diagnostic_bin25")
EXECUTOR_DIR = Path("/tmp/pstm_fixed_executor/migration_bin_25")


def main():
    print("=" * 70)
    print("FRESH OUTPUT COMPARISON")
    print("=" * 70)

    # Load both outputs
    diag_stack = zarr.open_array(str(DIAGNOSTIC_DIR / "migrated_stack.zarr"), mode='r')
    exec_stack = zarr.open_array(str(EXECUTOR_DIR / "migrated_stack.zarr"), mode='r')

    diag_fold = zarr.open_array(str(DIAGNOSTIC_DIR / "fold.zarr"), mode='r')
    exec_fold = zarr.open_array(str(EXECUTOR_DIR / "fold.zarr"), mode='r')

    print(f"\nDiagnostic stack shape: {diag_stack.shape}")
    print(f"Executor stack shape: {exec_stack.shape}")

    # Load into memory
    ds = np.array(diag_stack[:])
    es = np.array(exec_stack[:])
    df = np.array(diag_fold[:])
    ef = np.array(exec_fold[:])

    # Global statistics
    print("\n--- Global Statistics ---")
    print(f"Diagnostic: min={ds.min():.6f}, max={ds.max():.6f}, RMS={np.sqrt(np.mean(ds**2)):.6f}")
    print(f"Executor:   min={es.min():.6f}, max={es.max():.6f}, RMS={np.sqrt(np.mean(es**2)):.6f}")

    # Fold comparison
    fold_match = np.array_equal(df, ef)
    print(f"\nFold exact match: {fold_match}")

    # Image comparison
    diff = ds - es
    max_diff = np.abs(diff).max()
    mean_diff = np.abs(diff).mean()
    print(f"\nImage difference: max={max_diff:.8f}, mean={mean_diff:.8f}")

    if max_diff < 1e-6:
        print("\nOUTPUTS ARE IDENTICAL!")
    else:
        print("\nOUTPUTS DIFFER!")

        # Energy comparison
        diag_energy = np.sum(ds**2)
        exec_energy = np.sum(es**2)
        print(f"\nEnergy ratio (exec/diag): {exec_energy/diag_energy:.4f}")

        # Boundary comparison
        print("\n--- Boundary Analysis ---")
        t_range = slice(300, 500)

        for iy in [383, 384]:
            diag_rms = np.sqrt(np.mean(ds[:, iy, t_range]**2))
            exec_rms = np.sqrt(np.mean(es[:, iy, t_range]**2))
            print(f"Y={iy}: diag_rms={diag_rms:.8f}, exec_rms={exec_rms:.8f}, ratio={exec_rms/diag_rms:.4f}")

        # Sample comparison
        print("\nSample values at (x=64, t=400):")
        for iy in [383, 384]:
            it = 200
            print(f"  Y={iy}: diag={ds[64, iy, it]:.8f}, exec={es[64, iy, it]:.8f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
