#!/usr/bin/env python3
"""
Verify fold values are truly identical between diagnostic and executor.
"""

import numpy as np
import zarr
from pathlib import Path

DIAGNOSTIC_DIR = Path("/tmp/pstm_diagnostic_bin25")
EXECUTOR_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/migration_bin_25")


def main():
    print("=" * 70)
    print("FOLD EXACT COMPARISON")
    print("=" * 70)

    diag_fold = zarr.open_array(str(DIAGNOSTIC_DIR / "fold.zarr"), mode='r')
    exec_fold = zarr.open_array(str(EXECUTOR_DIR / "fold.zarr"), mode='r')

    print(f"Diagnostic fold shape: {diag_fold.shape}")
    print(f"Executor fold shape: {exec_fold.shape}")

    # Load into memory
    df = np.array(diag_fold[:])
    ef = np.array(exec_fold[:])

    print(f"\nDiagnostic fold: min={df.min()}, max={df.max()}, mean={df.mean():.2f}")
    print(f"Executor fold:   min={ef.min()}, max={ef.max()}, mean={ef.mean():.2f}")

    # Exact comparison
    exact_match = np.array_equal(df, ef)
    print(f"\nExact match (np.array_equal): {exact_match}")

    if not exact_match:
        diff = df - ef
        print(f"Difference: min={diff.min()}, max={diff.max()}, mean={diff.mean():.2f}")
        print(f"Non-zero differences: {np.count_nonzero(diff)}")

        # Show where they differ
        where_diff = np.where(diff != 0)
        if len(where_diff[0]) > 0:
            print(f"\nFirst 10 differing locations:")
            for i in range(min(10, len(where_diff[0]))):
                ix, iy, it = where_diff[0][i], where_diff[1][i], where_diff[2][i]
                print(f"  ({ix}, {iy}, {it}): diag={df[ix, iy, it]}, exec={ef[ix, iy, it]}, diff={diff[ix, iy, it]}")

    # Compare at specific rows
    print("\n--- Row-by-row comparison (Y axis) ---")
    for iy in [380, 381, 382, 383, 384, 385, 386, 387]:
        diag_row_sum = df[:, iy, :].sum()
        exec_row_sum = ef[:, iy, :].sum()
        match = "MATCH" if diag_row_sum == exec_row_sum else "DIFFER"
        print(f"  Y={iy}: diag_sum={diag_row_sum}, exec_sum={exec_row_sum} [{match}]")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
