#!/usr/bin/env python3
"""
Diagnostic Step 1: Verify Trace-Header Mapping

Check if bin_trace_idx correctly maps headers to zarr trace storage.
This is critical - wrong mapping would cause completely wrong geometry.

Tests:
1. Verify index range matches storage dimensions
2. Check for duplicate or missing indices
3. Sample traces and verify geometry makes sense
4. Cross-check with original trace_index from all_headers
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
import zarr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

BIN_NUM = 10
COMMON_OFFSET_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/diagnostic_qc")


def main():
    print("\n" + "="*70)
    print("STEP 1: Verify Trace-Header Mapping")
    print("="*70)

    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{BIN_NUM:02d}"
    traces_path = bin_dir / "traces.zarr"
    headers_path = bin_dir / "headers.parquet"

    # Load data
    print(f"\n[1.1] Loading data...")
    traces_store = zarr.open_array(traces_path, mode='r')
    df = pl.read_parquet(headers_path)

    print(f"      Traces zarr shape: {traces_store.shape}")
    print(f"      Headers count: {len(df)}")

    # Determine storage orientation
    if traces_store.shape[0] < traces_store.shape[1]:
        n_samples, n_traces_storage = traces_store.shape
        transposed = True
        print(f"      Storage: TRANSPOSED (n_samples, n_traces) = ({n_samples}, {n_traces_storage})")
    else:
        n_traces_storage, n_samples = traces_store.shape
        transposed = False
        print(f"      Storage: STANDARD (n_traces, n_samples) = ({n_traces_storage}, {n_samples})")

    # =========================================================================
    # Test 1.2: Check bin_trace_idx range
    # =========================================================================
    print(f"\n[1.2] Checking bin_trace_idx range...")

    bin_trace_idx = df['bin_trace_idx'].to_numpy()
    print(f"      bin_trace_idx min: {bin_trace_idx.min()}")
    print(f"      bin_trace_idx max: {bin_trace_idx.max()}")
    print(f"      Expected max for storage: {n_traces_storage - 1}")

    if bin_trace_idx.max() >= n_traces_storage:
        print(f"      ERROR: bin_trace_idx exceeds storage size!")
        return
    else:
        print(f"      OK: Index range within storage bounds")

    # =========================================================================
    # Test 1.3: Check for duplicates and gaps
    # =========================================================================
    print(f"\n[1.3] Checking for duplicates and gaps...")

    unique_indices = np.unique(bin_trace_idx)
    n_unique = len(unique_indices)
    n_duplicates = len(bin_trace_idx) - n_unique

    print(f"      Total headers: {len(bin_trace_idx)}")
    print(f"      Unique indices: {n_unique}")
    print(f"      Duplicates: {n_duplicates}")

    if n_duplicates > 0:
        print(f"      WARNING: {n_duplicates} duplicate indices found!")
        # Find which are duplicated
        idx_counts = pl.DataFrame({'idx': bin_trace_idx}).group_by('idx').agg(pl.count())
        dups = idx_counts.filter(pl.col('count') > 1)
        print(f"      First 10 duplicated indices: {dups.head(10)}")

    # Check for gaps
    expected_indices = set(range(n_traces_storage))
    actual_indices = set(bin_trace_idx)
    missing = expected_indices - actual_indices
    extra = actual_indices - expected_indices

    print(f"      Missing indices (not in headers): {len(missing)}")
    print(f"      Extra indices (beyond storage): {len(extra)}")

    # =========================================================================
    # Test 1.4: Verify geometry makes sense for sample traces
    # =========================================================================
    print(f"\n[1.4] Verifying geometry for sample traces...")

    # Get coordinate columns
    scalar = int(df['scalar_coord'][0])
    scale_factor = 1.0 / abs(scalar) if scalar < 0 else float(scalar) if scalar > 0 else 1.0

    sx = df['source_x'].to_numpy().astype(np.float64) * scale_factor
    sy = df['source_y'].to_numpy().astype(np.float64) * scale_factor
    rx = df['receiver_x'].to_numpy().astype(np.float64) * scale_factor
    ry = df['receiver_y'].to_numpy().astype(np.float64) * scale_factor
    header_offset = df['offset'].to_numpy()

    # Sample some traces and check
    sample_size = 20
    sample_header_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)

    print(f"\n      Sampling {sample_size} traces across the dataset:")
    print(f"      {'Header#':>8} {'Storage#':>10} {'HeaderOff':>10} {'ComputedOff':>12} {'Match':>6}")

    mismatches = 0
    for header_idx in sample_header_indices:
        storage_idx = bin_trace_idx[header_idx]
        h_offset = header_offset[header_idx]
        computed_offset = np.sqrt((rx[header_idx] - sx[header_idx])**2 +
                                  (ry[header_idx] - sy[header_idx])**2)
        match = "OK" if abs(computed_offset - h_offset) < 1.0 else "MISMATCH"
        if match == "MISMATCH":
            mismatches += 1
        print(f"      {header_idx:>8} {storage_idx:>10} {h_offset:>10.1f} {computed_offset:>12.1f} {match:>6}")

    if mismatches > 0:
        print(f"\n      WARNING: {mismatches} offset mismatches found!")

    # =========================================================================
    # Test 1.5: Check if storage order matches header order
    # =========================================================================
    print(f"\n[1.5] Checking storage order vs header order...")

    # Check if bin_trace_idx is sequential
    is_sequential = np.all(bin_trace_idx == np.arange(len(bin_trace_idx)))
    print(f"      bin_trace_idx is sequential (0,1,2,...): {is_sequential}")

    if not is_sequential:
        # Check if it's just offset by something
        diffs = np.diff(bin_trace_idx)
        print(f"      Index differences: min={diffs.min()}, max={diffs.max()}, mean={diffs.mean():.2f}")

        if np.all(diffs == 1):
            print(f"      Index is sequential but starts at {bin_trace_idx[0]}")
        else:
            print(f"      Index is NOT sequential - traces may be reordered")

    # =========================================================================
    # Test 1.6: Cross-check with original trace_index
    # =========================================================================
    print(f"\n[1.6] Cross-checking with original trace_index...")

    if 'trace_index' in df.columns:
        original_idx = df['trace_index'].to_numpy()
        print(f"      Original trace_index range: {original_idx.min()} - {original_idx.max()}")

        # Check relationship
        if np.array_equal(bin_trace_idx, original_idx):
            print(f"      bin_trace_idx == trace_index: True (identical)")
        else:
            correlation = np.corrcoef(bin_trace_idx, original_idx)[0, 1]
            print(f"      bin_trace_idx == trace_index: False")
            print(f"      Correlation: {correlation:.6f}")

            # Check if it's a simple offset
            diff = bin_trace_idx - original_idx
            if np.all(diff == diff[0]):
                print(f"      Relationship: bin_trace_idx = trace_index + {diff[0]}")
            else:
                print(f"      Difference range: {diff.min()} to {diff.max()}")
    else:
        print(f"      No 'trace_index' column found in headers")

    # =========================================================================
    # Test 1.7: Load actual traces and verify they have signal
    # =========================================================================
    print(f"\n[1.7] Loading sample traces to verify signal content...")

    # Load first, middle, and last traces
    test_indices = [0, len(df)//2, len(df)-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Trace Mapping Verification - Bin {BIN_NUM}', fontsize=12, fontweight='bold')

    for i, header_idx in enumerate(test_indices):
        storage_idx = bin_trace_idx[header_idx]

        # Load trace
        if transposed:
            trace = np.asarray(traces_store[:, storage_idx])
        else:
            trace = np.asarray(traces_store[storage_idx, :])

        # Get header info
        h_sx, h_sy = sx[header_idx], sy[header_idx]
        h_rx, h_ry = rx[header_idx], ry[header_idx]
        h_offset = header_offset[header_idx]

        # Plot
        ax = axes[i]
        t_axis = np.arange(len(trace)) * 2.0  # Assuming 2ms sample rate
        ax.plot(t_axis, trace, 'b-', linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Header #{header_idx}\nStorage #{storage_idx}\nOffset={h_offset:.0f}m')
        ax.set_xlim([0, 2000])

        # Print stats
        rms = np.sqrt(np.mean(trace**2))
        print(f"      Header {header_idx} -> Storage {storage_idx}: RMS={rms:.4f}, "
              f"offset={h_offset:.0f}m, src=({h_sx:.0f},{h_sy:.0f})")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "step1_trace_mapping.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n      Saved: {fig_path}")

    # =========================================================================
    # Test 1.8: Check if traces at same storage index have consistent headers
    # =========================================================================
    print(f"\n[1.8] Checking for header consistency at same storage indices...")

    if n_duplicates > 0:
        # Find duplicated indices
        idx_df = pl.DataFrame({
            'header_row': np.arange(len(bin_trace_idx)),
            'storage_idx': bin_trace_idx,
            'offset': header_offset,
            'sx': sx,
            'sy': sy,
        })

        dup_groups = idx_df.group_by('storage_idx').agg([
            pl.count().alias('count'),
            pl.col('offset').std().alias('offset_std'),
            pl.col('sx').std().alias('sx_std'),
        ]).filter(pl.col('count') > 1)

        print(f"      Found {len(dup_groups)} storage indices with multiple headers")

        high_variance = dup_groups.filter(pl.col('offset_std') > 1.0)
        if len(high_variance) > 0:
            print(f"      WARNING: {len(high_variance)} have offset variance > 1m")
            print(high_variance.head(5))
        else:
            print(f"      All duplicates have consistent geometry (offset variance < 1m)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("TRACE MAPPING SUMMARY")
    print("="*70)

    issues = []
    if bin_trace_idx.max() >= n_traces_storage:
        issues.append("Index exceeds storage size")
    if n_duplicates > 0:
        issues.append(f"{n_duplicates} duplicate indices")
    if len(missing) > 0:
        issues.append(f"{len(missing)} missing storage indices")
    if mismatches > 0:
        issues.append(f"{mismatches} offset computation mismatches")

    if issues:
        print(f"\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  NO MAJOR ISSUES FOUND")
        print(f"  Trace-header mapping appears correct")

    return len(issues) == 0


if __name__ == "__main__":
    main()
