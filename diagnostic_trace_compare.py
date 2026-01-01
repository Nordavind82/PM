#!/usr/bin/env python3
"""
Compare trace loading between executor path and direct zarr loading.

This checks if the same traces are loaded with the same amplitudes.
"""

import numpy as np
import zarr
import polars as pl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pstm.data.zarr_reader import ZarrTraceReader
from pstm.data.trace_cache import LRUTraceCache
from pstm.data.parquet_headers import ParquetHeaderManager

# Paths - use bin 25 like our diagnostic
DATA_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")
BIN_DIR = DATA_DIR / "offset_bin_25"

def main():
    print("=" * 70)
    print("DIAGNOSTIC: Trace Loading Comparison")
    print("=" * 70)

    traces_path = BIN_DIR / "traces.zarr"
    headers_path = BIN_DIR / "headers.parquet"

    # --- Method 1: Direct Zarr loading (like diagnostic_tile_by_tile.py) ---
    print("\n--- Method 1: Direct Zarr loading ---")
    z_direct = zarr.open_array(str(traces_path), mode='r')
    print(f"  Zarr shape: {z_direct.shape}")
    print(f"  Zarr dtype: {z_direct.dtype}")

    # Check if transposed (samples first or traces first)
    n_samples_check = 1001  # Expected number of samples (0-2000ms at 2ms)
    if z_direct.shape[0] == n_samples_check or z_direct.shape[0] < z_direct.shape[1]:
        print(f"  Data appears transposed: (n_samples={z_direct.shape[0]}, n_traces={z_direct.shape[1]})")
        is_transposed = True
        n_traces = z_direct.shape[1]
        n_samples = z_direct.shape[0]
    else:
        print(f"  Data appears standard: (n_traces={z_direct.shape[0]}, n_samples={z_direct.shape[1]})")
        is_transposed = False
        n_traces = z_direct.shape[0]
        n_samples = z_direct.shape[1]

    # Pick some test indices
    test_indices = np.array([0, 100, 1000, 10000, min(50000, n_traces-1)], dtype=np.int64)
    print(f"  Test indices: {test_indices}")

    # Load directly
    if is_transposed:
        # Data is (n_samples, n_traces) - select columns and transpose
        direct_traces = np.array(z_direct[:, test_indices]).T  # (n_test, n_samples)
    else:
        direct_traces = np.array(z_direct[test_indices, :])

    print(f"  Direct load shape: {direct_traces.shape}")
    print(f"  Direct load dtype: {direct_traces.dtype}")

    # --- Method 2: ZarrTraceReader (like executor) ---
    print("\n--- Method 2: ZarrTraceReader ---")
    reader = ZarrTraceReader(
        traces_path,
        transposed=is_transposed,
        n_traces=n_traces,
        n_samples=n_samples,
    )
    reader.open()
    print(f"  Reader n_traces: {reader.n_traces}")
    print(f"  Reader n_samples: {reader.n_samples}")

    reader_traces = reader.get_traces(test_indices)
    print(f"  Reader load shape: {reader_traces.shape}")
    print(f"  Reader load dtype: {reader_traces.dtype}")

    # --- Method 3: TraceCache (like executor with caching) ---
    print("\n--- Method 3: TraceCache (via ZarrTraceReader) ---")
    cache = LRUTraceCache(max_size_mb=100.0)
    cached_traces = cache.get_traces(test_indices, reader)
    print(f"  Cached load shape: {cached_traces.shape}")
    print(f"  Cached load dtype: {cached_traces.dtype}")

    # --- Compare results ---
    print("\n--- Comparison ---")

    # Direct vs Reader
    direct_f32 = direct_traces.astype(np.float32)  # Convert direct to float32 for fair comparison
    diff_reader = np.abs(reader_traces - direct_f32)
    print(f"Direct vs Reader:")
    print(f"  Max absolute difference: {diff_reader.max():.2e}")
    print(f"  Mean absolute difference: {diff_reader.mean():.2e}")

    if diff_reader.max() < 1e-5:
        print(f"  MATCH: Reader produces same values as direct loading")
    else:
        print(f"  MISMATCH: Values differ!")
        # Show where they differ
        max_idx = np.unravel_index(np.argmax(diff_reader), diff_reader.shape)
        print(f"    Largest diff at index {max_idx}")
        print(f"    Direct: {direct_f32[max_idx]:.8f}")
        print(f"    Reader: {reader_traces[max_idx]:.8f}")

    # Reader vs Cache
    diff_cache = np.abs(cached_traces - reader_traces)
    print(f"\nReader vs Cache:")
    print(f"  Max absolute difference: {diff_cache.max():.2e}")
    if diff_cache.max() < 1e-10:
        print(f"  MATCH: Cache produces same values as reader")
    else:
        print(f"  MISMATCH: Values differ!")

    # --- Check amplitude statistics ---
    print("\n--- Amplitude Statistics (test traces) ---")
    print(f"Direct (float32): min={direct_f32.min():.6f}, max={direct_f32.max():.6f}, RMS={np.sqrt(np.mean(direct_f32**2)):.6f}")
    print(f"Reader:           min={reader_traces.min():.6f}, max={reader_traces.max():.6f}, RMS={np.sqrt(np.mean(reader_traces**2)):.6f}")
    print(f"Cache:            min={cached_traces.min():.6f}, max={cached_traces.max():.6f}, RMS={np.sqrt(np.mean(cached_traces**2)):.6f}")

    # --- Check header loading ---
    print("\n--- Header Loading Comparison ---")

    # Direct polars loading
    df = pl.read_parquet(headers_path)
    print(f"  Polars columns: {df.columns}")

    source_x_polars = df['source_x'].to_numpy()[test_indices]
    source_y_polars = df['source_y'].to_numpy()[test_indices]
    receiver_x_polars = df['receiver_x'].to_numpy()[test_indices]
    receiver_y_polars = df['receiver_y'].to_numpy()[test_indices]
    scalar_polars = df['scalar_coord'].to_numpy()[test_indices]

    # Apply scalar
    scale = np.where(scalar_polars < 0, 1.0 / np.abs(scalar_polars), scalar_polars).astype(np.float64)
    source_x_polars = (source_x_polars * scale).astype(np.float64)
    source_y_polars = (source_y_polars * scale).astype(np.float64)
    receiver_x_polars = (receiver_x_polars * scale).astype(np.float64)
    receiver_y_polars = (receiver_y_polars * scale).astype(np.float64)

    print(f"  Polars source_x (scaled): {source_x_polars[:3]}")
    print(f"  Polars source_y (scaled): {source_y_polars[:3]}")

    # Try ParquetHeaderManager
    try:
        hdr_mgr = ParquetHeaderManager(headers_path)
        hdr_mgr.open()
        geom = hdr_mgr.get_geometry_for_indices(test_indices)
        print(f"  HeaderManager source_x: {geom.source_x[:3]}")
        print(f"  HeaderManager source_y: {geom.source_y[:3]}")

        # Compare
        diff_sx = np.abs(geom.source_x - source_x_polars)
        diff_sy = np.abs(geom.source_y - source_y_polars)
        print(f"  Source X diff: max={diff_sx.max():.6f}, mean={diff_sx.mean():.6f}")
        print(f"  Source Y diff: max={diff_sy.max():.6f}, mean={diff_sy.mean():.6f}")

        if diff_sx.max() < 0.1 and diff_sy.max() < 0.1:
            print(f"  MATCH: HeaderManager coordinates match polars")
        else:
            print(f"  MISMATCH: Coordinate differences!")
    except Exception as e:
        print(f"  HeaderManager error: {e}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
