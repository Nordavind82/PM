#!/usr/bin/env python3
"""
Sort seismic data into common offset bins (20m) for PSTM.
Multiprocessing version for faster I/O.

Creates offset bins compatible with run_pstm_all_offsets.py:
- bin_trace_idx column (0 to N-1, used by PSTM to index zarr)
- Transposed zarr format (n_samples, n_traces)
- All original header columns preserved

Usage:
    python sort_common_offset_20m.py [--workers 8]
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import zarr


# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = Path("/Users/olegadamovich/SeismicData/scd_xsd_data_new_20251230_214101")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")

OFFSET_BIN_SIZE = 20  # meters
DEFAULT_WORKERS = 8


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def process_bin(args: tuple) -> dict:
    """
    Process a single offset bin - extract traces and write to zarr.
    Runs in a separate process.
    """
    (bin_idx, trace_indices, input_zarr_path, output_dir,
     headers_path, n_samples) = args

    try:
        bin_idx = int(bin_idx)
        bin_dir = output_dir / f"offset_bin_{bin_idx:02d}"
        bin_dir.mkdir(exist_ok=True)

        n_traces = len(trace_indices)

        # Open input zarr
        input_zarr = zarr.open(str(input_zarr_path), mode="r")

        # Create output zarr (transposed: n_samples, n_traces)
        zarr_path = bin_dir / "traces.zarr"
        store = zarr.storage.LocalStore(str(zarr_path))
        output_zarr = zarr.create_array(
            store=store,
            shape=(int(n_samples), n_traces),
            dtype=np.float32,
            chunks=(int(n_samples), min(1000, n_traces)),
            overwrite=True,
        )

        # Read and write in chunks for memory efficiency
        chunk_size = 50000
        for i in range(0, n_traces, chunk_size):
            chunk_end = min(i + chunk_size, n_traces)
            chunk_indices = trace_indices[i:chunk_end]

            # Read traces (transposed format)
            traces = input_zarr[:, chunk_indices]
            output_zarr[:, i:chunk_end] = traces

        # Add metadata
        output_zarr.attrs["offset_bin"] = bin_idx
        output_zarr.attrs["offset_min"] = float(bin_idx * OFFSET_BIN_SIZE)
        output_zarr.attrs["offset_max"] = float((bin_idx + 1) * OFFSET_BIN_SIZE)
        output_zarr.attrs["n_traces"] = n_traces
        output_zarr.attrs["n_samples"] = int(n_samples)
        output_zarr.attrs["sample_rate_ms"] = 2.0
        output_zarr.attrs["transposed"] = True

        # Load and filter headers for this bin
        all_headers = pl.read_parquet(headers_path)
        bin_headers = all_headers.filter(
            pl.col("trace_index").is_in(trace_indices.tolist())
        )

        # Sort headers to match trace order in zarr
        trace_order = {int(idx): i for i, idx in enumerate(trace_indices)}
        bin_headers = bin_headers.with_columns(
            pl.col("trace_index").map_elements(
                lambda x: trace_order.get(int(x), -1),
                return_dtype=pl.Int64
            ).alias("_sort_order")
        ).sort("_sort_order").drop("_sort_order")

        # Add bin_trace_idx (0 to N-1) - CRITICAL for PSTM
        bin_headers = bin_headers.with_columns(
            pl.arange(0, n_traces, eager=True).alias("bin_trace_idx"),
            pl.lit(bin_idx).alias("offset_bin"),
        )

        # Save headers
        bin_headers.write_parquet(bin_dir / "headers.parquet")

        offset_mean = float(bin_headers["offset"].mean())

        return {
            "bin": bin_idx,
            "success": True,
            "n_traces": n_traces,
            "offset_mean": offset_mean,
        }

    except Exception as e:
        import traceback
        return {
            "bin": bin_idx,
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}",
            "n_traces": 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Sort common offset bins")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 70)
    print("Common Offset Sorting (20m bins) for PSTM - Multiprocess")
    print("=" * 70)
    print(f"[{timestamp()}] Started")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Bin size: {OFFSET_BIN_SIZE}m")
    print(f"Workers: {args.workers}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load headers
    print(f"[{timestamp()}] Loading headers...")
    t0 = time.time()
    headers = pl.read_parquet(args.input_dir / "headers.parquet")
    n_traces_total = len(headers)
    print(f"[{timestamp()}]   {n_traces_total:,} traces in {time.time()-t0:.1f}s")

    # Calculate offset bins
    print(f"[{timestamp()}] Calculating offset bins...")
    offsets = headers["offset"].to_numpy()
    trace_indices = headers["trace_index"].to_numpy()

    min_offset = offsets.min()
    max_offset = offsets.max()
    print(f"[{timestamp()}]   Offset range: {min_offset} - {max_offset} m")

    # Assign bins
    bin_assignments = (offsets / OFFSET_BIN_SIZE).astype(np.int32)
    unique_bins = np.unique(bin_assignments)
    n_bins = len(unique_bins)
    print(f"[{timestamp()}]   Number of bins: {n_bins}")

    # Group trace indices by bin
    print(f"[{timestamp()}] Grouping traces by bin...")
    bin_trace_indices = {}
    for bin_idx in unique_bins:
        mask = bin_assignments == bin_idx
        bin_trace_indices[int(bin_idx)] = trace_indices[mask]

    # Print bin summary
    counts = [len(v) for v in bin_trace_indices.values()]
    print(f"[{timestamp()}]   Traces per bin: {min(counts):,} - {max(counts):,}")

    # Get zarr info
    input_zarr = zarr.open(str(args.input_dir / "traces.zarr"), mode="r")
    n_samples = input_zarr.shape[0]
    print(f"[{timestamp()}]   Samples per trace: {n_samples}")

    # Prepare work items
    work_items = []
    for bin_idx in sorted(bin_trace_indices.keys()):
        work_items.append((
            bin_idx,
            bin_trace_indices[bin_idx],
            args.input_dir / "traces.zarr",
            args.output_dir,
            args.input_dir / "headers.parquet",
            n_samples,
        ))

    # Process bins in parallel
    print()
    print(f"[{timestamp()}] Processing {n_bins} bins with {args.workers} workers...")
    print("-" * 70)

    total_start = time.time()
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_bin, item): item[0] for item in work_items}

        for future in as_completed(futures):
            bin_idx = futures[future]
            completed += 1
            result = future.result()
            results.append(result)

            if result["success"]:
                print(f"[{timestamp()}] [{completed:3d}/{n_bins}] Bin {result['bin']:02d}: "
                      f"{result['n_traces']:,} traces, offset ~{result['offset_mean']:.0f}m")
            else:
                print(f"[{timestamp()}] [{completed:3d}/{n_bins}] Bin {bin_idx}: FAILED - {result['error'][:100]}")

    # Sort results by bin number
    results.sort(key=lambda r: r["bin"])

    total_time = time.time() - total_start

    # Create combined headers file
    print()
    print(f"[{timestamp()}] Creating combined headers file...")
    all_headers_list = []
    for bin_idx in sorted(bin_trace_indices.keys()):
        headers_path = args.output_dir / f"offset_bin_{bin_idx:02d}" / "headers.parquet"
        if headers_path.exists():
            df = pl.read_parquet(headers_path)
            all_headers_list.append(df)

    if all_headers_list:
        all_headers = pl.concat(all_headers_list)
        all_headers.write_parquet(args.output_dir / "all_headers.parquet")
        print(f"[{timestamp()}]   Combined: {len(all_headers):,} traces")

    # Summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    total_traces = sum(r["n_traces"] for r in successful)

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total traces: {total_traces:,}")
    print(f"Bins created: {len(successful)}")
    if failed:
        print(f"Bins failed: {len(failed)}")
        for r in failed:
            print(f"  Bin {r['bin']}: {r['error'][:100]}")
    print(f"Output: {args.output_dir}")
    print()

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
