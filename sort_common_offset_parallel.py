#!/usr/bin/env python3
"""
Parallel Common Offset Sorting Script.

Sorts seismic traces into common offset gathers with 50m bins.
Uses multiprocessing for efficient parallel I/O.

Input: Zarr traces + Parquet headers
Output: Separate Zarr/Parquet files per offset bin

Usage:
    python sort_common_offset_parallel.py [--workers N] [--bin-size 50]
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import zarr


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SortConfig:
    """Configuration for common offset sorting."""
    # Input paths
    input_traces: Path
    input_headers: Path

    # Output
    output_dir: Path

    # Binning parameters
    bin_size_m: float = 50.0
    min_offset_m: float = 0.0
    max_offset_m: Optional[float] = None

    # Processing
    n_workers: int = 8
    chunk_size: int = 50000  # Traces per chunk for parallel writing

    # Options
    overwrite: bool = False
    verbose: bool = True


# =============================================================================
# Sorting Functions
# =============================================================================

def get_offset_bins(headers_df: pl.DataFrame, bin_size: float,
                    min_offset: float = 0.0, max_offset: Optional[float] = None) -> dict:
    """
    Compute offset bin assignments for all traces.

    Returns:
        Dictionary mapping bin_id -> (bin_min, bin_max, trace_indices)
    """
    offsets = headers_df['offset'].to_numpy()

    if max_offset is None:
        max_offset = offsets.max()

    # Compute bin edges
    bin_edges = np.arange(min_offset, max_offset + bin_size, bin_size)
    n_bins = len(bin_edges) - 1

    # Assign traces to bins
    bin_ids = np.digitize(offsets, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    # Group traces by bin
    bins = {}
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        trace_indices = np.where(mask)[0]

        if len(trace_indices) > 0:
            bin_min = bin_edges[bin_id]
            bin_max = bin_edges[bin_id + 1]
            bins[bin_id] = {
                'min_offset': bin_min,
                'max_offset': bin_max,
                'trace_indices': trace_indices,
                'n_traces': len(trace_indices),
            }

    return bins


def write_offset_bin(args: tuple) -> dict:
    """
    Write a single offset bin to disk.

    This function is designed to be called in parallel.

    Args:
        args: Tuple of (bin_id, bin_info, config_dict)

    Returns:
        Result dictionary with status and stats
    """
    bin_id, bin_info, config_dict = args

    try:
        # Reconstruct paths from config dict
        input_traces = Path(config_dict['input_traces'])
        input_headers = Path(config_dict['input_headers'])
        output_dir = Path(config_dict['output_dir'])

        bin_min = bin_info['min_offset']
        bin_max = bin_info['max_offset']
        trace_indices = bin_info['trace_indices']
        n_traces = len(trace_indices)

        # Create output directory for this bin
        bin_dir = output_dir / f"offset_bin_{bin_id:02d}"
        bin_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        traces_out_path = bin_dir / "traces.zarr"
        headers_out_path = bin_dir / "headers.parquet"

        # Skip if exists and not overwriting
        if traces_out_path.exists() and not config_dict.get('overwrite', False):
            return {
                'bin_id': bin_id,
                'status': 'skipped',
                'message': 'Already exists',
                'n_traces': n_traces,
            }

        start_time = time.time()

        # Load input headers and filter
        headers_df = pl.read_parquet(input_headers)
        bin_headers = headers_df[trace_indices.tolist()]

        # Add bin metadata columns
        bin_headers = bin_headers.with_columns([
            pl.lit(bin_id).alias('offset_bin'),
            pl.arange(0, n_traces, eager=True).alias('bin_trace_idx'),
        ])

        # Open input traces
        input_store = zarr.storage.LocalStore(str(input_traces))
        input_zarr = zarr.open_array(store=input_store, mode='r')
        n_samples = input_zarr.shape[0]

        # Create output zarr array
        output_store = zarr.storage.LocalStore(str(traces_out_path))
        output_zarr = zarr.create_array(
            store=output_store,
            shape=(n_samples, n_traces),
            dtype=np.float32,
            chunks=(n_samples, min(1000, n_traces)),
            overwrite=True,
        )

        # Copy traces in chunks for memory efficiency
        chunk_size = config_dict.get('chunk_size', 50000)
        for i in range(0, n_traces, chunk_size):
            chunk_end = min(i + chunk_size, n_traces)
            chunk_indices = trace_indices[i:chunk_end]

            # Read traces (sorted indices for better I/O)
            sorted_order = np.argsort(chunk_indices)
            sorted_indices = chunk_indices[sorted_order]

            # Read chunk
            traces_chunk = input_zarr[:, sorted_indices]

            # Unsort back to original order
            unsort_order = np.argsort(sorted_order)
            traces_chunk = traces_chunk[:, unsort_order]

            # Write to output
            output_zarr[:, i:chunk_end] = traces_chunk

        # Add metadata to zarr
        output_zarr.attrs['offset_bin'] = bin_id
        output_zarr.attrs['offset_min'] = float(bin_min)
        output_zarr.attrs['offset_max'] = float(bin_max)
        output_zarr.attrs['n_traces'] = n_traces
        output_zarr.attrs['n_samples'] = n_samples
        output_zarr.attrs['sample_rate_ms'] = 2.0

        # Write headers
        bin_headers.write_parquet(headers_out_path)

        elapsed = time.time() - start_time

        return {
            'bin_id': bin_id,
            'status': 'success',
            'n_traces': n_traces,
            'offset_range': f"{bin_min:.0f}-{bin_max:.0f}m",
            'elapsed_s': elapsed,
            'traces_per_sec': n_traces / elapsed if elapsed > 0 else 0,
        }

    except Exception as e:
        return {
            'bin_id': bin_id,
            'status': 'error',
            'message': str(e),
            'n_traces': bin_info.get('n_traces', 0),
        }


def run_parallel_sort(config: SortConfig) -> bool:
    """
    Run parallel common offset sorting.

    Args:
        config: Sort configuration

    Returns:
        True if successful
    """
    print("=" * 70)
    print("Parallel Common Offset Sorting")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load headers to compute bins
    print("[1] Loading headers and computing offset bins...")
    headers_df = pl.read_parquet(config.input_headers)
    print(f"    Total traces: {len(headers_df):,}")

    bins = get_offset_bins(
        headers_df,
        config.bin_size_m,
        config.min_offset_m,
        config.max_offset_m,
    )

    print(f"    Offset bins: {len(bins)}")
    print(f"    Bin size: {config.bin_size_m} m")
    print()

    # Print bin summary
    print("[2] Offset bin summary:")
    total_traces = 0
    for bin_id, info in sorted(bins.items()):
        print(f"    Bin {bin_id:2d}: {info['min_offset']:6.0f}-{info['max_offset']:6.0f}m "
              f"-> {info['n_traces']:>10,} traces")
        total_traces += info['n_traces']
    print(f"    {'':8s} Total: {total_traces:>10,} traces")
    print()

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for parallel processing
    config_dict = {
        'input_traces': str(config.input_traces),
        'input_headers': str(config.input_headers),
        'output_dir': str(config.output_dir),
        'chunk_size': config.chunk_size,
        'overwrite': config.overwrite,
    }

    tasks = [(bin_id, info, config_dict) for bin_id, info in bins.items()]

    # Run parallel sorting
    print(f"[3] Sorting traces into offset bins ({config.n_workers} workers)...")
    print("-" * 70)

    start_time = time.time()
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        futures = {executor.submit(write_offset_bin, task): task[0] for task in tasks}

        for future in as_completed(futures):
            bin_id = futures[future]
            result = future.result()
            results.append(result)
            completed += 1

            # Print progress
            status = result['status']
            if status == 'success':
                print(f"    [{completed:2d}/{len(tasks)}] Bin {result['bin_id']:2d}: "
                      f"{result['n_traces']:>10,} traces "
                      f"({result['offset_range']}) "
                      f"in {result['elapsed_s']:.1f}s "
                      f"({result['traces_per_sec']/1000:.1f}k/s)")
            elif status == 'skipped':
                print(f"    [{completed:2d}/{len(tasks)}] Bin {result['bin_id']:2d}: SKIPPED ({result['message']})")
            else:
                print(f"    [{completed:2d}/{len(tasks)}] Bin {result['bin_id']:2d}: ERROR - {result['message']}")

    elapsed = time.time() - start_time

    # Summary
    print("-" * 70)
    print()
    print("[4] Summary:")

    successful = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    traces_written = sum(r['n_traces'] for r in results if r['status'] == 'success')

    print(f"    Successful: {successful}")
    print(f"    Skipped: {skipped}")
    print(f"    Errors: {errors}")
    print(f"    Traces written: {traces_written:,}")
    print(f"    Total time: {elapsed:.1f}s")
    print(f"    Throughput: {traces_written/elapsed/1000:.1f}k traces/s")
    print()
    print(f"    Output: {config.output_dir}")
    print()

    # Write combined headers (all bins)
    print("[5] Creating combined headers file...")
    all_headers = []
    for bin_id in sorted(bins.keys()):
        bin_headers_path = config.output_dir / f"offset_bin_{bin_id:02d}" / "headers.parquet"
        if bin_headers_path.exists():
            df = pl.read_parquet(bin_headers_path)
            df = df.with_columns(pl.lit(bin_id).alias('offset_bin'))
            all_headers.append(df)

    if all_headers:
        combined = pl.concat(all_headers)
        combined_path = config.output_dir / "all_headers.parquet"
        combined.write_parquet(combined_path)
        print(f"    Combined headers: {combined_path}")
        print(f"    Total traces: {len(combined):,}")

    print()
    print("=" * 70)
    print("Sorting Complete!")
    print("=" * 70)

    return errors == 0


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sort seismic traces into common offset gathers"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/olegadamovich/SeismicData/processing/processed_scd_xsd_data_new_20251221_225219_20251221_232357/output"),
        help="Input directory containing traces.zarr and headers.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new"),
        help="Output directory for sorted offset bins",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=50.0,
        help="Offset bin size in meters (default: 50)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Traces per chunk for I/O (default: 50000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--min-offset",
        type=float,
        default=0.0,
        help="Minimum offset to include (default: 0)",
    )
    parser.add_argument(
        "--max-offset",
        type=float,
        default=None,
        help="Maximum offset to include (default: all)",
    )

    args = parser.parse_args()

    # Resolve input paths
    input_traces = args.input_dir / "traces.zarr"
    input_headers = args.input_dir / "headers.parquet"

    # Handle symlink (traces.zarr -> noise.zarr)
    if input_traces.is_symlink():
        input_traces = input_traces.resolve()
        print(f"Note: traces.zarr is symlink to {input_traces.name}")

    # Validate inputs
    if not input_traces.exists():
        print(f"ERROR: Input traces not found: {input_traces}")
        return 1
    if not input_headers.exists():
        print(f"ERROR: Input headers not found: {input_headers}")
        return 1

    # Create configuration
    config = SortConfig(
        input_traces=input_traces,
        input_headers=input_headers,
        output_dir=args.output_dir,
        bin_size_m=args.bin_size,
        min_offset_m=args.min_offset,
        max_offset_m=args.max_offset,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )

    # Run sorting
    success = run_parallel_sort(config)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
