#!/usr/bin/env python3
"""
Prepare Common Offset Dataset for PSTM

This tool filters and sorts seismic data for optimal PSTM performance:
1. Filters traces to specified offset range
2. Sorts by Hilbert curve index of midpoints (optimal for spatial tile queries)
3. Writes new zarr + parquet with sorted data

Usage:
    python prepare_common_offset.py \
        --input-zarr /path/to/traces.zarr \
        --input-parquet /path/to/headers.parquet \
        --output-dir /path/to/output \
        --offset-min 200 \
        --offset-max 500

Why Hilbert sort?
- PSTM queries traces by spatial tile (midpoint location)
- Hilbert curve preserves 2D locality: nearby midpoints -> nearby storage
- Results in fewer I/O operations per tile (sequential reads vs random)
- Parquet row groups and Zarr chunks become spatially coherent

Alternatively use --sort-order for different strategies:
- hilbert: Hilbert curve (best for 2D, default)
- morton: Z-order/Morton curve (faster to compute, good locality)
- xy: Simple X then Y sort (fast, decent locality)
- offset: Sort by offset (for offset-ordered processing)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from numpy.typing import NDArray


def timestamp() -> str:
    """Return current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.0f}s"


def hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
    """Convert Hilbert curve index d to (x, y) coordinates."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def hilbert_xy2d(n: int, x: int, y: int) -> int:
    """Convert (x, y) coordinates to Hilbert curve index d."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def morton_xy2d(x: int, y: int) -> int:
    """Convert (x, y) to Morton/Z-order index (interleave bits)."""
    def spread_bits(v: int) -> int:
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF
        v = (v | (v << 8)) & 0x00FF00FF00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
        v = (v | (v << 2)) & 0x3333333333333333
        v = (v | (v << 1)) & 0x5555555555555555
        return v
    return spread_bits(x) | (spread_bits(y) << 1)


def compute_hilbert_indices(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    grid_size: int = 1024,
) -> NDArray[np.int64]:
    """
    Compute Hilbert curve indices for (x, y) coordinates.

    Args:
        x: X coordinates
        y: Y coordinates
        grid_size: Size of Hilbert grid (power of 2, default 1024)

    Returns:
        Array of Hilbert indices
    """
    # Normalize to [0, grid_size-1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Handle edge case of zero range
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    x_norm = ((x - x_min) / x_range * (grid_size - 1)).astype(np.int32)
    y_norm = ((y - y_min) / y_range * (grid_size - 1)).astype(np.int32)

    # Clamp to valid range
    x_norm = np.clip(x_norm, 0, grid_size - 1)
    y_norm = np.clip(y_norm, 0, grid_size - 1)

    # Compute Hilbert indices (vectorized using numpy)
    # For large arrays, use a faster approximation via Morton code
    n = len(x)
    indices = np.zeros(n, dtype=np.int64)

    print(f"  Computing Hilbert indices for {n:,} points...")

    # Use Morton as fast approximation (good enough for locality)
    # Morton is ~10x faster and provides similar locality
    for i in range(n):
        indices[i] = hilbert_xy2d(grid_size, int(x_norm[i]), int(y_norm[i]))
        if (i + 1) % 1_000_000 == 0:
            print(f"    {i+1:,}/{n:,} ({100*(i+1)/n:.1f}%)")

    return indices


def compute_morton_indices(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    grid_size: int = 65536,
) -> NDArray[np.int64]:
    """
    Compute Morton/Z-order indices for (x, y) coordinates.
    Much faster than Hilbert, still good locality.
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    x_norm = ((x - x_min) / x_range * (grid_size - 1)).astype(np.int32)
    y_norm = ((y - y_min) / y_range * (grid_size - 1)).astype(np.int32)

    x_norm = np.clip(x_norm, 0, grid_size - 1)
    y_norm = np.clip(y_norm, 0, grid_size - 1)

    print(f"  Computing Morton indices for {len(x):,} points...")

    # Vectorized Morton computation
    indices = np.zeros(len(x), dtype=np.int64)
    for i in range(len(x)):
        indices[i] = morton_xy2d(int(x_norm[i]), int(y_norm[i]))

    return indices


def compute_morton_indices_fast(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    grid_size: int = 65536,
) -> NDArray[np.int64]:
    """
    Fast vectorized Morton index computation using NumPy.
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    # Normalize to 16-bit range (0-65535)
    x_norm = ((x - x_min) / x_range * 65535).astype(np.uint32)
    y_norm = ((y - y_min) / y_range * 65535).astype(np.uint32)

    print(f"  Computing Morton indices (vectorized) for {len(x):,} points...")

    # Bit interleaving for 16-bit inputs -> 32-bit output
    # This is a simplified version that works for our grid sizes
    def spread_bits_vec(v: NDArray[np.uint32]) -> NDArray[np.uint64]:
        v = v.astype(np.uint64)
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF
        v = (v | (v << 8)) & 0x00FF00FF00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
        v = (v | (v << 2)) & 0x3333333333333333
        v = (v | (v << 1)) & 0x5555555555555555
        return v

    indices = spread_bits_vec(x_norm) | (spread_bits_vec(y_norm) << 1)
    return indices.astype(np.int64)


def prepare_common_offset(
    input_zarr: Path,
    input_parquet: Path,
    output_dir: Path,
    offset_min: float | None = None,
    offset_max: float | None = None,
    sort_order: str = "morton",
    chunk_size: int = 10000,
    source_x_col: str = "source_x",
    source_y_col: str = "source_y",
    receiver_x_col: str = "receiver_x",
    receiver_y_col: str = "receiver_y",
    offset_col: str = "offset",
    trace_idx_col: str = "trace_index",
    transposed: bool = False,
    coord_scalar: float | None = None,
) -> dict:
    """
    Prepare common offset dataset sorted for optimal PSTM performance.

    Args:
        input_zarr: Path to input trace data (zarr)
        input_parquet: Path to input headers (parquet)
        output_dir: Output directory for sorted data
        offset_min: Minimum offset to include (None = no minimum)
        offset_max: Maximum offset to include (None = no maximum)
        sort_order: Sort strategy - "hilbert", "morton", "xy", or "offset"
        chunk_size: Zarr chunk size for output
        source_x_col: Column name for source X
        source_y_col: Column name for source Y
        receiver_x_col: Column name for receiver X
        receiver_y_col: Column name for receiver Y
        offset_col: Column name for offset
        trace_idx_col: Column name for trace index
        transposed: If True, input zarr is (n_samples, n_traces)
        coord_scalar: Coordinate scalar to apply (e.g., -100 means divide by 100)

    Returns:
        Dict with statistics about the operation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PSTM Common Offset Data Preparation")
    print("=" * 70)
    print(f"[{timestamp()}] Started")
    print(f"  Input zarr:    {input_zarr}")
    print(f"  Input parquet: {input_parquet}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Offset range:  {offset_min} - {offset_max}")
    print(f"  Sort order:    {sort_order}")
    print(f"  Transposed:    {transposed}")
    print(f"  Coord scalar:  {coord_scalar}")
    print("=" * 70)

    t_start = time.time()
    step_times = {}

    # Step 1: Load and filter headers
    print(f"\n[{timestamp()}] [1/4] Loading and filtering headers...")
    t0 = time.time()

    df = pl.scan_parquet(input_parquet)

    # Apply offset filter at Parquet level (predicate pushdown)
    if offset_min is not None and offset_max is not None:
        df = df.filter(pl.col(offset_col).is_between(offset_min, offset_max))
    elif offset_min is not None:
        df = df.filter(pl.col(offset_col) >= offset_min)
    elif offset_max is not None:
        df = df.filter(pl.col(offset_col) <= offset_max)

    df = df.collect()
    n_traces = len(df)
    step_times['load_filter'] = time.time() - t0

    print(f"[{timestamp()}]   Loaded {n_traces:,} traces")
    print(f"[{timestamp()}]   Step 1 completed in {format_duration(step_times['load_filter'])}")

    if n_traces == 0:
        print("ERROR: No traces match the offset filter!")
        return {"error": "No traces match filter", "n_traces": 0}

    # Step 2: Compute midpoints and sort indices
    print(f"\n[{timestamp()}] [2/4] Computing spatial sort order...")
    t0 = time.time()

    source_x = df[source_x_col].to_numpy().astype(np.float64)
    source_y = df[source_y_col].to_numpy().astype(np.float64)
    receiver_x = df[receiver_x_col].to_numpy().astype(np.float64)
    receiver_y = df[receiver_y_col].to_numpy().astype(np.float64)

    # Apply coordinate scalar if provided
    if coord_scalar is not None and coord_scalar != 0:
        if coord_scalar < 0:
            # Negative scalar means divide
            scale_factor = 1.0 / abs(coord_scalar)
        else:
            scale_factor = coord_scalar
        source_x = source_x * scale_factor
        source_y = source_y * scale_factor
        receiver_x = receiver_x * scale_factor
        receiver_y = receiver_y * scale_factor
        print(f"  Applied coordinate scalar: {coord_scalar} (factor: {scale_factor})")

    midpoint_x = (source_x + receiver_x) / 2
    midpoint_y = (source_y + receiver_y) / 2

    print(f"  Midpoint X range: {midpoint_x.min():.1f} - {midpoint_x.max():.1f}")
    print(f"  Midpoint Y range: {midpoint_y.min():.1f} - {midpoint_y.max():.1f}")

    if sort_order == "hilbert":
        sort_keys = compute_hilbert_indices(midpoint_x, midpoint_y)
    elif sort_order == "morton":
        sort_keys = compute_morton_indices_fast(midpoint_x, midpoint_y)
    elif sort_order == "xy":
        # Simple X-major then Y sort
        x_norm = ((midpoint_x - midpoint_x.min()) / (midpoint_x.max() - midpoint_x.min()) * 1e9).astype(np.int64)
        y_norm = ((midpoint_y - midpoint_y.min()) / (midpoint_y.max() - midpoint_y.min()) * 1e9).astype(np.int64)
        sort_keys = x_norm * int(1e10) + y_norm
    elif sort_order == "offset":
        sort_keys = df[offset_col].to_numpy()
    else:
        raise ValueError(f"Unknown sort order: {sort_order}")

    # Get sort permutation
    sort_perm = np.argsort(sort_keys)
    step_times['compute_sort'] = time.time() - t0

    print(f"[{timestamp()}]   Sort order computed")
    print(f"[{timestamp()}]   Step 2 completed in {format_duration(step_times['compute_sort'])}")

    # Step 3: Reorder headers and write
    print(f"\n[{timestamp()}] [3/4] Writing sorted headers...")
    t0 = time.time()

    # Reorder dataframe
    df_sorted = df[sort_perm.tolist()]

    # Add new sequential trace index and update the trace_index column
    # IMPORTANT: trace_index must be updated to sequential 0..N-1 because
    # PSTM uses this column to index into the zarr array. The original
    # indices are preserved in trace_idx_original for reference.
    new_indices = np.arange(n_traces, dtype=np.int64)
    df_sorted = df_sorted.with_columns([
        pl.Series("trace_idx_original", df_sorted[trace_idx_col]),
        pl.Series("trace_idx", new_indices),
        pl.Series(trace_idx_col, new_indices),  # Update trace_index for PSTM compatibility
    ])

    # Write sorted parquet
    output_parquet = output_dir / "headers.parquet"
    df_sorted.write_parquet(str(output_parquet))
    step_times['write_headers'] = time.time() - t0

    print(f"[{timestamp()}]   Headers written to {output_parquet}")
    print(f"[{timestamp()}]   Step 3 completed in {format_duration(step_times['write_headers'])}")

    # Step 4: Reorder trace data and write
    print(f"\n[{timestamp()}] [4/4] Reordering and writing trace data...")
    t0 = time.time()

    # Open input zarr
    z_in = zarr.open(str(input_zarr), mode='r')
    if isinstance(z_in, zarr.Group):
        z_in = z_in['data']

    if transposed:
        n_samples, n_traces_in = z_in.shape
    else:
        n_traces_in, n_samples = z_in.shape

    print(f"  Input shape: {z_in.shape}, dtype: {z_in.dtype}")

    # Get original trace indices for reordering
    original_indices = df[trace_idx_col].to_numpy()
    sorted_original_indices = original_indices[sort_perm]

    # Create output zarr
    output_zarr = output_dir / "traces.zarr"
    z_out = zarr.open(
        str(output_zarr),
        mode='w',
        shape=(n_traces, n_samples),
        chunks=(chunk_size, n_samples),
        dtype=z_in.dtype,
    )

    # Copy metadata
    z_out.attrs['n_traces'] = n_traces
    z_out.attrs['n_samples'] = n_samples
    z_out.attrs['sort_order'] = sort_order
    z_out.attrs['offset_min'] = offset_min
    z_out.attrs['offset_max'] = offset_max
    z_out.attrs['source_file'] = str(input_zarr)

    # Process in chunks
    batch_size = 50000
    n_batches = (n_traces + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_traces)

        # Get original indices for this batch
        batch_orig_indices = sorted_original_indices[start:end]

        # Load traces (may be non-contiguous)
        if transposed:
            batch_data = z_in[:, batch_orig_indices].T
        else:
            batch_data = z_in[batch_orig_indices, :]

        # Write to output
        z_out[start:end, :] = batch_data

        elapsed = time.time() - t0
        rate = (end) / elapsed if elapsed > 0 else 0
        eta = (n_traces - end) / rate if rate > 0 else 0

        print(f"[{timestamp()}]   Batch {batch_idx+1}/{n_batches}: {end:,}/{n_traces:,} traces "
              f"({100*end/n_traces:.1f}%) - {rate:.0f} traces/s - ETA: {format_duration(eta)}")

    step_times['write_traces'] = time.time() - t0
    print(f"[{timestamp()}]   Traces written to {output_zarr}")
    print(f"[{timestamp()}]   Step 4 completed in {format_duration(step_times['write_traces'])}")

    # Summary
    t_total = time.time() - t_start

    print("\n" + "=" * 70)
    print(f"[{timestamp()}] COMPLETE")
    print("=" * 70)
    print(f"  Total runtime:  {format_duration(t_total)}")
    print(f"  Traces output:  {n_traces:,}")
    print(f"  Output zarr:    {output_zarr}")
    print(f"  Output parquet: {output_parquet}")
    print(f"  Sort order:     {sort_order}")
    print()
    print("  Step Breakdown:")
    print(f"    1. Load & filter headers: {format_duration(step_times['load_filter']):>12}")
    print(f"    2. Compute sort order:    {format_duration(step_times['compute_sort']):>12}")
    print(f"    3. Write sorted headers:  {format_duration(step_times['write_headers']):>12}")
    print(f"    4. Write sorted traces:   {format_duration(step_times['write_traces']):>12}")
    print(f"    ─────────────────────────────────────")
    print(f"    Total:                    {format_duration(t_total):>12}")

    # Compute locality statistics
    print()
    print("  Locality Quality (consecutive trace midpoint distances):")
    sample_size = min(10000, n_traces - 1)
    sample_indices = np.random.choice(n_traces - 1, sample_size, replace=False)

    mx_sorted = midpoint_x[sort_perm]
    my_sorted = midpoint_y[sort_perm]

    dx = mx_sorted[sample_indices + 1] - mx_sorted[sample_indices]
    dy = my_sorted[sample_indices + 1] - my_sorted[sample_indices]
    dist = np.sqrt(dx**2 + dy**2)

    print(f"    Mean distance:   {dist.mean():.1f} m")
    print(f"    Median distance: {np.median(dist):.1f} m")
    print(f"    95th percentile: {np.percentile(dist, 95):.1f} m")
    print(f"    (Lower = better spatial locality for PSTM tile queries)")

    print("=" * 70)

    return {
        "n_traces": n_traces,
        "n_samples": n_samples,
        "output_zarr": str(output_zarr),
        "output_parquet": str(output_parquet),
        "sort_order": sort_order,
        "offset_min": offset_min,
        "offset_max": offset_max,
        "time_total_seconds": t_total,
        "time_load_filter_seconds": step_times['load_filter'],
        "time_compute_sort_seconds": step_times['compute_sort'],
        "time_write_headers_seconds": step_times['write_headers'],
        "time_write_traces_seconds": step_times['write_traces'],
        "mean_consecutive_distance_m": float(dist.mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare common offset dataset for optimal PSTM performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract offset 200-500m, sort by Morton curve (fast, good locality)
  python prepare_common_offset.py \\
      --input-zarr /data/survey/traces.zarr \\
      --input-parquet /data/survey/headers.parquet \\
      --output-dir /data/survey/offset_200_500 \\
      --offset-min 200 --offset-max 500

  # Use Hilbert curve (best locality, slower to compute)
  python prepare_common_offset.py \\
      --input-zarr /data/survey/traces.zarr \\
      --input-parquet /data/survey/headers.parquet \\
      --output-dir /data/survey/offset_200_500_hilbert \\
      --offset-min 200 --offset-max 500 \\
      --sort-order hilbert

Sort order options:
  morton  - Z-order/Morton curve (default, fast, good locality)
  hilbert - Hilbert curve (best locality, slower)
  xy      - Simple X then Y sort (fastest, decent locality)
  offset  - Sort by offset value (for offset-ordered access)
        """,
    )

    parser.add_argument(
        "--input-zarr", "-z",
        type=Path,
        required=True,
        help="Path to input trace data (zarr format)",
    )
    parser.add_argument(
        "--input-parquet", "-p",
        type=Path,
        required=True,
        help="Path to input headers (parquet format)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for sorted data",
    )
    parser.add_argument(
        "--offset-min",
        type=float,
        default=None,
        help="Minimum offset to include (meters)",
    )
    parser.add_argument(
        "--offset-max",
        type=float,
        default=None,
        help="Maximum offset to include (meters)",
    )
    parser.add_argument(
        "--sort-order",
        choices=["morton", "hilbert", "xy", "offset"],
        default="morton",
        help="Sort strategy (default: morton)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Zarr chunk size for output (default: 10000)",
    )
    parser.add_argument(
        "--transposed",
        action="store_true",
        help="Input zarr is transposed (n_samples, n_traces)",
    )

    # Column name overrides
    parser.add_argument("--source-x-col", default="source_x", help="Source X column name")
    parser.add_argument("--source-y-col", default="source_y", help="Source Y column name")
    parser.add_argument("--receiver-x-col", default="receiver_x", help="Receiver X column name")
    parser.add_argument("--receiver-y-col", default="receiver_y", help="Receiver Y column name")
    parser.add_argument("--offset-col", default="offset", help="Offset column name")
    parser.add_argument("--trace-idx-col", default="trace_index", help="Trace index column name")
    parser.add_argument(
        "--coord-scalar",
        type=float,
        default=None,
        help="Coordinate scalar (e.g., -100 means divide by 100). Auto-detected from scalar_coord column if not specified.",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_zarr.exists():
        print(f"ERROR: Input zarr not found: {args.input_zarr}")
        sys.exit(1)
    if not args.input_parquet.exists():
        print(f"ERROR: Input parquet not found: {args.input_parquet}")
        sys.exit(1)

    # Auto-detect coord scalar if not provided
    coord_scalar = args.coord_scalar
    if coord_scalar is None:
        try:
            scalar_df = pl.scan_parquet(args.input_parquet).select("scalar_coord").head(1).collect()
            if len(scalar_df) > 0:
                coord_scalar = float(scalar_df["scalar_coord"][0])
                print(f"Auto-detected coord_scalar from scalar_coord column: {coord_scalar}")
        except Exception:
            pass  # No scalar_coord column, use None

    # Run preparation
    result = prepare_common_offset(
        input_zarr=args.input_zarr,
        input_parquet=args.input_parquet,
        output_dir=args.output_dir,
        offset_min=args.offset_min,
        offset_max=args.offset_max,
        sort_order=args.sort_order,
        chunk_size=args.chunk_size,
        source_x_col=args.source_x_col,
        source_y_col=args.source_y_col,
        receiver_x_col=args.receiver_x_col,
        receiver_y_col=args.receiver_y_col,
        offset_col=args.offset_col,
        trace_idx_col=args.trace_idx_col,
        transposed=args.transposed,
        coord_scalar=coord_scalar,
    )

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
