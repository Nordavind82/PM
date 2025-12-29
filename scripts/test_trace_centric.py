#!/usr/bin/env python3
"""
Test script for trace-centric PSTM migration.

Compares trace-centric vs tile-by-tile approach and measures performance.
Uses either real sorted data or generates synthetic data for testing.

Usage:
    # Use existing sorted data
    python scripts/test_trace_centric.py --input-dir /Volumes/AO_DISK/processing/offset_200_250_sorted

    # Generate synthetic data
    python scripts/test_trace_centric.py --synthetic --n-traces 300000

    # Quick test with smaller grid
    python scripts/test_trace_centric.py --synthetic --n-traces 100000 --quick
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{timestamp()}] {msg}", flush=True)


def generate_synthetic_data(
    n_traces: int = 300000,
    n_samples: int = 1001,
    output_dir: Path | None = None,
    survey_extent: float = 10000.0,  # 10km
) -> tuple[Path, Path]:
    """Generate synthetic data for testing."""
    import zarr
    import polars as pl

    log(f"Generating synthetic data: {n_traces:,} traces, {n_samples} samples")

    if output_dir is None:
        output_dir = Path("/tmp/pstm_test_synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate random midpoint positions within survey extent
    np.random.seed(42)

    # Grid of midpoints with some randomness
    n_per_side = int(np.sqrt(n_traces))
    x_base = np.linspace(0, survey_extent, n_per_side)
    y_base = np.linspace(0, survey_extent, n_per_side)
    xx, yy = np.meshgrid(x_base, y_base)
    midpoint_x = xx.ravel()[:n_traces] + np.random.randn(min(n_traces, n_per_side**2)) * 10
    midpoint_y = yy.ravel()[:n_traces] + np.random.randn(min(n_traces, n_per_side**2)) * 10

    # Pad if needed
    if len(midpoint_x) < n_traces:
        extra = n_traces - len(midpoint_x)
        midpoint_x = np.concatenate([midpoint_x, np.random.uniform(0, survey_extent, extra)])
        midpoint_y = np.concatenate([midpoint_y, np.random.uniform(0, survey_extent, extra)])

    # Generate source/receiver positions (offset ~200-250m)
    offsets = np.random.uniform(200, 250, n_traces)
    azimuths = np.random.uniform(0, 2*np.pi, n_traces)
    half_offset_x = offsets / 2 * np.cos(azimuths)
    half_offset_y = offsets / 2 * np.sin(azimuths)

    source_x = midpoint_x - half_offset_x
    source_y = midpoint_y - half_offset_y
    receiver_x = midpoint_x + half_offset_x
    receiver_y = midpoint_y + half_offset_y

    log(f"  Survey extent: {midpoint_x.min():.0f} to {midpoint_x.max():.0f} x {midpoint_y.min():.0f} to {midpoint_y.max():.0f}")
    log(f"  Offset range: {offsets.min():.0f} to {offsets.max():.0f} m")

    # Generate trace data (simple diffractor response)
    log("  Generating trace amplitudes...")
    t_axis = np.arange(n_samples) * 2.0  # 2ms sampling
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)

    # Add a diffractor at center
    diff_x, diff_y, diff_z = survey_extent/2, survey_extent/2, 1000.0
    velocity = 2500.0

    for i in range(n_traces):
        # DSR traveltime to diffractor
        ds = np.sqrt((source_x[i] - diff_x)**2 + (source_y[i] - diff_y)**2 + diff_z**2)
        dr = np.sqrt((receiver_x[i] - diff_x)**2 + (receiver_y[i] - diff_y)**2 + diff_z**2)
        t_travel = (ds + dr) / velocity * 1000  # ms

        # Add Ricker wavelet at traveltime
        if 0 < t_travel < t_axis[-1] - 50:
            t_idx = int(t_travel / 2.0)
            # Simple spike (Ricker wavelet approximation)
            for dt in range(-10, 11):
                if 0 <= t_idx + dt < n_samples:
                    traces[i, t_idx + dt] += np.exp(-0.5 * (dt/3)**2)

        if (i + 1) % 50000 == 0:
            log(f"    Generated {i+1:,}/{n_traces:,} traces")

    # Add noise
    traces += np.random.randn(n_traces, n_samples).astype(np.float32) * 0.1

    # Sort by Morton curve for spatial locality
    log("  Sorting by Morton curve...")
    from scripts.prepare_common_offset import compute_morton_indices_fast

    morton_idx = compute_morton_indices_fast(midpoint_x, midpoint_y)
    sort_perm = np.argsort(morton_idx)

    traces = traces[sort_perm]
    source_x = source_x[sort_perm]
    source_y = source_y[sort_perm]
    receiver_x = receiver_x[sort_perm]
    receiver_y = receiver_y[sort_perm]
    midpoint_x = midpoint_x[sort_perm]
    midpoint_y = midpoint_y[sort_perm]
    offsets = offsets[sort_perm]

    # Save to zarr
    log("  Saving traces to zarr...")
    traces_path = output_dir / "traces.zarr"
    z = zarr.open(str(traces_path), mode='w', shape=traces.shape, dtype=np.float32, chunks=(10000, n_samples))
    z[:] = traces

    # Save headers to parquet
    log("  Saving headers to parquet...")
    headers_path = output_dir / "headers.parquet"
    df = pl.DataFrame({
        'trace_index': np.arange(n_traces, dtype=np.int64),
        'source_x': source_x.astype(np.float64),
        'source_y': source_y.astype(np.float64),
        'receiver_x': receiver_x.astype(np.float64),
        'receiver_y': receiver_y.astype(np.float64),
        'offset': offsets.astype(np.float64),
        'scalar_coord': np.ones(n_traces, dtype=np.int16),  # No scaling
    })
    df.write_parquet(str(headers_path))

    log(f"  Saved to {output_dir}")
    return traces_path, headers_path


def test_trace_centric_kernel(
    traces_path: Path,
    headers_path: Path,
    output_grid_size: int = 64,
    max_time_ms: float = 2000.0,
    max_aperture: float = 3000.0,
    velocity: float = 2500.0,
) -> dict:
    """Test the trace-centric kernel."""
    import zarr
    import polars as pl

    from pstm.kernels.base import (
        KernelConfig, TraceBlock, OutputTile, VelocitySlice, create_trace_block
    )

    log("=" * 60)
    log("TRACE-CENTRIC KERNEL TEST")
    log("=" * 60)

    # Load data
    log("Loading data...")
    t0 = time.perf_counter()

    z = zarr.open(str(traces_path), mode='r')
    if isinstance(z, zarr.Group):
        z = z['data']
    n_traces, n_samples = z.shape
    log(f"  Zarr shape: {z.shape}")

    df = pl.read_parquet(headers_path)
    log(f"  Headers: {len(df):,} rows")

    t_load = time.perf_counter() - t0
    log(f"  Load time: {t_load:.2f}s")

    # Determine output grid from data extent
    source_x = df['source_x'].to_numpy()
    source_y = df['source_y'].to_numpy()
    receiver_x = df['receiver_x'].to_numpy()
    receiver_y = df['receiver_y'].to_numpy()

    midpoint_x = (source_x + receiver_x) / 2
    midpoint_y = (source_y + receiver_y) / 2

    x_min, x_max = midpoint_x.min(), midpoint_x.max()
    y_min, y_max = midpoint_y.min(), midpoint_y.max()

    log(f"  Data extent: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]")

    # Create output grid
    nx = ny = output_grid_size
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    dt_ms = 2.0
    nt = int(max_time_ms / dt_ms) + 1

    log(f"  Output grid: {nx}x{ny}x{nt} (dx={dx:.1f}m, dy={dy:.1f}m)")

    # Load ALL trace data
    log(f"Loading all trace data...")
    t0 = time.perf_counter()
    trace_data = z[:].astype(np.float32)
    t_trace_load = time.perf_counter() - t0
    log(f"  Loaded {trace_data.shape} in {t_trace_load:.2f}s ({trace_data.nbytes/1024**2:.1f} MB)")

    # Create trace block
    traces = create_trace_block(
        amplitudes=trace_data,
        source_x=source_x.astype(np.float32),
        source_y=source_y.astype(np.float32),
        receiver_x=receiver_x.astype(np.float32),
        receiver_y=receiver_y.astype(np.float32),
        sample_rate_ms=2.0,
        start_time_ms=0.0,
    )

    # Create output tile
    output = OutputTile(
        image=np.zeros((nx, ny, nt), dtype=np.float64),
        fold=np.zeros((nx, ny), dtype=np.int32),
        x_axis=np.linspace(x_min, x_max, nx).astype(np.float32),
        y_axis=np.linspace(y_min, y_max, ny).astype(np.float32),
        t_axis_ms=np.arange(0, max_time_ms + dt_ms/2, dt_ms).astype(np.float32),
    )

    # Create velocity (constant)
    velocity_1d = np.full(nt, velocity, dtype=np.float32)
    vel_slice = VelocitySlice(vrms=velocity_1d, is_1d=True)

    # Create kernel config
    config = KernelConfig(
        max_aperture_m=max_aperture,
        min_aperture_m=100.0,
        taper_fraction=0.1,
        max_dip_degrees=60.0,
        apply_spreading=True,
        apply_obliquity=True,
        aa_enabled=False,
    )

    # Test trace-centric kernel
    log("Initializing trace-centric kernel...")
    try:
        from pstm.kernels.trace_centric import TraceCentricKernel
        kernel = TraceCentricKernel()
        kernel.initialize(config)
        log("  Trace-centric kernel initialized")

        log("Running trace-centric migration...")
        t0 = time.perf_counter()
        metrics = kernel.migrate_tile(traces, output, vel_slice, config)
        t_kernel = time.perf_counter() - t0

        log(f"  Kernel time: {t_kernel:.2f}s")
        log(f"  Traces processed: {metrics.n_traces_processed:,}")
        log(f"  Throughput: {metrics.n_traces_processed / t_kernel:,.0f} traces/s")

        # Check output
        log(f"  Output image range: [{output.image.min():.4f}, {output.image.max():.4f}]")
        log(f"  Output fold range: [{output.fold.min()}, {output.fold.max()}]")

        result = {
            'kernel': 'trace_centric',
            'n_traces': n_traces,
            'grid_size': f"{nx}x{ny}x{nt}",
            'kernel_time': t_kernel,
            'traces_per_sec': metrics.n_traces_processed / t_kernel,
            'image_max': float(output.image.max()),
            'fold_max': int(output.fold.max()),
        }

    except Exception as e:
        log(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        result = {'error': str(e)}

    log("=" * 60)
    return result


def test_tile_by_tile_kernel(
    traces_path: Path,
    headers_path: Path,
    output_grid_size: int = 64,
    max_time_ms: float = 2000.0,
    max_aperture: float = 3000.0,
    velocity: float = 2500.0,
    tile_size: int = 16,
) -> dict:
    """Test the standard tile-by-tile kernel for comparison."""
    import zarr
    import polars as pl
    from scipy.spatial import cKDTree

    from pstm.kernels.base import (
        KernelConfig, TraceBlock, OutputTile, VelocitySlice, create_trace_block
    )
    from pstm.kernels.factory import create_kernel

    log("=" * 60)
    log("TILE-BY-TILE KERNEL TEST")
    log("=" * 60)

    # Load data
    log("Loading data...")
    z = zarr.open(str(traces_path), mode='r')
    if isinstance(z, zarr.Group):
        z = z['data']
    n_traces, n_samples = z.shape

    df = pl.read_parquet(headers_path)

    source_x = df['source_x'].to_numpy()
    source_y = df['source_y'].to_numpy()
    receiver_x = df['receiver_x'].to_numpy()
    receiver_y = df['receiver_y'].to_numpy()

    midpoint_x = (source_x + receiver_x) / 2
    midpoint_y = (source_y + receiver_y) / 2

    x_min, x_max = midpoint_x.min(), midpoint_x.max()
    y_min, y_max = midpoint_y.min(), midpoint_y.max()

    # Build spatial index
    log("Building spatial index...")
    t0 = time.perf_counter()
    coords = np.column_stack([midpoint_x, midpoint_y])
    tree = cKDTree(coords)
    t_index = time.perf_counter() - t0
    log(f"  Index built in {t_index:.2f}s")

    # Create output grid
    nx = ny = output_grid_size
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    dt_ms = 2.0
    nt = int(max_time_ms / dt_ms) + 1

    # Create kernel
    log("Initializing kernel...")
    config = KernelConfig(
        max_aperture_m=max_aperture,
        min_aperture_m=100.0,
        taper_fraction=0.1,
        max_dip_degrees=60.0,
        apply_spreading=True,
        apply_obliquity=True,
        aa_enabled=False,
    )

    kernel = create_kernel("metal_compiled")
    kernel.initialize(config)

    # Create velocity
    velocity_1d = np.full(nt, velocity, dtype=np.float32)
    vel_slice = VelocitySlice(vrms=velocity_1d, is_1d=True)

    # Process tiles
    n_tiles_x = (nx + tile_size - 1) // tile_size
    n_tiles_y = (ny + tile_size - 1) // tile_size
    n_tiles = n_tiles_x * n_tiles_y

    log(f"Processing {n_tiles} tiles ({n_tiles_x}x{n_tiles_y})...")

    total_traces_loaded = 0
    total_kernel_time = 0.0
    image_full = np.zeros((nx, ny, nt), dtype=np.float64)
    fold_full = np.zeros((nx, ny), dtype=np.int32)

    t_start = time.perf_counter()

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_id = ty * n_tiles_x + tx

            # Tile bounds
            ix_start = tx * tile_size
            ix_end = min(ix_start + tile_size, nx)
            iy_start = ty * tile_size
            iy_end = min(iy_start + tile_size, ny)

            tile_x_min = x_min + ix_start * dx
            tile_x_max = x_min + ix_end * dx
            tile_y_min = y_min + iy_start * dy
            tile_y_max = y_min + iy_end * dy

            tile_cx = (tile_x_min + tile_x_max) / 2
            tile_cy = (tile_y_min + tile_y_max) / 2

            # Query traces in aperture
            indices = tree.query_ball_point([tile_cx, tile_cy], max_aperture)
            if len(indices) == 0:
                continue

            indices = np.array(sorted(indices))
            total_traces_loaded += len(indices)

            # Load trace data
            trace_data = z[indices, :].astype(np.float32)

            # Create trace block
            traces = create_trace_block(
                amplitudes=trace_data,
                source_x=source_x[indices].astype(np.float32),
                source_y=source_y[indices].astype(np.float32),
                receiver_x=receiver_x[indices].astype(np.float32),
                receiver_y=receiver_y[indices].astype(np.float32),
                sample_rate_ms=2.0,
                start_time_ms=0.0,
            )

            # Create output tile
            tile_nx = ix_end - ix_start
            tile_ny = iy_end - iy_start
            output = OutputTile(
                image=np.zeros((tile_nx, tile_ny, nt), dtype=np.float64),
                fold=np.zeros((tile_nx, tile_ny), dtype=np.int32),
                x_axis=np.linspace(tile_x_min, tile_x_max, tile_nx).astype(np.float32),
                y_axis=np.linspace(tile_y_min, tile_y_max, tile_ny).astype(np.float32),
                t_axis_ms=np.arange(0, max_time_ms + dt_ms/2, dt_ms).astype(np.float32),
            )

            # Run kernel
            t0 = time.perf_counter()
            metrics = kernel.migrate_tile(traces, output, vel_slice, config)
            total_kernel_time += time.perf_counter() - t0

            # Accumulate
            image_full[ix_start:ix_end, iy_start:iy_end, :] += output.image
            fold_full[ix_start:ix_end, iy_start:iy_end] += output.fold

            if (tile_id + 1) % 10 == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (tile_id + 1) * (n_tiles - tile_id - 1)
                log(f"  Tile {tile_id+1}/{n_tiles} - ETA: {eta:.0f}s")

    t_total = time.perf_counter() - t_start

    log(f"  Total time: {t_total:.2f}s")
    log(f"  Kernel time: {total_kernel_time:.2f}s")
    log(f"  Traces loaded: {total_traces_loaded:,} (avg {total_traces_loaded/n_tiles:,.0f}/tile)")
    log(f"  Data reuse factor: {total_traces_loaded / n_traces:.1f}x")
    log(f"  Throughput: {total_traces_loaded / t_total:,.0f} traces/s (including reloads)")

    result = {
        'kernel': 'tile_by_tile',
        'n_traces': n_traces,
        'n_tiles': n_tiles,
        'grid_size': f"{nx}x{ny}x{nt}",
        'total_time': t_total,
        'kernel_time': total_kernel_time,
        'traces_loaded': total_traces_loaded,
        'reuse_factor': total_traces_loaded / n_traces,
        'traces_per_sec': total_traces_loaded / t_total,
        'image_max': float(image_full.max()),
        'fold_max': int(fold_full.max()),
    }

    log("=" * 60)
    return result


def main():
    parser = argparse.ArgumentParser(description="Test trace-centric PSTM migration")
    parser.add_argument("--input-dir", type=Path, help="Input directory with traces.zarr and headers.parquet")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--n-traces", type=int, default=300000, help="Number of traces for synthetic data")
    parser.add_argument("--quick", action="store_true", help="Quick test with smaller grid")
    parser.add_argument("--grid-size", type=int, default=128, help="Output grid size")
    parser.add_argument("--max-time-ms", type=float, default=2000.0, help="Maximum time in ms")
    parser.add_argument("--aperture", type=float, default=3000.0, help="Maximum aperture in meters")
    parser.add_argument("--tile-size", type=int, default=16, help="Tile size for tile-by-tile")
    parser.add_argument("--compare", action="store_true", help="Also run tile-by-tile for comparison")

    args = parser.parse_args()

    log("=" * 70)
    log("PSTM TRACE-CENTRIC KERNEL TEST")
    log("=" * 70)

    if args.quick:
        args.grid_size = 64
        args.max_time_ms = 1000.0
        args.n_traces = 100000

    # Get or generate data
    if args.synthetic or args.input_dir is None:
        log(f"Generating synthetic data ({args.n_traces:,} traces)...")
        traces_path, headers_path = generate_synthetic_data(
            n_traces=args.n_traces,
            n_samples=int(args.max_time_ms / 2) + 1,
        )
    else:
        traces_path = args.input_dir / "traces.zarr"
        headers_path = args.input_dir / "headers.parquet"

        if not traces_path.exists() or not headers_path.exists():
            log(f"ERROR: Data not found at {args.input_dir}")
            sys.exit(1)

    log(f"Traces: {traces_path}")
    log(f"Headers: {headers_path}")
    log("")

    # Test trace-centric
    tc_result = test_trace_centric_kernel(
        traces_path=traces_path,
        headers_path=headers_path,
        output_grid_size=args.grid_size,
        max_time_ms=args.max_time_ms,
        max_aperture=args.aperture,
    )

    # Optionally compare with tile-by-tile
    if args.compare:
        log("")
        tbt_result = test_tile_by_tile_kernel(
            traces_path=traces_path,
            headers_path=headers_path,
            output_grid_size=args.grid_size,
            max_time_ms=args.max_time_ms,
            max_aperture=args.aperture,
            tile_size=args.tile_size,
        )

        # Summary comparison
        log("")
        log("=" * 70)
        log("COMPARISON SUMMARY")
        log("=" * 70)

        if 'error' not in tc_result and 'error' not in tbt_result:
            speedup = tbt_result['total_time'] / tc_result['kernel_time']
            log(f"Trace-centric: {tc_result['kernel_time']:.2f}s")
            log(f"Tile-by-tile:  {tbt_result['total_time']:.2f}s")
            log(f"Speedup:       {speedup:.1f}x")
            log(f"Data reuse (tile-by-tile): {tbt_result['reuse_factor']:.1f}x")
        log("=" * 70)
    else:
        log("")
        log("Run with --compare to also test tile-by-tile approach")

    log("Done!")


if __name__ == "__main__":
    main()
