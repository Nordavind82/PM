#!/usr/bin/env python3
"""
Extract Common Image Gathers (CIG) from PSTM Migration Output.

Extracts migrated traces at sparse grid locations (every N inlines and M crosslines)
from all offset bins, sorts by offset to create CIGs, and outputs as zarr + parquet.

Input: Multiple migration_bin_XX folders with migrated_stack.zarr and bin_headers.parquet
Output: CIG zarr (traces) + parquet (headers with IL/XL/X/Y/offset)

Usage:
    python extract_cig_from_migration.py [--input-dir PATH] [--output-dir PATH]
                                         [--inline-step 10] [--xline-step 20]
"""

import argparse
import sys
import time
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
class CIGConfig:
    """Configuration for CIG extraction."""
    # Input
    input_dir: Path

    # Output
    output_dir: Path

    # Grid subsampling
    inline_step: int = 10   # Extract every Nth inline
    xline_step: int = 20    # Extract every Mth crossline

    # Options
    verbose: bool = True


# =============================================================================
# CIG Extraction Functions
# =============================================================================

def find_offset_bins(input_dir: Path) -> list[tuple[int, Path]]:
    """
    Find all migration offset bin directories.

    Returns:
        List of (bin_number, bin_path) tuples sorted by bin number
    """
    bins = []
    for d in input_dir.iterdir():
        if d.is_dir() and d.name.startswith("migration_bin_"):
            try:
                bin_num = int(d.name.replace("migration_bin_", ""))
                # Verify required files exist
                if (d / "migrated_stack.zarr").exists():
                    bins.append((bin_num, d))
            except ValueError:
                continue

    return sorted(bins, key=lambda x: x[0])


def load_migration_metadata(bin_path: Path) -> dict:
    """Load metadata from a migration bin's zarr file."""
    zarr_path = bin_path / "migrated_stack.zarr"
    store = zarr.storage.LocalStore(str(zarr_path))
    z = zarr.open_array(store=store, mode='r')

    return {
        'shape': z.shape,
        'nx': z.shape[0],
        'ny': z.shape[1],
        'nt': z.shape[2],
        'x_min': z.attrs.get('x_min'),
        'x_max': z.attrs.get('x_max'),
        'y_min': z.attrs.get('y_min'),
        'y_max': z.attrs.get('y_max'),
        'dx': z.attrs.get('dx'),
        'dy': z.attrs.get('dy'),
        'dt_ms': z.attrs.get('dt_ms'),
        't_min_ms': z.attrs.get('t_min_ms'),
        't_max_ms': z.attrs.get('t_max_ms'),
    }


def compute_grid_coordinates(metadata: dict, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute X and Y coordinates for each grid point.

    For a corner-point grid, we need to interpolate coordinates.
    This assumes the grid is defined by corner coordinates.
    """
    x_min = metadata['x_min']
    x_max = metadata['x_max']
    y_min = metadata['y_min']
    y_max = metadata['y_max']

    # Create coordinate axes
    x_axis = np.linspace(x_min, x_max, nx)
    y_axis = np.linspace(y_min, y_max, ny)

    return x_axis, y_axis


def extract_cig_locations(
    config: CIGConfig,
    metadata: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Determine CIG extraction locations.

    Returns:
        (inline_indices, xline_indices, x_coords, y_coords)
    """
    nx = metadata['nx']
    ny = metadata['ny']

    # Generate sparse indices
    inline_indices = np.arange(0, nx, config.inline_step)
    xline_indices = np.arange(0, ny, config.xline_step)

    # Compute coordinates
    x_axis, y_axis = compute_grid_coordinates(metadata, nx, ny)

    # Get coordinates at CIG locations
    x_coords = x_axis[inline_indices]
    y_coords = y_axis[xline_indices]

    return inline_indices, xline_indices, x_coords, y_coords


def extract_traces_from_bin(
    bin_path: Path,
    inline_indices: np.ndarray,
    xline_indices: np.ndarray,
) -> np.ndarray:
    """
    Extract traces at specified grid locations from one offset bin.

    Args:
        bin_path: Path to migration bin directory
        inline_indices: Inline indices to extract
        xline_indices: Crossline indices to extract

    Returns:
        Traces array of shape (n_inlines, n_xlines, nt)
    """
    zarr_path = bin_path / "migrated_stack.zarr"
    store = zarr.storage.LocalStore(str(zarr_path))
    z = zarr.open_array(store=store, mode='r')

    n_il = len(inline_indices)
    n_xl = len(xline_indices)
    nt = z.shape[2]

    # Extract traces at sparse locations
    # Use advanced indexing - need to handle this carefully for zarr
    traces = np.zeros((n_il, n_xl, nt), dtype=np.float32)

    for i, il_idx in enumerate(inline_indices):
        # Read one inline at a time for efficiency
        inline_data = z[il_idx, :, :]  # (ny, nt)
        traces[i, :, :] = inline_data[xline_indices, :]

    return traces


def get_offset_for_bin(bin_path: Path, bin_num: int) -> float:
    """
    Get representative offset value for an offset bin.

    Reads from bin_headers.parquet if available, otherwise estimates from bin number.
    """
    headers_path = bin_path / "bin_headers.parquet"

    if headers_path.exists():
        df = pl.read_parquet(headers_path)
        if 'offset_avg' in df.columns:
            # Get mean offset across all bins with data
            valid_offsets = df.filter(pl.col('trace_count') > 0)['offset_avg']
            if len(valid_offsets) > 0:
                return float(valid_offsets.mean())

    # Fallback: estimate from bin number (assuming 50m bins starting at 0)
    bin_size = 50.0
    return bin_num * bin_size + bin_size / 2


def extract_cigs(config: CIGConfig) -> bool:
    """
    Main CIG extraction function.

    Returns:
        True if successful
    """
    print("=" * 70)
    print("Common Image Gather (CIG) Extraction")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Find offset bins
    print("[1] Finding offset bins...")
    offset_bins = find_offset_bins(config.input_dir)

    if not offset_bins:
        print(f"ERROR: No migration bins found in {config.input_dir}")
        return False

    print(f"    Found {len(offset_bins)} offset bins:")
    for bin_num, bin_path in offset_bins:
        print(f"      Bin {bin_num:02d}: {bin_path.name}")
    print()

    # Load metadata from first bin
    print("[2] Loading grid metadata...")
    _, first_bin_path = offset_bins[0]
    metadata = load_migration_metadata(first_bin_path)

    print(f"    Grid size: {metadata['nx']} IL x {metadata['ny']} XL x {metadata['nt']} samples")
    print(f"    Grid spacing: dx={metadata['dx']}m, dy={metadata['dy']}m, dt={metadata['dt_ms']}ms")
    print(f"    X range: {metadata['x_min']:.2f} - {metadata['x_max']:.2f}")
    print(f"    Y range: {metadata['y_min']:.2f} - {metadata['y_max']:.2f}")
    print()

    # Determine CIG locations
    print("[3] Computing CIG locations...")
    inline_indices, xline_indices, x_coords, y_coords = extract_cig_locations(config, metadata)

    n_il = len(inline_indices)
    n_xl = len(xline_indices)
    n_cig = n_il * n_xl
    n_offsets = len(offset_bins)
    nt = metadata['nt']

    print(f"    Inline step: {config.inline_step} -> {n_il} inlines")
    print(f"    Crossline step: {config.xline_step} -> {n_xl} crosslines")
    print(f"    Total CIG locations: {n_cig}")
    print(f"    Traces per CIG: {n_offsets} (one per offset bin)")
    print(f"    Total output traces: {n_cig * n_offsets}")
    print()

    # Extract traces from all bins
    print("[4] Extracting traces from offset bins...")
    print("-" * 70)

    start_time = time.time()

    # Storage for all CIG data
    # Shape: (n_inlines, n_xlines, n_offsets, nt)
    all_traces = np.zeros((n_il, n_xl, n_offsets, nt), dtype=np.float32)
    offset_values = np.zeros(n_offsets, dtype=np.float32)

    for i, (bin_num, bin_path) in enumerate(offset_bins):
        t0 = time.time()

        # Extract traces
        traces = extract_traces_from_bin(bin_path, inline_indices, xline_indices)
        all_traces[:, :, i, :] = traces

        # Get offset value
        offset_values[i] = get_offset_for_bin(bin_path, bin_num)

        elapsed = time.time() - t0
        print(f"    Bin {bin_num:02d}: extracted {n_il}x{n_xl} traces "
              f"(offset ~{offset_values[i]:.0f}m) in {elapsed:.1f}s")

    extraction_time = time.time() - start_time
    print("-" * 70)
    print(f"    Total extraction time: {extraction_time:.1f}s")
    print()

    # Sort by offset (should already be sorted, but ensure it)
    print("[5] Sorting CIGs by offset...")
    sort_order = np.argsort(offset_values)
    offset_values = offset_values[sort_order]
    all_traces = all_traces[:, :, sort_order, :]

    print(f"    Offset range: {offset_values[0]:.0f} - {offset_values[-1]:.0f} m")
    print(f"    Offset values: {[f'{o:.0f}' for o in offset_values]}")
    print()

    # Reshape for output: flatten spatial dimensions, keep offset and time
    # Output shape: (n_traces, nt) where n_traces = n_il * n_xl * n_offsets
    print("[6] Preparing output data...")

    # Create header data for each trace
    headers_data = []

    for i_il, il_idx in enumerate(inline_indices):
        for i_xl, xl_idx in enumerate(xline_indices):
            for i_off, offset in enumerate(offset_values):
                headers_data.append({
                    'trace_index': len(headers_data),
                    'inline': int(il_idx + 1),  # 1-based inline number
                    'xline': int(xl_idx + 1),   # 1-based crossline number
                    'il_idx': int(il_idx),      # 0-based index
                    'xl_idx': int(xl_idx),      # 0-based index
                    'x': float(x_coords[i_il]),
                    'y': float(y_coords[i_xl]),
                    'offset': float(offset),
                    'offset_bin': int(i_off),
                })

    headers_df = pl.DataFrame(headers_data)

    # Reshape traces to 2D: (n_traces, nt)
    # Order: inline (slowest) -> xline -> offset (fastest)
    traces_2d = all_traces.reshape(-1, nt)

    print(f"    Output traces shape: {traces_2d.shape}")
    print(f"    Output headers: {len(headers_df)} rows")
    print()

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Write zarr traces
    print("[7] Writing output files...")
    traces_path = config.output_dir / "cig_traces.zarr"
    headers_path = config.output_dir / "cig_headers.parquet"

    print(f"    Writing traces to: {traces_path}")
    traces_store = zarr.storage.LocalStore(str(traces_path))
    traces_zarr = zarr.create_array(
        store=traces_store,
        shape=traces_2d.shape,
        dtype=np.float32,
        chunks=(min(1000, traces_2d.shape[0]), nt),
        overwrite=True,
    )
    traces_zarr[:] = traces_2d

    # Add metadata
    traces_zarr.attrs['n_traces'] = int(traces_2d.shape[0])
    traces_zarr.attrs['n_samples'] = int(nt)
    traces_zarr.attrs['sample_rate_ms'] = float(metadata['dt_ms'])
    traces_zarr.attrs['t_min_ms'] = float(metadata['t_min_ms'])
    traces_zarr.attrs['t_max_ms'] = float(metadata['t_max_ms'])
    traces_zarr.attrs['n_inlines'] = int(n_il)
    traces_zarr.attrs['n_xlines'] = int(n_xl)
    traces_zarr.attrs['n_offsets'] = int(n_offsets)
    traces_zarr.attrs['inline_step'] = config.inline_step
    traces_zarr.attrs['xline_step'] = config.xline_step
    traces_zarr.attrs['offset_values'] = offset_values.tolist()
    traces_zarr.attrs['inline_indices'] = inline_indices.tolist()
    traces_zarr.attrs['xline_indices'] = xline_indices.tolist()
    traces_zarr.attrs['description'] = 'Common Image Gathers extracted from PSTM offset bins'

    print(f"    Writing headers to: {headers_path}")
    headers_df.write_parquet(headers_path)

    # Also write a summary parquet with CIG locations only (one row per CIG)
    cig_locations_path = config.output_dir / "cig_locations.parquet"
    cig_locations = []
    for i_il, il_idx in enumerate(inline_indices):
        for i_xl, xl_idx in enumerate(xline_indices):
            cig_locations.append({
                'cig_index': len(cig_locations),
                'inline': int(il_idx + 1),
                'xline': int(xl_idx + 1),
                'il_idx': int(il_idx),
                'xl_idx': int(xl_idx),
                'x': float(x_coords[i_il]),
                'y': float(y_coords[i_xl]),
                'n_offsets': int(n_offsets),
            })
    cig_locations_df = pl.DataFrame(cig_locations)
    cig_locations_df.write_parquet(cig_locations_path)
    print(f"    Writing CIG locations to: {cig_locations_path}")

    print()

    # Summary
    total_time = time.time() - start_time
    data_size_mb = traces_2d.nbytes / (1024 * 1024)

    print("=" * 70)
    print("CIG Extraction Summary")
    print("=" * 70)
    print(f"  Grid sampling: every {config.inline_step} IL x {config.xline_step} XL")
    print(f"  CIG locations: {n_cig} ({n_il} IL x {n_xl} XL)")
    print(f"  Offset bins: {n_offsets}")
    print(f"  Total traces: {traces_2d.shape[0]:,}")
    print(f"  Samples per trace: {nt}")
    print(f"  Data size: {data_size_mb:.1f} MB")
    print(f"  Total time: {total_time:.1f}s")
    print()
    print(f"  Output directory: {config.output_dir}")
    print(f"    - cig_traces.zarr: Trace amplitudes (n_traces, nt)")
    print(f"    - cig_headers.parquet: Headers with IL/XL/X/Y/offset")
    print(f"    - cig_locations.parquet: CIG location summary")
    print("=" * 70)

    return True


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract Common Image Gathers from PSTM migration output"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Volumes/AO_DISK/PSTM_common_offset"),
        help="Input directory containing migration_bin_XX folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/AO_DISK/PSTM_common_offset/cig_output"),
        help="Output directory for CIG data",
    )
    parser.add_argument(
        "--inline-step",
        type=int,
        default=10,
        help="Extract every Nth inline (default: 10)",
    )
    parser.add_argument(
        "--xline-step",
        type=int,
        default=20,
        help="Extract every Mth crossline (default: 20)",
    )

    args = parser.parse_args()

    # Create configuration
    config = CIGConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        inline_step=args.inline_step,
        xline_step=args.xline_step,
    )

    # Run extraction
    success = extract_cigs(config)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
