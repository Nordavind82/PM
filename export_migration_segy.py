#!/usr/bin/env python3
"""
Export PSTM Migration Results to SEG-Y Format.

Exports migrated common offset volumes to SEG-Y files with proper header mapping:
- Inline/Crossline numbers
- CDP X/Y coordinates
- Offset values
- Trace coordinates

Usage:
    python export_migration_segy.py [--bins 0-37] [--output-dir PATH]
    python export_migration_segy.py --bins 10,15,20  # specific bins
    python export_migration_segy.py --swap-il-xl    # swap inline/crossline assignment
"""

import argparse
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import segyio
import zarr


# =============================================================================
# Configuration
# =============================================================================

MIGRATION_DIR = Path("/Volumes/AO_DISK/PSTM_common_offset")
INPUT_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
OUTPUT_DIR = Path("/Volumes/AO_DISK/PSTM_common_offset/segy_export")

# Coordinate scalar (negative means divide)
COORD_SCALAR = -100  # Coordinates stored as integers * 100

# Grid corners (will be populated from zarr attributes)
GRID_CORNERS = {
    'c1': (0, 0),
    'c2': (0, 0),
    'c3': (0, 0),
    'c4': (0, 0),
}


# =============================================================================
# Coordinate Computation
# =============================================================================

def compute_grid_coordinates_from_bounds(
    nx: int, ny: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute X and Y coordinates for axis-aligned grid from bounding box.

    Args:
        nx, ny: Grid dimensions
        x_min, x_max: X coordinate range
        y_min, y_max: Y coordinate range

    Returns:
        x_coords: (nx, ny) array of X coordinates
        y_coords: (nx, ny) array of Y coordinates
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def get_offset_for_bin(bin_num: int, input_dir: Path) -> float:
    """Get mean offset value for an offset bin from input headers."""
    headers_path = input_dir / f"offset_bin_{bin_num:02d}" / "headers.parquet"

    if headers_path.exists():
        df = pl.read_parquet(headers_path)
        if 'offset' in df.columns and len(df) > 0:
            return float(df['offset'].mean())

    # Fallback: estimate from bin number (50m bins)
    return bin_num * 50.0 + 25.0


# =============================================================================
# SEG-Y Export Functions
# =============================================================================

def create_text_header(bin_num: int, offset_m: float, nx: int, ny: int,
                       nt: int, dt_ms: float, x_min: float, x_max: float,
                       y_min: float, y_max: float) -> bytes:
    """Create EBCDIC text header for SEG-Y file."""
    lines = [
        f"C01 PSTM MIGRATION - COMMON OFFSET BIN {bin_num:02d}",
        f"C02 OFFSET: {offset_m:.0f} M (MEAN)",
        f"C03 ",
        f"C04 GRID: {nx} INLINES X {ny} CROSSLINES",
        f"C05 SAMPLES: {nt} @ {dt_ms:.1f} MS",
        f"C06 ",
        f"C07 COORDINATE SYSTEM: UTM",
        f"C08 COORDINATE SCALAR: {COORD_SCALAR}",
        f"C09 ",
        f"C10 TRACE HEADER MAPPING:",
        f"C11   INLINE:     BYTES 189-192 (INT32)",
        f"C12   CROSSLINE:  BYTES 193-196 (INT32)",
        f"C13   CDP X:      BYTES 181-184 (INT32)",
        f"C14   CDP Y:      BYTES 185-188 (INT32)",
        f"C15   OFFSET:     BYTES 37-40   (INT32)",
        f"C16   COORD SCALAR: BYTES 71-72 (INT16)",
        f"C17 ",
        f"C18 GRID BOUNDS:",
        f"C19   X RANGE: {x_min:.2f} - {x_max:.2f}",
        f"C20   Y RANGE: {y_min:.2f} - {y_max:.2f}",
        f"C21 ",
        f"C22 ",
        f"C23 ",
        f"C24 PROCESSING: KIRCHHOFF PSTM",
        f"C25 ALGORITHM: METAL GPU ACCELERATED",
        f"C26 ",
        f"C27 EXPORTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"C28 ",
        f"C29 ",
        f"C30 ",
        f"C31 ",
        f"C32 ",
        f"C33 ",
        f"C34 ",
        f"C35 ",
        f"C36 ",
        f"C37 ",
        f"C38 ",
        f"C39 ",
        f"C40 END TEXTUAL HEADER",
    ]

    # Pad each line to 80 characters and join
    text = ""
    for line in lines:
        text += line[:80].ljust(80)

    # Convert to EBCDIC
    return text.encode('cp500')


def export_bin_to_segy(bin_num: int, migration_dir: Path, input_dir: Path,
                       output_dir: Path, verbose: bool = True,
                       swap_il_xl: bool = False) -> Path:
    """
    Export a single offset bin to SEG-Y format.

    Args:
        bin_num: Offset bin number
        migration_dir: Directory containing migration results
        input_dir: Directory containing input data (for offset values)
        output_dir: Output directory for SEG-Y files
        verbose: Print progress messages
        swap_il_xl: If True, swap inline/crossline assignment (use if IL/XL appear swapped)

    Returns:
        Path to output SEG-Y file
    """
    bin_path = migration_dir / f"migration_bin_{bin_num:02d}"
    zarr_path = bin_path / "migrated_stack.zarr"

    if not zarr_path.exists():
        raise FileNotFoundError(f"Migration data not found: {zarr_path}")

    # Load data
    if verbose:
        print(f"  Loading migrated volume...")

    store = zarr.storage.LocalStore(str(zarr_path))
    z = zarr.open_array(store=store, mode='r')
    data = np.asarray(z)
    attrs = dict(z.attrs)

    nx, ny, nt = data.shape
    dt_ms = attrs.get('dt_ms', 2.0)
    dt_us = int(dt_ms * 1000)  # Convert to microseconds

    # Get coordinate bounds from zarr attributes
    x_min = attrs.get('x_min', 0)
    x_max = attrs.get('x_max', nx - 1)
    y_min = attrs.get('y_min', 0)
    y_max = attrs.get('y_max', ny - 1)

    # Get offset
    offset_m = get_offset_for_bin(bin_num, input_dir)

    # Determine inline/crossline counts based on swap setting
    if swap_il_xl:
        n_inlines, n_xlines = ny, nx
        if verbose:
            print(f"  Grid: {n_inlines} IL x {n_xlines} XL x {nt} samples (SWAPPED)")
    else:
        n_inlines, n_xlines = nx, ny
        if verbose:
            print(f"  Grid: {n_inlines} IL x {n_xlines} XL x {nt} samples")

    if verbose:
        print(f"  Sample rate: {dt_ms} ms")
        print(f"  Offset: {offset_m:.0f} m")
        print(f"  X range: {x_min:.1f} - {x_max:.1f}")
        print(f"  Y range: {y_min:.1f} - {y_max:.1f}")

    # Compute coordinates from bounds
    if verbose:
        print(f"  Computing grid coordinates...")

    x_coords, y_coords = compute_grid_coordinates_from_bounds(nx, ny, x_min, x_max, y_min, y_max)

    # Scale coordinates for storage
    coord_scale = abs(COORD_SCALAR)
    x_scaled = (x_coords * coord_scale).astype(np.int32)
    y_scaled = (y_coords * coord_scale).astype(np.int32)

    # Output path
    suffix = "_swapped" if swap_il_xl else ""
    output_path = output_dir / f"pstm_offset_bin_{bin_num:02d}{suffix}.sgy"

    # Create SEG-Y spec
    spec = segyio.spec()
    spec.sorting = 2  # Crossline sorting (inline-major order)
    spec.format = 1   # IBM float
    spec.samples = np.arange(nt) * dt_ms
    spec.ilines = np.arange(1, n_inlines + 1)
    spec.xlines = np.arange(1, n_xlines + 1)

    if verbose:
        print(f"  Writing SEG-Y file...")
        print(f"  Total traces: {n_inlines * n_xlines:,}")

    # Create file
    with segyio.create(str(output_path), spec) as f:
        # Set text header
        f.text[0] = create_text_header(bin_num, offset_m, n_inlines, n_xlines, nt, dt_ms,
                                       x_min, x_max, y_min, y_max)

        # Set binary header
        f.bin[segyio.BinField.Samples] = nt
        f.bin[segyio.BinField.Interval] = dt_us
        f.bin[segyio.BinField.Format] = 1  # IBM float
        f.bin[segyio.BinField.Traces] = n_inlines * n_xlines

        # Write traces
        trace_num = 0

        # Iterate based on swap setting
        if swap_il_xl:
            # Swapped: iterate y (now inline) first, then x (now crossline)
            for il_idx in range(ny):
                for xl_idx in range(nx):
                    # Get trace data (data is still [x, y, t])
                    trace = data[xl_idx, il_idx, :].astype(np.float32)

                    # Write trace
                    f.trace[trace_num] = trace

                    # Set trace headers
                    header = f.header[trace_num]

                    # Trace identification
                    header[segyio.TraceField.TRACE_SEQUENCE_LINE] = trace_num + 1
                    header[segyio.TraceField.TRACE_SEQUENCE_FILE] = trace_num + 1
                    header[segyio.TraceField.FieldRecord] = il_idx + 1
                    header[segyio.TraceField.TraceNumber] = xl_idx + 1

                    # Inline/Crossline (swapped)
                    header[segyio.TraceField.INLINE_3D] = il_idx + 1
                    header[segyio.TraceField.CROSSLINE_3D] = xl_idx + 1

                    # Coordinates (note: accessing with swapped indices)
                    header[segyio.TraceField.CDP_X] = x_scaled[xl_idx, il_idx]
                    header[segyio.TraceField.CDP_Y] = y_scaled[xl_idx, il_idx]
                    header[segyio.TraceField.SourceX] = x_scaled[xl_idx, il_idx]
                    header[segyio.TraceField.SourceY] = y_scaled[xl_idx, il_idx]
                    header[segyio.TraceField.GroupX] = x_scaled[xl_idx, il_idx]
                    header[segyio.TraceField.GroupY] = y_scaled[xl_idx, il_idx]

                    # Coordinate scalar
                    header[segyio.TraceField.SourceGroupScalar] = COORD_SCALAR

                    # CDP (using swapped dimensions)
                    header[segyio.TraceField.CDP] = il_idx * nx + xl_idx + 1
                    header[segyio.TraceField.CDP_TRACE] = 1

                    # Offset
                    header[segyio.TraceField.offset] = int(offset_m)

                    # Sample info
                    header[segyio.TraceField.TRACE_SAMPLE_COUNT] = nt
                    header[segyio.TraceField.TRACE_SAMPLE_INTERVAL] = dt_us

                    # Delay recording time
                    header[segyio.TraceField.DelayRecordingTime] = int(attrs.get('t_min_ms', 0))

                    trace_num += 1

                # Progress update
                if verbose and (il_idx + 1) % 50 == 0:
                    print(f"    Progress: {il_idx + 1}/{ny} inlines ({100*(il_idx+1)/ny:.0f}%)")
        else:
            # Normal: iterate x (inline) first, then y (crossline)
            for il_idx in range(nx):
                for xl_idx in range(ny):
                    # Get trace data
                    trace = data[il_idx, xl_idx, :].astype(np.float32)

                    # Write trace
                    f.trace[trace_num] = trace

                    # Set trace headers
                    header = f.header[trace_num]

                    # Trace identification
                    header[segyio.TraceField.TRACE_SEQUENCE_LINE] = trace_num + 1
                    header[segyio.TraceField.TRACE_SEQUENCE_FILE] = trace_num + 1
                    header[segyio.TraceField.FieldRecord] = il_idx + 1
                    header[segyio.TraceField.TraceNumber] = xl_idx + 1

                    # Inline/Crossline
                    header[segyio.TraceField.INLINE_3D] = il_idx + 1
                    header[segyio.TraceField.CROSSLINE_3D] = xl_idx + 1

                    # CDP (same as IL/XL for post-stack)
                    header[segyio.TraceField.CDP] = il_idx * ny + xl_idx + 1
                    header[segyio.TraceField.CDP_TRACE] = 1

                    # Offset
                    header[segyio.TraceField.offset] = int(offset_m)

                    # Coordinates
                    header[segyio.TraceField.CDP_X] = x_scaled[il_idx, xl_idx]
                    header[segyio.TraceField.CDP_Y] = y_scaled[il_idx, xl_idx]
                    header[segyio.TraceField.SourceX] = x_scaled[il_idx, xl_idx]
                    header[segyio.TraceField.SourceY] = y_scaled[il_idx, xl_idx]
                    header[segyio.TraceField.GroupX] = x_scaled[il_idx, xl_idx]
                    header[segyio.TraceField.GroupY] = y_scaled[il_idx, xl_idx]

                    # Coordinate scalar
                    header[segyio.TraceField.SourceGroupScalar] = COORD_SCALAR

                    # Sample info
                    header[segyio.TraceField.TRACE_SAMPLE_COUNT] = nt
                    header[segyio.TraceField.TRACE_SAMPLE_INTERVAL] = dt_us

                    # Delay recording time
                    header[segyio.TraceField.DelayRecordingTime] = int(attrs.get('t_min_ms', 0))

                    trace_num += 1

                # Progress update
                if verbose and (il_idx + 1) % 50 == 0:
                    print(f"    Progress: {il_idx + 1}/{nx} inlines ({100*(il_idx+1)/nx:.0f}%)")

    if verbose:
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Output: {output_path.name} ({file_size:.1f} MB)")

    return output_path


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export PSTM migration results to SEG-Y format"
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="all",
        help="Bins to export: 'all', '0-37', or '0,5,10' (default: all)",
    )
    parser.add_argument(
        "--migration-dir",
        type=Path,
        default=MIGRATION_DIR,
        help="Migration output directory",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Input data directory (for offset values)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for SEG-Y files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without writing files",
    )
    parser.add_argument(
        "--swap-il-xl",
        action="store_true",
        help="Swap inline/crossline assignment (use if IL/XL appear swapped in viewer)",
    )

    args = parser.parse_args()

    # Parse bins
    if args.bins.lower() == "all":
        bins = list(range(40))
    elif "-" in args.bins and "," not in args.bins:
        start, end = map(int, args.bins.split("-"))
        bins = list(range(start, end + 1))
    else:
        bins = [int(b.strip()) for b in args.bins.split(",")]

    # Filter to existing bins
    bins = [b for b in bins
            if (args.migration_dir / f"migration_bin_{b:02d}" / "migrated_stack.zarr").exists()]

    print("=" * 70)
    print("PSTM Migration SEG-Y Export")
    print("=" * 70)
    print(f"Migration directory: {args.migration_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Bins to export: {len(bins)} ({min(bins)}-{max(bins)})")
    if args.swap_il_xl:
        print("IL/XL SWAP: Enabled")
    print()

    if args.dry_run:
        suffix = "_swapped" if args.swap_il_xl else ""
        print("DRY RUN - would export:")
        for bin_num in bins:
            offset = get_offset_for_bin(bin_num, args.input_dir)
            print(f"  Bin {bin_num:02d}: offset ~{offset:.0f}m -> pstm_offset_bin_{bin_num:02d}{suffix}.sgy")
        return 0

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Export bins
    total_start = time.time()
    results = []

    for i, bin_num in enumerate(bins):
        print(f"\n[{i+1}/{len(bins)}] Exporting Bin {bin_num:02d}")
        print("-" * 50)

        try:
            start = time.time()
            output_path = export_bin_to_segy(
                bin_num,
                args.migration_dir,
                args.input_dir,
                args.output_dir,
                swap_il_xl=args.swap_il_xl,
            )
            elapsed = time.time() - start
            results.append({
                'bin': bin_num,
                'success': True,
                'elapsed': elapsed,
                'path': output_path,
            })
            print(f"  Completed in {elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'bin': bin_num,
                'success': False,
                'error': str(e),
            })

    total_elapsed = time.time() - total_start

    # Summary
    print()
    print("=" * 70)
    print("Export Summary")
    print("=" * 70)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"Total bins: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")

    if successful:
        total_size = sum(r['path'].stat().st_size for r in successful) / (1024**3)
        print(f"Total output size: {total_size:.2f} GB")

    if failed:
        print("\nFailed exports:")
        for r in failed:
            print(f"  Bin {r['bin']:02d}: {r['error']}")

    print()
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
