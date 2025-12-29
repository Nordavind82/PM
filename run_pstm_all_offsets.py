#!/usr/bin/env python3
"""
Batch PSTM Migration for All Offset Bins.

Runs Kirchhoff PSTM migration for each common offset bin, producing
migrated grids that can be used for CIG extraction.

This script processes offset bins sequentially and outputs to a
consistent directory structure for CIG extraction.

Usage:
    python run_pstm_all_offsets.py [--bins 0-39] [--skip-existing]
    python run_pstm_all_offsets.py --bins 0,5,10,15,20  # specific bins
    python run_pstm_all_offsets.py --min-traces 1000    # skip sparse bins
    python run_pstm_all_offsets.py --continue           # continue from last completed bin
"""

import argparse
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import zarr

# Add pstm to path
sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import (
    MigrationConfig,
    InputConfig,
    VelocityConfig,
    AlgorithmConfig,
    OutputConfig,
    ExecutionConfig,
    OutputGridConfig,
    ApertureConfig,
    AmplitudeConfig,
    ResourceConfig,
    TilingConfig,
    CheckpointConfig,
    VelocitySource,
    InterpolationMethod,
    ComputeBackend,
    OutputFormat,
    TimeVariantConfig,
    AntiAliasingConfig,
    AntiAliasingMethod,
    ColumnMapping,
)
from pstm.pipeline.executor import run_migration, ProgressInfo

# =============================================================================
# Configuration
# =============================================================================

# Input data paths
COMMON_OFFSET_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
VELOCITY_PATH = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm.zarr")

# Output directory - external disk for large migration output
OUTPUT_DIR = Path("/Volumes/AO_DISK/PSTM_common_offset")

# Grid parameters (must match original migration)
DX = 25.0   # Inline bin size (m)
DY = 12.5   # Crossline bin size (m)
DT_MS = 2.0 # Time sample interval (ms)
T_MIN_MS = 0.0
T_MAX_MS = 2000.0

# Grid corners (rotated grid)
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),  # Origin (IL=1, XL=1)
    'c2': (627094.02, 5106803.16),  # Inline end (IL=511, XL=1)
    'c3': (631143.35, 5110261.43),  # Far corner (IL=511, XL=427)
    'c4': (622862.92, 5119956.77),  # Crossline end (IL=1, XL=427)
}

# Algorithm parameters
MAX_APERTURE_M = 2000.0
MIN_APERTURE_M = 500.0
MAX_DIP_DEGREES = 65.0

# Tile size
TILE_NX = 512
TILE_NY = 512

# Time-variant sampling (disabled for CIG quality)
TIME_VARIANT_ENABLED = False
TIME_VARIANT_TABLE = [
    (0.0, 120.0),
    (800.0, 120.0),
    (1200.0, 80.0),
    (2000.0, 50.0),
    (3200.0, 50.0),
]

# Minimum traces per bin to process
MIN_TRACES_DEFAULT = 100


# =============================================================================
# Helper Functions
# =============================================================================

def setup_logging(output_dir: Path, bin_num: int) -> logging.Logger:
    """Setup logging for a migration run."""
    log_file = output_dir / f"pstm_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("pstm.batch")
    logger.setLevel(logging.INFO)

    # File handler
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(fh)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    return logger


def get_available_bins(common_offset_dir: Path) -> list[int]:
    """Get list of available offset bin numbers."""
    bins = []
    for d in common_offset_dir.iterdir():
        if d.is_dir() and d.name.startswith("offset_bin_"):
            try:
                bin_num = int(d.name.replace("offset_bin_", ""))
                # Verify data exists
                if (d / "traces.zarr").exists() and (d / "headers.parquet").exists():
                    bins.append(bin_num)
            except ValueError:
                continue
    return sorted(bins)


def get_bin_info(common_offset_dir: Path, bin_num: int) -> dict:
    """Get information about an offset bin."""
    bin_dir = common_offset_dir / f"offset_bin_{bin_num:02d}"
    headers_path = bin_dir / "headers.parquet"
    traces_path = bin_dir / "traces.zarr"

    info = {
        'bin_num': bin_num,
        'bin_dir': bin_dir,
        'exists': bin_dir.exists(),
        'n_traces': 0,
        'mean_offset': 0.0,
        'min_offset': 0.0,
        'max_offset': 0.0,
    }

    if headers_path.exists():
        df = pl.read_parquet(headers_path)
        info['n_traces'] = len(df)
        if 'offset' in df.columns and len(df) > 0:
            info['mean_offset'] = float(df['offset'].mean())
            info['min_offset'] = float(df['offset'].min())
            info['max_offset'] = float(df['offset'].max())

    return info


def parse_bin_range(bin_spec: str, available_bins: list[int]) -> list[int]:
    """
    Parse bin specification string.

    Formats:
        "0-39" -> range 0 to 39
        "0,5,10,15" -> specific bins
        "all" -> all available bins
    """
    if bin_spec.lower() == "all":
        return available_bins

    if "-" in bin_spec and "," not in bin_spec:
        # Range format: "0-39"
        start, end = bin_spec.split("-")
        requested = list(range(int(start), int(end) + 1))
    else:
        # List format: "0,5,10,15"
        requested = [int(b.strip()) for b in bin_spec.split(",")]

    # Filter to available bins
    return [b for b in requested if b in available_bins]


def check_migration_exists(output_dir: Path, bin_num: int) -> bool:
    """Check if migration output already exists for a bin."""
    bin_output_dir = output_dir / f"migration_bin_{bin_num:02d}"
    migrated_stack = bin_output_dir / "migrated_stack.zarr"
    return migrated_stack.exists()


def find_last_completed_bin(output_dir: Path, available_bins: list[int]) -> int | None:
    """Find the highest numbered bin that has completed migration."""
    completed_bins = [
        bin_num for bin_num in available_bins
        if check_migration_exists(output_dir, bin_num)
    ]
    return max(completed_bins) if completed_bins else None


# =============================================================================
# Migration Function
# =============================================================================

def run_migration_for_bin(
    bin_num: int,
    common_offset_dir: Path,
    output_dir: Path,
    velocity_path: Path,
    logger: logging.Logger,
) -> tuple[bool, float, str]:
    """
    Run PSTM migration for a single offset bin.

    Returns:
        (success, elapsed_time, message)
    """
    start_time = time.time()

    bin_dir = common_offset_dir / f"offset_bin_{bin_num:02d}"
    traces_path = bin_dir / "traces.zarr"
    headers_path = bin_dir / "headers.parquet"

    if not traces_path.exists() or not headers_path.exists():
        return False, 0.0, f"Input data not found for bin {bin_num}"

    # Get bin info
    bin_info = get_bin_info(common_offset_dir, bin_num)
    logger.info(f"  Traces: {bin_info['n_traces']:,}")
    logger.info(f"  Offset range: {bin_info['min_offset']:.0f} - {bin_info['max_offset']:.0f} m")

    # Output directory for this bin
    bin_output_dir = output_dir / f"migration_bin_{bin_num:02d}"
    bin_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Build migration configuration (matching run_pstm_common_offset.py)
        input_config = InputConfig(
            traces_path=traces_path,
            headers_path=headers_path,
            columns=ColumnMapping(
                source_x="source_x",
                source_y="source_y",
                receiver_x="receiver_x",
                receiver_y="receiver_y",
                offset="offset",
                azimuth="sr_azim",
                trace_index="bin_trace_idx",
                coord_scalar="scalar_coord",
            ),
            apply_coord_scalar=True,
            sample_rate_ms=DT_MS,
            transposed=True,
        )

        velocity_config = VelocityConfig(
            source=VelocitySource.CUBE_3D,
            velocity_path=velocity_path,
            precompute_to_output_grid=True,
        )

        aperture_config = ApertureConfig(
            max_dip_degrees=MAX_DIP_DEGREES,
            min_aperture_m=MIN_APERTURE_M,
            max_aperture_m=MAX_APERTURE_M,
            taper_fraction=0.1,
        )

        amplitude_config = AmplitudeConfig(
            geometrical_spreading=False,
            obliquity_factor=False,
        )

        time_variant_config = TimeVariantConfig(
            enabled=TIME_VARIANT_ENABLED,
            frequency_table=TIME_VARIANT_TABLE,
            min_downsample_factor=1,
            max_downsample_factor=8,
        )

        anti_aliasing_config = AntiAliasingConfig(
            enabled=False,
            method=AntiAliasingMethod.NONE,
        )

        algorithm_config = AlgorithmConfig(
            interpolation=InterpolationMethod.LINEAR,
            aperture=aperture_config,
            amplitude=amplitude_config,
            time_variant=time_variant_config,
            anti_aliasing=anti_aliasing_config,
        )

        output_grid = OutputGridConfig.from_corners(
            corner1=GRID_CORNERS['c1'],
            corner2=GRID_CORNERS['c2'],
            corner3=GRID_CORNERS['c3'],
            corner4=GRID_CORNERS['c4'],
            t_min_ms=T_MIN_MS,
            t_max_ms=T_MAX_MS,
            dx=DX,
            dy=DY,
            dt_ms=DT_MS,
        )

        output_config = OutputConfig(
            output_dir=bin_output_dir,
            grid=output_grid,
            format=OutputFormat.ZARR,
        )

        resource_config = ResourceConfig(
            backend=ComputeBackend.METAL_COMPILED,
            max_memory_gb=32.0,
        )

        tiling_config = TilingConfig(
            auto_tile_size=False,
            tile_nx=TILE_NX,
            tile_ny=TILE_NY,
            ordering='snake',
        )

        checkpoint_config = CheckpointConfig(
            enabled=True,
            interval_tiles=10,
        )

        execution_config = ExecutionConfig(
            resources=resource_config,
            tiling=tiling_config,
            checkpoint=checkpoint_config,
        )

        migration_config = MigrationConfig(
            name=f"PSTM_Offset_Bin_{bin_num:02d}",
            input=input_config,
            velocity=velocity_config,
            algorithm=algorithm_config,
            output=output_config,
            execution=execution_config,
        )

        # Run migration using the run_migration function
        run_migration(migration_config, resume=True)

        elapsed = time.time() - start_time

        # Force cleanup
        gc.collect()

        return True, elapsed, "Success"

    except Exception as e:
        elapsed = time.time() - start_time
        import traceback
        return False, elapsed, f"{str(e)}\n{traceback.format_exc()}"


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run PSTM migration for all offset bins"
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="all",
        help="Offset bins to process: 'all', '0-39', or '0,5,10,15,20' (default: all)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=COMMON_OFFSET_DIR,
        help="Input directory with offset bins",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for migration results",
    )
    parser.add_argument(
        "--velocity",
        type=Path,
        default=VELOCITY_PATH,
        help="Path to velocity model (zarr)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip bins that already have migration output",
    )
    parser.add_argument(
        "--continue",
        dest="continue_from_last",
        action="store_true",
        help="Continue from the last completed offset bin",
    )
    parser.add_argument(
        "--min-traces",
        type=int,
        default=MIN_TRACES_DEFAULT,
        help=f"Minimum traces per bin to process (default: {MIN_TRACES_DEFAULT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )

    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, 0)

    print("=" * 70)
    print("PSTM Batch Migration - All Offset Bins")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Find available bins
    available_bins = get_available_bins(args.input_dir)
    print(f"Available offset bins: {len(available_bins)}")
    print(f"  Range: {min(available_bins)} - {max(available_bins)}")
    print()

    # Parse requested bins
    requested_bins = parse_bin_range(args.bins, available_bins)
    print(f"Requested bins: {len(requested_bins)}")

    # Handle --continue flag: start from the bin after the last completed one
    if args.continue_from_last:
        last_completed = find_last_completed_bin(args.output_dir, available_bins)
        if last_completed is not None:
            print(f"Last completed bin: {last_completed:02d}")
            # Filter to only bins after the last completed one
            requested_bins = [b for b in requested_bins if b > last_completed]
            print(f"Continuing from bin {requested_bins[0]:02d}" if requested_bins else "All bins already completed!")
        else:
            print("No completed bins found, starting from the beginning")
        print()

    # Filter by existing and trace count
    bins_to_process = []
    bins_skipped_existing = []
    bins_skipped_sparse = []

    for bin_num in requested_bins:
        bin_info = get_bin_info(args.input_dir, bin_num)

        if args.skip_existing and check_migration_exists(args.output_dir, bin_num):
            bins_skipped_existing.append(bin_num)
            continue

        if bin_info['n_traces'] < args.min_traces:
            bins_skipped_sparse.append((bin_num, bin_info['n_traces']))
            continue

        bins_to_process.append((bin_num, bin_info))

    print(f"Bins to process: {len(bins_to_process)}")
    if bins_skipped_existing:
        print(f"  Skipped (existing): {len(bins_skipped_existing)} - {bins_skipped_existing}")
    if bins_skipped_sparse:
        print(f"  Skipped (sparse <{args.min_traces} traces): {len(bins_skipped_sparse)}")
        for bn, nt in bins_skipped_sparse:
            print(f"    Bin {bn:02d}: {nt} traces")
    print()

    if args.dry_run:
        print("DRY RUN - Bins that would be processed:")
        for bin_num, info in bins_to_process:
            print(f"  Bin {bin_num:02d}: {info['n_traces']:,} traces, "
                  f"offset {info['min_offset']:.0f}-{info['max_offset']:.0f}m")
        return 0

    if not bins_to_process:
        print("No bins to process!")
        return 0

    # Process bins
    print("=" * 70)
    print("Processing Offset Bins")
    print("=" * 70)

    total_start = time.time()
    results = []

    for i, (bin_num, bin_info) in enumerate(bins_to_process):
        print()
        print(f"[{i+1}/{len(bins_to_process)}] Processing Bin {bin_num:02d}")
        print("-" * 50)

        success, elapsed, message = run_migration_for_bin(
            bin_num=bin_num,
            common_offset_dir=args.input_dir,
            output_dir=args.output_dir,
            velocity_path=args.velocity,
            logger=logger,
        )

        results.append({
            'bin': bin_num,
            'success': success,
            'elapsed': elapsed,
            'message': message,
            'n_traces': bin_info['n_traces'],
        })

        if success:
            print(f"  COMPLETED in {elapsed:.1f}s")
        else:
            print(f"  FAILED: {message}")

        # Force garbage collection between bins
        gc.collect()

    total_elapsed = time.time() - total_start

    # Summary
    print()
    print("=" * 70)
    print("Batch Migration Summary")
    print("=" * 70)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"Total bins processed: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print()

    if successful:
        total_traces = sum(r['n_traces'] for r in successful)
        avg_time = sum(r['elapsed'] for r in successful) / len(successful)
        print(f"Successful migrations:")
        print(f"  Total traces migrated: {total_traces:,}")
        print(f"  Average time per bin: {avg_time:.1f}s")

    if failed:
        print()
        print("Failed migrations:")
        for r in failed:
            print(f"  Bin {r['bin']:02d}: {r['message']}")

    print()
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
