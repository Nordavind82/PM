#!/usr/bin/env python3
"""
Stack Common Offset PSTM Bins into Final PSTM Volume.

Combines multiple common offset migrated gathers into a single stacked volume
with options for:
- Offset range selection (min/max)
- Top mute application (offset-dependent or constant)
- Fold-weighted or simple stacking
- Output sorted by inline then crossline

Usage:
    python stack_offset_bins.py --output /path/to/output.zarr
    python stack_offset_bins.py --offset-min 100 --offset-max 2000
    python stack_offset_bins.py --mute-file mute_picks.csv
    python stack_offset_bins.py --mute-velocity 1500 --mute-offset 100
"""

import argparse
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import zarr
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# Default paths
MIGRATION_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset")

INPUT_HEADERS_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
DEFAULT_OUTPUT = MIGRATION_DIR / "pstm_stacked.zarr"

# Grid parameters (must match migration)
DX = 25.0   # Inline spacing (m)
DY = 12.5   # Crossline spacing (m)
DT_MS = 2.0 # Time sample interval (ms)
T_MIN_MS = 0.0
T_MAX_MS = 2000.0

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Mute Functions
# =============================================================================

class MuteFunction:
    """Base class for mute functions."""

    def get_mute_time_ms(self, offset: float) -> float:
        """Return mute time in ms for given offset."""
        raise NotImplementedError


class ConstantMute(MuteFunction):
    """Constant mute time regardless of offset."""

    def __init__(self, mute_time_ms: float):
        self.mute_time_ms = mute_time_ms

    def get_mute_time_ms(self, offset: float) -> float:
        return self.mute_time_ms


class LinearMute(MuteFunction):
    """Linear mute: t_mute = t0 + offset / velocity."""

    def __init__(self, t0_ms: float, velocity_m_s: float):
        self.t0_ms = t0_ms
        self.velocity = velocity_m_s

    def get_mute_time_ms(self, offset: float) -> float:
        return self.t0_ms + (abs(offset) / self.velocity) * 1000.0


class HyperbolicMute(MuteFunction):
    """Hyperbolic mute: t_mute = sqrt(t0^2 + (offset/velocity)^2)."""

    def __init__(self, t0_ms: float, velocity_m_s: float):
        self.t0_ms = t0_ms
        self.velocity = velocity_m_s

    def get_mute_time_ms(self, offset: float) -> float:
        t0_s = self.t0_ms / 1000.0
        offset_time = abs(offset) / self.velocity
        return np.sqrt(t0_s**2 + offset_time**2) * 1000.0


class TableMute(MuteFunction):
    """Mute from offset-time table (interpolated)."""

    def __init__(self, offsets: np.ndarray, times_ms: np.ndarray):
        self.offsets = np.array(offsets)
        self.times_ms = np.array(times_ms)

    def get_mute_time_ms(self, offset: float) -> float:
        return float(np.interp(abs(offset), self.offsets, self.times_ms))

    @classmethod
    def from_csv(cls, csv_path: Path) -> 'TableMute':
        """Load mute table from CSV file with columns: offset, time_ms."""
        df = pl.read_csv(csv_path)
        return cls(
            offsets=df['offset'].to_numpy(),
            times_ms=df['time_ms'].to_numpy()
        )


def apply_mute_to_volume(
    data: np.ndarray,
    mute_time_ms: float,
    dt_ms: float,
    taper_samples: int = 10
) -> np.ndarray:
    """
    Apply top mute to a 3D volume.

    Args:
        data: 3D array (nx, ny, nt)
        mute_time_ms: Mute time in milliseconds
        dt_ms: Sample interval in milliseconds
        taper_samples: Number of samples for cosine taper

    Returns:
        Muted data array
    """
    mute_sample = int(mute_time_ms / dt_ms)
    nt = data.shape[2]

    if mute_sample <= 0:
        return data

    if mute_sample >= nt:
        return np.zeros_like(data)

    # Create mute taper
    mute_array = np.ones(nt, dtype=np.float32)

    # Zero before mute
    if mute_sample > taper_samples:
        mute_array[:mute_sample - taper_samples] = 0.0
        # Cosine taper
        taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_samples)))
        mute_array[mute_sample - taper_samples:mute_sample] = taper
    else:
        mute_array[:mute_sample] = 0.0

    # Apply to all traces
    return data * mute_array[np.newaxis, np.newaxis, :]


# =============================================================================
# Stacking Functions
# =============================================================================

def get_available_bins(migration_dir: Path) -> list[tuple[int, float, int]]:
    """
    Get list of available migrated bins with their offset info.

    Returns:
        List of (bin_num, mean_offset, n_traces)
    """
    bins = []

    for d in sorted(migration_dir.iterdir()):
        if d.is_dir() and d.name.startswith("migration_bin_"):
            try:
                bin_num = int(d.name.replace("migration_bin_", ""))
                stack_path = d / "migrated_stack.zarr"

                if not stack_path.exists():
                    continue

                # Get offset info from input headers
                headers_path = INPUT_HEADERS_DIR / f"offset_bin_{bin_num:02d}" / "headers.parquet"
                if headers_path.exists():
                    df = pl.read_parquet(headers_path)
                    mean_offset = float(df['offset'].mean()) if 'offset' in df.columns else bin_num * 50.0
                    n_traces = len(df)
                else:
                    mean_offset = bin_num * 50.0
                    n_traces = 0

                bins.append((bin_num, mean_offset, n_traces))

            except (ValueError, Exception) as e:
                logger.warning(f"Error processing {d.name}: {e}")
                continue

    return sorted(bins, key=lambda x: x[1])  # Sort by offset


def load_migration_stack(migration_dir: Path, bin_num: int) -> Optional[np.ndarray]:
    """Load migrated stack for an offset bin."""
    stack_path = migration_dir / f"migration_bin_{bin_num:02d}" / "migrated_stack.zarr"

    if not stack_path.exists():
        return None

    try:
        z = zarr.open_array(stack_path, mode='r')
        return z[:]
    except Exception as e:
        logger.error(f"Error loading bin {bin_num}: {e}")
        return None


def load_fold(migration_dir: Path, bin_num: int) -> Optional[np.ndarray]:
    """Load fold for an offset bin."""
    fold_path = migration_dir / f"migration_bin_{bin_num:02d}" / "fold.zarr"

    if not fold_path.exists():
        return None

    try:
        z = zarr.open_array(fold_path, mode='r')
        return z[:]
    except Exception as e:
        logger.warning(f"Error loading fold for bin {bin_num}: {e}")
        return None


def stack_offset_bins(
    migration_dir: Path,
    output_path: Path,
    offset_min: float = 0.0,
    offset_max: float = float('inf'),
    mute_function: Optional[MuteFunction] = None,
    mute_taper_samples: int = 10,
    use_fold_weights: bool = True,
    normalize_by_fold: bool = True,
) -> dict:
    """
    Stack common offset PSTM bins into final volume.

    Args:
        migration_dir: Directory containing migration_bin_XX folders
        output_path: Path for output zarr
        offset_min: Minimum offset to include
        offset_max: Maximum offset to include
        mute_function: Optional mute function to apply
        mute_taper_samples: Taper length for mute
        use_fold_weights: Weight by fold during stacking
        normalize_by_fold: Normalize final stack by total fold

    Returns:
        Dictionary with stacking statistics
    """
    logger.info("=" * 60)
    logger.info("Stacking Common Offset PSTM Bins")
    logger.info("=" * 60)

    # Get available bins
    available_bins = get_available_bins(migration_dir)
    logger.info(f"Found {len(available_bins)} migrated offset bins")

    # Filter by offset range
    selected_bins = [
        (bn, off, nt) for bn, off, nt in available_bins
        if offset_min <= off <= offset_max
    ]

    if not selected_bins:
        raise ValueError(f"No bins found in offset range [{offset_min}, {offset_max}]")

    logger.info(f"Selected {len(selected_bins)} bins in offset range [{offset_min:.0f}, {offset_max:.0f}] m")
    for bn, off, nt in selected_bins:
        logger.info(f"  Bin {bn:02d}: offset={off:.0f}m, traces={nt:,}")

    # Load first bin to get dimensions
    first_bin = selected_bins[0][0]
    first_data = load_migration_stack(migration_dir, first_bin)
    if first_data is None:
        raise RuntimeError(f"Cannot load first bin {first_bin}")

    nx, ny, nt = first_data.shape
    logger.info(f"Grid dimensions: nx={nx}, ny={ny}, nt={nt}")
    logger.info(f"Grid size: {nx*DX/1000:.1f}km x {ny*DY/1000:.1f}km x {nt*DT_MS:.0f}ms")

    # Initialize accumulators
    stack_sum = np.zeros((nx, ny, nt), dtype=np.float64)
    fold_sum = np.zeros((nx, ny, nt), dtype=np.float64)
    n_bins_stacked = 0

    # Process each bin
    logger.info("")
    logger.info("Stacking bins...")

    for bin_num, mean_offset, n_traces in tqdm(selected_bins, desc="Stacking"):
        # Load data
        data = load_migration_stack(migration_dir, bin_num)
        if data is None:
            logger.warning(f"Skipping bin {bin_num}: cannot load data")
            continue

        # Verify dimensions
        if data.shape != (nx, ny, nt):
            logger.warning(f"Skipping bin {bin_num}: shape mismatch {data.shape} vs {(nx, ny, nt)}")
            continue

        # Apply mute if specified
        if mute_function is not None:
            mute_time = mute_function.get_mute_time_ms(mean_offset)
            data = apply_mute_to_volume(data, mute_time, DT_MS, mute_taper_samples)
            logger.debug(f"Bin {bin_num}: applied mute at {mute_time:.0f}ms for offset {mean_offset:.0f}m")

        # Load fold if using weights
        if use_fold_weights:
            fold = load_fold(migration_dir, bin_num)
            if fold is not None and fold.shape == (nx, ny, nt):
                # Weight by fold
                stack_sum += data.astype(np.float64) * fold.astype(np.float64)
                fold_sum += fold.astype(np.float64)
            else:
                # Fall back to unit weight
                stack_sum += data.astype(np.float64)
                fold_sum += np.where(data != 0, 1.0, 0.0)
        else:
            # Simple sum
            stack_sum += data.astype(np.float64)
            fold_sum += np.where(data != 0, 1.0, 0.0)

        n_bins_stacked += 1

        # Free memory
        del data
        gc.collect()

    logger.info(f"Stacked {n_bins_stacked} offset bins")

    # Normalize by fold
    if normalize_by_fold:
        logger.info("Normalizing by fold...")
        with np.errstate(divide='ignore', invalid='ignore'):
            stack_final = np.where(fold_sum > 0, stack_sum / fold_sum, 0.0).astype(np.float32)
    else:
        stack_final = stack_sum.astype(np.float32)

    # Compute statistics
    nonzero_mask = stack_final != 0
    if np.any(nonzero_mask):
        data_min = float(stack_final[nonzero_mask].min())
        data_max = float(stack_final[nonzero_mask].max())
        data_rms = float(np.sqrt(np.mean(stack_final[nonzero_mask]**2)))
    else:
        data_min = data_max = data_rms = 0.0

    fold_min = int(fold_sum.min())
    fold_max = int(fold_sum.max())
    fold_mean = float(fold_sum.mean())

    logger.info(f"Stack statistics:")
    logger.info(f"  Amplitude: min={data_min:.6f}, max={data_max:.6f}, RMS={data_rms:.6f}")
    logger.info(f"  Fold: min={fold_min}, max={fold_max}, mean={fold_mean:.1f}")

    # Save output
    logger.info(f"Saving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create zarr with metadata
    z = zarr.open_array(
        output_path,
        mode='w',
        shape=stack_final.shape,
        dtype=np.float32,
        chunks=(64, 64, nt),
    )
    z[:] = stack_final

    # Add metadata
    z.attrs['description'] = 'PSTM Stacked Volume'
    z.attrs['created'] = datetime.now().isoformat()
    z.attrs['nx'] = nx
    z.attrs['ny'] = ny
    z.attrs['nt'] = nt
    z.attrs['dx'] = DX
    z.attrs['dy'] = DY
    z.attrs['dt_ms'] = DT_MS
    z.attrs['t_min_ms'] = T_MIN_MS
    z.attrs['t_max_ms'] = T_MAX_MS
    z.attrs['n_bins_stacked'] = n_bins_stacked
    z.attrs['offset_min'] = offset_min
    z.attrs['offset_max'] = offset_max
    z.attrs['bins_used'] = [bn for bn, _, _ in selected_bins]
    z.attrs['offsets_used'] = [off for _, off, _ in selected_bins]

    # Save fold as separate array
    fold_path = output_path.parent / f"{output_path.stem}_fold.zarr"
    z_fold = zarr.open_array(
        fold_path,
        mode='w',
        shape=fold_sum.shape,
        dtype=np.float32,
        chunks=(64, 64, nt),
    )
    z_fold[:] = fold_sum.astype(np.float32)

    logger.info(f"Saved fold to: {fold_path}")

    # Return statistics
    stats = {
        'n_bins_stacked': n_bins_stacked,
        'bins_used': [bn for bn, _, _ in selected_bins],
        'offsets_used': [off for _, off, _ in selected_bins],
        'nx': nx,
        'ny': ny,
        'nt': nt,
        'data_min': data_min,
        'data_max': data_max,
        'data_rms': data_rms,
        'fold_min': fold_min,
        'fold_max': fold_max,
        'fold_mean': fold_mean,
        'output_path': str(output_path),
        'fold_path': str(fold_path),
    }

    logger.info("=" * 60)
    logger.info("Stacking complete!")
    logger.info("=" * 60)

    return stats


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stack common offset PSTM bins into final volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stack all available bins
  python stack_offset_bins.py

  # Stack with offset range selection
  python stack_offset_bins.py --offset-min 100 --offset-max 2000

  # Stack with linear mute (t = t0 + offset/velocity)
  python stack_offset_bins.py --mute-t0 50 --mute-velocity 1500

  # Stack with hyperbolic mute
  python stack_offset_bins.py --mute-t0 50 --mute-velocity 2000 --mute-type hyperbolic

  # Stack with mute from CSV file (columns: offset, time_ms)
  python stack_offset_bins.py --mute-file mute_picks.csv

  # Stack with constant mute
  python stack_offset_bins.py --mute-constant 200
        """
    )

    # Input/Output
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=MIGRATION_DIR,
        help=f"Directory containing migration_bin_XX folders (default: {MIGRATION_DIR})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output zarr path (default: {DEFAULT_OUTPUT})"
    )

    # Offset selection
    parser.add_argument(
        "--offset-min",
        type=float,
        default=0.0,
        help="Minimum offset to include (m) (default: 0)"
    )
    parser.add_argument(
        "--offset-max",
        type=float,
        default=float('inf'),
        help="Maximum offset to include (m) (default: inf)"
    )
    parser.add_argument(
        "--bins",
        type=str,
        default=None,
        help="Specific bins to include: '0,1,2,3' or '0-10' (default: all in offset range)"
    )

    # Mute options
    mute_group = parser.add_argument_group("Mute options")
    mute_group.add_argument(
        "--mute-type",
        choices=["linear", "hyperbolic", "constant", "table"],
        default="linear",
        help="Type of mute function (default: linear)"
    )
    mute_group.add_argument(
        "--mute-t0",
        type=float,
        default=None,
        help="Mute intercept time at zero offset (ms)"
    )
    mute_group.add_argument(
        "--mute-velocity",
        type=float,
        default=None,
        help="Mute velocity (m/s) for linear/hyperbolic mute"
    )
    mute_group.add_argument(
        "--mute-constant",
        type=float,
        default=None,
        help="Constant mute time (ms) for all offsets"
    )
    mute_group.add_argument(
        "--mute-file",
        type=Path,
        default=None,
        help="CSV file with mute picks (columns: offset, time_ms)"
    )
    mute_group.add_argument(
        "--mute-taper",
        type=int,
        default=10,
        help="Mute taper length in samples (default: 10)"
    )

    # Stacking options
    stack_group = parser.add_argument_group("Stacking options")
    stack_group.add_argument(
        "--no-fold-weights",
        action="store_true",
        help="Don't use fold weights during stacking"
    )
    stack_group.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize final stack by fold"
    )

    # Other
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without stacking"
    )

    args = parser.parse_args()

    # Build mute function
    mute_function = None

    if args.mute_constant is not None:
        mute_function = ConstantMute(args.mute_constant)
        logger.info(f"Using constant mute at {args.mute_constant}ms")

    elif args.mute_file is not None:
        if not args.mute_file.exists():
            print(f"ERROR: Mute file not found: {args.mute_file}")
            return 1
        mute_function = TableMute.from_csv(args.mute_file)
        logger.info(f"Using mute table from {args.mute_file}")

    elif args.mute_t0 is not None and args.mute_velocity is not None:
        if args.mute_type == "hyperbolic":
            mute_function = HyperbolicMute(args.mute_t0, args.mute_velocity)
            logger.info(f"Using hyperbolic mute: t0={args.mute_t0}ms, v={args.mute_velocity}m/s")
        else:
            mute_function = LinearMute(args.mute_t0, args.mute_velocity)
            logger.info(f"Using linear mute: t0={args.mute_t0}ms, v={args.mute_velocity}m/s")

    # Get available bins
    available_bins = get_available_bins(args.input_dir)

    if not available_bins:
        print(f"ERROR: No migrated bins found in {args.input_dir}")
        return 1

    print(f"\nAvailable offset bins: {len(available_bins)}")
    print(f"{'Bin':>4} {'Offset (m)':>12} {'Traces':>10}")
    print("-" * 30)
    for bn, off, nt in available_bins:
        print(f"{bn:4d} {off:12.0f} {nt:10,}")

    # Filter by offset range
    selected = [
        (bn, off, nt) for bn, off, nt in available_bins
        if args.offset_min <= off <= args.offset_max
    ]

    print(f"\nSelected for stacking: {len(selected)} bins")
    print(f"Offset range: [{args.offset_min:.0f}, {args.offset_max:.0f}] m")

    if args.dry_run:
        print("\nDRY RUN - would stack the following bins:")
        for bn, off, nt in selected:
            mute_str = ""
            if mute_function:
                mute_time = mute_function.get_mute_time_ms(off)
                mute_str = f", mute={mute_time:.0f}ms"
            print(f"  Bin {bn:02d}: offset={off:.0f}m{mute_str}")
        return 0

    # Run stacking
    try:
        stats = stack_offset_bins(
            migration_dir=args.input_dir,
            output_path=args.output,
            offset_min=args.offset_min,
            offset_max=args.offset_max,
            mute_function=mute_function,
            mute_taper_samples=args.mute_taper,
            use_fold_weights=not args.no_fold_weights,
            normalize_by_fold=not args.no_normalize,
        )

        print(f"\nOutput: {stats['output_path']}")
        print(f"Fold:   {stats['fold_path']}")

        return 0

    except Exception as e:
        logger.error(f"Stacking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
