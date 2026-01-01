#!/usr/bin/env python3
"""
Visualize Common Image Gathers (CIGs) from PSTM offset bin migrations.

CIGs show amplitude as a function of offset and time at specific (IL, XL) locations.
Flat events in CIGs indicate correct velocity; curved events indicate velocity errors.
"""

import argparse
import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Configuration
MIGRATION_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/cig_qc")

# Grid parameters
DT_MS = 2.0
T_MIN_MS = 0.0
T_MAX_MS = 2000.0


def get_available_bins(migration_dir: Path) -> list[tuple[int, float]]:
    """Get list of available offset bins with their mean offsets.

    Returns:
        List of (bin_number, mean_offset) tuples sorted by offset
    """
    bins = []
    for d in migration_dir.iterdir():
        if d.is_dir() and d.name.startswith("migration_bin_"):
            zarr_path = d / "migrated_stack.zarr"
            if zarr_path.exists():
                try:
                    bin_num = int(d.name.replace("migration_bin_", ""))
                    # Try to get offset from common_offset_gathers headers
                    header_path = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new") / f"offset_bin_{bin_num:02d}" / "headers.parquet"
                    if header_path.exists():
                        import polars as pl
                        df = pl.read_parquet(header_path)
                        mean_offset = float(df['offset'].mean())
                    else:
                        # Estimate offset from bin number (50m bins starting at ~25m)
                        mean_offset = 25 + bin_num * 50
                    bins.append((bin_num, mean_offset))
                except (ValueError, Exception):
                    continue
    return sorted(bins, key=lambda x: x[1])


def load_cig_data(migration_dir: Path, bins: list[tuple[int, float]],
                  il_idx: int, xl_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Load CIG data for a specific (IL, XL) location.

    Args:
        migration_dir: Path to migration output directory
        bins: List of (bin_number, mean_offset) tuples
        il_idx: Inline index (0-based)
        xl_idx: Crossline index (0-based)

    Returns:
        Tuple of (cig_data [n_offsets, n_times], offsets [n_offsets])
    """
    traces = []
    offsets = []

    for bin_num, offset in bins:
        zarr_path = migration_dir / f"migration_bin_{bin_num:02d}" / "migrated_stack.zarr"
        try:
            store = zarr.open_array(zarr_path, mode='r')
            trace = store[il_idx, xl_idx, :]
            traces.append(np.asarray(trace))
            offsets.append(offset)
        except Exception as e:
            print(f"  Warning: Could not load bin {bin_num}: {e}")
            continue

    if not traces:
        raise ValueError("No traces loaded")

    return np.array(traces), np.array(offsets)


def plot_cig(cig_data: np.ndarray, offsets: np.ndarray, t_axis: np.ndarray,
             title: str, ax: plt.Axes, clip_percentile: float = 99,
             cmap: str = 'gray', wiggle: bool = False):
    """Plot a single CIG panel.

    Args:
        cig_data: Array of shape [n_offsets, n_times]
        offsets: Array of offset values in meters
        t_axis: Array of time values in ms
        title: Plot title
        ax: Matplotlib axes
        clip_percentile: Percentile for amplitude clipping
        cmap: Colormap name
        wiggle: If True, overlay wiggle traces
    """
    # Amplitude scaling
    vmax = np.percentile(np.abs(cig_data), clip_percentile)
    vmin = -vmax

    # Plot as image
    extent = [offsets[0], offsets[-1], t_axis[-1], t_axis[0]]
    im = ax.imshow(cig_data.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, interpolation='bilinear')

    # Optional wiggle overlay
    if wiggle and len(offsets) <= 20:
        scale = (offsets[-1] - offsets[0]) / len(offsets) * 0.8
        for i, offset in enumerate(offsets):
            trace = cig_data[i] / (vmax + 1e-10) * scale
            ax.plot(offset + trace, t_axis, 'k-', linewidth=0.5, alpha=0.7)

    ax.set_xlabel('Offset (m)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)

    return im


def plot_cig_wiggle(cig_data: np.ndarray, offsets: np.ndarray, t_axis: np.ndarray,
                    title: str, ax: plt.Axes, clip_percentile: float = 99,
                    fill_positive: bool = True):
    """Plot CIG as wiggle traces with variable area fill.

    Args:
        cig_data: Array of shape [n_offsets, n_times]
        offsets: Array of offset values in meters
        t_axis: Array of time values in ms
        title: Plot title
        ax: Matplotlib axes
        clip_percentile: Percentile for amplitude clipping
        fill_positive: If True, fill positive amplitudes
    """
    # Amplitude scaling
    vmax = np.percentile(np.abs(cig_data), clip_percentile)
    scale = (offsets[-1] - offsets[0]) / len(offsets) * 0.4

    ax.set_xlim(offsets[0] - scale, offsets[-1] + scale)
    ax.set_ylim(t_axis[-1], t_axis[0])

    for i, offset in enumerate(offsets):
        trace = cig_data[i] / (vmax + 1e-10) * scale
        ax.plot(offset + trace, t_axis, 'k-', linewidth=0.5)

        if fill_positive:
            # Fill positive amplitudes
            trace_clipped = np.clip(trace, 0, None)
            ax.fill_betweenx(t_axis, offset, offset + trace_clipped,
                           color='black', alpha=0.7)

    ax.set_xlabel('Offset (m)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Common Image Gathers (CIGs) from PSTM migration"
    )
    parser.add_argument(
        "--locations", "-l",
        type=str,
        default="256,214;150,100;350,300",
        help="CIG locations as 'IL,XL;IL,XL;...' (default: 256,214;150,100;350,300)"
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=0,
        help="Minimum time to display (ms)"
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=2000,
        help="Maximum time to display (ms)"
    )
    parser.add_argument(
        "--wiggle",
        action="store_true",
        help="Use wiggle display instead of density"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for images"
    )
    parser.add_argument(
        "--migration-dir",
        type=Path,
        default=MIGRATION_DIR,
        help="Migration output directory"
    )

    args = parser.parse_args()

    # Parse locations
    locations = []
    for loc_str in args.locations.split(";"):
        il, xl = map(int, loc_str.strip().split(","))
        locations.append((il, xl))

    print("=" * 60)
    print("CIG Visualization")
    print("=" * 60)

    # Get available bins
    print("\nScanning for offset bins...")
    bins = get_available_bins(args.migration_dir)
    print(f"  Found {len(bins)} offset bins")
    print(f"  Offset range: {bins[0][1]:.0f} - {bins[-1][1]:.0f} m")

    # Create time axis
    n_times = int((T_MAX_MS - T_MIN_MS) / DT_MS) + 1
    t_axis = np.linspace(T_MIN_MS, T_MAX_MS, n_times)

    # Time window for display
    t_min_idx = int(args.t_min / DT_MS)
    t_max_idx = int(args.t_max / DT_MS)
    t_display = t_axis[t_min_idx:t_max_idx+1]

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load and plot CIGs
    print(f"\nGenerating CIGs for {len(locations)} locations...")

    # Individual CIG plots
    for il, xl in locations:
        print(f"\n  Loading CIG at IL={il}, XL={xl}...")
        try:
            cig_data, offsets = load_cig_data(args.migration_dir, bins, il-1, xl-1)
            cig_display = cig_data[:, t_min_idx:t_max_idx+1]

            # Density plot
            fig, ax = plt.subplots(figsize=(12, 10))
            plot_cig(cig_display, offsets, t_display,
                    f'CIG at IL={il}, XL={xl} ({len(offsets)} offsets)',
                    ax, clip_percentile=99)
            plt.colorbar(ax.images[0], ax=ax, label='Amplitude')
            plt.tight_layout()
            plt.savefig(args.output / f"cig_il{il}_xl{xl}_density.png", dpi=150)
            plt.close()
            print(f"    - Saved density plot")

            # Wiggle plot
            fig, ax = plt.subplots(figsize=(12, 10))
            plot_cig_wiggle(cig_display, offsets, t_display,
                           f'CIG at IL={il}, XL={xl} ({len(offsets)} offsets)',
                           ax, clip_percentile=99)
            plt.tight_layout()
            plt.savefig(args.output / f"cig_il{il}_xl{xl}_wiggle.png", dpi=150)
            plt.close()
            print(f"    - Saved wiggle plot")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Combined figure with all CIGs
    n_locs = len(locations)
    if n_locs > 0:
        fig, axes = plt.subplots(2, n_locs, figsize=(6*n_locs, 16))
        if n_locs == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(f'Common Image Gathers ({len(bins)} Offset Bins)',
                    fontsize=14, fontweight='bold')

        for i, (il, xl) in enumerate(locations):
            try:
                cig_data, offsets = load_cig_data(args.migration_dir, bins, il-1, xl-1)
                cig_display = cig_data[:, t_min_idx:t_max_idx+1]

                # Density plot
                plot_cig(cig_display, offsets, t_display,
                        f'IL={il}, XL={xl}',
                        axes[0, i], clip_percentile=99)

                # Wiggle plot
                plot_cig_wiggle(cig_display, offsets, t_display,
                               f'IL={il}, XL={xl}',
                               axes[1, i], clip_percentile=99)

            except Exception as e:
                axes[0, i].text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                               transform=axes[0, i].transAxes)
                axes[1, i].text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                               transform=axes[1, i].transAxes)

        axes[0, 0].set_ylabel('Time (ms)\n(Density)', fontsize=10)
        axes[1, 0].set_ylabel('Time (ms)\n(Wiggle)', fontsize=10)

        plt.tight_layout()
        plt.savefig(args.output / "cig_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  - Saved summary figure")

    # Print statistics
    print(f"\n{'='*60}")
    print("CIG Statistics")
    print(f"{'='*60}")
    for il, xl in locations:
        try:
            cig_data, offsets = load_cig_data(args.migration_dir, bins, il-1, xl-1)
            print(f"\nIL={il}, XL={xl}:")
            print(f"  Offsets: {len(offsets)} ({offsets[0]:.0f} - {offsets[-1]:.0f} m)")
            print(f"  Amplitude range: {cig_data.min():.6f} to {cig_data.max():.6f}")
            print(f"  RMS: {np.sqrt(np.mean(cig_data**2)):.6f}")
        except Exception as e:
            print(f"\nIL={il}, XL={xl}: Error - {e}")

    print(f"\nOutput saved to: {args.output}/")


if __name__ == "__main__":
    main()
