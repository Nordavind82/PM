#!/usr/bin/env python3
"""
QC Visualization: Inline and Crossline slices for Common Offset PSTM Gathers.

Creates comparison figures showing IL/XL slices across 8 offset bins.
"""

import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path("/Volumes/AO_DISK/PSTM_common_offset")
QC_DIR = OUTPUT_DIR / "qc_slices"
QC_DIR.mkdir(exist_ok=True)

# Grid parameters
DX = 25.0   # Inline bin size (m)
DY = 12.5   # Crossline bin size (m)
DT_MS = 2.0 # Time sample interval (ms)

# Visualization parameters
CLIP_PERCENTILE = 99  # For amplitude clipping
T_MAX_DISPLAY = 2000  # Max time to display (ms)

def load_migration_stack(bin_num: int) -> tuple[np.ndarray, dict]:
    """Load migrated stack for an offset bin."""
    bin_dir = OUTPUT_DIR / f"migration_bin_{bin_num:02d}"
    stack_path = bin_dir / "migrated_stack.zarr"

    if not stack_path.exists():
        return None, None

    z = zarr.open_array(stack_path, mode='r')
    data = z[:]

    # Get metadata
    attrs = dict(z.attrs) if hasattr(z, 'attrs') else {}

    return data, attrs


def get_offset_info(bin_num: int) -> float:
    """Get mean offset for a bin from headers."""
    import polars as pl

    headers_path = Path(f"/Users/olegadamovich/SeismicData/common_offset_gathers_new/offset_bin_{bin_num:02d}/headers.parquet")
    if headers_path.exists():
        df = pl.read_parquet(headers_path)
        if 'offset' in df.columns:
            return float(df['offset'].mean())
    return bin_num * 100.0  # Fallback estimate


def plot_inline_comparison(bins: list[int], inline_idx: int, output_path: Path):
    """Plot inline slices for multiple offset bins."""
    n_bins = len(bins)

    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, bin_num in enumerate(bins):
        ax = axes[i]
        data, attrs = load_migration_stack(bin_num)

        if data is None:
            ax.text(0.5, 0.5, f"Bin {bin_num:02d}\nNo data",
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"Offset Bin {bin_num:02d}")
            continue

        # Extract inline slice (x=inline_idx, all y, all t)
        nx, ny, nt = data.shape
        if inline_idx >= nx:
            inline_idx = nx // 2

        inline_slice = data[inline_idx, :, :].T  # (nt, ny)

        # Clip amplitudes
        clip_val = np.percentile(np.abs(inline_slice[inline_slice != 0]), CLIP_PERCENTILE) if np.any(inline_slice != 0) else 1.0

        # Get offset info
        offset = get_offset_info(bin_num)

        # Time axis
        t_max_idx = min(int(T_MAX_DISPLAY / DT_MS), nt)

        # Plot
        im = ax.imshow(inline_slice[:t_max_idx, :],
                       aspect='auto',
                       cmap='gray',
                       vmin=-clip_val, vmax=clip_val,
                       extent=[0, ny*DY, t_max_idx*DT_MS, 0])

        ax.set_title(f"Bin {bin_num:02d} | Offset: {offset:.0f}m", fontsize=12)
        ax.set_xlabel("Crossline (m)")
        ax.set_ylabel("Time (ms)")

    plt.suptitle(f"Inline {inline_idx} - Common Offset PSTM Comparison\n{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_crossline_comparison(bins: list[int], crossline_idx: int, output_path: Path):
    """Plot crossline slices for multiple offset bins."""
    n_bins = len(bins)

    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, bin_num in enumerate(bins):
        ax = axes[i]
        data, attrs = load_migration_stack(bin_num)

        if data is None:
            ax.text(0.5, 0.5, f"Bin {bin_num:02d}\nNo data",
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"Offset Bin {bin_num:02d}")
            continue

        # Extract crossline slice (all x, y=crossline_idx, all t)
        nx, ny, nt = data.shape
        if crossline_idx >= ny:
            crossline_idx = ny // 2

        crossline_slice = data[:, crossline_idx, :].T  # (nt, nx)

        # Clip amplitudes
        clip_val = np.percentile(np.abs(crossline_slice[crossline_slice != 0]), CLIP_PERCENTILE) if np.any(crossline_slice != 0) else 1.0

        # Get offset info
        offset = get_offset_info(bin_num)

        # Time axis
        t_max_idx = min(int(T_MAX_DISPLAY / DT_MS), nt)

        # Plot
        im = ax.imshow(crossline_slice[:t_max_idx, :],
                       aspect='auto',
                       cmap='gray',
                       vmin=-clip_val, vmax=clip_val,
                       extent=[0, nx*DX, t_max_idx*DT_MS, 0])

        ax.set_title(f"Bin {bin_num:02d} | Offset: {offset:.0f}m", fontsize=12)
        ax.set_xlabel("Inline (m)")
        ax.set_ylabel("Time (ms)")

    plt.suptitle(f"Crossline {crossline_idx} - Common Offset PSTM Comparison\n{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_time_slice_comparison(bins: list[int], time_ms: float, output_path: Path):
    """Plot time slices for multiple offset bins."""
    n_bins = len(bins)

    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, bin_num in enumerate(bins):
        ax = axes[i]
        data, attrs = load_migration_stack(bin_num)

        if data is None:
            ax.text(0.5, 0.5, f"Bin {bin_num:02d}\nNo data",
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"Offset Bin {bin_num:02d}")
            continue

        # Extract time slice
        nx, ny, nt = data.shape
        time_idx = int(time_ms / DT_MS)
        if time_idx >= nt:
            time_idx = nt // 2

        time_slice = data[:, :, time_idx].T  # (ny, nx)

        # Clip amplitudes
        clip_val = np.percentile(np.abs(time_slice[time_slice != 0]), CLIP_PERCENTILE) if np.any(time_slice != 0) else 1.0

        # Get offset info
        offset = get_offset_info(bin_num)

        # Plot
        im = ax.imshow(time_slice,
                       aspect='auto',
                       cmap='gray',
                       vmin=-clip_val, vmax=clip_val,
                       extent=[0, nx*DX, ny*DY, 0])

        ax.set_title(f"Bin {bin_num:02d} | Offset: {offset:.0f}m", fontsize=12)
        ax.set_xlabel("Inline (m)")
        ax.set_ylabel("Crossline (m)")

    plt.suptitle(f"Time Slice @ {time_ms:.0f}ms - Common Offset PSTM Comparison\n{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("QC Visualization: Common Offset PSTM Gathers")
    print("=" * 60)

    # Check which bins are available
    available_bins = []
    for i in range(40):
        bin_dir = OUTPUT_DIR / f"migration_bin_{i:02d}"
        if (bin_dir / "migrated_stack.zarr").exists():
            available_bins.append(i)

    print(f"Available migrated bins: {available_bins}")

    # Use first 8 available bins
    bins_to_plot = available_bins[:8]
    if len(bins_to_plot) < 8:
        print(f"Warning: Only {len(bins_to_plot)} bins available, expected 8")

    print(f"Plotting bins: {bins_to_plot}")

    # Load first bin to get grid dimensions
    data0, attrs0 = load_migration_stack(bins_to_plot[0])
    if data0 is None:
        print("ERROR: Cannot load first bin data")
        return

    nx, ny, nt = data0.shape
    print(f"Grid shape: nx={nx}, ny={ny}, nt={nt}")
    print(f"Grid size: {nx*DX/1000:.1f}km x {ny*DY/1000:.1f}km x {nt*DT_MS:.0f}ms")

    # Generate slice locations
    il_center = nx // 2
    xl_center = ny // 2

    # Multiple inline locations
    inline_indices = [nx//4, nx//2, 3*nx//4]

    # Multiple crossline locations
    crossline_indices = [ny//4, ny//2, 3*ny//4]

    # Time slices
    time_slices = [500, 800, 1000, 1200]

    print()
    print("Generating inline slices...")
    for il_idx in inline_indices:
        output_path = QC_DIR / f"inline_{il_idx:03d}_comparison.png"
        plot_inline_comparison(bins_to_plot, il_idx, output_path)

    print()
    print("Generating crossline slices...")
    for xl_idx in crossline_indices:
        output_path = QC_DIR / f"crossline_{xl_idx:03d}_comparison.png"
        plot_crossline_comparison(bins_to_plot, xl_idx, output_path)

    print()
    print("Generating time slices...")
    for t_ms in time_slices:
        output_path = QC_DIR / f"timeslice_{int(t_ms):04d}ms_comparison.png"
        plot_time_slice_comparison(bins_to_plot, t_ms, output_path)

    print()
    print("=" * 60)
    print(f"QC figures saved to: {QC_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
