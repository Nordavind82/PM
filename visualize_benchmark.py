#!/usr/bin/env python3
"""
Benchmark Visualization Script for PSTM Optimization Testing.

Generates standardized QC images for comparing optimization results.

Usage:
    python visualize_benchmark.py --bin 10 --output baseline
    python visualize_benchmark.py --bin 10 --output memory_44gb
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# Configuration
PSTM_OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset")
QC_OUTPUT_DIR = Path("/Users/olegadamovich/pstm/optimization_qc")

# Standard slice locations
INLINE_INDICES = [100, 255, 410]      # Near start, middle, near end
CROSSLINE_INDICES = [85, 213, 340]    # Near start, middle, near end
TIME_MS = [500, 1000, 1500]           # Standard time slices

# Grid parameters
DT_MS = 2.0  # Time sample interval


def load_migrated_volume(bin_num: int):
    """Load migrated volume and fold from zarr."""
    bin_dir = PSTM_OUTPUT_DIR / f"migration_bin_{bin_num:02d}"
    migrated_path = bin_dir / "migrated_stack.zarr"
    fold_path = bin_dir / "fold.zarr"

    if not migrated_path.exists():
        print(f"ERROR: Migrated volume not found: {migrated_path}")
        sys.exit(1)

    print(f"Loading migrated volume from: {migrated_path}")
    data = zarr.open_array(migrated_path, mode='r')
    data = np.asarray(data)
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Range: [{data.min():.6f}, {data.max():.6f}]")

    fold = None
    if fold_path.exists():
        print(f"Loading fold from: {fold_path}")
        fold = np.asarray(zarr.open_array(fold_path, mode='r'))
        print(f"  Fold shape: {fold.shape}")
        print(f"  Fold range: [{fold.min()}, {fold.max()}]")

    return data, fold


def time_to_index(time_ms: float, nt: int) -> int:
    """Convert time in ms to sample index."""
    idx = int(time_ms / DT_MS)
    return min(max(0, idx), nt - 1)


def plot_seismic(ax, data, x_axis, y_axis, title, xlabel, ylabel,
                 clip_percentile=99, cmap='gray'):
    """Plot seismic data with proper scaling."""
    if data.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return None

    vmax = np.percentile(np.abs(data[data != 0]), clip_percentile) if np.any(data != 0) else 1.0
    vmin = -vmax

    extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
    im = ax.imshow(data.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, interpolation='bilinear')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return im


def plot_fold_map(ax, fold, il_axis, xl_axis):
    """Plot fold map."""
    if fold is None:
        ax.text(0.5, 0.5, 'Fold not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Fold Map')
        return None

    extent = [il_axis[0], il_axis[-1], xl_axis[-1], xl_axis[0]]
    im = ax.imshow(fold.T, aspect='auto', cmap='viridis', extent=extent, interpolation='nearest')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title(f'Fold Map (min={fold.min()}, max={fold.max()}, mean={fold.mean():.1f})')
    plt.colorbar(im, ax=ax, label='Fold')

    return im


def generate_qc_images(data, fold, output_dir: Path, bin_num: int):
    """Generate all QC images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, nt = data.shape
    il_axis = np.arange(1, nx + 1)
    xl_axis = np.arange(1, ny + 1)
    t_axis = np.arange(nt) * DT_MS

    print(f"\nGenerating QC images to: {output_dir}")

    # 1. Inline slices
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f'Offset Bin {bin_num:02d} - Inline Slices', fontsize=14, fontweight='bold')

    for i, il_idx in enumerate(INLINE_INDICES):
        if il_idx < nx:
            slice_data = data[il_idx, :, :]
            plot_seismic(axes[i], slice_data, xl_axis, t_axis,
                        f'Inline {il_idx + 1}', 'Crossline', 'Time (ms)')
        else:
            axes[i].set_title(f'Inline {il_idx + 1} (out of range)')

    plt.tight_layout()
    plt.savefig(output_dir / 'inline_slices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - inline_slices.png")

    # 2. Crossline slices
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f'Offset Bin {bin_num:02d} - Crossline Slices', fontsize=14, fontweight='bold')

    for i, xl_idx in enumerate(CROSSLINE_INDICES):
        if xl_idx < ny:
            slice_data = data[:, xl_idx, :]
            plot_seismic(axes[i], slice_data, il_axis, t_axis,
                        f'Crossline {xl_idx + 1}', 'Inline', 'Time (ms)')
        else:
            axes[i].set_title(f'Crossline {xl_idx + 1} (out of range)')

    plt.tight_layout()
    plt.savefig(output_dir / 'crossline_slices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - crossline_slices.png")

    # 3. Time slices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Offset Bin {bin_num:02d} - Time Slices', fontsize=14, fontweight='bold')

    for i, time_ms in enumerate(TIME_MS):
        t_idx = time_to_index(time_ms, nt)
        slice_data = data[:, :, t_idx]
        plot_seismic(axes[i], slice_data, il_axis, xl_axis,
                    f'Time = {time_ms} ms', 'Inline', 'Crossline')

    plt.tight_layout()
    plt.savefig(output_dir / 'time_slices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - time_slices.png")

    # 4. Fold map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_fold_map(ax, fold, il_axis, xl_axis)
    plt.tight_layout()
    plt.savefig(output_dir / 'fold_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - fold_map.png")

    # 5. Summary figure (3x3 grid)
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(f'PSTM Migration QC - Offset Bin {bin_num:02d}', fontsize=16, fontweight='bold')

    # Row 1: Inline slices
    for i, il_idx in enumerate(INLINE_INDICES):
        ax = fig.add_subplot(3, 3, i + 1)
        if il_idx < nx:
            slice_data = data[il_idx, :, :]
            plot_seismic(ax, slice_data, xl_axis, t_axis,
                        f'Inline {il_idx + 1}', 'Crossline', 'Time (ms)')

    # Row 2: Crossline slices
    for i, xl_idx in enumerate(CROSSLINE_INDICES):
        ax = fig.add_subplot(3, 3, i + 4)
        if xl_idx < ny:
            slice_data = data[:, xl_idx, :]
            plot_seismic(ax, slice_data, il_axis, t_axis,
                        f'Crossline {xl_idx + 1}', 'Inline', 'Time (ms)')

    # Row 3: Time slices
    for i, time_ms in enumerate(TIME_MS):
        ax = fig.add_subplot(3, 3, i + 7)
        t_idx = time_to_index(time_ms, nt)
        slice_data = data[:, :, t_idx]
        plot_seismic(ax, slice_data, il_axis, xl_axis,
                    f'Time = {time_ms} ms', 'Inline', 'Crossline')

    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - summary.png")

    # 6. Individual inline/crossline/timeslice files for easy comparison
    for il_idx in INLINE_INDICES:
        if il_idx < nx:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            slice_data = data[il_idx, :, :]
            plot_seismic(ax, slice_data, xl_axis, t_axis,
                        f'Inline {il_idx + 1}', 'Crossline', 'Time (ms)')
            plt.tight_layout()
            plt.savefig(output_dir / f'inline_{il_idx + 1}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  - inline_{il_idx + 1}.png")

    for xl_idx in CROSSLINE_INDICES:
        if xl_idx < ny:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            slice_data = data[:, xl_idx, :]
            plot_seismic(ax, slice_data, il_axis, t_axis,
                        f'Crossline {xl_idx + 1}', 'Inline', 'Time (ms)')
            plt.tight_layout()
            plt.savefig(output_dir / f'crossline_{xl_idx + 1}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  - crossline_{xl_idx + 1}.png")

    for time_ms in TIME_MS:
        t_idx = time_to_index(time_ms, nt)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        slice_data = data[:, :, t_idx]
        plot_seismic(ax, slice_data, il_axis, xl_axis,
                    f'Time = {time_ms} ms', 'Inline', 'Crossline')
        plt.tight_layout()
        plt.savefig(output_dir / f'timeslice_{time_ms}ms.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  - timeslice_{time_ms}ms.png")

    print(f"\nAll QC images saved to: {output_dir}")


def compute_statistics(data, fold):
    """Compute and print statistics."""
    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)

    print(f"\nData Volume:")
    print(f"  Shape: {data.shape}")
    print(f"  Non-zero samples: {np.count_nonzero(data):,} / {data.size:,} ({100*np.count_nonzero(data)/data.size:.1f}%)")
    print(f"  Min: {data.min():.6f}")
    print(f"  Max: {data.max():.6f}")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Std: {data.std():.6f}")
    print(f"  RMS: {np.sqrt(np.mean(data**2)):.6f}")

    if fold is not None:
        print(f"\nFold:")
        print(f"  Min: {fold.min()}")
        print(f"  Max: {fold.max()}")
        print(f"  Mean: {fold.mean():.1f}")
        print(f"  Zero-fold bins: {np.sum(fold == 0):,} / {fold.size:,}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark QC images")
    parser.add_argument("--bin", type=int, default=10, help="Offset bin number (default: 10)")
    parser.add_argument("--output", type=str, default="baseline",
                        help="Output folder name (default: baseline)")
    parser.add_argument("--no-stats", action="store_true", help="Skip statistics output")

    args = parser.parse_args()

    print("=" * 60)
    print("PSTM Benchmark Visualization")
    print("=" * 60)
    print(f"Bin: {args.bin:02d}")
    print(f"Output: {args.output}")

    # Load data
    data, fold = load_migrated_volume(args.bin)

    # Statistics
    if not args.no_stats:
        compute_statistics(data, fold)

    # Generate QC images
    output_dir = QC_OUTPUT_DIR / args.output
    generate_qc_images(data, fold, output_dir, args.bin)

    print("\nDone!")


if __name__ == "__main__":
    main()
