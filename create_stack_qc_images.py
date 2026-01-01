#!/usr/bin/env python3
"""
Create QC images for PSTM stacked volume.

Generates individual inline, crossline, and timeslice images
matching the format of reference QC images.
"""

import argparse
import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Default configuration
DEFAULT_PSTM_STACK = "/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/pstm_stacked.zarr"
DEFAULT_FOLD_STACK = "/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/pstm_stacked_fold.zarr"
DEFAULT_OUTPUT_DIR = "/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/stack_qc_images"

# Slice selections
DEFAULT_INLINES = [128, 256, 384]
DEFAULT_CROSSLINES = [107, 214, 321]
DEFAULT_TIMESLICES_MS = [400, 600, 800, 1000, 1200]


def load_zarr_volume(zarr_path):
    """Load zarr volume and its attributes."""
    print(f"Loading: {zarr_path}")
    store = zarr.open_array(zarr_path, mode='r')
    data = np.asarray(store)
    attrs = dict(store.attrs)
    print(f"  Shape: {data.shape}")
    print(f"  Range: {data.min():.6f} to {data.max():.6f}")
    return data, attrs


def plot_inline_slice(data, inline_idx, attrs, output_path, clip_percentile=99):
    """Plot and save a single inline slice."""
    dt_ms = attrs.get('dt_ms', 2.0)
    dy = attrs.get('dy', 12.5)

    slice_data = data[inline_idx, :, :]
    n_xl, n_t = slice_data.shape

    # Create axes
    xl_axis = np.arange(1, n_xl + 1)
    t_axis = np.arange(n_t) * dt_ms

    # Amplitude clipping
    vmax = np.percentile(np.abs(slice_data[slice_data != 0]), clip_percentile) if np.any(slice_data != 0) else 1.0

    fig, ax = plt.subplots(figsize=(14, 10))

    extent = [xl_axis[0], xl_axis[-1], t_axis[-1], t_axis[0]]
    im = ax.imshow(slice_data.T, aspect='auto', cmap='gray',
                   vmin=-vmax, vmax=vmax, extent=extent, interpolation='bilinear')

    ax.set_xlabel('Crossline', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'Inline {inline_idx + 1}', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Amplitude', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_crossline_slice(data, crossline_idx, attrs, output_path, clip_percentile=99):
    """Plot and save a single crossline slice."""
    dt_ms = attrs.get('dt_ms', 2.0)
    dx = attrs.get('dx', 25.0)

    slice_data = data[:, crossline_idx, :]
    n_il, n_t = slice_data.shape

    # Create axes
    il_axis = np.arange(1, n_il + 1)
    t_axis = np.arange(n_t) * dt_ms

    # Amplitude clipping
    vmax = np.percentile(np.abs(slice_data[slice_data != 0]), clip_percentile) if np.any(slice_data != 0) else 1.0

    fig, ax = plt.subplots(figsize=(14, 10))

    extent = [il_axis[0], il_axis[-1], t_axis[-1], t_axis[0]]
    im = ax.imshow(slice_data.T, aspect='auto', cmap='gray',
                   vmin=-vmax, vmax=vmax, extent=extent, interpolation='bilinear')

    ax.set_xlabel('Inline', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'Crossline {crossline_idx + 1}', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Amplitude', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_timeslice(data, time_ms, dt_ms, attrs, output_path, clip_percentile=99):
    """Plot and save a single time slice."""
    time_idx = int(time_ms / dt_ms)

    if time_idx >= data.shape[2]:
        print(f"  Warning: Time {time_ms}ms exceeds data range, skipping")
        return

    slice_data = data[:, :, time_idx]
    n_il, n_xl = slice_data.shape

    # Create axes
    il_axis = np.arange(1, n_il + 1)
    xl_axis = np.arange(1, n_xl + 1)

    # Amplitude clipping
    vmax = np.percentile(np.abs(slice_data[slice_data != 0]), clip_percentile) if np.any(slice_data != 0) else 1.0

    fig, ax = plt.subplots(figsize=(12, 10))

    extent = [il_axis[0], il_axis[-1], xl_axis[-1], xl_axis[0]]
    im = ax.imshow(slice_data.T, aspect='auto', cmap='gray',
                   vmin=-vmax, vmax=vmax, extent=extent, interpolation='bilinear')

    ax.set_xlabel('Inline', fontsize=12)
    ax.set_ylabel('Crossline', fontsize=12)
    ax.set_title(f'Time Slice @ {time_ms:.0f} ms', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Amplitude', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_fold_map(fold_data, attrs, output_path):
    """Plot and save fold map (summed over time)."""
    # Sum fold over time axis to get total fold per CDP
    fold_sum = fold_data.sum(axis=2)

    n_il, n_xl = fold_sum.shape
    il_axis = np.arange(1, n_il + 1)
    xl_axis = np.arange(1, n_xl + 1)

    fig, ax = plt.subplots(figsize=(12, 10))

    extent = [il_axis[0], il_axis[-1], xl_axis[-1], xl_axis[0]]
    im = ax.imshow(fold_sum.T, aspect='auto', cmap='viridis', extent=extent)

    ax.set_xlabel('Inline', fontsize=12)
    ax.set_ylabel('Crossline', fontsize=12)
    ax.set_title('Fold Map (Total Fold per CDP)', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Total Fold', fontsize=10)

    # Add statistics text
    stats_text = f'Min: {fold_sum.min():.0f}  Max: {fold_sum.max():.0f}  Mean: {fold_sum.mean():.0f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Create QC images for PSTM stack')
    parser.add_argument('--stack', type=Path, default=DEFAULT_PSTM_STACK,
                        help='Path to PSTM stacked zarr')
    parser.add_argument('--fold', type=Path, default=DEFAULT_FOLD_STACK,
                        help='Path to fold zarr')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for images')
    parser.add_argument('--inlines', type=int, nargs='+', default=DEFAULT_INLINES,
                        help='Inline indices to plot (1-based)')
    parser.add_argument('--crosslines', type=int, nargs='+', default=DEFAULT_CROSSLINES,
                        help='Crossline indices to plot (1-based)')
    parser.add_argument('--timeslices', type=float, nargs='+', default=DEFAULT_TIMESLICES_MS,
                        help='Time slices to plot in ms')
    parser.add_argument('--clip', type=float, default=99,
                        help='Percentile for amplitude clipping')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load PSTM stack
    data, attrs = load_zarr_volume(args.stack)
    dt_ms = attrs.get('dt_ms', 2.0)

    nx, ny, nt = data.shape
    print(f"Grid: {nx} inlines x {ny} crosslines x {nt} samples")

    # Generate inline slices
    print("\nGenerating inline slices...")
    for il in args.inlines:
        if 1 <= il <= nx:
            output_path = output_dir / f"inline_{il}.png"
            plot_inline_slice(data, il - 1, attrs, output_path, args.clip)
        else:
            print(f"  Warning: Inline {il} out of range [1, {nx}]")

    # Generate crossline slices
    print("\nGenerating crossline slices...")
    for xl in args.crosslines:
        if 1 <= xl <= ny:
            output_path = output_dir / f"crossline_{xl}.png"
            plot_crossline_slice(data, xl - 1, attrs, output_path, args.clip)
        else:
            print(f"  Warning: Crossline {xl} out of range [1, {ny}]")

    # Generate time slices
    print("\nGenerating time slices...")
    max_time = (nt - 1) * dt_ms
    for t_ms in args.timeslices:
        if 0 <= t_ms <= max_time:
            output_path = output_dir / f"timeslice_{int(t_ms):04d}ms.png"
            plot_timeslice(data, t_ms, dt_ms, attrs, output_path, args.clip)
        else:
            print(f"  Warning: Time {t_ms}ms out of range [0, {max_time}]")

    # Generate fold map if available
    if args.fold.exists():
        print("\nGenerating fold map...")
        fold_data, fold_attrs = load_zarr_volume(args.fold)
        plot_fold_map(fold_data, fold_attrs, output_dir / "fold_map.png")
    else:
        print(f"\nFold file not found: {args.fold}")

    print(f"\n{'='*50}")
    print(f"QC images saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
