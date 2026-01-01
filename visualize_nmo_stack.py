#!/usr/bin/env python3
"""
Visualize NMO-stacked volume with inline, crossline, and time slices.

Similar to stacked_figures/ output format.

Usage:
    python visualize_nmo_stack.py
    python visualize_nmo_stack.py --input /path/to/nmo_stack.zarr
    python visualize_nmo_stack.py --output-dir ./my_figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr
import matplotlib.pyplot as plt


# Default paths
DEFAULT_ZARR = "/Users/olegadamovich/SeismicData/PSTM_common_offset/nmo_stack_stretch90.zarr"
DEFAULT_OUTPUT_DIR = "/Users/olegadamovich/pstm/nmo_stack_figures"


def load_volume(zarr_path: Path) -> tuple[np.ndarray, dict]:
    """Load volume from zarr."""
    print(f"Loading volume: {zarr_path}")
    store = zarr.open(str(zarr_path), mode='r')
    data = np.asarray(store)
    attrs = dict(store.attrs)

    print(f"  Shape: {data.shape}")
    print(f"  Data range: {data.min():.6f} to {data.max():.6f}")
    print(f"  Non-zero: {np.count_nonzero(data):,} / {data.size:,} ({100*np.count_nonzero(data)/data.size:.1f}%)")

    if attrs:
        x_min = attrs.get('x_min', 'N/A')
        x_max = attrs.get('x_max', 'N/A')
        y_min = attrs.get('y_min', 'N/A')
        y_max = attrs.get('y_max', 'N/A')
        t_min = attrs.get('t_min_ms', 0)
        t_max = attrs.get('t_max_ms', 'N/A')

        if isinstance(x_min, (int, float)):
            print(f"  X range: {x_min:.1f} - {x_max:.1f}")
        if isinstance(y_min, (int, float)):
            print(f"  Y range: {y_min:.1f} - {y_max:.1f}")
        if isinstance(t_max, (int, float)):
            print(f"  Time: {t_min:.0f} - {t_max:.0f} ms")
        if 'total_traces' in attrs:
            print(f"  Total traces stacked: {attrs['total_traces']:,}")
        if 'stretch_mute_percent' in attrs:
            print(f"  Stretch mute: {attrs['stretch_mute_percent']}%")

    return data, attrs


def plot_seismic_slice(data, x_axis, y_axis, title, xlabel, ylabel, ax,
                       clip_percentile=99, cmap='gray'):
    """Plot a seismic slice with proper amplitude scaling."""
    # Clip for display
    if np.any(data != 0):
        vmax = np.percentile(np.abs(data[data != 0]), clip_percentile)
    else:
        vmax = 1.0
    vmin = -vmax

    extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
    im = ax.imshow(data.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, interpolation='bilinear')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return im


def create_figures(data: np.ndarray, attrs: dict, output_dir: Path, prefix: str = "stacked"):
    """Create all visualization figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, nt = data.shape
    dt_ms = attrs.get('dt_ms', 2.0)

    # Create axis arrays
    il_axis = np.arange(1, nx + 1)  # Inline numbers
    xl_axis = np.arange(1, ny + 1)  # Crossline numbers
    t_axis = np.arange(nt) * dt_ms  # Time in ms

    # Select slices - evenly distributed
    inline_indices = [int(nx * 0.2), int(nx * 0.5), int(nx * 0.8)]
    crossline_indices = [int(ny * 0.2), int(ny * 0.5), int(ny * 0.8)]

    # Time slices at 500, 1000, 1500 ms (or proportional if different range)
    t_max = t_axis[-1]
    time_ms_targets = [t_max * 0.25, t_max * 0.5, t_max * 0.75]
    time_indices = [int(t / dt_ms) for t in time_ms_targets]
    time_indices = [min(i, nt - 1) for i in time_indices]

    print(f"\nGenerating slices...")
    print(f"  Inlines: {[il_axis[i] for i in inline_indices]}")
    print(f"  Crosslines: {[xl_axis[i] for i in crossline_indices]}")
    print(f"  Times: {[f'{t_axis[i]:.0f}' for i in time_indices]} ms")

    # === Inline Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    stretch_info = f" (Stretch Mute: {attrs.get('stretch_mute_percent', 'N/A')}%)" if 'stretch_mute_percent' in attrs else ""
    fig.suptitle(f'NMO Stack - Inline Slices{stretch_info}', fontsize=14, fontweight='bold')

    for i, il_idx in enumerate(inline_indices):
        ax = axes[i]
        slice_data = data[il_idx, :, :]
        plot_seismic_slice(slice_data, xl_axis, t_axis,
                          f'Inline {il_axis[il_idx]}',
                          'Crossline', 'Time (ms)', ax)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_inline_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Inline slices saved")

    # === Crossline Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f'NMO Stack - Crossline Slices{stretch_info}', fontsize=14, fontweight='bold')

    for i, xl_idx in enumerate(crossline_indices):
        ax = axes[i]
        slice_data = data[:, xl_idx, :]
        plot_seismic_slice(slice_data, il_axis, t_axis,
                          f'Crossline {xl_axis[xl_idx]}',
                          'Inline', 'Time (ms)', ax)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_crossline_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Crossline slices saved")

    # === Time Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'NMO Stack - Time Slices{stretch_info}', fontsize=14, fontweight='bold')

    for i, t_idx in enumerate(time_indices):
        ax = axes[i]
        slice_data = data[:, :, t_idx]
        plot_seismic_slice(slice_data, il_axis, xl_axis,
                          f'Time = {t_axis[t_idx]:.0f} ms',
                          'Inline', 'Crossline', ax)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_time_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Time slices saved")

    # === Combined figure ===
    fig = plt.figure(figsize=(20, 16))
    title = f'NMO Stack Results{stretch_info}'
    if 'total_traces' in attrs:
        title += f'\n{attrs["total_traces"]:,} traces from {attrs.get("n_bins_stacked", "?")} offset bins'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Row 1: Inline slices
    for i, il_idx in enumerate(inline_indices):
        ax = fig.add_subplot(3, 3, i + 1)
        slice_data = data[il_idx, :, :]
        plot_seismic_slice(slice_data, xl_axis, t_axis,
                          f'Inline {il_axis[il_idx]}',
                          'Crossline', 'Time (ms)', ax)

    # Row 2: Crossline slices
    for i, xl_idx in enumerate(crossline_indices):
        ax = fig.add_subplot(3, 3, i + 4)
        slice_data = data[:, xl_idx, :]
        plot_seismic_slice(slice_data, il_axis, t_axis,
                          f'Crossline {xl_axis[xl_idx]}',
                          'Inline', 'Time (ms)', ax)

    # Row 3: Time slices
    for i, t_idx in enumerate(time_indices):
        ax = fig.add_subplot(3, 3, i + 7)
        slice_data = data[:, :, t_idx]
        plot_seismic_slice(slice_data, il_axis, xl_axis,
                          f'Time = {t_axis[t_idx]:.0f} ms',
                          'Inline', 'Crossline', ax)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_all_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Combined figure saved")

    print(f"\nOutput saved to: {output_dir}/")
    print(f"  - {prefix}_inline_slices.png")
    print(f"  - {prefix}_crossline_slices.png")
    print(f"  - {prefix}_time_slices.png")
    print(f"  - {prefix}_all_slices.png")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Visualize NMO stack volume")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_ZARR,
                       help="Input zarr volume path")
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory for figures")
    parser.add_argument("--prefix", "-p", type=str, default="stacked",
                       help="Prefix for output filenames")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("The NMO stack may still be processing. Wait for it to complete.")
        return 1

    # Load and visualize
    data, attrs = load_volume(args.input)
    create_figures(data, attrs, args.output_dir, args.prefix)

    return 0


if __name__ == "__main__":
    sys.exit(main())
