#!/usr/bin/env python3
"""
Visualize migrated PSTM volume with inline, crossline, and time slices.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
MIGRATED_ZARR = "/Users/olegadamovich/SeismicData/PSTM_common_offset/migration_bin_00/migrated_stack.zarr"
OUTPUT_DIR = "/Users/olegadamovich/SeismicData/PSTM_common_offset/migration_qc"

# Grid info
NX, NY, NT = 511, 427, 1001
DT_MS = 3.2


def load_migrated_volume():
    """Load migrated volume from zarr."""
    print("Loading migrated volume...")
    store = zarr.open(MIGRATED_ZARR, mode='r')
    data = np.asarray(store)
    print(f"Shape: {data.shape}")
    print(f"Data range: {data.min():.4f} to {data.max():.4f}")

    # Get metadata
    attrs = dict(store.attrs)
    print(f"X range: {attrs.get('x_min', 'N/A')} - {attrs.get('x_max', 'N/A')}")
    print(f"Y range: {attrs.get('y_min', 'N/A')} - {attrs.get('y_max', 'N/A')}")
    print(f"T range: {attrs.get('t_min_ms', 0)} - {attrs.get('t_max_ms', 'N/A')} ms")

    return data, attrs


def plot_seismic_slice(data, x_axis, y_axis, title, xlabel, ylabel, ax,
                       clip_percentile=99, cmap='gray'):
    """Plot a seismic slice with proper amplitude scaling."""
    # Clip for display
    vmax = np.percentile(np.abs(data), clip_percentile)
    vmin = -vmax

    extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
    im = ax.imshow(data.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, interpolation='bilinear')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return im


def main():
    # Create output directory
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data, attrs = load_migrated_volume()

    # Create axis arrays
    il_axis = np.arange(1, data.shape[0] + 1)  # 1-511 (inline numbers)
    xl_axis = np.arange(1, data.shape[1] + 1)  # 1-427 (crossline numbers)
    t_axis = np.arange(data.shape[2]) * DT_MS  # Time in ms

    # Select slices - evenly distributed
    inline_indices = [100, 255, 410]  # Near start, middle, near end
    crossline_indices = [85, 213, 340]  # Near start, middle, near end
    time_indices = [312, 468, 625]  # ~1000ms, ~1500ms, ~2000ms

    print(f"\nGenerating slices...")
    print(f"  Inlines: {[il_axis[i] for i in inline_indices]}")
    print(f"  Crosslines: {[xl_axis[i] for i in crossline_indices]}")
    print(f"  Times: {[f'{t_axis[i]:.0f}' for i in time_indices]} ms")

    # === Inline Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Migrated Volume - Inline Slices', fontsize=14, fontweight='bold')

    for i, il_idx in enumerate(inline_indices):
        ax = axes[i]
        slice_data = data[il_idx, :, :]
        plot_seismic_slice(slice_data, xl_axis, t_axis,
                          f'Inline {il_axis[il_idx]}',
                          'Crossline', 'Time (ms)', ax)

    plt.tight_layout()
    plt.savefig(out_dir / "migrated_inline_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Inline slices saved")

    # === Crossline Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Migrated Volume - Crossline Slices', fontsize=14, fontweight='bold')

    for i, xl_idx in enumerate(crossline_indices):
        ax = axes[i]
        slice_data = data[:, xl_idx, :]
        plot_seismic_slice(slice_data, il_axis, t_axis,
                          f'Crossline {xl_axis[xl_idx]}',
                          'Inline', 'Time (ms)', ax)

    plt.tight_layout()
    plt.savefig(out_dir / "migrated_crossline_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Crossline slices saved")

    # === Time Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Migrated Volume - Time Slices', fontsize=14, fontweight='bold')

    for i, t_idx in enumerate(time_indices):
        ax = axes[i]
        slice_data = data[:, :, t_idx]
        plot_seismic_slice(slice_data, il_axis, xl_axis,
                          f'Time = {t_axis[t_idx]:.0f} ms',
                          'Inline', 'Crossline', ax)

    plt.tight_layout()
    plt.savefig(out_dir / "migrated_time_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Time slices saved")

    # === Combined figure for presentation ===
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('PSTM Migration Results - Offset Bin 00', fontsize=16, fontweight='bold')

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
    plt.savefig(out_dir / "migrated_all_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Combined figure saved")

    print(f"\nOutput saved to: {out_dir}/")
    print(f"  - migrated_inline_slices.png")
    print(f"  - migrated_crossline_slices.png")
    print(f"  - migrated_time_slices.png")
    print(f"  - migrated_all_slices.png")


if __name__ == "__main__":
    main()
