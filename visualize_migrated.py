#!/usr/bin/env python3
"""
Visualize PSTM stacked volume with inline, crossline, and time slices.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
PSTM_STACK_ZARR = "/Users/olegadamovich/SeismicData/PSTM_common_offset/pstm_stacked.zarr"
OUTPUT_DIR = "/Users/olegadamovich/SeismicData/PSTM_common_offset/pstm_stack_qc"


def load_pstm_stack():
    """Load PSTM stacked volume from zarr."""
    print("Loading PSTM stacked volume...")
    store = zarr.open_array(PSTM_STACK_ZARR, mode='r')
    data = np.asarray(store)
    print(f"  Shape: {data.shape}")
    print(f"  Data range: {data.min():.6f} to {data.max():.6f}")

    # Get metadata
    attrs = dict(store.attrs)
    print(f"  Grid: {attrs.get('nx', 'N/A')} x {attrs.get('ny', 'N/A')} x {attrs.get('nt', 'N/A')}")
    print(f"  Spacing: dx={attrs.get('dx', 'N/A')}m, dy={attrs.get('dy', 'N/A')}m, dt={attrs.get('dt_ms', 'N/A')}ms")
    print(f"  Time range: {attrs.get('t_min_ms', 0)} - {attrs.get('t_max_ms', 'N/A')} ms")
    print(f"  Offset bins stacked: {attrs.get('n_bins_stacked', 'N/A')}")
    print(f"  Offset range: {attrs.get('offset_min', 'N/A')} - {attrs.get('offset_max', 'N/A')} m")

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
    data, attrs = load_pstm_stack()

    # Get parameters from attributes
    dt_ms = attrs.get('dt_ms', 2.0)
    n_bins = attrs.get('n_bins_stacked', 'N/A')
    offset_range = f"{attrs.get('offset_min', 0):.0f}-{attrs.get('offset_max', 0):.0f}m"

    # Create axis arrays
    il_axis = np.arange(1, data.shape[0] + 1)  # 1-511 (inline numbers)
    xl_axis = np.arange(1, data.shape[1] + 1)  # 1-427 (crossline numbers)
    t_axis = np.arange(data.shape[2]) * dt_ms  # Time in ms

    # Select slices - evenly distributed
    inline_indices = [100, 255, 410]  # Near start, middle, near end
    crossline_indices = [85, 213, 340]  # Near start, middle, near end
    time_indices_ms = [500, 1000, 1500]  # Times in ms
    time_indices = [int(t / dt_ms) for t in time_indices_ms]

    print(f"\nGenerating slices...")
    print(f"  Inlines: {[il_axis[i] for i in inline_indices]}")
    print(f"  Crosslines: {[xl_axis[i] for i in crossline_indices]}")
    print(f"  Times: {time_indices_ms} ms")

    # === Inline Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f'PSTM Stack - Inline Slices ({n_bins} bins, {offset_range})',
                 fontsize=14, fontweight='bold')

    for i, il_idx in enumerate(inline_indices):
        ax = axes[i]
        slice_data = data[il_idx, :, :]
        plot_seismic_slice(slice_data, xl_axis, t_axis,
                          f'Inline {il_axis[il_idx]}',
                          'Crossline', 'Time (ms)', ax)

    plt.tight_layout()
    plt.savefig(out_dir / "pstm_stack_inline_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Inline slices saved")

    # === Crossline Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f'PSTM Stack - Crossline Slices ({n_bins} bins, {offset_range})',
                 fontsize=14, fontweight='bold')

    for i, xl_idx in enumerate(crossline_indices):
        ax = axes[i]
        slice_data = data[:, xl_idx, :]
        plot_seismic_slice(slice_data, il_axis, t_axis,
                          f'Crossline {xl_axis[xl_idx]}',
                          'Inline', 'Time (ms)', ax)

    plt.tight_layout()
    plt.savefig(out_dir / "pstm_stack_crossline_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Crossline slices saved")

    # === Time Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'PSTM Stack - Time Slices ({n_bins} bins, {offset_range})',
                 fontsize=14, fontweight='bold')

    for i, t_idx in enumerate(time_indices):
        ax = axes[i]
        slice_data = data[:, :, t_idx]
        plot_seismic_slice(slice_data, il_axis, xl_axis,
                          f'Time = {t_axis[t_idx]:.0f} ms',
                          'Inline', 'Crossline', ax)

    plt.tight_layout()
    plt.savefig(out_dir / "pstm_stack_time_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Time slices saved")

    # === Combined figure for presentation ===
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'PSTM Stack ({n_bins} Offset Bins, {offset_range})',
                 fontsize=16, fontweight='bold')

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
    plt.savefig(out_dir / "pstm_stack_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Summary figure saved")

    # === Statistics ===
    print(f"\n{'='*50}")
    print("PSTM STACK STATISTICS")
    print(f"{'='*50}")
    print(f"  Shape: {data.shape}")
    print(f"  Non-zero samples: {np.count_nonzero(data):,} / {data.size:,} ({100*np.count_nonzero(data)/data.size:.1f}%)")
    print(f"  Min: {data.min():.6f}")
    print(f"  Max: {data.max():.6f}")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Std: {data.std():.6f}")
    print(f"  RMS: {np.sqrt(np.mean(data**2)):.6f}")

    print(f"\nOutput saved to: {out_dir}/")
    print(f"  - pstm_stack_inline_slices.png")
    print(f"  - pstm_stack_crossline_slices.png")
    print(f"  - pstm_stack_time_slices.png")
    print(f"  - pstm_stack_summary.png")


if __name__ == "__main__":
    main()
