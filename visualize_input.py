#!/usr/bin/env python3
"""
Visualize input traces with inline, crossline, and time slices.
Grids the input traces based on inline/crossline headers.
"""

import zarr
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
INPUT_DIR = "/Users/olegadamovich/SeismicData/common_offset_gathers/offset_bin_00"
OUTPUT_DIR = "/Users/olegadamovich/SeismicData/PSTM_common_offset/migration_qc"

# Same slice positions as migrated output
INLINE_INDICES = [101, 256, 411]  # Inline numbers
CROSSLINE_INDICES = [86, 214, 341]  # Crossline numbers
TIME_INDICES = [312, 468, 625]  # Sample indices (~1000, 1500, 2000 ms at 3.2ms)

DT_MS = 2.0  # Input sample rate (2ms = 500Hz)


def load_input_data():
    """Load input traces and headers."""
    print("Loading input data...")

    traces_path = Path(INPUT_DIR) / "traces.zarr"
    headers_path = Path(INPUT_DIR) / "headers.parquet"

    traces = zarr.open(str(traces_path))
    headers = pl.read_parquet(str(headers_path))

    print(f"Traces shape: {traces.shape} (samples x traces)")
    print(f"Headers: {len(headers)} traces")

    return np.asarray(traces), headers


def grid_traces(traces, headers, inline_range, crossline_range):
    """Grid traces into a regular 3D volume based on inline/crossline."""
    n_samples = traces.shape[0]

    # Get unique sorted values
    inlines = np.array(headers['inline'].to_numpy())
    crosslines = np.array(headers['crossline'].to_numpy())

    il_min, il_max = inline_range
    xl_min, xl_max = crossline_range

    n_il = il_max - il_min + 1
    n_xl = xl_max - xl_min + 1

    print(f"Gridding to {n_il} x {n_xl} x {n_samples}")

    # Create output volume (IL, XL, T)
    volume = np.zeros((n_il, n_xl, n_samples), dtype=np.float32)
    fold = np.zeros((n_il, n_xl), dtype=np.int32)

    # Grid each trace
    for i in range(len(headers)):
        il = inlines[i]
        xl = crosslines[i]

        if il_min <= il <= il_max and xl_min <= xl <= xl_max:
            il_idx = il - il_min
            xl_idx = xl - xl_min
            volume[il_idx, xl_idx, :] += traces[:, i]
            fold[il_idx, xl_idx] += 1

    # Normalize by fold
    with np.errstate(divide='ignore', invalid='ignore'):
        volume = np.where(fold[:, :, np.newaxis] > 0,
                         volume / fold[:, :, np.newaxis],
                         0)

    print(f"Fold: min={fold.min()}, max={fold.max()}, mean={fold.mean():.1f}")
    print(f"Non-zero bins: {np.sum(fold > 0)} / {n_il * n_xl}")

    return volume, fold


def plot_seismic_slice(data, x_axis, y_axis, title, xlabel, ylabel, ax,
                       clip_percentile=99, cmap='gray'):
    """Plot a seismic slice with proper amplitude scaling."""
    vmax = np.percentile(np.abs(data[data != 0]), clip_percentile) if np.any(data != 0) else 1
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
    traces, headers = load_input_data()

    # Determine grid range
    il_min = int(headers['inline'].min())
    il_max = int(headers['inline'].max())
    xl_min = int(headers['crossline'].min())
    xl_max = int(headers['crossline'].max())

    print(f"Inline range: {il_min} - {il_max}")
    print(f"Crossline range: {xl_min} - {xl_max}")

    # Grid the traces
    volume, fold = grid_traces(traces, headers, (il_min, il_max), (xl_min, xl_max))

    # Create axis arrays
    n_samples = traces.shape[0]
    il_axis = np.arange(il_min, il_max + 1)
    xl_axis = np.arange(xl_min, xl_max + 1)
    t_axis = np.arange(n_samples) * DT_MS

    print(f"\nVolume shape: {volume.shape}")
    print(f"Time range: 0 - {t_axis[-1]:.0f} ms")
    print(f"Data range: {volume.min():.4f} to {volume.max():.4f}")

    # Convert slice positions to indices
    # Use the same inline/crossline numbers as the migrated output
    inline_indices = [il - il_min for il in INLINE_INDICES if il_min <= il <= il_max]
    crossline_indices = [xl - xl_min for xl in CROSSLINE_INDICES if xl_min <= xl <= xl_max]

    # Time indices - need to scale from 3.2ms output to 2ms input
    # Output time_idx * 3.2ms = input time in ms -> input time / 2ms = input_idx
    time_indices_input = [int(t_idx * 3.2 / DT_MS) for t_idx in TIME_INDICES]
    time_indices_input = [t for t in time_indices_input if t < n_samples]

    print(f"\nSlice positions:")
    print(f"  Inlines: {[il_axis[i] for i in inline_indices]}")
    print(f"  Crosslines: {[xl_axis[i] for i in crossline_indices]}")
    print(f"  Times: {[f'{t_axis[i]:.0f}' for i in time_indices_input]} ms")

    # === Inline Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Input Data (Offset Bin 00) - Inline Slices', fontsize=14, fontweight='bold')

    for i, il_idx in enumerate(inline_indices):
        ax = axes[i]
        slice_data = volume[il_idx, :, :]
        plot_seismic_slice(slice_data, xl_axis, t_axis,
                          f'Inline {il_axis[il_idx]}',
                          'Crossline', 'Time (ms)', ax)

    plt.tight_layout()
    plt.savefig(out_dir / "input_inline_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Inline slices saved")

    # === Crossline Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Input Data (Offset Bin 00) - Crossline Slices', fontsize=14, fontweight='bold')

    for i, xl_idx in enumerate(crossline_indices):
        ax = axes[i]
        slice_data = volume[:, xl_idx, :]
        plot_seismic_slice(slice_data, il_axis, t_axis,
                          f'Crossline {xl_axis[xl_idx]}',
                          'Inline', 'Time (ms)', ax)

    plt.tight_layout()
    plt.savefig(out_dir / "input_crossline_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Crossline slices saved")

    # === Time Slices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Input Data (Offset Bin 00) - Time Slices', fontsize=14, fontweight='bold')

    for i, t_idx in enumerate(time_indices_input):
        ax = axes[i]
        slice_data = volume[:, :, t_idx]
        plot_seismic_slice(slice_data, il_axis, xl_axis,
                          f'Time = {t_axis[t_idx]:.0f} ms',
                          'Inline', 'Crossline', ax)

    plt.tight_layout()
    plt.savefig(out_dir / "input_time_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Time slices saved")

    # === Combined figure ===
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Input Data (Offset Bin 00) - All Slices', fontsize=16, fontweight='bold')

    # Row 1: Inline slices
    for i, il_idx in enumerate(inline_indices):
        ax = fig.add_subplot(3, 3, i + 1)
        slice_data = volume[il_idx, :, :]
        plot_seismic_slice(slice_data, xl_axis, t_axis,
                          f'Inline {il_axis[il_idx]}',
                          'Crossline', 'Time (ms)', ax)

    # Row 2: Crossline slices
    for i, xl_idx in enumerate(crossline_indices):
        ax = fig.add_subplot(3, 3, i + 4)
        slice_data = volume[:, xl_idx, :]
        plot_seismic_slice(slice_data, il_axis, t_axis,
                          f'Crossline {xl_axis[xl_idx]}',
                          'Inline', 'Time (ms)', ax)

    # Row 3: Time slices
    for i, t_idx in enumerate(time_indices_input):
        ax = fig.add_subplot(3, 3, i + 7)
        slice_data = volume[:, :, t_idx]
        plot_seismic_slice(slice_data, il_axis, xl_axis,
                          f'Time = {t_axis[t_idx]:.0f} ms',
                          'Inline', 'Crossline', ax)

    plt.tight_layout()
    plt.savefig(out_dir / "input_all_slices.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Combined figure saved")

    # === Fold map ===
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(fold.T, aspect='auto', cmap='viridis', origin='lower',
                   extent=[il_axis[0], il_axis[-1], xl_axis[0], xl_axis[-1]])
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title('Input Data Fold Map (Offset Bin 00)')
    plt.colorbar(im, ax=ax, label='Fold')
    plt.tight_layout()
    plt.savefig(out_dir / "input_fold_map.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  - Fold map saved")

    print(f"\nOutput saved to: {out_dir}/")
    print(f"  - input_inline_slices.png")
    print(f"  - input_crossline_slices.png")
    print(f"  - input_time_slices.png")
    print(f"  - input_all_slices.png")
    print(f"  - input_fold_map.png")


if __name__ == "__main__":
    main()
