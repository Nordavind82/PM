#!/usr/bin/env python3
"""
Create QC pictures from stacked PSTM volume.
Generates inline, crossline, and time slice images.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zarr

# Configuration
STACK_PATH = Path("/Volumes/AO_DISK/PSTM_common_offset_20m/pstm_stacked.zarr")
FOLD_PATH = Path("/Volumes/AO_DISK/PSTM_common_offset_20m/pstm_stacked_fold.zarr")
OUTPUT_DIR = Path("/Volumes/AO_DISK/PSTM_common_offset_20m/stack_qc_images")


def load_stack():
    """Load stacked volume and metadata."""
    z = zarr.open_array(str(STACK_PATH), mode='r')
    data = np.array(z[:])
    attrs = dict(z.attrs)

    print(f"Loaded stack: {data.shape}")
    print(f"  Offset range: {attrs.get('offset_min', 'N/A')} - {attrs.get('offset_max', 'N/A')} m")
    print(f"  Bins stacked: {attrs.get('n_bins_stacked', 'N/A')}")

    return data, attrs


def load_fold():
    """Load fold volume."""
    if FOLD_PATH.exists():
        z = zarr.open_array(str(FOLD_PATH), mode='r')
        return np.array(z[:])
    return None


def plot_inline(data, attrs, il_idx, output_path):
    """Plot inline section."""
    nx, ny, nt = data.shape
    dt_ms = attrs.get('dt_ms', 2.0)
    t_axis = np.arange(nt) * dt_ms

    section = data[il_idx, :, :]

    # Compute clip value
    vmax = np.percentile(np.abs(section), 99)

    fig, ax = plt.subplots(figsize=(16, 10))

    extent = [1, ny, t_axis[-1], t_axis[0]]
    im = ax.imshow(section.T, aspect='auto', extent=extent,
                   cmap='gray', vmin=-vmax, vmax=vmax,
                   interpolation='bilinear')

    ax.set_xlabel('Crossline', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'PSTM Stack - Inline {il_idx+1}\n'
                 f'Offset: {attrs.get("offset_min", 0):.0f}-{attrs.get("offset_max", 0):.0f}m, '
                 f'{attrs.get("n_bins_stacked", 0)} bins', fontsize=14)

    plt.colorbar(im, ax=ax, label='Amplitude', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_crossline(data, attrs, xl_idx, output_path):
    """Plot crossline section."""
    nx, ny, nt = data.shape
    dt_ms = attrs.get('dt_ms', 2.0)
    t_axis = np.arange(nt) * dt_ms

    section = data[:, xl_idx, :]

    # Compute clip value
    vmax = np.percentile(np.abs(section), 99)

    fig, ax = plt.subplots(figsize=(16, 10))

    extent = [1, nx, t_axis[-1], t_axis[0]]
    im = ax.imshow(section.T, aspect='auto', extent=extent,
                   cmap='gray', vmin=-vmax, vmax=vmax,
                   interpolation='bilinear')

    ax.set_xlabel('Inline', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'PSTM Stack - Crossline {xl_idx+1}\n'
                 f'Offset: {attrs.get("offset_min", 0):.0f}-{attrs.get("offset_max", 0):.0f}m, '
                 f'{attrs.get("n_bins_stacked", 0)} bins', fontsize=14)

    plt.colorbar(im, ax=ax, label='Amplitude', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_timeslice(data, attrs, t_ms, output_path):
    """Plot time slice."""
    nx, ny, nt = data.shape
    dt_ms = attrs.get('dt_ms', 2.0)
    t_idx = int(t_ms / dt_ms)
    t_idx = min(max(t_idx, 0), nt - 1)
    actual_t_ms = t_idx * dt_ms

    slice_data = data[:, :, t_idx]

    # Compute clip value
    vmax = np.percentile(np.abs(slice_data), 99)
    if vmax == 0:
        vmax = 1

    fig, ax = plt.subplots(figsize=(14, 10))

    extent = [1, ny, nx, 1]
    im = ax.imshow(slice_data, aspect='auto', extent=extent,
                   cmap='gray', vmin=-vmax, vmax=vmax,
                   interpolation='bilinear', origin='lower')

    ax.set_xlabel('Crossline', fontsize=12)
    ax.set_ylabel('Inline', fontsize=12)
    ax.set_title(f'PSTM Stack - Time Slice @ {actual_t_ms:.0f} ms\n'
                 f'Offset: {attrs.get("offset_min", 0):.0f}-{attrs.get("offset_max", 0):.0f}m, '
                 f'{attrs.get("n_bins_stacked", 0)} bins', fontsize=14)

    plt.colorbar(im, ax=ax, label='Amplitude', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_fold_map(fold, attrs, t_ms, output_path):
    """Plot fold map at given time."""
    nx, ny, nt = fold.shape
    dt_ms = attrs.get('dt_ms', 2.0)
    t_idx = int(t_ms / dt_ms)
    t_idx = min(max(t_idx, 0), nt - 1)
    actual_t_ms = t_idx * dt_ms

    fold_slice = fold[:, :, t_idx]

    fig, ax = plt.subplots(figsize=(14, 10))

    extent = [1, ny, nx, 1]
    im = ax.imshow(fold_slice, aspect='auto', extent=extent,
                   cmap='viridis', interpolation='bilinear', origin='lower')

    ax.set_xlabel('Crossline', fontsize=12)
    ax.set_ylabel('Inline', fontsize=12)
    ax.set_title(f'Fold Map @ {actual_t_ms:.0f} ms\n'
                 f'Min: {fold_slice.min():.0f}, Max: {fold_slice.max():.0f}, Mean: {fold_slice.mean():.0f}',
                 fontsize=14)

    plt.colorbar(im, ax=ax, label='Fold', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_summary_plot(data, attrs, output_path):
    """Create a summary plot with multiple views."""
    nx, ny, nt = data.shape
    dt_ms = attrs.get('dt_ms', 2.0)
    t_axis = np.arange(nt) * dt_ms

    # Select representative locations
    il_center = nx // 2
    xl_center = ny // 2
    t_slices = [500, 1000, 1500]  # ms

    fig = plt.figure(figsize=(20, 16))

    # Inline section
    ax1 = fig.add_subplot(2, 2, 1)
    section = data[il_center, :, :]
    vmax = np.percentile(np.abs(section), 99)
    extent = [1, ny, t_axis[-1], t_axis[0]]
    ax1.imshow(section.T, aspect='auto', extent=extent,
               cmap='gray', vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax1.set_xlabel('Crossline')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title(f'Inline {il_center+1}')

    # Crossline section
    ax2 = fig.add_subplot(2, 2, 2)
    section = data[:, xl_center, :]
    vmax = np.percentile(np.abs(section), 99)
    extent = [1, nx, t_axis[-1], t_axis[0]]
    ax2.imshow(section.T, aspect='auto', extent=extent,
               cmap='gray', vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax2.set_xlabel('Inline')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title(f'Crossline {xl_center+1}')

    # Time slices (3 in one row)
    for i, t_ms in enumerate(t_slices):
        ax = fig.add_subplot(2, 3, 4 + i)
        t_idx = int(t_ms / dt_ms)
        t_idx = min(max(t_idx, 0), nt - 1)
        slice_data = data[:, :, t_idx]
        vmax = np.percentile(np.abs(slice_data), 99)
        if vmax == 0:
            vmax = 1
        extent = [1, ny, nx, 1]
        ax.imshow(slice_data, aspect='auto', extent=extent,
                  cmap='gray', vmin=-vmax, vmax=vmax,
                  interpolation='bilinear', origin='lower')
        ax.set_xlabel('Crossline')
        ax.set_ylabel('Inline')
        ax.set_title(f'Time Slice @ {t_ms} ms')

    plt.suptitle(f'PSTM Stack Summary\n'
                 f'Offset: {attrs.get("offset_min", 0):.0f}-{attrs.get("offset_max", 0):.0f}m, '
                 f'{attrs.get("n_bins_stacked", 0)} bins, '
                 f'Grid: {nx}x{ny}x{nt}', fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 70)
    print("Creating Stack QC Pictures")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data, attrs = load_stack()
    fold = load_fold()

    nx, ny, nt = data.shape
    dt_ms = attrs.get('dt_ms', 2.0)

    print(f"\nGenerating images...")

    # Inline sections (3 locations)
    il_positions = [nx//4, nx//2, 3*nx//4]
    print("\nInline sections:")
    for il_idx in il_positions:
        output_path = OUTPUT_DIR / f"inline_{il_idx+1:03d}.png"
        plot_inline(data, attrs, il_idx, output_path)

    # Crossline sections (3 locations)
    xl_positions = [ny//4, ny//2, 3*ny//4]
    print("\nCrossline sections:")
    for xl_idx in xl_positions:
        output_path = OUTPUT_DIR / f"crossline_{xl_idx+1:03d}.png"
        plot_crossline(data, attrs, xl_idx, output_path)

    # Time slices (5 times)
    time_slices = [400, 600, 800, 1000, 1200]  # ms
    print("\nTime slices:")
    for t_ms in time_slices:
        output_path = OUTPUT_DIR / f"timeslice_{t_ms:04d}ms.png"
        plot_timeslice(data, attrs, t_ms, output_path)

    # Fold maps
    if fold is not None:
        print("\nFold maps:")
        for t_ms in [500, 1000]:
            output_path = OUTPUT_DIR / f"fold_map_{t_ms:04d}ms.png"
            plot_fold_map(fold, attrs, t_ms, output_path)

    # Summary plot
    print("\nSummary plot:")
    summary_path = OUTPUT_DIR / "stack_summary.png"
    create_summary_plot(data, attrs, summary_path)

    print(f"\n" + "=" * 70)
    print("QC Pictures Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total images: {3 + 3 + 5 + 2 + 1} = 14")
    print("=" * 70)


if __name__ == "__main__":
    main()
