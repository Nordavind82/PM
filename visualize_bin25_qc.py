#!/usr/bin/env python3
"""
Create QC images for migrated offset bin 25.

Generates:
- Inline section
- Crossline section
- Time slice
"""

import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
MIGRATION_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/migration_bin_25")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/stack_qc_images")

# Grid parameters
DX = 25.0
DY = 12.5
DT_MS = 2.0
T_MIN_MS = 0.0
T_MAX_MS = 2000.0


def main():
    print("=" * 70)
    print("QC Visualization: Offset Bin 25")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load migrated stack
    stack_path = MIGRATION_DIR / "migrated_stack.zarr"
    fold_path = MIGRATION_DIR / "fold.zarr"

    print(f"Loading: {stack_path}")
    z_stack = zarr.open_array(str(stack_path), mode='r')
    z_fold = zarr.open_array(str(fold_path), mode='r')

    print(f"Stack shape: {z_stack.shape}")  # (nx, ny, nt) = (511, 427, 1001)

    nx, ny, nt = z_stack.shape

    # Load into memory
    print("Loading data into memory...")
    data = np.array(z_stack[:])
    fold = np.array(z_fold[:])

    # Calculate clip value (percentile-based)
    nonzero_data = data[fold > 0]
    clip_val = np.percentile(np.abs(nonzero_data), 99)
    print(f"Clip value (99th percentile): {clip_val:.4f}")

    # Time axis
    t_axis = np.arange(nt) * DT_MS + T_MIN_MS

    # Select slices
    il_idx = nx // 2  # Middle inline (index ~255)
    xl_idx = ny // 2  # Middle crossline (index ~213)
    t_slice_ms = 800.0  # Time slice at 800ms
    t_idx = int((t_slice_ms - T_MIN_MS) / DT_MS)

    print(f"\nSlice locations:")
    print(f"  Inline: {il_idx} (of {nx})")
    print(f"  Crossline: {xl_idx} (of {ny})")
    print(f"  Time slice: {t_slice_ms} ms (index {t_idx})")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Inline section (constant X, varying Y and T)
    ax1 = axes[0, 0]
    inline_data = data[il_idx, :, :].T  # (nt, ny)
    im1 = ax1.imshow(
        inline_data,
        aspect='auto',
        cmap='gray',
        vmin=-clip_val,
        vmax=clip_val,
        extent=[0, ny, t_axis[-1], t_axis[0]],
    )
    ax1.set_xlabel('Crossline Index')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title(f'Inline Section (IL={il_idx})')
    plt.colorbar(im1, ax=ax1, label='Amplitude')

    # Add vertical line at boundary (Y=384)
    ax1.axvline(x=384, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.text(386, 100, 'Y=384', color='red', fontsize=8)

    # 2. Crossline section (constant Y, varying X and T)
    ax2 = axes[0, 1]
    xline_data = data[:, xl_idx, :].T  # (nt, nx)
    im2 = ax2.imshow(
        xline_data,
        aspect='auto',
        cmap='gray',
        vmin=-clip_val,
        vmax=clip_val,
        extent=[0, nx, t_axis[-1], t_axis[0]],
    )
    ax2.set_xlabel('Inline Index')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title(f'Crossline Section (XL={xl_idx})')
    plt.colorbar(im2, ax=ax2, label='Amplitude')

    # 3. Time slice (constant T, varying X and Y)
    ax3 = axes[1, 0]
    time_slice = data[:, :, t_idx].T  # (ny, nx)
    im3 = ax3.imshow(
        time_slice,
        aspect='auto',
        cmap='gray',
        vmin=-clip_val,
        vmax=clip_val,
        extent=[0, nx, ny, 0],
    )
    ax3.set_xlabel('Inline Index')
    ax3.set_ylabel('Crossline Index')
    ax3.set_title(f'Time Slice (T={t_slice_ms} ms)')
    plt.colorbar(im3, ax=ax3, label='Amplitude')

    # Add horizontal line at Y=384 boundary
    ax3.axhline(y=384, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax3.text(10, 380, 'Y=384', color='red', fontsize=8)

    # 4. Boundary analysis plot
    ax4 = axes[1, 1]

    # Calculate RMS by Y row for time range 300-500 (600-1000 ms)
    t_range = slice(300, 500)
    rms_by_y = np.sqrt(np.mean(data[:, :, t_range]**2, axis=(0, 2)))
    fold_by_y = fold[:, :, t_range].mean(axis=(0, 2))

    y_axis = np.arange(ny)

    ax4_twin = ax4.twinx()
    line1, = ax4.plot(y_axis, rms_by_y, 'b-', linewidth=1, label='RMS')
    line2, = ax4_twin.plot(y_axis, fold_by_y, 'g-', linewidth=1, alpha=0.5, label='Fold')

    ax4.set_xlabel('Crossline Index')
    ax4.set_ylabel('RMS Amplitude', color='b')
    ax4_twin.set_ylabel('Mean Fold', color='g')
    ax4.set_title('Amplitude and Fold vs Crossline (Y)')

    # Mark tile boundaries
    for y_bound in [128, 256, 384]:
        ax4.axvline(x=y_bound, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax4.text(384, rms_by_y.max() * 0.95, 'Y=384\n(tile boundary)', color='red', fontsize=8, ha='center')
    ax4.legend([line1, line2], ['RMS', 'Fold'], loc='upper right')

    plt.tight_layout()

    # Save figure
    output_file = OUTPUT_DIR / "bin25_qc_sections.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")

    # Also save individual images
    # Inline
    fig_il, ax_il = plt.subplots(figsize=(12, 8))
    im_il = ax_il.imshow(
        inline_data,
        aspect='auto',
        cmap='gray',
        vmin=-clip_val,
        vmax=clip_val,
        extent=[0, ny, t_axis[-1], t_axis[0]],
    )
    ax_il.set_xlabel('Crossline Index')
    ax_il.set_ylabel('Time (ms)')
    ax_il.set_title(f'Offset Bin 25 - Inline {il_idx}')
    ax_il.axvline(x=384, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im_il, ax=ax_il, label='Amplitude')
    il_file = OUTPUT_DIR / "bin25_inline.png"
    plt.savefig(il_file, dpi=150, bbox_inches='tight')
    plt.close(fig_il)
    print(f"Saved: {il_file}")

    # Crossline
    fig_xl, ax_xl = plt.subplots(figsize=(12, 8))
    im_xl = ax_xl.imshow(
        xline_data,
        aspect='auto',
        cmap='gray',
        vmin=-clip_val,
        vmax=clip_val,
        extent=[0, nx, t_axis[-1], t_axis[0]],
    )
    ax_xl.set_xlabel('Inline Index')
    ax_xl.set_ylabel('Time (ms)')
    ax_xl.set_title(f'Offset Bin 25 - Crossline {xl_idx}')
    plt.colorbar(im_xl, ax=ax_xl, label='Amplitude')
    xl_file = OUTPUT_DIR / "bin25_crossline.png"
    plt.savefig(xl_file, dpi=150, bbox_inches='tight')
    plt.close(fig_xl)
    print(f"Saved: {xl_file}")

    # Time slice
    fig_ts, ax_ts = plt.subplots(figsize=(12, 10))
    im_ts = ax_ts.imshow(
        time_slice,
        aspect='auto',
        cmap='gray',
        vmin=-clip_val,
        vmax=clip_val,
        extent=[0, nx, ny, 0],
    )
    ax_ts.set_xlabel('Inline Index')
    ax_ts.set_ylabel('Crossline Index')
    ax_ts.set_title(f'Offset Bin 25 - Time Slice {t_slice_ms} ms')
    ax_ts.axhline(y=384, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im_ts, ax=ax_ts, label='Amplitude')
    ts_file = OUTPUT_DIR / "bin25_timeslice.png"
    plt.savefig(ts_file, dpi=150, bbox_inches='tight')
    plt.close(fig_ts)
    print(f"Saved: {ts_file}")

    plt.close('all')

    # Print boundary statistics
    print("\n" + "=" * 70)
    print("BOUNDARY STATISTICS")
    print("=" * 70)

    y383_rms = np.sqrt(np.mean(data[:, 383, t_range]**2))
    y384_rms = np.sqrt(np.mean(data[:, 384, t_range]**2))
    boundary_ratio = y384_rms / y383_rms

    print(f"Y=383 RMS: {y383_rms:.6f}")
    print(f"Y=384 RMS: {y384_rms:.6f}")
    print(f"Boundary ratio (Y=384/Y=383): {boundary_ratio:.4f}")

    if boundary_ratio > 0.95:
        print("\nBoundary looks GOOD (ratio > 0.95)")
    else:
        print(f"\nWARNING: Boundary ratio {boundary_ratio:.4f} indicates potential artifact")

    print("\n" + "=" * 70)
    print("QC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
