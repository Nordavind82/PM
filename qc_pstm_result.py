#!/usr/bin/env python3
"""
QC script for PSTM migration results.

Generates QC plots for the migrated stack:
- Time slices
- Inline sections
- Crossline sections
- Fold map
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Output grid parameters (from config)
NX = 1206  # Number of inlines
NY = 576   # Number of crosslines
NT = 1001  # Number of time samples

X_MIN = 617443.56
X_MAX = 632512.02
DX = 12.5

Y_MIN = 5106192.26
Y_MAX = 5120569.31
DY = 25.0

T_MIN = 0.0
T_MAX = 2000.0
DT = 2.0

# Data paths
WORK_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_offset_bins/.work")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_offset_bins/qc_plots")


def load_image():
    """Load the migrated image."""
    image_path = WORK_DIR / "image.dat"
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print(f"Loading image: {image_path}")
    image = np.memmap(
        str(image_path),
        dtype=np.float64,  # Executor stores as float64
        mode='r',
        shape=(NX, NY, NT)
    )
    print(f"  Shape: {image.shape}")
    print(f"  Min: {np.min(image):.4f}, Max: {np.max(image):.4f}")
    return image


def load_fold():
    """Load the fold map."""
    fold_path = WORK_DIR / "fold.dat"
    if not fold_path.exists():
        return None

    print(f"Loading fold: {fold_path}")
    fold = np.memmap(
        str(fold_path),
        dtype=np.int32,
        mode='r',
        shape=(NX, NY)
    )
    print(f"  Shape: {fold.shape}")
    print(f"  Min: {np.min(fold)}, Max: {np.max(fold)}")
    return fold


def plot_time_slices(image, times_ms=[200, 500, 800, 1200, 1600]):
    """Plot time slices."""
    print("\nGenerating time slice plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Compute clip value from data
    clip = np.percentile(np.abs(image), 99)

    for i, t_ms in enumerate(times_ms):
        if i >= len(axes):
            break

        t_idx = int((t_ms - T_MIN) / DT)
        if t_idx < 0 or t_idx >= NT:
            continue

        ax = axes[i]
        slice_data = image[:, :, t_idx].T

        im = ax.imshow(
            slice_data,
            aspect='auto',
            cmap='seismic',
            vmin=-clip,
            vmax=clip,
            extent=[0, NX, NY, 0]
        )
        ax.set_title(f'Time Slice: {t_ms} ms')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused axes
    for i in range(len(times_ms), len(axes)):
        axes[i].axis('off')

    plt.suptitle('PSTM Time Slices', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "time_slices.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_inline_sections(image, inlines=[200, 400, 600, 800, 1000]):
    """Plot inline sections."""
    print("\nGenerating inline section plots...")

    fig, axes = plt.subplots(len(inlines), 1, figsize=(16, 4*len(inlines)))
    if len(inlines) == 1:
        axes = [axes]

    clip = np.percentile(np.abs(image), 99)

    for i, il in enumerate(inlines):
        if il >= NX:
            continue

        ax = axes[i]
        section = image[il, :, :].T

        im = ax.imshow(
            section,
            aspect='auto',
            cmap='seismic',
            vmin=-clip,
            vmax=clip,
            extent=[0, NY, T_MAX, T_MIN]
        )
        ax.set_title(f'Inline {il} (X = {X_MIN + il * DX:.0f} m)')
        ax.set_xlabel('Crossline')
        ax.set_ylabel('Time (ms)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('PSTM Inline Sections', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "inline_sections.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_crossline_sections(image, crosslines=[100, 200, 300, 400, 500]):
    """Plot crossline sections."""
    print("\nGenerating crossline section plots...")

    fig, axes = plt.subplots(len(crosslines), 1, figsize=(16, 4*len(crosslines)))
    if len(crosslines) == 1:
        axes = [axes]

    clip = np.percentile(np.abs(image), 99)

    for i, xl in enumerate(crosslines):
        if xl >= NY:
            continue

        ax = axes[i]
        section = image[:, xl, :].T

        im = ax.imshow(
            section,
            aspect='auto',
            cmap='seismic',
            vmin=-clip,
            vmax=clip,
            extent=[0, NX, T_MAX, T_MIN]
        )
        ax.set_title(f'Crossline {xl} (Y = {Y_MIN + xl * DY:.0f} m)')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Time (ms)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('PSTM Crossline Sections', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "crossline_sections.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_fold_map(fold):
    """Plot fold map."""
    if fold is None:
        print("\nSkipping fold map (no fold data)")
        return

    print("\nGenerating fold map...")

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(
        fold.T,
        aspect='auto',
        cmap='viridis',
        extent=[0, NX, NY, 0]
    )
    ax.set_title('Fold Map')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    plt.colorbar(im, ax=ax, label='Fold')

    plt.tight_layout()

    output_path = OUTPUT_DIR / "fold_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_amplitude_spectrum(image):
    """Plot amplitude statistics."""
    print("\nGenerating amplitude analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMS amplitude per time sample
    ax = axes[0, 0]
    rms_t = np.sqrt(np.mean(image**2, axis=(0, 1)))
    times = np.arange(NT) * DT
    ax.plot(times, rms_t)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('RMS Amplitude')
    ax.set_title('RMS Amplitude vs Time')
    ax.grid(True, alpha=0.3)

    # RMS amplitude map
    ax = axes[0, 1]
    rms_map = np.sqrt(np.mean(image**2, axis=2))
    im = ax.imshow(rms_map.T, aspect='auto', cmap='hot', extent=[0, NX, NY, 0])
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title('RMS Amplitude Map')
    plt.colorbar(im, ax=ax)

    # Amplitude histogram
    ax = axes[1, 0]
    sample = image[::10, ::10, ::10].flatten()  # Subsample for speed
    ax.hist(sample, bins=100, edgecolor='none', alpha=0.7)
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Count')
    ax.set_title('Amplitude Histogram')
    ax.set_yscale('log')

    # Max amplitude per inline
    ax = axes[1, 1]
    max_il = np.max(np.abs(image), axis=(1, 2))
    ax.plot(max_il)
    ax.set_xlabel('Inline')
    ax.set_ylabel('Max Amplitude')
    ax.set_title('Max Amplitude per Inline')
    ax.grid(True, alpha=0.3)

    plt.suptitle('PSTM Amplitude Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "amplitude_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("PSTM QC Analysis")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Load data
    image = load_image()
    fold = load_fold()

    # Generate plots
    plot_time_slices(image)
    plot_inline_sections(image)
    plot_crossline_sections(image)
    plot_fold_map(fold)
    plot_amplitude_spectrum(image)

    print("\n" + "=" * 60)
    print("QC Complete!")
    print("=" * 60)
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
