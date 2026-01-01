#!/usr/bin/env python3
"""
Create Common Image Gathers (CIGs) from PSTM migrated offset bins.

Extracts amplitude vs offset at 18 spatially distributed locations
and saves them as images.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zarr

# Configuration
MIGRATION_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m")
INPUT_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")
OUTPUT_DIR = Path("/Volumes/AO_DISK/PSTM_common_offset_20m/cig_images")

N_CIGS = 18  # Number of CIG locations


def get_offset_for_bin(bin_num: int) -> float:
    """Get mean offset value for an offset bin from input headers."""
    import polars as pl
    headers_path = INPUT_DIR / f"offset_bin_{bin_num:02d}" / "headers.parquet"

    if headers_path.exists():
        df = pl.read_parquet(headers_path)
        if 'offset' in df.columns and len(df) > 0:
            return float(df['offset'].mean())

    # Fallback: estimate from bin number (20m bins)
    return bin_num * 20.0 + 10.0


def load_all_offsets():
    """Load all migrated offset volumes and their offset values."""
    print("Loading migrated offset volumes...")

    # Find all available bins
    bins = []
    for d in sorted(MIGRATION_DIR.glob("migration_bin_*")):
        zarr_path = d / "migrated_stack.zarr"
        if zarr_path.exists():
            bin_num = int(d.name.split("_")[-1])
            bins.append(bin_num)

    print(f"  Found {len(bins)} offset bins: {bins[0]}-{bins[-1]}")

    # Load first to get dimensions
    first_path = MIGRATION_DIR / f"migration_bin_{bins[0]:02d}" / "migrated_stack.zarr"
    first = zarr.open_array(str(first_path), mode='r')
    nx, ny, nt = first.shape
    attrs = dict(first.attrs)

    print(f"  Grid: {nx} x {ny} x {nt}")

    # Get coordinate bounds
    x_min = attrs.get('x_min', 0)
    x_max = attrs.get('x_max', nx - 1)
    y_min = attrs.get('y_min', 0)
    y_max = attrs.get('y_max', ny - 1)
    dt_ms = attrs.get('dt_ms', 2.0)

    print(f"  X range: {x_min:.1f} - {x_max:.1f}")
    print(f"  Y range: {y_min:.1f} - {y_max:.1f}")

    # Load offset values
    offsets = []
    for bin_num in bins:
        offset = get_offset_for_bin(bin_num)
        offsets.append(offset)

    offsets = np.array(offsets)
    print(f"  Offset range: {offsets.min():.0f} - {offsets.max():.0f} m")

    return bins, offsets, (nx, ny, nt), (x_min, x_max, y_min, y_max), dt_ms


def select_cig_locations(nx, ny, n_cigs=18):
    """Select CIG locations distributed across the grid."""
    # Create a grid pattern (approx sqrt(n) x sqrt(n))
    n_rows = int(np.ceil(np.sqrt(n_cigs * ny / nx)))
    n_cols = int(np.ceil(n_cigs / n_rows))

    # Adjust to get exactly n_cigs
    while n_rows * n_cols < n_cigs:
        n_cols += 1

    # Generate locations with margin from edges
    margin_x = nx // (n_cols + 1)
    margin_y = ny // (n_rows + 1)

    locations = []
    for row in range(n_rows):
        for col in range(n_cols):
            if len(locations) >= n_cigs:
                break
            ix = margin_x + col * (nx - 2 * margin_x) // max(1, n_cols - 1)
            iy = margin_y + row * (ny - 2 * margin_y) // max(1, n_rows - 1)
            # Clamp to valid range
            ix = min(max(ix, 10), nx - 10)
            iy = min(max(iy, 10), ny - 10)
            locations.append((ix, iy))

    return locations[:n_cigs]


def extract_cig(bins, ix, iy):
    """Extract a single CIG at location (ix, iy)."""
    traces = []

    for bin_num in bins:
        zarr_path = MIGRATION_DIR / f"migration_bin_{bin_num:02d}" / "migrated_stack.zarr"
        z = zarr.open_array(str(zarr_path), mode='r')
        trace = np.array(z[ix, iy, :])
        traces.append(trace)

    return np.array(traces)  # shape: (n_offsets, nt)


def plot_cig(cig, offsets, dt_ms, ix, iy, x_coord, y_coord, output_path, cig_num):
    """Plot and save a single CIG."""
    n_offsets, nt = cig.shape
    t_axis = np.arange(nt) * dt_ms

    # Normalize for display
    cig_norm = cig.copy()
    for i in range(n_offsets):
        trace_max = np.abs(cig_norm[i]).max()
        if trace_max > 0:
            cig_norm[i] /= trace_max

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))

    # Left: Wiggle plot
    ax1.set_title(f'CIG #{cig_num}: IL={ix+1}, XL={iy+1}\nX={x_coord:.0f}, Y={y_coord:.0f}', fontsize=12)

    # Plot wiggle traces
    scale = 0.8  # Scale factor for wiggle amplitude
    for i, offset in enumerate(offsets):
        trace = cig_norm[i] * scale
        ax1.plot(offset + trace * 20, t_axis, 'k-', linewidth=0.5)
        # Fill positive
        ax1.fill_betweenx(t_axis, offset, offset + trace * 20,
                          where=(trace > 0), color='black', alpha=0.5)

    ax1.set_xlabel('Offset (m)', fontsize=10)
    ax1.set_ylabel('Time (ms)', fontsize=10)
    ax1.set_xlim(offsets.min() - 50, offsets.max() + 50)
    ax1.set_ylim(t_axis.max(), 0)  # Invert for seismic convention
    ax1.grid(True, alpha=0.3)

    # Right: Image plot (variable density)
    extent = [offsets.min(), offsets.max(), t_axis.max(), t_axis.min()]

    # Compute percentile for clipping
    vmax = np.percentile(np.abs(cig), 98)

    im = ax2.imshow(cig.T, aspect='auto', extent=extent,
                    cmap='gray', vmin=-vmax, vmax=vmax,
                    interpolation='bilinear')
    ax2.set_xlabel('Offset (m)', fontsize=10)
    ax2.set_ylabel('Time (ms)', fontsize=10)
    ax2.set_title('Variable Density Display', fontsize=12)

    plt.colorbar(im, ax=ax2, label='Amplitude')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_cig_summary(all_cigs, offsets, dt_ms, locations, coords, output_path):
    """Create a summary plot showing all CIGs."""
    n_cigs = len(all_cigs)
    n_cols = 6
    n_rows = (n_cigs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    axes = axes.flatten()

    nt = all_cigs[0].shape[1]
    t_axis = np.arange(nt) * dt_ms
    extent = [offsets.min(), offsets.max(), t_axis.max(), t_axis.min()]

    for i, (cig, (ix, iy), (x, y)) in enumerate(zip(all_cigs, locations, coords)):
        ax = axes[i]

        # Compute percentile for clipping
        vmax = np.percentile(np.abs(cig), 98)
        if vmax == 0:
            vmax = 1

        ax.imshow(cig.T, aspect='auto', extent=extent,
                  cmap='gray', vmin=-vmax, vmax=vmax,
                  interpolation='bilinear')
        ax.set_title(f'CIG {i+1}: IL{ix+1}/XL{iy+1}', fontsize=9)
        ax.set_xlabel('Offset (m)', fontsize=8)
        ax.set_ylabel('Time (ms)', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for i in range(n_cigs, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Common Image Gathers (CIGs) - 18 Locations', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved summary: {output_path.name}")


def main():
    print("=" * 70)
    print("Creating Common Image Gathers (CIGs)")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data info
    bins, offsets, (nx, ny, nt), (x_min, x_max, y_min, y_max), dt_ms = load_all_offsets()

    # Select CIG locations
    locations = select_cig_locations(nx, ny, N_CIGS)
    print(f"\nSelected {len(locations)} CIG locations:")

    # Compute coordinates for each location
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)

    coords = []
    for ix, iy in locations:
        x = x_coords[ix]
        y = y_coords[iy]
        coords.append((x, y))
        print(f"  Location ({ix}, {iy}): X={x:.0f}, Y={y:.0f}")

    # Extract and plot CIGs
    print(f"\nExtracting and plotting CIGs...")
    all_cigs = []

    for i, ((ix, iy), (x, y)) in enumerate(zip(locations, coords)):
        print(f"  [{i+1}/{N_CIGS}] CIG at IL={ix+1}, XL={iy+1}")

        # Extract CIG
        cig = extract_cig(bins, ix, iy)
        all_cigs.append(cig)

        # Plot individual CIG
        output_path = OUTPUT_DIR / f"cig_{i+1:02d}_il{ix+1}_xl{iy+1}.png"
        plot_cig(cig, offsets, dt_ms, ix, iy, x, y, output_path, i+1)
        print(f"    Saved: {output_path.name}")

    # Create summary plot
    print(f"\nCreating summary plot...")
    summary_path = OUTPUT_DIR / "cig_summary_all.png"
    create_cig_summary(all_cigs, offsets, dt_ms, locations, coords, summary_path)

    print(f"\n" + "=" * 70)
    print("CIG Creation Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Individual CIGs: {N_CIGS} images")
    print(f"Summary: cig_summary_all.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
