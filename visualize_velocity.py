#!/usr/bin/env python3
"""
Velocity visualization script.
Creates inline, crossline, and time slice plots for PPT.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

# Configuration
VELOCITY_ZARR = "/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm.zarr"
OUTPUT_DIR = "/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_qc"

# Grid parameters from run script
NX, NY, NT = 511, 427, 1001
DT_MS = 3.2  # 3.2ms sampling

# Grid corners for coordinate reference
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),  # Origin (IL=1, XL=1)
    'c2': (627094.02, 5106803.16),  # Inline end (IL=511, XL=1)
    'c3': (631143.35, 5110261.43),  # Far corner (IL=511, XL=427)
    'c4': (622862.92, 5119956.77),  # Crossline end (IL=1, XL=427)
}


def load_and_interpolate_velocity():
    """Load velocity from zarr and interpolate to output grid."""
    print("Loading velocity from zarr...")
    store = zarr.open(VELOCITY_ZARR, mode='r')

    # Get source data and axes
    vel_data = np.asarray(store)  # shape: [52, 22, 108]
    x_axis = np.array(store.attrs['x_axis'])
    y_axis = np.array(store.attrs['y_axis'])
    t_axis_ms = np.array(store.attrs['t_axis_ms'])

    print(f"Source velocity shape: {vel_data.shape}")
    print(f"X range: {x_axis[0]:.1f} - {x_axis[-1]:.1f}")
    print(f"Y range: {y_axis[0]:.1f} - {y_axis[-1]:.1f}")
    print(f"T range: {t_axis_ms[0]:.1f} - {t_axis_ms[-1]:.1f} ms")
    print(f"Velocity range: {vel_data.min():.0f} - {vel_data.max():.0f} m/s")

    # Create interpolator with extrapolation
    interp = RegularGridInterpolator(
        (x_axis, y_axis, t_axis_ms),
        vel_data,
        method='linear',
        bounds_error=False,
        fill_value=None  # Enables extrapolation
    )

    # Create output grid coordinates
    # Using inline/crossline indices for axes labels
    il_axis = np.arange(1, NX + 1)  # 1-511
    xl_axis = np.arange(1, NY + 1)  # 1-427
    t_axis_out = np.arange(NT) * DT_MS  # 0 to 3200 ms

    # Create coordinate grids for interpolation
    # Map inline/crossline to X/Y coordinates
    x_out = np.linspace(x_axis[0], x_axis[-1], NX)
    y_out = np.linspace(y_axis[0], y_axis[-1], NY)

    return interp, x_out, y_out, t_axis_out, il_axis, xl_axis, vel_data


def create_inline_slice(interp, x_out, y_out, t_axis, il_idx, il_axis, xl_axis):
    """Create inline slice (constant inline, showing XL vs Time)."""
    x_val = x_out[il_idx]

    # Create meshgrid for this inline
    y_grid, t_grid = np.meshgrid(y_out, t_axis, indexing='ij')
    x_grid = np.full_like(y_grid, x_val)

    # Interpolate
    points = np.stack([x_grid.ravel(), y_grid.ravel(), t_grid.ravel()], axis=-1)
    vel_slice = interp(points).reshape(y_grid.shape)

    return vel_slice, xl_axis, t_axis


def create_crossline_slice(interp, x_out, y_out, t_axis, xl_idx, il_axis, xl_axis):
    """Create crossline slice (constant crossline, showing IL vs Time)."""
    y_val = y_out[xl_idx]

    # Create meshgrid for this crossline
    x_grid, t_grid = np.meshgrid(x_out, t_axis, indexing='ij')
    y_grid = np.full_like(x_grid, y_val)

    # Interpolate
    points = np.stack([x_grid.ravel(), y_grid.ravel(), t_grid.ravel()], axis=-1)
    vel_slice = interp(points).reshape(x_grid.shape)

    return vel_slice, il_axis, t_axis


def create_time_slice(interp, x_out, y_out, t_val, il_axis, xl_axis):
    """Create time slice (constant time, showing IL vs XL)."""
    # Create meshgrid for this time
    x_grid, y_grid = np.meshgrid(x_out, y_out, indexing='ij')
    t_grid = np.full_like(x_grid, t_val)

    # Interpolate
    points = np.stack([x_grid.ravel(), y_grid.ravel(), t_grid.ravel()], axis=-1)
    vel_slice = interp(points).reshape(x_grid.shape)

    return vel_slice, il_axis, xl_axis


def plot_slice(data, x_axis, y_axis, title, xlabel, ylabel, ax, cmap='jet_r',
               vmin=None, vmax=None, aspect='auto'):
    """Plot a velocity slice."""
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
    im = ax.imshow(data.T, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Velocity (m/s)')

    return im


def main():
    # Create output directory
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    interp, x_out, y_out, t_axis, il_axis, xl_axis, source_vel = load_and_interpolate_velocity()

    # Velocity range for consistent colorbar
    vmin, vmax = source_vel.min(), source_vel.max()
    print(f"\nUsing velocity range: {vmin:.0f} - {vmax:.0f} m/s")

    # Select slices
    inline_indices = [50, 150, 255, 350, 450]  # Sample inlines
    crossline_indices = [50, 150, 213, 300, 400]  # Sample crosslines
    time_values = [500.0, 1000.0, 1500.0, 2000.0, 2500.0]  # Time in ms

    # Create PDF with all slices
    pdf_path = out_dir / "velocity_slices.pdf"
    print(f"\nGenerating velocity slices to {pdf_path}...")

    with PdfPages(pdf_path) as pdf:
        # === Page 1: Inline slices ===
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Inline Slices (IL constant, XL vs Time)', fontsize=14)

        for i, il_idx in enumerate(inline_indices):
            ax = axes.flat[i]
            vel_slice, x_ax, y_ax = create_inline_slice(
                interp, x_out, y_out, t_axis, il_idx, il_axis, xl_axis
            )
            plot_slice(vel_slice, x_ax, y_ax,
                      f'Inline {il_axis[il_idx]}',
                      'Crossline', 'Time (ms)', ax,
                      vmin=vmin, vmax=vmax)

        # Hide empty subplot
        axes.flat[-1].set_visible(False)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  - Inline slices done")

        # === Page 2: Crossline slices ===
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Crossline Slices (XL constant, IL vs Time)', fontsize=14)

        for i, xl_idx in enumerate(crossline_indices):
            ax = axes.flat[i]
            vel_slice, x_ax, y_ax = create_crossline_slice(
                interp, x_out, y_out, t_axis, xl_idx, il_axis, xl_axis
            )
            plot_slice(vel_slice, x_ax, y_ax,
                      f'Crossline {xl_axis[xl_idx]}',
                      'Inline', 'Time (ms)', ax,
                      vmin=vmin, vmax=vmax)

        axes.flat[-1].set_visible(False)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  - Crossline slices done")

        # === Page 3: Time slices ===
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Time Slices (T constant, IL vs XL)', fontsize=14)

        for i, t_val in enumerate(time_values):
            ax = axes.flat[i]
            vel_slice, x_ax, y_ax = create_time_slice(
                interp, x_out, y_out, t_val, il_axis, xl_axis
            )
            plot_slice(vel_slice, x_ax, y_ax,
                      f'Time = {t_val:.0f} ms',
                      'Inline', 'Crossline', ax,
                      vmin=vmin, vmax=vmax)

        axes.flat[-1].set_visible(False)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  - Time slices done")

        # === Page 4: Source velocity overview ===
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Source Velocity Overview (52×22×108 from SEG-Y)', fontsize=14)

        # Mid inline from source
        ax = axes[0]
        mid_x = source_vel.shape[0] // 2
        im = ax.imshow(source_vel[mid_x, :, :].T, aspect='auto', cmap='jet_r',
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel('Y index')
        ax.set_ylabel('Time index')
        ax.set_title(f'Source X slice (idx={mid_x})')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Mid crossline from source
        ax = axes[1]
        mid_y = source_vel.shape[1] // 2
        im = ax.imshow(source_vel[:, mid_y, :].T, aspect='auto', cmap='jet_r',
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel('X index')
        ax.set_ylabel('Time index')
        ax.set_title(f'Source Y slice (idx={mid_y})')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Time slice from source
        ax = axes[2]
        mid_t = source_vel.shape[2] // 2
        im = ax.imshow(source_vel[:, :, mid_t].T, aspect='auto', cmap='jet_r',
                       vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')
        ax.set_title(f'Source T slice (idx={mid_t})')
        plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  - Source overview done")

    # Also save individual PNG files for easy PPT import
    print("\nSaving individual PNG files...")

    # One representative of each type
    for il_idx in [255]:  # Middle inline
        fig, ax = plt.subplots(figsize=(10, 8))
        vel_slice, x_ax, y_ax = create_inline_slice(
            interp, x_out, y_out, t_axis, il_idx, il_axis, xl_axis
        )
        plot_slice(vel_slice, x_ax, y_ax,
                  f'Velocity - Inline {il_axis[il_idx]}',
                  'Crossline', 'Time (ms)', ax,
                  vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.savefig(out_dir / f"velocity_inline_{il_axis[il_idx]}.png", dpi=150)
        plt.close(fig)

    for xl_idx in [213]:  # Middle crossline
        fig, ax = plt.subplots(figsize=(10, 8))
        vel_slice, x_ax, y_ax = create_crossline_slice(
            interp, x_out, y_out, t_axis, xl_idx, il_axis, xl_axis
        )
        plot_slice(vel_slice, x_ax, y_ax,
                  f'Velocity - Crossline {xl_axis[xl_idx]}',
                  'Inline', 'Time (ms)', ax,
                  vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.savefig(out_dir / f"velocity_crossline_{xl_axis[xl_idx]}.png", dpi=150)
        plt.close(fig)

    for t_val in [1000.0, 2000.0]:
        fig, ax = plt.subplots(figsize=(10, 8))
        vel_slice, x_ax, y_ax = create_time_slice(
            interp, x_out, y_out, t_val, il_axis, xl_axis
        )
        plot_slice(vel_slice, x_ax, y_ax,
                  f'Velocity - Time = {t_val:.0f} ms',
                  'Inline', 'Crossline', ax,
                  vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.savefig(out_dir / f"velocity_time_{int(t_val)}ms.png", dpi=150)
        plt.close(fig)

    print(f"\nOutput saved to: {out_dir}/")
    print(f"  - velocity_slices.pdf (all slices)")
    print(f"  - velocity_inline_*.png")
    print(f"  - velocity_crossline_*.png")
    print(f"  - velocity_time_*.png")


if __name__ == "__main__":
    main()
