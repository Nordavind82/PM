#!/usr/bin/env python3
"""
Create velocity cube from vels.sgy for PSTM (20m offset bins).

Reads velocity from SEGY with:
- Inline: byte 237
- Crossline: byte 21

Interpolates to full PSTM grid and saves as zarr.
Creates QC plots of inline/crossline sections.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio
import zarr
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# Configuration
# =============================================================================

VELOCITY_SEGY = Path("/Users/olegadamovich/SeismicData/vels.sgy")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")
OUTPUT_ZARR = OUTPUT_DIR / "velocity_pstm.zarr"
QC_DIR = OUTPUT_DIR / "velocity_qc"

# Header byte positions (user specified)
IL_BYTE = 237  # Inline
XL_BYTE = 21   # Crossline

# PSTM grid parameters (must match run_pstm_all_offsets.py)
GRID_IL_MIN = 1
GRID_IL_MAX = 511
GRID_XL_MIN = 1
GRID_XL_MAX = 427
DX = 25.0   # meters per inline
DY = 12.5   # meters per crossline
DT_MS = 2.0
T_MIN_MS = 0.0
T_MAX_MS = 2000.0

# Grid corners (for reference)
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),  # Origin (IL=1, XL=1)
    'c2': (627094.02, 5106803.16),  # Inline end (IL=511, XL=1)
    'c3': (631143.35, 5110261.43),  # Far corner (IL=511, XL=427)
    'c4': (622862.92, 5119956.77),  # Crossline end (IL=1, XL=427)
}


def main():
    print("=" * 70)
    print("Velocity Cube Creation for PSTM (20m bins)")
    print("=" * 70)
    print(f"Input SEGY: {VELOCITY_SEGY}")
    print(f"Output zarr: {OUTPUT_ZARR}")
    print(f"IL byte: {IL_BYTE}, XL byte: {XL_BYTE}")
    print()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Read velocity from SEGY
    # =========================================================================
    print("1. Reading velocity from SEGY...")

    with segyio.open(str(VELOCITY_SEGY), ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples_segy = len(f.samples)
        dt_segy = f.samples[1] - f.samples[0] if n_samples_segy > 1 else 30.0
        t_segy = f.samples

        print(f"   Traces: {n_traces}")
        print(f"   Samples: {n_samples_segy}, dt={dt_segy}ms")
        print(f"   Time range: {t_segy[0]}-{t_segy[-1]} ms")

        # Read headers
        ils = np.array([f.header[i][IL_BYTE] for i in range(n_traces)])
        xls = np.array([f.header[i][XL_BYTE] for i in range(n_traces)])

        # Get unique values
        unique_ils = np.sort(np.unique(ils))
        unique_xls = np.sort(np.unique(xls))

        print(f"   IL range: {unique_ils.min()}-{unique_ils.max()}, n={len(unique_ils)}")
        print(f"   XL range: {unique_xls.min()}-{unique_xls.max()}, n={len(unique_xls)}")

        # Read velocity data
        vel_data = f.trace.raw[:]  # (n_traces, n_samples)

        print(f"   Velocity range: {vel_data.min():.0f}-{vel_data.max():.0f} m/s")

    # Reshape to 3D grid (il, xl, t)
    n_il_segy = len(unique_ils)
    n_xl_segy = len(unique_xls)

    vel_3d = np.zeros((n_il_segy, n_xl_segy, n_samples_segy), dtype=np.float32)

    # Create index mapping
    il_idx = {il: i for i, il in enumerate(unique_ils)}
    xl_idx = {xl: i for i, xl in enumerate(unique_xls)}

    for i in range(n_traces):
        ii = il_idx[ils[i]]
        jj = xl_idx[xls[i]]
        vel_3d[ii, jj, :] = vel_data[i, :]

    print(f"   Reshaped to: {vel_3d.shape} (IL, XL, T)")

    # =========================================================================
    # Step 2: Define output grid
    # =========================================================================
    print("\n2. Defining output PSTM grid...")

    # Output grid
    out_ils = np.arange(GRID_IL_MIN, GRID_IL_MAX + 1)  # 1 to 511
    out_xls = np.arange(GRID_XL_MIN, GRID_XL_MAX + 1)  # 1 to 427
    out_times = np.arange(T_MIN_MS, T_MAX_MS + DT_MS, DT_MS)  # 0 to 2000 ms

    n_il_out = len(out_ils)
    n_xl_out = len(out_xls)
    n_t_out = len(out_times)

    print(f"   Output IL: {out_ils[0]}-{out_ils[-1]}, n={n_il_out}")
    print(f"   Output XL: {out_xls[0]}-{out_xls[-1]}, n={n_xl_out}")
    print(f"   Output T: {out_times[0]}-{out_times[-1]} ms, n={n_t_out}")
    print(f"   Output shape: ({n_il_out}, {n_xl_out}, {n_t_out})")

    # =========================================================================
    # Step 3: Interpolate velocity to output grid
    # =========================================================================
    print("\n3. Interpolating velocity to PSTM grid...")

    # Create interpolator (input grid)
    interpolator = RegularGridInterpolator(
        (unique_ils.astype(float), unique_xls.astype(float), t_segy.astype(float)),
        vel_3d,
        method='linear',
        bounds_error=False,
        fill_value=None,  # Extrapolate
    )

    # Create output grid points
    il_grid, xl_grid, t_grid = np.meshgrid(
        out_ils.astype(float),
        out_xls.astype(float),
        out_times.astype(float),
        indexing='ij'
    )
    points = np.stack([il_grid.ravel(), xl_grid.ravel(), t_grid.ravel()], axis=-1)

    # Interpolate
    vel_interp = interpolator(points).reshape(n_il_out, n_xl_out, n_t_out)
    vel_interp = vel_interp.astype(np.float32)

    print(f"   Interpolated shape: {vel_interp.shape}")
    print(f"   Velocity range: {vel_interp.min():.0f}-{vel_interp.max():.0f} m/s")

    # =========================================================================
    # Step 4: Save to zarr
    # =========================================================================
    print("\n4. Saving to zarr...")

    # Create zarr array
    store = zarr.storage.LocalStore(str(OUTPUT_ZARR))
    z = zarr.create_array(
        store=store,
        shape=vel_interp.shape,
        dtype=np.float32,
        chunks=(n_il_out, n_xl_out, 100),
        overwrite=True,
    )
    z[:] = vel_interp

    # Add metadata
    z.attrs['il_min'] = int(GRID_IL_MIN)
    z.attrs['il_max'] = int(GRID_IL_MAX)
    z.attrs['xl_min'] = int(GRID_XL_MIN)
    z.attrs['xl_max'] = int(GRID_XL_MAX)
    z.attrs['t_min_ms'] = float(T_MIN_MS)
    z.attrs['t_max_ms'] = float(T_MAX_MS)
    z.attrs['dt_ms'] = float(DT_MS)
    z.attrs['dx_m'] = float(DX)
    z.attrs['dy_m'] = float(DY)
    z.attrs['n_il'] = int(n_il_out)
    z.attrs['n_xl'] = int(n_xl_out)
    z.attrs['n_t'] = int(n_t_out)
    z.attrs['source_segy'] = str(VELOCITY_SEGY)
    z.attrs['il_byte'] = int(IL_BYTE)
    z.attrs['xl_byte'] = int(XL_BYTE)

    print(f"   Saved to: {OUTPUT_ZARR}")
    print(f"   Size: {vel_interp.nbytes / 1e9:.2f} GB")

    # =========================================================================
    # Step 5: Create QC plots
    # =========================================================================
    print("\n5. Creating QC plots...")

    # Plot inline sections
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Select inline locations
    il_locs = [50, 150, 255, 350, 450]
    for ax, il in zip(axes[0, :], il_locs[:3]):
        il_idx = il - GRID_IL_MIN
        if 0 <= il_idx < n_il_out:
            im = ax.imshow(
                vel_interp[il_idx, :, :].T,
                aspect='auto',
                extent=[out_xls[0], out_xls[-1], out_times[-1], out_times[0]],
                cmap='jet',
                vmin=1500, vmax=5500
            )
            ax.set_title(f'Inline {il}')
            ax.set_xlabel('Crossline')
            ax.set_ylabel('Time (ms)')
            plt.colorbar(im, ax=ax, label='Velocity (m/s)')

    # Plot crossline sections
    xl_locs = [50, 200, 350]
    for ax, xl in zip(axes[1, :], xl_locs):
        xl_idx = xl - GRID_XL_MIN
        if 0 <= xl_idx < n_xl_out:
            im = ax.imshow(
                vel_interp[:, xl_idx, :].T,
                aspect='auto',
                extent=[out_ils[0], out_ils[-1], out_times[-1], out_times[0]],
                cmap='jet',
                vmin=1500, vmax=5500
            )
            ax.set_title(f'Crossline {xl}')
            ax.set_xlabel('Inline')
            ax.set_ylabel('Time (ms)')
            plt.colorbar(im, ax=ax, label='Velocity (m/s)')

    plt.suptitle('Velocity Model QC - IL/XL Sections', fontsize=14)
    plt.tight_layout()
    plt.savefig(QC_DIR / 'velocity_ilxl_sections.png', dpi=150)
    plt.close()
    print(f"   Saved: {QC_DIR / 'velocity_ilxl_sections.png'}")

    # Time slice plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    t_locs = [200, 500, 800, 1000, 1500, 2000]

    for ax, t in zip(axes.flat, t_locs):
        t_idx = int((t - T_MIN_MS) / DT_MS)
        if 0 <= t_idx < n_t_out:
            im = ax.imshow(
                vel_interp[:, :, t_idx].T,
                aspect='auto',
                extent=[out_ils[0], out_ils[-1], out_xls[-1], out_xls[0]],
                cmap='jet',
                vmin=1500, vmax=5500
            )
            ax.set_title(f'Time = {t} ms')
            ax.set_xlabel('Inline')
            ax.set_ylabel('Crossline')
            plt.colorbar(im, ax=ax, label='Velocity (m/s)')

    plt.suptitle('Velocity Model QC - Time Slices', fontsize=14)
    plt.tight_layout()
    plt.savefig(QC_DIR / 'velocity_time_slices.png', dpi=150)
    plt.close()
    print(f"   Saved: {QC_DIR / 'velocity_time_slices.png'}")

    # Velocity gradient plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Average velocity profile (time)
    ax = axes[0]
    mean_vel = vel_interp.mean(axis=(0, 1))
    ax.plot(mean_vel, out_times)
    ax.invert_yaxis()
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Average Velocity Profile')
    ax.grid(True)

    # Velocity histogram
    ax = axes[1]
    ax.hist(vel_interp.ravel(), bins=100, edgecolor='none')
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Count')
    ax.set_title('Velocity Distribution')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(QC_DIR / 'velocity_statistics.png', dpi=150)
    plt.close()
    print(f"   Saved: {QC_DIR / 'velocity_statistics.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Velocity cube: {OUTPUT_ZARR}")
    print(f"Shape: {vel_interp.shape} (IL, XL, T)")
    print(f"IL: {GRID_IL_MIN}-{GRID_IL_MAX} (n={n_il_out})")
    print(f"XL: {GRID_XL_MIN}-{GRID_XL_MAX} (n={n_xl_out})")
    print(f"T: {T_MIN_MS}-{T_MAX_MS} ms (n={n_t_out})")
    print(f"Velocity: {vel_interp.min():.0f}-{vel_interp.max():.0f} m/s")
    print(f"QC plots: {QC_DIR}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
