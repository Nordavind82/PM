#!/usr/bin/env python3
"""
Create 3D Synthetic Common Offset Data with Point Diffractors.

VERSION 2: With realistic offset and azimuth variation within bins,
similar to real common offset gathers.

Creates:
- 3D velocity model with lateral and vertical variations
- Common offset synthetic data with offset/azimuth variation
- Multiple traces per CDP to simulate real acquisition
- 3 point diffractors at different times
"""

import numpy as np
import polars as pl
import zarr
from pathlib import Path
from scipy.ndimage import gaussian_filter

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/synthetic_diffractor_v2")

# Grid parameters
NX = 201  # Number of inlines
NY = 161  # Number of crosslines
NT = 501  # Number of time samples
DX = 25.0  # Inline spacing (m)
DY = 12.5  # Crossline spacing (m)
DT_MS = 2.0  # Time sample interval (ms)
T_MAX_MS = 1000.0  # Max time

# Offset bin parameters (realistic variation)
NOMINAL_OFFSET_M = 300.0  # Center of offset bin
OFFSET_VARIATION_M = 20.0  # +/- variation (like 20m bin width)
AZIMUTH_VARIATION_DEG = 15.0  # +/- azimuth variation (degrees)

# Number of traces per CDP (to simulate fold)
TRACES_PER_CDP = 3  # Creates ~3x more traces with offset/azimuth variation

# Grid origin
X_ORIGIN = 620000.0
Y_ORIGIN = 5115000.0

# Grid rotation
ROTATION_DEG = -49.5  # IL direction azimuth from East

# Diffractor locations (ix, iy, t_ms)
DIFFRACTORS = [
    (NX // 4, NY // 2, 300.0),      # Shallow diffractor
    (NX // 2, NY // 2, 500.0),      # Middle diffractor
    (3 * NX // 4, NY // 2, 700.0),  # Deep diffractor
]

# Velocity model parameters
V0 = 1700.0  # Surface velocity (m/s)
K = 0.8      # Velocity gradient (m/s per ms)
LATERAL_VAR = 0.03  # 3% lateral variation (reduced for cleaner test)

# Wavelet parameters
WAVELET_FREQ_HZ = 25.0  # Lower frequency for cleaner response


# =============================================================================
# Helper Functions
# =============================================================================

def create_velocity_model():
    """Create 3D velocity model with lateral and vertical variations."""
    print("Creating 3D velocity model...")

    t_axis = np.arange(NT) * DT_MS
    v_base = V0 + K * t_axis

    velocity = np.zeros((NX, NY, NT), dtype=np.float32)
    for it in range(NT):
        velocity[:, :, it] = v_base[it]

    # Add smooth lateral variation
    np.random.seed(42)
    lateral_field = np.random.randn(NX, NY) * LATERAL_VAR
    lateral_field = gaussian_filter(lateral_field, sigma=25)

    for it in range(NT):
        velocity[:, :, it] *= (1.0 + lateral_field)

    print(f"  Shape: {velocity.shape}")
    print(f"  V(t=0): {velocity[:,:,0].mean():.0f} m/s")
    print(f"  V(t=500ms): {velocity[:,:,250].mean():.0f} m/s")
    print(f"  V(t=1000ms): {velocity[:,:,-1].mean():.0f} m/s")
    print(f"  Lateral variation: {100*velocity[:,:,250].std()/velocity[:,:,250].mean():.1f}%")

    return velocity


def compute_grid_coordinates():
    """Compute rotated grid coordinates."""
    theta = np.radians(ROTATION_DEG)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    il_vec = np.array([cos_t, sin_t]) * DX
    xl_vec = np.array([-sin_t, cos_t]) * DY

    il_idx = np.arange(NX)
    xl_idx = np.arange(NY)
    IL, XL = np.meshgrid(il_idx, xl_idx, indexing='ij')

    X = X_ORIGIN + IL * il_vec[0] + XL * xl_vec[0]
    Y = Y_ORIGIN + IL * il_vec[1] + XL * xl_vec[1]

    return X, Y, il_vec, xl_vec


def compute_dsr_traveltime(sx, sy, rx, ry, ox, oy, t0_ms, velocity):
    """Compute DSR (Double Square Root) traveltime."""
    t0_s = t0_ms / 1000.0
    t0_half = t0_s / 2.0

    ds = np.sqrt((ox - sx)**2 + (oy - sy)**2)
    dr = np.sqrt((ox - rx)**2 + (oy - ry)**2)

    t_travel = np.sqrt(t0_half**2 + (ds/velocity)**2) + np.sqrt(t0_half**2 + (dr/velocity)**2)

    return t_travel * 1000.0


def ricker_wavelet(t_center_ms, t_axis_ms, f_dom):
    """Generate Ricker wavelet centered at t_center_ms."""
    wavelet = np.zeros(len(t_axis_ms), dtype=np.float32)
    for it, t in enumerate(t_axis_ms):
        tau = (t - t_center_ms) / 1000.0
        arg = (np.pi * f_dom * tau) ** 2
        wavelet[it] = (1 - 2 * arg) * np.exp(-arg)
    return wavelet


def create_synthetic_traces(velocity, X_grid, Y_grid, il_vec, xl_vec):
    """
    Create synthetic traces with realistic offset and azimuth variation.

    For each CDP, creates multiple traces with varying offset and azimuth
    to simulate real common offset gather acquisition.
    """
    print("\nCreating synthetic traces with offset/azimuth variation...")
    print(f"  Nominal offset: {NOMINAL_OFFSET_M} m")
    print(f"  Offset variation: +/- {OFFSET_VARIATION_M} m")
    print(f"  Azimuth variation: +/- {AZIMUTH_VARIATION_DEG} deg")
    print(f"  Traces per CDP: {TRACES_PER_CDP}")

    traces_list = []
    headers_list = []

    t_axis = np.arange(NT) * DT_MS

    # Base azimuth (perpendicular to inline)
    base_azimuth_deg = ROTATION_DEG + 90

    np.random.seed(123)  # Reproducible randomness

    trace_idx = 0
    for ix in range(NX):
        for iy in range(NY):
            # CDP location
            cdp_x = X_grid[ix, iy]
            cdp_y = Y_grid[ix, iy]

            # Generate multiple traces per CDP with varying offset/azimuth
            for trace_in_cdp in range(TRACES_PER_CDP):
                # Random offset within bin
                offset = NOMINAL_OFFSET_M + np.random.uniform(-OFFSET_VARIATION_M, OFFSET_VARIATION_M)
                half_offset = offset / 2.0

                # Random azimuth variation
                azimuth_deg = base_azimuth_deg + np.random.uniform(-AZIMUTH_VARIATION_DEG, AZIMUTH_VARIATION_DEG)
                azimuth_rad = np.radians(azimuth_deg)

                # Source and receiver positions
                offset_dx = half_offset * np.cos(azimuth_rad)
                offset_dy = half_offset * np.sin(azimuth_rad)

                sx = cdp_x - offset_dx
                sy = cdp_y - offset_dy
                rx = cdp_x + offset_dx
                ry = cdp_y + offset_dy

                # Create trace
                trace = np.zeros(NT, dtype=np.float32)

                # Add each diffractor
                for diff_ix, diff_iy, diff_t0 in DIFFRACTORS:
                    diff_x = X_grid[diff_ix, diff_iy]
                    diff_y = Y_grid[diff_ix, diff_iy]

                    t_idx = int(diff_t0 / DT_MS)
                    v_at_diff = velocity[diff_ix, diff_iy, t_idx]

                    t_travel = compute_dsr_traveltime(
                        sx, sy, rx, ry, diff_x, diff_y, diff_t0, v_at_diff
                    )

                    if 0 < t_travel < T_MAX_MS - 50:
                        # Distance-based amplitude decay
                        dist = np.sqrt((cdp_x - diff_x)**2 + (cdp_y - diff_y)**2)
                        amp = 1.0 / max(dist / 100.0, 1.0)

                        # Add Ricker wavelet at traveltime
                        wavelet = ricker_wavelet(t_travel, t_axis, WAVELET_FREQ_HZ)
                        trace += amp * wavelet

                traces_list.append(trace)

                # Header with actual offset and azimuth
                headers_list.append({
                    'trace_index': trace_idx,
                    'bin_trace_idx': trace_idx,
                    'inline': ix + 1,
                    'crossline': iy + 1,
                    'source_x': int(sx * 100),
                    'source_y': int(sy * 100),
                    'receiver_x': int(rx * 100),
                    'receiver_y': int(ry * 100),
                    'offset': offset,  # Actual offset (varies)
                    'scalar_coord': -100,
                    'sr_azim': azimuth_deg,  # Actual azimuth (varies)
                })

                trace_idx += 1

        if (ix + 1) % 50 == 0:
            print(f"  Progress: {ix+1}/{NX} inlines")

    traces = np.array(traces_list, dtype=np.float32)

    # Statistics
    offsets = np.array([h['offset'] for h in headers_list])
    azimuths = np.array([h['sr_azim'] for h in headers_list])

    print(f"\n  Created {len(traces)} traces")
    print(f"  Offset range: {offsets.min():.1f} - {offsets.max():.1f} m (mean: {offsets.mean():.1f})")
    print(f"  Azimuth range: {azimuths.min():.1f} - {azimuths.max():.1f} deg (mean: {azimuths.mean():.1f})")
    print(f"  Trace amplitude range: {traces.min():.4f} to {traces.max():.4f}")

    return traces, headers_list


def save_synthetic_data(velocity, traces, headers, X_grid, Y_grid):
    """Save synthetic data in pipeline-compatible format."""
    print("\nSaving synthetic data...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save velocity model
    vel_path = OUTPUT_DIR / "velocity.zarr"
    print(f"  Saving velocity to {vel_path}")

    z_vel = zarr.open_array(str(vel_path), mode='w',
                            shape=velocity.shape, dtype=velocity.dtype,
                            chunks=(32, 32, NT))
    z_vel[:] = velocity

    z_vel.attrs['x_axis'] = list(range(1, NX + 1))
    z_vel.attrs['y_axis'] = list(range(1, NY + 1))
    z_vel.attrs['t_axis_ms'] = list(np.arange(NT) * DT_MS)
    z_vel.attrs['dt_ms'] = DT_MS
    z_vel.attrs['dx_m'] = DX
    z_vel.attrs['dy_m'] = DY
    z_vel.attrs['n_il'] = NX
    z_vel.attrs['n_xl'] = NY
    z_vel.attrs['n_t'] = NT

    # Save traces (transposed format: nt, n_traces)
    traces_path = OUTPUT_DIR / "offset_bin_15" / "traces.zarr"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving traces to {traces_path}")

    traces_t = traces.T
    z_traces = zarr.open_array(str(traces_path), mode='w',
                               shape=traces_t.shape, dtype=traces_t.dtype,
                               chunks=(NT, min(10000, traces_t.shape[1])))
    z_traces[:] = traces_t

    # Save headers
    headers_path = OUTPUT_DIR / "offset_bin_15" / "headers.parquet"
    print(f"  Saving headers to {headers_path}")
    df = pl.DataFrame(headers)
    df.write_parquet(headers_path)

    # Compute statistics
    offsets = df['offset'].to_numpy()
    azimuths = df['sr_azim'].to_numpy()

    # Save grid info
    grid_info = {
        'nx': NX,
        'ny': NY,
        'nt': NT,
        'dx': DX,
        'dy': DY,
        'dt_ms': DT_MS,
        'x_origin': X_ORIGIN,
        'y_origin': Y_ORIGIN,
        'rotation_deg': ROTATION_DEG,
        'nominal_offset_m': NOMINAL_OFFSET_M,
        'offset_variation_m': OFFSET_VARIATION_M,
        'azimuth_variation_deg': AZIMUTH_VARIATION_DEG,
        'traces_per_cdp': TRACES_PER_CDP,
        'offset_min': float(offsets.min()),
        'offset_max': float(offsets.max()),
        'offset_mean': float(offsets.mean()),
        'azimuth_min': float(azimuths.min()),
        'azimuth_max': float(azimuths.max()),
        'azimuth_mean': float(azimuths.mean()),
        'diffractors': DIFFRACTORS,
        'wavelet_freq_hz': WAVELET_FREQ_HZ,
        'c1': (float(X_grid[0, 0]), float(Y_grid[0, 0])),
        'c2': (float(X_grid[-1, 0]), float(Y_grid[-1, 0])),
        'c3': (float(X_grid[-1, -1]), float(Y_grid[-1, -1])),
        'c4': (float(X_grid[0, -1]), float(Y_grid[0, -1])),
    }

    import json
    with open(OUTPUT_DIR / "grid_info.json", 'w') as f:
        json.dump(grid_info, f, indent=2, default=str)

    print(f"\nGrid corners (UTM):")
    print(f"  C1 (IL=1, XL=1): {grid_info['c1']}")
    print(f"  C2 (IL={NX}, XL=1): {grid_info['c2']}")
    print(f"  C3 (IL={NX}, XL={NY}): {grid_info['c3']}")
    print(f"  C4 (IL=1, XL={NY}): {grid_info['c4']}")

    return grid_info


def create_qc_images(traces, headers, X_grid, Y_grid):
    """Create QC images showing offset/azimuth distribution."""
    import matplotlib.pyplot as plt

    print("\nCreating QC images...")

    images_dir = OUTPUT_DIR / "qc_images"
    images_dir.mkdir(exist_ok=True)

    offsets = np.array([h['offset'] for h in headers])
    azimuths = np.array([h['sr_azim'] for h in headers])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Synthetic Data QC - Offset/Azimuth Distribution\n'
                 f'Nominal offset: {NOMINAL_OFFSET_M}m, Variation: ±{OFFSET_VARIATION_M}m, '
                 f'Azimuth var: ±{AZIMUTH_VARIATION_DEG}°',
                 fontsize=12, fontweight='bold')

    # Offset histogram
    ax = axes[0, 0]
    ax.hist(offsets, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=NOMINAL_OFFSET_M, color='r', linestyle='--', linewidth=2, label=f'Nominal: {NOMINAL_OFFSET_M}m')
    ax.set_xlabel('Offset (m)')
    ax.set_ylabel('Count')
    ax.set_title('Offset Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Azimuth histogram
    ax = axes[0, 1]
    ax.hist(azimuths, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(x=ROTATION_DEG + 90, color='r', linestyle='--', linewidth=2, label=f'Base: {ROTATION_DEG+90:.1f}°')
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Azimuth Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Offset vs Azimuth scatter
    ax = axes[1, 0]
    ax.scatter(offsets[::10], azimuths[::10], alpha=0.3, s=1)  # Subsample for speed
    ax.set_xlabel('Offset (m)')
    ax.set_ylabel('Azimuth (degrees)')
    ax.set_title('Offset vs Azimuth (subsampled)')
    ax.grid(True, alpha=0.3)

    # Sample traces
    ax = axes[1, 1]
    t_axis = np.arange(NT) * DT_MS
    center_il = NX // 2
    center_xl = NY // 2
    center_cdp_traces = [i for i, h in enumerate(headers)
                         if h['inline'] == center_il + 1 and h['crossline'] == center_xl + 1]
    for i, trace_idx in enumerate(center_cdp_traces[:5]):
        trace = traces[trace_idx]
        offset = headers[trace_idx]['offset']
        ax.plot(t_axis, trace + i*0.5, label=f'Offset: {offset:.0f}m')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (offset)')
    ax.set_title(f'Sample Traces at Center CDP (IL={center_il+1}, XL={center_xl+1})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = images_dir / 'offset_azimuth_distribution.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")

    # Crossline gather
    fig, ax = plt.subplots(figsize=(16, 10))
    center_xl = NY // 2
    xl_traces = [(i, h) for i, h in enumerate(headers) if h['crossline'] == center_xl + 1]

    # Sort by inline
    xl_traces.sort(key=lambda x: (x[1]['inline'], x[1]['offset']))

    # Plot as gather
    gather = np.zeros((len(xl_traces), NT))
    for i, (trace_idx, h) in enumerate(xl_traces):
        gather[i, :] = traces[trace_idx]

    vmax = np.percentile(np.abs(gather), 99)
    ax.imshow(gather.T, aspect='auto', cmap='gray',
              extent=[0, len(xl_traces), t_axis[-1], 0],
              vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax.set_xlabel('Trace Number')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Crossline Gather (XL={center_xl+1}) - All Traces')

    plt.tight_layout()
    fig_path = images_dir / 'crossline_gather.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")


def main():
    print("=" * 70)
    print("Creating 3D Synthetic with Offset/Azimuth Variation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Grid: {NX} x {NY} x {NT} (IL x XL x T)")
    print(f"  Spacing: dx={DX}m, dy={DY}m, dt={DT_MS}ms")
    print(f"  Nominal offset: {NOMINAL_OFFSET_M}m (+/- {OFFSET_VARIATION_M}m)")
    print(f"  Azimuth variation: +/- {AZIMUTH_VARIATION_DEG} degrees")
    print(f"  Traces per CDP: {TRACES_PER_CDP}")
    print(f"  Wavelet frequency: {WAVELET_FREQ_HZ} Hz")
    print(f"  Rotation: {ROTATION_DEG} degrees")
    print(f"\nDiffractors:")
    for i, (ix, iy, t0) in enumerate(DIFFRACTORS):
        print(f"  {i+1}. IL={ix+1}, XL={iy+1}, t0={t0}ms")

    # Create velocity model
    velocity = create_velocity_model()

    # Compute grid coordinates
    X_grid, Y_grid, il_vec, xl_vec = compute_grid_coordinates()

    # Create synthetic traces
    traces, headers = create_synthetic_traces(velocity, X_grid, Y_grid, il_vec, xl_vec)

    # Save data
    grid_info = save_synthetic_data(velocity, traces, headers, X_grid, Y_grid)

    # Create QC images
    create_qc_images(traces, headers, X_grid, Y_grid)

    print("\n" + "=" * 70)
    print("Synthetic Data Creation Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total traces: {len(traces)}")
    print("=" * 70)

    return grid_info


if __name__ == "__main__":
    main()
