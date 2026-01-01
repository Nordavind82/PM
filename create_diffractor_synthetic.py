#!/usr/bin/env python3
"""
Create 3D Synthetic Common Offset Data with Point Diffractors.

Creates:
- 3D velocity model with lateral and vertical variations
- 300m common offset synthetic data with 3 point diffractors at different times
- Saves in format compatible with existing PSTM pipeline
"""

import numpy as np
import polars as pl
import zarr
from pathlib import Path
from scipy.ndimage import gaussian_filter

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/synthetic_diffractor")

# Grid parameters (matching real data)
NX = 201  # Number of inlines (smaller for faster processing)
NY = 161  # Number of crosslines
NT = 501  # Number of time samples
DX = 25.0  # Inline spacing (m)
DY = 12.5  # Crossline spacing (m)
DT_MS = 2.0  # Time sample interval (ms)
T_MAX_MS = 1000.0  # Max time

# Common offset
OFFSET_M = 300.0  # meters

# Grid origin (UTM-like coordinates)
X_ORIGIN = 620000.0
Y_ORIGIN = 5115000.0

# Grid rotation (similar to real data: ~50 degrees)
ROTATION_DEG = -49.5  # IL direction azimuth from East

# Diffractor locations (ix, iy, t_ms)
DIFFRACTORS = [
    (NX // 4, NY // 2, 300.0),      # Shallow diffractor
    (NX // 2, NY // 2, 500.0),      # Middle diffractor
    (3 * NX // 4, NY // 2, 700.0),  # Deep diffractor
]

# Velocity model parameters
V0 = 1700.0  # Surface velocity (m/s)
K = 0.8      # Velocity gradient (m/s per ms of two-way time)
LATERAL_VAR = 0.05  # 5% lateral variation


# =============================================================================
# Helper Functions
# =============================================================================

def create_velocity_model():
    """Create 3D velocity model with lateral and vertical variations."""
    print("Creating 3D velocity model...")

    # Time axis
    t_axis = np.arange(NT) * DT_MS

    # Base velocity: V(t) = V0 + K * t
    v_base = V0 + K * t_axis

    # Create 3D cube
    velocity = np.zeros((NX, NY, NT), dtype=np.float32)

    # Fill with base velocity
    for it in range(NT):
        velocity[:, :, it] = v_base[it]

    # Add smooth lateral variation
    np.random.seed(42)
    # Create random field and smooth it
    lateral_field = np.random.randn(NX, NY) * LATERAL_VAR
    lateral_field = gaussian_filter(lateral_field, sigma=20)

    # Apply lateral variation
    for it in range(NT):
        velocity[:, :, it] *= (1.0 + lateral_field)

    print(f"  Shape: {velocity.shape}")
    print(f"  V(t=0): {velocity[:,:,0].mean():.0f} m/s (range: {velocity[:,:,0].min():.0f}-{velocity[:,:,0].max():.0f})")
    print(f"  V(t=500ms): {velocity[:,:,250].mean():.0f} m/s")
    print(f"  V(t=1000ms): {velocity[:,:,-1].mean():.0f} m/s")
    print(f"  Lateral variation: {100*velocity[:,:,250].std()/velocity[:,:,250].mean():.1f}%")

    return velocity


def compute_grid_coordinates():
    """Compute rotated grid coordinates."""
    # Rotation matrix
    theta = np.radians(ROTATION_DEG)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # IL and XL unit vectors in UTM space
    il_vec = np.array([cos_t, sin_t]) * DX
    xl_vec = np.array([-sin_t, cos_t]) * DY

    # Grid indices
    il_idx = np.arange(NX)
    xl_idx = np.arange(NY)
    IL, XL = np.meshgrid(il_idx, xl_idx, indexing='ij')

    # UTM coordinates
    X = X_ORIGIN + IL * il_vec[0] + XL * xl_vec[0]
    Y = Y_ORIGIN + IL * il_vec[1] + XL * xl_vec[1]

    return X, Y, il_vec, xl_vec


def compute_dsr_traveltime(sx, sy, rx, ry, ox, oy, t0_ms, velocity):
    """
    Compute DSR (Double Square Root) traveltime.

    t = sqrt((t0/2)^2 + ds^2/v^2) + sqrt((t0/2)^2 + dr^2/v^2)
    """
    t0_s = t0_ms / 1000.0
    t0_half = t0_s / 2.0

    # Distances
    ds = np.sqrt((ox - sx)**2 + (oy - sy)**2)
    dr = np.sqrt((ox - rx)**2 + (oy - ry)**2)

    # DSR formula
    t_travel = np.sqrt(t0_half**2 + (ds/velocity)**2) + np.sqrt(t0_half**2 + (dr/velocity)**2)

    return t_travel * 1000.0  # Return in ms


def create_synthetic_traces(velocity, X_grid, Y_grid, il_vec, xl_vec):
    """Create synthetic traces with point diffractors."""
    print("\nCreating synthetic traces...")

    # Generate trace geometry for common offset
    # Create traces at each CDP location
    traces_list = []
    headers_list = []

    t_axis = np.arange(NT) * DT_MS

    # Azimuth for offset (perpendicular to inline direction for simplicity)
    azimuth_deg = ROTATION_DEG + 90  # Perpendicular to IL
    azimuth_rad = np.radians(azimuth_deg)

    half_offset = OFFSET_M / 2.0
    offset_dx = half_offset * np.cos(azimuth_rad)
    offset_dy = half_offset * np.sin(azimuth_rad)

    trace_idx = 0
    for ix in range(NX):
        for iy in range(NY):
            # CDP location
            cdp_x = X_grid[ix, iy]
            cdp_y = Y_grid[ix, iy]

            # Source and receiver positions
            sx = cdp_x - offset_dx
            sy = cdp_y - offset_dy
            rx = cdp_x + offset_dx
            ry = cdp_y + offset_dy

            # Create trace (start with zeros)
            trace = np.zeros(NT, dtype=np.float32)

            # Add each diffractor
            for diff_ix, diff_iy, diff_t0 in DIFFRACTORS:
                # Diffractor location in UTM
                diff_x = X_grid[diff_ix, diff_iy]
                diff_y = Y_grid[diff_ix, diff_iy]

                # Get velocity at diffractor time
                t_idx = int(diff_t0 / DT_MS)
                v_at_diff = velocity[diff_ix, diff_iy, t_idx]

                # Compute DSR traveltime
                t_travel = compute_dsr_traveltime(
                    sx, sy, rx, ry, diff_x, diff_y, diff_t0, v_at_diff
                )

                # Add Ricker wavelet at traveltime
                if 0 < t_travel < T_MAX_MS - 50:
                    # Ricker wavelet (30 Hz)
                    f_dom = 30.0
                    t_center = t_travel

                    # Compute amplitude based on distance (simple 1/r spreading)
                    dist = np.sqrt((cdp_x - diff_x)**2 + (cdp_y - diff_y)**2)
                    amp = 1.0 / max(dist / 100.0, 1.0)

                    # Add wavelet
                    for it, t in enumerate(t_axis):
                        tau = (t - t_center) / 1000.0  # Convert to seconds
                        arg = (np.pi * f_dom * tau) ** 2
                        wavelet = amp * (1 - 2 * arg) * np.exp(-arg)
                        trace[it] += wavelet

            traces_list.append(trace)

            # Header
            headers_list.append({
                'trace_index': trace_idx,
                'bin_trace_idx': trace_idx,
                'inline': ix + 1,
                'crossline': iy + 1,
                'source_x': int(sx * 100),  # Store scaled
                'source_y': int(sy * 100),
                'receiver_x': int(rx * 100),
                'receiver_y': int(ry * 100),
                'offset': OFFSET_M,
                'scalar_coord': -100,
                'sr_azim': azimuth_deg,
            })

            trace_idx += 1

        if (ix + 1) % 50 == 0:
            print(f"  Progress: {ix+1}/{NX} inlines")

    traces = np.array(traces_list, dtype=np.float32)
    print(f"  Created {len(traces)} traces")
    print(f"  Trace shape: {traces.shape}")
    print(f"  Amplitude range: {traces.min():.4f} to {traces.max():.4f}")

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

    # Axes as IL/XL numbers (like real data)
    z_vel.attrs['x_axis'] = list(range(1, NX + 1))
    z_vel.attrs['y_axis'] = list(range(1, NY + 1))
    z_vel.attrs['t_axis_ms'] = list(np.arange(NT) * DT_MS)
    z_vel.attrs['dt_ms'] = DT_MS
    z_vel.attrs['dx_m'] = DX
    z_vel.attrs['dy_m'] = DY
    z_vel.attrs['n_il'] = NX
    z_vel.attrs['n_xl'] = NY
    z_vel.attrs['n_t'] = NT

    # Save traces
    traces_path = OUTPUT_DIR / "offset_bin_15" / "traces.zarr"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving traces to {traces_path}")

    # Transpose to (nt, n_traces) format
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
        'offset_m': OFFSET_M,
        'diffractors': DIFFRACTORS,
        # Grid corners
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


def main():
    print("=" * 70)
    print("Creating 3D Synthetic with Point Diffractors")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Grid: {NX} x {NY} x {NT} (IL x XL x T)")
    print(f"  Spacing: dx={DX}m, dy={DY}m, dt={DT_MS}ms")
    print(f"  Offset: {OFFSET_M}m")
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

    print("\n" + "=" * 70)
    print("Synthetic Data Creation Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    return grid_info


if __name__ == "__main__":
    main()
