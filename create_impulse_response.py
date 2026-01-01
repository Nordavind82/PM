#!/usr/bin/env python3
"""
Create 3D Impulse Response for PSTM Migration.

Generates impulse responses at different times/depths to visualize:
1. Migration operator shape (ellipsoid for Kirchhoff)
2. Aperture limits
3. Velocity effects on operator width
4. Artifacts and evanescent energy

Usage:
    python create_impulse_response.py
    python create_impulse_response.py --velocity 2000
    python create_impulse_response.py --impulse-times 300,500,700
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import zarr

sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import (
    MigrationConfig,
    InputConfig,
    VelocityConfig,
    AlgorithmConfig,
    OutputConfig,
    ExecutionConfig,
    OutputGridConfig,
    ApertureConfig,
    AmplitudeConfig,
    ResourceConfig,
    TilingConfig,
    CheckpointConfig,
    VelocitySource,
    InterpolationMethod,
    ComputeBackend,
    OutputFormat,
    TimeVariantConfig,
    AntiAliasingConfig,
    AntiAliasingMethod,
    ColumnMapping,
)
from pstm.pipeline.executor import run_migration

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_BASE = Path("/Users/olegadamovich/SeismicData/impulse_response")

# Grid parameters
NX = 201          # Inlines
NY = 201          # Crosslines
NT = 501          # Time samples
DX = 25.0         # Inline spacing (m)
DY = 25.0         # Crossline spacing (m) - square grid for symmetry
DT_MS = 2.0       # Sample interval (ms)
T_MAX_MS = 1000.0 # Max time

# Common offset
OFFSET_M = 300.0

# Grid origin (arbitrary UTM-like)
X_ORIGIN = 620000.0
Y_ORIGIN = 5115000.0

# Default impulse locations (time in ms)
DEFAULT_IMPULSE_TIMES = [300, 500, 700]

# Default velocity (constant)
DEFAULT_VELOCITY = 2000.0  # m/s

# Migration parameters (matching real data)
MAX_APERTURE_M = 2000.0
MIN_APERTURE_M = 500.0
MAX_DIP_DEGREES = 65.0


def create_impulse_data(impulse_times: list[float], velocity: float) -> tuple[np.ndarray, list[dict], dict]:
    """
    Create synthetic data with impulses at specified times.

    For each impulse, we place a spike at the center of the grid.
    The spike travels along the DSR hyperbola.

    Returns:
        traces: (n_traces, nt) array
        headers: list of header dicts
        grid_info: dict with grid parameters
    """
    print(f"\nCreating impulse data...")
    print(f"  Impulse times: {impulse_times} ms")
    print(f"  Velocity: {velocity} m/s")

    # Grid coordinates (no rotation for simplicity)
    il_idx = np.arange(NX)
    xl_idx = np.arange(NY)

    # UTM coordinates
    X_grid = X_ORIGIN + il_idx[:, None] * DX
    Y_grid = Y_ORIGIN + xl_idx[None, :] * DY

    # Center of grid (where impulses are located)
    center_il = NX // 2
    center_xl = NY // 2
    center_x = X_grid[center_il, 0]
    center_y = Y_grid[0, center_xl]

    print(f"  Grid center: IL={center_il+1}, XL={center_xl+1}")
    print(f"  UTM center: X={center_x:.1f}, Y={center_y:.1f}")

    # Time axis
    t_axis = np.arange(NT) * DT_MS

    # Offset geometry (inline direction)
    half_offset = OFFSET_M / 2.0

    traces_list = []
    headers_list = []

    trace_idx = 0
    for ix in range(NX):
        for iy in range(NY):
            # CDP location
            cdp_x = X_grid[ix, 0]
            cdp_y = Y_grid[0, iy]

            # Source and receiver (offset in X direction)
            sx = cdp_x - half_offset
            sy = cdp_y
            rx = cdp_x + half_offset
            ry = cdp_y

            # Create trace
            trace = np.zeros(NT, dtype=np.float32)

            # Add impulse for each specified time
            for t0_ms in impulse_times:
                # Distance from CDP to impulse location (center)
                dist_x = cdp_x - center_x
                dist_y = cdp_y - center_y

                # Source and receiver distances to impulse
                ds = np.sqrt((sx - center_x)**2 + (sy - center_y)**2)
                dr = np.sqrt((rx - center_x)**2 + (ry - center_y)**2)

                # DSR traveltime
                t0_s = t0_ms / 1000.0
                t0_half = t0_s / 2.0

                t_travel_s = np.sqrt(t0_half**2 + (ds/velocity)**2) + \
                             np.sqrt(t0_half**2 + (dr/velocity)**2)
                t_travel_ms = t_travel_s * 1000.0

                # Place spike at traveltime
                if 0 < t_travel_ms < T_MAX_MS - 10:
                    t_idx = int(t_travel_ms / DT_MS)
                    if 0 <= t_idx < NT:
                        # Use a short wavelet instead of pure spike for stability
                        # Ricker wavelet centered at t_travel
                        f_dom = 40.0  # Hz - higher frequency for sharper impulse
                        for it in range(max(0, t_idx-25), min(NT, t_idx+25)):
                            tau = (it * DT_MS - t_travel_ms) / 1000.0
                            arg = (np.pi * f_dom * tau) ** 2
                            trace[it] += (1 - 2 * arg) * np.exp(-arg)

            traces_list.append(trace)

            # Header
            headers_list.append({
                'trace_index': trace_idx,
                'bin_trace_idx': trace_idx,
                'inline': ix + 1,
                'crossline': iy + 1,
                'source_x': int(sx * 100),
                'source_y': int(sy * 100),
                'receiver_x': int(rx * 100),
                'receiver_y': int(ry * 100),
                'offset': OFFSET_M,
                'scalar_coord': -100,
                'sr_azim': 90.0,  # E-W offset
            })

            trace_idx += 1

        if (ix + 1) % 50 == 0:
            print(f"    Progress: {ix+1}/{NX} inlines")

    traces = np.array(traces_list, dtype=np.float32)

    # Grid info
    grid_info = {
        'nx': NX,
        'ny': NY,
        'nt': NT,
        'dx': DX,
        'dy': DY,
        'dt_ms': DT_MS,
        'x_origin': X_ORIGIN,
        'y_origin': Y_ORIGIN,
        'offset_m': OFFSET_M,
        'velocity': velocity,
        'impulse_times': impulse_times,
        'center_il': center_il,
        'center_xl': center_xl,
        'c1': (X_ORIGIN, Y_ORIGIN),
        'c2': (X_ORIGIN + (NX-1)*DX, Y_ORIGIN),
        'c3': (X_ORIGIN + (NX-1)*DX, Y_ORIGIN + (NY-1)*DY),
        'c4': (X_ORIGIN, Y_ORIGIN + (NY-1)*DY),
    }

    print(f"  Created {len(traces)} traces")
    print(f"  Trace amplitude range: {traces.min():.4f} to {traces.max():.4f}")

    return traces, headers_list, grid_info


def create_velocity_model(velocity: float, gradient: float = 0.0) -> np.ndarray:
    """Create velocity model (constant or with gradient)."""
    print(f"\nCreating velocity model...")
    print(f"  Base velocity: {velocity} m/s")
    print(f"  Gradient: {gradient} m/s per ms")

    vel = np.zeros((NX, NY, NT), dtype=np.float32)

    t_axis = np.arange(NT) * DT_MS
    for it in range(NT):
        vel[:, :, it] = velocity + gradient * t_axis[it]

    print(f"  V(t=0): {vel[:,:,0].mean():.0f} m/s")
    print(f"  V(t=500ms): {vel[:,:,250].mean():.0f} m/s")
    print(f"  V(t=1000ms): {vel[:,:,-1].mean():.0f} m/s")

    return vel


def save_impulse_data(traces: np.ndarray, headers: list[dict],
                      velocity: np.ndarray, grid_info: dict, output_dir: Path):
    """Save impulse data to disk."""
    print(f"\nSaving data to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save velocity
    vel_path = output_dir / "velocity.zarr"
    z_vel = zarr.open_array(str(vel_path), mode='w',
                            shape=velocity.shape, dtype=velocity.dtype,
                            chunks=(32, 32, NT))
    z_vel[:] = velocity
    z_vel.attrs['x_axis'] = list(range(1, NX + 1))
    z_vel.attrs['y_axis'] = list(range(1, NY + 1))
    z_vel.attrs['t_axis_ms'] = list(np.arange(NT) * DT_MS)
    z_vel.attrs['dt_ms'] = DT_MS

    # Save traces (transposed: nt, n_traces)
    traces_dir = output_dir / "impulse_data"
    traces_dir.mkdir(exist_ok=True)
    traces_path = traces_dir / "traces.zarr"
    traces_t = traces.T
    z_traces = zarr.open_array(str(traces_path), mode='w',
                               shape=traces_t.shape, dtype=traces_t.dtype,
                               chunks=(NT, min(10000, traces_t.shape[1])))
    z_traces[:] = traces_t

    # Save headers
    headers_path = traces_dir / "headers.parquet"
    df = pl.DataFrame(headers)
    df.write_parquet(headers_path)

    # Save grid info
    with open(output_dir / "grid_info.json", 'w') as f:
        json.dump(grid_info, f, indent=2)

    print("  Saved velocity, traces, headers, and grid info")


def run_impulse_migration(output_dir: Path, grid_info: dict) -> bool:
    """Run PSTM migration on impulse data."""
    print(f"\n{'='*60}")
    print("Running PSTM Migration on Impulse Data")
    print(f"{'='*60}")

    input_dir = output_dir / "impulse_data"
    migration_dir = output_dir / "migration_output"
    migration_dir.mkdir(parents=True, exist_ok=True)

    velocity_path = output_dir / "velocity.zarr"

    # Build configuration
    input_config = InputConfig(
        traces_path=input_dir / "traces.zarr",
        headers_path=input_dir / "headers.parquet",
        columns=ColumnMapping(
            source_x="source_x",
            source_y="source_y",
            receiver_x="receiver_x",
            receiver_y="receiver_y",
            offset="offset",
            azimuth="sr_azim",
            trace_index="bin_trace_idx",
            coord_scalar="scalar_coord",
        ),
        apply_coord_scalar=True,
        sample_rate_ms=grid_info['dt_ms'],
        transposed=True,
    )

    velocity_config = VelocityConfig(
        source=VelocitySource.CUBE_3D,
        velocity_path=velocity_path,
        precompute_to_output_grid=True,
    )

    aperture_config = ApertureConfig(
        max_dip_degrees=MAX_DIP_DEGREES,
        min_aperture_m=MIN_APERTURE_M,
        max_aperture_m=MAX_APERTURE_M,
        taper_fraction=0.1,
    )

    amplitude_config = AmplitudeConfig(
        geometrical_spreading=False,
        obliquity_factor=False,
    )

    time_variant_config = TimeVariantConfig(enabled=False)
    anti_aliasing_config = AntiAliasingConfig(enabled=False, method=AntiAliasingMethod.NONE)

    algorithm_config = AlgorithmConfig(
        interpolation=InterpolationMethod.LINEAR,
        aperture=aperture_config,
        amplitude=amplitude_config,
        time_variant=time_variant_config,
        anti_aliasing=anti_aliasing_config,
    )

    output_grid = OutputGridConfig.from_corners(
        corner1=tuple(grid_info['c1']),
        corner2=tuple(grid_info['c2']),
        corner3=tuple(grid_info['c3']),
        corner4=tuple(grid_info['c4']),
        t_min_ms=0.0,
        t_max_ms=grid_info['nt'] * grid_info['dt_ms'],
        dx=grid_info['dx'],
        dy=grid_info['dy'],
        dt_ms=grid_info['dt_ms'],
    )

    output_config = OutputConfig(
        output_dir=migration_dir,
        grid=output_grid,
        format=OutputFormat.ZARR,
    )

    resource_config = ResourceConfig(
        backend=ComputeBackend.METAL_COMPILED,
        max_memory_gb=16.0,
    )

    tiling_config = TilingConfig(
        auto_tile_size=False,
        tile_nx=64,
        tile_ny=64,
        ordering='snake',
    )

    checkpoint_config = CheckpointConfig(enabled=False)

    execution_config = ExecutionConfig(
        resources=resource_config,
        tiling=tiling_config,
        checkpoint=checkpoint_config,
    )

    migration_config = MigrationConfig(
        name="Impulse_Response",
        input=input_config,
        velocity=velocity_config,
        algorithm=algorithm_config,
        output=output_config,
        execution=execution_config,
    )

    import time
    start_time = time.time()
    try:
        run_migration(migration_config, resume=False)
        elapsed = time.time() - start_time
        print(f"Migration completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"Migration FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_impulse_response_images(output_dir: Path, grid_info: dict):
    """Create comprehensive QC images of the impulse response."""
    print(f"\n{'='*60}")
    print("Creating Impulse Response Images")
    print(f"{'='*60}")

    migration_dir = output_dir / "migration_output"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load migrated data
    z = zarr.open_array(str(migration_dir / "migrated_stack.zarr"), mode='r')
    image = np.array(z[:])
    nx, ny, nt = image.shape

    # Load fold
    fold_path = migration_dir / "fold.zarr"
    if fold_path.exists():
        fold = np.array(zarr.open_array(str(fold_path), mode='r'))
    else:
        fold = np.ones_like(image)

    dt_ms = grid_info['dt_ms']
    dx = grid_info['dx']
    dy = grid_info['dy']
    t_axis = np.arange(nt) * dt_ms

    center_il = nx // 2
    center_xl = ny // 2
    impulse_times = grid_info['impulse_times']
    velocity = grid_info['velocity']

    print(f"  Image shape: {image.shape}")
    print(f"  Amplitude range: {image.min():.4f} to {image.max():.4f}")

    # Compute clip values
    vmax = np.abs(image).max()
    vmax_95 = np.percentile(np.abs(image), 95)

    # === Figure 1: Full crossline and inline through center ===
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'PSTM Impulse Response - V={velocity} m/s, Offset={OFFSET_M}m\n'
                 f'Impulses at t={impulse_times} ms, Center: IL={center_il+1}, XL={center_xl+1}',
                 fontsize=14, fontweight='bold')

    # Crossline through center (different clips)
    clips = [('100% max', vmax), ('95%', vmax_95), ('50% max', vmax*0.5)]
    for i, (name, vm) in enumerate(clips):
        ax = axes[0, i]
        section = image[:, center_xl, :].T
        ax.imshow(section, aspect='auto', cmap='gray',
                  extent=[1, nx, t_axis[-1], 0],
                  vmin=-vm, vmax=vm, interpolation='bilinear')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Crossline {center_xl+1}: {name}')
        ax.axvline(x=center_il+1, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
        for t0 in impulse_times:
            ax.axhline(y=t0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)

    # Inline through center
    for i, (name, vm) in enumerate(clips):
        ax = axes[1, i]
        section = image[center_il, :, :].T
        ax.imshow(section, aspect='auto', cmap='gray',
                  extent=[1, ny, t_axis[-1], 0],
                  vmin=-vm, vmax=vm, interpolation='bilinear')
        ax.set_xlabel('Crossline')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Inline {center_il+1}: {name}')
        ax.axvline(x=center_xl+1, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
        for t0 in impulse_times:
            ax.axhline(y=t0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    fig_path = images_dir / 'impulse_response_sections.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")

    # === Figure 2: Time slices at impulse times ===
    n_impulses = len(impulse_times)
    fig, axes = plt.subplots(2, n_impulses, figsize=(6*n_impulses, 12))
    fig.suptitle(f'Impulse Response Time Slices - V={velocity} m/s\n'
                 f'Shows migration operator footprint at each time',
                 fontsize=14, fontweight='bold')

    for i, t0 in enumerate(impulse_times):
        t_idx = int(t0 / dt_ms)
        t_idx = min(max(t_idx, 0), nt-1)

        slice_data = image[:, :, t_idx]
        local_vmax = np.abs(slice_data).max()

        # Full clip
        ax = axes[0, i]
        ax.imshow(slice_data.T, aspect='equal', cmap='gray',
                  extent=[1, nx, ny, 1],
                  vmin=-local_vmax, vmax=local_vmax,
                  interpolation='bilinear', origin='lower')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        ax.set_title(f't = {t0} ms (100% clip)')
        ax.plot(center_il+1, center_xl+1, 'r+', markersize=15, markeredgewidth=2)

        # Tight clip to show structure
        ax = axes[1, i]
        ax.imshow(slice_data.T, aspect='equal', cmap='gray',
                  extent=[1, nx, ny, 1],
                  vmin=-local_vmax*0.3, vmax=local_vmax*0.3,
                  interpolation='bilinear', origin='lower')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        ax.set_title(f't = {t0} ms (30% clip)')
        ax.plot(center_il+1, center_xl+1, 'r+', markersize=15, markeredgewidth=2)

        # Draw theoretical aperture circle
        # Aperture = v * t * tan(max_dip) for Kirchhoff
        aperture_m = min(velocity * (t0/1000) * np.tan(np.radians(MAX_DIP_DEGREES)), MAX_APERTURE_M)
        aperture_il = aperture_m / dx
        aperture_xl = aperture_m / dy
        theta = np.linspace(0, 2*np.pi, 100)
        circle_il = center_il + 1 + aperture_il * np.cos(theta)
        circle_xl = center_xl + 1 + aperture_xl * np.sin(theta)
        ax.plot(circle_il, circle_xl, 'r--', linewidth=1, alpha=0.7, label=f'Aperture={aperture_m:.0f}m')
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    fig_path = images_dir / 'impulse_response_timeslices.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")

    # === Figure 3: Zoomed views around each impulse ===
    fig, axes = plt.subplots(2, n_impulses, figsize=(6*n_impulses, 12))
    fig.suptitle(f'Impulse Response - Zoomed Crossline Views\n'
                 f'V={velocity} m/s, showing operator shape at each time',
                 fontsize=14, fontweight='bold')

    zoom_half_t = 100  # ms window half-width
    zoom_half_x = 50   # trace window half-width

    for i, t0 in enumerate(impulse_times):
        t_start = max(0, int((t0 - zoom_half_t) / dt_ms))
        t_end = min(nt, int((t0 + zoom_half_t) / dt_ms))
        x_start = max(0, center_il - zoom_half_x)
        x_end = min(nx, center_il + zoom_half_x)

        section = image[x_start:x_end, center_xl, t_start:t_end].T
        local_vmax = np.abs(section).max()

        # Full clip
        ax = axes[0, i]
        extent = [x_start+1, x_end, (t_end)*dt_ms, (t_start)*dt_ms]
        ax.imshow(section, aspect='auto', cmap='gray',
                  extent=extent,
                  vmin=-local_vmax, vmax=local_vmax,
                  interpolation='bilinear')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f't0 = {t0} ms (100% clip)')
        ax.axhline(y=t0, color='r', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=center_il+1, color='r', linestyle='--', linewidth=1, alpha=0.7)

        # Tight clip
        ax = axes[1, i]
        ax.imshow(section, aspect='auto', cmap='gray',
                  extent=extent,
                  vmin=-local_vmax*0.3, vmax=local_vmax*0.3,
                  interpolation='bilinear')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f't0 = {t0} ms (30% clip)')
        ax.axhline(y=t0, color='r', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=center_il+1, color='r', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    fig_path = images_dir / 'impulse_response_zoomed.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")

    # === Figure 4: Amplitude profiles ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Impulse Response Amplitude Analysis\n'
                 f'V={velocity} m/s, Aperture: {MIN_APERTURE_M}-{MAX_APERTURE_M}m, Max Dip: {MAX_DIP_DEGREES}°',
                 fontsize=14, fontweight='bold')

    # Amplitude vs inline at each impulse time
    ax = axes[0, 0]
    for t0 in impulse_times:
        t_idx = int(t0 / dt_ms)
        profile = image[:, center_xl, t_idx]
        ax.plot(np.arange(nx)+1, profile, label=f't={t0}ms')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Amplitude')
    ax.set_title('Amplitude Profile (Crossline center)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=center_il+1, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Amplitude vs crossline
    ax = axes[0, 1]
    for t0 in impulse_times:
        t_idx = int(t0 / dt_ms)
        profile = image[center_il, :, t_idx]
        ax.plot(np.arange(ny)+1, profile, label=f't={t0}ms')
    ax.set_xlabel('Crossline')
    ax.set_ylabel('Amplitude')
    ax.set_title('Amplitude Profile (Inline center)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=center_xl+1, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Amplitude vs time at center
    ax = axes[1, 0]
    trace_center = image[center_il, center_xl, :]
    ax.plot(t_axis, trace_center, 'b-', linewidth=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Center Trace (IL/XL center)')
    for t0 in impulse_times:
        ax.axvline(x=t0, color='r', linestyle='--', linewidth=1, alpha=0.7, label=f't0={t0}ms')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fold profile
    ax = axes[1, 1]
    for t0 in impulse_times:
        t_idx = int(t0 / dt_ms)
        fold_profile = fold[:, center_xl, t_idx]
        ax.plot(np.arange(nx)+1, fold_profile, label=f't={t0}ms')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Fold')
    ax.set_title('Fold Profile (Crossline center)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = images_dir / 'impulse_response_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")

    # === Figure 5: 3D operator visualization (depth slices) ===
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'3D Impulse Response - Multiple Depth Slices\n'
                 f'V={velocity} m/s, showing operator evolution with time',
                 fontsize=14, fontweight='bold')

    # Create slices at multiple times around each impulse
    n_rows = 3
    n_cols = 5
    t_samples = []
    for t0 in impulse_times[:2]:  # First two impulses
        for dt in [-40, -20, 0, 20, 40]:
            t_samples.append(t0 + dt)
    # Add some from third impulse
    t_samples.extend([impulse_times[2] - 20, impulse_times[2], impulse_times[2] + 20])

    for idx, t_ms in enumerate(t_samples[:n_rows*n_cols]):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        t_idx = int(t_ms / dt_ms)
        t_idx = min(max(t_idx, 0), nt-1)

        slice_data = image[:, :, t_idx]
        local_vmax = np.abs(slice_data).max()
        if local_vmax == 0:
            local_vmax = 1

        ax.imshow(slice_data.T, aspect='equal', cmap='gray',
                  extent=[1, nx, ny, 1],
                  vmin=-local_vmax*0.5, vmax=local_vmax*0.5,
                  interpolation='bilinear', origin='lower')
        ax.set_xlabel('IL' if idx >= (n_rows-1)*n_cols else '')
        ax.set_ylabel('XL' if idx % n_cols == 0 else '')

        # Highlight if this is an impulse time
        title_color = 'red' if t_ms in impulse_times else 'black'
        ax.set_title(f't={t_ms:.0f}ms', fontsize=10, color=title_color)
        ax.plot(center_il+1, center_xl+1, 'r+', markersize=8, markeredgewidth=1)

    plt.tight_layout()
    fig_path = images_dir / 'impulse_response_3d_slices.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")

    print(f"\nAll images saved to: {images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create PSTM impulse response")
    parser.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY,
                        help=f"Constant velocity (m/s), default: {DEFAULT_VELOCITY}")
    parser.add_argument("--gradient", type=float, default=0.0,
                        help="Velocity gradient (m/s per ms), default: 0")
    parser.add_argument("--impulse-times", type=str, default=None,
                        help=f"Comma-separated impulse times (ms), default: {DEFAULT_IMPULSE_TIMES}")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_BASE,
                        help=f"Output directory, default: {OUTPUT_BASE}")
    parser.add_argument("--skip-migration", action="store_true",
                        help="Skip migration, only create images from existing output")

    args = parser.parse_args()

    # Parse impulse times
    if args.impulse_times:
        impulse_times = [float(t.strip()) for t in args.impulse_times.split(',')]
    else:
        impulse_times = DEFAULT_IMPULSE_TIMES

    print("=" * 70)
    print("PSTM IMPULSE RESPONSE GENERATOR")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Grid: {NX} x {NY} x {NT}")
    print(f"  Spacing: dx={DX}m, dy={DY}m, dt={DT_MS}ms")
    print(f"  Offset: {OFFSET_M}m")
    print(f"  Velocity: {args.velocity} m/s (gradient: {args.gradient})")
    print(f"  Impulse times: {impulse_times} ms")
    print(f"  Migration aperture: {MIN_APERTURE_M}-{MAX_APERTURE_M}m")
    print(f"  Max dip: {MAX_DIP_DEGREES} degrees")

    # Create output directory with velocity suffix
    output_dir = args.output_dir / f"v{int(args.velocity)}"

    if not args.skip_migration:
        # Create impulse data
        traces, headers, grid_info = create_impulse_data(impulse_times, args.velocity)

        # Create velocity model
        velocity = create_velocity_model(args.velocity, args.gradient)

        # Save data
        save_impulse_data(traces, headers, velocity, grid_info, output_dir)

        # Run migration
        success = run_impulse_migration(output_dir, grid_info)

        if not success:
            print("\nMigration failed!")
            return 1
    else:
        # Load existing grid info
        with open(output_dir / "grid_info.json", 'r') as f:
            grid_info = json.load(f)

    # Create images
    create_impulse_response_images(output_dir, grid_info)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
