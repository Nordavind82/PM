#!/usr/bin/env python3
"""
Create Clean 3D Impulse Response for PSTM Migration.

Creates a clean impulse response visualization showing the migration
operator ellipsoid shape without wavelet ringing.

Uses a single spike (delta function) at each location to show
the true migration operator shape.
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
    MigrationConfig, InputConfig, VelocityConfig, AlgorithmConfig,
    OutputConfig, ExecutionConfig, OutputGridConfig, ApertureConfig,
    AmplitudeConfig, ResourceConfig, TilingConfig, CheckpointConfig,
    VelocitySource, InterpolationMethod, ComputeBackend, OutputFormat,
    TimeVariantConfig, AntiAliasingConfig, AntiAliasingMethod, ColumnMapping,
)
from pstm.pipeline.executor import run_migration

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_BASE = Path("/Users/olegadamovich/SeismicData/impulse_response_clean")

# Grid parameters - smaller for cleaner visualization
NX = 201
NY = 201
NT = 501
DX = 25.0
DY = 25.0
DT_MS = 2.0
T_MAX_MS = 1000.0

# Common offset
OFFSET_M = 300.0

# Grid origin
X_ORIGIN = 620000.0
Y_ORIGIN = 5115000.0

# Default impulse time
DEFAULT_IMPULSE_TIME = 500.0  # Single impulse for cleaner response

# Default velocity
DEFAULT_VELOCITY = 2000.0

# Migration parameters
MAX_APERTURE_M = 2000.0
MIN_APERTURE_M = 200.0  # Smaller min aperture for better near-offset response
MAX_DIP_DEGREES = 65.0


def create_spike_data(impulse_time: float, velocity: float) -> tuple[np.ndarray, list[dict], dict]:
    """
    Create synthetic data with a SINGLE SPIKE at the center.

    Uses a true delta function (single sample = 1.0) to show the
    migration operator shape without wavelet effects.
    """
    print(f"\nCreating spike impulse data...")
    print(f"  Impulse time: {impulse_time} ms")
    print(f"  Velocity: {velocity} m/s")

    # Grid coordinates
    X_grid = X_ORIGIN + np.arange(NX)[:, None] * DX
    Y_grid = Y_ORIGIN + np.arange(NY)[None, :] * DY

    # Center of grid
    center_il = NX // 2
    center_xl = NY // 2
    center_x = X_grid[center_il, 0]
    center_y = Y_grid[0, center_xl]

    print(f"  Grid center: IL={center_il+1}, XL={center_xl+1}")
    print(f"  UTM center: X={center_x:.1f}, Y={center_y:.1f}")

    # Offset geometry
    half_offset = OFFSET_M / 2.0

    traces_list = []
    headers_list = []
    trace_idx = 0

    for ix in range(NX):
        for iy in range(NY):
            cdp_x = X_grid[ix, 0]
            cdp_y = Y_grid[0, iy]

            # Source and receiver (offset in X direction)
            sx = cdp_x - half_offset
            sy = cdp_y
            rx = cdp_x + half_offset
            ry = cdp_y

            # Create trace with single spike
            trace = np.zeros(NT, dtype=np.float32)

            # Distance from CDP to impulse location (center)
            ds = np.sqrt((sx - center_x)**2 + (sy - center_y)**2)
            dr = np.sqrt((rx - center_x)**2 + (ry - center_y)**2)

            # DSR traveltime
            t0_s = impulse_time / 1000.0
            t0_half = t0_s / 2.0

            t_travel_s = np.sqrt(t0_half**2 + (ds/velocity)**2) + \
                         np.sqrt(t0_half**2 + (dr/velocity)**2)
            t_travel_ms = t_travel_s * 1000.0

            # Place SINGLE SPIKE at traveltime (no wavelet)
            if 0 < t_travel_ms < T_MAX_MS - 10:
                t_idx = int(round(t_travel_ms / DT_MS))
                if 0 <= t_idx < NT:
                    trace[t_idx] = 1.0  # Delta function

            traces_list.append(trace)

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
                'sr_azim': 90.0,
            })
            trace_idx += 1

        if (ix + 1) % 50 == 0:
            print(f"    Progress: {ix+1}/{NX} inlines")

    traces = np.array(traces_list, dtype=np.float32)

    grid_info = {
        'nx': NX, 'ny': NY, 'nt': NT,
        'dx': DX, 'dy': DY, 'dt_ms': DT_MS,
        'x_origin': X_ORIGIN, 'y_origin': Y_ORIGIN,
        'offset_m': OFFSET_M, 'velocity': velocity,
        'impulse_time': impulse_time,
        'center_il': center_il, 'center_xl': center_xl,
        'c1': (X_ORIGIN, Y_ORIGIN),
        'c2': (X_ORIGIN + (NX-1)*DX, Y_ORIGIN),
        'c3': (X_ORIGIN + (NX-1)*DX, Y_ORIGIN + (NY-1)*DY),
        'c4': (X_ORIGIN, Y_ORIGIN + (NY-1)*DY),
    }

    print(f"  Created {len(traces)} traces")
    print(f"  Non-zero samples: {np.count_nonzero(traces)}")

    return traces, headers_list, grid_info


def create_velocity_model(velocity: float) -> np.ndarray:
    """Create constant velocity model."""
    vel = np.full((NX, NY, NT), velocity, dtype=np.float32)
    return vel


def save_data(traces, headers, velocity, grid_info, output_dir):
    """Save data to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Velocity
    vel_path = output_dir / "velocity.zarr"
    z_vel = zarr.open_array(str(vel_path), mode='w', shape=velocity.shape,
                            dtype=velocity.dtype, chunks=(32, 32, NT))
    z_vel[:] = velocity
    z_vel.attrs['x_axis'] = list(range(1, NX + 1))
    z_vel.attrs['y_axis'] = list(range(1, NY + 1))
    z_vel.attrs['t_axis_ms'] = list(np.arange(NT) * DT_MS)
    z_vel.attrs['dt_ms'] = DT_MS

    # Traces
    traces_dir = output_dir / "impulse_data"
    traces_dir.mkdir(exist_ok=True)
    traces_t = traces.T
    z_traces = zarr.open_array(str(traces_dir / "traces.zarr"), mode='w',
                               shape=traces_t.shape, dtype=traces_t.dtype,
                               chunks=(NT, min(10000, traces_t.shape[1])))
    z_traces[:] = traces_t

    # Headers
    df = pl.DataFrame(headers)
    df.write_parquet(traces_dir / "headers.parquet")

    # Grid info
    with open(output_dir / "grid_info.json", 'w') as f:
        json.dump(grid_info, f, indent=2)


def run_impulse_migration(output_dir, grid_info):
    """Run PSTM migration."""
    print(f"\n{'='*60}")
    print("Running PSTM Migration")
    print(f"{'='*60}")

    input_dir = output_dir / "impulse_data"
    migration_dir = output_dir / "migration_output"
    migration_dir.mkdir(parents=True, exist_ok=True)

    input_config = InputConfig(
        traces_path=input_dir / "traces.zarr",
        headers_path=input_dir / "headers.parquet",
        columns=ColumnMapping(
            source_x="source_x", source_y="source_y",
            receiver_x="receiver_x", receiver_y="receiver_y",
            offset="offset", azimuth="sr_azim",
            trace_index="bin_trace_idx", coord_scalar="scalar_coord",
        ),
        apply_coord_scalar=True,
        sample_rate_ms=grid_info['dt_ms'],
        transposed=True,
    )

    velocity_config = VelocityConfig(
        source=VelocitySource.CUBE_3D,
        velocity_path=output_dir / "velocity.zarr",
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

    algorithm_config = AlgorithmConfig(
        interpolation=InterpolationMethod.LINEAR,
        aperture=aperture_config,
        amplitude=amplitude_config,
        time_variant=TimeVariantConfig(enabled=False),
        anti_aliasing=AntiAliasingConfig(enabled=False, method=AntiAliasingMethod.NONE),
    )

    output_grid = OutputGridConfig.from_corners(
        corner1=tuple(grid_info['c1']),
        corner2=tuple(grid_info['c2']),
        corner3=tuple(grid_info['c3']),
        corner4=tuple(grid_info['c4']),
        t_min_ms=0.0,
        t_max_ms=grid_info['nt'] * grid_info['dt_ms'],
        dx=grid_info['dx'], dy=grid_info['dy'], dt_ms=grid_info['dt_ms'],
    )

    output_config = OutputConfig(
        output_dir=migration_dir,
        grid=output_grid,
        format=OutputFormat.ZARR,
    )

    execution_config = ExecutionConfig(
        resources=ResourceConfig(backend=ComputeBackend.METAL_COMPILED, max_memory_gb=16.0),
        tiling=TilingConfig(auto_tile_size=False, tile_nx=64, tile_ny=64, ordering='snake'),
        checkpoint=CheckpointConfig(enabled=False),
    )

    migration_config = MigrationConfig(
        name="Clean_Impulse_Response",
        input=input_config,
        velocity=velocity_config,
        algorithm=algorithm_config,
        output=output_config,
        execution=execution_config,
    )

    import time
    start = time.time()
    try:
        run_migration(migration_config, resume=False)
        print(f"Migration completed in {time.time()-start:.1f}s")
        return True
    except Exception as e:
        print(f"Migration FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_clean_images(output_dir, grid_info):
    """Create clean impulse response images showing ellipsoid operator."""
    print(f"\n{'='*60}")
    print("Creating Clean Impulse Response Images")
    print(f"{'='*60}")

    migration_dir = output_dir / "migration_output"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    z = zarr.open_array(str(migration_dir / "migrated_stack.zarr"), mode='r')
    image = np.array(z[:])
    nx, ny, nt = image.shape

    dt_ms = grid_info['dt_ms']
    dx = grid_info['dx']
    dy = grid_info['dy']
    velocity = grid_info['velocity']
    impulse_time = grid_info['impulse_time']
    center_il = nx // 2
    center_xl = ny // 2

    t_axis = np.arange(nt) * dt_ms

    print(f"  Image shape: {image.shape}")
    print(f"  Amplitude range: {image.min():.4f} to {image.max():.4f}")

    # Take absolute value for cleaner display of operator shape
    image_abs = np.abs(image)
    vmax = image_abs.max()

    # === Figure 1: Classic impulse response view ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'PSTM Migration Impulse Response\n'
                 f'V={velocity} m/s, t0={impulse_time}ms, Offset={OFFSET_M}m',
                 fontsize=14, fontweight='bold')

    # Crossline section - amplitude
    ax = axes[0, 0]
    section = image[:, center_xl, :].T
    sec_vmax = np.abs(section).max()
    ax.imshow(section, aspect='auto', cmap='seismic',
              extent=[1, nx, t_axis[-1], 0],
              vmin=-sec_vmax, vmax=sec_vmax, interpolation='bilinear')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Crossline {center_xl+1} (amplitude)')
    ax.axvline(x=center_il+1, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(y=impulse_time, color='k', linestyle='--', linewidth=0.5)

    # Crossline section - envelope (absolute value)
    ax = axes[0, 1]
    section_abs = image_abs[:, center_xl, :].T
    ax.imshow(section_abs, aspect='auto', cmap='hot',
              extent=[1, nx, t_axis[-1], 0],
              vmin=0, vmax=section_abs.max(), interpolation='bilinear')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Crossline {center_xl+1} (envelope)')
    ax.axvline(x=center_il+1, color='w', linestyle='--', linewidth=0.5)
    ax.axhline(y=impulse_time, color='w', linestyle='--', linewidth=0.5)

    # Inline section - envelope
    ax = axes[0, 2]
    section_abs = image_abs[center_il, :, :].T
    ax.imshow(section_abs, aspect='auto', cmap='hot',
              extent=[1, ny, t_axis[-1], 0],
              vmin=0, vmax=section_abs.max(), interpolation='bilinear')
    ax.set_xlabel('Crossline')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Inline {center_il+1} (envelope)')
    ax.axvline(x=center_xl+1, color='w', linestyle='--', linewidth=0.5)
    ax.axhline(y=impulse_time, color='w', linestyle='--', linewidth=0.5)

    # Time slice at impulse time - amplitude
    t_idx = int(impulse_time / dt_ms)
    ax = axes[1, 0]
    slice_data = image[:, :, t_idx]
    slice_vmax = np.abs(slice_data).max()
    ax.imshow(slice_data.T, aspect='equal', cmap='seismic',
              extent=[1, nx, ny, 1],
              vmin=-slice_vmax, vmax=slice_vmax,
              interpolation='bilinear', origin='lower')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title(f'Time Slice @ {impulse_time}ms (amplitude)')
    ax.plot(center_il+1, center_xl+1, 'k+', markersize=15, markeredgewidth=2)

    # Time slice - envelope
    ax = axes[1, 1]
    slice_abs = image_abs[:, :, t_idx]
    ax.imshow(slice_abs.T, aspect='equal', cmap='hot',
              extent=[1, nx, ny, 1],
              vmin=0, vmax=slice_abs.max(),
              interpolation='bilinear', origin='lower')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title(f'Time Slice @ {impulse_time}ms (envelope)')
    ax.plot(center_il+1, center_xl+1, 'w+', markersize=15, markeredgewidth=2)

    # Add theoretical aperture ellipse
    aperture_m = min(velocity * (impulse_time/1000) * np.tan(np.radians(MAX_DIP_DEGREES)), MAX_APERTURE_M)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_il = center_il + 1 + (aperture_m/dx) * np.cos(theta)
    circle_xl = center_xl + 1 + (aperture_m/dy) * np.sin(theta)
    ax.plot(circle_il, circle_xl, 'w--', linewidth=1.5, label=f'Aperture={aperture_m:.0f}m')
    ax.legend(loc='upper right')

    # Amplitude profiles
    ax = axes[1, 2]
    # Profile along inline
    profile_il = image_abs[:, center_xl, t_idx]
    profile_xl = image_abs[center_il, :, t_idx]
    ax.plot(np.arange(nx)+1, profile_il / profile_il.max(), 'b-', label='Along Inline', linewidth=1.5)
    ax.plot(np.arange(ny)+1, profile_xl / profile_xl.max(), 'r-', label='Along Crossline', linewidth=1.5)
    ax.set_xlabel('Trace Number')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title('Amplitude Profile @ Impulse Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=center_il+1, color='k', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    fig_path = images_dir / 'impulse_response_clean.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")

    # === Figure 2: 3D operator visualization ===
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'Migration Operator Evolution - Time Slices\n'
                 f'V={velocity} m/s, Impulse @ t={impulse_time}ms',
                 fontsize=14, fontweight='bold')

    # Time slices around impulse time
    t_offsets = [-100, -50, -20, 0, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    for idx, dt in enumerate(t_offsets):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        t_ms = impulse_time + dt
        t_idx = int(t_ms / dt_ms)
        t_idx = min(max(t_idx, 0), nt-1)

        slice_data = image_abs[:, :, t_idx]
        local_vmax = slice_data.max()
        if local_vmax == 0:
            local_vmax = 1

        ax.imshow(slice_data.T, aspect='equal', cmap='hot',
                  extent=[1, nx, ny, 1],
                  vmin=0, vmax=local_vmax*0.8,
                  interpolation='bilinear', origin='lower')

        # Highlight impulse time
        title_color = 'red' if dt == 0 else 'black'
        ax.set_title(f't={t_ms:.0f}ms', fontsize=10, color=title_color, fontweight='bold' if dt==0 else 'normal')
        ax.plot(center_il+1, center_xl+1, 'w+', markersize=8, markeredgewidth=1)

        if row == 2:
            ax.set_xlabel('IL')
        if col == 0:
            ax.set_ylabel('XL')

    plt.tight_layout()
    fig_path = images_dir / 'impulse_response_evolution.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")

    # === Figure 3: Operator shape analysis ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Migration Operator Analysis\n'
                 f'V={velocity} m/s, Aperture: {MIN_APERTURE_M}-{MAX_APERTURE_M}m, Max Dip: {MAX_DIP_DEGREES}°',
                 fontsize=14, fontweight='bold')

    # Zoomed crossline view
    ax = axes[0, 0]
    zoom_il = 40  # traces around center
    zoom_t = 150  # ms around impulse
    il_start = max(0, center_il - zoom_il)
    il_end = min(nx, center_il + zoom_il)
    t_start = max(0, int((impulse_time - zoom_t) / dt_ms))
    t_end = min(nt, int((impulse_time + zoom_t) / dt_ms))

    section = image[il_start:il_end, center_xl, t_start:t_end].T
    sec_vmax = np.abs(section).max()
    extent = [il_start+1, il_end, t_end*dt_ms, t_start*dt_ms]
    ax.imshow(section, aspect='auto', cmap='seismic',
              extent=extent, vmin=-sec_vmax, vmax=sec_vmax, interpolation='bilinear')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Zoomed Crossline (amplitude)')
    ax.axhline(y=impulse_time, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=center_il+1, color='k', linestyle='--', linewidth=1)

    # Zoomed envelope
    ax = axes[0, 1]
    section_abs = np.abs(section)
    ax.imshow(section_abs, aspect='auto', cmap='hot',
              extent=extent, vmin=0, vmax=section_abs.max(), interpolation='bilinear')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Zoomed Crossline (envelope)')
    ax.axhline(y=impulse_time, color='w', linestyle='--', linewidth=1)
    ax.axvline(x=center_il+1, color='w', linestyle='--', linewidth=1)

    # Center trace
    ax = axes[1, 0]
    center_trace = image[center_il, center_xl, :]
    ax.plot(t_axis, center_trace, 'b-', linewidth=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Center Trace')
    ax.axvline(x=impulse_time, color='r', linestyle='--', linewidth=1, label=f't0={impulse_time}ms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(impulse_time - 200, impulse_time + 200)

    # Operator width vs time
    ax = axes[1, 1]
    widths_t = []
    times = []
    for t_ms in range(int(impulse_time - 100), int(impulse_time + 300), 10):
        t_idx = int(t_ms / dt_ms)
        if 0 <= t_idx < nt:
            profile = image_abs[:, center_xl, t_idx]
            # Find width at half maximum
            max_amp = profile.max()
            if max_amp > 0:
                half_max = max_amp / 2
                above_half = np.where(profile > half_max)[0]
                if len(above_half) > 0:
                    width_traces = above_half[-1] - above_half[0]
                    width_m = width_traces * dx
                    widths_t.append(width_m)
                    times.append(t_ms)

    ax.plot(times, widths_t, 'b-o', markersize=4)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Operator Width (m)')
    ax.set_title('Migration Operator Width vs Time')
    ax.axvline(x=impulse_time, color='r', linestyle='--', linewidth=1, label=f't0={impulse_time}ms')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = images_dir / 'impulse_response_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path.name}")

    print(f"\nAll images saved to: {images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create clean PSTM impulse response")
    parser.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY)
    parser.add_argument("--impulse-time", type=float, default=DEFAULT_IMPULSE_TIME)
    parser.add_argument("--skip-migration", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("CLEAN PSTM IMPULSE RESPONSE")
    print("=" * 70)
    print(f"Velocity: {args.velocity} m/s")
    print(f"Impulse time: {args.impulse_time} ms")

    output_dir = OUTPUT_BASE / f"v{int(args.velocity)}_t{int(args.impulse_time)}"

    if not args.skip_migration:
        traces, headers, grid_info = create_spike_data(args.impulse_time, args.velocity)
        velocity = create_velocity_model(args.velocity)
        save_data(traces, headers, velocity, grid_info, output_dir)

        if not run_impulse_migration(output_dir, grid_info):
            return 1
    else:
        with open(output_dir / "grid_info.json", 'r') as f:
            grid_info = json.load(f)

    create_clean_images(output_dir, grid_info)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
