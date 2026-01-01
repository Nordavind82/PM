#!/usr/bin/env python3
"""
Create Synthetic Common Offset Data for PSTM Testing

Generates two common offset gathers with different azimuths and a 3D velocity
model, saved in the format compatible with run_pstm_all_offsets.py.

Output:
- Velocity model: velocity_synthetic.zarr
- Offset 1 (Az 0-90): synthetic_offset_01/traces.zarr, headers.parquet
- Offset 2 (Az 90-180): synthetic_offset_02/traces.zarr, headers.parquet
- Combined: synthetic_offset_joint/traces.zarr, headers.parquet
"""

import numpy as np
import polars as pl
import zarr
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_BASE = Path("/Users/olegadamovich/SeismicData/synthetic_azimuth_test")


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    # Grid parameters - matching real data scale but smaller
    nx: int = 101
    ny: int = 85
    nt: int = 501
    dx: float = 25.0  # m (matching DX from run_pstm_all_offsets)
    dy: float = 12.5  # m (matching DY from run_pstm_all_offsets)
    dt_ms: float = 2.0  # ms

    # Grid origin (matching approximate real coordinates)
    x0: float = 618813.59
    y0: float = 5116498.50

    # Grid rotation angle (to match real data)
    rotation_deg: float = -52.0  # Approximate rotation of real grid

    # Velocity model parameters
    v0: float = 2000.0  # m/s at surface
    v_gradient: float = 0.4  # m/s per ms (vertical gradient)
    v_lateral_x: float = 80.0  # m/s variation in x direction
    v_lateral_y: float = 50.0  # m/s variation in y direction

    # Acquisition parameters
    offset: float = 400.0  # m (same offset for both gathers)
    traces_per_cmp: int = 8  # traces per CMP location

    # Wavelet
    f_peak: float = 30.0  # Hz

    @property
    def dt_s(self) -> float:
        return self.dt_ms / 1000.0


def rotate_point(x: float, y: float, angle_deg: float, cx: float, cy: float) -> Tuple[float, float]:
    """Rotate point (x,y) around center (cx,cy) by angle_deg degrees."""
    angle_rad = np.radians(angle_deg)
    dx = x - cx
    dy = y - cy
    x_rot = cx + dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    y_rot = cy + dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    return x_rot, y_rot


def create_ricker_wavelet(f_peak: float, dt_s: float, duration_s: float = 0.1) -> np.ndarray:
    """Create a Ricker wavelet."""
    t = np.arange(-duration_s/2, duration_s/2, dt_s)
    wavelet = (1 - 2*(np.pi*f_peak*t)**2) * np.exp(-(np.pi*f_peak*t)**2)
    return wavelet.astype(np.float32)


def dsr_traveltime(ox: float, oy: float, sx: float, sy: float,
                   rx: float, ry: float, t0: float, v: float) -> float:
    """Compute DSR traveltime."""
    ds2 = (ox - sx)**2 + (oy - sy)**2
    dr2 = (ox - rx)**2 + (oy - ry)**2
    t0_half_sq = (t0 / 2)**2
    inv_v_sq = 1 / (v * v)
    return np.sqrt(t0_half_sq + ds2 * inv_v_sq) + np.sqrt(t0_half_sq + dr2 * inv_v_sq)


def create_3d_velocity_model(config: SyntheticConfig, output_dir: Path) -> Path:
    """
    Create and save 3D velocity model in zarr format compatible with the pipeline.
    """
    print("\n[1] Creating 3D velocity model...")

    # Create velocity array
    velocity = np.zeros((config.nx, config.ny, config.nt), dtype=np.float32)

    # Time axis
    t_axis_ms = np.arange(config.nt) * config.dt_ms

    # Build velocity with gradient and lateral variations
    for ix in range(config.nx):
        for iy in range(config.ny):
            # Base velocity with vertical gradient
            v_profile = config.v0 + config.v_gradient * t_axis_ms

            # Lateral variation (sinusoidal patterns)
            x_var = config.v_lateral_x * np.sin(2 * np.pi * ix / (config.nx / 2))
            y_var = config.v_lateral_y * np.cos(2 * np.pi * iy / (config.ny / 2))

            velocity[ix, iy, :] = v_profile + x_var + y_var

    print(f"    Shape: {velocity.shape}")
    print(f"    V range: {velocity.min():.0f} - {velocity.max():.0f} m/s")
    print(f"    V at center surface: {velocity[config.nx//2, config.ny//2, 0]:.0f} m/s")
    print(f"    V at center deep: {velocity[config.nx//2, config.ny//2, -1]:.0f} m/s")

    # Save as zarr array with metadata for the pipeline
    velocity_path = output_dir / "velocity_synthetic.zarr"

    import shutil
    if velocity_path.exists():
        shutil.rmtree(velocity_path)

    # Create coordinate arrays for CubeVelocityModel
    x_coords = np.arange(config.nx) * config.dx
    y_coords = np.arange(config.ny) * config.dy
    t_coords = np.arange(config.nt) * config.dt_ms

    # Save velocity directly as array (not in a group) with axes as attributes
    velocity = velocity.astype(np.float32)
    z = zarr.open_array(str(velocity_path), mode='w',
                        shape=velocity.shape, dtype=velocity.dtype,
                        chunks=(32, 32, config.nt))
    z[:] = velocity

    # Set axes as attributes (CubeVelocityModel looks for these)
    z.attrs['x_axis'] = x_coords.tolist()
    z.attrs['y_axis'] = y_coords.tolist()
    z.attrs['t_axis_ms'] = t_coords.tolist()

    # Additional metadata
    z.attrs['nx'] = config.nx
    z.attrs['ny'] = config.ny
    z.attrs['nt'] = config.nt
    z.attrs['dx'] = config.dx
    z.attrs['dy'] = config.dy
    z.attrs['dt_ms'] = config.dt_ms
    z.attrs['x0'] = config.x0
    z.attrs['y0'] = config.y0
    z.attrs['rotation_deg'] = config.rotation_deg

    print(f"    Saved: {velocity_path}")

    return velocity_path


def get_world_coords(ix: int, iy: int, config: SyntheticConfig) -> Tuple[float, float]:
    """Convert grid indices to world coordinates."""
    # Local coordinates
    local_x = ix * config.dx
    local_y = iy * config.dy

    # Rotate and translate to world coordinates
    world_x, world_y = rotate_point(local_x, local_y, config.rotation_deg, 0, 0)
    world_x += config.x0
    world_y += config.y0

    return world_x, world_y


def generate_common_offset_gather(
    config: SyntheticConfig,
    velocity: np.ndarray,
    reflector_params: List[Tuple[float, float, float, float]],
    azimuth_range: Tuple[float, float],
    output_dir: Path,
    name: str
) -> Tuple[Path, Path]:
    """
    Generate a common offset gather and save in pipeline-compatible format.

    Args:
        reflector_params: List of (t0_ms, dip_x, dip_y, amplitude)

    Returns:
        (traces_path, headers_path)
    """
    print(f"\n[2] Generating {name}...")
    print(f"    Azimuth range: {azimuth_range[0]:.0f} - {azimuth_range[1]:.0f} degrees")

    wavelet = create_ricker_wavelet(config.f_peak, config.dt_s)
    wavelet_half = len(wavelet) // 2

    # Generate azimuths within range
    n_azimuths = config.traces_per_cmp
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], n_azimuths, endpoint=False)

    # Total traces
    n_traces = config.nx * config.ny * n_azimuths

    # Preallocate
    traces = np.zeros((n_traces, config.nt), dtype=np.float32)

    # Headers matching the pipeline expectations
    headers = {
        'source_x': np.zeros(n_traces, dtype=np.float64),
        'source_y': np.zeros(n_traces, dtype=np.float64),
        'receiver_x': np.zeros(n_traces, dtype=np.float64),
        'receiver_y': np.zeros(n_traces, dtype=np.float64),
        'offset': np.zeros(n_traces, dtype=np.float64),
        'sr_azim': np.zeros(n_traces, dtype=np.float64),
        'bin_trace_idx': np.zeros(n_traces, dtype=np.int32),
        'scalar_coord': np.ones(n_traces, dtype=np.float64),  # No scaling needed
    }

    trace_idx = 0
    for ix in range(config.nx):
        for iy in range(config.ny):
            # CMP location in world coordinates
            mx, my = get_world_coords(ix, iy, config)

            # Get velocity at this grid cell
            v_local = velocity[ix, iy, config.nt // 2]

            for az_idx, azimuth in enumerate(azimuths):
                # Compute source/receiver from midpoint, offset, azimuth
                half_offset = config.offset / 2
                dx = half_offset * np.sin(np.radians(azimuth))
                dy = half_offset * np.cos(np.radians(azimuth))

                sx, sy = mx - dx, my - dy
                rx, ry = mx + dx, my + dy

                # Create trace with reflections
                trace = np.zeros(config.nt, dtype=np.float32)

                for t0_ms, dip_x, dip_y, amp in reflector_params:
                    # Get reflector time at this location (dip in grid units)
                    dx_grid = ix - config.nx // 2
                    dy_grid = iy - config.ny // 2
                    t0_ref = t0_ms + dip_x * dx_grid + dip_y * dy_grid
                    t0 = t0_ref / 1000.0

                    # Skip if outside time range
                    if t0_ref < 50 or t0_ref > (config.nt - 50) * config.dt_ms:
                        continue

                    # Compute traveltime using DSR equation
                    t_travel = dsr_traveltime(mx, my, sx, sy, rx, ry, t0, v_local)
                    sample_idx = int(t_travel / config.dt_s)

                    # Add wavelet
                    if wavelet_half <= sample_idx < config.nt - wavelet_half:
                        start = sample_idx - wavelet_half
                        end = start + len(wavelet)
                        if end <= config.nt:
                            trace[start:end] += wavelet * amp

                traces[trace_idx] = trace

                headers['source_x'][trace_idx] = sx
                headers['source_y'][trace_idx] = sy
                headers['receiver_x'][trace_idx] = rx
                headers['receiver_y'][trace_idx] = ry
                headers['offset'][trace_idx] = config.offset
                headers['sr_azim'][trace_idx] = azimuth
                headers['bin_trace_idx'][trace_idx] = trace_idx

                trace_idx += 1

    print(f"    Traces: {n_traces}")
    print(f"    Non-zero traces: {np.sum(np.any(traces != 0, axis=1))}")

    # Save traces as zarr array directly (transposed format for pipeline compatibility)
    traces_path = output_dir / "traces.zarr"

    # Pipeline expects (nt, n_traces) with transposed=True
    traces_t = traces.T.astype(np.float32)  # Shape: (nt, n_traces)

    import shutil
    if traces_path.exists():
        shutil.rmtree(traces_path)

    # Save directly as array (not in a group)
    z = zarr.open_array(str(traces_path), mode='w',
                        shape=traces_t.shape, dtype=traces_t.dtype,
                        chunks=(config.nt, min(10000, n_traces)))
    z[:] = traces_t

    # Save headers as parquet
    headers_path = output_dir / "headers.parquet"
    df = pl.DataFrame(headers)
    df.write_parquet(headers_path)

    print(f"    Saved traces: {traces_path}")
    print(f"    Saved headers: {headers_path}")

    return traces_path, headers_path


def create_grid_config(config: SyntheticConfig, output_dir: Path):
    """Create grid configuration file for visualization."""
    grid_config = {
        'nx': config.nx,
        'ny': config.ny,
        'nt': config.nt,
        'dx': config.dx,
        'dy': config.dy,
        'dt_ms': config.dt_ms,
        'x0': config.x0,
        'y0': config.y0,
        'rotation_deg': config.rotation_deg,
        'corners': {
            'c1': (config.x0, config.y0),
            'c2': get_world_coords(config.nx - 1, 0, config),
            'c3': get_world_coords(config.nx - 1, config.ny - 1, config),
            'c4': get_world_coords(0, config.ny - 1, config),
        }
    }

    config_path = output_dir / "grid_config.json"
    with open(config_path, 'w') as f:
        # Convert tuples to lists for JSON
        grid_config['corners'] = {k: list(v) for k, v in grid_config['corners'].items()}
        json.dump(grid_config, f, indent=2)

    print(f"    Saved grid config: {config_path}")
    return grid_config


def main():
    """Create all synthetic data."""
    print("=" * 70)
    print("CREATE SYNTHETIC AZIMUTH TEST DATA")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_BASE}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = SyntheticConfig()

    print(f"\nConfiguration:")
    print(f"  Grid: {config.nx} x {config.ny} x {config.nt}")
    print(f"  Cell size: {config.dx} x {config.dy} m")
    print(f"  Time: 0 - {config.nt * config.dt_ms:.0f} ms @ {config.dt_ms} ms")
    print(f"  Offset: {config.offset} m")
    print(f"  Traces per CMP: {config.traces_per_cmp}")

    # Create velocity model
    velocity_path = create_3d_velocity_model(config, OUTPUT_BASE)

    # Load velocity for trace generation
    velocity = np.array(zarr.open_array(str(velocity_path), mode='r'))

    # Create grid config
    grid_config = create_grid_config(config, OUTPUT_BASE)

    # Define reflectors: (t0_ms, dip_x, dip_y, amplitude)
    reflector_params = [
        (200, 0.0, 0.0, 1.0),     # Flat at 200ms
        (400, 0.0, 0.0, 0.8),     # Flat at 400ms
        (600, 0.0, 0.0, 0.6),     # Flat at 600ms
        (500, 0.6, 0.4, 0.7),     # Dipping reflector
    ]

    print(f"\nReflectors:")
    for t0, dx, dy, amp in reflector_params:
        dip_type = "flat" if dx == 0 and dy == 0 else f"dipping (dx={dx}, dy={dy})"
        print(f"  t0={t0}ms, {dip_type}, amp={amp}")

    # Generate offset gather 1 (azimuths 0-90)
    offset1_dir = OUTPUT_BASE / "synthetic_offset_01"
    offset1_dir.mkdir(parents=True, exist_ok=True)
    generate_common_offset_gather(
        config, velocity, reflector_params,
        azimuth_range=(0, 90),
        output_dir=offset1_dir,
        name="Offset Gather 1 (Azimuth 0-90)"
    )

    # Generate offset gather 2 (azimuths 90-180)
    offset2_dir = OUTPUT_BASE / "synthetic_offset_02"
    offset2_dir.mkdir(parents=True, exist_ok=True)
    generate_common_offset_gather(
        config, velocity, reflector_params,
        azimuth_range=(90, 180),
        output_dir=offset2_dir,
        name="Offset Gather 2 (Azimuth 90-180)"
    )

    # Generate combined offset gather (all azimuths)
    print("\n[3] Creating combined offset gather...")
    joint_dir = OUTPUT_BASE / "synthetic_offset_joint"
    joint_dir.mkdir(parents=True, exist_ok=True)

    # Load and combine traces
    traces1 = zarr.open_array(str(offset1_dir / "traces.zarr"), mode='r')[:]
    traces2 = zarr.open_array(str(offset2_dir / "traces.zarr"), mode='r')[:]
    traces_joint = np.concatenate([traces1, traces2], axis=1)  # Concatenate along trace axis

    # Save combined traces as direct array
    import shutil
    joint_traces_path = joint_dir / "traces.zarr"
    if joint_traces_path.exists():
        shutil.rmtree(joint_traces_path)

    joint_traces = traces_joint.astype(np.float32)
    z = zarr.open_array(str(joint_traces_path), mode='w',
                        shape=joint_traces.shape, dtype=joint_traces.dtype,
                        chunks=(config.nt, min(10000, joint_traces.shape[1])))
    z[:] = joint_traces

    # Combine headers
    df1 = pl.read_parquet(offset1_dir / "headers.parquet")
    df2 = pl.read_parquet(offset2_dir / "headers.parquet")
    # Update bin_trace_idx for second gather
    df2 = df2.with_columns(
        (pl.col('bin_trace_idx') + len(df1)).alias('bin_trace_idx')
    )
    df_joint = pl.concat([df1, df2])
    df_joint.write_parquet(joint_dir / "headers.parquet")

    print(f"    Combined traces: {traces_joint.shape[1]}")
    print(f"    Saved: {joint_dir}")

    # Summary
    print("\n" + "=" * 70)
    print("SYNTHETIC DATA CREATED")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_BASE}")
    print(f"\nCreated:")
    print(f"  - Velocity model: velocity_synthetic.zarr")
    print(f"  - Offset 1 (Az 0-90): synthetic_offset_01/")
    print(f"  - Offset 2 (Az 90-180): synthetic_offset_02/")
    print(f"  - Combined (Az 0-180): synthetic_offset_joint/")
    print(f"\nNext steps:")
    print(f"  1. Run migration on each offset gather:")
    print(f"     python run_synthetic_migration.py")

    return True


if __name__ == "__main__":
    main()
