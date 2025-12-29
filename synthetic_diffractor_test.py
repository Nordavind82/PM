#!/usr/bin/env python3
"""
Synthetic Diffractor Test for PSTM Migration Validation.

Creates synthetic common offset gathers with known diffractor positions,
then migrates them to verify the migration algorithm is working correctly.

The test uses:
- Two point diffractors at different depths
- Two offset volumes (50m and 1200m)
- Same grid parameters as the real survey
- Known constant velocity for verification

If migration is correct, diffractors should collapse to points at correct (x, y, t0).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import zarr


# =============================================================================
# Configuration - Match Real Survey Parameters
# =============================================================================

# Output directories
SYNTHETIC_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_synthetic_test")
SYNTHETIC_INPUT_DIR = SYNTHETIC_DIR / "input"
SYNTHETIC_OUTPUT_DIR = SYNTHETIC_DIR / "migration_output"

# Grid parameters (small grid for fast testing)
GRID_CONFIG = {
    # Spatial grid (small size for fast testing)
    'x_min': 623000.0,
    'x_max': 625000.0,  # 2km range
    'y_min': 5113000.0,
    'y_max': 5115000.0,  # 2km range
    'dx': 50.0,  # inline spacing
    'dy': 50.0,  # crossline spacing

    # Time grid
    't_min_ms': 0.0,
    't_max_ms': 2000.0,
    'dt_ms': 4.0,  # Coarser time sampling

    # Trace parameters
    'n_samples': 501,
    'sample_rate_ms': 4.0,
}

# Velocity model (constant for synthetic test)
VELOCITY_MS = 2500.0  # m/s - constant RMS velocity

# Diffractor definitions (x, y, t0_ms, amplitude)
DIFFRACTORS = [
    {
        'name': 'shallow',
        'x': 624000.0,
        'y': 5114000.0,
        't0_ms': 500.0,  # Zero-offset two-way time
        'amplitude': 1.0,
    },
    {
        'name': 'deep',
        'x': 624000.0,
        'y': 5114000.0,
        't0_ms': 1200.0,
        'amplitude': 0.8,
    },
]

# Offsets to generate
OFFSETS = [50.0, 1200.0]

# Wavelet parameters
WAVELET_FREQ_HZ = 30.0  # Dominant frequency


# =============================================================================
# Wavelet Generation
# =============================================================================

def ricker_wavelet(t: np.ndarray, f: float) -> np.ndarray:
    """
    Generate Ricker (Mexican hat) wavelet.

    Args:
        t: Time array (seconds), centered at 0
        f: Dominant frequency (Hz)

    Returns:
        Wavelet amplitude array
    """
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


# =============================================================================
# Travel Time Computation
# =============================================================================

def compute_dsr_traveltime(t0_s: float, ds: float, dr: float, velocity: float) -> float:
    """
    Compute Double Square Root (DSR) travel time.

    Args:
        t0_s: Zero-offset two-way time (seconds)
        ds: Distance from image point to source (meters)
        dr: Distance from image point to receiver (meters)
        velocity: RMS velocity (m/s)

    Returns:
        Total travel time (seconds)
    """
    t0_half = t0_s / 2.0
    t0_half_sq = t0_half ** 2
    inv_v_sq = 1.0 / (velocity ** 2)

    # Source leg
    t_src = np.sqrt(t0_half_sq + ds**2 * inv_v_sq)
    # Receiver leg
    t_rec = np.sqrt(t0_half_sq + dr**2 * inv_v_sq)

    return t_src + t_rec


def compute_nmo_traveltime(t0_s: float, offset: float, velocity: float) -> float:
    """
    Compute NMO travel time (for reference).

    Args:
        t0_s: Zero-offset two-way time (seconds)
        offset: Source-receiver offset (meters)
        velocity: NMO velocity (m/s)

    Returns:
        NMO travel time (seconds)
    """
    return np.sqrt(t0_s**2 + (offset / velocity)**2)


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_traces(
    offset_m: float,
    grid_config: dict,
    diffractors: list,
    velocity: float,
    wavelet_freq: float,
) -> tuple[np.ndarray, pl.DataFrame]:
    """
    Generate synthetic common offset traces for given diffractors.

    Args:
        offset_m: Source-receiver offset (meters)
        grid_config: Grid configuration dictionary
        diffractors: List of diffractor definitions
        velocity: Constant velocity (m/s)
        wavelet_freq: Wavelet dominant frequency (Hz)

    Returns:
        traces: (n_traces, n_samples) array
        headers: DataFrame with trace headers
    """
    # Compute grid
    x_coords = np.arange(grid_config['x_min'], grid_config['x_max'] + grid_config['dx'], grid_config['dx'])
    y_coords = np.arange(grid_config['y_min'], grid_config['y_max'] + grid_config['dy'], grid_config['dy'])

    nx = len(x_coords)
    ny = len(y_coords)
    n_traces = nx * ny
    n_samples = grid_config['n_samples']
    dt_ms = grid_config['sample_rate_ms']
    dt_s = dt_ms / 1000.0

    print(f"  Grid: {nx} x {ny} = {n_traces} traces")
    print(f"  Samples: {n_samples} @ {dt_ms} ms")

    # Time axis
    t_axis_s = np.arange(n_samples) * dt_s

    # Create wavelet (centered at t=0)
    wavelet_duration = 0.1  # seconds
    wavelet_samples = int(wavelet_duration / dt_s)
    wavelet_t = np.linspace(-wavelet_duration/2, wavelet_duration/2, wavelet_samples)
    wavelet = ricker_wavelet(wavelet_t, wavelet_freq)

    # Initialize traces
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)

    # Header data
    header_data = {
        'trace_index': [],
        'source_x': [],
        'source_y': [],
        'receiver_x': [],
        'receiver_y': [],
        'midpoint_x': [],
        'midpoint_y': [],
        'offset': [],
        'inline': [],
        'crossline': [],
    }

    # Generate traces
    trace_idx = 0
    half_offset = offset_m / 2.0

    # Assume inline direction is X, crossline is Y
    # Source and receiver are offset along X direction (inline)

    for ix, mx in enumerate(x_coords):
        for iy, my in enumerate(y_coords):
            # Midpoint location
            # Source and receiver positions (offset along x-direction)
            sx = mx - half_offset
            sy = my
            rx = mx + half_offset
            ry = my

            # Compute trace for each diffractor
            trace = np.zeros(n_samples, dtype=np.float32)

            for diff in diffractors:
                # Distance from diffractor to source
                ds = np.sqrt((diff['x'] - sx)**2 + (diff['y'] - sy)**2)
                # Distance from diffractor to receiver
                dr = np.sqrt((diff['x'] - rx)**2 + (diff['y'] - ry)**2)

                # Compute travel time using DSR
                t0_s = diff['t0_ms'] / 1000.0
                t_travel_s = compute_dsr_traveltime(t0_s, ds, dr, velocity)
                t_travel_ms = t_travel_s * 1000.0

                # Convert to sample index
                sample_idx = int(t_travel_ms / dt_ms)

                if 0 <= sample_idx < n_samples:
                    # Add wavelet centered at travel time
                    start_idx = sample_idx - wavelet_samples // 2
                    end_idx = start_idx + wavelet_samples

                    # Clip to valid range
                    wav_start = max(0, -start_idx)
                    wav_end = wavelet_samples - max(0, end_idx - n_samples)
                    trace_start = max(0, start_idx)
                    trace_end = min(n_samples, end_idx)

                    if trace_end > trace_start and wav_end > wav_start:
                        # Apply amplitude with geometric spreading approximation
                        dist = np.sqrt(ds**2 + dr**2)
                        spreading = 1.0 / max(dist, 100.0) * 1000.0  # Normalize
                        trace[trace_start:trace_end] += (
                            diff['amplitude'] * spreading *
                            wavelet[wav_start:wav_end]
                        )

            traces[trace_idx] = trace

            # Store headers
            header_data['trace_index'].append(trace_idx)
            header_data['source_x'].append(sx)
            header_data['source_y'].append(sy)
            header_data['receiver_x'].append(rx)
            header_data['receiver_y'].append(ry)
            header_data['midpoint_x'].append(mx)
            header_data['midpoint_y'].append(my)
            header_data['offset'].append(offset_m)
            header_data['inline'].append(ix + 1)
            header_data['crossline'].append(iy + 1)

            trace_idx += 1

        # Progress
        if (ix + 1) % 50 == 0:
            print(f"    Progress: {ix + 1}/{nx} inlines")

    headers = pl.DataFrame(header_data)

    return traces, headers


def save_synthetic_data(
    offset_m: float,
    traces: np.ndarray,
    headers: pl.DataFrame,
    output_dir: Path,
    grid_config: dict,
):
    """
    Save synthetic data in the same format as real common offset data.

    Args:
        offset_m: Offset value (for naming)
        traces: Trace data array
        headers: Headers DataFrame
        output_dir: Output directory
        grid_config: Grid configuration
    """
    # Create bin directory (use offset as bin "number")
    bin_name = f"offset_{int(offset_m):04d}m"
    bin_dir = output_dir / bin_name
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Save traces as zarr
    traces_path = bin_dir / "traces.zarr"
    if traces_path.exists():
        shutil.rmtree(traces_path)

    store = zarr.storage.LocalStore(str(traces_path))
    z = zarr.open_array(
        store=store,
        mode='w',
        shape=traces.shape,
        dtype=traces.dtype,
        chunks=(min(1000, traces.shape[0]), traces.shape[1]),
    )
    z[:] = traces
    z.attrs['sample_rate_ms'] = grid_config['sample_rate_ms']
    z.attrs['n_samples'] = grid_config['n_samples']
    z.attrs['n_traces'] = traces.shape[0]

    # Save headers as parquet
    headers_path = bin_dir / "headers.parquet"
    headers.write_parquet(headers_path)

    print(f"  Saved: {bin_dir}")
    print(f"    Traces: {traces.shape}")
    print(f"    Headers: {len(headers)} rows")


# =============================================================================
# Velocity Model Creation
# =============================================================================

def create_constant_velocity_model(
    grid_config: dict,
    velocity: float,
    output_path: Path,
):
    """
    Create a constant velocity model for migration.

    Args:
        grid_config: Grid configuration
        velocity: Constant velocity (m/s)
        output_path: Output zarr path
    """
    # Compute grid dimensions
    x_coords = np.arange(grid_config['x_min'], grid_config['x_max'] + grid_config['dx'], grid_config['dx'])
    y_coords = np.arange(grid_config['y_min'], grid_config['y_max'] + grid_config['dy'], grid_config['dy'])
    t_coords = np.arange(grid_config['t_min_ms'], grid_config['t_max_ms'] + grid_config['dt_ms'], grid_config['dt_ms'])

    nx = len(x_coords)
    ny = len(y_coords)
    nt = len(t_coords)

    print(f"  Velocity model: {nx} x {ny} x {nt}")
    print(f"  Constant velocity: {velocity} m/s")

    # Create constant velocity cube
    vel_cube = np.full((nx, ny, nt), velocity, dtype=np.float32)

    # Save as zarr
    if output_path.exists():
        shutil.rmtree(output_path)

    store = zarr.storage.LocalStore(str(output_path))
    z = zarr.open_array(
        store=store,
        mode='w',
        shape=vel_cube.shape,
        dtype=vel_cube.dtype,
        chunks=(nx, ny, min(100, nt)),
    )
    z[:] = vel_cube

    # Set attributes
    z.attrs['x_min'] = grid_config['x_min']
    z.attrs['x_max'] = grid_config['x_max']
    z.attrs['y_min'] = grid_config['y_min']
    z.attrs['y_max'] = grid_config['y_max']
    z.attrs['dx'] = grid_config['dx']
    z.attrs['dy'] = grid_config['dy']
    z.attrs['t_min_ms'] = grid_config['t_min_ms']
    z.attrs['t_max_ms'] = grid_config['t_max_ms']
    z.attrs['dt_ms'] = grid_config['dt_ms']
    z.attrs['t_axis_ms'] = t_coords.tolist()
    z.attrs['x_axis'] = x_coords.tolist()
    z.attrs['y_axis'] = y_coords.tolist()

    print(f"  Saved: {output_path}")


# =============================================================================
# Migration Runner
# =============================================================================

def create_migration_config(
    input_dir: Path,
    output_dir: Path,
    velocity_path: Path,
    offset_m: float,
    grid_config: dict,
) -> dict:
    """Create migration configuration for synthetic data."""

    bin_name = f"offset_{int(offset_m):04d}m"

    config = {
        "input_data": {
            "dataset_path": str(input_dir / bin_name),
            "traces_path": str(input_dir / bin_name / "traces.zarr"),
            "traces_format": "zarr",
            "headers_path": str(input_dir / bin_name / "headers.parquet"),
            "headers_format": "parquet",
            "col_source_x": "source_x",
            "col_source_y": "source_y",
            "col_receiver_x": "receiver_x",
            "col_receiver_y": "receiver_y",
            "col_midpoint_x": "midpoint_x",
            "col_midpoint_y": "midpoint_y",
            "col_offset": "offset",
            "col_trace_index": "trace_index",
            "n_samples": grid_config['n_samples'],
            "sample_rate_ms": grid_config['sample_rate_ms'],
            "is_loaded": True,
            "traces_transposed": False,
            "apply_coord_scalar": False,
        },
        "survey": {
            "x_min": grid_config['x_min'],
            "x_max": grid_config['x_max'],
            "y_min": grid_config['y_min'],
            "y_max": grid_config['y_max'],
        },
        "output_grid": {
            "dx": grid_config['dx'],
            "dy": grid_config['dy'],
            "dt_ms": grid_config['dt_ms'],
            "t_min_ms": grid_config['t_min_ms'],
            "t_max_ms": grid_config['t_max_ms'],
        },
        "velocity": {
            "source": "cube_3d",
            "cube_path": str(velocity_path),
        },
        "data_selection": {
            "mode": "all",
        },
        "algorithm": {
            "max_aperture_m": 5000.0,
            "min_aperture_m": 100.0,
            "max_dip_degrees": 65.0,
            "taper_type": "cosine",
            "taper_fraction": 0.1,
            "interpolation_method": "linear",
            "apply_spreading": False,
            "apply_obliquity": False,
            "enable_antialiasing": False,
            "kernel_type": "straight_ray",
        },
        "execution": {
            "backend": "metal_cpp",
            "max_memory_gb": 16.0,
            "auto_tile_size": True,
            "tile_nx": 64,
            "tile_ny": 64,
        },
        "output": {
            "output_dir": str(output_dir),
            "project_name": f"synthetic_{bin_name}",
            "output_stacked_image": True,
            "output_fold_map": True,
            "output_format": "zarr",
        },
    }

    return config


def run_simple_migration(
    input_dir: Path,
    output_dir: Path,
    velocity_path: Path,
    offset_m: float,
    grid_config: dict,
):
    """
    Run a simple Python-based migration for the synthetic test.
    Uses fully vectorized operations for speed.
    """
    bin_name = f"offset_{int(offset_m):04d}m"

    print(f"\n{'='*60}")
    print(f"Migrating {bin_name}")
    print(f"{'='*60}")

    # Load input traces
    traces_path = input_dir / bin_name / "traces.zarr"
    headers_path = input_dir / bin_name / "headers.parquet"

    store = zarr.storage.LocalStore(str(traces_path))
    traces_z = zarr.open_array(store=store, mode='r')
    traces = np.asarray(traces_z)
    headers = pl.read_parquet(headers_path)

    print(f"  Input traces: {traces.shape}")

    # Load velocity
    vel_store = zarr.storage.LocalStore(str(velocity_path))
    vel_z = zarr.open_array(store=vel_store, mode='r')
    velocity = float(vel_z[0, 0, 0])  # Constant velocity

    print(f"  Velocity: {velocity} m/s")

    # Output grid
    x_out = np.arange(grid_config['x_min'], grid_config['x_max'] + grid_config['dx'], grid_config['dx'])
    y_out = np.arange(grid_config['y_min'], grid_config['y_max'] + grid_config['dy'], grid_config['dy'])
    t_out = np.arange(grid_config['t_min_ms'], grid_config['t_max_ms'] + grid_config['dt_ms'], grid_config['dt_ms'])

    nx_out = len(x_out)
    ny_out = len(y_out)
    nt_out = len(t_out)

    print(f"  Output grid: {nx_out} x {ny_out} x {nt_out}")

    # Initialize output
    output = np.zeros((nx_out, ny_out, nt_out), dtype=np.float32)
    fold = np.zeros((nx_out, ny_out, nt_out), dtype=np.int32)

    # Get trace geometry
    sx = headers['source_x'].to_numpy()
    sy = headers['source_y'].to_numpy()
    rx = headers['receiver_x'].to_numpy()
    ry = headers['receiver_y'].to_numpy()

    n_traces = len(headers)
    input_dt_ms = grid_config['sample_rate_ms']
    input_nt = traces.shape[1]

    inv_v_sq = 1.0 / (velocity ** 2)
    max_aperture = 2000.0

    print(f"  Migrating {n_traces} traces...")

    # Process output points (this is more efficient for small grids)
    for ix in range(nx_out):
        ox = x_out[ix]

        for iy in range(ny_out):
            oy = y_out[iy]

            # Compute distances to all traces
            ds_all = np.sqrt((ox - sx)**2 + (oy - sy)**2)
            dr_all = np.sqrt((ox - rx)**2 + (oy - ry)**2)

            # Aperture mask
            valid_traces = (ds_all <= max_aperture) & (dr_all <= max_aperture)
            n_valid = np.sum(valid_traces)

            if n_valid == 0:
                continue

            # Get valid trace data
            valid_idx = np.where(valid_traces)[0]
            valid_traces_data = traces[valid_idx]  # (n_valid, n_samples)
            ds_valid = ds_all[valid_idx]
            dr_valid = dr_all[valid_idx]

            # For each output time
            for it in range(nt_out):
                t0_ms = t_out[it]
                if t0_ms < 10:
                    continue

                t0_s = t0_ms / 1000.0
                t0_half = t0_s / 2.0
                t0_half_sq = t0_half ** 2

                # Compute travel times for all valid traces (vectorized)
                t_src = np.sqrt(t0_half_sq + ds_valid**2 * inv_v_sq)
                t_rec = np.sqrt(t0_half_sq + dr_valid**2 * inv_v_sq)
                t_travel_ms = (t_src + t_rec) * 1000.0

                # Sample indices
                sample_idx = t_travel_ms / input_dt_ms

                # Valid samples
                valid_samples = (sample_idx >= 0) & (sample_idx < input_nt - 1)

                if not np.any(valid_samples):
                    continue

                # Interpolate
                sample_idx_valid = sample_idx[valid_samples]
                idx_lo = sample_idx_valid.astype(np.int32)
                frac = sample_idx_valid - idx_lo

                # Get values from valid traces
                trace_indices = np.arange(n_valid)[valid_samples]
                values = (
                    valid_traces_data[trace_indices, idx_lo] * (1 - frac) +
                    valid_traces_data[trace_indices, idx_lo + 1] * frac
                )

                # Sum contributions
                output[ix, iy, it] = np.sum(values)
                fold[ix, iy, it] = len(values)

        # Progress
        if (ix + 1) % 10 == 0:
            print(f"    Progress: {ix + 1}/{nx_out} inlines ({100*(ix+1)/nx_out:.0f}%)")

    # Normalize by fold
    mask = fold > 0
    output[mask] /= fold[mask]

    print(f"  Max fold: {fold.max()}")
    print(f"  Output amplitude range: {output.min():.6f} to {output.max():.6f}")

    # Save output
    output_path = output_dir / f"synthetic_{bin_name}" / "migrated_stack.zarr"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        shutil.rmtree(output_path)

    store = zarr.storage.LocalStore(str(output_path))
    z = zarr.open_array(
        store=store,
        mode='w',
        shape=output.shape,
        dtype=output.dtype,
        chunks=(min(64, nx_out), min(64, ny_out), nt_out),
    )
    z[:] = output

    # Set attributes
    z.attrs['x_min'] = grid_config['x_min']
    z.attrs['x_max'] = grid_config['x_max']
    z.attrs['y_min'] = grid_config['y_min']
    z.attrs['y_max'] = grid_config['y_max']
    z.attrs['dx'] = grid_config['dx']
    z.attrs['dy'] = grid_config['dy']
    z.attrs['dt_ms'] = grid_config['dt_ms']
    z.attrs['t_min_ms'] = grid_config['t_min_ms']
    z.attrs['t_max_ms'] = grid_config['t_max_ms']
    z.attrs['offset_m'] = offset_m
    z.attrs['velocity_ms'] = velocity

    print(f"  Saved: {output_path}")

    return output, fold


# =============================================================================
# Analysis and Visualization
# =============================================================================

def analyze_migration_result(
    output_dir: Path,
    offset_m: float,
    diffractors: list,
    grid_config: dict,
):
    """Analyze migration result and compare with expected diffractor positions."""

    bin_name = f"offset_{int(offset_m):04d}m"
    output_path = output_dir / f"synthetic_{bin_name}" / "migrated_stack.zarr"

    print(f"\n{'='*60}")
    print(f"Analyzing {bin_name}")
    print(f"{'='*60}")

    # Load migrated data
    store = zarr.storage.LocalStore(str(output_path))
    z = zarr.open_array(store=store, mode='r')
    data = np.asarray(z)

    nx, ny, nt = data.shape

    # Grid coordinates
    x_coords = np.arange(grid_config['x_min'], grid_config['x_max'] + grid_config['dx'], grid_config['dx'])[:nx]
    y_coords = np.arange(grid_config['y_min'], grid_config['y_max'] + grid_config['dy'], grid_config['dy'])[:ny]
    t_coords = np.arange(grid_config['t_min_ms'], grid_config['t_max_ms'] + grid_config['dt_ms'], grid_config['dt_ms'])[:nt]

    print(f"  Output shape: {data.shape}")
    print(f"  Amplitude range: {data.min():.6f} to {data.max():.6f}")

    # Find peaks and compare with expected diffractor positions
    for diff in diffractors:
        print(f"\n  Expected diffractor '{diff['name']}':")
        print(f"    Position: x={diff['x']}, y={diff['y']}, t0={diff['t0_ms']} ms")

        # Find nearest grid point
        ix_exp = np.argmin(np.abs(x_coords - diff['x']))
        iy_exp = np.argmin(np.abs(y_coords - diff['y']))
        it_exp = np.argmin(np.abs(t_coords - diff['t0_ms']))

        print(f"    Expected grid indices: ix={ix_exp}, iy={iy_exp}, it={it_exp}")

        # Search for maximum in a window around expected position
        window = 20  # samples
        ix_min = max(0, ix_exp - window)
        ix_max = min(nx, ix_exp + window)
        iy_min = max(0, iy_exp - window)
        iy_max = min(ny, iy_exp + window)
        it_min = max(0, it_exp - window)
        it_max = min(nt, it_exp + window)

        search_volume = data[ix_min:ix_max, iy_min:iy_max, it_min:it_max]

        # Find maximum
        max_idx = np.unravel_index(np.argmax(np.abs(search_volume)), search_volume.shape)

        ix_found = ix_min + max_idx[0]
        iy_found = iy_min + max_idx[1]
        it_found = it_min + max_idx[2]

        x_found = x_coords[ix_found]
        y_found = y_coords[iy_found]
        t_found = t_coords[it_found]

        print(f"    Found peak at: x={x_found}, y={y_found}, t={t_found} ms")
        print(f"    Found grid indices: ix={ix_found}, iy={iy_found}, it={it_found}")

        # Compute error
        dx_err = x_found - diff['x']
        dy_err = y_found - diff['y']
        dt_err = t_found - diff['t0_ms']

        print(f"    Position error: dx={dx_err:.1f}m, dy={dy_err:.1f}m, dt={dt_err:.1f}ms")

        # Check if within tolerance
        spatial_err = np.sqrt(dx_err**2 + dy_err**2)
        if spatial_err < 50 and abs(dt_err) < 10:
            print(f"    ✓ PASS: Diffractor correctly positioned")
        else:
            print(f"    ✗ FAIL: Position error too large!")


def create_qc_figures(
    output_dir: Path,
    offsets: list,
    diffractors: list,
    grid_config: dict,
):
    """Create QC figures for migration results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping figures")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Grid coordinates
    x_coords = np.arange(grid_config['x_min'], grid_config['x_max'] + grid_config['dx'], grid_config['dx'])
    y_coords = np.arange(grid_config['y_min'], grid_config['y_max'] + grid_config['dy'], grid_config['dy'])
    t_coords = np.arange(grid_config['t_min_ms'], grid_config['t_max_ms'] + grid_config['dt_ms'], grid_config['dt_ms'])

    for offset_m in offsets:
        bin_name = f"offset_{int(offset_m):04d}m"
        output_path = output_dir / f"synthetic_{bin_name}" / "migrated_stack.zarr"

        if not output_path.exists():
            continue

        store = zarr.storage.LocalStore(str(output_path))
        z = zarr.open_array(store=store, mode='r')
        data = np.asarray(z)

        nx, ny, nt = data.shape
        clip = np.percentile(np.abs(data), 99)

        # Find diffractor indices
        diff = diffractors[0]  # Use first diffractor
        ix_diff = np.argmin(np.abs(x_coords[:nx] - diff['x']))
        iy_diff = np.argmin(np.abs(y_coords[:ny] - diff['y']))

        # Create figure with inline, crossline, and time slice
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Synthetic Migration QC - Offset {offset_m}m', fontsize=14)

        # Inline through diffractor
        ax = axes[0, 0]
        section = data[ix_diff, :, :].T
        ax.imshow(section, aspect='auto', cmap='gray', vmin=-clip, vmax=clip,
                  extent=[0, ny, t_coords[nt-1], t_coords[0]])
        ax.axhline(diff['t0_ms'], color='r', linestyle='--', alpha=0.7, label=f"Expected t0={diff['t0_ms']}ms")
        ax.set_title(f'Inline through diffractor (IL={ix_diff})')
        ax.set_xlabel('Crossline')
        ax.set_ylabel('Time (ms)')
        ax.legend()

        # Crossline through diffractor
        ax = axes[0, 1]
        section = data[:, iy_diff, :].T
        ax.imshow(section, aspect='auto', cmap='gray', vmin=-clip, vmax=clip,
                  extent=[0, nx, t_coords[nt-1], t_coords[0]])
        ax.axhline(diff['t0_ms'], color='r', linestyle='--', alpha=0.7)
        ax.set_title(f'Crossline through diffractor (XL={iy_diff})')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Time (ms)')

        # Time slice at shallow diffractor
        ax = axes[1, 0]
        it_shallow = np.argmin(np.abs(t_coords[:nt] - diffractors[0]['t0_ms']))
        slice_data = data[:, :, it_shallow]
        ax.imshow(slice_data.T, aspect='auto', cmap='gray', vmin=-clip, vmax=clip, origin='lower')
        ax.plot(ix_diff, iy_diff, 'r+', markersize=15, markeredgewidth=2, label='Expected')
        ax.set_title(f'Time slice @ {t_coords[it_shallow]:.0f}ms (shallow diffractor)')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        ax.legend()

        # Time slice at deep diffractor
        ax = axes[1, 1]
        it_deep = np.argmin(np.abs(t_coords[:nt] - diffractors[1]['t0_ms']))
        slice_data = data[:, :, it_deep]
        ax.imshow(slice_data.T, aspect='auto', cmap='gray', vmin=-clip, vmax=clip, origin='lower')
        ax.plot(ix_diff, iy_diff, 'r+', markersize=15, markeredgewidth=2, label='Expected')
        ax.set_title(f'Time slice @ {t_coords[it_deep]:.0f}ms (deep diffractor)')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        ax.legend()

        plt.tight_layout()

        fig_path = fig_dir / f"migration_qc_{bin_name}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("PSTM Synthetic Diffractor Test")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Output directory: {SYNTHETIC_DIR}")
    print(f"  Velocity: {VELOCITY_MS} m/s")
    print(f"  Offsets: {OFFSETS}")
    print(f"  Diffractors: {len(DIFFRACTORS)}")
    for d in DIFFRACTORS:
        print(f"    - {d['name']}: x={d['x']}, y={d['y']}, t0={d['t0_ms']}ms")

    # Create directories
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Create velocity model
    print(f"\n{'='*60}")
    print("Step 1: Creating velocity model")
    print(f"{'='*60}")

    velocity_path = SYNTHETIC_DIR / "velocity_constant.zarr"
    create_constant_velocity_model(GRID_CONFIG, VELOCITY_MS, velocity_path)

    # Step 2: Generate synthetic data for each offset
    print(f"\n{'='*60}")
    print("Step 2: Generating synthetic traces")
    print(f"{'='*60}")

    for offset_m in OFFSETS:
        print(f"\nGenerating offset {offset_m}m...")
        traces, headers = generate_synthetic_traces(
            offset_m, GRID_CONFIG, DIFFRACTORS, VELOCITY_MS, WAVELET_FREQ_HZ
        )
        save_synthetic_data(offset_m, traces, headers, SYNTHETIC_INPUT_DIR, GRID_CONFIG)

    # Step 3: Migrate each offset volume
    print(f"\n{'='*60}")
    print("Step 3: Running migration")
    print(f"{'='*60}")

    for offset_m in OFFSETS:
        run_simple_migration(
            SYNTHETIC_INPUT_DIR,
            SYNTHETIC_OUTPUT_DIR,
            velocity_path,
            offset_m,
            GRID_CONFIG,
        )

    # Step 4: Analyze results
    print(f"\n{'='*60}")
    print("Step 4: Analyzing results")
    print(f"{'='*60}")

    for offset_m in OFFSETS:
        analyze_migration_result(SYNTHETIC_OUTPUT_DIR, offset_m, DIFFRACTORS, GRID_CONFIG)

    # Step 5: Create QC figures
    print(f"\n{'='*60}")
    print("Step 5: Creating QC figures")
    print(f"{'='*60}")

    create_qc_figures(SYNTHETIC_OUTPUT_DIR, OFFSETS, DIFFRACTORS, GRID_CONFIG)

    print(f"\n{'='*70}")
    print("SYNTHETIC TEST COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {SYNTHETIC_DIR}")
    print(f"Figures saved to: {SYNTHETIC_OUTPUT_DIR / 'figures'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
