#!/usr/bin/env python3
"""
PSTM Pipeline Diagnostic Script

This script implements systematic debugging to identify why real data
produces noisy output while synthetic tests pass.

Debugging Steps:
1. Synthetic test using ACTUAL Metal pipeline (not simple Python loop)
2. Velocity value logging
3. Constant velocity test on real data
4. Time-variant sampling disabled test
5. Fold normalization check
6. Small subset debug run

Usage:
    python diagnose_pstm_pipeline.py --test synthetic_pipeline
    python diagnose_pstm_pipeline.py --test constant_velocity
    python diagnose_pstm_pipeline.py --test no_time_variant
    python diagnose_pstm_pipeline.py --test small_subset
    python diagnose_pstm_pipeline.py --test all
"""

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import zarr

# Add pstm to path
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
from pstm.pipeline.executor import run_migration, ProgressInfo


# =============================================================================
# Configuration
# =============================================================================

DIAGNOSTIC_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_diagnostic")
COMMON_OFFSET_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
VELOCITY_SEGY_PATH = Path("/Users/olegadamovich/SeismicData/vels.sgy")

# Use offset bin 0 for testing (smallest, fastest)
TEST_OFFSET_BIN = 0

# Small grid for fast debugging (subset of full grid)
DEBUG_GRID = {
    'nx': 64,
    'ny': 64,
    'dx': 25.0,
    'dy': 12.5,
    'dt_ms': 2.0,
    't_min_ms': 0.0,
    't_max_ms': 1000.0,  # Only 500 samples for speed
}

# Full grid corners (from run_pstm_common_offset.py)
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}


def setup_logging(test_name: str) -> logging.Logger:
    """Setup detailed logging for diagnostics."""
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = DIAGNOSTIC_DIR / f"diagnostic_{test_name}_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger("pstm.diagnostic")
    logger.info(f"Diagnostic log: {log_path}")

    return logger


# =============================================================================
# Test 1: Synthetic Test Using Actual Pipeline
# =============================================================================

def create_synthetic_data_for_pipeline(
    output_dir: Path,
    grid_config: dict,
    velocity_ms: float = 2500.0,
) -> tuple[Path, Path, Path]:
    """
    Create synthetic diffractor data in the EXACT format expected by the pipeline.

    Returns:
        traces_path, headers_path, velocity_path
    """
    logger = logging.getLogger("pstm.diagnostic")
    logger.info("Creating synthetic data for pipeline test...")

    # Grid parameters
    nx = grid_config['nx']
    ny = grid_config['ny']
    dx = grid_config['dx']
    dy = grid_config['dy']
    dt_ms = grid_config['dt_ms']
    t_max_ms = grid_config['t_max_ms']
    n_samples = int(t_max_ms / dt_ms) + 1

    # Create coordinate arrays
    x_coords = np.arange(nx) * dx + GRID_CORNERS['c1'][0]
    y_coords = np.arange(ny) * dy + GRID_CORNERS['c1'][1]

    n_traces = nx * ny

    logger.info(f"  Grid: {nx}x{ny} = {n_traces} traces")
    logger.info(f"  Samples: {n_samples} @ {dt_ms} ms")
    logger.info(f"  X range: {x_coords[0]:.1f} - {x_coords[-1]:.1f}")
    logger.info(f"  Y range: {y_coords[0]:.1f} - {y_coords[-1]:.1f}")

    # Diffractor at center of grid
    diff_x = x_coords[nx // 2]
    diff_y = y_coords[ny // 2]
    diff_t0_ms = 500.0  # Zero-offset time

    logger.info(f"  Diffractor: x={diff_x:.1f}, y={diff_y:.1f}, t0={diff_t0_ms:.1f} ms")

    # Generate Ricker wavelet
    wavelet_freq = 30.0  # Hz
    wavelet_duration = 0.1  # seconds
    wavelet_samples = int(wavelet_duration / (dt_ms / 1000))
    wavelet_t = np.linspace(-wavelet_duration/2, wavelet_duration/2, wavelet_samples)
    a = (np.pi * wavelet_freq * wavelet_t) ** 2
    wavelet = (1 - 2 * a) * np.exp(-a)

    # Generate traces
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)  # Transposed format

    # Header data
    header_data = {
        'bin_trace_idx': [],
        'source_x': [],
        'source_y': [],
        'receiver_x': [],
        'receiver_y': [],
        'offset': [],
        'sr_azim': [],
        'scalar_coord': [],
    }

    offset_m = 100.0  # Small offset
    half_offset = offset_m / 2.0

    trace_idx = 0
    for ix, mx in enumerate(x_coords):
        for iy, my in enumerate(y_coords):
            # Source/receiver positions (offset along X)
            sx = mx - half_offset
            sy = my
            rx = mx + half_offset
            ry = my

            # Distance to diffractor
            ds = np.sqrt((diff_x - sx)**2 + (diff_y - sy)**2)
            dr = np.sqrt((diff_x - rx)**2 + (diff_y - ry)**2)

            # DSR traveltime
            t0_s = diff_t0_ms / 1000.0
            t0_half = t0_s / 2.0
            inv_v_sq = 1.0 / (velocity_ms ** 2)

            t_src = np.sqrt(t0_half**2 + ds**2 * inv_v_sq)
            t_rec = np.sqrt(t0_half**2 + dr**2 * inv_v_sq)
            t_travel_ms = (t_src + t_rec) * 1000.0

            # Place wavelet at traveltime
            sample_idx = int(t_travel_ms / dt_ms)
            if 0 <= sample_idx < n_samples - wavelet_samples:
                start = sample_idx - wavelet_samples // 2
                end = start + wavelet_samples
                if start >= 0 and end < n_samples:
                    traces[start:end, trace_idx] += wavelet.astype(np.float32)

            # Store headers (coordinates in raw units, will be scaled by -100)
            header_data['bin_trace_idx'].append(trace_idx)
            header_data['source_x'].append(int(sx * 100))  # Scale for -100 scalar
            header_data['source_y'].append(int(sy * 100))
            header_data['receiver_x'].append(int(rx * 100))
            header_data['receiver_y'].append(int(ry * 100))
            header_data['offset'].append(offset_m)
            header_data['sr_azim'].append(0.0)
            header_data['scalar_coord'].append(-100)

            trace_idx += 1

    # Save traces as zarr (transposed format: n_samples x n_traces)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.zarr"
    if traces_path.exists():
        shutil.rmtree(traces_path)

    store = zarr.storage.LocalStore(str(traces_path))
    z = zarr.open_array(
        store=store,
        mode='w',
        shape=traces.shape,
        dtype=traces.dtype,
        chunks=(n_samples, min(1000, n_traces)),
    )
    z[:] = traces
    z.attrs['n_samples'] = n_samples
    z.attrs['n_traces'] = n_traces
    z.attrs['sample_rate_ms'] = dt_ms

    logger.info(f"  Saved traces: {traces_path} (shape: {traces.shape})")

    # Save headers
    headers_path = output_dir / "headers.parquet"
    df = pl.DataFrame(header_data)
    df.write_parquet(headers_path)

    logger.info(f"  Saved headers: {headers_path} ({len(df)} rows)")

    # Create constant velocity model
    velocity_path = output_dir / "velocity_constant.zarr"
    if velocity_path.exists():
        shutil.rmtree(velocity_path)

    t_axis_ms = np.arange(0, t_max_ms + dt_ms, dt_ms)
    vel_cube = np.full((nx, ny, len(t_axis_ms)), velocity_ms, dtype=np.float32)

    vel_store = zarr.storage.LocalStore(str(velocity_path))
    vel_z = zarr.open_array(
        store=vel_store,
        mode='w',
        shape=vel_cube.shape,
        dtype=vel_cube.dtype,
        chunks=(nx, ny, len(t_axis_ms)),
    )
    vel_z[:] = vel_cube
    vel_z.attrs['x_axis'] = x_coords.tolist()
    vel_z.attrs['y_axis'] = y_coords.tolist()
    vel_z.attrs['t_axis_ms'] = t_axis_ms.tolist()
    vel_z.attrs['units'] = 'm/s'

    logger.info(f"  Saved velocity: {velocity_path} (constant {velocity_ms} m/s)")

    return traces_path, headers_path, velocity_path


def build_diagnostic_config(
    traces_path: Path,
    headers_path: Path,
    velocity_path: Path,
    output_dir: Path,
    grid_config: dict,
    use_time_variant: bool = False,
    use_metal_compiled: bool = True,
) -> MigrationConfig:
    """Build a diagnostic migration configuration."""

    nx = grid_config['nx']
    ny = grid_config['ny']
    dx = grid_config['dx']
    dy = grid_config['dy']
    dt_ms = grid_config['dt_ms']
    t_min_ms = grid_config['t_min_ms']
    t_max_ms = grid_config['t_max_ms']

    # Simple rectangular grid (not rotated)
    x_min = grid_config.get('x_min', GRID_CORNERS['c1'][0])
    y_min = grid_config.get('y_min', GRID_CORNERS['c1'][1])
    x_max = x_min + (nx - 1) * dx
    y_max = y_min + (ny - 1) * dy

    input_config = InputConfig(
        traces_path=traces_path,
        headers_path=headers_path,
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
        sample_rate_ms=dt_ms,
        transposed=True,
    )

    velocity_config = VelocityConfig(
        source=VelocitySource.CUBE_3D,
        velocity_path=velocity_path,
        precompute_to_output_grid=True,
    )

    aperture_config = ApertureConfig(
        max_dip_degrees=65.0,
        min_aperture_m=100.0,
        max_aperture_m=2000.0,
        taper_fraction=0.1,
    )

    amplitude_config = AmplitudeConfig(
        geometrical_spreading=False,
        obliquity_factor=False,
    )

    if use_time_variant:
        time_variant_config = TimeVariantConfig(
            enabled=True,
            frequency_table=[
                (0.0, 120.0),
                (500.0, 80.0),
                (1000.0, 50.0),
            ],
            min_downsample_factor=1,
            max_downsample_factor=4,
        )
    else:
        time_variant_config = TimeVariantConfig(enabled=False)

    anti_aliasing_config = AntiAliasingConfig(
        enabled=False,
        method=AntiAliasingMethod.NONE,
    )

    algorithm_config = AlgorithmConfig(
        interpolation=InterpolationMethod.LINEAR,
        aperture=aperture_config,
        amplitude=amplitude_config,
        time_variant=time_variant_config,
        anti_aliasing=anti_aliasing_config,
    )

    # Simple axis-aligned grid (not rotated)
    output_grid = OutputGridConfig(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        t_min_ms=t_min_ms,
        t_max_ms=t_max_ms,
        dx=dx,
        dy=dy,
        dt_ms=dt_ms,
    )

    output_config = OutputConfig(
        output_dir=output_dir,
        grid=output_grid,
        format=OutputFormat.ZARR,
    )

    backend = ComputeBackend.METAL_COMPILED if use_metal_compiled else ComputeBackend.NUMPY

    resource_config = ResourceConfig(
        backend=backend,
        max_memory_gb=16.0,
    )

    tiling_config = TilingConfig(
        auto_tile_size=False,
        tile_nx=min(64, nx),
        tile_ny=min(64, ny),
        ordering='snake',
    )

    checkpoint_config = CheckpointConfig(
        enabled=False,  # Disable for fast debugging
    )

    execution_config = ExecutionConfig(
        resources=resource_config,
        tiling=tiling_config,
        checkpoint=checkpoint_config,
    )

    return MigrationConfig(
        name="PSTM_Diagnostic",
        input=input_config,
        velocity=velocity_config,
        algorithm=algorithm_config,
        output=output_config,
        execution=execution_config,
    )


# Alias for centered grid config
build_diagnostic_config_centered = build_diagnostic_config


def test_synthetic_pipeline():
    """
    Test 1: Run synthetic data through the ACTUAL Metal pipeline.

    This tests whether the Metal kernel correctly collapses a known diffractor.
    """
    logger = setup_logging("synthetic_pipeline")
    logger.info("=" * 70)
    logger.info("TEST 1: Synthetic Data Through Actual Pipeline")
    logger.info("=" * 70)

    test_dir = DIAGNOSTIC_DIR / "test_synthetic_pipeline"
    input_dir = test_dir / "input"
    output_dir = test_dir / "output"

    # Create synthetic data
    traces_path, headers_path, velocity_path = create_synthetic_data_for_pipeline(
        input_dir, DEBUG_GRID, velocity_ms=2500.0
    )

    # Build config
    config = build_diagnostic_config(
        traces_path=traces_path,
        headers_path=headers_path,
        velocity_path=velocity_path,
        output_dir=output_dir,
        grid_config=DEBUG_GRID,
        use_time_variant=False,
        use_metal_compiled=True,
    )

    logger.info(f"\nConfiguration:")
    logger.info(f"  Output grid: {config.output.grid.nx}x{config.output.grid.ny}x{config.output.grid.nt}")
    logger.info(f"  Backend: {config.execution.resources.backend}")
    logger.info(f"  Time-variant: {config.algorithm.time_variant.enabled}")

    # Run migration
    logger.info("\nRunning migration...")
    start_time = time.time()

    try:
        success = run_migration(config, resume=False)
        elapsed = time.time() - start_time

        if success:
            logger.info(f"\nMigration completed in {elapsed:.1f}s")

            # Analyze output
            analyze_migration_output(output_dir, DEBUG_GRID, logger)
        else:
            logger.error("Migration failed!")

    except Exception as e:
        logger.error(f"Migration error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    return success


def analyze_migration_output(output_dir: Path, grid_config: dict, logger: logging.Logger):
    """Analyze migration output to check for diffractor focusing."""

    # Find output zarr
    output_zarr = output_dir / "migrated_stack.zarr"
    if not output_zarr.exists():
        # Try to find it in subdirectories
        for p in output_dir.rglob("*.zarr"):
            if "migrated" in p.name or "stack" in p.name:
                output_zarr = p
                break

    if not output_zarr.exists():
        logger.warning(f"Output not found in {output_dir}")
        return

    store = zarr.storage.LocalStore(str(output_zarr))
    z = zarr.open_array(store=store, mode='r')
    data = np.asarray(z)

    logger.info(f"\nOutput Analysis:")
    logger.info(f"  Shape: {data.shape}")
    logger.info(f"  Dtype: {data.dtype}")
    logger.info(f"  Value range: {data.min():.6f} to {data.max():.6f}")
    logger.info(f"  Mean: {np.mean(data):.6f}")
    logger.info(f"  Std: {np.std(data):.6f}")
    logger.info(f"  Non-zero fraction: {np.count_nonzero(data) / data.size:.4f}")

    # Find maximum amplitude location
    max_idx = np.unravel_index(np.argmax(np.abs(data)), data.shape)
    logger.info(f"  Max amplitude at: ix={max_idx[0]}, iy={max_idx[1]}, it={max_idx[2]}")

    # Expected diffractor location
    nx, ny = grid_config['nx'], grid_config['ny']
    expected_ix = nx // 2
    expected_iy = ny // 2
    expected_it = int(500.0 / grid_config['dt_ms'])  # t0 = 500ms

    logger.info(f"  Expected diffractor at: ix={expected_ix}, iy={expected_iy}, it={expected_it}")

    # Check if diffractor is focused
    dx_err = abs(max_idx[0] - expected_ix)
    dy_err = abs(max_idx[1] - expected_iy)
    dt_err = abs(max_idx[2] - expected_it)

    if dx_err <= 2 and dy_err <= 2 and dt_err <= 5:
        logger.info(f"  ✓ PASS: Diffractor correctly focused (error: {dx_err}, {dy_err}, {dt_err})")
    else:
        logger.error(f"  ✗ FAIL: Diffractor not focused (error: {dx_err}, {dy_err}, {dt_err})")


# =============================================================================
# Test 2: Real Data with Constant Velocity
# =============================================================================

def create_constant_velocity_model(
    output_path: Path,
    velocity_ms: float,
    x_range: tuple,
    y_range: tuple,
    t_max_ms: float,
    dt_ms: float = 4.0,
    nx: int = 50,
    ny: int = 50,
) -> Path:
    """Create a constant velocity model matching survey extent."""
    logger = logging.getLogger("pstm.diagnostic")

    x_axis = np.linspace(x_range[0], x_range[1], nx)
    y_axis = np.linspace(y_range[0], y_range[1], ny)
    t_axis_ms = np.arange(0, t_max_ms + dt_ms, dt_ms)
    nt = len(t_axis_ms)

    vel_cube = np.full((nx, ny, nt), velocity_ms, dtype=np.float32)

    if output_path.exists():
        shutil.rmtree(output_path)

    store = zarr.storage.LocalStore(str(output_path))
    z = zarr.open_array(
        store=store,
        mode='w',
        shape=vel_cube.shape,
        dtype=vel_cube.dtype,
        chunks=(nx, ny, nt),
    )
    z[:] = vel_cube
    z.attrs['x_axis'] = x_axis.tolist()
    z.attrs['y_axis'] = y_axis.tolist()
    z.attrs['t_axis_ms'] = t_axis_ms.tolist()
    z.attrs['units'] = 'm/s'

    logger.info(f"Created constant velocity model: {output_path}")
    logger.info(f"  Velocity: {velocity_ms} m/s")
    logger.info(f"  Shape: {vel_cube.shape}")
    logger.info(f"  X range: {x_axis[0]:.1f} - {x_axis[-1]:.1f}")
    logger.info(f"  Y range: {y_axis[0]:.1f} - {y_axis[-1]:.1f}")

    return output_path


def test_constant_velocity():
    """
    Test 2: Run real data with constant velocity.

    If output is still noisy with constant velocity, the problem is NOT velocity.
    """
    logger = setup_logging("constant_velocity")
    logger.info("=" * 70)
    logger.info("TEST 2: Real Data with Constant Velocity")
    logger.info("=" * 70)

    test_dir = DIAGNOSTIC_DIR / "test_constant_velocity"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Use real data paths
    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{TEST_OFFSET_BIN:02d}"
    traces_path = bin_dir / "traces.zarr"
    headers_path = bin_dir / "headers.parquet"

    if not traces_path.exists():
        logger.error(f"Traces not found: {traces_path}")
        return False

    # Create constant velocity
    velocity_path = test_dir / "velocity_constant.zarr"
    create_constant_velocity_model(
        velocity_path,
        velocity_ms=3000.0,  # Reasonable average velocity
        x_range=(618000, 632000),
        y_range=(5106000, 5120000),
        t_max_ms=2000.0,
    )

    # Build config with small grid for speed
    small_grid = {
        'nx': 64,
        'ny': 64,
        'dx': 50.0,
        'dy': 50.0,
        'dt_ms': 4.0,
        't_min_ms': 0.0,
        't_max_ms': 1000.0,
    }

    output_dir = test_dir / "output"

    config = build_diagnostic_config(
        traces_path=traces_path,
        headers_path=headers_path,
        velocity_path=velocity_path,
        output_dir=output_dir,
        grid_config=small_grid,
        use_time_variant=False,
        use_metal_compiled=True,
    )

    logger.info(f"\nConfiguration:")
    logger.info(f"  Input traces: {traces_path}")
    logger.info(f"  Velocity: CONSTANT 3000 m/s")
    logger.info(f"  Output grid: {config.output.grid.nx}x{config.output.grid.ny}x{config.output.grid.nt}")
    logger.info(f"  Time-variant: {config.algorithm.time_variant.enabled}")

    # Run migration
    logger.info("\nRunning migration...")
    start_time = time.time()

    try:
        success = run_migration(config, resume=False)
        elapsed = time.time() - start_time

        if success:
            logger.info(f"\nMigration completed in {elapsed:.1f}s")
            analyze_real_data_output(output_dir, small_grid, logger)
        else:
            logger.error("Migration failed!")

    except Exception as e:
        logger.error(f"Migration error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    return success


def analyze_real_data_output(output_dir: Path, grid_config: dict, logger: logging.Logger):
    """Analyze real data migration output."""

    output_zarr = None
    for p in output_dir.rglob("*.zarr"):
        if "migrated" in p.name.lower() or "stack" in p.name.lower():
            output_zarr = p
            break

    if output_zarr is None:
        logger.warning(f"Output not found in {output_dir}")
        return

    store = zarr.storage.LocalStore(str(output_zarr))
    z = zarr.open_array(store=store, mode='r')
    data = np.asarray(z)

    logger.info(f"\nOutput Analysis:")
    logger.info(f"  Shape: {data.shape}")
    logger.info(f"  Value range: {data.min():.6f} to {data.max():.6f}")
    logger.info(f"  Mean: {np.mean(data):.6f}")
    logger.info(f"  Std: {np.std(data):.6f}")
    logger.info(f"  Non-zero fraction: {np.count_nonzero(data) / data.size:.4f}")

    # Signal-to-noise estimate
    # Use median as noise estimate, max as signal
    noise_est = np.median(np.abs(data))
    signal_est = np.percentile(np.abs(data), 99)
    snr = signal_est / (noise_est + 1e-10)

    logger.info(f"  Estimated SNR: {snr:.1f}")

    if snr > 10:
        logger.info("  ✓ Output appears to have reasonable signal-to-noise")
    else:
        logger.warning("  ✗ Output appears noisy (low SNR)")

    # Check for NaN/Inf
    n_nan = np.count_nonzero(np.isnan(data))
    n_inf = np.count_nonzero(np.isinf(data))
    if n_nan > 0 or n_inf > 0:
        logger.error(f"  ✗ Found {n_nan} NaN and {n_inf} Inf values!")


# =============================================================================
# Test 3: Disable Time-Variant Sampling
# =============================================================================

def test_no_time_variant():
    """
    Test 3: Run real data WITHOUT time-variant sampling.

    If output is clean, the problem is in time-variant resampling.
    """
    logger = setup_logging("no_time_variant")
    logger.info("=" * 70)
    logger.info("TEST 3: Real Data WITHOUT Time-Variant Sampling")
    logger.info("=" * 70)

    test_dir = DIAGNOSTIC_DIR / "test_no_time_variant"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Use real data paths
    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{TEST_OFFSET_BIN:02d}"
    traces_path = bin_dir / "traces.zarr"
    headers_path = bin_dir / "headers.parquet"

    if not traces_path.exists():
        logger.error(f"Traces not found: {traces_path}")
        return False

    # Use real velocity (load and convert)
    velocity_path = test_dir / "velocity_real.zarr"
    load_and_convert_velocity(VELOCITY_SEGY_PATH, velocity_path)

    # Small grid
    small_grid = {
        'nx': 64,
        'ny': 64,
        'dx': 50.0,
        'dy': 50.0,
        'dt_ms': 4.0,
        't_min_ms': 0.0,
        't_max_ms': 1000.0,
    }

    output_dir = test_dir / "output"

    config = build_diagnostic_config(
        traces_path=traces_path,
        headers_path=headers_path,
        velocity_path=velocity_path,
        output_dir=output_dir,
        grid_config=small_grid,
        use_time_variant=False,  # KEY: Disabled
        use_metal_compiled=True,
    )

    logger.info(f"\nConfiguration:")
    logger.info(f"  Input traces: {traces_path}")
    logger.info(f"  Velocity: Real (from SEG-Y)")
    logger.info(f"  Time-variant: DISABLED")

    # Run migration
    logger.info("\nRunning migration...")
    start_time = time.time()

    try:
        success = run_migration(config, resume=False)
        elapsed = time.time() - start_time

        if success:
            logger.info(f"\nMigration completed in {elapsed:.1f}s")
            analyze_real_data_output(output_dir, small_grid, logger)
        else:
            logger.error("Migration failed!")

    except Exception as e:
        logger.error(f"Migration error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    return success


def load_and_convert_velocity(segy_path: Path, output_path: Path) -> Path:
    """Load velocity from SEG-Y and convert to zarr."""
    import segyio
    from scipy.interpolate import griddata

    logger = logging.getLogger("pstm.diagnostic")
    logger.info(f"Loading velocity from: {segy_path}")

    with segyio.open(str(segy_path), 'r', strict=False) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)
        t_axis_ms = f.samples.astype(np.float64)

        # Get coordinate scalar
        h0 = f.header[0]
        scalar = h0[segyio.TraceField.SourceGroupScalar]
        if scalar < 0:
            scale = 1.0 / abs(scalar)
        elif scalar > 0:
            scale = float(scalar)
        else:
            scale = 1.0

        # Extract coordinates and data
        x_coords = np.zeros(n_traces, dtype=np.float64)
        y_coords = np.zeros(n_traces, dtype=np.float64)
        velocity_data = np.zeros((n_traces, n_samples), dtype=np.float32)

        for i in range(n_traces):
            h = f.header[i]
            x_coords[i] = h[segyio.TraceField.CDP_X] * scale
            y_coords[i] = h[segyio.TraceField.CDP_Y] * scale
            velocity_data[i, :] = f.trace[i]

    logger.info(f"  X range: {x_coords.min():.1f} - {x_coords.max():.1f}")
    logger.info(f"  Y range: {y_coords.min():.1f} - {y_coords.max():.1f}")
    logger.info(f"  Velocity range: {velocity_data.min():.0f} - {velocity_data.max():.0f} m/s")

    # Interpolate to regular grid
    target_nx, target_ny = 50, 50
    x_axis = np.linspace(x_coords.min(), x_coords.max(), target_nx)
    y_axis = np.linspace(y_coords.min(), y_coords.max(), target_ny)

    xi, yi = np.meshgrid(x_axis, y_axis, indexing='ij')
    points = np.column_stack([x_coords, y_coords])

    velocity_cube = np.zeros((target_nx, target_ny, n_samples), dtype=np.float32)

    for t in range(n_samples):
        velocity_cube[:, :, t] = griddata(
            points,
            velocity_data[:, t],
            (xi, yi),
            method='linear',
            fill_value=np.nanmean(velocity_data[:, t])
        )

    # Save as zarr
    if output_path.exists():
        shutil.rmtree(output_path)

    store = zarr.storage.LocalStore(str(output_path))
    z = zarr.open_array(
        store=store,
        mode='w',
        shape=velocity_cube.shape,
        dtype=velocity_cube.dtype,
        chunks=(target_nx, target_ny, n_samples),
    )
    z[:] = velocity_cube
    z.attrs['x_axis'] = x_axis.tolist()
    z.attrs['y_axis'] = y_axis.tolist()
    z.attrs['t_axis_ms'] = t_axis_ms.tolist()
    z.attrs['units'] = 'm/s'

    logger.info(f"  Saved: {output_path}")

    return output_path


# =============================================================================
# Test 4: Small Subset Debug Run
# =============================================================================

def test_small_subset():
    """
    Test 4: Run a very small subset with maximum debugging.

    Single tile, verbose logging, check every step.
    """
    logger = setup_logging("small_subset")
    logger.info("=" * 70)
    logger.info("TEST 4: Small Subset Debug Run")
    logger.info("=" * 70)

    test_dir = DIAGNOSTIC_DIR / "test_small_subset"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Use real data
    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{TEST_OFFSET_BIN:02d}"
    traces_path = bin_dir / "traces.zarr"
    headers_path = bin_dir / "headers.parquet"

    if not traces_path.exists():
        logger.error(f"Traces not found: {traces_path}")
        return False

    # Load and inspect traces
    store = zarr.storage.LocalStore(str(traces_path))
    z = zarr.open_array(store=store, mode='r')
    logger.info(f"\nInput traces:")
    logger.info(f"  Shape: {z.shape}")
    logger.info(f"  Dtype: {z.dtype}")

    # Sample a few traces
    sample_traces = z[:, :5]
    logger.info(f"  First 5 traces stats:")
    logger.info(f"    Min: {sample_traces.min():.6f}")
    logger.info(f"    Max: {sample_traces.max():.6f}")
    logger.info(f"    Mean: {sample_traces.mean():.6f}")
    logger.info(f"    Std: {sample_traces.std():.6f}")

    # Load headers
    df = pl.read_parquet(headers_path)
    logger.info(f"\nInput headers:")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Columns: {df.columns}")

    # Check coordinate ranges after scaling
    scalar = df['scalar_coord'][0]
    scale = 1.0 / abs(scalar) if scalar < 0 else (scalar if scalar > 0 else 1.0)

    sx = df['source_x'].to_numpy() * scale
    sy = df['source_y'].to_numpy() * scale
    rx = df['receiver_x'].to_numpy() * scale
    ry = df['receiver_y'].to_numpy() * scale
    mx = (sx + rx) / 2
    my = (sy + ry) / 2

    logger.info(f"\nCoordinates (after scaling by {scale}):")
    logger.info(f"  Midpoint X: {mx.min():.1f} - {mx.max():.1f}")
    logger.info(f"  Midpoint Y: {my.min():.1f} - {my.max():.1f}")

    # Very small grid - just 16x16
    tiny_grid = {
        'nx': 16,
        'ny': 16,
        'dx': 100.0,
        'dy': 100.0,
        'dt_ms': 4.0,
        't_min_ms': 0.0,
        't_max_ms': 800.0,
    }

    # Center grid on midpoint center
    center_x = (mx.min() + mx.max()) / 2
    center_y = (my.min() + my.max()) / 2

    logger.info(f"\nOutput grid centered at: ({center_x:.1f}, {center_y:.1f})")

    # Create constant velocity
    velocity_path = test_dir / "velocity_constant.zarr"
    create_constant_velocity_model(
        velocity_path,
        velocity_ms=3000.0,
        x_range=(mx.min() - 1000, mx.max() + 1000),
        y_range=(my.min() - 1000, my.max() + 1000),
        t_max_ms=1000.0,
    )

    output_dir = test_dir / "output"

    # Create a centered grid config
    centered_grid = {
        'nx': tiny_grid['nx'],
        'ny': tiny_grid['ny'],
        'dx': tiny_grid['dx'],
        'dy': tiny_grid['dy'],
        'dt_ms': tiny_grid['dt_ms'],
        't_min_ms': tiny_grid['t_min_ms'],
        't_max_ms': tiny_grid['t_max_ms'],
        'x_min': center_x - (tiny_grid['nx'] // 2) * tiny_grid['dx'],
        'y_min': center_y - (tiny_grid['ny'] // 2) * tiny_grid['dy'],
    }

    config = build_diagnostic_config_centered(
        traces_path=traces_path,
        headers_path=headers_path,
        velocity_path=velocity_path,
        output_dir=output_dir,
        grid_config=centered_grid,
        use_time_variant=False,
        use_metal_compiled=True,
    )

    logger.info(f"\nFinal output grid:")
    logger.info(f"  X: {config.output.grid.x_min:.1f} - {config.output.grid.x_max:.1f}")
    logger.info(f"  Y: {config.output.grid.y_min:.1f} - {config.output.grid.y_max:.1f}")
    logger.info(f"  Size: {config.output.grid.nx}x{config.output.grid.ny}x{config.output.grid.nt}")

    # Run migration
    logger.info("\nRunning migration...")
    start_time = time.time()

    try:
        success = run_migration(config, resume=False)
        elapsed = time.time() - start_time

        if success:
            logger.info(f"\nMigration completed in {elapsed:.1f}s")
            analyze_real_data_output(output_dir, tiny_grid, logger)
        else:
            logger.error("Migration failed!")

    except Exception as e:
        logger.error(f"Migration error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    return success


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PSTM Pipeline Diagnostics")
    parser.add_argument(
        "--test",
        choices=["synthetic_pipeline", "constant_velocity", "no_time_variant", "small_subset", "all"],
        default="all",
        help="Which diagnostic test to run",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PSTM PIPELINE DIAGNOSTIC")
    print("=" * 70)
    print(f"Test: {args.test}")
    print(f"Output: {DIAGNOSTIC_DIR}")
    print()

    results = {}

    if args.test in ["synthetic_pipeline", "all"]:
        print("\n" + "=" * 70)
        print("Running: Synthetic Pipeline Test")
        print("=" * 70)
        results["synthetic_pipeline"] = test_synthetic_pipeline()

    if args.test in ["constant_velocity", "all"]:
        print("\n" + "=" * 70)
        print("Running: Constant Velocity Test")
        print("=" * 70)
        results["constant_velocity"] = test_constant_velocity()

    if args.test in ["no_time_variant", "all"]:
        print("\n" + "=" * 70)
        print("Running: No Time-Variant Test")
        print("=" * 70)
        results["no_time_variant"] = test_no_time_variant()

    if args.test in ["small_subset", "all"]:
        print("\n" + "=" * 70)
        print("Running: Small Subset Test")
        print("=" * 70)
        results["small_subset"] = test_small_subset()

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nLogs saved to: {DIAGNOSTIC_DIR}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
