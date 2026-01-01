#!/usr/bin/env python3
"""
Create Pre-PSTM Stacked Cube using NMO Correction.

Applies Normal Moveout (NMO) correction to common offset gathers and stacks
them to create a pre-migration stacked volume. Includes stretch mute control.

NMO Formula:
    t_nmo = sqrt(t0^2 + (offset/v_rms)^2)

Stretch Mute:
    stretch = (t_nmo - t0) / t0
    Mute if stretch > threshold (in percent)

Usage:
    python create_nmo_stack.py --velocity velocity.zarr --output stack.zarr
    python create_nmo_stack.py --stretch-mute 30  # 30% stretch mute
    python create_nmo_stack.py --bins 0-20 --velocity-1d 1500,0.5
"""

import argparse
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from numba import njit, prange

# Add pstm to path
sys.path.insert(0, str(Path(__file__).parent))

from pstm.data.velocity_model import (
    VelocityModel,
    ConstantVelocityModel,
    LinearVelocityModel,
    TableVelocityModel,
    CubeVelocityModel,
)


# =============================================================================
# Configuration
# =============================================================================

COMMON_OFFSET_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
VELOCITY_PATH = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm_full.zarr")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset")

# Default grid parameters
DX = 25.0   # meters
DY = 12.5   # meters
DT_MS = 2.0 # milliseconds

# Grid corners (rotated grid) - same as PSTM migration
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),  # Origin (IL=1, XL=1)
    'c2': (627094.02, 5106803.16),  # Inline end
    'c3': (631143.35, 5110261.43),  # Far corner
    'c4': (622862.92, 5119956.77),  # Crossline end
}

# Default stretch mute (percent)
DEFAULT_STRETCH_MUTE_PERCENT = 30.0


# =============================================================================
# Grid Computation
# =============================================================================

def compute_rotated_grid(
    c1: tuple[float, float],
    c2: tuple[float, float],
    c3: tuple[float, float],
    c4: tuple[float, float],
    dx: float,
    dy: float,
) -> tuple[NDArray, NDArray, int, int]:
    """
    Compute rotated grid coordinates from corner points.

    Args:
        c1-c4: Corner coordinates (x, y)
        dx, dy: Grid spacing in meters

    Returns:
        X_grid, Y_grid: 2D coordinate arrays (nx, ny)
        nx, ny: Grid dimensions
    """
    c1, c2, c3, c4 = np.array(c1), np.array(c2), np.array(c3), np.array(c4)

    # Inline direction (c1 -> c2)
    inline_vec = c2 - c1
    inline_length = np.linalg.norm(inline_vec)
    inline_unit = inline_vec / inline_length

    # Crossline direction (c1 -> c4)
    xline_vec = c4 - c1
    xline_length = np.linalg.norm(xline_vec)
    xline_unit = xline_vec / xline_length

    # Grid dimensions
    nx = int(np.round(inline_length / dx)) + 1
    ny = int(np.round(xline_length / dy)) + 1

    # Create grid
    il_offsets = np.arange(nx) * dx
    xl_offsets = np.arange(ny) * dy

    X = np.zeros((nx, ny), dtype=np.float64)
    Y = np.zeros((nx, ny), dtype=np.float64)

    for i, il_off in enumerate(il_offsets):
        for j, xl_off in enumerate(xl_offsets):
            pos = c1 + il_off * inline_unit + xl_off * xline_unit
            X[i, j] = pos[0]
            Y[i, j] = pos[1]

    return X, Y, nx, ny


def find_cdp_bin(
    cdp_x: float,
    cdp_y: float,
    X_grid: NDArray,
    Y_grid: NDArray,
    c1: tuple[float, float],
    inline_unit: NDArray,
    xline_unit: NDArray,
    dx: float,
    dy: float,
) -> tuple[int, int] | None:
    """
    Find the grid bin (i, j) for a given CDP location.

    Returns None if outside grid bounds.
    """
    # Vector from origin to CDP
    vec = np.array([cdp_x - c1[0], cdp_y - c1[1]])

    # Project onto inline and crossline directions
    il_dist = np.dot(vec, inline_unit)
    xl_dist = np.dot(vec, xline_unit)

    # Convert to bin indices
    i = int(np.round(il_dist / dx))
    j = int(np.round(xl_dist / dy))

    nx, ny = X_grid.shape
    if 0 <= i < nx and 0 <= j < ny:
        return i, j
    return None


# =============================================================================
# NMO Functions (Numba-accelerated)
# =============================================================================

@njit(cache=True, parallel=True)
def resample_traces_fast(
    traces: np.ndarray,
    input_t: np.ndarray,
    output_t: np.ndarray,
) -> np.ndarray:
    """Resample all traces to new time axis (parallel)."""
    n_traces = traces.shape[0]
    n_out = len(output_t)
    output = np.zeros((n_traces, n_out), dtype=np.float32)

    for tr in prange(n_traces):
        for i in range(n_out):
            t = output_t[i]
            # Find bracketing samples in input
            if t <= input_t[0]:
                output[tr, i] = traces[tr, 0]
            elif t >= input_t[-1]:
                output[tr, i] = traces[tr, -1]
            else:
                # Binary search would be faster but linear is fine for regular grids
                for j in range(len(input_t) - 1):
                    if input_t[j] <= t < input_t[j + 1]:
                        frac = (t - input_t[j]) / (input_t[j + 1] - input_t[j])
                        output[tr, i] = traces[tr, j] * (1 - frac) + traces[tr, j + 1] * frac
                        break
    return output


@njit(cache=True)
def compute_cdp_bins_fast(
    cdp_x: np.ndarray,
    cdp_y: np.ndarray,
    origin_x: float,
    origin_y: float,
    inline_ux: float,
    inline_uy: float,
    xline_ux: float,
    xline_uy: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CDP bin indices for all traces at once (Numba accelerated).

    Returns:
        i_bins, j_bins: Bin indices for each trace
        valid: Boolean mask for traces within grid
    """
    n_traces = len(cdp_x)
    i_bins = np.empty(n_traces, dtype=np.int32)
    j_bins = np.empty(n_traces, dtype=np.int32)
    valid = np.empty(n_traces, dtype=np.bool_)

    for t in range(n_traces):
        # Vector from origin to CDP
        vec_x = cdp_x[t] - origin_x
        vec_y = cdp_y[t] - origin_y

        # Project onto inline and crossline directions
        il_dist = vec_x * inline_ux + vec_y * inline_uy
        xl_dist = vec_x * xline_ux + vec_y * xline_uy

        # Convert to bin indices
        i = int(round(il_dist / dx))
        j = int(round(xl_dist / dy))

        if 0 <= i < nx and 0 <= j < ny:
            i_bins[t] = i
            j_bins[t] = j
            valid[t] = True
        else:
            i_bins[t] = 0
            j_bins[t] = 0
            valid[t] = False

    return i_bins, j_bins, valid


@njit(cache=True, parallel=True)
def apply_nmo_batch(
    traces: np.ndarray,
    t_axis_ms: np.ndarray,
    offsets: np.ndarray,
    vrms_1d: np.ndarray,
    stretch_mute_percent: float,
) -> np.ndarray:
    """
    Apply NMO correction to a batch of traces (Numba parallel).

    Args:
        traces: Input traces (n_traces, n_samples)
        t_axis_ms: Time axis in milliseconds
        offsets: Offset for each trace (n_traces,)
        vrms_1d: 1D velocity profile (n_samples,)
        stretch_mute_percent: Maximum stretch in percent

    Returns:
        NMO-corrected traces (n_traces, n_samples)
    """
    n_traces, nt = traces.shape
    output = np.zeros((n_traces, nt), dtype=np.float32)

    t0_s = t_axis_ms / 1000.0
    t_max = t_axis_ms[-1]
    dt = t_axis_ms[1] - t_axis_ms[0]

    for tr in prange(n_traces):
        offset = offsets[tr]

        for it in range(nt):
            t0 = t0_s[it]
            v = vrms_1d[it]
            if v < 100.0:
                v = 100.0

            # NMO time
            t_nmo_s = np.sqrt(t0 * t0 + (offset / v) ** 2)
            t_nmo_ms = t_nmo_s * 1000.0

            # Stretch mute
            if t0 > 0:
                stretch = (t_nmo_s - t0) / t0 * 100.0
                if stretch > stretch_mute_percent:
                    continue
            else:
                continue  # Mute t0=0

            # Check bounds
            if t_nmo_ms >= t_max:
                continue

            # Linear interpolation
            idx_f = t_nmo_ms / dt
            idx0 = int(idx_f)
            if idx0 < 0 or idx0 >= nt - 1:
                continue

            frac = idx_f - idx0
            val = traces[tr, idx0] * (1.0 - frac) + traces[tr, idx0 + 1] * frac
            output[tr, it] = val

    return output


@njit(cache=True)
def accumulate_to_stack(
    stack: np.ndarray,
    fold: np.ndarray,
    nmo_traces: np.ndarray,
    i_bins: np.ndarray,
    j_bins: np.ndarray,
    valid: np.ndarray,
) -> int:
    """
    Accumulate NMO-corrected traces into stack (thread-safe per bin).

    Returns number of traces added.
    """
    n_traces, nt = nmo_traces.shape
    added = 0

    # Note: This is not fully parallel-safe for overlapping bins
    # but works well enough for seismic data with sparse bin overlap
    for tr in range(n_traces):
        if not valid[tr]:
            continue

        i = i_bins[tr]
        j = j_bins[tr]

        for it in range(nt):
            val = nmo_traces[tr, it]
            if val != 0.0:
                stack[i, j, it] += val
                fold[i, j, it] += 1

        added += 1

    return added


def apply_nmo_correction(
    trace: NDArray[np.float32],
    t_axis_ms: NDArray[np.float64],
    offset_m: float,
    vrms: NDArray[np.float64],
    stretch_mute_percent: float = 30.0,
) -> NDArray[np.float32]:
    """
    Apply NMO correction to a single trace with stretch mute (legacy fallback).
    """
    nt = len(trace)
    t0_s = t_axis_ms / 1000.0
    vrms_safe = np.maximum(vrms, 100.0)

    # NMO times
    t_nmo_s = np.sqrt(t0_s**2 + (offset_m / vrms_safe)**2)
    t_nmo_ms = t_nmo_s * 1000.0

    # Stretch
    with np.errstate(divide='ignore', invalid='ignore'):
        stretch = np.where(t0_s > 0, (t_nmo_s - t0_s) / t0_s * 100.0, np.inf)

    # Interpolate
    trace_interp = interp1d(t_axis_ms, trace, kind='linear',
                            bounds_error=False, fill_value=0.0)
    nmo_trace = trace_interp(t_nmo_ms).astype(np.float32)

    # Mute
    nmo_trace[stretch > stretch_mute_percent] = 0.0
    nmo_trace[t_nmo_ms > t_axis_ms[-1]] = 0.0

    return nmo_trace


# =============================================================================
# Data Loading
# =============================================================================

def get_available_bins(common_offset_dir: Path) -> list[int]:
    """Get list of available offset bin numbers."""
    bins = []
    for d in common_offset_dir.iterdir():
        if d.is_dir() and d.name.startswith("offset_bin_"):
            try:
                bin_num = int(d.name.replace("offset_bin_", ""))
                if (d / "traces.zarr").exists() and (d / "headers.parquet").exists():
                    bins.append(bin_num)
            except ValueError:
                continue
    return sorted(bins)


def load_bin_data(
    common_offset_dir: Path,
    bin_num: int,
) -> tuple[NDArray[np.float32], pl.DataFrame, dict]:
    """
    Load trace data and headers for an offset bin.

    Returns:
        traces: (n_traces, n_samples) array
        headers: Polars DataFrame with trace headers
        attrs: Zarr attributes (sample_rate_ms, etc.)
    """
    bin_dir = common_offset_dir / f"offset_bin_{bin_num:02d}"
    traces_path = bin_dir / "traces.zarr"
    headers_path = bin_dir / "headers.parquet"

    # Load traces
    z = zarr.open(str(traces_path), mode='r')
    attrs = dict(z.attrs)

    # Check if transposed (n_samples, n_traces) vs (n_traces, n_samples)
    # Typically n_samples ~1000-2000, n_traces much larger
    if z.shape[0] < z.shape[1]:
        # Transposed: (n_samples, n_traces) -> need to transpose
        traces = np.asarray(z[:]).T
    else:
        traces = np.asarray(z[:])

    # Load headers
    headers = pl.read_parquet(headers_path)

    # Ensure traces and headers match
    n_traces = min(len(traces), len(headers))
    if len(traces) != len(headers):
        print(f"    WARNING: traces ({len(traces)}) != headers ({len(headers)}), using {n_traces}")
        traces = traces[:n_traces]
        headers = headers.head(n_traces)

    return traces.astype(np.float32), headers, attrs


# =============================================================================
# Main NMO Stack Function
# =============================================================================

def create_nmo_stack(
    common_offset_dir: Path,
    velocity_model: VelocityModel,
    output_path: Path,
    bins: list[int],
    grid_corners: dict,
    dx: float = 25.0,
    dy: float = 12.5,
    dt_ms: float = 2.0,
    t_min_ms: float = 0.0,
    t_max_ms: float = 2000.0,
    stretch_mute_percent: float = 30.0,
    verbose: bool = True,
) -> Path:
    """
    Create NMO-corrected stacked volume from common offset gathers.

    Args:
        common_offset_dir: Directory with offset_bin_XX subdirectories
        velocity_model: Velocity model for NMO correction
        output_path: Output zarr path
        bins: List of offset bin numbers to process
        grid_corners: Dict with c1, c2, c3, c4 corner coordinates
        dx, dy: Grid spacing in meters
        dt_ms: Time sample interval in milliseconds
        t_min_ms, t_max_ms: Time range
        stretch_mute_percent: Maximum allowed stretch (percent)
        verbose: Print progress

    Returns:
        Path to output zarr file
    """
    # Compute output grid
    if verbose:
        print("Computing output grid...")

    c1 = np.array(grid_corners['c1'])
    c2 = np.array(grid_corners['c2'])
    c4 = np.array(grid_corners['c4'])

    # Inline/crossline unit vectors
    inline_vec = c2 - c1
    inline_unit = inline_vec / np.linalg.norm(inline_vec)
    xline_vec = c4 - c1
    xline_unit = xline_vec / np.linalg.norm(xline_vec)

    X_grid, Y_grid, nx, ny = compute_rotated_grid(
        grid_corners['c1'],
        grid_corners['c2'],
        grid_corners['c3'],
        grid_corners['c4'],
        dx, dy,
    )

    # Time axis
    nt = int((t_max_ms - t_min_ms) / dt_ms) + 1
    t_axis_ms = np.linspace(t_min_ms, t_max_ms, nt)

    if verbose:
        print(f"  Grid: {nx} IL x {ny} XL x {nt} samples")
        print(f"  Grid X: {X_grid.min():.1f} - {X_grid.max():.1f}")
        print(f"  Grid Y: {Y_grid.min():.1f} - {Y_grid.max():.1f}")
        print(f"  Time: {t_min_ms:.0f} - {t_max_ms:.0f} ms @ {dt_ms} ms")
        print(f"  Stretch mute: {stretch_mute_percent:.0f}%")

    # Initialize output arrays
    stack = np.zeros((nx, ny, nt), dtype=np.float64)
    fold = np.zeros((nx, ny, nt), dtype=np.int32)

    # Get velocity profile
    # For batch NMO, we use 1D velocity (center of grid for 3D models)
    is_1d = velocity_model.is_laterally_constant
    if is_1d:
        vrms_1d = velocity_model.get_vrms_1d(t_axis_ms)
        if verbose:
            print(f"  Using 1D velocity: {vrms_1d.min():.0f} - {vrms_1d.max():.0f} m/s")
    else:
        # For 3D velocity, extract 1D profile from center of grid for NMO
        # (NMO is relatively insensitive to lateral velocity variations)
        center_x = (X_grid.min() + X_grid.max()) / 2
        center_y = (Y_grid.min() + Y_grid.max()) / 2
        vrms_1d = velocity_model.get_vrms_at_point(center_x, center_y, t_axis_ms)
        if verbose:
            print(f"  Using 3D velocity (center profile): {vrms_1d.min():.0f} - {vrms_1d.max():.0f} m/s")

    # Process each offset bin
    total_traces = 0
    total_start = time.time()

    for bin_idx, bin_num in enumerate(bins):
        bin_start = time.time()

        if verbose:
            print(f"\n[{bin_idx+1}/{len(bins)}] Processing Bin {bin_num:02d}")

        try:
            traces, headers, attrs = load_bin_data(common_offset_dir, bin_num)
        except Exception as e:
            print(f"  ERROR loading bin {bin_num}: {e}")
            continue

        n_traces = len(traces)
        if n_traces == 0:
            print(f"  Skipping empty bin")
            continue

        # Get input time axis
        input_dt_ms = attrs.get('sample_rate_ms', attrs.get('dt_ms', dt_ms))
        input_start_ms = attrs.get('start_time_ms', 0.0)
        input_nt = traces.shape[1]
        input_t_axis = input_start_ms + np.arange(input_nt) * input_dt_ms

        # Get coordinate scalar (negative means divide)
        coord_scalar = 1.0
        if 'scalar_coord' in headers.columns:
            scalar_val = headers['scalar_coord'][0]
            if scalar_val < 0:
                coord_scalar = 1.0 / abs(scalar_val)
            elif scalar_val > 0:
                coord_scalar = scalar_val
            # scalar_val == 0 means no scaling

        # Get source/receiver coordinates
        src_x_col = 'source_x' if 'source_x' in headers.columns else 'SOU_X'
        src_y_col = 'source_y' if 'source_y' in headers.columns else 'SOU_Y'
        rec_x_col = 'receiver_x' if 'receiver_x' in headers.columns else 'REC_X'
        rec_y_col = 'receiver_y' if 'receiver_y' in headers.columns else 'REC_Y'

        src_x = headers[src_x_col].to_numpy().astype(np.float64) * coord_scalar
        src_y = headers[src_y_col].to_numpy().astype(np.float64) * coord_scalar
        rec_x = headers[rec_x_col].to_numpy().astype(np.float64) * coord_scalar
        rec_y = headers[rec_y_col].to_numpy().astype(np.float64) * coord_scalar

        # Compute CDP (midpoint)
        cdp_x = (src_x + rec_x) / 2.0
        cdp_y = (src_y + rec_y) / 2.0

        # Get offset from header (preferred) or compute from coordinates
        if 'offset' in headers.columns:
            offsets = np.abs(headers['offset'].to_numpy().astype(np.float64))
        elif 'OFFSET' in headers.columns:
            offsets = np.abs(headers['OFFSET'].to_numpy().astype(np.float64))
        else:
            # Compute from source-receiver distance
            offsets = np.sqrt((rec_x - src_x)**2 + (rec_y - src_y)**2)

        mean_offset = float(np.mean(offsets))
        if verbose:
            print(f"  Traces: {n_traces:,}, Mean offset: {mean_offset:.0f} m")
            print(f"  CDP X: {cdp_x.min():.1f} - {cdp_x.max():.1f}")
            print(f"  CDP Y: {cdp_y.min():.1f} - {cdp_y.max():.1f}")

        # === VECTORIZED PROCESSING ===

        # 1. Compute CDP bins for all traces at once
        i_bins, j_bins, valid = compute_cdp_bins_fast(
            cdp_x.astype(np.float64),
            cdp_y.astype(np.float64),
            float(c1[0]), float(c1[1]),
            float(inline_unit[0]), float(inline_unit[1]),
            float(xline_unit[0]), float(xline_unit[1]),
            dx, dy, nx, ny,
        )

        n_valid = np.sum(valid)
        if verbose:
            print(f"  Valid traces in grid: {n_valid:,} ({100*n_valid/n_traces:.1f}%)")

        if n_valid == 0:
            if verbose:
                print(f"  Skipping - no traces in grid")
            continue

        # 2. Resample input traces to output time axis if needed
        need_resample = (len(input_t_axis) != len(t_axis_ms) or
                         not np.allclose(input_t_axis[:min(len(input_t_axis), len(t_axis_ms))],
                                        t_axis_ms[:min(len(input_t_axis), len(t_axis_ms))]))
        if need_resample:
            if verbose:
                print(f"  Resampling traces {len(input_t_axis)} -> {nt} samples (parallel)...")
            # Batch resample using numba
            traces_to_process = resample_traces_fast(
                traces.astype(np.float32),
                input_t_axis.astype(np.float64),
                t_axis_ms.astype(np.float64),
            )
            processing_t_axis = t_axis_ms
        else:
            traces_to_process = traces
            processing_t_axis = input_t_axis

        # 3. Apply NMO correction (vectorized with Numba)
        if verbose:
            print(f"  Applying NMO correction (parallel)...")

        nmo_traces = apply_nmo_batch(
            traces_to_process.astype(np.float32),
            processing_t_axis.astype(np.float64),
            offsets.astype(np.float64),
            vrms_1d.astype(np.float64),
            stretch_mute_percent,
        )

        # 4. Accumulate to stack
        if verbose:
            print(f"  Accumulating to stack...")

        traces_added = accumulate_to_stack(
            stack, fold, nmo_traces, i_bins, j_bins, valid
        )

        total_traces += traces_added
        bin_elapsed = time.time() - bin_start

        if verbose:
            print(f"  Added {traces_added:,} traces in {bin_elapsed:.1f}s")

        # Cleanup
        del traces, headers
        gc.collect()

        # Save intermediate result every 2 bins for live preview
        if (bin_idx + 1) % 2 == 0 or bin_idx == len(bins) - 1:
            if verbose:
                print(f"  Saving intermediate result...")
            # Normalize current stack for preview
            with np.errstate(divide='ignore', invalid='ignore'):
                preview_stack = np.where(fold > 0, stack / fold, 0.0)

            # Save preview
            preview_path = output_path.parent / (output_path.stem + "_preview.zarr")
            preview_store = zarr.storage.LocalStore(str(preview_path))
            preview_z = zarr.create_array(
                store=preview_store,
                shape=preview_stack.shape,
                chunks=(min(64, nx), min(64, ny), nt),
                dtype=np.float32,
                overwrite=True,
            )
            preview_z[:] = preview_stack.astype(np.float32)
            preview_z.attrs['bins_processed'] = bin_idx + 1
            preview_z.attrs['total_traces'] = total_traces
            preview_z.attrs['dt_ms'] = dt_ms
            preview_z.attrs['stretch_mute_percent'] = stretch_mute_percent
            if verbose:
                print(f"  Preview saved to: {preview_path}")

    # Normalize by fold
    if verbose:
        print("\nNormalizing by fold...")

    with np.errstate(divide='ignore', invalid='ignore'):
        stack = np.where(fold > 0, stack / fold, 0.0)

    # Save to zarr
    if verbose:
        print(f"Saving to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.storage.LocalStore(str(output_path))
    z = zarr.create_array(
        store=store,
        shape=stack.shape,
        chunks=(min(64, nx), min(64, ny), nt),
        dtype=np.float32,
        overwrite=True,
    )
    z[:] = stack.astype(np.float32)

    # Set attributes
    z.attrs['nx'] = nx
    z.attrs['ny'] = ny
    z.attrs['nt'] = nt
    z.attrs['dx'] = dx
    z.attrs['dy'] = dy
    z.attrs['dt_ms'] = dt_ms
    z.attrs['t_min_ms'] = t_min_ms
    z.attrs['t_max_ms'] = t_max_ms
    z.attrs['x_min'] = float(X_grid.min())
    z.attrs['x_max'] = float(X_grid.max())
    z.attrs['y_min'] = float(Y_grid.min())
    z.attrs['y_max'] = float(Y_grid.max())
    z.attrs['stretch_mute_percent'] = stretch_mute_percent
    z.attrs['n_bins_stacked'] = len(bins)
    z.attrs['total_traces'] = total_traces
    z.attrs['created'] = datetime.now().isoformat()
    z.attrs['description'] = 'NMO-corrected pre-PSTM stack'

    # Also save fold volume
    fold_path = output_path.parent / (output_path.stem + "_fold.zarr")
    fold_store = zarr.storage.LocalStore(str(fold_path))
    fold_z = zarr.create_array(
        store=fold_store,
        shape=fold.shape,
        chunks=(min(64, nx), min(64, ny), nt),
        dtype=np.int32,
        overwrite=True,
    )
    fold_z[:] = fold
    fold_z.attrs.update(dict(z.attrs))
    fold_z.attrs['description'] = 'Fold map for NMO stack'

    total_elapsed = time.time() - total_start

    if verbose:
        max_fold = int(fold.max())
        mean_fold = float(fold[fold > 0].mean()) if np.any(fold > 0) else 0
        print(f"\nCompleted in {total_elapsed/60:.1f} minutes")
        print(f"Total traces stacked: {total_traces:,}")
        print(f"Max fold: {max_fold}, Mean fold: {mean_fold:.1f}")
        print(f"Output: {output_path}")
        print(f"Fold:   {fold_path}")

    return output_path


# =============================================================================
# Velocity Model Creation
# =============================================================================

def create_velocity_from_args(args) -> VelocityModel:
    """Create velocity model from command-line arguments."""

    if args.velocity:
        # 3D velocity cube
        return CubeVelocityModel(args.velocity)

    elif args.velocity_1d:
        # Parse "v0,k" format: V(t) = v0 + k * t
        parts = args.velocity_1d.split(',')
        v0 = float(parts[0])
        k = float(parts[1]) if len(parts) > 1 else 0.0
        return LinearVelocityModel(v0, k)

    elif args.velocity_table:
        # Parse "t1:v1,t2:v2,..." format
        pairs = args.velocity_table.split(',')
        times = []
        velocities = []
        for pair in pairs:
            t, v = pair.split(':')
            times.append(float(t))
            velocities.append(float(v))
        return TableVelocityModel(np.array(times), np.array(velocities))

    elif args.velocity_constant:
        return ConstantVelocityModel(args.velocity_constant)

    else:
        # Default: linear velocity 1500 + 0.5*t
        print("No velocity specified, using default: V(t) = 1500 + 0.5*t")
        return LinearVelocityModel(1500.0, 0.5)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create pre-PSTM NMO stack from common offset gathers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use 3D velocity cube
  python create_nmo_stack.py --velocity velocity.zarr

  # Use linear velocity V(t) = 1500 + 0.5*t
  python create_nmo_stack.py --velocity-1d 1500,0.5

  # Use velocity table
  python create_nmo_stack.py --velocity-table "0:1500,500:2000,1000:2500"

  # Adjust stretch mute
  python create_nmo_stack.py --stretch-mute 50  # 50% stretch mute (more aggressive)
  python create_nmo_stack.py --stretch-mute 20  # 20% stretch mute (less muting)

  # Process specific bins
  python create_nmo_stack.py --bins 0-20  # bins 0 to 20
  python create_nmo_stack.py --bins 0,5,10,15,20  # specific bins
        """
    )

    # Velocity options (mutually exclusive)
    vel_group = parser.add_mutually_exclusive_group()
    vel_group.add_argument(
        "--velocity", "-v",
        type=Path,
        help="Path to 3D velocity cube (zarr)",
    )
    vel_group.add_argument(
        "--velocity-1d",
        type=str,
        help="Linear velocity 'v0,k' where V(t) = v0 + k*t (m/s, m/s per second)",
    )
    vel_group.add_argument(
        "--velocity-table",
        type=str,
        help="Velocity table 't1:v1,t2:v2,...' (times in ms, velocities in m/s)",
    )
    vel_group.add_argument(
        "--velocity-constant",
        type=float,
        help="Constant velocity (m/s)",
    )

    # Stretch mute
    parser.add_argument(
        "--stretch-mute", "-s",
        type=float,
        default=DEFAULT_STRETCH_MUTE_PERCENT,
        help=f"Maximum allowed stretch in percent (default: {DEFAULT_STRETCH_MUTE_PERCENT})",
    )

    # Input/output
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=COMMON_OFFSET_DIR,
        help="Input directory with offset bins",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output zarr path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (if --output not specified)",
    )

    # Bin selection
    parser.add_argument(
        "--bins", "-b",
        type=str,
        default="all",
        help="Bins to process: 'all', '0-20', or '0,5,10' (default: all)",
    )

    # Grid parameters
    parser.add_argument("--dx", type=float, default=DX, help="Inline spacing (m)")
    parser.add_argument("--dy", type=float, default=DY, help="Crossline spacing (m)")
    parser.add_argument("--dt", type=float, default=DT_MS, help="Time sample (ms)")
    parser.add_argument("--t-min", type=float, default=0.0, help="Start time (ms)")
    parser.add_argument("--t-max", type=float, default=2000.0, help="End time (ms)")

    # Misc
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Get available bins
    available_bins = get_available_bins(args.input_dir)
    if not available_bins:
        print(f"ERROR: No offset bins found in {args.input_dir}")
        return 1

    # Parse bin selection
    if args.bins.lower() == "all":
        bins = available_bins
    elif "-" in args.bins and "," not in args.bins:
        start, end = map(int, args.bins.split("-"))
        bins = [b for b in range(start, end + 1) if b in available_bins]
    else:
        bins = [int(b.strip()) for b in args.bins.split(",") if int(b.strip()) in available_bins]

    if not bins:
        print("ERROR: No valid bins to process")
        return 1

    # Output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.output_dir / f"nmo_stack_stretch{int(args.stretch_mute)}.zarr"

    # Print configuration
    print("=" * 70)
    print("NMO Stack Creation")
    print("=" * 70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output path:      {output_path}")
    print(f"Bins to process:  {len(bins)} ({min(bins)}-{max(bins)})")
    print(f"Stretch mute:     {args.stretch_mute}%")
    print(f"Grid spacing:     {args.dx}m x {args.dy}m x {args.dt}ms")
    print(f"Time range:       {args.t_min} - {args.t_max} ms")
    print()

    if args.dry_run:
        print("DRY RUN - would process:")
        for bin_num in bins:
            bin_dir = args.input_dir / f"offset_bin_{bin_num:02d}"
            headers_path = bin_dir / "headers.parquet"
            if headers_path.exists():
                df = pl.read_parquet(headers_path)
                n_traces = len(df)
                offset = float(df['offset'].mean()) if 'offset' in df.columns else bin_num * 50 + 25
                print(f"  Bin {bin_num:02d}: {n_traces:,} traces, ~{offset:.0f}m offset")
        return 0

    # Create velocity model
    velocity_model = create_velocity_from_args(args)
    print(f"Velocity model: {type(velocity_model).__name__}")

    # Run
    try:
        create_nmo_stack(
            common_offset_dir=args.input_dir,
            velocity_model=velocity_model,
            output_path=output_path,
            bins=bins,
            grid_corners=GRID_CORNERS,
            dx=args.dx,
            dy=args.dy,
            dt_ms=args.dt,
            t_min_ms=args.t_min,
            t_max_ms=args.t_max,
            stretch_mute_percent=args.stretch_mute,
            verbose=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 70)
    print("Done!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
