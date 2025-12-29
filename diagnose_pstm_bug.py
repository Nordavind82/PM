#!/usr/bin/env python3
"""
PSTM Bug Diagnostic Script.

Investigates potential issues with travel time calculation, velocity handling,
and sample rate consistency.

Run this to identify the root cause of residual moveout in CIGs.
"""

import numpy as np
import polars as pl
import zarr
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

MIGRATION_DIR = Path("/Volumes/AO_DISK/PSTM_common_offset")
INPUT_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
VELOCITY_PATH = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm.zarr")

BIN_NUM = 10  # Bin to analyze

# =============================================================================
# Load Data
# =============================================================================

def load_migration_metadata(bin_num: int):
    """Load migration output metadata."""
    zarr_path = MIGRATION_DIR / f"migration_bin_{bin_num:02d}" / "migrated_stack.zarr"
    store = zarr.storage.LocalStore(str(zarr_path))
    z = zarr.open_array(store=store, mode='r')
    return dict(z.attrs), z.shape

def load_input_metadata(bin_num: int):
    """Load input trace metadata."""
    traces_path = INPUT_DIR / f"offset_bin_{bin_num:02d}" / "traces.zarr"
    headers_path = INPUT_DIR / f"offset_bin_{bin_num:02d}" / "headers.parquet"

    store = zarr.storage.LocalStore(str(traces_path))
    z = zarr.open_array(store=store, mode='r')
    trace_attrs = dict(z.attrs)

    headers = pl.read_parquet(headers_path)

    return trace_attrs, z.shape, headers

def load_velocity():
    """Load velocity model."""
    store = zarr.storage.LocalStore(str(VELOCITY_PATH))
    z = zarr.open_array(store=store, mode='r')
    return dict(z.attrs), np.array(z)

# =============================================================================
# DSR Travel Time Calculation (Reference Implementation)
# =============================================================================

def compute_dsr_traveltime(t0_s, ds, dr, velocity):
    """
    Compute DSR travel time.

    Args:
        t0_s: Zero-offset two-way time (seconds)
        ds: Distance from image point to source (meters)
        dr: Distance from image point to receiver (meters)
        velocity: RMS velocity at time t0 (m/s)

    Returns:
        Total travel time (seconds)
    """
    t0_half = t0_s / 2.0
    t0_half_sq = t0_half ** 2
    inv_v_sq = 1.0 / (velocity ** 2)

    t_src = np.sqrt(t0_half_sq + ds**2 * inv_v_sq)
    t_rec = np.sqrt(t0_half_sq + dr**2 * inv_v_sq)

    return t_src + t_rec

def compute_nmo_traveltime(t0_s, offset, velocity):
    """
    Compute NMO travel time.

    Args:
        t0_s: Zero-offset two-way time (seconds)
        offset: Source-receiver offset (meters)
        velocity: NMO velocity (m/s)

    Returns:
        NMO travel time (seconds)
    """
    return np.sqrt(t0_s**2 + (offset/velocity)**2)

# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_sample_rates():
    """Check consistency of sample rates."""
    print("\n" + "="*70)
    print("SAMPLE RATE ANALYSIS")
    print("="*70)

    # Input traces
    trace_attrs, trace_shape, headers = load_input_metadata(BIN_NUM)
    input_dt = trace_attrs.get('sample_rate_ms', 'UNKNOWN')
    input_ns = trace_shape[0] if trace_shape[0] < trace_shape[1] else trace_shape[1]

    # Migration output
    mig_attrs, mig_shape = load_migration_metadata(BIN_NUM)
    output_dt = mig_attrs.get('dt_ms', 'UNKNOWN')
    output_nt = mig_shape[2]

    print(f"\nINPUT TRACES:")
    print(f"  Sample rate: {input_dt} ms")
    print(f"  Samples: {input_ns}")
    print(f"  Time range: 0 - {input_ns * float(input_dt):.0f} ms")

    print(f"\nOUTPUT GRID:")
    print(f"  Sample rate: {output_dt} ms")
    print(f"  Samples: {output_nt}")
    print(f"  t_min: {mig_attrs.get('t_min_ms', 0)} ms")
    print(f"  t_max: {mig_attrs.get('t_max_ms', 'N/A')} ms")

    if input_dt != output_dt:
        print(f"\n⚠️  WARNING: Sample rate mismatch! Input={input_dt}ms, Output={output_dt}ms")
    else:
        print(f"\n✓ Sample rates match: {input_dt} ms")

    return float(input_dt), float(output_dt)

def analyze_velocity():
    """Check velocity model alignment."""
    print("\n" + "="*70)
    print("VELOCITY MODEL ANALYSIS")
    print("="*70)

    velo_attrs, velo_data = load_velocity()
    mig_attrs, _ = load_migration_metadata(BIN_NUM)

    # Velocity time axis
    velo_t = np.array(velo_attrs.get('t_axis_ms', []))

    print(f"\nVELOCITY CUBE:")
    print(f"  Shape: {velo_data.shape}")
    print(f"  Time samples: {len(velo_t)}")
    print(f"  Time range: {velo_t[0]:.0f} - {velo_t[-1]:.0f} ms")
    print(f"  dt: {velo_t[1] - velo_t[0]:.0f} ms")
    print(f"  Velocity range: {velo_data.min():.0f} - {velo_data.max():.0f} m/s")

    # Check alignment with output grid
    output_t_min = mig_attrs.get('t_min_ms', 0)
    output_t_max = mig_attrs.get('t_max_ms', 2000)

    print(f"\nOUTPUT GRID TIME:")
    print(f"  Range: {output_t_min} - {output_t_max} ms")

    if output_t_max > velo_t[-1]:
        print(f"\n⚠️  WARNING: Output extends beyond velocity ({output_t_max} > {velo_t[-1]})")

    # Show velocity at key times
    print(f"\nVELOCITY AT KEY TIMES (center pillar):")
    cx, cy = velo_data.shape[0]//2, velo_data.shape[1]//2
    for t_ms in [0, 200, 500, 1000, 1500, 2000]:
        idx = np.searchsorted(velo_t, t_ms)
        idx = min(idx, len(velo_t)-1)
        v = velo_data[cx, cy, idx]
        print(f"  t={t_ms:4d} ms: v={v:.0f} m/s")

    return velo_attrs, velo_data

def analyze_travel_times():
    """Compare DSR vs NMO travel times."""
    print("\n" + "="*70)
    print("TRAVEL TIME ANALYSIS")
    print("="*70)

    trace_attrs, _, headers = load_input_metadata(BIN_NUM)
    velo_attrs, velo_data = load_velocity()

    # Get a sample trace
    sample_idx = len(headers) // 2
    row = headers.row(sample_idx)

    # Extract geometry
    sx = row[headers.columns.index('source_x')]
    sy = row[headers.columns.index('source_y')]
    rx = row[headers.columns.index('receiver_x')]
    ry = row[headers.columns.index('receiver_y')]
    offset = row[headers.columns.index('offset')]

    mx = (sx + rx) / 2
    my = (sy + ry) / 2

    print(f"\nSAMPLE TRACE GEOMETRY:")
    print(f"  Source: ({sx:.2f}, {sy:.2f})")
    print(f"  Receiver: ({rx:.2f}, {ry:.2f})")
    print(f"  Midpoint: ({mx:.2f}, {my:.2f})")
    print(f"  Offset: {offset:.0f} m")

    # Get velocity (center pillar)
    velo_t = np.array(velo_attrs.get('t_axis_ms', []))
    cx, cy = velo_data.shape[0]//2, velo_data.shape[1]//2
    vrms_center = velo_data[cx, cy, :]

    # Compare travel times at different output times
    print(f"\nTRAVEL TIME COMPARISON:")
    print(f"  (Assuming image point = midpoint)")
    print(f"  {'t0 (ms)':<10} {'v (m/s)':<10} {'DSR (ms)':<12} {'NMO (ms)':<12} {'Diff (ms)':<10}")
    print(f"  {'-'*54}")

    for t0_ms in [200, 400, 600, 800, 1000, 1200, 1500, 2000]:
        # Interpolate velocity
        v = np.interp(t0_ms, velo_t, vrms_center)

        t0_s = t0_ms / 1000.0

        # For image point = midpoint: ds = dr = offset/2
        ds = dr = offset / 2

        t_dsr = compute_dsr_traveltime(t0_s, ds, dr, v) * 1000  # Convert to ms
        t_nmo = compute_nmo_traveltime(t0_s, offset, v) * 1000

        diff = t_dsr - t_nmo

        print(f"  {t0_ms:<10} {v:<10.0f} {t_dsr:<12.1f} {t_nmo:<12.1f} {diff:<10.2f}")

    # Show expected NMO moveout
    print(f"\nEXPECTED NMO MOVEOUT (t_nmo - t0):")
    for t0_ms in [200, 500, 1000, 1500, 2000]:
        v = np.interp(t0_ms, velo_t, vrms_center)
        t0_s = t0_ms / 1000.0
        t_nmo = compute_nmo_traveltime(t0_s, offset, v) * 1000
        moveout = t_nmo - t0_ms
        print(f"  t0={t0_ms:4d} ms, v={v:.0f} m/s: moveout = {moveout:.1f} ms")

def analyze_coordinate_units():
    """Check coordinate units and scaling."""
    print("\n" + "="*70)
    print("COORDINATE UNITS ANALYSIS")
    print("="*70)

    trace_attrs, _, headers = load_input_metadata(BIN_NUM)
    mig_attrs, _ = load_migration_metadata(BIN_NUM)

    # Check header coordinate ranges
    print(f"\nINPUT TRACE COORDINATES (from headers):")
    for col in ['source_x', 'source_y', 'receiver_x', 'receiver_y', 'offset']:
        if col in headers.columns:
            vals = headers[col]
            print(f"  {col}: {vals.min():.2f} - {vals.max():.2f}")

    print(f"\nMIGRATION OUTPUT COORDINATES:")
    print(f"  x_min: {mig_attrs.get('x_min', 'N/A')}")
    print(f"  x_max: {mig_attrs.get('x_max', 'N/A')}")
    print(f"  y_min: {mig_attrs.get('y_min', 'N/A')}")
    print(f"  y_max: {mig_attrs.get('y_max', 'N/A')}")

    # Check if coordinates are in meters or other units
    offset_vals = headers['offset'] if 'offset' in headers.columns else None
    if offset_vals is not None:
        mean_offset = offset_vals.mean()
        print(f"\nOFFSET ANALYSIS:")
        print(f"  Mean offset: {mean_offset:.1f} m")
        print(f"  Expected for bin {BIN_NUM}: {BIN_NUM*50 + 25:.0f} m (if 50m bins)")

        if abs(mean_offset - (BIN_NUM*50 + 25)) > 100:
            print(f"  ⚠️  WARNING: Offset doesn't match expected bin center!")

def analyze_cig_moveout():
    """Analyze CIG to measure actual residual moveout."""
    print("\n" + "="*70)
    print("CIG MOVEOUT ANALYSIS")
    print("="*70)

    # Load a few offset bins to check moveout
    bins_to_check = [5, 10, 15, 20, 25]
    traces = []
    offsets = []

    print(f"\nLoading CIG at survey center...")

    mig_attrs, mig_shape = load_migration_metadata(bins_to_check[0])
    nx, ny, nt = mig_shape

    # Center location
    ix, iy = nx // 2, ny // 2

    for bin_num in bins_to_check:
        zarr_path = MIGRATION_DIR / f"migration_bin_{bin_num:02d}" / "migrated_stack.zarr"
        if not zarr_path.exists():
            continue

        store = zarr.storage.LocalStore(str(zarr_path))
        z = zarr.open_array(store=store, mode='r')

        trace = z[ix, iy, :]
        traces.append(trace)

        # Get offset from input headers
        headers_path = INPUT_DIR / f"offset_bin_{bin_num:02d}" / "headers.parquet"
        if headers_path.exists():
            headers = pl.read_parquet(headers_path)
            mean_offset = headers['offset'].mean()
            offsets.append(mean_offset)

    if len(traces) < 2:
        print("  Not enough bins to analyze moveout")
        return

    traces = np.array(traces)
    offsets = np.array(offsets)

    print(f"  Location: IL={ix+1}, XL={iy+1}")
    print(f"  Offsets: {offsets}")

    # Find a strong event and track its moveout
    dt_ms = mig_attrs.get('dt_ms', 2.0)

    # Look for maximum amplitude in near-offset trace
    near_trace = traces[0]
    t_peak_near = np.argmax(np.abs(near_trace[100:500])) + 100
    t_peak_ms = t_peak_near * dt_ms

    print(f"\n  Strong event at t={t_peak_ms:.0f} ms in near-offset bin")

    # Track this event across offsets using cross-correlation
    print(f"\n  MOVEOUT ANALYSIS:")
    print(f"  {'Bin':<6} {'Offset (m)':<12} {'t_peak (ms)':<12} {'Moveout (ms)':<12}")
    print(f"  {'-'*42}")

    window = 50
    for i, (trace, offset) in enumerate(zip(traces, offsets)):
        # Find peak in window around expected position
        search_start = max(0, t_peak_near - window)
        search_end = min(len(trace), t_peak_near + window)

        local_peak = np.argmax(np.abs(trace[search_start:search_end])) + search_start
        local_t_ms = local_peak * dt_ms
        moveout = local_t_ms - t_peak_ms

        print(f"  {bins_to_check[i]:<6} {offset:<12.0f} {local_t_ms:<12.0f} {moveout:<12.1f}")

# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("PSTM BUG DIAGNOSTIC REPORT")
    print("="*70)
    print(f"Analyzing bin {BIN_NUM}")

    analyze_sample_rates()
    analyze_velocity()
    analyze_coordinate_units()
    analyze_travel_times()
    analyze_cig_moveout()

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
