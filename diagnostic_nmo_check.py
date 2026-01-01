#!/usr/bin/env python3
"""
Diagnostic Step 2: Check if Input Data is Already NMO-Corrected

If traces have pre-existing NMO correction, applying DSR would double-correct
and cause misalignment. This test:
1. Builds a CDP gather from common-offset traces near same midpoint
2. Checks if events are flat (NMO applied) or curved (raw)
3. Computes expected NMO curve and compares

Note: For common-offset data, we expect moveout to be minimal since
offset is constant. But we can still check if shallow/deep events align.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
import zarr
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent))

BIN_NUM = 10
COMMON_OFFSET_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
VELOCITY_PATH = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm_ilxl.zarr")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/diagnostic_qc")

GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}
NX, NY, NT = 511, 427, 1001
DT_MS = 2.0


def main():
    print("\n" + "="*70)
    print("STEP 2: Check if Input Data has Pre-existing NMO Correction")
    print("="*70)

    # Load data
    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{BIN_NUM:02d}"
    traces_store = zarr.open_array(bin_dir / "traces.zarr", mode='r')
    df = pl.read_parquet(bin_dir / "headers.parquet")

    transposed = traces_store.shape[0] < traces_store.shape[1]
    n_samples = traces_store.shape[0] if transposed else traces_store.shape[1]

    scalar = int(df['scalar_coord'][0])
    scale_factor = 1.0 / abs(scalar) if scalar < 0 else float(scalar) if scalar > 0 else 1.0

    sx = df['source_x'].to_numpy().astype(np.float64) * scale_factor
    sy = df['source_y'].to_numpy().astype(np.float64) * scale_factor
    rx = df['receiver_x'].to_numpy().astype(np.float64) * scale_factor
    ry = df['receiver_y'].to_numpy().astype(np.float64) * scale_factor
    mx = (sx + rx) / 2.0
    my = (sy + ry) / 2.0
    offsets = df['offset'].to_numpy()
    bin_trace_idx = df['bin_trace_idx'].to_numpy()

    # Load velocity
    vel_store = zarr.open(VELOCITY_PATH, mode='r')
    velocity = np.asarray(vel_store)

    # =========================================================================
    # Test 2.1: Build a gather from traces at similar midpoint locations
    # =========================================================================
    print(f"\n[2.1] Building gather from traces at similar midpoint...")

    # Choose a center point (grid center)
    c1 = np.array(GRID_CORNERS['c1'])
    c2 = np.array(GRID_CORNERS['c2'])
    c4 = np.array(GRID_CORNERS['c4'])
    il_dir = (c2 - c1) / (NX - 1)
    xl_dir = (c4 - c1) / (NY - 1)

    test_il, test_xl = 256, 214
    center_pt = c1 + (test_il - 1) * il_dir + (test_xl - 1) * xl_dir
    cx, cy = center_pt

    # Find traces within 100m of this midpoint
    midpoint_dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
    near_indices = np.where(midpoint_dist < 100.0)[0]

    print(f"      Center point: ({cx:.1f}, {cy:.1f})")
    print(f"      Traces within 100m: {len(near_indices)}")

    if len(near_indices) < 10:
        print("      Not enough traces for analysis. Expanding search radius...")
        near_indices = np.where(midpoint_dist < 500.0)[0]
        print(f"      Traces within 500m: {len(near_indices)}")

    # Sample up to 50 traces
    n_gather = min(50, len(near_indices))
    sample_indices = np.random.choice(near_indices, n_gather, replace=False)

    # Sort by azimuth for better visualization
    azimuths = df['sr_azim'].to_numpy() if 'sr_azim' in df.columns else None
    if azimuths is not None:
        sort_order = np.argsort(azimuths[sample_indices])
        sample_indices = sample_indices[sort_order]

    # Load traces
    gather = np.zeros((n_gather, n_samples), dtype=np.float32)
    gather_offsets = np.zeros(n_gather)
    gather_azimuths = np.zeros(n_gather)

    for i, header_idx in enumerate(sample_indices):
        storage_idx = bin_trace_idx[header_idx]
        if transposed:
            gather[i, :] = np.asarray(traces_store[:, storage_idx])
        else:
            gather[i, :] = np.asarray(traces_store[storage_idx, :])
        gather_offsets[i] = offsets[header_idx]
        if azimuths is not None:
            gather_azimuths[i] = azimuths[header_idx]

    print(f"      Loaded {n_gather} traces")
    print(f"      Offset range: {gather_offsets.min():.0f} - {gather_offsets.max():.0f} m")
    if azimuths is not None:
        print(f"      Azimuth range: {gather_azimuths.min():.0f} - {gather_azimuths.max():.0f} deg")

    # =========================================================================
    # Test 2.2: Check trace-to-trace correlation at different time windows
    # =========================================================================
    print(f"\n[2.2] Computing trace-to-trace correlation...")

    # Reference trace (middle of gather)
    ref_trace = gather[n_gather // 2, :]

    # Time windows to test
    windows = [
        (200, 400, "Shallow (400-800ms)"),
        (400, 600, "Mid (800-1200ms)"),
        (600, 800, "Deep (1200-1600ms)"),
    ]

    print(f"\n      Correlation with reference trace (trace #{n_gather//2}):")
    for start_idx, end_idx, name in windows:
        correlations = []
        for i in range(n_gather):
            if i == n_gather // 2:
                continue
            ref_win = ref_trace[start_idx:end_idx]
            test_win = gather[i, start_idx:end_idx]

            # Normalize
            ref_norm = ref_win / (np.std(ref_win) + 1e-10)
            test_norm = test_win / (np.std(test_win) + 1e-10)

            corr = np.corrcoef(ref_norm, test_norm)[0, 1]
            correlations.append(corr)

        mean_corr = np.nanmean(correlations)
        std_corr = np.nanstd(correlations)
        print(f"      {name}: mean={mean_corr:.3f}, std={std_corr:.3f}")

    # =========================================================================
    # Test 2.3: Check for systematic time shifts between traces
    # =========================================================================
    print(f"\n[2.3] Checking for systematic time shifts...")

    # Cross-correlate to find time shifts
    time_shifts = []
    ref_trace_smooth = gaussian_filter1d(ref_trace, sigma=2)

    for i in range(n_gather):
        if i == n_gather // 2:
            time_shifts.append(0)
            continue

        test_trace = gaussian_filter1d(gather[i, :], sigma=2)

        # Cross-correlation in a window around t=1000ms
        win_start, win_end = 400, 600
        ref_win = ref_trace_smooth[win_start:win_end]
        test_win = test_trace[win_start-50:win_end+50]  # Larger window for search

        xcorr = correlate(ref_win, test_win, mode='valid')
        shift = np.argmax(xcorr) - 50  # Relative to expected position
        time_shifts.append(shift)

    time_shifts = np.array(time_shifts) * DT_MS  # Convert to ms
    print(f"      Time shift range: {time_shifts.min():.1f} to {time_shifts.max():.1f} ms")
    print(f"      Time shift std: {time_shifts.std():.1f} ms")

    # Check if shifts correlate with azimuth
    if azimuths is not None:
        valid_mask = ~np.isnan(time_shifts) & ~np.isnan(gather_azimuths)
        if valid_mask.sum() > 5:
            corr_az_shift = np.corrcoef(gather_azimuths[valid_mask], time_shifts[valid_mask])[0, 1]
            print(f"      Correlation (azimuth vs time shift): {corr_az_shift:.3f}")

    # =========================================================================
    # Test 2.4: Compare with expected NMO moveout
    # =========================================================================
    print(f"\n[2.4] Comparing with expected NMO moveout...")

    # Get velocity at this location
    v_at_point = velocity[test_il-1, test_xl-1, :]

    # For common-offset data, all traces have similar offset (~500-550m)
    mean_offset = gather_offsets.mean()
    print(f"      Mean offset: {mean_offset:.0f} m")

    # Expected NMO time shift at different t0
    print(f"\n      Expected NMO moveout for offset={mean_offset:.0f}m:")
    print(f"      {'t0 (ms)':<10} {'V (m/s)':<10} {'t_nmo (ms)':<12} {'Moveout (ms)':<12}")

    for t0_ms in [500, 750, 1000, 1250, 1500]:
        t_idx = int(t0_ms / DT_MS)
        v = v_at_point[t_idx]
        t0_s = t0_ms / 1000.0

        # NMO formula
        t_nmo = np.sqrt(t0_s**2 + (mean_offset / v)**2)
        moveout = (t_nmo - t0_s) * 1000  # in ms

        print(f"      {t0_ms:<10} {v:<10.0f} {t_nmo*1000:<12.2f} {moveout:<12.2f}")

    # =========================================================================
    # Test 2.5: Visual inspection - are events flat or curved?
    # =========================================================================
    print(f"\n[2.5] Creating visual comparison...")

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'NMO Check - Bin {BIN_NUM}, Location IL={test_il} XL={test_xl}',
                fontsize=14, fontweight='bold')

    t_axis = np.arange(n_samples) * DT_MS

    # Raw gather (sorted by azimuth)
    ax = axes[0, 0]
    vmax = np.percentile(np.abs(gather), 98)
    if azimuths is not None:
        extent = [gather_azimuths.min(), gather_azimuths.max(), t_axis[-1], t_axis[0]]
        ax.imshow(gather.T, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_xlabel('Azimuth (degrees)')
    else:
        ax.imshow(gather.T, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
                 extent=[0, n_gather, t_axis[-1], t_axis[0]])
        ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Raw Gather (sorted by azimuth)')

    # Zoom on shallow events
    ax = axes[0, 1]
    t_start, t_end = 400, 800
    gather_zoom = gather[:, int(t_start/DT_MS):int(t_end/DT_MS)]
    vmax_zoom = np.percentile(np.abs(gather_zoom), 98)
    ax.imshow(gather_zoom.T, aspect='auto', cmap='gray', vmin=-vmax_zoom, vmax=vmax_zoom,
              extent=[0, n_gather, t_end, t_start])
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Zoom: 400-800ms (look for flat/curved events)')

    # Zoom on mid events
    ax = axes[0, 2]
    t_start, t_end = 800, 1200
    gather_zoom = gather[:, int(t_start/DT_MS):int(t_end/DT_MS)]
    vmax_zoom = np.percentile(np.abs(gather_zoom), 98)
    ax.imshow(gather_zoom.T, aspect='auto', cmap='gray', vmin=-vmax_zoom, vmax=vmax_zoom,
              extent=[0, n_gather, t_end, t_start])
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Zoom: 800-1200ms (look for flat/curved events)')

    # Time shifts vs azimuth
    ax = axes[1, 0]
    if azimuths is not None:
        ax.scatter(gather_azimuths, time_shifts, s=30, alpha=0.7)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Time Shift (ms)')
        ax.set_title('Time Shift vs Azimuth')
        ax.grid(True, alpha=0.3)

    # Stack of all traces
    ax = axes[1, 1]
    stack = gather.sum(axis=0) / n_gather
    ax.plot(t_axis, stack, 'b-', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Stack of {n_gather} traces')
    ax.set_xlim([0, 2000])

    # Velocity profile
    ax = axes[1, 2]
    ax.plot(t_axis[:len(v_at_point)], v_at_point, 'b-')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity at this location')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "step2_nmo_check.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n      Saved: {fig_path}")

    # =========================================================================
    # Test 2.6: Check trace headers for NMO flag or processing history
    # =========================================================================
    print(f"\n[2.6] Checking headers for processing indicators...")

    # Look for columns that might indicate processing state
    processing_cols = ['nmo_applied', 'processed', 'correction', 'static']
    for col in processing_cols:
        matches = [c for c in df.columns if col.lower() in c.lower()]
        if matches:
            print(f"      Found columns matching '{col}': {matches}")

    # Check sample_interval consistency (different for NMO'd data?)
    if 'sample_interval' in df.columns:
        sample_intervals = df['sample_interval'].unique().to_numpy()
        print(f"      Sample intervals in headers: {sample_intervals}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("NMO CHECK SUMMARY")
    print("="*70)

    time_shift_range = time_shifts.max() - time_shifts.min()
    expected_moveout = (mean_offset / v_at_point[500])**2 / (2 * 1.0) * 1000  # Approximate at 1s

    print(f"""
  Time shift analysis:
    Observed time shift range: {time_shift_range:.1f} ms
    Time shift std: {time_shifts.std():.1f} ms

  Expected moveout for offset={mean_offset:.0f}m:
    At t0=1000ms with V={v_at_point[500]:.0f}m/s: ~{expected_moveout:.1f} ms

  Interpretation:
    - If events appear FLAT in the gather → Data may be NMO-corrected
    - If events show CURVATURE → Data is likely raw (not NMO-corrected)
    - If time shifts correlate with AZIMUTH → Possible coordinate issue

  VISUAL INSPECTION REQUIRED: Check step2_nmo_check.png
    - Look at the gather images for flat vs curved events
    - For common-offset data, minimal moveout is expected
    - Random time shifts suggest geometry or processing issues
""")


if __name__ == "__main__":
    main()
