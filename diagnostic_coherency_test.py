#!/usr/bin/env python3
"""
PSTM Coherency Diagnostic - Test if traces stack coherently.

The amplitude diagnostic showed traces have random amplitudes that cancel.
This test checks:
1. If there's coherent signal at specific horizons
2. How traces align after DSR correction
3. Semblance/coherency measures
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
import zarr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# Configuration
# =============================================================================

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

TEST_IL, TEST_XL = 256, 214
TEST_IX, TEST_IY = TEST_IL - 1, TEST_XL - 1


def compute_semblance(gather, window_samples=25):
    """Compute semblance (coherency measure) for a gather.

    Semblance = (sum of traces)^2 / (N * sum of traces^2)
    High semblance (near 1) indicates coherent signal.
    Low semblance (near 0) indicates random noise.
    """
    n_traces, n_samples = gather.shape
    semblance = np.zeros(n_samples)

    half_win = window_samples // 2

    for i in range(half_win, n_samples - half_win):
        win = gather[:, i-half_win:i+half_win+1]

        # Sum of traces squared
        stack = win.sum(axis=0)
        num = np.sum(stack ** 2)

        # Sum of squared traces
        denom = n_traces * np.sum(win ** 2)

        if denom > 0:
            semblance[i] = num / denom

    return semblance


def main():
    print("\n" + "="*70)
    print("COHERENCY DIAGNOSTIC - Testing trace alignment and signal coherency")
    print("="*70)

    # Load data
    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{BIN_NUM:02d}"
    traces_store = zarr.open_array(bin_dir / "traces.zarr", mode='r')
    df = pl.read_parquet(bin_dir / "headers.parquet")

    scalar = int(df['scalar_coord'][0])
    scale_factor = 1.0 / abs(scalar) if scalar < 0 else float(scalar) if scalar > 0 else 1.0

    sx = df['source_x'].to_numpy().astype(np.float64) * scale_factor
    sy = df['source_y'].to_numpy().astype(np.float64) * scale_factor
    rx = df['receiver_x'].to_numpy().astype(np.float64) * scale_factor
    ry = df['receiver_y'].to_numpy().astype(np.float64) * scale_factor
    mx = (sx + rx) / 2.0
    my = (sy + ry) / 2.0

    bin_trace_idx = df['bin_trace_idx'].to_numpy()
    azimuths = df['sr_azim'].to_numpy() if 'sr_azim' in df.columns else None

    # Compute output point
    c1 = np.array(GRID_CORNERS['c1'])
    c2 = np.array(GRID_CORNERS['c2'])
    c4 = np.array(GRID_CORNERS['c4'])
    il_dir = (c2 - c1) / (NX - 1)
    xl_dir = (c4 - c1) / (NY - 1)
    output_pt = c1 + TEST_IX * il_dir + TEST_IY * xl_dir
    ox, oy = output_pt

    # Load velocity
    vel_store = zarr.open(VELOCITY_PATH, mode='r')
    velocity = np.asarray(vel_store)

    # Find traces near output point (within small aperture for focused test)
    distances = np.sqrt((mx - ox)**2 + (my - oy)**2)

    # =========================================================================
    # Test A: Compare trace alignment before and after DSR correction
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST A: Trace Alignment Before/After DSR Correction")
    print("-"*70)

    # Select traces within 500m (closer traces should have better alignment)
    test_aperture = 500.0
    within_aperture = distances <= test_aperture
    indices = np.where(within_aperture)[0]
    n_traces = min(100, len(indices))  # Use 100 traces for visualization

    print(f"  Traces within {test_aperture}m: {len(indices):,}")
    print(f"  Using {n_traces} traces for analysis")

    if n_traces < 10:
        print("  ERROR: Not enough traces for analysis")
        return

    # Sample traces
    sample_indices = np.random.choice(indices, n_traces, replace=False)

    # Load trace data
    n_samples = traces_store.shape[0] if traces_store.shape[0] < traces_store.shape[1] else traces_store.shape[1]
    raw_gather = np.zeros((n_traces, n_samples), dtype=np.float32)
    migrated_gather = np.zeros((n_traces, NT), dtype=np.float32)

    trace_azimuths = []
    trace_distances = []

    for i, trace_idx in enumerate(sample_indices):
        storage_idx = bin_trace_idx[trace_idx]

        # Load trace
        if traces_store.shape[0] < traces_store.shape[1]:
            trace_data = np.asarray(traces_store[:, storage_idx]).astype(np.float32)
        else:
            trace_data = np.asarray(traces_store[storage_idx, :]).astype(np.float32)

        raw_gather[i, :] = trace_data
        trace_distances.append(distances[trace_idx])
        if azimuths is not None:
            trace_azimuths.append(azimuths[trace_idx])

        # Apply DSR correction (migrate to output time)
        trace_sx, trace_sy = sx[trace_idx], sy[trace_idx]
        trace_rx, trace_ry = rx[trace_idx], ry[trace_idx]

        ds2 = (ox - trace_sx)**2 + (oy - trace_sy)**2
        dr2 = (ox - trace_rx)**2 + (oy - trace_ry)**2

        for it in range(NT):
            t0_s = it * DT_MS / 1000.0
            if t0_s < 0.1:
                continue

            v = velocity[TEST_IX, TEST_IY, it]
            t0_half_sq = (t0_s / 2)**2
            inv_v_sq = 1 / (v * v)

            t_travel = np.sqrt(t0_half_sq + ds2 * inv_v_sq) + np.sqrt(t0_half_sq + dr2 * inv_v_sq)
            sample_idx = (t_travel * 1000.0) / DT_MS

            if 0 <= sample_idx < len(trace_data) - 1:
                idx0 = int(sample_idx)
                frac = sample_idx - idx0
                migrated_gather[i, it] = trace_data[idx0] * (1-frac) + trace_data[idx0+1] * frac

    # =========================================================================
    # Test B: Compute Semblance
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST B: Semblance Analysis")
    print("-"*70)

    semblance_raw = compute_semblance(raw_gather)
    semblance_migrated = compute_semblance(migrated_gather)

    t_axis = np.arange(NT) * DT_MS

    # Find peaks in semblance
    from scipy.signal import find_peaks
    peaks_raw, _ = find_peaks(semblance_raw, height=0.1, distance=50)
    peaks_mig, _ = find_peaks(semblance_migrated, height=0.1, distance=50)

    print(f"  Raw gather semblance:")
    print(f"    Mean: {semblance_raw.mean():.4f}")
    print(f"    Max: {semblance_raw.max():.4f}")
    print(f"    High coherency events (>0.1): {len(peaks_raw)}")

    print(f"  Migrated gather semblance:")
    print(f"    Mean: {semblance_migrated.mean():.4f}")
    print(f"    Max: {semblance_migrated.max():.4f}")
    print(f"    High coherency events (>0.1): {len(peaks_mig)}")

    # =========================================================================
    # Test C: Azimuth-sorted Analysis
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST C: Azimuth-Sorted Gather")
    print("-"*70)

    if trace_azimuths:
        # Sort by azimuth
        az_order = np.argsort(trace_azimuths)
        az_sorted_gather = migrated_gather[az_order, :]
        sorted_azimuths = np.array(trace_azimuths)[az_order]

        print(f"  Azimuth range: {min(trace_azimuths):.1f} - {max(trace_azimuths):.1f} degrees")

        # Check for azimuth-dependent time shifts
        # Look at a specific time window around t=1000ms
        t_center = 500  # index for t=1000ms
        t_win = 25  # +/- 50ms

        window_data = az_sorted_gather[:, t_center-t_win:t_center+t_win+1]

        # Find peak position for each trace
        peak_positions = []
        for i in range(n_traces):
            trace_win = window_data[i, :]
            if np.abs(trace_win).max() > 0:
                peak_idx = np.argmax(np.abs(trace_win))
                peak_positions.append(peak_idx - t_win)  # Offset from center
            else:
                peak_positions.append(np.nan)

        valid_peaks = [p for p in peak_positions if not np.isnan(p)]
        if valid_peaks:
            print(f"  Peak time shifts around t=1000ms:")
            print(f"    Mean shift: {np.mean(valid_peaks) * DT_MS:.2f} ms")
            print(f"    Std shift: {np.std(valid_peaks) * DT_MS:.2f} ms")
            print(f"    Range: {(max(valid_peaks) - min(valid_peaks)) * DT_MS:.2f} ms")

    # =========================================================================
    # Test D: Stack Improvement with Trace Count
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST D: Stack Quality vs Number of Traces")
    print("-"*70)

    # Progressive stacking
    stack_sizes = [5, 10, 20, 50, 100]
    snr_results = []

    for n in stack_sizes:
        if n > n_traces:
            break
        subset = migrated_gather[:n, :]
        stack = subset.sum(axis=0)

        # Estimate SNR from signal window vs noise window
        signal_win = stack[400:600]  # Around t=800-1200ms
        noise_win = stack[50:150]    # Shallow (mostly noise)

        signal_power = np.sqrt(np.mean(signal_win**2))
        noise_power = np.sqrt(np.mean(noise_win**2))
        snr = signal_power / (noise_power + 1e-10)

        snr_results.append({'n': n, 'snr': snr, 'signal': signal_power, 'noise': noise_power})
        print(f"  n={n:3d}: SNR={snr:.2f}, signal={signal_power:.2f}, noise={noise_power:.2f}")

    # =========================================================================
    # Create QC Figure
    # =========================================================================
    print("\n" + "-"*70)
    print("Creating QC Figure")
    print("-"*70)

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)
    fig.suptitle(f'Coherency Diagnostic - Bin {BIN_NUM}, Test Point IL={TEST_IL} XL={TEST_XL}',
                fontsize=14, fontweight='bold')

    # Raw gather
    ax = fig.add_subplot(gs[0, 0])
    t_raw = np.arange(n_samples) * DT_MS
    vmax = np.percentile(np.abs(raw_gather), 99)
    ax.imshow(raw_gather.T, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
              extent=[0, n_traces, t_raw[-1], t_raw[0]])
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Raw Gather (before DSR)')

    # Migrated gather
    ax = fig.add_subplot(gs[0, 1])
    vmax = np.percentile(np.abs(migrated_gather), 99)
    ax.imshow(migrated_gather.T, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
              extent=[0, n_traces, t_axis[-1], t_axis[0]])
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Migrated Gather (after DSR)')

    # Azimuth-sorted gather
    ax = fig.add_subplot(gs[0, 2])
    if trace_azimuths:
        vmax = np.percentile(np.abs(az_sorted_gather), 99)
        im = ax.imshow(az_sorted_gather.T, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
                      extent=[sorted_azimuths[0], sorted_azimuths[-1], t_axis[-1], t_axis[0]])
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Migrated Gather (sorted by azimuth)')
    else:
        ax.text(0.5, 0.5, 'No azimuth data', ha='center', va='center', transform=ax.transAxes)

    # Semblance comparison
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_raw, semblance_raw, 'b-', alpha=0.7, label='Raw')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Semblance')
    ax.set_title('Semblance - Raw Gather')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_axis, semblance_migrated, 'r-', alpha=0.7, label='Migrated')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Semblance')
    ax.set_title('Semblance - Migrated Gather')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Stack comparison
    ax = fig.add_subplot(gs[1, 2])
    raw_stack = raw_gather.sum(axis=0)
    mig_stack = migrated_gather.sum(axis=0)
    ax.plot(t_raw, raw_stack / n_traces, 'b-', alpha=0.7, label='Raw (normalized)')
    ax.plot(t_axis, mig_stack / n_traces, 'r-', alpha=0.7, label='Migrated (normalized)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Normalized Stacks')
    ax.legend()
    ax.set_xlim([0, 2000])

    # SNR vs trace count
    ax = fig.add_subplot(gs[2, 0])
    if snr_results:
        ns = [r['n'] for r in snr_results]
        snrs = [r['snr'] for r in snr_results]
        ax.plot(ns, snrs, 'b-o')
        ax.set_xlabel('Number of Traces')
        ax.set_ylabel('SNR')
        ax.set_title('SNR vs Number of Traces Stacked')
        ax.grid(True, alpha=0.3)

        # Expected sqrt(N) improvement
        if len(ns) > 1:
            expected = snrs[0] * np.sqrt(np.array(ns) / ns[0])
            ax.plot(ns, expected, 'r--', label='Expected sqrt(N)')
            ax.legend()

    # Distance distribution of traces
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(trace_distances, bins=20, alpha=0.7)
    ax.set_xlabel('Distance to Output Point (m)')
    ax.set_ylabel('Count')
    ax.set_title(f'Trace Distance Distribution (aperture={test_aperture}m)')

    # Peak time shifts vs azimuth
    ax = fig.add_subplot(gs[2, 2])
    if trace_azimuths and valid_peaks:
        valid_az = [trace_azimuths[i] for i in range(len(peak_positions)) if not np.isnan(peak_positions[i])]
        valid_shifts = [p * DT_MS for p in peak_positions if not np.isnan(p)]
        ax.scatter(valid_az, valid_shifts, s=20, alpha=0.5)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Time Shift (ms)')
        ax.set_title('Peak Time Shift vs Azimuth at t~1000ms')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "coherency_diagnostic.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("COHERENCY DIAGNOSTIC SUMMARY")
    print("="*70)

    print(f"""
Test Point: IL={TEST_IL}, XL={TEST_XL}
Traces analyzed: {n_traces} (within {test_aperture}m aperture)

Semblance Results:
  Raw gather:      mean={semblance_raw.mean():.4f}, max={semblance_raw.max():.4f}
  Migrated gather: mean={semblance_migrated.mean():.4f}, max={semblance_migrated.max():.4f}

Key Observations:
1. Low semblance values indicate limited coherent signal
2. DSR correction {"improves" if semblance_migrated.mean() > semblance_raw.mean() else "does not improve"} coherency
3. Check figures for:
   - Raw gather: Are events visible before correction?
   - Migrated gather: Are events flattened after DSR?
   - Azimuth-sorted: Is there azimuth-dependent misalignment?
""")


if __name__ == "__main__":
    main()
