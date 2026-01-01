#!/usr/bin/env python3
"""
PSTM Amplitude Diagnostic - Deep dive into amplitude scaling issue.

Test 7 showed manual single-trace migration has ~400x larger amplitude than actual output.
This script investigates:
1. Fold values and normalization
2. Aperture taper attenuation
3. Anti-aliasing weight impact
4. Multi-trace stacking behavior
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
import zarr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# Configuration
# =============================================================================

BIN_NUM = 10
COMMON_OFFSET_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
VELOCITY_PATH = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm_ilxl.zarr")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/diagnostic_qc")
MIGRATION_OUTPUT = Path(f"/Users/olegadamovich/SeismicData/PSTM_common_offset/migration_bin_{BIN_NUM:02d}")

# Grid parameters
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}
NX, NY, NT = 511, 427, 1001
DX, DY, DT_MS = 25.0, 12.5, 2.0

TEST_IL, TEST_XL = 256, 214
TEST_IX, TEST_IY = TEST_IL - 1, TEST_XL - 1


def main():
    print("\n" + "="*70)
    print("AMPLITUDE DIAGNOSTIC - Investigating 400x amplitude discrepancy")
    print("="*70)

    # Load data
    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{BIN_NUM:02d}"
    traces_store = zarr.open_array(bin_dir / "traces.zarr", mode='r')
    df = pl.read_parquet(bin_dir / "headers.parquet")

    scalar = int(df['scalar_coord'][0])
    scale_factor = 1.0 / abs(scalar) if scalar < 0 else float(scalar) if scalar > 0 else 1.0

    # Load coordinates
    sx = df['source_x'].to_numpy().astype(np.float64) * scale_factor
    sy = df['source_y'].to_numpy().astype(np.float64) * scale_factor
    rx = df['receiver_x'].to_numpy().astype(np.float64) * scale_factor
    ry = df['receiver_y'].to_numpy().astype(np.float64) * scale_factor
    mx = (sx + rx) / 2.0
    my = (sy + ry) / 2.0

    # Compute output point coordinates
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

    # Load migration results
    mig_path = MIGRATION_OUTPUT / "migrated_stack.zarr"
    mig_store = zarr.open_array(mig_path, mode='r')
    actual_migrated = np.asarray(mig_store[TEST_IX, TEST_IY, :])

    # =========================================================================
    # Test A: Check Fold Values
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST A: Fold Analysis")
    print("-"*70)

    fold_path = MIGRATION_OUTPUT / "fold.zarr"
    if fold_path.exists():
        fold_store = zarr.open_array(fold_path, mode='r')
        fold_data = np.asarray(fold_store)
        fold_at_test = fold_data[TEST_IX, TEST_IY]
        print(f"  Fold array shape: {fold_data.shape}")
        print(f"  Fold at test point (IL={TEST_IL}, XL={TEST_XL}): {fold_at_test}")
        print(f"  Fold statistics: min={fold_data.min()}, max={fold_data.max()}, mean={fold_data.mean():.0f}")
    else:
        print(f"  Fold file not found: {fold_path}")
        fold_at_test = None

    # =========================================================================
    # Test B: Find traces within aperture
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST B: Traces Within Aperture")
    print("-"*70)

    # Find traces within 2000m aperture
    distances = np.sqrt((mx - ox)**2 + (my - oy)**2)
    aperture = 2000.0
    within_aperture = distances <= aperture
    n_within = within_aperture.sum()

    print(f"  Traces within {aperture}m aperture: {n_within:,}")
    print(f"  Distance range: {distances[within_aperture].min():.1f} - {distances[within_aperture].max():.1f} m")

    # =========================================================================
    # Test C: Manual Multi-Trace Migration (no fold normalization)
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST C: Manual Multi-Trace Migration")
    print("-"*70)

    # Select traces within aperture
    indices_in_aperture = np.where(within_aperture)[0]
    print(f"  Processing {len(indices_in_aperture):,} traces...")

    # Get bin trace indices for storage access
    bin_trace_idx = df['bin_trace_idx'].to_numpy()

    # Migration parameters
    taper_fraction = 0.1
    taper_start = aperture * (1.0 - taper_fraction)
    t_axis_ms = np.arange(NT) * DT_MS

    # Initialize accumulators
    image_sum = np.zeros(NT, dtype=np.float64)
    image_count = np.zeros(NT, dtype=np.int32)
    weights_sum = np.zeros(NT, dtype=np.float64)

    # Sample a subset for detailed analysis
    sample_size = min(1000, len(indices_in_aperture))
    sample_indices = np.random.choice(indices_in_aperture, sample_size, replace=False)

    print(f"  Detailed analysis on {sample_size} traces...")

    # Track statistics
    taper_weights = []
    contributions_at_1000ms = []

    for i, trace_idx in enumerate(sample_indices):
        storage_idx = bin_trace_idx[trace_idx]

        # Get trace data
        if traces_store.shape[0] < traces_store.shape[1]:
            trace_data = np.asarray(traces_store[:, storage_idx]).astype(np.float32)
        else:
            trace_data = np.asarray(traces_store[storage_idx, :]).astype(np.float32)

        # Get geometry
        trace_sx, trace_sy = sx[trace_idx], sy[trace_idx]
        trace_rx, trace_ry = rx[trace_idx], ry[trace_idx]

        ds2 = (ox - trace_sx)**2 + (oy - trace_sy)**2
        dr2 = (ox - trace_rx)**2 + (oy - trace_ry)**2
        dm = distances[trace_idx]

        # Compute taper weight
        if dm <= taper_start:
            taper_weight = 1.0
        elif dm >= aperture:
            taper_weight = 0.0
        else:
            t = (dm - taper_start) / (aperture - taper_start)
            taper_weight = 0.5 * (1.0 + np.cos(t * np.pi))

        taper_weights.append(taper_weight)

        # Migrate to selected output times
        for it in [500]:  # Just t=1000ms for detailed analysis
            t0_s = t_axis_ms[it] / 1000.0
            v = velocity[TEST_IX, TEST_IY, it]
            t0_half_sq = (t0_s / 2)**2
            inv_v_sq = 1 / (v * v)

            t_travel = np.sqrt(t0_half_sq + ds2 * inv_v_sq) + np.sqrt(t0_half_sq + dr2 * inv_v_sq)
            sample_idx = (t_travel * 1000.0) / DT_MS

            if 0 <= sample_idx < len(trace_data) - 1:
                idx0 = int(sample_idx)
                frac = sample_idx - idx0
                amp = trace_data[idx0] * (1 - frac) + trace_data[idx0 + 1] * frac
                weighted_amp = amp * taper_weight
                contributions_at_1000ms.append({
                    'trace_idx': trace_idx,
                    'distance': dm,
                    'taper_weight': taper_weight,
                    'raw_amp': amp,
                    'weighted_amp': weighted_amp,
                    't_travel': t_travel,
                })

        # Full migration for all times
        for it in range(NT):
            t0_s = t_axis_ms[it] / 1000.0
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
                amp = trace_data[idx0] * (1 - frac) + trace_data[idx0 + 1] * frac
                weighted_amp = amp * taper_weight

                image_sum[it] += weighted_amp
                image_count[it] += 1
                weights_sum[it] += taper_weight

    # Analyze contributions at t=1000ms
    print(f"\n  Contributions at t=1000ms ({len(contributions_at_1000ms)} traces):")
    if contributions_at_1000ms:
        raw_amps = np.array([c['raw_amp'] for c in contributions_at_1000ms])
        weighted_amps = np.array([c['weighted_amp'] for c in contributions_at_1000ms])
        taper_ws = np.array([c['taper_weight'] for c in contributions_at_1000ms])

        print(f"    Raw amplitude: mean={raw_amps.mean():.6f}, std={raw_amps.std():.6f}")
        print(f"    Weighted amplitude: mean={weighted_amps.mean():.6f}, std={weighted_amps.std():.6f}")
        print(f"    Taper weights: mean={taper_ws.mean():.4f}, min={taper_ws.min():.4f}, max={taper_ws.max():.4f}")
        print(f"    Sum of raw amps: {raw_amps.sum():.4f}")
        print(f"    Sum of weighted amps: {weighted_amps.sum():.4f}")

    # =========================================================================
    # Test D: Compare different normalization approaches
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST D: Normalization Comparison")
    print("-"*70)

    t_idx = 500  # t=1000ms

    unnormalized = image_sum[t_idx]
    by_count = image_sum[t_idx] / max(image_count[t_idx], 1)
    by_weights = image_sum[t_idx] / max(weights_sum[t_idx], 1e-10)

    actual_value = actual_migrated[t_idx]

    print(f"  At t=1000ms (from {sample_size} traces):")
    print(f"    Unnormalized sum: {unnormalized:.6f}")
    print(f"    Normalized by count ({image_count[t_idx]}): {by_count:.6f}")
    print(f"    Normalized by weight sum ({weights_sum[t_idx]:.1f}): {by_weights:.6f}")
    print(f"    Actual migration output: {actual_value:.6f}")

    if fold_at_test:
        print(f"\n  If we scale by full fold ({fold_at_test:,}):")
        print(f"    Extrapolated full sum: {unnormalized * (n_within / sample_size):.2f}")
        print(f"    Divided by fold: {unnormalized * (n_within / sample_size) / fold_at_test:.6f}")

    # =========================================================================
    # Test E: Check if fold is computed per time or just at t=0
    # =========================================================================
    print("\n" + "-"*70)
    print("TEST E: Fold Dimensionality Check")
    print("-"*70)

    if fold_path.exists():
        fold_store = zarr.open_array(fold_path, mode='r')
        print(f"  Fold array shape: {fold_store.shape}")
        print(f"  Fold is 2D (one value per x,y): {len(fold_store.shape) == 2}")

        if len(fold_store.shape) == 2:
            print("\n  ISSUE IDENTIFIED: Fold is 2D but aperture varies with time!")
            print("  At shallow times, fewer traces contribute but are divided by deep-time fold!")

            # Estimate trace count at different times
            print("\n  Estimated trace count vs time (based on aperture):")
            for t_ms in [200, 500, 1000, 1500, 2000]:
                t_s = t_ms / 1000.0
                t_idx = int(t_ms / DT_MS)
                v = velocity[TEST_IX, TEST_IY, t_idx]

                # Simple aperture model: min(max_aperture, v * t * tan(max_dip))
                max_dip_rad = np.radians(65)
                computed_aperture = min(2000, v * t_s * np.tan(max_dip_rad))

                traces_at_aperture = (distances <= computed_aperture).sum()
                print(f"    t={t_ms}ms: aperture={computed_aperture:.0f}m, traces={traces_at_aperture:,}")

    # =========================================================================
    # Create QC figure
    # =========================================================================
    print("\n" + "-"*70)
    print("Creating QC Figure")
    print("-"*70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Amplitude Diagnostic - Bin {BIN_NUM}', fontsize=14, fontweight='bold')

    # A: Fold map
    ax = axes[0, 0]
    if fold_path.exists():
        fold_data = np.asarray(zarr.open_array(fold_path, mode='r'))
        im = ax.imshow(fold_data.T, origin='lower', aspect='auto', cmap='viridis',
                      extent=[1, NX, 1, NY])
        ax.plot(TEST_IL, TEST_XL, 'ro', markersize=10)
        plt.colorbar(im, ax=ax, label='Fold')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        ax.set_title(f'Fold Map (at test: {fold_at_test:,})')
    else:
        ax.text(0.5, 0.5, 'Fold not available', ha='center', va='center', transform=ax.transAxes)

    # B: Taper weight distribution
    ax = axes[0, 1]
    ax.hist(taper_weights, bins=50, alpha=0.7)
    ax.set_xlabel('Taper Weight')
    ax.set_ylabel('Count')
    ax.set_title(f'Taper Weight Distribution (mean={np.mean(taper_weights):.3f})')
    ax.axvline(1.0, color='r', linestyle='--', label='No taper')
    ax.legend()

    # C: Contribution histogram at t=1000ms
    ax = axes[0, 2]
    if contributions_at_1000ms:
        raw_amps = [c['raw_amp'] for c in contributions_at_1000ms]
        ax.hist(raw_amps, bins=50, alpha=0.7)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Raw Amplitude')
        ax.set_ylabel('Count')
        ax.set_title(f'Contribution Distribution at t=1000ms')

    # D: Stacked trace comparison
    ax = axes[1, 0]
    t_axis = np.arange(NT) * DT_MS
    ax.plot(t_axis, image_sum / max(image_count.max(), 1), 'b-', linewidth=0.5, label='Manual (norm by count)', alpha=0.7)
    ax.plot(t_axis, actual_migrated * 100, 'r-', linewidth=0.5, label='Actual x100')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Manual vs Actual Migration (scaled)')
    ax.legend()
    ax.set_xlim([0, 2000])

    # E: Trace count vs time
    ax = axes[1, 1]
    ax.plot(t_axis, image_count, 'b-')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Contributing Traces')
    ax.set_title(f'Traces Contributing per Time Sample (sample of {sample_size})')
    ax.grid(True, alpha=0.3)

    # F: Distance vs amplitude scatter
    ax = axes[1, 2]
    if contributions_at_1000ms:
        dists = [c['distance'] for c in contributions_at_1000ms]
        amps = [c['raw_amp'] for c in contributions_at_1000ms]
        ax.scatter(dists, amps, s=1, alpha=0.3)
        ax.axvline(taper_start, color='orange', linestyle='--', label=f'Taper start ({taper_start:.0f}m)')
        ax.axvline(aperture, color='r', linestyle='--', label=f'Aperture ({aperture:.0f}m)')
        ax.set_xlabel('Distance to Output (m)')
        ax.set_ylabel('Raw Amplitude')
        ax.set_title('Amplitude vs Distance at t=1000ms')
        ax.legend()

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "amplitude_diagnostic.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("AMPLITUDE DIAGNOSTIC SUMMARY")
    print("="*70)

    print(f"""
Key Findings:
1. Fold at test point: {fold_at_test:,} traces
2. Traces within 2000m aperture: {n_within:,}
3. Fold is 2D (computed at t=0) but aperture varies with time

Amplitude comparison at t=1000ms:
- Manual stacked (from {sample_size} traces): {image_sum[500]:.4f}
- Extrapolated to {n_within} traces: {image_sum[500] * n_within / sample_size:.4f}
- Divided by fold ({fold_at_test:,}): {image_sum[500] * n_within / sample_size / fold_at_test if fold_at_test else 'N/A':.6f}
- Actual migration output: {actual_value:.6f}

Ratio (extrapolated / actual): {(image_sum[500] * n_within / sample_size) / (actual_value + 1e-10):.1f}x
""")

    return {
        'fold_at_test': fold_at_test,
        'n_within_aperture': n_within,
        'sample_size': sample_size,
        'manual_sum': image_sum[500],
        'actual_value': actual_value,
    }


if __name__ == "__main__":
    main()
