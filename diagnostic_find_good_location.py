#!/usr/bin/env python3
"""
Diagnostic Step 3: Find Location with Strong Reflections

The previous test found NO coherent signal at IL=256, XL=214.
This script searches across the grid to find locations with actual signal.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
import zarr
import matplotlib.pyplot as plt
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).parent))

BIN_NUM = 10
COMMON_OFFSET_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
VELOCITY_PATH = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm_ilxl.zarr")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/diagnostic_qc")
MIGRATION_OUTPUT = Path(f"/Users/olegadamovich/SeismicData/PSTM_common_offset/migration_bin_{BIN_NUM:02d}")

GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}
NX, NY, NT = 511, 427, 1001
DT_MS = 2.0


def compute_local_semblance(traces, window=25):
    """Compute average semblance for a gather."""
    n_traces, n_samples = traces.shape
    if n_traces < 3:
        return 0.0

    # Compute semblance in middle portion
    start = n_samples // 4
    end = 3 * n_samples // 4

    semblance_values = []
    for i in range(start, end, window):
        win = traces[:, i:i+window]
        stack = win.sum(axis=0)
        num = np.sum(stack ** 2)
        denom = n_traces * np.sum(win ** 2)
        if denom > 0:
            semblance_values.append(num / denom)

    return np.mean(semblance_values) if semblance_values else 0.0


def main():
    print("\n" + "="*70)
    print("STEP 3: Find Location with Strong Reflections")
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
    bin_trace_idx = df['bin_trace_idx'].to_numpy()

    # Grid direction vectors
    c1 = np.array(GRID_CORNERS['c1'])
    c2 = np.array(GRID_CORNERS['c2'])
    c4 = np.array(GRID_CORNERS['c4'])
    il_dir = (c2 - c1) / (NX - 1)
    xl_dir = (c4 - c1) / (NY - 1)

    # =========================================================================
    # Scan grid for locations with high semblance
    # =========================================================================
    print(f"\n[3.1] Scanning grid for locations with coherent signal...")

    # Test points across the grid
    test_ils = np.linspace(50, 460, 10, dtype=int)
    test_xls = np.linspace(50, 380, 10, dtype=int)

    results = []
    n_locations = len(test_ils) * len(test_xls)

    for i, il in enumerate(test_ils):
        for j, xl in enumerate(test_xls):
            # Compute location coordinates
            pt = c1 + (il - 1) * il_dir + (xl - 1) * xl_dir
            px, py = pt

            # Find traces within 200m
            dist = np.sqrt((mx - px)**2 + (my - py)**2)
            near_idx = np.where(dist < 200.0)[0]

            if len(near_idx) < 10:
                results.append({
                    'il': il, 'xl': xl, 'x': px, 'y': py,
                    'n_traces': len(near_idx), 'semblance': 0, 'rms': 0
                })
                continue

            # Sample traces
            sample_size = min(30, len(near_idx))
            sample_idx = np.random.choice(near_idx, sample_size, replace=False)

            # Load traces
            gather = np.zeros((sample_size, n_samples), dtype=np.float32)
            for k, header_idx in enumerate(sample_idx):
                storage_idx = bin_trace_idx[header_idx]
                if transposed:
                    gather[k, :] = np.asarray(traces_store[:, storage_idx])
                else:
                    gather[k, :] = np.asarray(traces_store[storage_idx, :])

            # Compute semblance
            sem = compute_local_semblance(gather)
            rms = np.sqrt(np.mean(gather**2))

            results.append({
                'il': il, 'xl': xl, 'x': px, 'y': py,
                'n_traces': len(near_idx), 'semblance': sem, 'rms': rms
            })

    # Sort by semblance
    results_sorted = sorted(results, key=lambda x: x['semblance'], reverse=True)

    print(f"\n      Top 10 locations by semblance:")
    print(f"      {'IL':>6} {'XL':>6} {'Traces':>8} {'Semblance':>10} {'RMS':>8}")
    for r in results_sorted[:10]:
        print(f"      {r['il']:>6} {r['xl']:>6} {r['n_traces']:>8} {r['semblance']:>10.4f} {r['rms']:>8.2f}")

    # =========================================================================
    # Detailed analysis at best location
    # =========================================================================
    best = results_sorted[0]
    print(f"\n[3.2] Detailed analysis at best location: IL={best['il']}, XL={best['xl']}")

    pt = c1 + (best['il'] - 1) * il_dir + (best['xl'] - 1) * xl_dir
    px, py = pt

    dist = np.sqrt((mx - px)**2 + (my - py)**2)
    near_idx = np.where(dist < 300.0)[0]

    print(f"      Traces within 300m: {len(near_idx)}")

    # Load more traces for better visualization
    sample_size = min(100, len(near_idx))
    sample_idx = np.random.choice(near_idx, sample_size, replace=False)

    gather = np.zeros((sample_size, n_samples), dtype=np.float32)
    for k, header_idx in enumerate(sample_idx):
        storage_idx = bin_trace_idx[header_idx]
        if transposed:
            gather[k, :] = np.asarray(traces_store[:, storage_idx])
        else:
            gather[k, :] = np.asarray(traces_store[storage_idx, :])

    # Compute correlation matrix
    print(f"\n[3.3] Computing trace correlation at best location...")

    # Sample correlation
    n_corr_test = min(20, sample_size)
    corr_matrix = np.zeros((n_corr_test, n_corr_test))

    for i in range(n_corr_test):
        for j in range(n_corr_test):
            # Use middle portion of trace
            t1 = gather[i, 200:800]
            t2 = gather[j, 200:800]
            t1_norm = t1 / (np.std(t1) + 1e-10)
            t2_norm = t2 / (np.std(t2) + 1e-10)
            corr_matrix[i, j] = np.corrcoef(t1_norm, t2_norm)[0, 1]

    mean_corr = np.mean(corr_matrix[np.triu_indices(n_corr_test, k=1)])
    print(f"      Mean correlation (off-diagonal): {mean_corr:.4f}")

    # =========================================================================
    # Check migration output at this location
    # =========================================================================
    print(f"\n[3.4] Checking migration output at best location...")

    mig_path = MIGRATION_OUTPUT / "migrated_stack.zarr"
    if mig_path.exists():
        mig_store = zarr.open_array(mig_path, mode='r')
        il_idx, xl_idx = best['il'] - 1, best['xl'] - 1
        migrated_trace = np.asarray(mig_store[il_idx, xl_idx, :])

        print(f"      Migrated trace at IL={best['il']}, XL={best['xl']}:")
        print(f"        Min: {migrated_trace.min():.6f}")
        print(f"        Max: {migrated_trace.max():.6f}")
        print(f"        RMS: {np.sqrt(np.mean(migrated_trace**2)):.6f}")

        # Compare with stack of raw traces
        raw_stack = gather.mean(axis=0)
        print(f"      Raw stack RMS: {np.sqrt(np.mean(raw_stack**2)):.4f}")
        print(f"      Ratio (migrated/raw): {np.sqrt(np.mean(migrated_trace**2)) / (np.sqrt(np.mean(raw_stack**2)) + 1e-10):.4f}")

    # =========================================================================
    # Create figure
    # =========================================================================
    print(f"\n[3.5] Creating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Best Location Analysis - Bin {BIN_NUM}, IL={best["il"]} XL={best["xl"]}',
                fontsize=14, fontweight='bold')

    t_axis = np.arange(n_samples) * DT_MS

    # Semblance map
    ax = axes[0, 0]
    sem_grid = np.zeros((len(test_ils), len(test_xls)))
    for r in results:
        i = list(test_ils).index(r['il'])
        j = list(test_xls).index(r['xl'])
        sem_grid[i, j] = r['semblance']

    im = ax.imshow(sem_grid.T, origin='lower', aspect='auto', cmap='hot',
                   extent=[test_ils[0], test_ils[-1], test_xls[0], test_xls[-1]])
    ax.plot(best['il'], best['xl'], 'c*', markersize=15)
    plt.colorbar(im, ax=ax, label='Semblance')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title('Semblance Map')

    # Gather at best location
    ax = axes[0, 1]
    vmax = np.percentile(np.abs(gather), 98)
    ax.imshow(gather.T, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
              extent=[0, sample_size, t_axis[-1], t_axis[0]])
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Gather at Best Location ({sample_size} traces)')

    # Zoom on potential reflections
    ax = axes[0, 2]
    t_start, t_end = 600, 1200
    gather_zoom = gather[:, int(t_start/DT_MS):int(t_end/DT_MS)]
    vmax_zoom = np.percentile(np.abs(gather_zoom), 98)
    ax.imshow(gather_zoom.T, aspect='auto', cmap='gray', vmin=-vmax_zoom, vmax=vmax_zoom,
              extent=[0, sample_size, t_end, t_start])
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Zoom: 600-1200ms')

    # Correlation matrix
    ax = axes[1, 0]
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Trace #')
    ax.set_title(f'Correlation Matrix (mean off-diag: {mean_corr:.3f})')

    # Stack comparison
    ax = axes[1, 1]
    raw_stack = gather.mean(axis=0)
    ax.plot(t_axis, raw_stack, 'b-', linewidth=0.5, label='Raw stack', alpha=0.7)
    if mig_path.exists():
        # Scale migrated for comparison
        scale = np.std(raw_stack) / (np.std(migrated_trace) + 1e-10)
        ax.plot(t_axis[:len(migrated_trace)], migrated_trace * scale, 'r-',
               linewidth=0.5, label=f'Migrated (x{scale:.0f})', alpha=0.7)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Raw Stack vs Migrated')
    ax.legend()
    ax.set_xlim([0, 2000])

    # Trace count map
    ax = axes[1, 2]
    count_grid = np.zeros((len(test_ils), len(test_xls)))
    for r in results:
        i = list(test_ils).index(r['il'])
        j = list(test_xls).index(r['xl'])
        count_grid[i, j] = r['n_traces']

    im = ax.imshow(count_grid.T, origin='lower', aspect='auto', cmap='viridis',
                   extent=[test_ils[0], test_ils[-1], test_xls[0], test_xls[-1]])
    plt.colorbar(im, ax=ax, label='Trace Count')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title('Trace Count Map (200m radius)')

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "step3_best_location.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n      Saved: {fig_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("LOCATION SEARCH SUMMARY")
    print("="*70)

    max_sem = max(r['semblance'] for r in results)
    mean_sem = np.mean([r['semblance'] for r in results])

    print(f"""
  Grid scan results:
    Locations tested: {n_locations}
    Max semblance: {max_sem:.4f}
    Mean semblance: {mean_sem:.4f}

  Best location: IL={best['il']}, XL={best['xl']}
    Semblance: {best['semblance']:.4f}
    Traces within 200m: {best['n_traces']}
    Mean correlation: {mean_corr:.4f}

  Interpretation:
    - Semblance > 0.1: Some coherent signal present
    - Semblance > 0.3: Good coherent reflections
    - Semblance < 0.05: Mostly noise

  Current maximum semblance ({max_sem:.4f}) suggests:
    {"GOOD: Coherent signal exists" if max_sem > 0.1 else "POOR: Limited coherent signal in data"}
""")

    return best


if __name__ == "__main__":
    main()
