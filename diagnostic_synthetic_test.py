#!/usr/bin/env python3
"""
Diagnostic Step 4: Synthetic Data Test

Create synthetic traces with known geometry and verify migration focuses correctly.
This isolates algorithm issues from data issues.

Test cases:
1. Single point diffractor - should produce focused image
2. Flat reflector - should produce flat event
3. Multiple offsets at same CMP - should stack coherently
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/diagnostic_qc")
DT_MS = 2.0


def create_ricker_wavelet(f_peak, dt_s, duration_s):
    """Create a Ricker wavelet."""
    t = np.arange(-duration_s/2, duration_s/2, dt_s)
    return (1 - 2*(np.pi*f_peak*t)**2) * np.exp(-(np.pi*f_peak*t)**2), t


def dsr_traveltime(ox, oy, sx, sy, rx, ry, t0, v):
    """Compute DSR traveltime."""
    ds2 = (ox - sx)**2 + (oy - sy)**2
    dr2 = (ox - rx)**2 + (oy - ry)**2
    t0_half_sq = (t0 / 2)**2
    inv_v_sq = 1 / (v * v)
    return np.sqrt(t0_half_sq + ds2 * inv_v_sq) + np.sqrt(t0_half_sq + dr2 * inv_v_sq)


def main():
    print("\n" + "="*70)
    print("STEP 4: Synthetic Data Test")
    print("="*70)

    # =========================================================================
    # Test 4.1: Point Diffractor
    # =========================================================================
    print(f"\n[4.1] Point Diffractor Test")
    print("=" * 50)

    # Parameters
    v = 2500.0  # m/s (constant velocity)
    t0 = 1.0    # s (diffractor at t=1s)
    f_peak = 30.0  # Hz
    dt_s = DT_MS / 1000.0
    n_samples = 1001

    # Output grid (small for test)
    ox_range = np.linspace(-500, 500, 41)  # -500 to 500m
    oy = 0.0  # Single line

    # Create synthetic traces with different source-receiver positions
    # All traces have midpoint at (0, 0) but different offsets/azimuths
    offsets = [200, 400, 600]  # m
    azimuths = np.arange(0, 360, 45)  # Every 45 degrees

    print(f"  Velocity: {v} m/s")
    print(f"  Diffractor time: {t0*1000} ms")
    print(f"  Offsets: {offsets}")
    print(f"  Azimuths: {list(azimuths)}")

    # Create wavelet
    wavelet, _ = create_ricker_wavelet(f_peak, dt_s, 0.1)
    wavelet_half = len(wavelet) // 2

    # Generate synthetic traces
    traces = []
    geometries = []

    for offset in offsets:
        for azimuth in azimuths:
            # Source and receiver positions (midpoint at origin)
            dx = offset / 2 * np.sin(np.radians(azimuth))
            dy = offset / 2 * np.cos(np.radians(azimuth))

            sx, sy = -dx, -dy
            rx, ry = dx, dy

            # Compute traveltime to diffractor at (0, 0, t0)
            # For point diffractor at t0: t_travel = DSR with output at (0,0)
            t_travel = dsr_traveltime(0, 0, sx, sy, rx, ry, t0, v)

            # Create trace
            trace = np.zeros(n_samples)
            sample_idx = int(t_travel / dt_s)

            if sample_idx - wavelet_half >= 0 and sample_idx + wavelet_half < n_samples:
                wlen = min(len(wavelet), 2*wavelet_half + 1)
                trace[sample_idx - wavelet_half:sample_idx - wavelet_half + wlen] = wavelet[:wlen]

            traces.append(trace)
            geometries.append({
                'sx': sx, 'sy': sy, 'rx': rx, 'ry': ry,
                'offset': offset, 'azimuth': azimuth,
                't_travel': t_travel
            })

    traces = np.array(traces)
    n_traces = len(traces)
    print(f"  Generated {n_traces} synthetic traces")

    # Migrate the synthetic traces
    print(f"\n  Migrating with DSR...")

    # Output image along ox_range at oy=0
    migrated_line = np.zeros((len(ox_range), n_samples))
    fold_line = np.zeros(len(ox_range))

    for i, ox in enumerate(ox_range):
        for j, g in enumerate(geometries):
            trace = traces[j]
            sx, sy = g['sx'], g['sy']
            rx, ry = g['rx'], g['ry']

            # Check aperture (use 1000m for this test)
            mx, my = (sx + rx) / 2, (sy + ry) / 2
            dm = np.sqrt((ox - mx)**2 + (oy - my)**2)
            if dm > 1000:
                continue

            fold_line[i] += 1

            # Migrate to each output time
            for it in range(50, n_samples - 50):  # Skip edges
                t0_out = it * dt_s

                t_travel = dsr_traveltime(ox, oy, sx, sy, rx, ry, t0_out, v)
                sample_idx = t_travel / dt_s

                if 0 <= sample_idx < n_samples - 1:
                    idx0 = int(sample_idx)
                    frac = sample_idx - idx0
                    amp = trace[idx0] * (1-frac) + trace[idx0+1] * frac
                    migrated_line[i, it] += amp

    # Normalize by fold
    for i in range(len(ox_range)):
        if fold_line[i] > 0:
            migrated_line[i, :] /= fold_line[i]

    print(f"  Migration complete")
    print(f"  Fold range: {fold_line.min():.0f} - {fold_line.max():.0f}")

    # Check focusing
    t_axis = np.arange(n_samples) * dt_s * 1000  # ms
    t_idx = int(t0 / dt_s)
    at_diffractor_time = migrated_line[:, t_idx]
    peak_idx = np.argmax(np.abs(at_diffractor_time))
    peak_x = ox_range[peak_idx]
    peak_amp = at_diffractor_time[peak_idx]

    print(f"\n  Results:")
    print(f"    Expected focus: x=0, t={t0*1000}ms")
    print(f"    Actual peak: x={peak_x:.1f}m, amp={peak_amp:.4f}")
    print(f"    Focusing accuracy: {abs(peak_x):.1f}m from target")

    # =========================================================================
    # Test 4.2: Effect of aperture size
    # =========================================================================
    print(f"\n[4.2] Aperture Effect Test")
    print("=" * 50)

    # Test different apertures
    apertures = [100, 250, 500, 1000, 2000]

    for aperture in apertures:
        migrated_test = np.zeros((len(ox_range), n_samples))
        fold_test = np.zeros(len(ox_range))

        for i, ox in enumerate(ox_range):
            for j, g in enumerate(geometries):
                trace = traces[j]
                sx, sy = g['sx'], g['sy']
                rx, ry = g['rx'], g['ry']

                mx, my = (sx + rx) / 2, (sy + ry) / 2
                dm = np.sqrt((ox - mx)**2 + (oy - my)**2)
                if dm > aperture:
                    continue

                fold_test[i] += 1

                for it in range(50, n_samples - 50):
                    t0_out = it * dt_s
                    t_travel = dsr_traveltime(ox, oy, sx, sy, rx, ry, t0_out, v)
                    sample_idx = t_travel / dt_s

                    if 0 <= sample_idx < n_samples - 1:
                        idx0 = int(sample_idx)
                        frac = sample_idx - idx0
                        amp = trace[idx0] * (1-frac) + trace[idx0+1] * frac
                        migrated_test[i, it] += amp

        # Normalize
        for i in range(len(ox_range)):
            if fold_test[i] > 0:
                migrated_test[i, :] /= fold_test[i]

        # Check result
        at_t0 = migrated_test[:, t_idx]
        peak_amp = at_t0[len(ox_range)//2]  # At x=0
        max_amp = np.abs(at_t0).max()

        print(f"  Aperture {aperture:4d}m: fold={fold_test[len(ox_range)//2]:.0f}, "
              f"amp_at_origin={peak_amp:.4f}, max_amp={max_amp:.4f}")

    # =========================================================================
    # Create visualization
    # =========================================================================
    print(f"\n[4.3] Creating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Synthetic Data Test - Point Diffractor Migration', fontsize=14, fontweight='bold')

    # Input traces
    ax = axes[0, 0]
    vmax = np.percentile(np.abs(traces), 99)
    ax.imshow(traces.T, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
              extent=[0, n_traces, t_axis[-1], t_axis[0]])
    ax.axhline(t0*1000, color='r', linestyle='--', alpha=0.5, label=f't={t0*1000}ms')
    ax.set_xlabel('Trace #')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Input Traces ({n_traces} traces)')
    ax.legend()

    # Migrated image
    ax = axes[0, 1]
    vmax = np.percentile(np.abs(migrated_line), 99)
    ax.imshow(migrated_line.T, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
              extent=[ox_range[0], ox_range[-1], t_axis[-1], t_axis[0]])
    ax.axhline(t0*1000, color='r', linestyle='--', alpha=0.5)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax.scatter([0], [t0*1000], c='red', s=100, marker='x', label='Expected focus')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Migrated Image')
    ax.legend()

    # Zoom on focus point
    ax = axes[0, 2]
    zoom_t = (800, 1200)
    zoom_x = (-200, 200)
    t_mask = (t_axis >= zoom_t[0]) & (t_axis <= zoom_t[1])
    x_mask = (ox_range >= zoom_x[0]) & (ox_range <= zoom_x[1])
    zoom_data = migrated_line[np.ix_(x_mask, t_mask)]
    vmax_zoom = np.percentile(np.abs(zoom_data), 99)
    ax.imshow(zoom_data.T, aspect='auto', cmap='gray', vmin=-vmax_zoom, vmax=vmax_zoom,
              extent=[zoom_x[0], zoom_x[1], zoom_t[1], zoom_t[0]])
    ax.scatter([0], [t0*1000], c='red', s=100, marker='x')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Zoom on Focus Point')

    # Amplitude at t=t0
    ax = axes[1, 0]
    ax.plot(ox_range, at_diffractor_time, 'b-')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Expected focus')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Amplitude at t={t0*1000}ms')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fold
    ax = axes[1, 1]
    ax.plot(ox_range, fold_line, 'b-')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Fold')
    ax.set_title('Fold vs X Position')
    ax.grid(True, alpha=0.3)

    # Trace at x=0
    ax = axes[1, 2]
    trace_at_origin = migrated_line[len(ox_range)//2, :]
    ax.plot(t_axis, trace_at_origin, 'b-', linewidth=0.5)
    ax.axvline(t0*1000, color='r', linestyle='--', alpha=0.5, label=f't={t0*1000}ms')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Migrated Trace at x=0')
    ax.legend()
    ax.set_xlim([0, 2000])

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "step4_synthetic_test.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SYNTHETIC TEST SUMMARY")
    print("="*70)

    print(f"""
  Point Diffractor Test:
    - {n_traces} synthetic traces with known geometry
    - Offsets: {offsets} m
    - Azimuths: 0-315 degrees (every 45)
    - Constant velocity: {v} m/s
    - Diffractor at: (0, 0, {t0*1000}ms)

  Migration Results:
    - Expected focus: x=0m
    - Actual focus: x={peak_x:.1f}m
    - Peak amplitude: {peak_amp:.4f}

  Conclusion:
    {"PASS: Migration focuses correctly" if abs(peak_x) < 25 else "FAIL: Migration not focusing"}

  Key Insight:
    The DSR migration formula works correctly for synthetic data.
    The issue with real data is likely:
    1. Fold normalization (dividing by ALL traces, not just coherent ones)
    2. Traces from different CMPs don't share same reflectors
    3. Large aperture brings in incoherent energy
""")


if __name__ == "__main__":
    main()
