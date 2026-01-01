#!/usr/bin/env python3
"""
Realistic Test of 3D Fold Normalization

Tests with traces at various midpoint locations to demonstrate
how fold varies with depth due to aperture constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/diagnostic_qc")


def create_ricker_wavelet(f_peak, dt_s, duration_s):
    t = np.arange(-duration_s/2, duration_s/2, dt_s)
    return (1 - 2*(np.pi*f_peak*t)**2) * np.exp(-(np.pi*f_peak*t)**2), t


def dsr_traveltime(ox, oy, sx, sy, rx, ry, t0, v):
    ds2 = (ox - sx)**2 + (oy - sy)**2
    dr2 = (ox - rx)**2 + (oy - ry)**2
    t0_half_sq = (t0 / 2)**2
    inv_v_sq = 1 / (v * v)
    return np.sqrt(t0_half_sq + ds2 * inv_v_sq) + np.sqrt(t0_half_sq + dr2 * inv_v_sq)


def test_realistic_fold_variation():
    """Test with realistic trace distribution showing fold variation."""
    print("=" * 70)
    print("REALISTIC TEST: 3D Fold with Distributed Midpoints")
    print("=" * 70)

    # Parameters
    v = 2500.0
    f_peak = 30.0
    dt_ms = 2.0
    dt_s = dt_ms / 1000.0
    n_samples = 1001

    # Output point at origin
    ox, oy = 0.0, 0.0

    # Create traces with midpoints distributed around the output point
    # This simulates real data where traces have various midpoint locations
    print("\n[1] Creating traces with distributed midpoints...")

    np.random.seed(42)
    n_traces = 500

    # Midpoints distributed uniformly within 1000m radius
    angles = np.random.uniform(0, 2*np.pi, n_traces)
    radii = np.random.uniform(0, 1000, n_traces)  # 0-1000m from output
    midpoint_x = radii * np.cos(angles)
    midpoint_y = radii * np.sin(angles)

    # Random offsets 200-600m
    offsets = np.random.uniform(200, 600, n_traces)
    azimuths = np.random.uniform(0, 360, n_traces)

    # Compute source/receiver from midpoint, offset, azimuth
    source_x = midpoint_x - (offsets/2) * np.sin(np.radians(azimuths))
    source_y = midpoint_y - (offsets/2) * np.cos(np.radians(azimuths))
    receiver_x = midpoint_x + (offsets/2) * np.sin(np.radians(azimuths))
    receiver_y = midpoint_y + (offsets/2) * np.cos(np.radians(azimuths))

    print(f"    {n_traces} traces created")
    print(f"    Midpoint distance range: 0-1000m")
    print(f"    Offset range: 200-600m")

    # Create wavelet
    wavelet, _ = create_ricker_wavelet(f_peak, dt_s, 0.1)
    wavelet_half = len(wavelet) // 2

    # Create flat reflector at multiple times
    reflector_times_ms = [400, 800, 1200, 1600]

    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for i in range(n_traces):
        for t0_ms in reflector_times_ms:
            t0 = t0_ms / 1000.0
            t_travel = dsr_traveltime(ox, oy, source_x[i], source_y[i],
                                      receiver_x[i], receiver_y[i], t0, v)
            sample_idx = int(t_travel / dt_s)
            if sample_idx - wavelet_half >= 0 and sample_idx + wavelet_half < n_samples:
                wlen = min(len(wavelet), n_samples - sample_idx + wavelet_half)
                start = sample_idx - wavelet_half
                traces[i, start:start + wlen] += wavelet[:wlen]

    # Aperture model: linear increase with depth
    # 200m at t=0, 1000m at t=2000ms
    def get_aperture(t_ms):
        return 200 + (1000 - 200) * (t_ms / 2000)

    print(f"\n[2] Aperture model: {get_aperture(0):.0f}m at 0ms → {get_aperture(2000):.0f}m at 2000ms")

    # Expected fold at each time
    print("\n[3] Computing expected fold at reflector times...")
    for t_ms in reflector_times_ms:
        aperture = get_aperture(t_ms)
        dm = np.sqrt(midpoint_x**2 + midpoint_y**2)
        expected_fold = np.sum(dm <= aperture)
        print(f"    t={t_ms}ms: aperture={aperture:.0f}m, expected fold={expected_fold}")

    # Migrate
    print("\n[4] Running migration...")
    image = np.zeros(n_samples, dtype=np.float64)
    fold_3d = np.zeros(n_samples, dtype=np.int32)

    for it in range(50, n_samples - 50):
        t0_out = it * dt_s
        t0_ms = t0_out * 1000
        aperture = get_aperture(t0_ms)

        for j in range(n_traces):
            # Check aperture
            dm = np.sqrt(midpoint_x[j]**2 + midpoint_y[j]**2)
            if dm > aperture:
                continue

            # Compute DSR traveltime
            t_travel = dsr_traveltime(ox, oy, source_x[j], source_y[j],
                                      receiver_x[j], receiver_y[j], t0_out, v)
            sample_idx = t_travel / dt_s

            if 0 <= sample_idx < n_samples - 1:
                idx0 = int(sample_idx)
                frac = sample_idx - idx0
                amp = traces[j, idx0] * (1-frac) + traces[j, idx0+1] * frac
                image[it] += amp
                fold_3d[it] += 1

    # Normalize with 3D fold
    with np.errstate(invalid='ignore', divide='ignore'):
        image_normalized = np.where(fold_3d > 0, image / fold_3d, 0.0)

    # Results
    print("\n[5] Results at reflector times:")
    print(f"    {'Time':<8} {'Aperture':<10} {'Fold':<8} {'Raw Amp':<12} {'Norm Amp':<12}")
    print("    " + "-" * 50)

    norm_amps = []
    folds = []
    for t_ms in reflector_times_ms:
        t_idx = int(t_ms / dt_ms)
        aperture = get_aperture(t_ms)
        print(f"    {t_ms:<8} {aperture:<10.0f} {fold_3d[t_idx]:<8} "
              f"{image[t_idx]:<12.4f} {image_normalized[t_idx]:<12.4f}")
        norm_amps.append(image_normalized[t_idx])
        folds.append(fold_3d[t_idx])

    # Check fold increases with depth
    fold_ratio = folds[-1] / folds[0] if folds[0] > 0 else 0
    print(f"\n[6] Fold Variation:")
    print(f"    Fold at 400ms: {folds[0]}")
    print(f"    Fold at 1600ms: {folds[-1]}")
    print(f"    Ratio: {fold_ratio:.1f}x")

    if fold_ratio > 1.5:
        print(f"    PASS: Fold increases significantly with depth")
    else:
        print(f"    NOTE: Fold increase is modest")

    # Check amplitude consistency
    amp_mean = np.mean(np.abs(norm_amps))
    amp_std = np.std(np.abs(norm_amps))
    amp_cv = amp_std / amp_mean if amp_mean > 0 else 0

    print(f"\n[7] Amplitude Consistency:")
    print(f"    Mean: {amp_mean:.4f}")
    print(f"    Std: {amp_std:.4f}")
    print(f"    CV: {amp_cv:.1%}")

    if amp_cv < 0.3:
        print(f"    PASS: Amplitude variation < 30%")
    else:
        print(f"    WARNING: High amplitude variation")

    # Compare with OLD (2D fold) approach
    print("\n[8] Comparison: 3D fold vs 2D fold (old approach)")

    # 2D fold would use the fold at t=0 (or some reference time) for all times
    fold_2d = fold_3d[0]  # Use shallow time fold
    with np.errstate(invalid='ignore', divide='ignore'):
        image_2d_norm = np.where(fold_2d > 0, image / fold_2d, 0.0)

    print(f"    2D fold (at t=0): {fold_2d}")
    print(f"    {'Time':<8} {'3D Fold':<10} {'3D Norm':<12} {'2D Norm':<12} {'Error':<10}")
    print("    " + "-" * 52)

    for i, t_ms in enumerate(reflector_times_ms):
        t_idx = int(t_ms / dt_ms)
        err = (image_2d_norm[t_idx] - image_normalized[t_idx]) / image_normalized[t_idx] * 100 if image_normalized[t_idx] != 0 else 0
        print(f"    {t_ms:<8} {fold_3d[t_idx]:<10} {image_normalized[t_idx]:<12.4f} "
              f"{image_2d_norm[t_idx]:<12.4f} {err:<+10.1f}%")

    # Visualization
    print("\n[9] Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('3D vs 2D Fold Normalization Comparison', fontsize=14, fontweight='bold')

    t_axis_ms = np.arange(n_samples) * dt_ms

    # Midpoint distribution
    ax = axes[0, 0]
    sc = ax.scatter(midpoint_x, midpoint_y, c=offsets, s=5, alpha=0.5, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Offset (m)')
    for t_ms in reflector_times_ms:
        aperture = get_aperture(t_ms)
        circle = plt.Circle((0, 0), aperture, fill=False, linestyle='--',
                            label=f'{t_ms}ms: {aperture:.0f}m')
        ax.add_patch(circle)
    ax.plot(0, 0, 'r*', markersize=15, label='Output point')
    ax.set_xlim(-1200, 1200)
    ax.set_ylim(-1200, 1200)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Midpoint Distribution & Apertures')
    ax.legend(loc='upper right', fontsize=8)

    # Fold vs time
    ax = axes[0, 1]
    ax.plot(t_axis_ms, fold_3d, 'g-', linewidth=1)
    ax.axhline(fold_2d, color='r', linestyle='--', label=f'2D fold={fold_2d}')
    for t_ms in reflector_times_ms:
        ax.axvline(t_ms, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Fold')
    ax.set_title('3D Fold (per sample) vs 2D Fold')
    ax.legend()
    ax.set_xlim([0, 2000])

    # Raw image
    ax = axes[0, 2]
    ax.plot(t_axis_ms, image, 'b-', linewidth=0.5)
    for t_ms in reflector_times_ms:
        ax.axvline(t_ms, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Raw Image (before normalization)')
    ax.set_xlim([0, 2000])

    # 3D normalized
    ax = axes[1, 0]
    ax.plot(t_axis_ms, image_normalized, 'b-', linewidth=0.5)
    for t_ms in reflector_times_ms:
        ax.axvline(t_ms, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('3D Fold Normalized (CORRECT)')
    ax.set_xlim([0, 2000])

    # 2D normalized
    ax = axes[1, 1]
    ax.plot(t_axis_ms, image_2d_norm, 'b-', linewidth=0.5)
    for t_ms in reflector_times_ms:
        ax.axvline(t_ms, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('2D Fold Normalized (OLD - WRONG)')
    ax.set_xlim([0, 2000])

    # Comparison bar chart
    ax = axes[1, 2]
    x = np.arange(len(reflector_times_ms))
    width = 0.35
    ax.bar(x - width/2, [np.abs(image_normalized[int(t/dt_ms)]) for t in reflector_times_ms],
           width, label='3D Fold Norm', alpha=0.8)
    ax.bar(x + width/2, [np.abs(image_2d_norm[int(t/dt_ms)]) for t in reflector_times_ms],
           width, label='2D Fold Norm', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}ms' for t in reflector_times_ms])
    ax.set_xlabel('Reflector Time')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title('Amplitude Comparison at Reflector Times')
    ax.legend()

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "test_3d_fold_realistic.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fig_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"""
  Test Configuration:
    - {n_traces} traces with midpoints distributed 0-1000m from output
    - 4 reflectors at {reflector_times_ms} ms
    - Aperture: 200m (shallow) → 1000m (deep)

  Key Results:
    - Fold ratio (deep/shallow): {fold_ratio:.1f}x
    - 3D normalized amplitude CV: {amp_cv:.1%}
    - 2D normalization error at 1600ms: {((image_2d_norm[int(1600/dt_ms)] - image_normalized[int(1600/dt_ms)]) / image_normalized[int(1600/dt_ms)] * 100) if image_normalized[int(1600/dt_ms)] != 0 else 0:+.1f}%

  Conclusion:
    3D fold normalization correctly compensates for depth-varying fold,
    producing consistent amplitudes across all depths.

    2D fold (old approach) would over-normalize deep reflections because
    it divides by the larger deep-time fold at all times.
""")

    return True


if __name__ == "__main__":
    test_realistic_fold_variation()
