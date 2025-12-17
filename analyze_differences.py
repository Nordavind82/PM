#!/usr/bin/env python3
"""
Analyze differences between anti-aliasing benchmark results.
Computes and visualizes differences between migration outputs.
"""

import numpy as np
from pathlib import Path
import zarr
import matplotlib.pyplot as plt


def load_crossline_slice(output_dir: Path, crossline_idx: int) -> np.ndarray:
    """Load crossline slice from migration output."""
    stack_path = output_dir / "migrated_stack.zarr"
    z = zarr.open(str(stack_path), mode='r')

    if isinstance(z, zarr.Array):
        data = np.array(z)
    else:
        key = list(z.keys())[0]
        data = np.array(z[key])

    return data[:, crossline_idx, :]


def compute_difference_stats(data1: np.ndarray, data2: np.ndarray, name1: str, name2: str) -> dict:
    """Compute statistics on the difference between two arrays."""
    diff = data1 - data2

    # Handle NaN values
    valid_diff = diff[~np.isnan(diff)]
    valid_data1 = data1[~np.isnan(data1)]

    if len(valid_diff) == 0 or len(valid_data1) == 0:
        return {"error": "No valid data"}

    max_abs_diff = np.max(np.abs(valid_diff))
    mean_abs_diff = np.mean(np.abs(valid_diff))
    rms_diff = np.sqrt(np.mean(valid_diff**2))

    # Relative differences (normalized by max of reference)
    max_ref = np.max(np.abs(valid_data1))
    if max_ref > 1e-15:
        rel_max_diff = max_abs_diff / max_ref * 100
        rel_rms_diff = rms_diff / max_ref * 100
    else:
        rel_max_diff = 0
        rel_rms_diff = 0

    # Correlation coefficient
    if np.std(valid_data1) > 1e-15 and np.std(data2[~np.isnan(data2)]) > 1e-15:
        corr = np.corrcoef(valid_data1.flatten(), data2[~np.isnan(data2)].flatten())[0, 1]
    else:
        corr = 1.0

    return {
        "comparison": f"{name1} vs {name2}",
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rms_diff": rms_diff,
        "rel_max_diff_pct": rel_max_diff,
        "rel_rms_diff_pct": rel_rms_diff,
        "correlation": corr,
        "diff_array": diff,
    }


def main():
    base_dir = Path("./benchmark_antialiasing_output")

    # Crossline index through diffractor (Y=1000m, dy=50m)
    crossline_idx = 20

    # Load all slices
    configs = ["no_antialiasing", "triangle_aa", "triangle_aa_spreading", "triangle_aa_both_amp"]
    labels = ["No AA", "Triangle AA", "AA + Spreading", "AA + Both"]

    slices = {}
    for config in configs:
        try:
            slices[config] = load_crossline_slice(base_dir / config, crossline_idx)
            print(f"Loaded {config}: shape={slices[config].shape}, "
                  f"range=[{np.nanmin(slices[config]):.6f}, {np.nanmax(slices[config]):.6f}]")
        except Exception as e:
            print(f"Error loading {config}: {e}")
            return

    print("\n" + "=" * 70)
    print("DIFFERENCE ANALYSIS")
    print("=" * 70)

    # Compute differences
    comparisons = [
        ("no_antialiasing", "triangle_aa", "Effect of Anti-Aliasing"),
        ("triangle_aa", "triangle_aa_spreading", "Effect of Spreading (with AA)"),
        ("triangle_aa_spreading", "triangle_aa_both_amp", "Effect of Obliquity (with AA+Spreading)"),
        ("no_antialiasing", "triangle_aa_both_amp", "Total Effect (No AA vs Full Processing)"),
    ]

    stats_list = []
    for config1, config2, desc in comparisons:
        stats = compute_difference_stats(
            slices[config1], slices[config2], config1, config2
        )
        stats["description"] = desc
        stats_list.append(stats)

        print(f"\n{desc}:")
        print(f"  Max absolute difference: {stats['max_abs_diff']:.6e}")
        print(f"  Mean absolute difference: {stats['mean_abs_diff']:.6e}")
        print(f"  RMS difference: {stats['rms_diff']:.6e}")
        print(f"  Relative max diff: {stats['rel_max_diff_pct']:.4f}%")
        print(f"  Relative RMS diff: {stats['rel_rms_diff_pct']:.4f}%")
        print(f"  Correlation: {stats['correlation']:.6f}")

    # Create visualization
    fig = plt.figure(figsize=(16, 14))

    # Top row: Original images (normalized to same scale for comparison)
    # Normalize all to the no_antialiasing scale for fair visual comparison
    ref_data = slices["no_antialiasing"]
    ref_clip = np.percentile(np.abs(ref_data[~np.isnan(ref_data)]), 99)

    ax1 = fig.add_subplot(3, 4, 1)
    im1 = ax1.imshow(slices["no_antialiasing"].T, aspect='auto', extent=[0, 41, 2000, 0],
                     cmap='seismic', vmin=-ref_clip, vmax=ref_clip)
    ax1.set_title("No Anti-Aliasing")
    ax1.set_ylabel("Time (ms)")
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    ax2 = fig.add_subplot(3, 4, 2)
    im2 = ax2.imshow(slices["triangle_aa"].T, aspect='auto', extent=[0, 41, 2000, 0],
                     cmap='seismic', vmin=-ref_clip, vmax=ref_clip)
    ax2.set_title("Triangle AA")
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    # For spreading/both, use their own scale (much larger amplitudes)
    spread_data = slices["triangle_aa_spreading"]
    spread_clip = np.percentile(np.abs(spread_data[~np.isnan(spread_data)]), 99)

    ax3 = fig.add_subplot(3, 4, 3)
    im3 = ax3.imshow(slices["triangle_aa_spreading"].T, aspect='auto', extent=[0, 41, 2000, 0],
                     cmap='seismic', vmin=-spread_clip, vmax=spread_clip)
    ax3.set_title("AA + Spreading")
    plt.colorbar(im3, ax=ax3, shrink=0.7)

    both_data = slices["triangle_aa_both_amp"]
    both_clip = np.percentile(np.abs(both_data[~np.isnan(both_data)]), 99)

    ax4 = fig.add_subplot(3, 4, 4)
    im4 = ax4.imshow(slices["triangle_aa_both_amp"].T, aspect='auto', extent=[0, 41, 2000, 0],
                     cmap='seismic', vmin=-both_clip, vmax=both_clip)
    ax4.set_title("AA + Both Corrections")
    plt.colorbar(im4, ax=ax4, shrink=0.7)

    # Middle row: Difference images
    diff_cmap = 'RdBu_r'

    # No AA vs Triangle AA difference
    diff1 = stats_list[0]["diff_array"]
    diff1_clip = np.percentile(np.abs(diff1[~np.isnan(diff1)]), 99)
    diff1_clip = max(diff1_clip, 1e-15)

    ax5 = fig.add_subplot(3, 4, 5)
    im5 = ax5.imshow(diff1.T, aspect='auto', extent=[0, 41, 2000, 0],
                     cmap=diff_cmap, vmin=-diff1_clip, vmax=diff1_clip)
    ax5.set_title(f"Diff: No AA - Triangle AA\nRMS: {stats_list[0]['rms_diff']:.2e}")
    ax5.set_ylabel("Time (ms)")
    plt.colorbar(im5, ax=ax5, shrink=0.7)

    # Triangle AA vs AA+Spreading difference
    diff2 = stats_list[1]["diff_array"]
    diff2_clip = np.percentile(np.abs(diff2[~np.isnan(diff2)]), 99)
    diff2_clip = max(diff2_clip, 1e-15)

    ax6 = fig.add_subplot(3, 4, 6)
    im6 = ax6.imshow(diff2.T, aspect='auto', extent=[0, 41, 2000, 0],
                     cmap=diff_cmap, vmin=-diff2_clip, vmax=diff2_clip)
    ax6.set_title(f"Diff: AA - AA+Spread\nRMS: {stats_list[1]['rms_diff']:.2e}")
    plt.colorbar(im6, ax=ax6, shrink=0.7)

    # AA+Spreading vs AA+Both difference
    diff3 = stats_list[2]["diff_array"]
    diff3_clip = np.percentile(np.abs(diff3[~np.isnan(diff3)]), 99)
    diff3_clip = max(diff3_clip, 1e-15)

    ax7 = fig.add_subplot(3, 4, 7)
    im7 = ax7.imshow(diff3.T, aspect='auto', extent=[0, 41, 2000, 0],
                     cmap=diff_cmap, vmin=-diff3_clip, vmax=diff3_clip)
    ax7.set_title(f"Diff: AA+Spread - AA+Both\nRMS: {stats_list[2]['rms_diff']:.2e}")
    plt.colorbar(im7, ax=ax7, shrink=0.7)

    # Total difference: No AA vs AA+Both
    diff4 = stats_list[3]["diff_array"]
    diff4_clip = np.percentile(np.abs(diff4[~np.isnan(diff4)]), 99)
    diff4_clip = max(diff4_clip, 1e-15)

    ax8 = fig.add_subplot(3, 4, 8)
    im8 = ax8.imshow(diff4.T, aspect='auto', extent=[0, 41, 2000, 0],
                     cmap=diff_cmap, vmin=-diff4_clip, vmax=diff4_clip)
    ax8.set_title(f"Diff: No AA - AA+Both\nRMS: {stats_list[3]['rms_diff']:.2e}")
    plt.colorbar(im8, ax=ax8, shrink=0.7)

    # Bottom row: Normalized differences (percentage of max amplitude)
    ax9 = fig.add_subplot(3, 4, 9)
    norm_diff1 = diff1 / ref_clip * 100
    im9 = ax9.imshow(norm_diff1.T, aspect='auto', extent=[0, 41, 2000, 0],
                     cmap=diff_cmap, vmin=-10, vmax=10)
    ax9.set_title(f"Normalized Diff (%)\nNo AA - Triangle AA")
    ax9.set_ylabel("Time (ms)")
    ax9.set_xlabel("Inline Index")
    plt.colorbar(im9, ax=ax9, shrink=0.7, label='%')

    ax10 = fig.add_subplot(3, 4, 10)
    norm_diff2 = diff2 / spread_clip * 100
    im10 = ax10.imshow(norm_diff2.T, aspect='auto', extent=[0, 41, 2000, 0],
                      cmap=diff_cmap, vmin=-100, vmax=100)
    ax10.set_title(f"Normalized Diff (%)\nAA - AA+Spread")
    ax10.set_xlabel("Inline Index")
    plt.colorbar(im10, ax=ax10, shrink=0.7, label='%')

    ax11 = fig.add_subplot(3, 4, 11)
    norm_diff3 = diff3 / both_clip * 100
    im11 = ax11.imshow(norm_diff3.T, aspect='auto', extent=[0, 41, 2000, 0],
                      cmap=diff_cmap, vmin=-100, vmax=100)
    ax11.set_title(f"Normalized Diff (%)\nAA+Spread - AA+Both")
    ax11.set_xlabel("Inline Index")
    plt.colorbar(im11, ax=ax11, shrink=0.7, label='%')

    ax12 = fig.add_subplot(3, 4, 12)
    norm_diff4 = diff4 / ref_clip * 100
    im12 = ax12.imshow(norm_diff4.T, aspect='auto', extent=[0, 41, 2000, 0],
                      cmap=diff_cmap, vmin=-100, vmax=100)
    ax12.set_title(f"Normalized Diff (%)\nNo AA - AA+Both")
    ax12.set_xlabel("Inline Index")
    plt.colorbar(im12, ax=ax12, shrink=0.7, label='%')

    plt.suptitle("Anti-Aliasing Difference Analysis\n"
                 "Top: Original Images | Middle: Absolute Differences | Bottom: Normalized Differences (%)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(base_dir / "difference_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved difference analysis to: {base_dir / 'difference_analysis.png'}")
    plt.close()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Comparison':<40} {'Max Diff':<12} {'RMS Diff':<12} {'Rel RMS %':<12} {'Corr':<8}")
    print("-" * 84)
    for stats in stats_list:
        print(f"{stats['description']:<40} {stats['max_abs_diff']:<12.2e} "
              f"{stats['rms_diff']:<12.2e} {stats['rel_rms_diff_pct']:<12.4f} "
              f"{stats['correlation']:<8.6f}")

    # Check if AA actually makes a difference
    aa_diff_stats = stats_list[0]
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if aa_diff_stats['rel_rms_diff_pct'] < 0.1:
        print("⚠️  Anti-aliasing has MINIMAL effect (<0.1% RMS difference)")
        print("   This may be because:")
        print("   - The synthetic data has limited high-frequency content")
        print("   - The wavelet frequency (30 Hz) is well below Nyquist")
        print("   - The offset range is small, limiting steep-dip aliasing")
    else:
        print(f"✓ Anti-aliasing shows {aa_diff_stats['rel_rms_diff_pct']:.2f}% RMS difference")


if __name__ == "__main__":
    main()
