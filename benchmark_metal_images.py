#!/usr/bin/env python3
"""Generate and compare images from compiled Metal kernel with different corrections."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 70)
print("Compiled Metal - Image Comparison with Different Corrections")
print("=" * 70)

# Output directory
output_dir = Path("metal_correction_images")
output_dir.mkdir(exist_ok=True)

# Test configuration - larger grid for better visualization
n_traces = 15000
n_samples = 1000
nx, ny, nt = 81, 81, 500

np.random.seed(42)
survey_size = 2000.0

# Mock classes
class MockTraceBlock:
    def __init__(self):
        self.n_traces = n_traces
        self.n_samples = n_samples
        self.amplitudes = np.ascontiguousarray(np.random.randn(n_traces, n_samples).astype(np.float32))
        self.source_x = np.ascontiguousarray(np.random.uniform(0, survey_size, n_traces).astype(np.float64))
        self.source_y = np.ascontiguousarray(np.random.uniform(0, survey_size, n_traces).astype(np.float64))
        self.receiver_x = np.ascontiguousarray(np.random.uniform(0, survey_size, n_traces).astype(np.float64))
        self.receiver_y = np.ascontiguousarray(np.random.uniform(0, survey_size, n_traces).astype(np.float64))
        self.midpoint_x = np.ascontiguousarray((self.source_x + self.receiver_x) / 2)
        self.midpoint_y = np.ascontiguousarray((self.source_y + self.receiver_y) / 2)
        self.offset = np.ascontiguousarray(np.sqrt((self.receiver_x - self.source_x)**2 + (self.receiver_y - self.source_y)**2))
        self.sample_rate_ms = 2.0
        self.start_time_ms = 0.0

class MockOutputTile:
    def __init__(self):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.x_axis = np.linspace(0, survey_size, nx).astype(np.float64)
        self.y_axis = np.linspace(0, survey_size, ny).astype(np.float64)
        self.t_axis_ms = np.linspace(0, 1000, nt).astype(np.float64)
        self.image = np.zeros((nx, ny, nt), dtype=np.float64)
        self.fold = np.zeros((nx, ny), dtype=np.int32)

class MockVelocity:
    def __init__(self):
        self.is_1d = True
        self.vrms = np.full(nt, 2500.0, dtype=np.float32)

class MockConfig:
    def __init__(self, apply_spreading=False, apply_obliquity=False, apply_aa=False):
        self.max_aperture_m = 1500.0
        self.min_aperture_m = 100.0
        self.taper_fraction = 0.1
        self.max_dip_degrees = 60.0
        self.apply_spreading = apply_spreading
        self.apply_obliquity = apply_obliquity
        self.apply_aa = apply_aa
        self.aa_dominant_freq = 30.0

print(f"\nTest Configuration:")
print(f"  Traces: {n_traces:,}")
print(f"  Output grid: {nx}x{ny}x{nt}")
print(f"  Output dir: {output_dir}")

# Check Metal availability
from pstm.kernels.metal_compiled import CompiledMetalKernel, check_metal_available

if not check_metal_available():
    print("\nERROR: Metal not available")
    exit(1)

# Shared data
traces = MockTraceBlock()
velocity = MockVelocity()

# Test configurations
test_configs = [
    ("no_corrections", "No Corrections", {"apply_spreading": False, "apply_obliquity": False, "apply_aa": False}),
    ("aa_only", "AA Only", {"apply_spreading": False, "apply_obliquity": False, "apply_aa": True}),
    ("spreading_only", "Spreading Only", {"apply_spreading": True, "apply_obliquity": False, "apply_aa": False}),
    ("obliquity_only", "Obliquity Only", {"apply_spreading": False, "apply_obliquity": True, "apply_aa": False}),
    ("spreading_obliquity", "Spreading + Obliquity", {"apply_spreading": True, "apply_obliquity": True, "apply_aa": False}),
    ("all_corrections", "All Corrections", {"apply_spreading": True, "apply_obliquity": True, "apply_aa": True}),
]

# Store results
images = {}
stats = {}

# Run migrations
for key, name, cfg_params in test_configs:
    print(f"\nProcessing: {name}...")

    config = MockConfig(**cfg_params)
    kernel = CompiledMetalKernel(use_simd=True)
    kernel.initialize(config)

    output = MockOutputTile()
    kernel.migrate_tile(traces, output, velocity, config)

    images[key] = output.image.copy()

    # Calculate statistics
    img = output.image
    stats[key] = {
        "name": name,
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "rms": float(np.sqrt(np.mean(img**2))),
        "absmax": float(np.max(np.abs(img))),
    }

    kernel.cleanup()
    print(f"  Done. RMS: {stats[key]['rms']:.2e}, Max: {stats[key]['absmax']:.2e}")

# Slice indices
inline_idx = nx // 2  # Middle inline
xline_idx = ny // 2   # Middle crossline
time_idx = nt // 2    # Middle time slice

t_axis = np.linspace(0, 1000, nt)
x_axis = np.linspace(0, survey_size, nx)
y_axis = np.linspace(0, survey_size, ny)

# Create comparison figures
print("\n" + "=" * 70)
print("Generating comparison images...")
print("=" * 70)

# 1. Inline comparison (all configs side by side)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (key, name, _) in enumerate(test_configs):
    ax = axes[i]
    img = images[key]
    inline_slice = img[inline_idx, :, :].T  # (nt, ny)

    # Normalize for display
    vmax = np.percentile(np.abs(inline_slice), 99) if np.any(inline_slice != 0) else 1

    im = ax.imshow(inline_slice, aspect='auto', cmap='seismic',
                   extent=[y_axis[0], y_axis[-1], t_axis[-1], t_axis[0]],
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"{name}\nRMS: {stats[key]['rms']:.2e}", fontsize=10)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Time (ms)")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle(f"Inline {inline_idx} Comparison (X = {x_axis[inline_idx]:.0f} m)", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "inline_comparison.png", dpi=150)
plt.close()
print(f"  Saved: {output_dir / 'inline_comparison.png'}")

# 2. Crossline comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (key, name, _) in enumerate(test_configs):
    ax = axes[i]
    img = images[key]
    xline_slice = img[:, xline_idx, :].T  # (nt, nx)

    vmax = np.percentile(np.abs(xline_slice), 99) if np.any(xline_slice != 0) else 1

    im = ax.imshow(xline_slice, aspect='auto', cmap='seismic',
                   extent=[x_axis[0], x_axis[-1], t_axis[-1], t_axis[0]],
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"{name}\nRMS: {stats[key]['rms']:.2e}", fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Time (ms)")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle(f"Crossline {xline_idx} Comparison (Y = {y_axis[xline_idx]:.0f} m)", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "crossline_comparison.png", dpi=150)
plt.close()
print(f"  Saved: {output_dir / 'crossline_comparison.png'}")

# 3. Time slice comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (key, name, _) in enumerate(test_configs):
    ax = axes[i]
    img = images[key]
    time_slice = img[:, :, time_idx].T  # (ny, nx)

    vmax = np.percentile(np.abs(time_slice), 99) if np.any(time_slice != 0) else 1

    im = ax.imshow(time_slice, aspect='equal', cmap='seismic',
                   extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]],
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"{name}\nRMS: {stats[key]['rms']:.2e}", fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle(f"Time Slice at {t_axis[time_idx]:.0f} ms", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "timeslice_comparison.png", dpi=150)
plt.close()
print(f"  Saved: {output_dir / 'timeslice_comparison.png'}")

# 4. Difference images (relative to "no corrections")
print("\nGenerating difference images...")

baseline = images["no_corrections"]
baseline_rms = stats["no_corrections"]["rms"]

diff_stats = {}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

compare_keys = ["aa_only", "spreading_only", "obliquity_only", "spreading_obliquity", "all_corrections"]

for i, key in enumerate(compare_keys):
    ax = axes[i]

    # Calculate difference
    diff = images[key] - baseline
    diff_inline = diff[inline_idx, :, :].T

    # Statistics
    diff_rms = np.sqrt(np.mean(diff**2))
    diff_max = np.max(np.abs(diff))
    rel_diff = 100 * diff_rms / baseline_rms if baseline_rms > 0 else 0

    diff_stats[key] = {
        "name": stats[key]["name"],
        "diff_rms": diff_rms,
        "diff_max": diff_max,
        "rel_diff_pct": rel_diff,
        "correlation": np.corrcoef(images[key].flatten(), baseline.flatten())[0, 1],
    }

    vmax = np.percentile(np.abs(diff_inline), 99) if np.any(diff_inline != 0) else 1

    im = ax.imshow(diff_inline, aspect='auto', cmap='seismic',
                   extent=[y_axis[0], y_axis[-1], t_axis[-1], t_axis[0]],
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"{stats[key]['name']} - No Corr\nDiff RMS: {diff_rms:.2e} ({rel_diff:.1f}%)", fontsize=10)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Time (ms)")
    plt.colorbar(im, ax=ax, shrink=0.8)

# Empty last subplot - add summary text
ax = axes[5]
ax.axis('off')
summary_text = "DIFFERENCE SUMMARY\n" + "=" * 30 + "\n\n"
for key in compare_keys:
    ds = diff_stats[key]
    summary_text += f"{ds['name']}:\n"
    summary_text += f"  Diff RMS: {ds['diff_rms']:.2e}\n"
    summary_text += f"  Rel Diff: {ds['rel_diff_pct']:.1f}%\n"
    summary_text += f"  Corr: {ds['correlation']:.4f}\n\n"
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace')

plt.suptitle(f"Inline {inline_idx} - Difference from No Corrections", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "inline_differences.png", dpi=150)
plt.close()
print(f"  Saved: {output_dir / 'inline_differences.png'}")

# 5. Time slice differences
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, key in enumerate(compare_keys):
    ax = axes[i]

    diff = images[key] - baseline
    diff_slice = diff[:, :, time_idx].T

    vmax = np.percentile(np.abs(diff_slice), 99) if np.any(diff_slice != 0) else 1

    im = ax.imshow(diff_slice, aspect='equal', cmap='seismic',
                   extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]],
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"{stats[key]['name']} - No Corr\nDiff RMS: {diff_stats[key]['diff_rms']:.2e}", fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[5]
ax.axis('off')

plt.suptitle(f"Time Slice at {t_axis[time_idx]:.0f} ms - Difference from No Corrections", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "timeslice_differences.png", dpi=150)
plt.close()
print(f"  Saved: {output_dir / 'timeslice_differences.png'}")

# 6. Normalized comparison (same scale for all)
print("\nGenerating normalized comparison...")

# Find global max for normalization
all_rms = [stats[k]["rms"] for k in images.keys()]
max_rms = max(all_rms)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (key, name, _) in enumerate(test_configs):
    ax = axes[i]
    img = images[key]

    # Normalize by its own RMS for fair comparison
    img_rms = stats[key]["rms"]
    if img_rms > 0:
        img_normalized = img / img_rms
    else:
        img_normalized = img

    inline_slice = img_normalized[inline_idx, :, :].T

    vmax = 5  # Fixed scale for normalized data

    im = ax.imshow(inline_slice, aspect='auto', cmap='seismic',
                   extent=[y_axis[0], y_axis[-1], t_axis[-1], t_axis[0]],
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"{name}\n(Normalized by RMS)", fontsize=10)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Time (ms)")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle(f"Inline {inline_idx} - Normalized Comparison (Data / RMS)", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "inline_normalized.png", dpi=150)
plt.close()
print(f"  Saved: {output_dir / 'inline_normalized.png'}")

# Print summary tables
print("\n" + "=" * 70)
print("IMAGE STATISTICS SUMMARY")
print("=" * 70)
print(f"{'Configuration':<25} {'RMS':<12} {'Max Abs':<12} {'Mean':<12}")
print("-" * 70)
for key, name, _ in test_configs:
    s = stats[key]
    print(f"{name:<25} {s['rms']:<12.2e} {s['absmax']:<12.2e} {s['mean']:<12.2e}")

print("\n" + "=" * 70)
print("DIFFERENCE ANALYSIS (vs No Corrections)")
print("=" * 70)
print(f"{'Configuration':<25} {'Diff RMS':<12} {'Rel Diff %':<12} {'Correlation':<12}")
print("-" * 70)
for key in compare_keys:
    ds = diff_stats[key]
    print(f"{ds['name']:<25} {ds['diff_rms']:<12.2e} {ds['rel_diff_pct']:<12.1f} {ds['correlation']:<12.4f}")

print("\n" + "=" * 70)
print(f"All images saved to: {output_dir.absolute()}")
print("=" * 70)
