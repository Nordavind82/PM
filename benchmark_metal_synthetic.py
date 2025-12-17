#!/usr/bin/env python3
"""Benchmark compiled Metal kernel using real synthetic data with point diffractor."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zarr
import polars as pl

print("=" * 70)
print("Compiled Metal - Synthetic Diffractor Migration")
print("=" * 70)

# Output directory
output_dir = Path("metal_synthetic_images")
output_dir.mkdir(exist_ok=True)

# Load synthetic data
traces_path = Path("synthetic_output2/diffractor_traces.zarr")
headers_path = Path("synthetic_output2/diffractor_headers.parquet")

print(f"\nLoading synthetic data...")
print(f"  Traces: {traces_path}")
print(f"  Headers: {headers_path}")

# Load traces
z = zarr.open(traces_path, mode='r')
traces_data = np.array(z[:])
print(f"  Traces shape: {traces_data.shape}")

# Load headers
df = pl.read_parquet(headers_path)
print(f"  Headers: {len(df)} rows")
print(f"  Columns: {df.columns}")

# Extract geometry
n_traces = len(df)
n_samples = traces_data.shape[1]

source_x = df["SOU_X"].to_numpy().astype(np.float64)
source_y = df["SOU_Y"].to_numpy().astype(np.float64)
receiver_x = df["REC_X"].to_numpy().astype(np.float64)
receiver_y = df["REC_Y"].to_numpy().astype(np.float64)
midpoint_x = df["CDP_X"].to_numpy().astype(np.float64)
midpoint_y = df["CDP_Y"].to_numpy().astype(np.float64)
offset = df["OFFSET"].to_numpy().astype(np.float64)

print(f"\nGeometry summary:")
print(f"  X range: {midpoint_x.min():.0f} - {midpoint_x.max():.0f} m")
print(f"  Y range: {midpoint_y.min():.0f} - {midpoint_y.max():.0f} m")
print(f"  Offset range: {offset.min():.0f} - {offset.max():.0f} m")

# Diffractor location (from synthetic_gather_2.py example_4)
diffractor_x = 2000.0
diffractor_y = 2000.0
diffractor_z = 800.0  # depth
velocity = 2500.0

print(f"\nDiffractor location: ({diffractor_x}, {diffractor_y}, {diffractor_z}) m")
print(f"Velocity: {velocity} m/s")

# Expected two-way time at diffractor
t0_ms = 2 * diffractor_z / velocity * 1000
print(f"Expected t0 at diffractor: {t0_ms:.0f} ms")

# Output grid - centered on diffractor
margin = 1000.0
x_min = diffractor_x - margin
x_max = diffractor_x + margin
y_min = diffractor_y - margin
y_max = diffractor_y + margin
dx, dy = 25.0, 25.0
dt_ms = 2.0
t_min_ms = 0.0
t_max_ms = 1500.0

nx = int((x_max - x_min) / dx) + 1
ny = int((y_max - y_min) / dy) + 1
nt = int((t_max_ms - t_min_ms) / dt_ms) + 1

print(f"\nOutput grid:")
print(f"  X: {x_min:.0f} - {x_max:.0f} m ({nx} bins)")
print(f"  Y: {y_min:.0f} - {y_max:.0f} m ({ny} bins)")
print(f"  T: {t_min_ms:.0f} - {t_max_ms:.0f} ms ({nt} samples)")

# Create trace block
class TraceBlock:
    def __init__(self):
        self.n_traces = n_traces
        self.n_samples = n_samples
        self.amplitudes = np.ascontiguousarray(traces_data.astype(np.float32))
        self.source_x = np.ascontiguousarray(source_x)
        self.source_y = np.ascontiguousarray(source_y)
        self.receiver_x = np.ascontiguousarray(receiver_x)
        self.receiver_y = np.ascontiguousarray(receiver_y)
        self.midpoint_x = np.ascontiguousarray(midpoint_x)
        self.midpoint_y = np.ascontiguousarray(midpoint_y)
        self.offset = np.ascontiguousarray(offset)
        self.sample_rate_ms = 2.0
        self.start_time_ms = 0.0

class OutputTile:
    def __init__(self):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.x_axis = np.linspace(x_min, x_max, nx).astype(np.float64)
        self.y_axis = np.linspace(y_min, y_max, ny).astype(np.float64)
        self.t_axis_ms = np.linspace(t_min_ms, t_max_ms, nt).astype(np.float64)
        self.image = np.zeros((nx, ny, nt), dtype=np.float64)
        self.fold = np.zeros((nx, ny), dtype=np.int32)

class VelocityModel:
    def __init__(self):
        self.is_1d = True
        self.vrms = np.full(nt, velocity, dtype=np.float32)

class Config:
    def __init__(self, apply_spreading=False, apply_obliquity=False, apply_aa=False):
        self.max_aperture_m = 1500.0
        self.min_aperture_m = 100.0
        self.taper_fraction = 0.1
        self.max_dip_degrees = 60.0
        self.apply_spreading = apply_spreading
        self.apply_obliquity = apply_obliquity
        self.apply_aa = apply_aa
        self.aa_dominant_freq = 25.0  # Match wavelet frequency

# Check Metal
from pstm.kernels.metal_compiled import CompiledMetalKernel, check_metal_available

if not check_metal_available():
    print("\nERROR: Metal not available")
    exit(1)

# Shared data
traces = TraceBlock()
velocity_model = VelocityModel()

# Test configurations
test_configs = [
    ("no_corrections", "No Corrections", {"apply_spreading": False, "apply_obliquity": False, "apply_aa": False}),
    ("aa_only", "AA Only", {"apply_spreading": False, "apply_obliquity": False, "apply_aa": True}),
    ("spreading_only", "Spreading Only", {"apply_spreading": True, "apply_obliquity": False, "apply_aa": False}),
    ("obliquity_only", "Obliquity Only", {"apply_spreading": False, "apply_obliquity": True, "apply_aa": False}),
    ("all_corrections", "All Corrections", {"apply_spreading": True, "apply_obliquity": True, "apply_aa": True}),
]

# Store results
images = {}
stats = {}

print("\n" + "=" * 70)
print("Running migrations...")
print("=" * 70)

for key, name, cfg_params in test_configs:
    print(f"\nProcessing: {name}...")

    config = Config(**cfg_params)
    kernel = CompiledMetalKernel(use_simd=True)
    kernel.initialize(config)

    output = OutputTile()
    import time
    start = time.perf_counter()
    kernel.migrate_tile(traces, output, velocity_model, config)
    elapsed = time.perf_counter() - start

    images[key] = output.image.copy()

    # Statistics
    img = output.image
    stats[key] = {
        "name": name,
        "time": elapsed,
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "rms": float(np.sqrt(np.mean(img**2))),
        "absmax": float(np.max(np.abs(img))),
    }

    kernel.cleanup()
    print(f"  Done in {elapsed:.3f}s. RMS: {stats[key]['rms']:.2e}")

# Axes
x_axis = np.linspace(x_min, x_max, nx)
y_axis = np.linspace(y_min, y_max, ny)
t_axis = np.linspace(t_min_ms, t_max_ms, nt)

# Find diffractor position in grid
diff_ix = int((diffractor_x - x_min) / dx)
diff_iy = int((diffractor_y - y_min) / dy)
diff_it = int(t0_ms / dt_ms)

print(f"\nDiffractor grid position: ix={diff_ix}, iy={diff_iy}, it={diff_it}")

# Generate comparison images
print("\n" + "=" * 70)
print("Generating images...")
print("=" * 70)

# 1. Inline through diffractor
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (key, name, _) in enumerate(test_configs):
    ax = axes[i]
    img = images[key]
    inline_slice = img[diff_ix, :, :].T  # (nt, ny)

    vmax = np.percentile(np.abs(inline_slice), 99) if np.any(inline_slice != 0) else 1

    im = ax.imshow(inline_slice, aspect='auto', cmap='seismic',
                   extent=[y_axis[0], y_axis[-1], t_axis[-1], t_axis[0]],
                   vmin=-vmax, vmax=vmax)
    ax.axhline(y=t0_ms, color='green', linestyle='--', alpha=0.5, label=f't0={t0_ms:.0f}ms')
    ax.axvline(x=diffractor_y, color='yellow', linestyle='--', alpha=0.5)
    ax.set_title(f"{name}\nRMS: {stats[key]['rms']:.2e}", fontsize=10)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Time (ms)")
    plt.colorbar(im, ax=ax, shrink=0.8)

# Summary in last panel
ax = axes[5]
ax.axis('off')
summary = f"DIFFRACTOR MIGRATION\n{'='*30}\n\n"
summary += f"Diffractor: ({diffractor_x:.0f}, {diffractor_y:.0f}, {diffractor_z:.0f}m)\n"
summary += f"Velocity: {velocity:.0f} m/s\n"
summary += f"Expected t0: {t0_ms:.0f} ms\n\n"
summary += f"Input: {n_traces:,} traces\n"
summary += f"Output: {nx}x{ny}x{nt}\n\n"
summary += "PERFORMANCE:\n"
for key, name, _ in test_configs:
    summary += f"  {name}: {stats[key]['time']:.3f}s\n"
ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace')

plt.suptitle(f"Inline through Diffractor (X = {diffractor_x:.0f} m)", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "inline_diffractor.png", dpi=150)
plt.close()
print(f"  Saved: {output_dir / 'inline_diffractor.png'}")

# 2. Crossline through diffractor
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (key, name, _) in enumerate(test_configs):
    ax = axes[i]
    img = images[key]
    xline_slice = img[:, diff_iy, :].T  # (nt, nx)

    vmax = np.percentile(np.abs(xline_slice), 99) if np.any(xline_slice != 0) else 1

    im = ax.imshow(xline_slice, aspect='auto', cmap='seismic',
                   extent=[x_axis[0], x_axis[-1], t_axis[-1], t_axis[0]],
                   vmin=-vmax, vmax=vmax)
    ax.axhline(y=t0_ms, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=diffractor_x, color='yellow', linestyle='--', alpha=0.5)
    ax.set_title(f"{name}\nRMS: {stats[key]['rms']:.2e}", fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Time (ms)")
    plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[5]
ax.axis('off')

plt.suptitle(f"Crossline through Diffractor (Y = {diffractor_y:.0f} m)", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "crossline_diffractor.png", dpi=150)
plt.close()
print(f"  Saved: {output_dir / 'crossline_diffractor.png'}")

# 3. Time slice at diffractor time
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (key, name, _) in enumerate(test_configs):
    ax = axes[i]
    img = images[key]
    time_slice = img[:, :, diff_it].T  # (ny, nx)

    vmax = np.percentile(np.abs(time_slice), 99) if np.any(time_slice != 0) else 1

    im = ax.imshow(time_slice, aspect='equal', cmap='seismic',
                   extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]],
                   vmin=-vmax, vmax=vmax)
    ax.plot(diffractor_x, diffractor_y, 'g+', markersize=15, markeredgewidth=3)
    ax.set_title(f"{name}\nRMS: {stats[key]['rms']:.2e}", fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[5]
ax.axis('off')

plt.suptitle(f"Time Slice at Diffractor Time (t = {t0_ms:.0f} ms)", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "timeslice_diffractor.png", dpi=150)
plt.close()
print(f"  Saved: {output_dir / 'timeslice_diffractor.png'}")

# 4. Difference images
print("\nGenerating difference images...")

baseline = images["no_corrections"]
baseline_rms = stats["no_corrections"]["rms"]

diff_stats = {}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

compare_keys = ["aa_only", "spreading_only", "obliquity_only", "all_corrections"]

for i, key in enumerate(compare_keys):
    ax = axes[i]

    diff = images[key] - baseline
    diff_inline = diff[diff_ix, :, :].T

    diff_rms = np.sqrt(np.mean(diff**2))
    rel_diff = 100 * diff_rms / baseline_rms if baseline_rms > 0 else 0

    # Correlation
    corr = np.corrcoef(images[key].flatten(), baseline.flatten())[0, 1]

    diff_stats[key] = {
        "name": stats[key]["name"],
        "diff_rms": diff_rms,
        "rel_diff_pct": rel_diff,
        "correlation": corr,
    }

    vmax = np.percentile(np.abs(diff_inline), 99) if np.any(diff_inline != 0) else 1

    im = ax.imshow(diff_inline, aspect='auto', cmap='seismic',
                   extent=[y_axis[0], y_axis[-1], t_axis[-1], t_axis[0]],
                   vmin=-vmax, vmax=vmax)
    ax.axhline(y=t0_ms, color='green', linestyle='--', alpha=0.5)
    ax.set_title(f"{stats[key]['name']} - No Corrections\nDiff: {rel_diff:.1f}%, Corr: {corr:.3f}", fontsize=10)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Time (ms)")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Inline Differences from No Corrections", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "inline_differences.png", dpi=150)
plt.close()
print(f"  Saved: {output_dir / 'inline_differences.png'}")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Configuration':<20} {'Time (s)':<10} {'RMS':<12} {'Max Abs':<12}")
print("-" * 70)
for key, name, _ in test_configs:
    s = stats[key]
    print(f"{name:<20} {s['time']:<10.3f} {s['rms']:<12.2e} {s['absmax']:<12.2e}")

print("\n" + "=" * 70)
print("DIFFERENCE ANALYSIS")
print("=" * 70)
print(f"{'Configuration':<20} {'Diff RMS':<12} {'Rel Diff %':<12} {'Correlation':<12}")
print("-" * 70)
for key in compare_keys:
    ds = diff_stats[key]
    print(f"{ds['name']:<20} {ds['diff_rms']:<12.2e} {ds['rel_diff_pct']:<12.1f} {ds['correlation']:<12.4f}")

print("\n" + "=" * 70)
print(f"All images saved to: {output_dir.absolute()}")
print("=" * 70)
