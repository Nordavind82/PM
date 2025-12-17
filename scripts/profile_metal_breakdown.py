#!/usr/bin/env python3
"""Profile Metal kernel time breakdown."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np

# Test parameters
n_traces = 50_000
n_samples = 500
tile_size = 32
aperture_m = 2500.0

np.random.seed(42)

# Generate test data
tile_extent = tile_size * 25
spread = aperture_m * 0.8

midpoint_x = np.random.randn(n_traces).astype(np.float64) * spread + tile_extent / 2
midpoint_y = np.random.randn(n_traces).astype(np.float64) * spread + tile_extent / 2

offset = 500.0
angles = np.random.rand(n_traces) * 2 * np.pi
source_x = midpoint_x - offset * np.cos(angles)
source_y = midpoint_y - offset * np.sin(angles)
receiver_x = midpoint_x + offset * np.cos(angles)
receiver_y = midpoint_y + offset * np.sin(angles)

x_coords = np.linspace(0, tile_extent, tile_size).astype(np.float64)
y_coords = np.linspace(0, tile_extent, tile_size).astype(np.float64)
t_coords_ms = np.linspace(0, n_samples * 4.0, n_samples).astype(np.float64)
vrms = np.linspace(1800, 3500, n_samples).astype(np.float64)
amplitudes = np.random.randn(n_traces, n_samples).astype(np.float32)

image = np.zeros((tile_size, tile_size, n_samples), dtype=np.float64)
fold = np.zeros((tile_size, tile_size), dtype=np.int32)

config = {
    "max_dip_deg": 45.0,
    "min_aperture": 100.0,
    "max_aperture": aperture_m,
    "taper_fraction": 0.1,
    "dt_ms": 4.0,
    "t_start_ms": 0.0,
    "apply_spreading": True,
    "apply_obliquity": True,
}

print("=" * 60)
print("METAL KERNEL TIME BREAKDOWN")
print("=" * 60)
print(f"Traces: {n_traces:,}, Samples: {n_samples}, Tile: {tile_size}x{tile_size}")
print()

# Import and get the raw module
from pstm.metal.python import _find_and_import_module
metal = _find_and_import_module()

if metal is None:
    print("Metal module not available")
    sys.exit(1)

# Find shader path
module_dir = Path(__file__).parent.parent / "pstm" / "metal"
shader_path = module_dir / "build" / "migrate_tile.metallib"
config["shader_path"] = str(shader_path)

# Warmup
print("Warming up...")
metal.migrate_tile(
    amplitudes[:100], source_x[:100], source_y[:100],
    receiver_x[:100], receiver_y[:100], midpoint_x[:100], midpoint_y[:100],
    np.zeros((tile_size, tile_size, n_samples), dtype=np.float64),
    np.zeros((tile_size, tile_size), dtype=np.int32),
    x_coords, y_coords, t_coords_ms, vrms, config
)

print()
print("Profiling components...")
print()

# Profile: Data preparation (Python side)
n_runs = 3
times = {"prep": [], "call": [], "total": []}

for _ in range(n_runs):
    image[:] = 0
    fold[:] = 0

    total_start = time.perf_counter()

    # This includes all the work: data conversion, buffer creation, kernel, copy-back
    result = metal.migrate_tile(
        amplitudes, source_x, source_y,
        receiver_x, receiver_y, midpoint_x, midpoint_y,
        image, fold,
        x_coords, y_coords, t_coords_ms, vrms, config
    )

    total_end = time.perf_counter()

    times["total"].append((total_end - total_start) * 1000)
    times["call"].append(result["kernel_time_ms"])

# Calculate
avg_total = np.mean(times["total"])
avg_kernel = np.mean(times["call"])
avg_overhead = avg_total - avg_kernel

print(f"{'Component':<30} {'Time (ms)':<15} {'% of Total':<15}")
print("-" * 60)
print(f"{'GPU Kernel Execution':<30} {avg_kernel:<15.2f} {avg_kernel/avg_total*100:<15.1f}%")
print(f"{'Python/C++ Overhead':<30} {avg_overhead:<15.2f} {avg_overhead/avg_total*100:<15.1f}%")
print(f"{'  (data conversion, buffers)':<30}")
print("-" * 60)
print(f"{'TOTAL':<30} {avg_total:<15.2f} {'100.0':<15}%")

print()
print("Breakdown from C++ metrics:")
print(f"  kernel_time_ms: {result['kernel_time_ms']:.2f}")
print(f"  total_time_ms: {result['total_time_ms']:.2f}")
print(f"  traces_per_second: {result['traces_per_second']:,.0f}")

# Calculate theoretical limits
print()
print("=" * 60)
print("ANALYSIS")
print("=" * 60)

# Memory bandwidth calculation
trace_data_mb = (n_traces * n_samples * 4) / (1024 * 1024)  # amplitudes
coord_data_mb = (n_traces * 6 * 8) / (1024 * 1024)  # 6 coord arrays, float64
output_data_mb = (tile_size * tile_size * n_samples * 8) / (1024 * 1024)  # image
total_data_mb = trace_data_mb + coord_data_mb + output_data_mb

print(f"Data sizes:")
print(f"  Trace amplitudes: {trace_data_mb:.1f} MB")
print(f"  Coordinates: {coord_data_mb:.1f} MB")
print(f"  Output image: {output_data_mb:.1f} MB")
print(f"  Total: {total_data_mb:.1f} MB")

# M4 Max bandwidth: ~546 GB/s
bandwidth_gbs = 546
theoretical_transfer_ms = (total_data_mb / 1024) / bandwidth_gbs * 1000
print(f"\nTheoretical transfer time @ {bandwidth_gbs} GB/s: {theoretical_transfer_ms:.2f} ms")

# Compute analysis
n_output_samples = tile_size * tile_size * n_samples
n_operations = n_traces * n_output_samples  # Each trace contributes to each output
print(f"\nCompute complexity:")
print(f"  Output samples: {n_output_samples:,}")
print(f"  Trace × output combinations: {n_operations:,}")
print(f"  Current throughput: {n_operations / (avg_kernel/1000):,.0f} ops/s")
