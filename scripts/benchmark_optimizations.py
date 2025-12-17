#!/usr/bin/env python3
"""
Benchmark Metal Kernel Optimizations

Tests each optimization variant and measures performance impact.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time

# Test parameters
n_traces = 50_000
n_samples = 500
tile_size = 32
aperture_m = 2500.0

print("=" * 70)
print("METAL KERNEL OPTIMIZATION BENCHMARK")
print("=" * 70)
print(f"Configuration: {n_traces:,} traces, {tile_size}x{tile_size} tile, {n_samples} samples")
print()

# Generate test data
np.random.seed(42)
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

# Get Numba baseline first
print("Getting Numba CPU baseline...")
from pstm.kernels.base import KernelConfig, OutputTile, TraceBlock, VelocitySlice
from pstm.kernels.numba_cpu_optimized import OptimizedNumbaKernel

offset_arr = np.sqrt((receiver_x - source_x)**2 + (receiver_y - source_y)**2)
traces = TraceBlock(
    amplitudes=amplitudes,
    source_x=source_x, source_y=source_y,
    receiver_x=receiver_x, receiver_y=receiver_y,
    offset=offset_arr,
    midpoint_x=midpoint_x, midpoint_y=midpoint_y,
    sample_rate_ms=4.0, start_time_ms=0.0,
)
output = OutputTile(
    image=np.zeros((tile_size, tile_size, n_samples), dtype=np.float64),
    fold=np.zeros((tile_size, tile_size), dtype=np.int32),
    x_axis=x_coords.copy(), y_axis=y_coords.copy(), t_axis_ms=t_coords_ms.copy(),
)
velocity = VelocitySlice(vrms=vrms.copy(), is_1d=True, t_axis_ms=t_coords_ms.copy())
kernel_config = KernelConfig(
    max_aperture_m=aperture_m, min_aperture_m=100.0,
    max_dip_degrees=45.0, taper_fraction=0.1,
    apply_spreading=True, apply_obliquity=True,
)

numba_kernel = OptimizedNumbaKernel()
numba_kernel.initialize(kernel_config)

# Warmup
output.reset()
numba_kernel.migrate_tile(traces, output, velocity, kernel_config)

# Benchmark
numba_times = []
for _ in range(3):
    output.reset()
    start = time.perf_counter()
    numba_kernel.migrate_tile(traces, output, velocity, kernel_config)
    numba_times.append(time.perf_counter() - start)

numba_mean = np.mean(numba_times) * 1000  # ms
numba_image_sum = np.sum(np.abs(output.image))
numba_kernel.cleanup()

print(f"Numba CPU: {numba_mean:.1f} ms ({n_traces/numba_mean*1000:,.0f} traces/s)")
print(f"Numba image sum: {numba_image_sum:.2f}")
print()

# Import Metal benchmark
from pstm.metal.python import _find_and_import_module
metal = _find_and_import_module()

if metal is None or not hasattr(metal, 'benchmark_kernel_variant'):
    print("Metal benchmark module not available")
    sys.exit(1)

# Test variants
variants = [
    ("baseline", "Baseline (2D grid, nx*ny threads)"),
    ("3d_parallel", "3D Parallel (nx*ny*nt threads)"),
    ("shared_memory", "Shared Memory (cached velocity)"),
]

print("-" * 70)
print(f"{'Variant':<40} {'Time (ms)':<12} {'Speedup':<10} {'vs Numba':<10}")
print("-" * 70)

results = {}
baseline_time = None

for variant_name, description in variants:
    try:
        result = metal.benchmark_kernel_variant(
            variant_name,
            amplitudes,
            source_x, source_y,
            receiver_x, receiver_y,
            midpoint_x, midpoint_y,
            x_coords, y_coords,
            t_coords_ms, vrms,
            config,
            3  # n_runs
        )

        time_ms = result["mean_time_ms"]
        results[variant_name] = result

        if baseline_time is None:
            baseline_time = time_ms
            speedup = "1.00x"
        else:
            speedup = f"{baseline_time/time_ms:.2f}x"

        vs_numba = f"{numba_mean/time_ms:.2f}x"

        print(f"{description:<40} {time_ms:<12.1f} {speedup:<10} {vs_numba:<10}")

        # Check numerical accuracy
        rel_diff = abs(result["image_sum"] - numba_image_sum) / numba_image_sum * 100
        if rel_diff > 1.0:
            print(f"  WARNING: Image sum differs by {rel_diff:.1f}%")

    except Exception as e:
        print(f"{description:<40} ERROR: {e}")

print("-" * 70)
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
if results:
    best_variant = min(results.keys(), key=lambda k: results[k]["mean_time_ms"])
    best_time = results[best_variant]["mean_time_ms"]
    print(f"Best Metal variant: {best_variant}")
    print(f"Best Metal time: {best_time:.1f} ms")
    print(f"Best Metal vs Numba: {numba_mean/best_time:.2f}x")

    if best_time < numba_mean:
        print(f"\n>>> Metal GPU is {numba_mean/best_time:.1f}x FASTER than Numba CPU")
    else:
        print(f"\n>>> Numba CPU is {best_time/numba_mean:.1f}x faster than best Metal GPU")
