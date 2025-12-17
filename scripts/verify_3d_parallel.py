#!/usr/bin/env python3
"""Verify 3D parallel kernel numerical accuracy."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Small test case for easier debugging
n_traces = 5000
n_samples = 200
tile_size = 16

np.random.seed(42)
tile_extent = tile_size * 50  # Larger tile

# Traces clustered around tile center with large spread to ensure contributions
midpoint_x = np.random.randn(n_traces).astype(np.float64) * 500 + tile_extent / 2
midpoint_y = np.random.randn(n_traces).astype(np.float64) * 500 + tile_extent / 2

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
    "max_aperture": 2500.0,  # Larger aperture
    "taper_fraction": 0.1,
    "dt_ms": 4.0,
    "t_start_ms": 0.0,
    "apply_spreading": True,
    "apply_obliquity": True,
}

print("=" * 60)
print("3D PARALLEL KERNEL NUMERICAL VERIFICATION")
print("=" * 60)
print(f"Test: {n_traces} traces, {tile_size}x{tile_size} tile, {n_samples} samples")
print()

# Numba baseline
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
numba_output = OutputTile(
    image=np.zeros((tile_size, tile_size, n_samples), dtype=np.float64),
    fold=np.zeros((tile_size, tile_size), dtype=np.int32),
    x_axis=x_coords.copy(), y_axis=y_coords.copy(), t_axis_ms=t_coords_ms.copy(),
)
velocity = VelocitySlice(vrms=vrms.copy(), is_1d=True, t_axis_ms=t_coords_ms.copy())
kernel_config = KernelConfig(
    max_aperture_m=2500.0, min_aperture_m=100.0,
    max_dip_degrees=45.0, taper_fraction=0.1,
    apply_spreading=True, apply_obliquity=True,
)

numba_kernel = OptimizedNumbaKernel()
numba_kernel.initialize(kernel_config)
numba_kernel.migrate_tile(traces, numba_output, velocity, kernel_config)
numba_kernel.cleanup()

print("Numba results:")
print(f"  Image sum: {np.sum(numba_output.image):.6f}")
print(f"  Image abs sum: {np.sum(np.abs(numba_output.image)):.6f}")
print(f"  Image max: {np.max(numba_output.image):.6f}")
print(f"  Image min: {np.min(numba_output.image):.6f}")
print(f"  Non-zero count: {np.count_nonzero(numba_output.image)}")

# Metal 3D parallel
from pstm.metal.python import _find_and_import_module
metal = _find_and_import_module()

result = metal.benchmark_kernel_variant(
    "3d_parallel",
    amplitudes,
    source_x, source_y,
    receiver_x, receiver_y,
    midpoint_x, midpoint_y,
    x_coords, y_coords,
    t_coords_ms, vrms,
    config,
    1
)

print()
print("Metal 3D parallel results:")
print(f"  Image abs sum: {result['image_sum']:.6f}")

# Calculate difference
numba_abs_sum = np.sum(np.abs(numba_output.image))
metal_abs_sum = result['image_sum']
rel_diff = abs(metal_abs_sum - numba_abs_sum) / numba_abs_sum * 100

print()
print("Comparison:")
print(f"  Numba abs sum: {numba_abs_sum:.6f}")
print(f"  Metal abs sum: {metal_abs_sum:.6f}")
print(f"  Relative difference: {rel_diff:.4f}%")

if rel_diff < 1.0:
    print()
    print(">>> PASSED: Results match within 1% tolerance")
else:
    print()
    print(">>> FAILED: Results differ by more than 1%")
