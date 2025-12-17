#!/usr/bin/env python3
"""Test compiled Metal kernel."""

import numpy as np
import time
from pathlib import Path

# First, make sure the metal library exists
from pstm.kernels.metal_compiled import CompiledMetalKernel, check_metal_available

print("=" * 60)
print("Testing Compiled Metal Kernel")
print("=" * 60)

if not check_metal_available():
    print("ERROR: Metal not available")
    exit(1)

print("Metal is available!")

# Check metallib exists
metallib_path = Path("pstm/metal/pstm_kernels.metallib")
print(f"Metallib path: {metallib_path}")
print(f"Metallib exists: {metallib_path.exists()}")

# Create synthetic test data
print("\nCreating synthetic test data...")
n_traces = 5000
n_samples = 500
nx, ny, nt = 32, 32, 500

# Random trace amplitudes
np.random.seed(42)
amplitudes = np.random.randn(n_traces, n_samples).astype(np.float32)

# Random geometry within a 2km x 2km survey
survey_size = 2000.0
source_x = np.random.uniform(0, survey_size, n_traces).astype(np.float32)
source_y = np.random.uniform(0, survey_size, n_traces).astype(np.float32)
receiver_x = np.random.uniform(0, survey_size, n_traces).astype(np.float32)
receiver_y = np.random.uniform(0, survey_size, n_traces).astype(np.float32)
midpoint_x = (source_x + receiver_x) / 2
midpoint_y = (source_y + receiver_y) / 2

print(f"  Traces: {n_traces}")
print(f"  Samples per trace: {n_samples}")
print(f"  Output grid: {nx} x {ny} x {nt}")
print(f"  Total output samples: {nx * ny * nt:,}")

# Create mock trace block and output tile
class MockTraceBlock:
    def __init__(self):
        self.n_traces = n_traces
        self.n_samples = n_samples
        self.amplitudes = amplitudes
        self.source_x = source_x
        self.source_y = source_y
        self.receiver_x = receiver_x
        self.receiver_y = receiver_y
        self.midpoint_x = midpoint_x
        self.midpoint_y = midpoint_y
        self.sample_rate_ms = 2.0
        self.start_time_ms = 0.0

class MockOutputTile:
    def __init__(self):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.x_axis = np.linspace(0, survey_size, nx).astype(np.float32)
        self.y_axis = np.linspace(0, survey_size, ny).astype(np.float32)
        self.t_axis_ms = np.linspace(0, 1000, nt).astype(np.float32)
        self.image = np.zeros((nx, ny, nt), dtype=np.float64)
        self.fold = np.zeros((nx, ny), dtype=np.int32)

class MockVelocity:
    def __init__(self):
        self.is_1d = True
        self.vrms = np.full(nt, 2500.0, dtype=np.float32)

class MockConfig:
    def __init__(self):
        self.max_aperture_m = 1500.0
        self.min_aperture_m = 100.0
        self.taper_fraction = 0.1
        self.max_dip_degrees = 60.0
        self.apply_spreading = True
        self.apply_obliquity = True
        self.apply_aa = True
        self.aa_dominant_freq = 30.0

traces = MockTraceBlock()
output = MockOutputTile()
velocity = MockVelocity()
config = MockConfig()

print("\nInitializing compiled Metal kernel...")
try:
    kernel = CompiledMetalKernel(use_simd=True)
    kernel.initialize(config)
    print("Kernel initialized successfully!")
except Exception as e:
    print(f"ERROR initializing kernel: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nRunning migration...")
try:
    start = time.perf_counter()
    metrics = kernel.migrate_tile(traces, output, velocity, config)
    elapsed = time.perf_counter() - start
    
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Compute time: {metrics.compute_time_s:.3f} s")
    print(f"Total time: {elapsed:.3f} s")
    print(f"Traces processed: {metrics.n_traces_processed:,}")
    print(f"Output samples: {metrics.n_samples_output:,}")
    print(f"Throughput: {metrics.n_traces_processed / metrics.compute_time_s:,.0f} traces/s")
    print(f"Output rate: {metrics.n_samples_output / metrics.compute_time_s / 1e6:.2f} M samples/s")
    
    print(f"\nOutput image stats:")
    print(f"  Shape: {output.image.shape}")
    print(f"  Min: {output.image.min():.6f}")
    print(f"  Max: {output.image.max():.6f}")
    print(f"  Mean: {output.image.mean():.6f}")
    print(f"  Non-zero: {np.count_nonzero(output.image):,}")
    
    print(f"\nFold stats:")
    print(f"  Min: {output.fold.min()}")
    print(f"  Max: {output.fold.max()}")
    print(f"  Mean: {output.fold.mean():.1f}")
    
except Exception as e:
    print(f"ERROR during migration: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n{'=' * 60}")
print("TEST PASSED!")
print(f"{'=' * 60}")
