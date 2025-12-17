#!/usr/bin/env python3
"""Benchmark different compute backends for PSTM migration."""

import numpy as np
import time
from pathlib import Path

print("=" * 70)
print("PSTM Backend Benchmark")
print("=" * 70)

# Create synthetic test data
n_traces = 10000
n_samples = 1000
nx, ny, nt = 41, 41, 500

np.random.seed(42)
survey_size = 2000.0

# Mock classes matching kernel interface
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
        self.weights = None
        self.sample_rate_ms = 2.0
        self.start_time_ms = 0.0

    def ensure_contiguous(self):
        """Return self (already contiguous)."""
        return self

class MockOutputTile:
    def __init__(self):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.x_axis = np.linspace(0, survey_size, nx).astype(np.float64)
        self.y_axis = np.linspace(0, survey_size, ny).astype(np.float64)
        self.t_axis_ms = np.linspace(0, 1000, nt).astype(np.float64)
        self.t_coords_ms = self.t_axis_ms  # Alias for Numba kernel
        self.x_coords = self.x_axis  # Alias for Numba kernel
        self.y_coords = self.y_axis  # Alias for Numba kernel
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

print(f"\nTest Configuration:")
print(f"  Traces: {n_traces:,}")
print(f"  Samples/trace: {n_samples:,}")
print(f"  Output grid: {nx}x{ny}x{nt} = {nx*ny*nt:,} samples")
print(f"  Total operations: {n_traces * nx * ny * nt:,} trace-sample pairs")

traces = MockTraceBlock()
velocity = MockVelocity()
config = MockConfig()

results = {}

# Test 1: Compiled Metal
print("\n" + "-" * 70)
print("Backend: Compiled Metal (PyObjC)")
print("-" * 70)
try:
    from pstm.kernels.metal_compiled import CompiledMetalKernel, check_metal_available
    if check_metal_available():
        kernel = CompiledMetalKernel(use_simd=True)
        kernel.initialize(config)
        
        # Warmup
        output = MockOutputTile()
        kernel.migrate_tile(traces, output, velocity, config)
        
        # Benchmark (3 runs)
        times = []
        for i in range(3):
            output = MockOutputTile()
            start = time.perf_counter()
            metrics = kernel.migrate_tile(traces, output, velocity, config)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s")
        
        avg_time = np.mean(times)
        results["Compiled Metal"] = {
            "time": avg_time,
            "traces_per_s": n_traces / avg_time,
            "samples_per_s": (nx * ny * nt) / avg_time,
        }
        print(f"  Average: {avg_time:.3f}s ({n_traces/avg_time:,.0f} traces/s)")
    else:
        print("  Metal not available")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: Numba CPU
print("\n" + "-" * 70)
print("Backend: Numba CPU (Optimized)")
print("-" * 70)
try:
    from pstm.kernels.numba_cpu_optimized import OptimizedNumbaKernel
    from pstm.kernels.base import KernelConfig
    
    kernel_config = KernelConfig(
        max_aperture_m=config.max_aperture_m,
        min_aperture_m=config.min_aperture_m,
        max_dip_degrees=config.max_dip_degrees,
        taper_fraction=config.taper_fraction,
        apply_spreading=config.apply_spreading,
        apply_obliquity=config.apply_obliquity,
        aa_enabled=config.apply_aa,
    )
    
    kernel = OptimizedNumbaKernel()
    kernel.initialize(kernel_config)
    
    # Warmup
    output = MockOutputTile()
    kernel.migrate_tile(traces, output, velocity, kernel_config)
    
    # Benchmark (3 runs)
    times = []
    for i in range(3):
        output = MockOutputTile()
        start = time.perf_counter()
        metrics = kernel.migrate_tile(traces, output, velocity, kernel_config)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
    
    avg_time = np.mean(times)
    results["Numba CPU"] = {
        "time": avg_time,
        "traces_per_s": n_traces / avg_time,
        "samples_per_s": (nx * ny * nt) / avg_time,
    }
    print(f"  Average: {avg_time:.3f}s ({n_traces/avg_time:,.0f} traces/s)")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)
print(f"{'Backend':<25} {'Time (s)':<12} {'Traces/s':<15} {'Speedup':<10}")
print("-" * 70)

if results:
    baseline = min(r["time"] for r in results.values())
    for name, r in sorted(results.items(), key=lambda x: x[1]["time"]):
        speedup = baseline / r["time"] if r["time"] > 0 else 0
        print(f"{name:<25} {r['time']:<12.3f} {r['traces_per_s']:<15,.0f} {speedup:<10.2f}x")

print("=" * 70)
