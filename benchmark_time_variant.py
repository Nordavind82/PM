#!/usr/bin/env python3
"""Benchmark time-variant sampling vs uniform sampling."""

import numpy as np
import time
from pathlib import Path

print("=" * 70)
print("Time-Variant Sampling Benchmark")
print("=" * 70)

# Test configuration
n_traces = 10000
n_samples = 1500  # 3 seconds at 2ms
nx, ny = 41, 41
base_dt_ms = 2.0
t_max_ms = 3000.0
nt_uniform = int(t_max_ms / base_dt_ms) + 1

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
        self.sample_rate_ms = base_dt_ms
        self.start_time_ms = 0.0

class MockOutputTile:
    def __init__(self, nt):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.x_axis = np.linspace(0, survey_size, nx).astype(np.float64)
        self.y_axis = np.linspace(0, survey_size, ny).astype(np.float64)
        self.t_axis_ms = np.linspace(0, t_max_ms, nt).astype(np.float64)
        self.image = np.zeros((nx, ny, nt), dtype=np.float64)
        self.fold = np.zeros((nx, ny), dtype=np.int32)

class MockVelocity:
    def __init__(self, nt):
        self.is_1d = True
        self.vrms = np.full(nt, 2500.0, dtype=np.float32)

class MockConfig:
    def __init__(self):
        self.max_aperture_m = 1500.0
        self.min_aperture_m = 100.0
        self.taper_fraction = 0.1
        self.max_dip_degrees = 60.0
        self.apply_spreading = False
        self.apply_obliquity = False
        self.apply_aa = False
        self.aa_dominant_freq = 30.0

print(f"\nTest Configuration:")
print(f"  Traces: {n_traces:,}")
print(f"  Samples/trace: {n_samples:,}")
print(f"  Output grid: {nx}x{ny}")
print(f"  Time range: 0 - {t_max_ms:.0f} ms")
print(f"  Base dt: {base_dt_ms} ms")
print(f"  Uniform samples: {nt_uniform}")

# Import modules
from pstm.kernels.metal_compiled import CompiledMetalKernel, check_metal_available
from pstm.algorithm.time_variant import (
    FrequencyTimeTable,
    compute_time_windows,
    estimate_speedup,
    get_window_info_string,
    resample_to_uniform,
)

if not check_metal_available():
    print("\nERROR: Metal not available")
    exit(1)

# Create frequency table
freq_table = FrequencyTimeTable(
    times_ms=[0, 500, 1500, 3000],
    frequencies_hz=[80, 50, 30, 20],
)

# Compute time windows
windows = compute_time_windows(0, t_max_ms, base_dt_ms, freq_table)
total_tv_samples = sum(w.n_samples for w in windows)

print(f"\n{get_window_info_string(windows)}")
print(f"\nEstimated speedup: {estimate_speedup(0, t_max_ms, base_dt_ms, freq_table):.2f}x")

# Shared data
traces = MockTraceBlock()
config = MockConfig()

results = {}

# Test 1: Uniform sampling
print("\n" + "-" * 70)
print("Test 1: Uniform Sampling")
print("-" * 70)

kernel = CompiledMetalKernel(use_simd=True)
kernel.initialize(config)

# Warmup
output = MockOutputTile(nt_uniform)
velocity = MockVelocity(nt_uniform)
kernel.migrate_tile(traces, output, velocity, config)

# Benchmark
times = []
for i in range(3):
    output = MockOutputTile(nt_uniform)
    start = time.perf_counter()
    kernel.migrate_tile(traces, output, velocity, config)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.3f}s")

uniform_time = np.mean(times)
uniform_output = output.image.copy()
results["Uniform"] = {
    "time": uniform_time,
    "samples": nt_uniform,
    "traces_per_s": n_traces / uniform_time,
}
print(f"  Average: {uniform_time:.3f}s ({n_traces/uniform_time:,.0f} traces/s)")
print(f"  Output samples: {nt_uniform}")

kernel.cleanup()

# Test 2: Time-variant sampling
print("\n" + "-" * 70)
print("Test 2: Time-Variant Sampling")
print("-" * 70)

kernel = CompiledMetalKernel(use_simd=True)
kernel.initialize(config)

# Create output tile for TV (will be resized by kernel)
output_tv = MockOutputTile(total_tv_samples)
velocity_tv = MockVelocity(n_samples)  # Velocity at input sampling

# Warmup
kernel.migrate_tile_time_variant(traces, output_tv, velocity_tv, config, windows)

# Benchmark
times = []
for i in range(3):
    output_tv = MockOutputTile(total_tv_samples)
    start = time.perf_counter()
    kernel.migrate_tile_time_variant(traces, output_tv, velocity_tv, config, windows)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.3f}s")

tv_time = np.mean(times)
tv_output_raw = output_tv.image.copy()
results["Time-Variant"] = {
    "time": tv_time,
    "samples": total_tv_samples,
    "traces_per_s": n_traces / tv_time,
}
print(f"  Average: {tv_time:.3f}s ({n_traces/tv_time:,.0f} traces/s)")
print(f"  Output samples: {total_tv_samples}")

kernel.cleanup()

# Resample TV output to uniform for comparison
print("\n" + "-" * 70)
print("Resampling time-variant output to uniform...")
print("-" * 70)

start = time.perf_counter()
tv_output_resampled = resample_to_uniform(tv_output_raw, windows, base_dt_ms, nt_uniform)
resample_time = time.perf_counter() - start
print(f"  Resample time: {resample_time:.3f}s")

# Total TV time including resample
tv_total_time = tv_time + resample_time
results["TV + Resample"] = {
    "time": tv_total_time,
    "samples": nt_uniform,
    "traces_per_s": n_traces / tv_total_time,
}

# Summary
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)
print(f"{'Method':<20} {'Time (s)':<12} {'Samples':<12} {'Traces/s':<15} {'Speedup':<10}")
print("-" * 70)

baseline = results["Uniform"]["time"]
for name, r in results.items():
    speedup = baseline / r["time"]
    print(f"{name:<20} {r['time']:<12.3f} {r['samples']:<12,} {r['traces_per_s']:<15,.0f} {speedup:<10.2f}x")

# Quality comparison
print("\n" + "=" * 70)
print("QUALITY COMPARISON (Uniform vs Time-Variant Resampled)")
print("=" * 70)

# Compare at matching time samples
uniform_rms = np.sqrt(np.mean(uniform_output**2))
tv_rms = np.sqrt(np.mean(tv_output_resampled**2))
diff = uniform_output - tv_output_resampled
diff_rms = np.sqrt(np.mean(diff**2))
rel_diff = 100 * diff_rms / uniform_rms if uniform_rms > 0 else 0

# Correlation
corr = np.corrcoef(uniform_output.flatten(), tv_output_resampled.flatten())[0, 1]

print(f"  Uniform RMS:     {uniform_rms:.4e}")
print(f"  TV Resampled RMS: {tv_rms:.4e}")
print(f"  Difference RMS:  {diff_rms:.4e} ({rel_diff:.2f}%)")
print(f"  Correlation:     {corr:.6f}")

print("\n" + "=" * 70)
actual_speedup = uniform_time / tv_total_time
print(f"ACTUAL SPEEDUP (including resample): {actual_speedup:.2f}x")
print("=" * 70)
