#!/usr/bin/env python3
"""Benchmark compiled Metal kernel with different correction configurations."""

import numpy as np
import time
from pathlib import Path

print("=" * 70)
print("Compiled Metal - Amplitude Corrections Benchmark")
print("=" * 70)

# Test configuration
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
print(f"  Samples/trace: {n_samples:,}")
print(f"  Output grid: {nx}x{ny}x{nt} = {nx*ny*nt:,} samples")
print(f"  Total operations: {n_traces * nx * ny * nt:,} trace-sample pairs")

# Check Metal availability
from pstm.kernels.metal_compiled import CompiledMetalKernel, check_metal_available

if not check_metal_available():
    print("\nERROR: Metal not available on this system")
    exit(1)

# Shared data
traces = MockTraceBlock()
velocity = MockVelocity()

# Test configurations
test_configs = [
    ("No corrections", {"apply_spreading": False, "apply_obliquity": False, "apply_aa": False}),
    ("AA only", {"apply_spreading": False, "apply_obliquity": False, "apply_aa": True}),
    ("Spreading only", {"apply_spreading": True, "apply_obliquity": False, "apply_aa": False}),
    ("Obliquity only", {"apply_spreading": False, "apply_obliquity": True, "apply_aa": False}),
    ("Spreading + Obliquity", {"apply_spreading": True, "apply_obliquity": True, "apply_aa": False}),
    ("AA + Spreading", {"apply_spreading": True, "apply_obliquity": False, "apply_aa": True}),
    ("AA + Obliquity", {"apply_spreading": False, "apply_obliquity": True, "apply_aa": True}),
    ("All corrections", {"apply_spreading": True, "apply_obliquity": True, "apply_aa": True}),
]

results = {}
image_stats = {}

for name, cfg_params in test_configs:
    print("\n" + "-" * 70)
    print(f"Test: {name}")
    print(f"  Spreading: {cfg_params['apply_spreading']}, Obliquity: {cfg_params['apply_obliquity']}, AA: {cfg_params['apply_aa']}")
    print("-" * 70)

    config = MockConfig(**cfg_params)
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
    results[name] = {
        "time": avg_time,
        "traces_per_s": n_traces / avg_time,
        "config": cfg_params,
    }

    # Capture image statistics
    img = output.image
    image_stats[name] = {
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "nonzero": int(np.count_nonzero(img)),
        "rms": float(np.sqrt(np.mean(img**2))),
    }

    print(f"  Average: {avg_time:.3f}s ({n_traces/avg_time:,.0f} traces/s)")
    print(f"  Image stats: min={image_stats[name]['min']:.2e}, max={image_stats[name]['max']:.2e}, rms={image_stats[name]['rms']:.2e}")

    kernel.cleanup()

# Summary
print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)
print(f"{'Configuration':<25} {'Time (s)':<12} {'Traces/s':<15} {'Rel Speed':<10}")
print("-" * 70)

baseline_time = results["No corrections"]["time"]
for name in [t[0] for t in test_configs]:
    r = results[name]
    rel_speed = baseline_time / r["time"]
    print(f"{name:<25} {r['time']:<12.3f} {r['traces_per_s']:<15,.0f} {rel_speed:<10.2f}x")

# Image statistics comparison
print("\n" + "=" * 70)
print("IMAGE OUTPUT STATISTICS")
print("=" * 70)
print(f"{'Configuration':<25} {'RMS':<15} {'Max':<15} {'Non-zero':<12}")
print("-" * 70)

for name in [t[0] for t in test_configs]:
    s = image_stats[name]
    print(f"{name:<25} {s['rms']:<15.2e} {s['max']:<15.2e} {s['nonzero']:<12,}")

# Check if corrections actually affect output
print("\n" + "=" * 70)
print("CORRECTION EFFECT ANALYSIS")
print("=" * 70)

baseline_rms = image_stats["No corrections"]["rms"]
for name in [t[0] for t in test_configs]:
    if name == "No corrections":
        continue
    s = image_stats[name]
    rms_change = ((s["rms"] / baseline_rms) - 1) * 100 if baseline_rms > 0 else 0
    print(f"{name:<25} RMS change: {rms_change:+.1f}%")

print("=" * 70)
