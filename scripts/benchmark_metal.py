#!/usr/bin/env python3
"""
Benchmark: Metal C++ vs Numba CPU Kernel

Compares performance of the new Metal GPU kernel against optimized Numba CPU.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pstm.kernels.base import (
    KernelConfig,
    OutputTile,
    TraceBlock,
    VelocitySlice,
)


def generate_test_data(n_traces: int, tile_size: int, n_samples: int, aperture_m: float):
    """Generate test data for benchmarking."""
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

    # Calculate offset for TraceBlock
    offset_arr = np.sqrt(
        (receiver_x - source_x) ** 2 + (receiver_y - source_y) ** 2
    )

    traces = TraceBlock(
        amplitudes=amplitudes,
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        offset=offset_arr,
        midpoint_x=midpoint_x,
        midpoint_y=midpoint_y,
        sample_rate_ms=4.0,
        start_time_ms=0.0,
    )

    output = OutputTile(
        image=np.zeros((tile_size, tile_size, n_samples), dtype=np.float64),
        fold=np.zeros((tile_size, tile_size), dtype=np.int32),
        x_axis=x_coords,
        y_axis=y_coords,
        t_axis_ms=t_coords_ms,
    )

    velocity = VelocitySlice(
        vrms=vrms,
        is_1d=True,
        t_axis_ms=t_coords_ms,
    )

    config = KernelConfig(
        max_aperture_m=aperture_m,
        min_aperture_m=100.0,
        max_dip_degrees=45.0,
        taper_fraction=0.1,
        apply_spreading=True,
        apply_obliquity=True,
    )

    return traces, output, velocity, config


def benchmark_kernel(kernel, traces, output, velocity, config, n_warmup=1, n_runs=3):
    """Benchmark a kernel."""
    kernel.initialize(config)

    # Warmup
    for _ in range(n_warmup):
        output.reset()
        kernel.migrate_tile(traces, output, velocity, config)
        kernel.synchronize()

    # Benchmark runs
    times = []
    for _ in range(n_runs):
        output.reset()
        start = time.perf_counter()
        kernel.migrate_tile(traces, output, velocity, config)
        kernel.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    kernel.cleanup()

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "traces_per_s": traces.n_traces / np.mean(times),
        "image_sum": float(np.sum(np.abs(output.image))),
    }


def main():
    print("=" * 70)
    print("METAL C++ vs NUMBA CPU BENCHMARK")
    print("=" * 70)
    print()

    # Test configurations
    configs = [
        {"n_traces": 10_000, "tile_size": 32, "n_samples": 500, "aperture_m": 2500.0},
        {"n_traces": 50_000, "tile_size": 32, "n_samples": 500, "aperture_m": 2500.0},
        {"n_traces": 100_000, "tile_size": 32, "n_samples": 500, "aperture_m": 2500.0},
    ]

    # Check available kernels
    from pstm.kernels.metal_cpp import MetalCppKernel, is_metal_cpp_available
    from pstm.kernels.numba_cpu_optimized import OptimizedNumbaKernel

    metal_available = is_metal_cpp_available()
    print(f"Metal C++ available: {metal_available}")
    if metal_available:
        from pstm.metal.python import get_device_info
        info = get_device_info()
        print(f"  Device: {info['device_name']}")
        print(f"  Memory: {info['device_memory_gb']:.1f} GB")
    print()

    for cfg in configs:
        print("-" * 70)
        print(f"Configuration: {cfg['n_traces']:,} traces, "
              f"{cfg['tile_size']}x{cfg['tile_size']} tile, "
              f"{cfg['n_samples']} samples")
        print("-" * 70)

        traces, output, velocity, kernel_config = generate_test_data(**cfg)

        results = {}

        # Benchmark Numba CPU
        print("Benchmarking Numba CPU (optimized)...")
        try:
            numba_kernel = OptimizedNumbaKernel()
            results["Numba CPU"] = benchmark_kernel(
                numba_kernel, traces, output, velocity, kernel_config
            )
            print(f"  Time: {results['Numba CPU']['mean_time']:.3f}s "
                  f"(±{results['Numba CPU']['std_time']:.3f}s)")
            print(f"  Traces/s: {results['Numba CPU']['traces_per_s']:,.0f}")
        except Exception as e:
            print(f"  Error: {e}")

        # Benchmark Metal C++
        if metal_available:
            print("Benchmarking Metal C++...")
            try:
                metal_kernel = MetalCppKernel()
                results["Metal C++"] = benchmark_kernel(
                    metal_kernel, traces, output, velocity, kernel_config
                )
                print(f"  Time: {results['Metal C++']['mean_time']:.3f}s "
                      f"(±{results['Metal C++']['std_time']:.3f}s)")
                print(f"  Traces/s: {results['Metal C++']['traces_per_s']:,.0f}")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

        # Compare
        if "Numba CPU" in results and "Metal C++" in results:
            speedup = results["Numba CPU"]["mean_time"] / results["Metal C++"]["mean_time"]
            print()
            print(f"  >>> Metal C++ speedup: {speedup:.2f}x vs Numba CPU")

            # Check numerical consistency
            numba_sum = results["Numba CPU"]["image_sum"]
            metal_sum = results["Metal C++"]["image_sum"]
            if numba_sum > 0:
                rel_diff = abs(metal_sum - numba_sum) / numba_sum
                print(f"  >>> Image sum difference: {rel_diff:.2%}")

        print()

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
