#!/usr/bin/env python3
"""
Benchmark: Original vs Optimized Numba Kernel

Compares performance of the original and optimized Numba CPU kernels.

Usage:
    python scripts/benchmark_optimized.py [--traces N] [--tile-size N] [--samples N]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    time_s: float
    n_traces: int
    traces_per_s: float


def generate_test_data(n_traces: int, tile_size: int, n_samples: int, aperture_m: float) -> dict:
    """Generate test data."""
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

    return {
        "amplitudes": amplitudes,
        "source_x": source_x,
        "source_y": source_y,
        "receiver_x": receiver_x,
        "receiver_y": receiver_y,
        "midpoint_x": midpoint_x,
        "midpoint_y": midpoint_y,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "t_coords_ms": t_coords_ms,
        "vrms": vrms,
        "dt_ms": 4.0,
        "t_start_ms": 0.0,
        "n_traces": n_traces,
        "tile_size": tile_size,
        "n_samples": n_samples,
    }


def benchmark_original_kernel(data: dict, aperture_m: float) -> BenchmarkResult:
    """Benchmark original Numba kernel."""
    from pstm.kernels.numba_cpu import _migrate_tile_kernel

    tile_size = data["tile_size"]
    n_samples = data["n_samples"]
    n_traces = data["n_traces"]

    # Warmup
    image_w = np.zeros((4, 4, 50), dtype=np.float64)
    fold_w = np.zeros((4, 4), dtype=np.int32)
    _migrate_tile_kernel(
        data["amplitudes"][:10],
        data["source_x"][:10], data["source_y"][:10],
        data["receiver_x"][:10], data["receiver_y"][:10],
        data["midpoint_x"][:10], data["midpoint_y"][:10],
        np.ones(10, dtype=np.float64),
        4.0, 0.0, image_w, fold_w,
        data["x_coords"][:4], data["y_coords"][:4],
        data["t_coords_ms"][:50], data["vrms"][:50],
        45.0, 100.0, aperture_m, 0.1, True, True, 1,
    )

    # Benchmark
    image = np.zeros((tile_size, tile_size, n_samples), dtype=np.float64)
    fold = np.zeros((tile_size, tile_size), dtype=np.int32)
    trace_weights = np.ones(n_traces, dtype=np.float64)

    start = time.perf_counter()
    _migrate_tile_kernel(
        data["amplitudes"],
        data["source_x"], data["source_y"],
        data["receiver_x"], data["receiver_y"],
        data["midpoint_x"], data["midpoint_y"],
        trace_weights,
        data["dt_ms"], data["t_start_ms"],
        image, fold,
        data["x_coords"], data["y_coords"],
        data["t_coords_ms"], data["vrms"],
        45.0, 100.0, aperture_m, 0.1, True, True, 1,
    )
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name="Original Numba",
        time_s=elapsed,
        n_traces=n_traces,
        traces_per_s=n_traces / elapsed,
    )


def benchmark_optimized_kernel(data: dict, aperture_m: float, use_fast_sqrt: bool = False) -> BenchmarkResult:
    """Benchmark optimized Numba kernel."""
    from pstm.kernels.numba_cpu_optimized import (
        _migrate_tile_kernel_optimized,
        _migrate_tile_kernel_fast_sqrt,
        _precompute_time_terms,
    )

    tile_size = data["tile_size"]
    n_samples = data["n_samples"]
    n_traces = data["n_traces"]

    # Pre-compute time terms
    t0_half_sq, inv_v_sq, t0_s, apertures = _precompute_time_terms(
        data["t_coords_ms"], data["vrms"], 45.0, 100.0, aperture_m
    )

    kernel_func = _migrate_tile_kernel_fast_sqrt if use_fast_sqrt else _migrate_tile_kernel_optimized

    # Warmup
    image_w = np.zeros((4, 4, 50), dtype=np.float64)
    fold_w = np.zeros((4, 4), dtype=np.int32)
    t0_w, iv_w, ts_w, ap_w = _precompute_time_terms(
        data["t_coords_ms"][:50], data["vrms"][:50], 45.0, 100.0, aperture_m
    )
    kernel_func(
        data["amplitudes"][:10],
        data["source_x"][:10], data["source_y"][:10],
        data["receiver_x"][:10], data["receiver_y"][:10],
        data["midpoint_x"][:10], data["midpoint_y"][:10],
        np.ones(10, dtype=np.float64),
        4.0, 0.0, image_w, fold_w,
        data["x_coords"][:4], data["y_coords"][:4],
        t0_w, iv_w, ts_w,
        45.0, 100.0, aperture_m, 0.1, True, True, ap_w,
    )

    # Benchmark
    image = np.zeros((tile_size, tile_size, n_samples), dtype=np.float64)
    fold = np.zeros((tile_size, tile_size), dtype=np.int32)
    trace_weights = np.ones(n_traces, dtype=np.float64)

    start = time.perf_counter()
    kernel_func(
        data["amplitudes"],
        data["source_x"], data["source_y"],
        data["receiver_x"], data["receiver_y"],
        data["midpoint_x"], data["midpoint_y"],
        trace_weights,
        data["dt_ms"], data["t_start_ms"],
        image, fold,
        data["x_coords"], data["y_coords"],
        t0_half_sq, inv_v_sq, t0_s,
        45.0, 100.0, aperture_m, 0.1, True, True, apertures,
    )
    elapsed = time.perf_counter() - start

    name = "Optimized + Fast sqrt" if use_fast_sqrt else "Optimized Numba"
    return BenchmarkResult(
        name=name,
        time_s=elapsed,
        n_traces=n_traces,
        traces_per_s=n_traces / elapsed,
    )


def benchmark_with_prefilter(data: dict, aperture_m: float) -> BenchmarkResult:
    """Benchmark with trace pre-filtering."""
    from pstm.kernels.numba_cpu_optimized import (
        _migrate_tile_kernel_optimized,
        _precompute_time_terms,
        prefilter_traces_for_tile,
    )

    tile_size = data["tile_size"]
    n_samples = data["n_samples"]
    tile_extent = tile_size * 25

    # Pre-filter traces
    tile_center_x = tile_extent / 2
    tile_center_y = tile_extent / 2

    start_filter = time.perf_counter()
    mask = prefilter_traces_for_tile(
        data["midpoint_x"], data["midpoint_y"],
        tile_center_x, tile_center_y, aperture_m, margin=1.2
    )
    filter_time = time.perf_counter() - start_filter

    # Get filtered traces
    filtered_amp = data["amplitudes"][mask]
    filtered_sx = data["source_x"][mask]
    filtered_sy = data["source_y"][mask]
    filtered_rx = data["receiver_x"][mask]
    filtered_ry = data["receiver_y"][mask]
    filtered_mx = data["midpoint_x"][mask]
    filtered_my = data["midpoint_y"][mask]
    n_filtered = filtered_amp.shape[0]

    # Pre-compute
    t0_half_sq, inv_v_sq, t0_s, apertures = _precompute_time_terms(
        data["t_coords_ms"], data["vrms"], 45.0, 100.0, aperture_m
    )

    # Warmup
    image_w = np.zeros((4, 4, 50), dtype=np.float64)
    fold_w = np.zeros((4, 4), dtype=np.int32)
    t0_w, iv_w, ts_w, ap_w = _precompute_time_terms(
        data["t_coords_ms"][:50], data["vrms"][:50], 45.0, 100.0, aperture_m
    )
    _migrate_tile_kernel_optimized(
        filtered_amp[:min(10, n_filtered)],
        filtered_sx[:min(10, n_filtered)], filtered_sy[:min(10, n_filtered)],
        filtered_rx[:min(10, n_filtered)], filtered_ry[:min(10, n_filtered)],
        filtered_mx[:min(10, n_filtered)], filtered_my[:min(10, n_filtered)],
        np.ones(min(10, n_filtered), dtype=np.float64),
        4.0, 0.0, image_w, fold_w,
        data["x_coords"][:4], data["y_coords"][:4],
        t0_w, iv_w, ts_w,
        45.0, 100.0, aperture_m, 0.1, True, True, ap_w,
    )

    # Benchmark
    image = np.zeros((tile_size, tile_size, n_samples), dtype=np.float64)
    fold = np.zeros((tile_size, tile_size), dtype=np.int32)
    trace_weights = np.ones(n_filtered, dtype=np.float64)

    start = time.perf_counter()
    _migrate_tile_kernel_optimized(
        filtered_amp,
        filtered_sx, filtered_sy,
        filtered_rx, filtered_ry,
        filtered_mx, filtered_my,
        trace_weights,
        data["dt_ms"], data["t_start_ms"],
        image, fold,
        data["x_coords"], data["y_coords"],
        t0_half_sq, inv_v_sq, t0_s,
        45.0, 100.0, aperture_m, 0.1, True, True, apertures,
    )
    kernel_time = time.perf_counter() - start

    total_time = filter_time + kernel_time

    print(f"  Pre-filter: {data['n_traces']:,} → {n_filtered:,} traces ({n_filtered/data['n_traces']*100:.1f}%)")
    print(f"  Filter time: {filter_time*1000:.2f} ms, Kernel time: {kernel_time*1000:.2f} ms")

    return BenchmarkResult(
        name="Optimized + Pre-filter",
        time_s=total_time,
        n_traces=data["n_traces"],
        traces_per_s=data["n_traces"] / total_time,
    )


def run_benchmarks(n_traces: int, tile_size: int, n_samples: int, aperture_m: float) -> list[BenchmarkResult]:
    """Run all benchmarks."""
    print(f"Configuration:")
    print(f"  Traces:      {n_traces:,}")
    print(f"  Tile size:   {tile_size}×{tile_size}")
    print(f"  Samples:     {n_samples}")
    print(f"  Aperture:    {aperture_m:.0f} m")
    print()

    data = generate_test_data(n_traces, tile_size, n_samples, aperture_m)
    results = []

    print("1/4: Benchmarking Original Numba kernel...")
    results.append(benchmark_original_kernel(data, aperture_m))

    print("2/4: Benchmarking Optimized Numba kernel...")
    results.append(benchmark_optimized_kernel(data, aperture_m, use_fast_sqrt=False))

    print("3/4: Benchmarking Optimized + Fast sqrt...")
    results.append(benchmark_optimized_kernel(data, aperture_m, use_fast_sqrt=True))

    print("4/4: Benchmarking Optimized + Pre-filter...")
    results.append(benchmark_with_prefilter(data, aperture_m))

    return results


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    baseline = results[0].time_s

    print(f"\n{'Kernel':<30} {'Time (s)':<12} {'Speedup':<12} {'Traces/s':<15}")
    print("-" * 80)

    for r in results:
        speedup = baseline / r.time_s
        print(f"{r.name:<30} {r.time_s:<12.3f} {speedup:<12.2f}x {r.traces_per_s:<15,.0f}")

    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION IMPACT")
    print("=" * 80)

    orig = results[0].time_s
    opt = results[1].time_s
    fast = results[2].time_s
    prefilt = results[3].time_s

    print(f"\nPre-computed terms: {(orig-opt)/orig*100:.1f}% improvement")
    print(f"Fast sqrt:          {(opt-fast)/opt*100:.1f}% additional improvement")
    print(f"Pre-filtering:      {(opt-prefilt)/opt*100:.1f}% additional improvement")
    print(f"\nTotal speedup:      {orig/prefilt:.2f}x faster than original")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Numba kernels")
    parser.add_argument("--traces", type=int, default=50_000, help="Number of traces")
    parser.add_argument("--tile-size", type=int, default=32, help="Tile size")
    parser.add_argument("--samples", type=int, default=500, help="Time samples")
    parser.add_argument("--aperture", type=float, default=2500.0, help="Aperture (m)")

    args = parser.parse_args()

    print("=" * 80)
    print("NUMBA KERNEL OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print()

    results = run_benchmarks(
        n_traces=args.traces,
        tile_size=args.tile_size,
        n_samples=args.samples,
        aperture_m=args.aperture,
    )

    print_results(results)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
