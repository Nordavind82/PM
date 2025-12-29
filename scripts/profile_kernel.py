#!/usr/bin/env python3
"""
Profile PSTM kernel performance to identify bottlenecks.

Measures:
- Trace loading time
- Kernel execution time
- Memory bandwidth
- Impact of grid size on performance
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np


def timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{timestamp()}] {msg}", flush=True)


def profile_kernel_execution(
    n_traces: int = 50000,
    n_samples: int = 501,
    grid_sizes: list[int] = [32, 64, 128],
    aperture_m: float = 2000.0,
    n_iterations: int = 3,
):
    """Profile kernel with different grid sizes."""

    from pstm.kernels.base import (
        KernelConfig, OutputTile, VelocitySlice, create_trace_block
    )
    from pstm.kernels.factory import create_kernel

    log("=" * 70)
    log("PSTM KERNEL PROFILING")
    log("=" * 70)
    log(f"Traces: {n_traces:,}, Samples: {n_samples}, Aperture: {aperture_m}m")
    log("")

    # Generate test data
    log("Generating test data...")
    np.random.seed(42)

    survey_extent = 5000.0  # 5km survey
    n_per_side = int(np.sqrt(n_traces))

    x_base = np.linspace(0, survey_extent, n_per_side)
    y_base = np.linspace(0, survey_extent, n_per_side)
    xx, yy = np.meshgrid(x_base, y_base)

    midpoint_x = xx.ravel()[:n_traces] + np.random.randn(min(n_traces, n_per_side**2)) * 10
    midpoint_y = yy.ravel()[:n_traces] + np.random.randn(min(n_traces, n_per_side**2)) * 10

    if len(midpoint_x) < n_traces:
        extra = n_traces - len(midpoint_x)
        midpoint_x = np.concatenate([midpoint_x, np.random.uniform(0, survey_extent, extra)])
        midpoint_y = np.concatenate([midpoint_y, np.random.uniform(0, survey_extent, extra)])

    offsets = np.random.uniform(200, 250, n_traces)
    azimuths = np.random.uniform(0, 2*np.pi, n_traces)
    half_offset_x = offsets / 2 * np.cos(azimuths)
    half_offset_y = offsets / 2 * np.sin(azimuths)

    source_x = (midpoint_x - half_offset_x).astype(np.float32)
    source_y = (midpoint_y - half_offset_y).astype(np.float32)
    receiver_x = (midpoint_x + half_offset_x).astype(np.float32)
    receiver_y = (midpoint_y + half_offset_y).astype(np.float32)

    # Generate trace amplitudes
    traces = np.random.randn(n_traces, n_samples).astype(np.float32) * 0.1
    log(f"  Trace data: {traces.nbytes / 1024**2:.1f} MB")

    # Create kernel
    config = KernelConfig(
        max_aperture_m=aperture_m,
        min_aperture_m=100.0,
        taper_fraction=0.1,
        max_dip_degrees=60.0,
        apply_spreading=True,
        apply_obliquity=True,
        aa_enabled=False,
    )

    kernel = create_kernel("metal_compiled")
    kernel.initialize(config)
    log("  Kernel initialized")

    # Velocity model
    velocity = 2500.0
    velocity_1d = np.full(n_samples, velocity, dtype=np.float32)
    vel_slice = VelocitySlice(vrms=velocity_1d, is_1d=True)

    x_min, x_max = midpoint_x.min(), midpoint_x.max()
    y_min, y_max = midpoint_y.min(), midpoint_y.max()
    dt_ms = 2.0
    max_time_ms = 1000.0

    log("")
    log("GRID SIZE COMPARISON")
    log("-" * 70)
    log(f"{'Grid Size':>12} {'Traces':>12} {'Kernel (ms)':>12} {'Throughput':>15}")
    log("-" * 70)

    results = []

    for grid_size in grid_sizes:
        nx = ny = grid_size
        nt = int(max_time_ms / dt_ms) + 1
        dx = (x_max - x_min) / (nx - 1) if nx > 1 else 25.0
        dy = (y_max - y_min) / (ny - 1) if ny > 1 else 25.0

        # Use single-tile processing (all traces to one output)
        x_axis = np.linspace(x_min, x_max, nx).astype(np.float32)
        y_axis = np.linspace(y_min, y_max, ny).astype(np.float32)
        t_axis_ms = np.arange(0, max_time_ms + dt_ms/2, dt_ms).astype(np.float32)

        # Create trace block with all traces
        trace_block = create_trace_block(
            amplitudes=traces,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            sample_rate_ms=dt_ms,
            start_time_ms=0.0,
        )

        kernel_times = []

        for iteration in range(n_iterations):
            # Create fresh output tile
            output = OutputTile(
                image=np.zeros((nx, ny, nt), dtype=np.float64),
                fold=np.zeros((nx, ny), dtype=np.int32),
                x_axis=x_axis,
                y_axis=y_axis,
                t_axis_ms=t_axis_ms,
            )

            # Time kernel
            gc.collect()
            t0 = time.perf_counter()
            metrics = kernel.migrate_tile(trace_block, output, vel_slice, config)
            kernel_time = time.perf_counter() - t0
            kernel_times.append(kernel_time)

        avg_time = np.mean(kernel_times)
        throughput = n_traces / avg_time

        log(f"{grid_size}x{grid_size}x{nt:>4} {n_traces:>12,} {avg_time*1000:>12.1f} {throughput:>12,.0f}/s")

        results.append({
            'grid_size': grid_size,
            'nx': nx, 'ny': ny, 'nt': nt,
            'n_output_points': nx * ny * nt,
            'kernel_time_ms': avg_time * 1000,
            'throughput': throughput,
        })

    log("-" * 70)

    # Analyze scaling
    if len(results) >= 2:
        log("")
        log("SCALING ANALYSIS")
        log("-" * 70)

        base = results[0]
        for r in results[1:]:
            grid_ratio = r['n_output_points'] / base['n_output_points']
            time_ratio = r['kernel_time_ms'] / base['kernel_time_ms']
            log(f"  {base['grid_size']}x{base['grid_size']} -> {r['grid_size']}x{r['grid_size']}: "
                f"output {grid_ratio:.1f}x, time {time_ratio:.1f}x, "
                f"efficiency {grid_ratio/time_ratio:.2f}")

    log("")
    log("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile PSTM kernel performance")
    parser.add_argument("--n-traces", type=int, default=50000, help="Number of traces")
    parser.add_argument("--n-samples", type=int, default=501, help="Samples per trace")

    args = parser.parse_args()

    profile_kernel_execution(
        n_traces=args.n_traces,
        n_samples=args.n_samples,
        grid_sizes=[32, 64, 128, 256],
    )


if __name__ == "__main__":
    main()
