#!/usr/bin/env python3
"""
Simple kernel scaling test - runs kernels directly and measures time.
Tests both backends at different trace counts.
"""

import sys
import time
import numpy as np
from dataclasses import dataclass

# Test configs
TRACE_COUNTS = [1000, 5000, 10000, 20000, 50000, 100000]
OUTPUT_GRID = (50, 50, 100)  # nx, ny, nt


def create_data(n_traces: int, n_samples: int = 1000):
    """Create synthetic test data."""
    rng = np.random.default_rng(42)
    return {
        'amplitudes': rng.standard_normal((n_traces, n_samples)).astype(np.float32),
        'source_x': rng.uniform(0, 10000, n_traces).astype(np.float64),
        'source_y': rng.uniform(0, 10000, n_traces).astype(np.float64),
        'receiver_x': rng.uniform(0, 10000, n_traces).astype(np.float64),
        'receiver_y': rng.uniform(0, 10000, n_traces).astype(np.float64),
        'midpoint_x': rng.uniform(2000, 8000, n_traces).astype(np.float64),
        'midpoint_y': rng.uniform(2000, 8000, n_traces).astype(np.float64),
    }


def test_kernel(backend: str, n_traces: int) -> tuple[float, str]:
    """Test a kernel with given trace count. Returns (time, error)."""
    from pstm.kernels.factory import create_kernel
    from pstm.kernels.base import (
        KernelConfig, TraceBlock, OutputTile, VelocitySlice, create_trace_block
    )

    nx, ny, nt = OUTPUT_GRID

    try:
        # Create kernel
        kernel = create_kernel(backend)

        config = KernelConfig(
            max_aperture_m=3000.0,
            min_aperture_m=100.0,
            max_dip_degrees=45.0,
            taper_fraction=0.1,
            apply_spreading=True,
            apply_obliquity=True,
        )
        kernel.initialize(config)

        # Create data
        data = create_data(n_traces)
        traces = create_trace_block(
            amplitudes=data['amplitudes'],
            source_x=data['source_x'],
            source_y=data['source_y'],
            receiver_x=data['receiver_x'],
            receiver_y=data['receiver_y'],
            sample_rate_ms=2.0,
            start_time_ms=0.0,
        )

        # Output
        output = OutputTile(
            image=np.zeros((nx, ny, nt), dtype=np.float64),
            fold=np.zeros((nx, ny), dtype=np.int32),
            x_axis=np.linspace(3000, 7000, nx),
            y_axis=np.linspace(3000, 7000, ny),
            t_axis_ms=np.linspace(0, 4000, nt),
        )

        # Velocity
        velocity = VelocitySlice(
            vrms=np.linspace(1500, 3500, nt),
            is_1d=True,
            t_axis_ms=output.t_axis_ms,
        )

        # Run
        print(f"    Running {backend} with {n_traces:,} traces...", flush=True)
        start = time.time()
        metrics = kernel.migrate_tile(traces, output, velocity, config)
        elapsed = time.time() - start

        kernel.cleanup()
        return elapsed, ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 0, str(e)


def main():
    print("=" * 60)
    print("KERNEL SCALING TEST (Direct Execution)")
    print("=" * 60)
    print(f"Output grid: {OUTPUT_GRID}")
    print()

    # Get backends
    from pstm.kernels.factory import get_available_backends
    backends = [b.value for b in get_available_backends()]
    print(f"Available: {backends}")

    test_backends = []
    if 'numba_cpu' in backends:
        test_backends.append('numba_cpu')
    if 'mlx_metal' in backends:
        test_backends.append('mlx_metal')

    print(f"Testing: {test_backends}")
    print()

    # Get trace counts from args
    trace_counts = TRACE_COUNTS
    if len(sys.argv) > 1:
        trace_counts = [int(x) for x in sys.argv[1].split(",")]

    results = []

    for backend in test_backends:
        print(f"\n--- {backend.upper()} ---")

        for n_traces in trace_counts:
            elapsed, error = test_kernel(backend, n_traces)

            if error:
                print(f"  {n_traces:>10,}: ERROR - {error}")
                results.append((backend, n_traces, 0, error))
            else:
                rate = n_traces / elapsed if elapsed > 0 else 0
                print(f"  {n_traces:>10,}: {elapsed:6.2f}s ({rate:,.0f} traces/s)")
                results.append((backend, n_traces, elapsed, ""))

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Backend':<12} {'Traces':>10} {'Time':>10} {'Rate':>15}")
    print("-" * 50)

    for backend, n_traces, elapsed, error in results:
        if error:
            print(f"{backend:<12} {n_traces:>10,} {'ERROR':>10} {error[:20]}")
        else:
            rate = n_traces / elapsed if elapsed > 0 else 0
            print(f"{backend:<12} {n_traces:>10,} {elapsed:>9.2f}s {rate:>12,.0f}/s")

    print()
    return 0


if __name__ == "__main__":
    main()
