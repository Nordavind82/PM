#!/usr/bin/env python3
"""
Kernel Profiling Script for PSTM.

Profiles individual components of the Numba kernel to identify bottlenecks.
Creates timing breakdown table for each computational step.

Usage:
    python scripts/profile_kernel.py [--traces N] [--tile-size N] [--samples N]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit, prange

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])


# =============================================================================
# Component kernels for profiling (isolated versions of kernel steps)
# =============================================================================


@njit(cache=True, fastmath=True)
def _profile_distance_calc(
    ox_coords: np.ndarray,
    oy_coords: np.ndarray,
    midpoint_x: np.ndarray,
    midpoint_y: np.ndarray,
    max_aperture: float,
) -> int:
    """Profile distance calculation and aperture check."""
    n_traces = len(midpoint_x)
    nx = len(ox_coords)
    ny = len(oy_coords)
    n_pillars = nx * ny
    count = 0

    for idx in prange(n_pillars):
        ix = idx // ny
        iy = idx % ny
        ox = ox_coords[ix]
        oy = oy_coords[iy]

        for it in range(n_traces):
            mx = midpoint_x[it]
            my = midpoint_y[it]
            dist = np.sqrt((ox - mx) ** 2 + (oy - my) ** 2)
            if dist <= max_aperture:
                count += 1

    return count


@njit(cache=True, fastmath=True)
def _profile_aperture_calc(
    ot_coords_ms: np.ndarray,
    vrms_1d: np.ndarray,
    max_dip_deg: float,
    min_aperture: float,
    max_aperture: float,
    n_iterations: int,
) -> float:
    """Profile time-varying aperture calculation."""
    nt = len(ot_coords_ms)
    total = 0.0

    for _ in range(n_iterations):
        for iot in range(nt):
            t_ms = ot_coords_ms[iot]
            vrms = vrms_1d[iot]
            t_s = t_ms / 1000.0
            tan_dip = np.tan(max_dip_deg * np.pi / 180.0)
            aperture = vrms * t_s * 0.5 * tan_dip
            if aperture < min_aperture:
                aperture = min_aperture
            elif aperture > max_aperture:
                aperture = max_aperture
            total += aperture

    return total


@njit(cache=True, fastmath=True)
def _profile_dsr_travel_time(
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    ox: float,
    oy: float,
    t0_s_arr: np.ndarray,
    vrms_arr: np.ndarray,
) -> float:
    """Profile DSR travel time calculation."""
    n_traces = len(source_x)
    nt = len(t0_s_arr)
    total = 0.0

    for it in range(n_traces):
        sx = source_x[it]
        sy = source_y[it]
        rx = receiver_x[it]
        ry = receiver_y[it]

        ds2 = (ox - sx) ** 2 + (oy - sy) ** 2
        dr2 = (ox - rx) ** 2 + (oy - ry) ** 2

        for iot in range(nt):
            t0_s = t0_s_arr[iot]
            vrms = vrms_arr[iot]

            t0_half = t0_s * 0.5
            t0_half_sq = t0_half * t0_half
            v_sq = vrms * vrms

            t_src = np.sqrt(t0_half_sq + ds2 / v_sq)
            t_rec = np.sqrt(t0_half_sq + dr2 / v_sq)
            total += t_src + t_rec

    return total


@njit(cache=True, fastmath=True)
def _profile_interpolation_linear(
    amplitudes: np.ndarray,
    t_samples: np.ndarray,
) -> float:
    """Profile linear interpolation."""
    n_traces = amplitudes.shape[0]
    n_queries = len(t_samples)
    n_samples = amplitudes.shape[1]
    total = 0.0

    for it in range(n_traces):
        trace = amplitudes[it]
        for iq in range(n_queries):
            t = t_samples[iq]
            if t < 0.0 or t >= n_samples - 1:
                continue
            i0 = int(t)
            frac = t - i0
            total += trace[i0] * (1.0 - frac) + trace[i0 + 1] * frac

    return total


@njit(cache=True, fastmath=True)
def _sinc_value(x: float) -> float:
    """Compute sinc(x) = sin(pi*x) / (pi*x)."""
    if abs(x) < 1e-8:
        return 1.0
    px = np.pi * x
    return np.sin(px) / px


@njit(cache=True, fastmath=True)
def _profile_interpolation_sinc8(
    amplitudes: np.ndarray,
    t_samples: np.ndarray,
) -> float:
    """Profile 8-point sinc interpolation."""
    n_traces = amplitudes.shape[0]
    n_queries = len(t_samples)
    n_samples = amplitudes.shape[1]
    total = 0.0

    for it in range(n_traces):
        trace = amplitudes[it]
        for iq in range(n_queries):
            t = t_samples[iq]
            if t < 3.0 or t >= n_samples - 4:
                continue

            i_center = int(t + 0.5)
            frac = t - i_center

            result = 0.0
            for k in range(-3, 5):
                idx = i_center + k
                if 0 <= idx < n_samples:
                    result += trace[idx] * _sinc_value(frac - k)

            total += result

    return total


@njit(cache=True, fastmath=True)
def _profile_weight_taper(
    distances: np.ndarray,
    aperture: float,
    taper_fraction: float,
) -> float:
    """Profile taper weight calculation."""
    total = 0.0
    taper_start = aperture * (1.0 - taper_fraction)

    for dist in distances:
        if dist >= aperture:
            w = 0.0
        elif dist <= taper_start:
            w = 1.0
        else:
            x = (dist - taper_start) / (aperture - taper_start)
            w = 0.5 * (1.0 + np.cos(np.pi * x))
        total += w

    return total


@njit(cache=True, fastmath=True)
def _profile_weight_spreading(
    t_s_arr: np.ndarray,
    vrms_arr: np.ndarray,
) -> float:
    """Profile spreading weight calculation."""
    total = 0.0
    n = len(t_s_arr)

    for i in range(n):
        t_s = t_s_arr[i]
        vrms = vrms_arr[i]
        if t_s < 0.001:
            w = 0.0
        else:
            w = 1.0 / (vrms * t_s)
        total += w

    return total


@njit(cache=True, fastmath=True)
def _profile_weight_obliquity(
    t0_s_arr: np.ndarray,
    t_total_arr: np.ndarray,
) -> float:
    """Profile obliquity weight calculation."""
    total = 0.0
    n = len(t0_s_arr)

    for i in range(n):
        t0 = t0_s_arr[i]
        t_total = t_total_arr[i]
        if t_total < 0.001:
            w = 0.0
        else:
            w = t0 / t_total
        total += w

    return total


@njit(parallel=True, cache=True, fastmath=True)
def _profile_accumulation(
    image: np.ndarray,
    values: np.ndarray,
    ix_arr: np.ndarray,
    iy_arr: np.ndarray,
    it_arr: np.ndarray,
) -> None:
    """Profile accumulation to output."""
    n = len(values)

    for i in prange(n):
        ix = ix_arr[i]
        iy = iy_arr[i]
        it = it_arr[i]
        image[ix, iy, it] += values[i]


@njit(parallel=True, cache=True, fastmath=True)
def _profile_full_inner_loop(
    amplitudes: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    midpoint_x: np.ndarray,
    midpoint_y: np.ndarray,
    dt_in_ms: float,
    t_start_in_ms: float,
    image: np.ndarray,
    ox_coords: np.ndarray,
    oy_coords: np.ndarray,
    ot_coords_ms: np.ndarray,
    vrms_1d: np.ndarray,
    max_aperture: float,
) -> int:
    """Profile full inner loop without weights (DSR + interpolation + accumulation)."""
    n_traces = amplitudes.shape[0]
    n_samples_in = amplitudes.shape[1]
    nx = len(ox_coords)
    ny = len(oy_coords)
    nt = len(ot_coords_ms)
    n_pillars = nx * ny
    count = 0

    for idx in prange(n_pillars):
        ix = idx // ny
        iy = idx % ny
        ox = ox_coords[ix]
        oy = oy_coords[iy]

        for it in range(n_traces):
            sx = source_x[it]
            sy = source_y[it]
            rx = receiver_x[it]
            ry = receiver_y[it]
            mx = midpoint_x[it]
            my = midpoint_y[it]

            dist = np.sqrt((ox - mx) ** 2 + (oy - my) ** 2)
            if dist > max_aperture:
                continue

            ds2 = (ox - sx) ** 2 + (oy - sy) ** 2
            dr2 = (ox - rx) ** 2 + (oy - ry) ** 2

            for iot in range(nt):
                t0_ms = ot_coords_ms[iot]
                t0_s = t0_ms / 1000.0
                vrms = vrms_1d[iot]

                t0_half = t0_s * 0.5
                t0_half_sq = t0_half * t0_half
                v_sq = vrms * vrms

                t_src = np.sqrt(t0_half_sq + ds2 / v_sq)
                t_rec = np.sqrt(t0_half_sq + dr2 / v_sq)
                t_total_s = t_src + t_rec
                t_total_ms = t_total_s * 1000.0

                t_sample = (t_total_ms - t_start_in_ms) / dt_in_ms

                if t_sample < 0.0 or t_sample >= n_samples_in - 1:
                    continue

                # Linear interpolation
                i0 = int(t_sample)
                frac = t_sample - i0
                amp = amplitudes[it, i0] * (1.0 - frac) + amplitudes[it, i0 + 1] * frac

                image[ix, iy, iot] += amp
                count += 1

    return count


# =============================================================================
# Profiling harness
# =============================================================================


@dataclass
class ProfileResult:
    """Result from profiling a single component."""
    name: str
    time_s: float
    iterations: int
    ops_per_iter: int

    @property
    def time_per_iter_us(self) -> float:
        """Time per iteration in microseconds."""
        return (self.time_s / self.iterations) * 1e6

    @property
    def ops_per_s(self) -> float:
        """Operations per second."""
        return (self.iterations * self.ops_per_iter) / self.time_s if self.time_s > 0 else 0


def warmup_jit():
    """Warm up JIT compilation for all profiled functions."""
    print("Warming up JIT compilation...")

    n = 10
    amp = np.random.randn(n, 100).astype(np.float32)
    coords = np.random.randn(n).astype(np.float64) * 1000
    ox = np.linspace(0, 100, 4)
    oy = np.linspace(0, 100, 4)
    ot_ms = np.linspace(0, 1000, 50)
    vrms = np.full(50, 2000.0)
    t_samples = np.random.rand(n) * 90 + 5
    distances = np.random.rand(n) * 2000
    image = np.zeros((4, 4, 50), dtype=np.float64)

    # Call each function once to compile
    _profile_distance_calc(ox, oy, coords, coords, 2500.0)
    _profile_aperture_calc(ot_ms, vrms, 45.0, 100.0, 5000.0, 10)
    _profile_dsr_travel_time(coords, coords, coords, coords, 50.0, 50.0, ot_ms / 1000.0, vrms)
    _profile_interpolation_linear(amp, t_samples)
    _profile_interpolation_sinc8(amp, t_samples)
    _profile_weight_taper(distances, 2500.0, 0.1)
    _profile_weight_spreading(ot_ms / 1000.0, vrms)
    _profile_weight_obliquity(ot_ms / 1000.0, ot_ms / 1000.0 * 1.1)
    _profile_accumulation(
        image,
        np.random.randn(100).astype(np.float64),
        np.random.randint(0, 4, 100).astype(np.int64),
        np.random.randint(0, 4, 100).astype(np.int64),
        np.random.randint(0, 50, 100).astype(np.int64),
    )
    _profile_full_inner_loop(
        amp, coords, coords, coords, coords, coords, coords,
        4.0, 0.0, image, ox, oy, ot_ms, vrms, 2500.0
    )

    print("JIT compilation complete.\n")


def generate_test_data(
    n_traces: int,
    tile_size: int,
    n_samples: int,
    aperture_m: float,
) -> dict:
    """Generate realistic test data."""
    # Trace geometry - distributed around tile center
    spread = aperture_m * 0.8  # Keep most traces within aperture

    midpoint_x = np.random.randn(n_traces).astype(np.float64) * spread + tile_size * 12.5
    midpoint_y = np.random.randn(n_traces).astype(np.float64) * spread + tile_size * 12.5

    # Random offsets for source/receiver
    offset = 500.0
    angles = np.random.rand(n_traces) * 2 * np.pi
    source_x = midpoint_x - offset * np.cos(angles)
    source_y = midpoint_y - offset * np.sin(angles)
    receiver_x = midpoint_x + offset * np.cos(angles)
    receiver_y = midpoint_y + offset * np.sin(angles)

    # Output grid
    ox = np.linspace(0, tile_size * 25, tile_size).astype(np.float64)
    oy = np.linspace(0, tile_size * 25, tile_size).astype(np.float64)
    ot_ms = np.linspace(0, n_samples * 4.0, n_samples).astype(np.float64)  # 4ms sample rate

    # Velocity model
    vrms = np.linspace(1800, 3500, n_samples).astype(np.float64)

    # Amplitudes
    amplitudes = np.random.randn(n_traces, n_samples).astype(np.float32)

    return {
        "amplitudes": amplitudes,
        "source_x": source_x,
        "source_y": source_y,
        "receiver_x": receiver_x,
        "receiver_y": receiver_y,
        "midpoint_x": midpoint_x,
        "midpoint_y": midpoint_y,
        "ox": ox,
        "oy": oy,
        "ot_ms": ot_ms,
        "vrms": vrms,
        "n_traces": n_traces,
        "tile_size": tile_size,
        "n_samples": n_samples,
    }


def profile_component(
    name: str,
    func: Callable,
    min_time_s: float = 0.5,
) -> ProfileResult:
    """Profile a component function."""
    # Run once to get baseline
    start = time.perf_counter()
    result = func()
    single_time = time.perf_counter() - start

    # Determine iteration count
    if single_time > 0.001:
        n_iter = max(1, int(min_time_s / single_time))
    else:
        n_iter = max(10, int(min_time_s / max(single_time, 1e-6)))

    n_iter = min(n_iter, 10000)  # Cap iterations

    # Timed run
    start = time.perf_counter()
    for _ in range(n_iter):
        result = func()
    elapsed = time.perf_counter() - start

    ops = result if isinstance(result, int) else 1

    return ProfileResult(
        name=name,
        time_s=elapsed,
        iterations=n_iter,
        ops_per_iter=ops,
    )


def run_profiling(
    n_traces: int = 100_000,
    tile_size: int = 32,
    n_samples: int = 500,
    aperture_m: float = 2500.0,
) -> list[ProfileResult]:
    """Run profiling on all kernel components."""

    print(f"Configuration:")
    print(f"  Traces:      {n_traces:,}")
    print(f"  Tile size:   {tile_size}×{tile_size} = {tile_size**2:,} pillars")
    print(f"  Samples:     {n_samples}")
    print(f"  Aperture:    {aperture_m:.0f} m")
    print()

    data = generate_test_data(n_traces, tile_size, n_samples, aperture_m)

    # Calculate expected operations for reference
    n_pillars = tile_size * tile_size
    traces_in_aperture = int(n_traces * 0.43)  # Estimate based on gaussian distribution
    total_dsr_calls = n_pillars * traces_in_aperture * n_samples

    print(f"Expected operations per tile:")
    print(f"  Pillars × Traces × Samples ≈ {n_pillars:,} × {traces_in_aperture:,} × {n_samples:,}")
    print(f"  Total DSR calls ≈ {total_dsr_calls:,}")
    print()

    results = []

    # 1. Distance calculation (pillar × trace)
    print("Profiling 1/9: Distance calculation...")
    results.append(profile_component(
        "Distance calc (pillar×trace)",
        lambda: _profile_distance_calc(
            data["ox"], data["oy"],
            data["midpoint_x"], data["midpoint_y"],
            aperture_m
        ),
    ))

    # 2. Aperture calculation (time samples)
    print("Profiling 2/9: Aperture calculation...")
    n_aperture_iter = n_pillars * traces_in_aperture
    results.append(profile_component(
        "Aperture calc (per time)",
        lambda: _profile_aperture_calc(
            data["ot_ms"], data["vrms"],
            45.0, 100.0, aperture_m, n_aperture_iter
        ),
    ))

    # 3. DSR travel time
    print("Profiling 3/9: DSR travel time...")
    results.append(profile_component(
        "DSR travel time",
        lambda: _profile_dsr_travel_time(
            data["source_x"][:1000],
            data["source_y"][:1000],
            data["receiver_x"][:1000],
            data["receiver_y"][:1000],
            data["ox"][tile_size//2],
            data["oy"][tile_size//2],
            data["ot_ms"] / 1000.0,
            data["vrms"],
        ),
    ))

    # 4. Linear interpolation
    print("Profiling 4/9: Linear interpolation...")
    t_samples = np.random.rand(n_samples) * (n_samples - 10) + 5
    results.append(profile_component(
        "Linear interpolation",
        lambda: _profile_interpolation_linear(
            data["amplitudes"][:1000], t_samples
        ),
    ))

    # 5. Sinc8 interpolation
    print("Profiling 5/9: Sinc8 interpolation...")
    results.append(profile_component(
        "Sinc8 interpolation",
        lambda: _profile_interpolation_sinc8(
            data["amplitudes"][:1000], t_samples
        ),
    ))

    # 6. Taper weight
    print("Profiling 6/9: Taper weight...")
    distances = np.sqrt(
        (data["ox"][tile_size//2] - data["midpoint_x"])**2 +
        (data["oy"][tile_size//2] - data["midpoint_y"])**2
    )
    results.append(profile_component(
        "Taper weight",
        lambda: _profile_weight_taper(distances, aperture_m, 0.1),
    ))

    # 7. Spreading weight
    print("Profiling 7/9: Spreading weight...")
    t_s_arr = np.random.rand(n_traces * n_samples // 100) * 2 + 0.1
    vrms_arr = np.full_like(t_s_arr, 2500.0)
    results.append(profile_component(
        "Spreading weight",
        lambda: _profile_weight_spreading(t_s_arr, vrms_arr),
    ))

    # 8. Obliquity weight
    print("Profiling 8/9: Obliquity weight...")
    t0_arr = np.random.rand(n_traces * n_samples // 100) * 2 + 0.1
    t_total_arr = t0_arr * 1.1
    results.append(profile_component(
        "Obliquity weight",
        lambda: _profile_weight_obliquity(t0_arr, t_total_arr),
    ))

    # 9. Full inner loop (DSR + interp + accumulation, no weights)
    print("Profiling 9/9: Full inner loop (no weights)...")
    image = np.zeros((tile_size, tile_size, n_samples), dtype=np.float64)
    subset_traces = min(n_traces, 50000)  # Limit for reasonable timing
    results.append(profile_component(
        "Full inner loop (no weights)",
        lambda: _profile_full_inner_loop(
            data["amplitudes"][:subset_traces],
            data["source_x"][:subset_traces],
            data["source_y"][:subset_traces],
            data["receiver_x"][:subset_traces],
            data["receiver_y"][:subset_traces],
            data["midpoint_x"][:subset_traces],
            data["midpoint_y"][:subset_traces],
            4.0, 0.0, image,
            data["ox"], data["oy"], data["ot_ms"], data["vrms"],
            aperture_m,
        ),
        min_time_s=1.0,
    ))

    return results


def estimate_tile_breakdown(results: list[ProfileResult], n_traces: int, tile_size: int, n_samples: int) -> dict:
    """Estimate time breakdown for a full tile migration."""
    n_pillars = tile_size * tile_size
    traces_in_aperture = int(n_traces * 0.43)

    # Extrapolate from profiled results
    estimates = {}

    for r in results:
        if "Distance" in r.name:
            # Already tested at full scale
            estimates["Distance calculation"] = r.time_s / r.iterations
        elif "Aperture" in r.name:
            # Scales with pillars × traces × samples
            estimates["Aperture calculation"] = r.time_s / r.iterations
        elif "DSR" in r.name:
            # Scale: tested 1000 traces × n_samples, need pillars × traces × samples
            scale = (n_pillars * traces_in_aperture) / 1000
            estimates["DSR travel time"] = (r.time_s / r.iterations) * scale
        elif "Linear" in r.name:
            # Scale from 1000 traces
            scale = (n_pillars * traces_in_aperture) / 1000
            estimates["Linear interpolation"] = (r.time_s / r.iterations) * scale
        elif "Sinc8" in r.name:
            scale = (n_pillars * traces_in_aperture) / 1000
            estimates["Sinc8 interpolation"] = (r.time_s / r.iterations) * scale
        elif "Taper" in r.name:
            # Scale: tested n_traces, need pillars × samples per contributing trace
            scale = (n_pillars * n_samples) / n_traces
            estimates["Taper weight"] = (r.time_s / r.iterations) * scale
        elif "Spreading" in r.name:
            # Small contribution, scale similarly
            estimates["Spreading weight"] = (r.time_s / r.iterations) * 0.1
        elif "Obliquity" in r.name:
            estimates["Obliquity weight"] = (r.time_s / r.iterations) * 0.1
        elif "Full" in r.name:
            # This is the actual full loop - use directly
            # Scale to full trace count
            tested_traces = min(n_traces, 50000)
            scale = n_traces / tested_traces
            estimates["Full kernel (no weights)"] = (r.time_s / r.iterations) * scale

    return estimates


def print_results(results: list[ProfileResult], n_traces: int, tile_size: int, n_samples: int):
    """Print profiling results as a table."""
    print("\n" + "=" * 80)
    print("KERNEL COMPONENT PROFILING RESULTS")
    print("=" * 80)

    print(f"\n{'Component':<35} {'Time (ms)':<12} {'Time/iter (µs)':<15} {'Ops/sec':<15}")
    print("-" * 80)

    for r in results:
        time_ms = r.time_s * 1000
        print(f"{r.name:<35} {time_ms:<12.3f} {r.time_per_iter_us:<15.2f} {r.ops_per_s:<15,.0f}")

    # Estimate full tile breakdown
    print("\n" + "=" * 80)
    print("ESTIMATED TIME BREAKDOWN FOR SINGLE TILE MIGRATION")
    print("=" * 80)

    estimates = estimate_tile_breakdown(results, n_traces, tile_size, n_samples)

    total_estimated = sum(estimates.values())

    print(f"\n{'Component':<40} {'Est. Time (s)':<15} {'% of Total':<15}")
    print("-" * 70)

    for name, time_s in sorted(estimates.items(), key=lambda x: -x[1]):
        pct = (time_s / total_estimated * 100) if total_estimated > 0 else 0
        print(f"{name:<40} {time_s:<15.3f} {pct:<15.1f}%")

    print("-" * 70)
    print(f"{'TOTAL ESTIMATED':<40} {total_estimated:<15.3f}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY BOTTLENECK ANALYSIS")
    print("=" * 80)

    if estimates:
        top_component = max(estimates.items(), key=lambda x: x[1])
        print(f"\n1. PRIMARY BOTTLENECK: {top_component[0]}")
        print(f"   Accounts for {top_component[1]/total_estimated*100:.1f}% of kernel time")

        if "DSR" in top_component[0]:
            print("\n   ANALYSIS: DSR travel time dominates due to:")
            print("   - 2 sqrt() operations per sample")
            print("   - Called for every (pillar × trace × time_sample) combination")
            print(f"   - At {n_traces:,} traces, {tile_size}²={tile_size**2} pillars, {n_samples} samples:")
            print(f"     That's {tile_size**2 * int(n_traces*0.43) * n_samples:,} DSR calls!")
            print("\n   OPTIMIZATION OPPORTUNITIES:")
            print("   - Pre-compute distance² terms outside time loop")
            print("   - Vectorize over time samples using numpy")
            print("   - Consider lookup table for sqrt approximation")

        if "interpolation" in top_component[0].lower():
            print("\n   ANALYSIS: Interpolation dominates due to:")
            print("   - Memory access pattern (trace samples)")
            print("   - Sinc8 requires 8 multiplies + 8 sinc() calls per sample")
            print("\n   OPTIMIZATION OPPORTUNITIES:")
            print("   - Use linear interpolation (3-5x faster)")
            print("   - Pre-compute sinc coefficients")
            print("   - Optimize memory access with blocked processing")

    # Compare linear vs sinc8
    linear_time = next((r.time_s/r.iterations for r in results if "Linear" in r.name), 0)
    sinc8_time = next((r.time_s/r.iterations for r in results if "Sinc8" in r.name), 0)
    if linear_time > 0 and sinc8_time > 0:
        print(f"\n2. INTERPOLATION COMPARISON:")
        print(f"   Linear: {linear_time*1000:.3f} ms")
        print(f"   Sinc8:  {sinc8_time*1000:.3f} ms")
        print(f"   Sinc8 is {sinc8_time/linear_time:.1f}x slower than linear")


def main():
    parser = argparse.ArgumentParser(description="Profile PSTM kernel components")
    parser.add_argument("--traces", type=int, default=100_000, help="Number of traces")
    parser.add_argument("--tile-size", type=int, default=32, help="Tile size (NxN)")
    parser.add_argument("--samples", type=int, default=500, help="Time samples per trace")
    parser.add_argument("--aperture", type=float, default=2500.0, help="Aperture radius in meters")

    args = parser.parse_args()

    print("=" * 80)
    print("PSTM KERNEL PROFILING")
    print("=" * 80)
    print()

    warmup_jit()

    results = run_profiling(
        n_traces=args.traces,
        tile_size=args.tile_size,
        n_samples=args.samples,
        aperture_m=args.aperture,
    )

    print_results(results, args.traces, args.tile_size, args.samples)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
