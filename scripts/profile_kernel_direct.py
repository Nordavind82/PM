#!/usr/bin/env python3
"""
Direct Kernel Profiling for PSTM.

Profiles the actual kernel function with different configurations
to identify bottlenecks. Creates timing breakdown table.

Usage:
    python scripts/profile_kernel_direct.py [--traces N] [--tile-size N] [--samples N]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from pstm.kernels.numba_cpu import _migrate_tile_kernel
from pstm.kernels.interpolation import get_method_code


@dataclass
class KernelTimingResult:
    """Result from a kernel timing run."""
    name: str
    config_desc: str
    time_s: float
    n_traces: int
    n_pillars: int
    n_samples: int
    contributions: int

    @property
    def traces_per_s(self) -> float:
        return self.n_traces / self.time_s if self.time_s > 0 else 0

    @property
    def ops_per_s(self) -> float:
        """Operations: traces × pillars × samples (max possible)."""
        return (self.n_traces * self.n_pillars * self.n_samples) / self.time_s if self.time_s > 0 else 0


def generate_test_data(
    n_traces: int,
    tile_size: int,
    n_samples: int,
    aperture_m: float,
) -> dict:
    """Generate realistic test data for kernel profiling."""
    np.random.seed(42)  # Reproducible

    # Tile covers 800m × 800m (32 × 25m spacing)
    tile_extent = tile_size * 25

    # Trace geometry - distributed around tile center
    spread = aperture_m * 0.8

    midpoint_x = np.random.randn(n_traces).astype(np.float64) * spread + tile_extent / 2
    midpoint_y = np.random.randn(n_traces).astype(np.float64) * spread + tile_extent / 2

    # Random offsets for source/receiver (500m half-offset)
    offset = 500.0
    angles = np.random.rand(n_traces) * 2 * np.pi
    source_x = midpoint_x - offset * np.cos(angles)
    source_y = midpoint_y - offset * np.sin(angles)
    receiver_x = midpoint_x + offset * np.cos(angles)
    receiver_y = midpoint_y + offset * np.sin(angles)

    # Output grid
    ox = np.linspace(0, tile_extent, tile_size).astype(np.float64)
    oy = np.linspace(0, tile_extent, tile_size).astype(np.float64)
    ot_ms = np.linspace(0, n_samples * 4.0, n_samples).astype(np.float64)

    # Velocity model (increasing with depth)
    vrms = np.linspace(1800, 3500, n_samples).astype(np.float64)

    # Amplitudes with realistic seismic wavelet
    amplitudes = np.random.randn(n_traces, n_samples).astype(np.float32)

    # Trace weights
    trace_weights = np.ones(n_traces, dtype=np.float64)

    return {
        "amplitudes": amplitudes,
        "source_x": source_x,
        "source_y": source_y,
        "receiver_x": receiver_x,
        "receiver_y": receiver_y,
        "midpoint_x": midpoint_x,
        "midpoint_y": midpoint_y,
        "trace_weights": trace_weights,
        "ox": ox,
        "oy": oy,
        "ot_ms": ot_ms,
        "vrms": vrms,
        "n_traces": n_traces,
        "tile_size": tile_size,
        "n_samples": n_samples,
        "dt_ms": 4.0,
        "t_start_ms": 0.0,
    }


def warmup_kernel(data: dict, interp_method: int = 1):
    """Warm up JIT compilation."""
    print("Warming up JIT compilation...")

    # Small warmup arrays
    n_warmup = 100
    warmup_amp = data["amplitudes"][:n_warmup].copy()
    warmup_sx = data["source_x"][:n_warmup].copy()
    warmup_sy = data["source_y"][:n_warmup].copy()
    warmup_rx = data["receiver_x"][:n_warmup].copy()
    warmup_ry = data["receiver_y"][:n_warmup].copy()
    warmup_mx = data["midpoint_x"][:n_warmup].copy()
    warmup_my = data["midpoint_y"][:n_warmup].copy()
    warmup_w = np.ones(n_warmup, dtype=np.float64)

    image = np.zeros((4, 4, 50), dtype=np.float64)
    fold = np.zeros((4, 4), dtype=np.int32)
    ox = data["ox"][:4].copy()
    oy = data["oy"][:4].copy()
    ot = data["ot_ms"][:50].copy()
    vrms = data["vrms"][:50].copy()

    # Run once to compile
    _migrate_tile_kernel(
        warmup_amp, warmup_sx, warmup_sy, warmup_rx, warmup_ry,
        warmup_mx, warmup_my, warmup_w,
        4.0, 0.0,
        image, fold, ox, oy, ot, vrms,
        45.0, 100.0, 5000.0, 0.1,
        True, True, interp_method,
    )

    print("JIT compilation complete.\n")


def run_kernel_timing(
    data: dict,
    name: str,
    config_desc: str,
    interp_method: int = 1,
    apply_spreading: bool = True,
    apply_obliquity: bool = True,
    max_aperture: float = 2500.0,
    n_runs: int = 1,
) -> KernelTimingResult:
    """Run kernel with specific configuration and measure time."""
    tile_size = data["tile_size"]
    n_samples = data["n_samples"]
    n_traces = data["n_traces"]

    # Fresh output arrays
    image = np.zeros((tile_size, tile_size, n_samples), dtype=np.float64)
    fold = np.zeros((tile_size, tile_size), dtype=np.int32)

    # Time the kernel
    total_time = 0.0
    for _ in range(n_runs):
        image.fill(0)
        fold.fill(0)

        start = time.perf_counter()
        contributions = _migrate_tile_kernel(
            data["amplitudes"],
            data["source_x"],
            data["source_y"],
            data["receiver_x"],
            data["receiver_y"],
            data["midpoint_x"],
            data["midpoint_y"],
            data["trace_weights"],
            data["dt_ms"],
            data["t_start_ms"],
            image, fold,
            data["ox"],
            data["oy"],
            data["ot_ms"],
            data["vrms"],
            45.0,  # max_dip_deg
            100.0,  # min_aperture
            max_aperture,
            0.1,  # taper_fraction
            apply_spreading,
            apply_obliquity,
            interp_method,
        )
        elapsed = time.perf_counter() - start
        total_time += elapsed

    avg_time = total_time / n_runs

    return KernelTimingResult(
        name=name,
        config_desc=config_desc,
        time_s=avg_time,
        n_traces=n_traces,
        n_pillars=tile_size * tile_size,
        n_samples=n_samples,
        contributions=contributions,
    )


def run_profiling(
    n_traces: int = 50_000,
    tile_size: int = 32,
    n_samples: int = 500,
    aperture_m: float = 2500.0,
) -> list[KernelTimingResult]:
    """Run comprehensive kernel profiling."""

    print(f"Configuration:")
    print(f"  Traces:      {n_traces:,}")
    print(f"  Tile size:   {tile_size}×{tile_size} = {tile_size**2:,} pillars")
    print(f"  Samples:     {n_samples}")
    print(f"  Aperture:    {aperture_m:.0f} m")
    print()

    data = generate_test_data(n_traces, tile_size, n_samples, aperture_m)

    warmup_kernel(data)

    results = []

    # 1. Nearest neighbor (fastest baseline)
    print("Test 1/12: Nearest neighbor (fastest baseline)...")
    results.append(run_kernel_timing(
        data,
        "Nearest neighbor",
        "interp=nearest, weights=off",
        interp_method=0,
        apply_spreading=False,
        apply_obliquity=False,
        max_aperture=aperture_m,
    ))

    # 2. Linear interpolation (no weights)
    print("Test 2/12: Linear interpolation (no weights)...")
    results.append(run_kernel_timing(
        data,
        "Linear (no weights)",
        "interp=linear, weights=off",
        interp_method=1,
        apply_spreading=False,
        apply_obliquity=False,
        max_aperture=aperture_m,
    ))

    # 3. Linear interpolation + weights
    print("Test 3/12: Linear + all weights...")
    results.append(run_kernel_timing(
        data,
        "Linear + weights",
        "interp=linear, weights=on",
        interp_method=1,
        apply_spreading=True,
        apply_obliquity=True,
        max_aperture=aperture_m,
    ))

    # 4. Cubic interpolation
    print("Test 4/12: Cubic interpolation + weights...")
    results.append(run_kernel_timing(
        data,
        "Cubic + weights",
        "interp=cubic, weights=on",
        interp_method=2,
        apply_spreading=True,
        apply_obliquity=True,
        max_aperture=aperture_m,
    ))

    # 5. Sinc4 interpolation
    print("Test 5/12: Sinc4 interpolation + weights...")
    results.append(run_kernel_timing(
        data,
        "Sinc4 + weights",
        "interp=sinc4, weights=on",
        interp_method=3,
        apply_spreading=True,
        apply_obliquity=True,
        max_aperture=aperture_m,
    ))

    # 6. Sinc8 interpolation
    print("Test 6/12: Sinc8 interpolation + weights...")
    results.append(run_kernel_timing(
        data,
        "Sinc8 + weights",
        "interp=sinc8, weights=on",
        interp_method=4,
        apply_spreading=True,
        apply_obliquity=True,
        max_aperture=aperture_m,
    ))

    # 7. Sinc16 interpolation
    print("Test 7/12: Sinc16 interpolation + weights...")
    results.append(run_kernel_timing(
        data,
        "Sinc16 + weights",
        "interp=sinc16, weights=on",
        interp_method=5,
        apply_spreading=True,
        apply_obliquity=True,
        max_aperture=aperture_m,
    ))

    # 8. Lanczos3 interpolation
    print("Test 8/12: Lanczos3 interpolation + weights...")
    results.append(run_kernel_timing(
        data,
        "Lanczos3 + weights",
        "interp=lanczos3, weights=on",
        interp_method=6,
        apply_spreading=True,
        apply_obliquity=True,
        max_aperture=aperture_m,
    ))

    # 9. Lanczos5 interpolation
    print("Test 9/12: Lanczos5 interpolation + weights...")
    results.append(run_kernel_timing(
        data,
        "Lanczos5 + weights",
        "interp=lanczos5, weights=on",
        interp_method=7,
        apply_spreading=True,
        apply_obliquity=True,
        max_aperture=aperture_m,
    ))

    # 10. Impact of aperture - half
    small_aperture = aperture_m * 0.5
    print(f"Test 10/12: Half aperture ({small_aperture:.0f}m)...")
    results.append(run_kernel_timing(
        data,
        f"Half aperture ({small_aperture:.0f}m)",
        f"interp=linear, aperture={small_aperture:.0f}m",
        interp_method=1,
        apply_spreading=True,
        apply_obliquity=True,
        max_aperture=small_aperture,
    ))

    # 11. Impact of aperture - quarter
    quarter_aperture = aperture_m * 0.25
    print(f"Test 11/12: Quarter aperture ({quarter_aperture:.0f}m)...")
    results.append(run_kernel_timing(
        data,
        f"Quarter aperture ({quarter_aperture:.0f}m)",
        f"interp=linear, aperture={quarter_aperture:.0f}m",
        interp_method=1,
        apply_spreading=True,
        apply_obliquity=True,
        max_aperture=quarter_aperture,
    ))

    # 12. Impact of spreading vs obliquity
    print("Test 12/12: Linear + spreading only...")
    results.append(run_kernel_timing(
        data,
        "Linear + spreading only",
        "interp=linear, spreading=on, obliquity=off",
        interp_method=1,
        apply_spreading=True,
        apply_obliquity=False,
        max_aperture=aperture_m,
    ))

    return results


def print_results(results: list[KernelTimingResult]):
    """Print results as formatted table."""

    print("\n" + "=" * 100)
    print("KERNEL TIMING RESULTS")
    print("=" * 100)

    baseline = results[0].time_s

    print(f"\n{'Configuration':<40} {'Time (s)':<12} {'Relative':<12} {'Traces/s':<15}")
    print("-" * 100)

    for r in results:
        relative = r.time_s / baseline if baseline > 0 else 0
        print(f"{r.name:<40} {r.time_s:<12.3f} {relative:<12.2f}x {r.traces_per_s:<15,.0f}")

    # Calculate component contributions
    print("\n" + "=" * 100)
    print("COMPONENT TIME BREAKDOWN (estimated from differential timing)")
    print("=" * 100)

    # Find times
    full_weights = next((r for r in results if "linear + weights" in r.name.lower()), None)
    linear_only = next((r for r in results if r.name == "Linear interp only"), None)
    spreading_only = next((r for r in results if r.name == "Linear + spreading"), None)
    obliquity_only = next((r for r in results if r.name == "Linear + obliquity"), None)
    nearest = next((r for r in results if "Nearest" in r.name), None)
    sinc8 = next((r for r in results if "Sinc8" in r.name), None)
    cubic = next((r for r in results if "Cubic" in r.name), None)

    breakdown = {}

    if linear_only and nearest:
        breakdown["Linear interpolation overhead"] = linear_only.time_s - nearest.time_s

    if full_weights and linear_only:
        breakdown["All weights overhead"] = full_weights.time_s - linear_only.time_s

    if spreading_only and linear_only:
        breakdown["Spreading weight"] = spreading_only.time_s - linear_only.time_s

    if obliquity_only and linear_only:
        breakdown["Obliquity weight"] = obliquity_only.time_s - linear_only.time_s

    if sinc8 and full_weights:
        breakdown["Sinc8 vs Linear (extra)"] = sinc8.time_s - full_weights.time_s

    if cubic and full_weights:
        breakdown["Cubic vs Linear (extra)"] = cubic.time_s - full_weights.time_s

    if nearest:
        breakdown["Core computation (DSR + accumulate)"] = nearest.time_s

    print(f"\n{'Component':<45} {'Time (s)':<12} {'% of Total':<12}")
    print("-" * 70)

    total = sum(max(0, v) for v in breakdown.values())

    for name, time_s in sorted(breakdown.items(), key=lambda x: -x[1]):
        pct = (time_s / total * 100) if total > 0 else 0
        print(f"{name:<45} {time_s:<12.3f} {pct:<12.1f}%")

    # Interpolation comparison table
    print("\n" + "=" * 100)
    print("INTERPOLATION METHOD COMPARISON")
    print("=" * 100)

    interp_results = [r for r in results if any(x in r.name.lower() for x in ["nearest", "linear", "cubic", "sinc"])]

    if interp_results:
        fastest = min(r.time_s for r in interp_results)
        print(f"\n{'Method':<25} {'Time (s)':<12} {'Slowdown':<12} {'Quality':<15}")
        print("-" * 70)

        quality_notes = {
            "nearest": "Low (staircasing)",
            "linear": "Good (standard)",
            "cubic": "Better (smooth)",
            "sinc8": "Best (accurate)",
        }

        for r in sorted(interp_results, key=lambda x: x.time_s):
            slowdown = r.time_s / fastest if fastest > 0 else 0
            method = r.name.split()[0].lower()
            quality = quality_notes.get(method, "")
            print(f"{r.name:<25} {r.time_s:<12.3f} {slowdown:<12.1f}x {quality:<15}")

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    if sinc8 and full_weights:
        sinc_penalty = sinc8.time_s / full_weights.time_s
        print(f"\n1. INTERPOLATION CHOICE:")
        print(f"   Sinc8 is {sinc_penalty:.1f}x slower than linear")
        if sinc_penalty > 2:
            print(f"   → Consider using linear interpolation for {(sinc_penalty-1)*100:.0f}% speedup")
            print(f"   → Linear is suitable for most seismic processing")

    small_aperture = next((r for r in results if "Reduced aperture" in r.name), None)
    if small_aperture and full_weights:
        aperture_speedup = full_weights.time_s / small_aperture.time_s
        print(f"\n2. APERTURE SIZE:")
        print(f"   Halving aperture gives {aperture_speedup:.1f}x speedup")
        print(f"   → Current aperture may be larger than needed")
        print(f"   → Consider 2-3km aperture for typical shallow targets")

    if full_weights and linear_only:
        weight_overhead = (full_weights.time_s - linear_only.time_s) / full_weights.time_s * 100
        print(f"\n3. WEIGHT COMPUTATIONS:")
        print(f"   Spreading + obliquity adds {weight_overhead:.1f}% overhead")
        print(f"   → Keep enabled for proper amplitude preservation")


def main():
    parser = argparse.ArgumentParser(description="Profile PSTM kernel directly")
    parser.add_argument("--traces", type=int, default=50_000, help="Number of traces")
    parser.add_argument("--tile-size", type=int, default=32, help="Tile size (NxN)")
    parser.add_argument("--samples", type=int, default=500, help="Time samples per trace")
    parser.add_argument("--aperture", type=float, default=2500.0, help="Aperture radius in meters")

    args = parser.parse_args()

    print("=" * 100)
    print("PSTM KERNEL DIRECT PROFILING")
    print("=" * 100)
    print()

    results = run_profiling(
        n_traces=args.traces,
        tile_size=args.tile_size,
        n_samples=args.samples,
        aperture_m=args.aperture,
    )

    print_results(results)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
