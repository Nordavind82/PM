#!/usr/bin/env python3
"""
Kernel Comparison: Numba CPU vs MLX GPU vs Metal C++ GPU

Compares performance AND visual output of all kernels.
Generates comparison images for different survey sizes.

Usage:
    python scripts/compare_kernels.py [--output-dir DIR] [--sizes small,medium,large]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])


@dataclass
class KernelResult:
    """Results from a kernel benchmark."""
    kernel_name: str
    total_time_s: float
    n_traces: int
    n_pillars: int
    n_samples: int
    traces_per_s: float
    image: np.ndarray | None = None
    fold: np.ndarray | None = None


@dataclass
class SurveySize:
    """Survey size configuration."""
    name: str
    n_traces: int
    tile_size: int
    n_samples: int
    aperture_m: float


# Predefined survey sizes for testing
SURVEY_SIZES = {
    "tiny": SurveySize("tiny", 1000, 16, 250, 1000.0),
    "small": SurveySize("small", 5000, 32, 500, 2000.0),
    "medium": SurveySize("medium", 20000, 48, 750, 3000.0),
    "large": SurveySize("large", 50000, 64, 1000, 4000.0),
}


def generate_diffractor_data(
    n_traces: int,
    tile_size: int,
    n_samples: int,
    aperture_m: float,
    n_diffractors: int = 3,
) -> dict:
    """
    Generate synthetic data with point diffractors.

    Point diffractors create recognizable hyperbolic patterns in migration,
    making it easy to verify kernel correctness visually.
    """
    np.random.seed(42)

    tile_extent = tile_size * 25  # 25m spacing
    dt_ms = 4.0
    t_max_ms = n_samples * dt_ms

    # Trace geometry - regular grid of midpoints covering the tile
    n_x = int(np.sqrt(n_traces))
    n_y = n_traces // n_x
    actual_n_traces = n_x * n_y

    # Create midpoint grid with some random perturbation
    mx_grid = np.linspace(0, tile_extent, n_x)
    my_grid = np.linspace(0, tile_extent, n_y)
    mx, my = np.meshgrid(mx_grid, my_grid)
    midpoint_x = mx.ravel() + np.random.randn(actual_n_traces) * 10
    midpoint_y = my.ravel() + np.random.randn(actual_n_traces) * 10

    # Variable offsets (source-receiver distance)
    offsets = np.random.uniform(200, 1500, actual_n_traces)
    angles = np.random.rand(actual_n_traces) * 2 * np.pi

    source_x = midpoint_x - 0.5 * offsets * np.cos(angles)
    source_y = midpoint_y - 0.5 * offsets * np.sin(angles)
    receiver_x = midpoint_x + 0.5 * offsets * np.cos(angles)
    receiver_y = midpoint_y + 0.5 * offsets * np.sin(angles)

    # Output grid
    x_coords = np.linspace(0, tile_extent, tile_size).astype(np.float64)
    y_coords = np.linspace(0, tile_extent, tile_size).astype(np.float64)
    t_coords_ms = np.linspace(0, t_max_ms, n_samples).astype(np.float64)

    # Velocity model (increasing with depth)
    vrms = np.linspace(2000, 4000, n_samples).astype(np.float64)

    # Create amplitudes with point diffractors
    amplitudes = np.zeros((actual_n_traces, n_samples), dtype=np.float32)

    # Place diffractors at different locations and depths
    diffractor_positions = [
        (tile_extent * 0.3, tile_extent * 0.3, 800),   # x, y, t0_ms
        (tile_extent * 0.5, tile_extent * 0.5, 1200),  # center
        (tile_extent * 0.7, tile_extent * 0.7, 1600),
    ][:n_diffractors]

    for diff_x, diff_y, diff_t0_ms in diffractor_positions:
        diff_t0_s = diff_t0_ms / 1000.0

        for i in range(actual_n_traces):
            sx, sy = source_x[i], source_y[i]
            rx, ry = receiver_x[i], receiver_y[i]

            # DSR travel time from diffractor
            v_idx = int(diff_t0_ms / dt_ms)
            v_idx = min(v_idx, n_samples - 1)
            v = vrms[v_idx]

            # Distance from diffractor to source and receiver
            ds = np.sqrt((diff_x - sx)**2 + (diff_y - sy)**2)
            dr = np.sqrt((diff_x - rx)**2 + (diff_y - ry)**2)

            # DSR travel time: t = sqrt(t0^2/4 + ds^2/v^2) + sqrt(t0^2/4 + dr^2/v^2)
            t0_half_sq = (diff_t0_s / 2) ** 2
            inv_v_sq = 1.0 / (v * v)
            t_travel = np.sqrt(t0_half_sq + ds**2 * inv_v_sq) + np.sqrt(t0_half_sq + dr**2 * inv_v_sq)

            # Convert to sample index
            sample_idx = t_travel * 1000.0 / dt_ms
            idx = int(sample_idx)

            if 0 <= idx < n_samples - 1:
                # Ricker wavelet centered at this time
                f = 30.0  # dominant frequency Hz
                for j in range(-20, 21):
                    t_idx = idx + j
                    if 0 <= t_idx < n_samples:
                        t = j * dt_ms / 1000.0
                        # Ricker wavelet
                        a = (np.pi * f * t) ** 2
                        wavelet = (1 - 2 * a) * np.exp(-a)
                        amplitudes[i, t_idx] += wavelet * 1000.0

    # Add some noise
    amplitudes += np.random.randn(actual_n_traces, n_samples).astype(np.float32) * 10

    return {
        "amplitudes": amplitudes,
        "source_x": source_x.astype(np.float64),
        "source_y": source_y.astype(np.float64),
        "receiver_x": receiver_x.astype(np.float64),
        "receiver_y": receiver_y.astype(np.float64),
        "midpoint_x": midpoint_x.astype(np.float64),
        "midpoint_y": midpoint_y.astype(np.float64),
        "x_coords": x_coords,
        "y_coords": y_coords,
        "t_coords_ms": t_coords_ms,
        "vrms": vrms,
        "dt_ms": dt_ms,
        "t_start_ms": 0.0,
        "n_traces": actual_n_traces,
        "tile_size": tile_size,
        "n_samples": n_samples,
        "diffractor_positions": diffractor_positions,
    }


def run_numba_cpu(data: dict, aperture_m: float) -> KernelResult:
    """Run Numba CPU kernel and return result with image."""
    from pstm.kernels.numba_cpu import _migrate_tile_kernel

    tile_size = data["tile_size"]
    n_samples = data["n_samples"]
    n_traces = data["n_traces"]

    # Warmup
    image_warmup = np.zeros((4, 4, 50), dtype=np.float64)
    fold_warmup = np.zeros((4, 4), dtype=np.int32)
    _migrate_tile_kernel(
        data["amplitudes"][:10],
        data["source_x"][:10], data["source_y"][:10],
        data["receiver_x"][:10], data["receiver_y"][:10],
        data["midpoint_x"][:10], data["midpoint_y"][:10],
        np.ones(10, dtype=np.float64),
        4.0, 0.0, image_warmup, fold_warmup,
        data["x_coords"][:4], data["y_coords"][:4],
        data["t_coords_ms"][:50], data["vrms"][:50],
        45.0, 100.0, aperture_m, 0.1, True, True, 1,
    )

    # Actual run
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
        45.0, 100.0, aperture_m, 0.1,
        True, True, 1,
    )
    elapsed = time.perf_counter() - start

    return KernelResult(
        kernel_name="Numba CPU",
        total_time_s=elapsed,
        n_traces=n_traces,
        n_pillars=tile_size * tile_size,
        n_samples=n_samples,
        traces_per_s=n_traces / elapsed,
        image=image,
        fold=fold,
    )


def run_mlx_gpu(data: dict, aperture_m: float) -> KernelResult | None:
    """Run MLX GPU kernel and return result with image."""
    try:
        import mlx.core as mx
        from pstm.kernels.mlx_metal import MLXKernel, check_mlx_available

        if not check_mlx_available():
            return None
    except ImportError:
        return None

    from pstm.kernels.base import TraceBlock, OutputTile, VelocitySlice, KernelConfig

    tile_size = data["tile_size"]
    n_samples = data["n_samples"]
    n_traces = data["n_traces"]

    kernel = MLXKernel(chunk_size=1000)

    offset = np.sqrt(
        (data["receiver_x"] - data["source_x"]) ** 2 +
        (data["receiver_y"] - data["source_y"]) ** 2
    )

    traces = TraceBlock(
        amplitudes=data["amplitudes"],
        source_x=data["source_x"],
        source_y=data["source_y"],
        receiver_x=data["receiver_x"],
        receiver_y=data["receiver_y"],
        offset=offset,
        midpoint_x=data["midpoint_x"],
        midpoint_y=data["midpoint_y"],
        sample_rate_ms=data["dt_ms"],
        start_time_ms=data["t_start_ms"],
    )

    output = OutputTile(
        image=np.zeros((tile_size, tile_size, n_samples), dtype=np.float64),
        fold=np.zeros((tile_size, tile_size), dtype=np.int32),
        x_axis=data["x_coords"],
        y_axis=data["y_coords"],
        t_axis_ms=data["t_coords_ms"],
    )

    velocity = VelocitySlice(
        vrms=data["vrms"],
        t_axis_ms=data["t_coords_ms"],
    )

    config = KernelConfig(
        max_aperture_m=aperture_m,
        min_aperture_m=100.0,
        max_dip_degrees=45.0,
        taper_fraction=0.1,
        apply_spreading=True,
        apply_obliquity=True,
        interpolation_method="linear",
    )

    kernel.initialize(config)

    start = time.perf_counter()
    kernel.migrate_tile(traces, output, velocity, config)
    elapsed = time.perf_counter() - start

    return KernelResult(
        kernel_name="MLX Metal",
        total_time_s=elapsed,
        n_traces=n_traces,
        n_pillars=tile_size * tile_size,
        n_samples=n_samples,
        traces_per_s=n_traces / elapsed,
        image=output.image.copy(),
        fold=output.fold.copy(),
    )


def run_metal_cpp(data: dict, aperture_m: float) -> KernelResult | None:
    """Run Metal C++ GPU kernel and return result with image."""
    try:
        from pstm.kernels.metal_cpp import MetalCppKernel, is_metal_cpp_available

        if not is_metal_cpp_available():
            return None
    except ImportError:
        return None

    from pstm.kernels.base import TraceBlock, OutputTile, VelocitySlice, KernelConfig

    tile_size = data["tile_size"]
    n_samples = data["n_samples"]
    n_traces = data["n_traces"]

    kernel = MetalCppKernel()

    offset = np.sqrt(
        (data["receiver_x"] - data["source_x"]) ** 2 +
        (data["receiver_y"] - data["source_y"]) ** 2
    )

    traces = TraceBlock(
        amplitudes=data["amplitudes"],
        source_x=data["source_x"],
        source_y=data["source_y"],
        receiver_x=data["receiver_x"],
        receiver_y=data["receiver_y"],
        offset=offset,
        midpoint_x=data["midpoint_x"],
        midpoint_y=data["midpoint_y"],
        sample_rate_ms=data["dt_ms"],
        start_time_ms=data["t_start_ms"],
    )

    output = OutputTile(
        image=np.zeros((tile_size, tile_size, n_samples), dtype=np.float64),
        fold=np.zeros((tile_size, tile_size), dtype=np.int32),
        x_axis=data["x_coords"],
        y_axis=data["y_coords"],
        t_axis_ms=data["t_coords_ms"],
    )

    velocity = VelocitySlice(
        vrms=data["vrms"],
        t_axis_ms=data["t_coords_ms"],
    )

    config = KernelConfig(
        max_aperture_m=aperture_m,
        min_aperture_m=100.0,
        max_dip_degrees=45.0,
        taper_fraction=0.1,
        apply_spreading=True,
        apply_obliquity=True,
        interpolation_method="linear",
    )

    kernel.initialize(config)

    start = time.perf_counter()
    kernel.migrate_tile(traces, output, velocity, config)
    elapsed = time.perf_counter() - start

    kernel.cleanup()

    return KernelResult(
        kernel_name="Metal C++",
        total_time_s=elapsed,
        n_traces=n_traces,
        n_pillars=tile_size * tile_size,
        n_samples=n_samples,
        traces_per_s=n_traces / elapsed,
        image=output.image.copy(),
        fold=output.fold.copy(),
    )


def create_comparison_figure(
    results: list[KernelResult],
    survey_name: str,
    data: dict,
    output_path: Path,
):
    """Create and save comparison figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n_kernels = len(results)
    if n_kernels == 0:
        print(f"  No results to plot for {survey_name}")
        return

    # Figure with multiple rows
    # Row 1: Vertical slice at center x (y vs t)
    # Row 2: Horizontal slice at center time (x vs y)
    # Row 3: Fold map
    fig, axes = plt.subplots(3, n_kernels + 1, figsize=(5 * (n_kernels + 1), 12))
    fig.suptitle(f"Kernel Comparison - {survey_name}\n"
                 f"({data['n_traces']:,} traces, {data['tile_size']}x{data['tile_size']} tile, "
                 f"{data['n_samples']} samples)", fontsize=14)

    # Get diffractor positions for reference
    diff_positions = data.get("diffractor_positions", [])
    tile_extent = data["tile_size"] * 25

    # Determine common color scale
    all_images = [r.image for r in results if r.image is not None]
    if not all_images:
        plt.close(fig)
        return

    vmax_list = [np.percentile(np.abs(img), 99) for img in all_images]
    vmax = max(vmax_list) if vmax_list else 1.0
    vmin = -vmax

    # First column: Input trace gather (center CDP)
    center_idx = data["n_traces"] // 2
    gather_range = min(100, data["n_traces"])
    start_idx = max(0, center_idx - gather_range // 2)
    end_idx = min(data["n_traces"], start_idx + gather_range)

    # Plot input gather
    ax = axes[0, 0]
    gather = data["amplitudes"][start_idx:end_idx, :].T
    im = ax.imshow(
        gather,
        aspect="auto",
        cmap="seismic",
        extent=[start_idx, end_idx, data["n_samples"] * data["dt_ms"], 0],
        vmin=-np.percentile(np.abs(gather), 98),
        vmax=np.percentile(np.abs(gather), 98),
    )
    ax.set_title("Input Traces")
    ax.set_xlabel("Trace #")
    ax.set_ylabel("Time (ms)")

    # Empty plots for input column
    axes[1, 0].text(0.5, 0.5, "Input\nGather", ha="center", va="center", fontsize=14)
    axes[1, 0].axis("off")
    axes[2, 0].text(0.5, 0.5, f"Diffractors:\n" +
                    "\n".join([f"({x:.0f}, {y:.0f}, {t:.0f}ms)" for x, y, t in diff_positions]),
                    ha="center", va="center", fontsize=10)
    axes[2, 0].axis("off")

    # Plot each kernel result
    for col, result in enumerate(results, 1):
        if result.image is None:
            for row in range(3):
                axes[row, col].text(0.5, 0.5, f"{result.kernel_name}\nN/A",
                                   ha="center", va="center")
                axes[row, col].axis("off")
            continue

        img = result.image
        fold = result.fold
        tile_size = img.shape[0]
        n_samples = img.shape[2]

        # Row 0: Vertical slice (inline at center)
        ax = axes[0, col]
        center_x = tile_size // 2
        slice_data = img[center_x, :, :].T  # (nt, ny)
        im = ax.imshow(
            slice_data,
            aspect="auto",
            cmap="seismic",
            extent=[0, tile_extent, data["n_samples"] * data["dt_ms"], 0],
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"{result.kernel_name}\n{result.total_time_s:.2f}s ({result.traces_per_s:,.0f} tr/s)")
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Time (ms)")
        # Mark diffractor positions
        for dx, dy, dt in diff_positions:
            if abs(dx - tile_extent/2) < tile_extent/4:  # If near this slice
                ax.axhline(dt, color='green', linestyle='--', alpha=0.5)

        # Row 1: Horizontal slice (time slice)
        ax = axes[1, col]
        center_t = n_samples // 3  # Show slice at first diffractor depth
        if diff_positions:
            center_t = int(diff_positions[0][2] / data["dt_ms"])
        center_t = min(center_t, n_samples - 1)

        slice_data = img[:, :, center_t]  # (nx, ny)
        im = ax.imshow(
            slice_data.T,
            aspect="equal",
            cmap="seismic",
            extent=[0, tile_extent, tile_extent, 0],
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"Time slice @ {center_t * data['dt_ms']:.0f} ms")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        # Mark diffractor positions
        for dx, dy, dt in diff_positions:
            ax.plot(dx, dy, 'g+', markersize=10, markeredgewidth=2)

        # Row 2: Fold map
        ax = axes[2, col]
        im = ax.imshow(
            fold.T,
            aspect="equal",
            cmap="viridis",
            extent=[0, tile_extent, tile_extent, 0],
        )
        ax.set_title(f"Fold (max={fold.max():,})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_difference_figure(
    results: list[KernelResult],
    survey_name: str,
    data: dict,
    output_path: Path,
):
    """Create figure showing differences between kernels."""
    import matplotlib.pyplot as plt

    # Need at least 2 results to compare
    valid_results = [r for r in results if r.image is not None]
    if len(valid_results) < 2:
        print(f"  Need at least 2 kernels to compare differences")
        return

    # Use first result as reference
    ref = valid_results[0]
    others = valid_results[1:]

    n_compare = len(others)
    fig, axes = plt.subplots(2, n_compare, figsize=(6 * n_compare, 10))
    if n_compare == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"Kernel Differences vs {ref.kernel_name} - {survey_name}", fontsize=14)

    tile_extent = data["tile_size"] * 25

    for col, other in enumerate(others):
        # Compute difference
        diff = other.image - ref.image

        # Relative difference (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.where(np.abs(ref.image) > 1e-10,
                               np.abs(diff) / np.abs(ref.image) * 100,
                               0)

        max_abs_diff = np.max(np.abs(diff))
        mean_abs_diff = np.mean(np.abs(diff))
        max_rel_diff = np.percentile(rel_diff, 99)

        # Vertical slice of difference
        ax = axes[0, col]
        center_x = diff.shape[0] // 2
        slice_diff = diff[center_x, :, :].T

        vmax = np.percentile(np.abs(slice_diff), 99)
        im = ax.imshow(
            slice_diff,
            aspect="auto",
            cmap="RdBu",
            extent=[0, tile_extent, data["n_samples"] * data["dt_ms"], 0],
            vmin=-vmax, vmax=vmax,
        )
        ax.set_title(f"{other.kernel_name} - {ref.kernel_name}\n"
                    f"Max diff: {max_abs_diff:.2e}, Mean: {mean_abs_diff:.2e}")
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Time (ms)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Time slice of relative difference
        ax = axes[1, col]
        center_t = diff.shape[2] // 3
        slice_rel = rel_diff[:, :, center_t].T

        im = ax.imshow(
            slice_rel,
            aspect="equal",
            cmap="hot",
            extent=[0, tile_extent, tile_extent, 0],
            vmin=0, vmax=min(max_rel_diff, 10),
        )
        ax.set_title(f"Relative diff (%) @ {center_t * data['dt_ms']:.0f} ms\n"
                    f"99th percentile: {max_rel_diff:.2f}%")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def run_comparison(survey: SurveySize, output_dir: Path, skip_mlx: bool = True) -> list[KernelResult]:
    """Run comparison for a single survey size."""
    print(f"\n{'='*70}")
    print(f"Survey: {survey.name}")
    print(f"  Traces: {survey.n_traces:,}")
    print(f"  Tile: {survey.tile_size}x{survey.tile_size}")
    print(f"  Samples: {survey.n_samples}")
    print(f"  Aperture: {survey.aperture_m:.0f} m")
    print(f"{'='*70}")

    # Generate test data
    print("\nGenerating diffractor data...")
    data = generate_diffractor_data(
        survey.n_traces,
        survey.tile_size,
        survey.n_samples,
        survey.aperture_m,
    )
    print(f"  Generated {data['n_traces']:,} traces")

    results = []

    # Run Numba CPU
    print("\nRunning Numba CPU kernel...")
    try:
        result = run_numba_cpu(data, survey.aperture_m)
        results.append(result)
        print(f"  Time: {result.total_time_s:.2f}s ({result.traces_per_s:,.0f} traces/s)")
    except Exception as e:
        print(f"  Error: {e}")

    # Run Metal C++ (fastest GPU)
    print("\nRunning Metal C++ kernel...")
    try:
        result = run_metal_cpp(data, survey.aperture_m)
        if result:
            results.append(result)
            print(f"  Time: {result.total_time_s:.2f}s ({result.traces_per_s:,.0f} traces/s)")
        else:
            print("  Not available")
    except Exception as e:
        print(f"  Error: {e}")

    # Run MLX (for comparison - slower)
    if skip_mlx:
        print("\nSkipping MLX Metal kernel (use --include-mlx to enable)")
    else:
        print("\nRunning MLX Metal kernel (warning: very slow)...")
        try:
            result = run_mlx_gpu(data, survey.aperture_m)
            if result:
                results.append(result)
                print(f"  Time: {result.total_time_s:.2f}s ({result.traces_per_s:,.0f} traces/s)")
            else:
                print("  Not available")
        except Exception as e:
            print(f"  Error: {e}")

    # Create comparison images
    print("\nGenerating comparison images...")

    comparison_path = output_dir / f"comparison_{survey.name}.png"
    create_comparison_figure(results, survey.name, data, comparison_path)

    diff_path = output_dir / f"differences_{survey.name}.png"
    create_difference_figure(results, survey.name, data, diff_path)

    return results


def print_summary(all_results: dict[str, list[KernelResult]]):
    """Print summary table of all results."""
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Collect all kernel names
    kernel_names = set()
    for results in all_results.values():
        for r in results:
            kernel_names.add(r.kernel_name)
    kernel_names = sorted(kernel_names)

    # Print header
    print(f"\n{'Survey':<12}", end="")
    for name in kernel_names:
        print(f"{name:<20}", end="")
    print("Speedup (Metal C++ vs Numba)")
    print("-" * 90)

    # Print results
    for survey_name, results in all_results.items():
        print(f"{survey_name:<12}", end="")

        result_dict = {r.kernel_name: r for r in results}
        numba_time = None
        metal_time = None

        for name in kernel_names:
            if name in result_dict:
                r = result_dict[name]
                print(f"{r.total_time_s:.2f}s ({r.traces_per_s/1000:.1f}k/s)".ljust(20), end="")
                if name == "Numba CPU":
                    numba_time = r.total_time_s
                elif name == "Metal C++":
                    metal_time = r.total_time_s
            else:
                print("N/A".ljust(20), end="")

        if numba_time and metal_time:
            speedup = numba_time / metal_time
            print(f"{speedup:.1f}x")
        else:
            print("")

    print("-" * 90)


def main():
    parser = argparse.ArgumentParser(description="Compare PSTM kernels with visual output")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="kernel_comparison_output",
        help="Output directory for images"
    )
    parser.add_argument(
        "--sizes", "-s",
        type=str,
        default="tiny,small",
        help="Comma-separated survey sizes to test (tiny,small,medium,large)"
    )
    parser.add_argument(
        "--skip-mlx",
        action="store_true",
        default=True,
        help="Skip MLX kernel (it's very slow)"
    )
    parser.add_argument(
        "--include-mlx",
        action="store_true",
        help="Include MLX kernel (warning: very slow)"
    )

    args = parser.parse_args()

    # Handle MLX flag
    skip_mlx = args.skip_mlx and not args.include_mlx

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")

    # Parse survey sizes
    size_names = [s.strip().lower() for s in args.sizes.split(",")]
    surveys = []
    for name in size_names:
        if name in SURVEY_SIZES:
            surveys.append(SURVEY_SIZES[name])
        else:
            print(f"Warning: Unknown survey size '{name}', skipping")

    if not surveys:
        print("No valid survey sizes specified")
        return 1

    print("\n" + "=" * 90)
    print("PSTM KERNEL COMPARISON: Numba CPU vs Metal C++ GPU vs MLX Metal")
    print("=" * 90)

    all_results = {}

    for survey in surveys:
        results = run_comparison(survey, output_dir, skip_mlx=skip_mlx)
        all_results[survey.name] = results

    print_summary(all_results)

    print(f"\n{'='*90}")
    print(f"Images saved to: {output_dir.absolute()}")
    print(f"{'='*90}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
