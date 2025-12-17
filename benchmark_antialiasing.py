#!/usr/bin/env python3
"""
Benchmark: PSTM with different anti-aliasing settings.

Tests configurations:
1. No anti-aliasing (baseline)
2. Triangle filter anti-aliasing
3. Triangle + geometrical spreading
4. Triangle + both amplitude corrections

Creates timing comparison table and crossline slice visualizations.
"""

import time
import numpy as np
from pathlib import Path
import shutil

# Generate synthetic data
from pstm.synthetic import (
    create_simple_synthetic,
    export_to_zarr_parquet,
)

# Migration imports
from pstm.config.models import (
    MigrationConfig, InputConfig, OutputConfig, OutputGridConfig,
    VelocityConfig, VelocitySource, AlgorithmConfig,
    ApertureConfig, AmplitudeConfig, AntiAliasingConfig,
    AntiAliasingMethod, InterpolationMethod,
    ExecutionConfig, ResourceConfig, ComputeBackend,
    TilingConfig, CheckpointConfig, ColumnMapping,
    OutputProductsConfig, OutputFormat,
)
from pstm.pipeline.executor import MigrationExecutor
import zarr


def generate_synthetic_data(output_dir: Path) -> dict:
    """Generate synthetic data with single diffractor."""
    print("=" * 60)
    print("Generating Synthetic Data")
    print("=" * 60)

    # Diffractor position (centered in smaller survey)
    diffractor_x = 1000.0
    diffractor_y = 1000.0
    diffractor_z = 500.0
    velocity = 2500.0

    # Smaller dataset for faster benchmarking
    result = create_simple_synthetic(
        diffractor_x=diffractor_x,
        diffractor_y=diffractor_y,
        diffractor_z=diffractor_z,
        survey_extent=2000.0,
        grid_spacing=50.0,
        offsets=[200, 400],
        azimuths=[0, 90, 180, 270],
        velocity=velocity,
        n_samples=1001,
        dt_ms=2.0,
        wavelet_freq=30.0,
        noise_level=0.05,
    )

    # Export to Zarr/Parquet
    traces_path, headers_path = export_to_zarr_parquet(
        result,
        output_dir,
        traces_name="traces.zarr",
        headers_name="headers.parquet",
    )

    print(f"Generated {result.n_traces:,} traces")
    print(f"Trace shape: {result.traces.shape}")
    print(f"Diffractor at: ({diffractor_x}, {diffractor_y}, {diffractor_z})")
    print(f"Velocity: {velocity} m/s")
    print(f"Expected t0: {2 * diffractor_z / velocity * 1000:.1f} ms")

    return {
        "traces_path": traces_path,
        "headers_path": headers_path,
        "diffractor_x": diffractor_x,
        "diffractor_y": diffractor_y,
        "diffractor_z": diffractor_z,
        "velocity": velocity,
        "n_traces": result.n_traces,
        "survey_extent": 2000.0,
    }


def create_migration_config(
    input_info: dict,
    output_dir: Path,
    anti_aliasing_enabled: bool,
    anti_aliasing_method: AntiAliasingMethod = AntiAliasingMethod.TRIANGLE,
    geometrical_spreading: bool = False,
    obliquity_factor: bool = False,
) -> MigrationConfig:
    """Create migration config with specified anti-aliasing settings."""

    # Output grid
    grid = OutputGridConfig(
        x_min=0.0,
        x_max=input_info["survey_extent"],
        y_min=0.0,
        y_max=input_info["survey_extent"],
        dx=50.0,
        dy=50.0,
        t_min_ms=0.0,
        t_max_ms=2000.0,
        dt_ms=2.0,
    )

    return MigrationConfig(
        name="benchmark_run",
        input=InputConfig(
            traces_path=input_info["traces_path"],
            headers_path=input_info["headers_path"],
            columns=ColumnMapping(
                source_x="SOU_X",
                source_y="SOU_Y",
                receiver_x="REC_X",
                receiver_y="REC_Y",
                cdp_x="CDP_X",
                cdp_y="CDP_Y",
                offset="OFFSET",
            ),
        ),
        output=OutputConfig(
            output_dir=output_dir,
            grid=grid,
            products=OutputProductsConfig(
                stacked_image=True,
                fold_volume=True,
            ),
            format=OutputFormat.ZARR,
        ),
        velocity=VelocityConfig(
            source=VelocitySource.CONSTANT,
            constant_velocity=input_info["velocity"],
        ),
        algorithm=AlgorithmConfig(
            interpolation=InterpolationMethod.LINEAR,
            aperture=ApertureConfig(
                max_aperture_m=1500.0,
                min_aperture_m=100.0,
                max_dip_degrees=60.0,
                taper_fraction=0.1,
            ),
            anti_aliasing=AntiAliasingConfig(
                enabled=anti_aliasing_enabled,
                method=anti_aliasing_method,
                num_filters=32,
                min_frequency_hz=5.0,
                max_frequency_hz=80.0,
            ),
            amplitude=AmplitudeConfig(
                geometrical_spreading=geometrical_spreading,
                obliquity_factor=obliquity_factor,
            ),
        ),
        execution=ExecutionConfig(
            resources=ResourceConfig(
                backend=ComputeBackend.NUMBA_CPU,
                max_memory_gb=8.0,
                num_workers=8,
            ),
            tiling=TilingConfig(
                auto_tile_size=True,
            ),
            checkpoint=CheckpointConfig(
                enabled=False,
            ),
        ),
    )


def run_migration(config: MigrationConfig, run_name: str) -> tuple[float, Path]:
    """Run migration and return elapsed time and output path."""
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"  Anti-aliasing: {config.algorithm.anti_aliasing.enabled}")
    if config.algorithm.anti_aliasing.enabled:
        print(f"  AA Method: {config.algorithm.anti_aliasing.method.value}")
    print(f"  Geometrical Spreading: {config.algorithm.amplitude.geometrical_spreading}")
    print(f"  Obliquity Factor: {config.algorithm.amplitude.obliquity_factor}")
    print(f"{'='*60}")

    start_time = time.time()

    executor = MigrationExecutor(config)
    success = executor.run()

    elapsed = time.time() - start_time

    if success:
        print(f"Completed in {elapsed:.2f} seconds")
    else:
        print(f"FAILED after {elapsed:.2f} seconds")

    return elapsed, config.output.output_dir


def extract_crossline_slice(output_dir: Path, crossline_idx: int) -> np.ndarray:
    """Extract crossline slice from migration output."""
    stack_path = output_dir / "migrated_stack.zarr"
    z = zarr.open(str(stack_path), mode='r')

    if isinstance(z, zarr.Array):
        data = np.array(z)
    else:
        key = list(z.keys())[0]
        data = np.array(z[key])

    return data[:, crossline_idx, :]


def create_comparison_figure(
    slices: dict,
    input_info: dict,
    output_path: Path,
    titles: list,
):
    """Create comparison figure of all crossline slices."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Calculate diffractor position in grid indices
    dx = 50.0
    diffractor_ix = int(input_info["diffractor_x"] / dx)
    expected_t0 = 2 * input_info["diffractor_z"] / input_info["velocity"] * 1000

    for ax, (key, data), title in zip(axes, slices.items(), titles):
        nx, nt = data.shape

        valid = data[~np.isnan(data)]
        if len(valid) > 0:
            clip = np.percentile(np.abs(valid), 99)
        else:
            clip = 1.0
        clip = max(clip, 1e-10)

        im = ax.imshow(
            data.T,
            aspect='auto',
            extent=[0, nx, 2000, 0],
            cmap='seismic',
            vmin=-clip,
            vmax=clip,
        )

        ax.axvline(x=diffractor_ix, color='green', linestyle='--', alpha=0.7, label='Diffractor X')
        ax.axhline(y=expected_t0, color='yellow', linestyle='--', alpha=0.7, label=f'Expected t0={expected_t0:.0f}ms')

        ax.set_xlabel('Inline Index')
        ax.set_ylabel('Time (ms)')
        ax.set_title(title)
        ax.legend(loc='lower right', fontsize=8)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(
        f'Anti-Aliasing Comparison - Crossline Slice (Y={input_info["diffractor_y"]:.0f}m)\n'
        f'Diffractor: ({input_info["diffractor_x"]:.0f}, {input_info["diffractor_y"]:.0f}, {input_info["diffractor_z"]:.0f})m, '
        f'V={input_info["velocity"]:.0f} m/s',
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison figure to: {output_path}")
    plt.close()


def main():
    """Run benchmark."""
    print("\n" + "=" * 60)
    print("PSTM ANTI-ALIASING BENCHMARK")
    print("=" * 60)

    # Setup directories
    base_dir = Path("./benchmark_antialiasing_output")
    input_dir = base_dir / "input_data"

    # Clean previous runs
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)

    # Generate synthetic data
    input_info = generate_synthetic_data(input_dir)

    # Define test configurations: (name, aa_enabled, aa_method, spreading, obliquity)
    configs = [
        ("no_antialiasing", False, AntiAliasingMethod.NONE, False, False),
        ("triangle_aa", True, AntiAliasingMethod.TRIANGLE, False, False),
        ("triangle_aa_spreading", True, AntiAliasingMethod.TRIANGLE, True, False),
        ("triangle_aa_both_amp", True, AntiAliasingMethod.TRIANGLE, True, True),
    ]

    titles = [
        "No Anti-Aliasing (Baseline)",
        "Triangle Anti-Aliasing",
        "Triangle AA + Spreading",
        "Triangle AA + Both Corrections",
    ]

    # Run migrations and collect results
    results = {}
    output_dirs = {}

    for name, aa_enabled, aa_method, spreading, obliquity in configs:
        output_dir = base_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)

        config = create_migration_config(
            input_info, output_dir, aa_enabled, aa_method, spreading, obliquity
        )

        elapsed, out_path = run_migration(config, name)
        results[name] = elapsed
        output_dirs[name] = out_path

    # Print timing table
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS - TIMING COMPARISON")
    print("=" * 60)
    print(f"{'Configuration':<30} {'Time (s)':<12} {'Relative':<10}")
    print("-" * 55)

    baseline = results["no_antialiasing"]
    for name, elapsed in results.items():
        relative = elapsed / baseline if baseline > 0 else 0
        print(f"{name:<30} {elapsed:<12.2f} {relative:<10.2f}x")

    print("-" * 55)
    print(f"{'Total traces:':<30} {input_info['n_traces']:,}")
    print(f"{'Backend:':<30} Numba CPU")

    # Extract crossline slices through diffractor
    print("\n" + "=" * 60)
    print("Extracting Crossline Slices")
    print("=" * 60)

    dy = 50.0
    crossline_idx = int(input_info["diffractor_y"] / dy)
    print(f"Crossline index: {crossline_idx} (Y={input_info['diffractor_y']:.0f}m)")

    slices = {}
    for name in results.keys():
        try:
            slice_data = extract_crossline_slice(output_dirs[name], crossline_idx)
            slices[name] = slice_data
            print(f"  {name}: shape={slice_data.shape}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Create comparison figure
    if slices:
        create_comparison_figure(
            slices,
            input_info,
            base_dir / "antialiasing_comparison.png",
            titles,
        )

    # Save results to text file
    results_file = base_dir / "benchmark_results.txt"
    with open(results_file, "w") as f:
        f.write("PSTM ANTI-ALIASING BENCHMARK\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Input Data:\n")
        f.write(f"  Traces: {input_info['n_traces']:,}\n")
        f.write(f"  Diffractor: ({input_info['diffractor_x']}, {input_info['diffractor_y']}, {input_info['diffractor_z']})\n")
        f.write(f"  Velocity: {input_info['velocity']} m/s\n\n")
        f.write("Timing Results:\n")
        f.write(f"{'Configuration':<30} {'Time (s)':<12} {'Relative':<10}\n")
        f.write("-" * 55 + "\n")
        for name, elapsed in results.items():
            relative = elapsed / baseline if baseline > 0 else 0
            f.write(f"{name:<30} {elapsed:<12.2f} {relative:<10.2f}x\n")

    print(f"\nResults saved to: {results_file}")
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
