#!/usr/bin/env python3
"""
PSTM Scaling Test Script

Generates synthetic datasets of varying sizes and benchmarks the PSTM pipeline
performance for each size. Creates a comprehensive table of results.

Survey sizes tested:
- 1km x 1km
- 2km x 2km
- 5km x 5km
- 10km x 10km
- 20km x 20km
- 50km x 50km
- 100km x 100km

All use:
- Same grid spacing (dx=dy=25m)
- Same time parameters (0-3s at 2ms)
- Single diffractor in the center
- Two offset values (200m, 400m)
- Fixed aperture (2500m)
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ScalingResult:
    """Result from a single scaling test."""
    survey_size_km: float
    survey_extent_m: float
    n_traces: int
    n_midpoints: int
    data_size_mb: float

    # Timing breakdown (seconds)
    generation_time: float = 0.0
    export_time: float = 0.0
    header_load_time: float = 0.0
    trace_access_time: float = 0.0
    spatial_index_time: float = 0.0
    aperture_query_time: float = 0.0
    trace_load_time: float = 0.0
    kernel_init_time: float = 0.0
    single_tile_time: float = 0.0

    # Performance metrics
    traces_per_tile: int = 0
    traces_per_s: float = 0.0
    n_tiles_total: int = 0
    estimated_total_time_s: float = 0.0

    # Additional stats
    stats: dict[str, Any] = field(default_factory=dict)


class ScalingTest:
    """Run scaling tests across different survey sizes."""

    # Survey sizes to test (in kilometers)
    SURVEY_SIZES_KM = [1, 2, 5, 10, 20, 50, 100]

    def __init__(
        self,
        output_dir: Path,
        velocity: float = 2500.0,
        aperture: float = 2500.0,
        tile_size: int = 32,
        grid_spacing: float = 25.0,
        offsets: list[float] | None = None,
        max_size_km: float | None = None,
        backend: str = "auto",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.velocity = velocity
        self.aperture = aperture
        self.tile_size = tile_size
        self.grid_spacing = grid_spacing
        self.offsets = offsets or [200, 400]
        self.max_size_km = max_size_km
        self.backend = backend

        self.results: list[ScalingResult] = []

    def _log(self, msg: str) -> None:
        """Print timestamped log message."""
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def run_all(self) -> None:
        """Run all scaling tests."""
        sizes = self.SURVEY_SIZES_KM
        if self.max_size_km:
            sizes = [s for s in sizes if s <= self.max_size_km]

        self._log(f"Starting scaling tests for {len(sizes)} survey sizes")
        self._log(f"Grid spacing: {self.grid_spacing}m")
        self._log(f"Offsets: {self.offsets}")
        self._log(f"Aperture: {self.aperture}m")
        self._log(f"Tile size: {self.tile_size}x{self.tile_size}")
        print()

        for size_km in sizes:
            try:
                result = self.run_single_test(size_km)
                self.results.append(result)
                self._print_result(result)
            except Exception as e:
                self._log(f"FAILED for {size_km}km: {e}")
                import traceback
                traceback.print_exc()

            # Clean up between tests
            gc.collect()

        self._print_summary_table()

    def run_single_test(self, size_km: float) -> ScalingResult:
        """Run test for a single survey size."""
        self._log(f"Testing {size_km}km x {size_km}km survey...")

        survey_extent_m = size_km * 1000

        # Initialize result
        result = ScalingResult(
            survey_size_km=size_km,
            survey_extent_m=survey_extent_m,
            n_traces=0,
            n_midpoints=0,
            data_size_mb=0.0,
        )

        # Step 1: Generate synthetic data
        traces_path, headers_path = self._generate_synthetic(
            survey_extent_m, result
        )

        # Step 2: Run diagnostics
        self._run_diagnostics(traces_path, headers_path, result)

        # Clean up synthetic files to save disk space
        self._cleanup_files(traces_path, headers_path)

        return result

    def _generate_synthetic(
        self, survey_extent_m: float, result: ScalingResult
    ) -> tuple[Path, Path]:
        """Generate synthetic data for given survey extent."""
        from pstm.synthetic import (
            create_simple_synthetic,
            export_to_zarr_parquet,
        )

        # Diffractor in center
        center = survey_extent_m / 2

        self._log(f"  Generating synthetic data...")
        t0 = time.perf_counter()

        synth_result = create_simple_synthetic(
            diffractor_x=center,
            diffractor_y=center,
            diffractor_z=800.0,
            survey_extent=survey_extent_m,
            grid_spacing=self.grid_spacing,
            offsets=self.offsets,
            azimuths=[0, 360],
            velocity=self.velocity,
            n_samples=1501,  # 3s at 2ms
            dt_ms=2.0,
            wavelet_freq=25.0,
            noise_level=0.1,
        )

        result.generation_time = time.perf_counter() - t0
        result.n_traces = synth_result.n_traces
        result.n_midpoints = synth_result.config.survey.nx * synth_result.config.survey.ny
        result.data_size_mb = synth_result.traces.nbytes / 1024**2

        self._log(f"    Generated {result.n_traces:,} traces in {result.generation_time:.2f}s")

        # Export to files
        self._log(f"  Exporting to Zarr/Parquet...")
        t0 = time.perf_counter()

        test_dir = self.output_dir / f"test_{int(result.survey_size_km)}km"
        test_dir.mkdir(parents=True, exist_ok=True)

        traces_path, headers_path = export_to_zarr_parquet(
            synth_result,
            test_dir,
            traces_name="traces.zarr",
            headers_name="headers.parquet",
        )

        result.export_time = time.perf_counter() - t0
        self._log(f"    Exported in {result.export_time:.2f}s")

        return traces_path, headers_path

    def _run_diagnostics(
        self, traces_path: Path, headers_path: Path, result: ScalingResult
    ) -> None:
        """Run diagnostic benchmarks."""
        import pandas as pd
        import zarr
        from scipy.spatial import cKDTree

        # Load headers
        self._log(f"  Loading headers...")
        t0 = time.perf_counter()
        headers_df = pd.read_parquet(headers_path)
        result.header_load_time = time.perf_counter() - t0

        # Access traces
        self._log(f"  Accessing trace store...")
        t0 = time.perf_counter()
        store = zarr.open(str(traces_path), mode='r')
        if isinstance(store, zarr.Array):
            traces_array = store
        else:
            traces_array = store['data']
        n_traces, n_samples = traces_array.shape
        result.trace_access_time = time.perf_counter() - t0

        # Build spatial index
        self._log(f"  Building spatial index...")
        mx = headers_df['CDP_X'].values
        my = headers_df['CDP_Y'].values

        t0 = time.perf_counter()
        coords = np.column_stack([mx, my])
        tree = cKDTree(coords)
        result.spatial_index_time = time.perf_counter() - t0

        # Query center tile
        cx = (mx.min() + mx.max()) / 2
        cy = (my.min() + my.max()) / 2

        self._log(f"  Querying aperture...")
        t0 = time.perf_counter()
        indices = tree.query_ball_point([cx, cy], self.aperture)
        result.aperture_query_time = time.perf_counter() - t0

        indices = np.array(sorted(indices))
        n_traces_in_aperture = min(len(indices), 100000)
        indices = indices[:n_traces_in_aperture]
        result.traces_per_tile = n_traces_in_aperture

        # Load trace data
        self._log(f"  Loading {n_traces_in_aperture:,} traces...")
        t0 = time.perf_counter()
        amplitudes = traces_array[indices, :].astype(np.float32)
        result.trace_load_time = time.perf_counter() - t0

        # Run kernel benchmark
        self._log(f"  Running kernel benchmark...")
        self._run_kernel_benchmark(
            amplitudes, headers_df, indices, mx, my, result
        )

        # Calculate total tiles and estimated time
        tile_span = self.tile_size * self.grid_spacing
        n_tiles_x = max(1, int((mx.max() - mx.min()) / tile_span))
        n_tiles_y = max(1, int((my.max() - my.min()) / tile_span))
        result.n_tiles_total = n_tiles_x * n_tiles_y
        result.estimated_total_time_s = result.single_tile_time * result.n_tiles_total

    def _run_kernel_benchmark(
        self,
        amplitudes: np.ndarray,
        headers_df,
        indices: np.ndarray,
        mx: np.ndarray,
        my: np.ndarray,
        result: ScalingResult,
    ) -> None:
        """Run kernel benchmark on loaded data."""
        from pstm.kernels.factory import create_kernel
        from pstm.kernels.base import (
            KernelConfig, TraceBlock, OutputTile, VelocitySlice
        )

        # Initialize kernel based on backend setting
        t0 = time.perf_counter()
        if self.backend == "auto":
            try:
                from pstm.kernels.metal_cpp import is_metal_cpp_available
                if is_metal_cpp_available():
                    kernel = create_kernel("metal_cpp")
                    self._log(f"  Using Metal C++ GPU kernel (auto-selected)")
                else:
                    kernel = create_kernel("numba_cpu")
                    self._log(f"  Using Numba CPU kernel (Metal not available)")
            except ImportError:
                kernel = create_kernel("numba_cpu")
                self._log(f"  Using Numba CPU kernel")
        else:
            kernel = create_kernel(self.backend)
            self._log(f"  Using {self.backend} kernel")
        config = KernelConfig(
            max_aperture_m=self.aperture,
            min_aperture_m=100.0,
            max_dip_degrees=45.0,
            apply_spreading=False,
            apply_obliquity=False,
        )
        kernel.initialize(config)
        result.kernel_init_time = time.perf_counter() - t0

        # Prepare trace block
        sx = headers_df['SOU_X'].values[indices].astype(np.float64)
        sy = headers_df['SOU_Y'].values[indices].astype(np.float64)
        rx = headers_df['REC_X'].values[indices].astype(np.float64)
        ry = headers_df['REC_Y'].values[indices].astype(np.float64)
        mid_x = mx[indices].astype(np.float64)
        mid_y = my[indices].astype(np.float64)
        offset = np.sqrt((rx - sx)**2 + (ry - sy)**2)

        traces = TraceBlock(
            amplitudes=amplitudes,
            source_x=sx,
            source_y=sy,
            receiver_x=rx,
            receiver_y=ry,
            midpoint_x=mid_x,
            midpoint_y=mid_y,
            offset=offset,
            sample_rate_ms=2.0,
            start_time_ms=0.0,
        )

        # Create output tile
        cx = (mx.min() + mx.max()) / 2
        cy = (my.min() + my.max()) / 2
        tile_span = self.tile_size * self.grid_spacing

        ox = np.linspace(cx - tile_span/2, cx + tile_span/2, self.tile_size)
        oy = np.linspace(cy - tile_span/2, cy + tile_span/2, self.tile_size)
        ot = np.linspace(0, 3000, 1501)

        output = OutputTile(
            image=np.zeros((self.tile_size, self.tile_size, len(ot)), dtype=np.float64),
            fold=np.zeros((self.tile_size, self.tile_size), dtype=np.int32),
            x_axis=ox,
            y_axis=oy,
            t_axis_ms=ot,
        )

        velocity = VelocitySlice(
            vrms=np.full(len(ot), self.velocity),
            is_1d=True,
            t_axis_ms=ot,
        )

        # Run migration
        t0 = time.perf_counter()
        metrics = kernel.migrate_tile(traces, output, velocity)
        result.single_tile_time = time.perf_counter() - t0

        # Calculate rates
        n_traces_processed = len(indices)
        result.traces_per_s = n_traces_processed / result.single_tile_time if result.single_tile_time > 0 else 0

        result.stats['max_fold'] = int(output.fold.max())
        result.stats['mean_fold'] = float(output.fold.mean())
        result.stats['image_max'] = float(np.abs(output.image).max())

    def _cleanup_files(self, traces_path: Path, headers_path: Path) -> None:
        """Clean up test files to save disk space."""
        try:
            if traces_path.exists():
                shutil.rmtree(traces_path)
            if headers_path.exists():
                headers_path.unlink()
            # Remove parent directory if empty
            parent = traces_path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception as e:
            self._log(f"  Warning: cleanup failed: {e}")

    def _print_result(self, result: ScalingResult) -> None:
        """Print single test result."""
        print(f"\n  Results for {result.survey_size_km}km x {result.survey_size_km}km:")
        print(f"    Traces: {result.n_traces:,}")
        print(f"    Midpoints: {result.n_midpoints:,}")
        print(f"    Data size: {result.data_size_mb:.1f} MB")
        print(f"    Generation: {result.generation_time:.2f}s")
        print(f"    Export: {result.export_time:.2f}s")
        print(f"    Spatial index: {result.spatial_index_time:.3f}s")
        print(f"    Aperture query: {result.aperture_query_time:.4f}s")
        print(f"    Trace load: {result.trace_load_time:.3f}s")
        print(f"    Kernel init: {result.kernel_init_time:.3f}s")
        print(f"    Single tile: {result.single_tile_time:.2f}s ({result.traces_per_tile:,} traces)")
        print(f"    Traces/s: {result.traces_per_s:,.0f}")
        print(f"    Total tiles: {result.n_tiles_total}")
        print(f"    Est. total time: {result.estimated_total_time_s/60:.1f} min")
        print()

    def _print_summary_table(self) -> None:
        """Print summary table of all results."""
        print("\n" + "=" * 120)
        print("SCALING TEST SUMMARY")
        print("=" * 120)

        # Header
        headers = [
            "Size (km)", "Traces", "Midpoints", "Data (MB)",
            "Gen (s)", "Index (s)", "Query (ms)", "Load (s)",
            "Tile (s)", "Traces/s", "Tiles", "Est. Total"
        ]

        # Calculate column widths
        widths = [12, 12, 12, 10, 8, 10, 10, 8, 8, 12, 8, 12]

        # Print header
        header_row = " | ".join(f"{h:>{w}}" for h, w in zip(headers, widths))
        print(header_row)
        print("-" * len(header_row))

        # Print data rows
        for r in self.results:
            est_total = f"{r.estimated_total_time_s/60:.1f} min" if r.estimated_total_time_s < 3600 else f"{r.estimated_total_time_s/3600:.1f} hr"

            row = [
                f"{r.survey_size_km}x{r.survey_size_km}",
                f"{r.n_traces:,}",
                f"{r.n_midpoints:,}",
                f"{r.data_size_mb:.0f}",
                f"{r.generation_time:.1f}",
                f"{r.spatial_index_time:.3f}",
                f"{r.aperture_query_time*1000:.2f}",
                f"{r.trace_load_time:.2f}",
                f"{r.single_tile_time:.1f}",
                f"{r.traces_per_s:,.0f}",
                f"{r.n_tiles_total}",
                est_total,
            ]

            print(" | ".join(f"{v:>{w}}" for v, w in zip(row, widths)))

        print("=" * 120)

        # Print detailed breakdown table
        print("\n" + "=" * 100)
        print("DETAILED TIMING BREAKDOWN (seconds)")
        print("=" * 100)

        detail_headers = [
            "Size", "Generate", "Export", "Headers", "Index",
            "Query", "Load", "Kernel", "Tile"
        ]
        detail_widths = [10, 10, 10, 10, 10, 10, 10, 10, 10]

        header_row = " | ".join(f"{h:>{w}}" for h, w in zip(detail_headers, detail_widths))
        print(header_row)
        print("-" * len(header_row))

        for r in self.results:
            row = [
                f"{r.survey_size_km}km",
                f"{r.generation_time:.2f}",
                f"{r.export_time:.2f}",
                f"{r.header_load_time:.3f}",
                f"{r.spatial_index_time:.3f}",
                f"{r.aperture_query_time:.4f}",
                f"{r.trace_load_time:.3f}",
                f"{r.kernel_init_time:.3f}",
                f"{r.single_tile_time:.2f}",
            ]
            print(" | ".join(f"{v:>{w}}" for v, w in zip(row, detail_widths)))

        print("=" * 100)

        # Save to CSV
        self._save_csv()

    def _save_csv(self) -> None:
        """Save results to CSV file."""
        csv_path = self.output_dir / "scaling_results.csv"

        with open(csv_path, 'w') as f:
            # Header
            f.write("survey_size_km,n_traces,n_midpoints,data_size_mb,")
            f.write("generation_s,export_s,header_load_s,spatial_index_s,")
            f.write("aperture_query_s,trace_load_s,kernel_init_s,single_tile_s,")
            f.write("traces_per_tile,traces_per_s,n_tiles_total,estimated_total_s\n")

            # Data
            for r in self.results:
                f.write(f"{r.survey_size_km},{r.n_traces},{r.n_midpoints},{r.data_size_mb:.2f},")
                f.write(f"{r.generation_time:.3f},{r.export_time:.3f},{r.header_load_time:.4f},{r.spatial_index_time:.4f},")
                f.write(f"{r.aperture_query_time:.6f},{r.trace_load_time:.4f},{r.kernel_init_time:.4f},{r.single_tile_time:.3f},")
                f.write(f"{r.traces_per_tile},{r.traces_per_s:.1f},{r.n_tiles_total},{r.estimated_total_time_s:.1f}\n")

        print(f"\nResults saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PSTM Scaling Test - benchmark across survey sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with small sizes only
    python scripts/scaling_test.py --max-size 10

    # Full test
    python scripts/scaling_test.py

    # Custom parameters
    python scripts/scaling_test.py --aperture 5000 --tile-size 48 --velocity 3000
        """
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./scaling_test_output"),
        help="Output directory for test files (default: ./scaling_test_output)"
    )
    parser.add_argument(
        "--velocity", "-v",
        type=float,
        default=2500.0,
        help="Velocity in m/s (default: 2500)"
    )
    parser.add_argument(
        "--aperture", "-a",
        type=float,
        default=2500.0,
        help="Migration aperture in meters (default: 2500)"
    )
    parser.add_argument(
        "--tile-size", "-t",
        type=int,
        default=32,
        help="Tile size in bins (default: 32)"
    )
    parser.add_argument(
        "--grid-spacing", "-g",
        type=float,
        default=25.0,
        help="Grid spacing in meters (default: 25)"
    )
    parser.add_argument(
        "--max-size", "-m",
        type=float,
        default=None,
        help="Maximum survey size in km to test (default: all)"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="auto",
        choices=["auto", "metal_cpp", "numba_cpu", "numpy"],
        help="Compute backend (default: auto - uses Metal GPU if available)"
    )

    args = parser.parse_args()

    test = ScalingTest(
        output_dir=args.output_dir,
        velocity=args.velocity,
        aperture=args.aperture,
        tile_size=args.tile_size,
        grid_spacing=args.grid_spacing,
        max_size_km=args.max_size,
        backend=args.backend,
    )

    test.run_all()


if __name__ == "__main__":
    main()
