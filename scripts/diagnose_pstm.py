#!/usr/bin/env python3
"""
PSTM Pipeline Diagnostic Script

This script runs step-by-step diagnostics on all relevant parts of the PSTM
CPU/Numba kernel, providing detailed statistics and timing information.

Usage:
    python scripts/diagnose_pstm.py \
        --traces /path/to/traces.zarr \
        --headers /path/to/headers.parquet \
        [--velocity 2500] \
        [--aperture 2500] \
        [--tile-size 32] \
        [--n-tiles 1]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DiagnosticResult:
    """Result from a diagnostic step."""
    name: str
    elapsed_s: float
    success: bool
    message: str
    stats: dict[str, Any] | None = None


class PSTMDiagnostics:
    """Run comprehensive diagnostics on PSTM pipeline components."""

    def __init__(
        self,
        traces_path: str,
        headers_path: str,
        velocity: float = 2500.0,
        aperture: float = 2500.0,
        tile_size: int = 32,
        n_tiles: int = 1,
    ):
        self.traces_path = Path(traces_path)
        self.headers_path = Path(headers_path)
        self.velocity = velocity
        self.aperture = aperture
        self.tile_size = tile_size
        self.n_tiles = n_tiles
        self.results: list[DiagnosticResult] = []

        # Data loaded during diagnostics
        self._headers_df = None
        self._traces_array = None
        self._n_traces = 0
        self._n_samples = 0
        self._sample_rate_ms = 2.0

    def _log(self, msg: str) -> None:
        """Print log message with timestamp."""
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _log_result(self, result: DiagnosticResult) -> None:
        """Print diagnostic result."""
        status = "OK" if result.success else "FAILED"
        print(f"\n{'='*60}")
        print(f"  {result.name}")
        print(f"{'='*60}")
        print(f"  Status: {status}")
        print(f"  Time: {result.elapsed_s:.3f}s")
        print(f"  {result.message}")
        if result.stats:
            print(f"  Statistics:")
            for key, value in result.stats.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

    def run_all(self) -> bool:
        """Run all diagnostics and return True if all passed."""
        self._log("Starting PSTM Pipeline Diagnostics")
        self._log(f"Traces: {self.traces_path}")
        self._log(f"Headers: {self.headers_path}")
        self._log(f"Velocity: {self.velocity} m/s")
        self._log(f"Aperture: {self.aperture} m")
        self._log(f"Tile size: {self.tile_size}x{self.tile_size}")
        print()

        diagnostics = [
            self.diagnose_file_access,
            self.diagnose_header_loading,
            self.diagnose_trace_loading,
            self.diagnose_spatial_index,
            self.diagnose_aperture_query,
            self.diagnose_trace_data_loading,
            self.diagnose_kernel_initialization,
            self.diagnose_kernel_warmup,
            self.diagnose_single_tile_migration,
            self.diagnose_multi_tile_performance,
        ]

        all_passed = True
        for diag in diagnostics:
            try:
                result = diag()
                self.results.append(result)
                self._log_result(result)
                if not result.success:
                    all_passed = False
                    self._log(f"Stopping due to failure in {result.name}")
                    break
            except Exception as e:
                result = DiagnosticResult(
                    name=diag.__name__,
                    elapsed_s=0.0,
                    success=False,
                    message=f"Exception: {e}",
                )
                self.results.append(result)
                self._log_result(result)
                all_passed = False
                break

        self._print_summary()
        return all_passed

    def _print_summary(self) -> None:
        """Print summary of all diagnostics."""
        print("\n" + "="*60)
        print("  DIAGNOSTIC SUMMARY")
        print("="*60)

        total_time = sum(r.elapsed_s for r in self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed

        print(f"\n  Total tests: {len(self.results)}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Total time: {total_time:.3f}s")

        print("\n  Timing breakdown:")
        for result in self.results:
            status = "OK" if result.success else "FAILED"
            pct = (result.elapsed_s / total_time * 100) if total_time > 0 else 0
            print(f"    {result.name:40s} {result.elapsed_s:8.3f}s ({pct:5.1f}%) [{status}]")

    # =========================================================================
    # Individual Diagnostics
    # =========================================================================

    def diagnose_file_access(self) -> DiagnosticResult:
        """Check if input files exist and are readable."""
        start = time.perf_counter()

        issues = []
        if not self.traces_path.exists():
            issues.append(f"Traces file not found: {self.traces_path}")
        if not self.headers_path.exists():
            issues.append(f"Headers file not found: {self.headers_path}")

        elapsed = time.perf_counter() - start

        if issues:
            return DiagnosticResult(
                name="1. File Access",
                elapsed_s=elapsed,
                success=False,
                message="\n    ".join(issues),
            )

        return DiagnosticResult(
            name="1. File Access",
            elapsed_s=elapsed,
            success=True,
            message="All input files accessible",
            stats={
                "traces_path": str(self.traces_path),
                "headers_path": str(self.headers_path),
            }
        )

    def diagnose_header_loading(self) -> DiagnosticResult:
        """Load and validate trace headers."""
        import pandas as pd

        start = time.perf_counter()

        try:
            self._headers_df = pd.read_parquet(self.headers_path)
            n_traces = len(self._headers_df)
            columns = list(self._headers_df.columns)

            # Check required columns
            required = ['SOU_X', 'SOU_Y', 'REC_X', 'REC_Y', 'CDP_X', 'CDP_Y']
            missing = [c for c in required if c not in columns]

            elapsed = time.perf_counter() - start

            if missing:
                return DiagnosticResult(
                    name="2. Header Loading",
                    elapsed_s=elapsed,
                    success=False,
                    message=f"Missing columns: {missing}",
                )

            # Compute header statistics
            mx = self._headers_df['CDP_X'].values
            my = self._headers_df['CDP_Y'].values

            return DiagnosticResult(
                name="2. Header Loading",
                elapsed_s=elapsed,
                success=True,
                message=f"Loaded {n_traces:,} trace headers",
                stats={
                    "n_traces": n_traces,
                    "n_columns": len(columns),
                    "x_range": f"[{mx.min():.1f}, {mx.max():.1f}]",
                    "y_range": f"[{my.min():.1f}, {my.max():.1f}]",
                    "memory_mb": self._headers_df.memory_usage(deep=True).sum() / 1024**2,
                    "load_rate_traces_per_s": n_traces / elapsed if elapsed > 0 else 0,
                }
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            return DiagnosticResult(
                name="2. Header Loading",
                elapsed_s=elapsed,
                success=False,
                message=f"Failed to load headers: {e}",
            )

    def diagnose_trace_loading(self) -> DiagnosticResult:
        """Load trace data and get array dimensions."""
        import zarr

        start = time.perf_counter()

        try:
            store = zarr.open(str(self.traces_path), mode='r')

            if isinstance(store, zarr.Array):
                self._traces_array = store
            elif 'data' in store:
                self._traces_array = store['data']
            else:
                elapsed = time.perf_counter() - start
                return DiagnosticResult(
                    name="3. Trace Data Access",
                    elapsed_s=elapsed,
                    success=False,
                    message="Could not find trace data in zarr store",
                )

            self._n_traces, self._n_samples = self._traces_array.shape

            # Get array info
            dtype = self._traces_array.dtype
            chunks = self._traces_array.chunks
            compressor = getattr(self._traces_array, 'compressor', None)

            elapsed = time.perf_counter() - start

            return DiagnosticResult(
                name="3. Trace Data Access",
                elapsed_s=elapsed,
                success=True,
                message=f"Opened {self._n_traces:,} traces x {self._n_samples} samples",
                stats={
                    "n_traces": self._n_traces,
                    "n_samples": self._n_samples,
                    "dtype": str(dtype),
                    "chunks": str(chunks),
                    "compressor": str(compressor),
                    "total_size_gb": (self._n_traces * self._n_samples * dtype.itemsize) / 1024**3,
                }
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            return DiagnosticResult(
                name="3. Trace Data Access",
                elapsed_s=elapsed,
                success=False,
                message=f"Failed to open traces: {e}",
            )

    def diagnose_spatial_index(self) -> DiagnosticResult:
        """Build and test spatial index for trace lookup."""
        start = time.perf_counter()

        try:
            from scipy.spatial import cKDTree

            mx = self._headers_df['CDP_X'].values
            my = self._headers_df['CDP_Y'].values

            # Build index
            t0 = time.perf_counter()
            coords = np.column_stack([mx, my])
            tree = cKDTree(coords)
            build_time = time.perf_counter() - t0

            # Test query
            center_x = (mx.min() + mx.max()) / 2
            center_y = (my.min() + my.max()) / 2

            t0 = time.perf_counter()
            indices = tree.query_ball_point([center_x, center_y], self.aperture)
            query_time = time.perf_counter() - t0

            elapsed = time.perf_counter() - start

            return DiagnosticResult(
                name="4. Spatial Index",
                elapsed_s=elapsed,
                success=True,
                message=f"Built KDTree and queried {len(indices):,} traces in aperture",
                stats={
                    "build_time_s": build_time,
                    "query_time_s": query_time,
                    "traces_in_aperture": len(indices),
                    "aperture_m": self.aperture,
                    "pct_of_total": 100 * len(indices) / self._n_traces,
                }
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            return DiagnosticResult(
                name="4. Spatial Index",
                elapsed_s=elapsed,
                success=False,
                message=f"Spatial index failed: {e}",
            )

    def diagnose_aperture_query(self) -> DiagnosticResult:
        """Test aperture query at multiple tile locations."""
        start = time.perf_counter()

        try:
            from scipy.spatial import cKDTree

            mx = self._headers_df['CDP_X'].values
            my = self._headers_df['CDP_Y'].values
            coords = np.column_stack([mx, my])
            tree = cKDTree(coords)

            # Generate tile centers
            x_min, x_max = mx.min(), mx.max()
            y_min, y_max = my.min(), my.max()

            tile_span = self.tile_size * 25  # Assume 25m bin size
            n_tiles_x = max(1, int((x_max - x_min) / tile_span))
            n_tiles_y = max(1, int((y_max - y_min) / tile_span))

            # Query multiple tiles
            trace_counts = []
            query_times = []

            for i in range(min(9, n_tiles_x * n_tiles_y)):
                ix = i % n_tiles_x
                iy = i // n_tiles_x

                cx = x_min + (ix + 0.5) * tile_span
                cy = y_min + (iy + 0.5) * tile_span

                t0 = time.perf_counter()
                indices = tree.query_ball_point([cx, cy], self.aperture)
                query_times.append(time.perf_counter() - t0)
                trace_counts.append(len(indices))

            elapsed = time.perf_counter() - start

            return DiagnosticResult(
                name="5. Aperture Query Performance",
                elapsed_s=elapsed,
                success=True,
                message=f"Tested {len(trace_counts)} tile queries",
                stats={
                    "min_traces_per_tile": min(trace_counts),
                    "max_traces_per_tile": max(trace_counts),
                    "mean_traces_per_tile": np.mean(trace_counts),
                    "min_query_time_ms": 1000 * min(query_times),
                    "max_query_time_ms": 1000 * max(query_times),
                    "mean_query_time_ms": 1000 * np.mean(query_times),
                }
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            return DiagnosticResult(
                name="5. Aperture Query Performance",
                elapsed_s=elapsed,
                success=False,
                message=f"Aperture query failed: {e}",
            )

    def diagnose_trace_data_loading(self) -> DiagnosticResult:
        """Test loading actual trace amplitudes from zarr."""
        start = time.perf_counter()

        try:
            from scipy.spatial import cKDTree

            mx = self._headers_df['CDP_X'].values
            my = self._headers_df['CDP_Y'].values
            coords = np.column_stack([mx, my])
            tree = cKDTree(coords)

            # Query center tile
            cx = (mx.min() + mx.max()) / 2
            cy = (my.min() + my.max()) / 2
            indices = tree.query_ball_point([cx, cy], self.aperture)
            indices = np.array(sorted(indices))

            # Load trace data
            n_to_load = min(len(indices), 50000)
            indices_to_load = indices[:n_to_load]

            t0 = time.perf_counter()
            amplitudes = self._traces_array[indices_to_load, :]
            load_time = time.perf_counter() - t0

            elapsed = time.perf_counter() - start

            data_size_mb = amplitudes.nbytes / 1024**2
            load_rate_mb_s = data_size_mb / load_time if load_time > 0 else 0

            return DiagnosticResult(
                name="6. Trace Data Loading",
                elapsed_s=elapsed,
                success=True,
                message=f"Loaded {n_to_load:,} traces ({data_size_mb:.1f} MB)",
                stats={
                    "n_traces_loaded": n_to_load,
                    "data_size_mb": data_size_mb,
                    "load_time_s": load_time,
                    "load_rate_mb_s": load_rate_mb_s,
                    "load_rate_traces_s": n_to_load / load_time if load_time > 0 else 0,
                    "amp_min": float(amplitudes.min()),
                    "amp_max": float(amplitudes.max()),
                    "amp_std": float(amplitudes.std()),
                }
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            return DiagnosticResult(
                name="6. Trace Data Loading",
                elapsed_s=elapsed,
                success=False,
                message=f"Trace data loading failed: {e}",
            )

    def diagnose_kernel_initialization(self) -> DiagnosticResult:
        """Test Numba kernel initialization and JIT compilation."""
        start = time.perf_counter()

        try:
            from pstm.kernels.factory import create_kernel
            from pstm.kernels.base import KernelConfig

            # Create kernel
            kernel = create_kernel("numba_cpu")

            # Initialize with config
            config = KernelConfig(
                max_aperture_m=self.aperture,
                min_aperture_m=100.0,
                max_dip_degrees=45.0,
                apply_spreading=False,
                apply_obliquity=False,
            )

            t0 = time.perf_counter()
            kernel.initialize(config)
            init_time = time.perf_counter() - t0

            elapsed = time.perf_counter() - start

            return DiagnosticResult(
                name="7. Kernel Initialization",
                elapsed_s=elapsed,
                success=True,
                message=f"Numba kernel initialized with JIT compilation",
                stats={
                    "kernel_name": kernel.name,
                    "init_time_s": init_time,
                    "max_aperture_m": self.aperture,
                }
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            return DiagnosticResult(
                name="7. Kernel Initialization",
                elapsed_s=elapsed,
                success=False,
                message=f"Kernel init failed: {e}",
            )

    def diagnose_kernel_warmup(self) -> DiagnosticResult:
        """Run kernel warmup with small synthetic data."""
        start = time.perf_counter()

        try:
            from pstm.kernels.factory import create_kernel
            from pstm.kernels.base import (
                KernelConfig, TraceBlock, OutputTile, VelocitySlice
            )

            kernel = create_kernel("numba_cpu")
            config = KernelConfig(
                max_aperture_m=self.aperture,
                min_aperture_m=100.0,
                max_dip_degrees=45.0,
            )
            kernel.initialize(config)

            # Small synthetic test
            n_warmup = 100
            n_samples = 500
            tile_size = 8

            rng = np.random.default_rng(42)
            traces = TraceBlock(
                amplitudes=rng.standard_normal((n_warmup, n_samples)).astype(np.float32),
                source_x=rng.uniform(0, 1000, n_warmup),
                source_y=rng.uniform(0, 1000, n_warmup),
                receiver_x=rng.uniform(0, 1000, n_warmup),
                receiver_y=rng.uniform(0, 1000, n_warmup),
                midpoint_x=rng.uniform(400, 600, n_warmup),
                midpoint_y=rng.uniform(400, 600, n_warmup),
                offset=rng.uniform(100, 3000, n_warmup),
                sample_rate_ms=2.0,
                start_time_ms=0.0,
            )

            ot = np.linspace(0, 1000, n_samples // 2)
            output = OutputTile(
                image=np.zeros((tile_size, tile_size, len(ot)), dtype=np.float64),
                fold=np.zeros((tile_size, tile_size), dtype=np.int32),
                x_axis=np.linspace(450, 550, tile_size),
                y_axis=np.linspace(450, 550, tile_size),
                t_axis_ms=ot,
            )

            velocity = VelocitySlice(
                vrms=np.full(len(ot), self.velocity),
                is_1d=True,
                t_axis_ms=ot,
            )

            # Warmup run - config is optional, kernel uses initialized config
            t0 = time.perf_counter()
            metrics = kernel.migrate_tile(traces, output, velocity)
            warmup_time = time.perf_counter() - t0

            elapsed = time.perf_counter() - start

            return DiagnosticResult(
                name="8. Kernel Warmup",
                elapsed_s=elapsed,
                success=True,
                message="JIT-compiled kernel warmed up",
                stats={
                    "warmup_traces": n_warmup,
                    "warmup_tile_size": tile_size,
                    "warmup_time_s": warmup_time,
                    "warmup_samples_s": metrics.n_samples_output / warmup_time if warmup_time > 0 else 0,
                }
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            return DiagnosticResult(
                name="8. Kernel Warmup",
                elapsed_s=elapsed,
                success=False,
                message=f"Kernel warmup failed: {e}",
            )

    def diagnose_single_tile_migration(self) -> DiagnosticResult:
        """Run migration on a single tile with real data."""
        start = time.perf_counter()

        try:
            from scipy.spatial import cKDTree
            from pstm.kernels.factory import create_kernel
            from pstm.kernels.base import (
                KernelConfig, TraceBlock, OutputTile, VelocitySlice
            )

            # Get tile location
            mx = self._headers_df['CDP_X'].values
            my = self._headers_df['CDP_Y'].values
            cx = (mx.min() + mx.max()) / 2
            cy = (my.min() + my.max()) / 2

            # Query traces in aperture
            coords = np.column_stack([mx, my])
            tree = cKDTree(coords)

            t0 = time.perf_counter()
            indices = tree.query_ball_point([cx, cy], self.aperture)
            query_time = time.perf_counter() - t0

            indices = np.array(sorted(indices))
            n_traces_tile = min(len(indices), 100000)  # Cap for diagnostics
            indices = indices[:n_traces_tile]

            # Load trace data
            t0 = time.perf_counter()
            amplitudes = self._traces_array[indices, :].astype(np.float32)
            load_time = time.perf_counter() - t0

            # Get geometry
            sx = self._headers_df['SOU_X'].values[indices]
            sy = self._headers_df['SOU_Y'].values[indices]
            rx = self._headers_df['REC_X'].values[indices]
            ry = self._headers_df['REC_Y'].values[indices]
            mid_x = mx[indices]
            mid_y = my[indices]
            offset = np.sqrt((rx - sx)**2 + (ry - sy)**2)

            traces = TraceBlock(
                amplitudes=amplitudes,
                source_x=sx.astype(np.float64),
                source_y=sy.astype(np.float64),
                receiver_x=rx.astype(np.float64),
                receiver_y=ry.astype(np.float64),
                midpoint_x=mid_x.astype(np.float64),
                midpoint_y=mid_y.astype(np.float64),
                offset=offset.astype(np.float64),
                sample_rate_ms=self._sample_rate_ms,
                start_time_ms=0.0,
            )

            # Create output tile
            tile_span = self.tile_size * 25  # 25m bin size
            ox = np.linspace(cx - tile_span/2, cx + tile_span/2, self.tile_size)
            oy = np.linspace(cy - tile_span/2, cy + tile_span/2, self.tile_size)
            ot = np.linspace(0, 3000, 1501)  # 0-3s at 2ms

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

            # Initialize and run kernel
            kernel = create_kernel("numba_cpu")
            config = KernelConfig(
                max_aperture_m=self.aperture,
                min_aperture_m=100.0,
                max_dip_degrees=45.0,
                apply_spreading=False,
                apply_obliquity=False,
            )
            kernel.initialize(config)

            t0 = time.perf_counter()
            metrics = kernel.migrate_tile(traces, output, velocity)
            kernel_time = time.perf_counter() - t0

            elapsed = time.perf_counter() - start

            # Calculate rates
            traces_per_s = n_traces_tile / kernel_time if kernel_time > 0 else 0
            samples_per_s = metrics.n_samples_output / kernel_time if kernel_time > 0 else 0

            return DiagnosticResult(
                name="9. Single Tile Migration",
                elapsed_s=elapsed,
                success=True,
                message=f"Migrated {n_traces_tile:,} traces to {self.tile_size}x{self.tile_size} tile",
                stats={
                    "n_traces": n_traces_tile,
                    "tile_size": self.tile_size,
                    "n_output_samples": output.image.size,
                    "query_time_s": query_time,
                    "load_time_s": load_time,
                    "kernel_time_s": kernel_time,
                    "traces_per_s": traces_per_s,
                    "samples_per_s": samples_per_s,
                    "max_fold": int(output.fold.max()),
                    "mean_fold": float(output.fold.mean()),
                    "image_max": float(np.abs(output.image).max()),
                }
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            import traceback
            return DiagnosticResult(
                name="9. Single Tile Migration",
                elapsed_s=elapsed,
                success=False,
                message=f"Migration failed: {e}\n{traceback.format_exc()}",
            )

    def diagnose_multi_tile_performance(self) -> DiagnosticResult:
        """Run migration on multiple tiles to estimate full job time."""
        start = time.perf_counter()

        try:
            from scipy.spatial import cKDTree
            from pstm.kernels.factory import create_kernel
            from pstm.kernels.base import (
                KernelConfig, TraceBlock, OutputTile, VelocitySlice
            )

            mx = self._headers_df['CDP_X'].values
            my = self._headers_df['CDP_Y'].values

            x_min, x_max = mx.min(), mx.max()
            y_min, y_max = my.min(), my.max()

            coords = np.column_stack([mx, my])
            tree = cKDTree(coords)

            # Initialize kernel once
            kernel = create_kernel("numba_cpu")
            config = KernelConfig(
                max_aperture_m=self.aperture,
                min_aperture_m=100.0,
                max_dip_degrees=45.0,
                apply_spreading=False,
                apply_obliquity=False,
            )
            kernel.initialize(config)

            # Output time axis
            ot = np.linspace(0, 3000, 1501)

            velocity = VelocitySlice(
                vrms=np.full(len(ot), self.velocity),
                is_1d=True,
                t_axis_ms=ot,
            )

            tile_span = self.tile_size * 25
            n_tiles_x = max(1, int((x_max - x_min) / tile_span))
            n_tiles_y = max(1, int((y_max - y_min) / tile_span))
            total_tiles = n_tiles_x * n_tiles_y

            # Process N tiles
            n_to_process = min(self.n_tiles, total_tiles)
            tile_times = []
            tile_trace_counts = []

            for i in range(n_to_process):
                ix = i % n_tiles_x
                iy = i // n_tiles_x

                cx = x_min + (ix + 0.5) * tile_span
                cy = y_min + (iy + 0.5) * tile_span

                # Query traces
                t0 = time.perf_counter()
                indices = tree.query_ball_point([cx, cy], self.aperture)
                indices = np.array(sorted(indices))
                n_traces_tile = min(len(indices), 100000)
                indices = indices[:n_traces_tile]

                # Load data
                amplitudes = self._traces_array[indices, :].astype(np.float32)

                sx = self._headers_df['SOU_X'].values[indices]
                sy = self._headers_df['SOU_Y'].values[indices]
                rx = self._headers_df['REC_X'].values[indices]
                ry = self._headers_df['REC_Y'].values[indices]
                mid_x = mx[indices]
                mid_y = my[indices]
                offset = np.sqrt((rx - sx)**2 + (ry - sy)**2)

                traces = TraceBlock(
                    amplitudes=amplitudes,
                    source_x=sx.astype(np.float64),
                    source_y=sy.astype(np.float64),
                    receiver_x=rx.astype(np.float64),
                    receiver_y=ry.astype(np.float64),
                    midpoint_x=mid_x.astype(np.float64),
                    midpoint_y=mid_y.astype(np.float64),
                    offset=offset.astype(np.float64),
                    sample_rate_ms=self._sample_rate_ms,
                    start_time_ms=0.0,
                )

                ox = np.linspace(cx - tile_span/2, cx + tile_span/2, self.tile_size)
                oy = np.linspace(cy - tile_span/2, cy + tile_span/2, self.tile_size)

                output = OutputTile(
                    image=np.zeros((self.tile_size, self.tile_size, len(ot)), dtype=np.float64),
                    fold=np.zeros((self.tile_size, self.tile_size), dtype=np.int32),
                    x_axis=ox,
                    y_axis=oy,
                    t_axis_ms=ot,
                )

                kernel.migrate_tile(traces, output, velocity)

                tile_time = time.perf_counter() - t0
                tile_times.append(tile_time)
                tile_trace_counts.append(n_traces_tile)

            elapsed = time.perf_counter() - start

            mean_tile_time = np.mean(tile_times)
            estimated_total_s = mean_tile_time * total_tiles

            return DiagnosticResult(
                name="10. Multi-Tile Performance",
                elapsed_s=elapsed,
                success=True,
                message=f"Processed {n_to_process} tiles, estimated {total_tiles} total",
                stats={
                    "tiles_processed": n_to_process,
                    "total_tiles": total_tiles,
                    "mean_tile_time_s": mean_tile_time,
                    "min_tile_time_s": min(tile_times),
                    "max_tile_time_s": max(tile_times),
                    "mean_traces_per_tile": np.mean(tile_trace_counts),
                    "estimated_total_time_s": estimated_total_s,
                    "estimated_total_time_min": estimated_total_s / 60,
                    "estimated_total_time_hr": estimated_total_s / 3600,
                }
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            import traceback
            return DiagnosticResult(
                name="10. Multi-Tile Performance",
                elapsed_s=elapsed,
                success=False,
                message=f"Multi-tile test failed: {e}\n{traceback.format_exc()}",
            )


def main():
    parser = argparse.ArgumentParser(
        description="PSTM Pipeline Diagnostic Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/diagnose_pstm.py \\
        --traces synthetic_output/diffractor_traces.zarr \\
        --headers synthetic_output/diffractor_headers.parquet

    python scripts/diagnose_pstm.py \\
        --traces data/traces.zarr \\
        --headers data/headers.parquet \\
        --velocity 3000 \\
        --aperture 2000 \\
        --tile-size 48 \\
        --n-tiles 5
        """
    )

    parser.add_argument(
        "--traces", "-t",
        required=True,
        help="Path to traces zarr file"
    )
    parser.add_argument(
        "--headers", "-H",
        required=True,
        help="Path to headers parquet file"
    )
    parser.add_argument(
        "--velocity", "-v",
        type=float,
        default=2500.0,
        help="Constant velocity in m/s (default: 2500)"
    )
    parser.add_argument(
        "--aperture", "-a",
        type=float,
        default=2500.0,
        help="Migration aperture in meters (default: 2500)"
    )
    parser.add_argument(
        "--tile-size", "-s",
        type=int,
        default=32,
        help="Tile size in bins (default: 32)"
    )
    parser.add_argument(
        "--n-tiles", "-n",
        type=int,
        default=3,
        help="Number of tiles to process for timing (default: 3)"
    )

    args = parser.parse_args()

    diagnostics = PSTMDiagnostics(
        traces_path=args.traces,
        headers_path=args.headers,
        velocity=args.velocity,
        aperture=args.aperture,
        tile_size=args.tile_size,
        n_tiles=args.n_tiles,
    )

    success = diagnostics.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
