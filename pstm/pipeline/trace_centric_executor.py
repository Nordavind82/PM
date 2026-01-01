"""
Trace-Centric Migration Executor.

This module provides an alternative migration strategy that processes all traces
in a single pass, eliminating redundant trace loading for overlapping tiles.

Key advantages:
- Each trace loaded exactly once (vs. N times in tile-centric approach)
- O(traces) complexity instead of O(tiles × traces_per_tile)
- Dramatically faster when trace overlap is high (>50% per tile)

Usage:
    from pstm.pipeline.trace_centric_executor import run_trace_centric_migration

    # Call instead of tile-by-tile processing when overlap is high
    run_trace_centric_migration(
        trace_reader=reader,
        header_manager=headers,
        velocity_manager=velocity,
        output_grid=grid,
        config=kernel_config,
        memmap_manager=memmap,
        progress_callback=callback,
    )
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pstm.data.zarr_reader import ZarrTraceReader
    from pstm.data.parquet_headers import ParquetHeaderManager
    from pstm.migration.velocity import VelocityManager
    from pstm.config.models import OutputGrid
    from pstm.kernels.base import KernelConfig
    from pstm.pipeline.memmap import MemmapManager

from pstm.kernels.base import (
    KernelMetrics,
    OutputTile,
    TraceBlock,
    VelocitySlice,
    create_trace_block,
)
from pstm.utils.logging import get_logger

logger = get_logger(__name__)
debug_logger = logging.getLogger("pstm.migration.debug")


@dataclass
class TraceCentricConfig:
    """Configuration for trace-centric migration."""
    batch_size: int = 50_000  # Traces per batch (GPU memory limit)
    report_interval: int = 10_000  # Report progress every N traces
    use_cache: bool = True  # Enable GPU trace caching


@dataclass
class TraceCentricProgress:
    """Progress information for trace-centric migration."""
    traces_processed: int
    total_traces: int
    elapsed_seconds: float
    traces_per_second: float
    estimated_remaining_seconds: float
    message: str


def run_trace_centric_migration(
    trace_reader: "ZarrTraceReader",
    header_manager: "ParquetHeaderManager",
    velocity_manager: "VelocityManager",
    output_grid: "OutputGrid",
    kernel_config: "KernelConfig",
    memmap_manager: "MemmapManager",
    progress_callback: Callable[[TraceCentricProgress], None] | None = None,
    trace_indices: NDArray[np.int64] | None = None,
    tc_config: TraceCentricConfig | None = None,
) -> KernelMetrics:
    """
    Run trace-centric PSTM migration.

    Processes all traces in a single pass, scattering contributions to output grid.

    Args:
        trace_reader: Zarr trace data reader
        header_manager: Parquet header manager for geometry
        velocity_manager: Velocity model manager
        output_grid: Output grid specification
        kernel_config: Kernel configuration
        memmap_manager: Memory-mapped output manager
        progress_callback: Optional callback for progress updates
        trace_indices: Optional subset of trace indices to process (None = all)
        tc_config: Trace-centric configuration

    Returns:
        KernelMetrics with processing statistics
    """
    if tc_config is None:
        tc_config = TraceCentricConfig()

    start_time = time.time()

    # Determine traces to process
    if trace_indices is None:
        n_traces = trace_reader.n_traces
        trace_indices = np.arange(n_traces, dtype=np.int64)
    else:
        n_traces = len(trace_indices)

    debug_logger.info(f"[TRACE-CENTRIC] Starting migration of {n_traces:,} traces")
    debug_logger.info(f"[TRACE-CENTRIC] Output grid: {output_grid.nx} x {output_grid.ny} x {output_grid.nt}")
    debug_logger.info(f"[TRACE-CENTRIC] Batch size: {tc_config.batch_size:,}")

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"TRACE-CENTRIC MIGRATION", file=sys.stderr, flush=True)
    print(f"  Traces: {n_traces:,}", file=sys.stderr, flush=True)
    print(f"  Output: {output_grid.nx}x{output_grid.ny}x{output_grid.nt}", file=sys.stderr, flush=True)
    print(f"{'='*60}", file=sys.stderr, flush=True)

    # Get output arrays from memmap
    image = memmap_manager.get("image")
    fold = memmap_manager.get("fold")

    # Import and initialize trace-centric kernel
    try:
        from pstm.kernels.trace_centric import TraceCentricKernel
        kernel = TraceCentricKernel()
        kernel.initialize(kernel_config)
        debug_logger.info("[TRACE-CENTRIC] Using GPU trace-centric kernel")
    except Exception as e:
        debug_logger.warning(f"[TRACE-CENTRIC] GPU kernel not available: {e}")
        debug_logger.info("[TRACE-CENTRIC] Falling back to CPU implementation")
        kernel = None

    # Prepare velocity slice (full grid)
    velocity = velocity_manager.get_velocity_slice_for_tile(
        0, output_grid.nx,
        0, output_grid.ny,
    )

    # Prepare output tile for full grid
    output_tile = OutputTile(
        image=np.zeros((output_grid.nx, output_grid.ny, output_grid.nt), dtype=np.float64),
        fold=np.zeros((output_grid.nx, output_grid.ny, output_grid.nt), dtype=np.int32),  # 3D fold per sample
        x_axis=np.linspace(
            output_grid.x_min, output_grid.x_max, output_grid.nx
        ),
        y_axis=np.linspace(
            output_grid.y_min, output_grid.y_max, output_grid.ny
        ),
        t_axis_ms=np.arange(
            output_grid.t_min_ms,
            output_grid.t_max_ms + output_grid.dt_ms / 2,
            output_grid.dt_ms,
        ),
    )

    total_traces_processed = 0
    total_kernel_time = 0.0

    # Process in batches to manage GPU memory
    n_batches = (n_traces + tc_config.batch_size - 1) // tc_config.batch_size

    for batch_idx in range(n_batches):
        batch_start = batch_idx * tc_config.batch_size
        batch_end = min(batch_start + tc_config.batch_size, n_traces)
        batch_indices = trace_indices[batch_start:batch_end]
        batch_size = len(batch_indices)

        batch_start_time = time.time()

        # Load trace data
        t0 = time.time()
        trace_data = trace_reader.get_traces(batch_indices)
        t_load = time.time() - t0
        trace_data_mb = trace_data.nbytes / 1024**2

        debug_logger.info(f"[BATCH {batch_idx+1}/{n_batches}] Loaded {batch_size:,} traces in {t_load:.2f}s ({trace_data_mb:.1f} MB)")

        # Load geometry
        t0 = time.time()
        geometry = header_manager.get_geometry_for_indices(batch_indices)
        t_geom = time.time() - t0

        debug_logger.info(f"[BATCH {batch_idx+1}/{n_batches}] Loaded geometry in {t_geom:.2f}s")

        # Create trace block
        traces = create_trace_block(
            amplitudes=trace_data,
            source_x=geometry.source_x,
            source_y=geometry.source_y,
            receiver_x=geometry.receiver_x,
            receiver_y=geometry.receiver_y,
            sample_rate_ms=trace_reader.sample_rate_ms or 2.0,
            start_time_ms=trace_reader.info.start_time_ms or 0.0,
        )

        # Execute kernel
        t0 = time.time()
        if kernel is not None:
            batch_metrics = kernel.migrate_tile(traces, output_tile, velocity, kernel_config)
        else:
            batch_metrics = _cpu_trace_centric_migrate(
                traces, output_tile, velocity, kernel_config
            )
        t_kernel = time.time() - t0
        total_kernel_time += t_kernel

        total_traces_processed += batch_size

        # Report progress
        elapsed = time.time() - start_time
        traces_per_sec = total_traces_processed / elapsed if elapsed > 0 else 0
        remaining = n_traces - total_traces_processed
        eta = remaining / traces_per_sec if traces_per_sec > 0 else 0

        debug_logger.info(
            f"[BATCH {batch_idx+1}/{n_batches}] Kernel: {t_kernel:.2f}s | "
            f"Rate: {batch_size/t_kernel:,.0f} traces/s | "
            f"Progress: {100*total_traces_processed/n_traces:.1f}%"
        )

        print(
            f"  Batch {batch_idx+1}/{n_batches}: {total_traces_processed:,}/{n_traces:,} "
            f"({100*total_traces_processed/n_traces:.1f}%) - "
            f"{traces_per_sec:,.0f} traces/s - ETA: {eta:.0f}s",
            file=sys.stderr, flush=True
        )

        if progress_callback:
            progress = TraceCentricProgress(
                traces_processed=total_traces_processed,
                total_traces=n_traces,
                elapsed_seconds=elapsed,
                traces_per_second=traces_per_sec,
                estimated_remaining_seconds=eta,
                message=f"Batch {batch_idx+1}/{n_batches}",
            )
            progress_callback(progress)

    # Accumulate to main output
    image[:] += output_tile.image
    fold[:] += output_tile.fold

    total_time = time.time() - start_time

    debug_logger.info(f"[TRACE-CENTRIC] Complete!")
    debug_logger.info(f"[TRACE-CENTRIC] Total time: {total_time:.1f}s")
    debug_logger.info(f"[TRACE-CENTRIC] Kernel time: {total_kernel_time:.1f}s ({100*total_kernel_time/total_time:.1f}%)")
    debug_logger.info(f"[TRACE-CENTRIC] Overall rate: {n_traces/total_time:,.0f} traces/s")

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"TRACE-CENTRIC COMPLETE", file=sys.stderr, flush=True)
    print(f"  Total time: {total_time:.1f}s", file=sys.stderr, flush=True)
    print(f"  Traces processed: {total_traces_processed:,}", file=sys.stderr, flush=True)
    print(f"  Overall rate: {n_traces/total_time:,.0f} traces/s", file=sys.stderr, flush=True)
    print(f"{'='*60}\n", file=sys.stderr, flush=True)

    return KernelMetrics(
        n_traces_processed=total_traces_processed,
        n_samples_output=output_grid.nx * output_grid.ny * output_grid.nt,
        compute_time_s=total_kernel_time,
    )


def _cpu_trace_centric_migrate(
    traces: TraceBlock,
    output: OutputTile,
    velocity: VelocitySlice,
    config: "KernelConfig",
) -> KernelMetrics:
    """
    CPU fallback for trace-centric migration.

    This is much slower than GPU but provides a working reference.
    """
    debug_logger.warning("[TRACE-CENTRIC] Using CPU fallback - this will be slow!")

    start_time = time.time()

    n_traces = traces.n_traces
    nx, ny, nt = output.nx, output.ny, output.nt
    n_samples = traces.n_samples

    # Get coordinates
    x_coords = output.x_axis
    y_coords = output.y_axis
    t_axis_ms = output.t_axis_ms
    t_axis_s = t_axis_ms / 1000.0

    # Get velocity
    if velocity.is_1d:
        vrms = velocity.vrms
    else:
        vrms = velocity.vrms[nx//2, ny//2, :]

    # Pre-compute time-dependent values
    t0_half_sq = (t_axis_s / 2.0) ** 2
    inv_v_sq = 1.0 / (vrms ** 2)

    dx = x_coords[1] - x_coords[0] if nx > 1 else 25.0
    dy = y_coords[1] - y_coords[0] if ny > 1 else 25.0
    x_min = x_coords[0]
    y_min = y_coords[0]
    max_aperture = config.max_aperture_m

    # Process each trace
    for tr in range(n_traces):
        sx, sy = traces.source_x[tr], traces.source_y[tr]
        rx, ry = traces.receiver_x[tr], traces.receiver_y[tr]
        mx, my = traces.midpoint_x[tr], traces.midpoint_y[tr]
        trace_amp = traces.amplitudes[tr]

        # Determine output bin range
        ix_min = max(0, int((mx - max_aperture - x_min) / dx))
        ix_max = min(nx - 1, int((mx + max_aperture - x_min) / dx))
        iy_min = max(0, int((my - max_aperture - y_min) / dy))
        iy_max = min(ny - 1, int((my + max_aperture - y_min) / dy))

        # Loop over output points
        for ix in range(ix_min, ix_max + 1):
            ox = x_coords[ix]
            for iy in range(iy_min, iy_max + 1):
                oy = y_coords[iy]

                dm = np.sqrt((ox - mx)**2 + (oy - my)**2)
                if dm > max_aperture:
                    continue

                ds2 = (ox - sx)**2 + (oy - sy)**2
                dr2 = (ox - rx)**2 + (oy - ry)**2

                # Loop over output times
                for it in range(nt):
                    t0_half_sq_val = t0_half_sq[it]
                    inv_v_sq_val = inv_v_sq[it]

                    # DSR traveltime
                    t_travel = (
                        np.sqrt(t0_half_sq_val + ds2 * inv_v_sq_val) +
                        np.sqrt(t0_half_sq_val + dr2 * inv_v_sq_val)
                    )

                    # Sample index
                    sample_idx = (t_travel * 1000.0 - traces.start_time_ms) / traces.sample_rate_ms

                    if sample_idx < 0 or sample_idx >= n_samples - 1:
                        continue

                    # Linear interpolation
                    idx0 = int(sample_idx)
                    frac = sample_idx - idx0
                    amp = trace_amp[idx0] * (1 - frac) + trace_amp[idx0 + 1] * frac

                    # Apply taper
                    taper_start = max_aperture * (1 - config.taper_fraction)
                    if dm > taper_start:
                        t = (dm - taper_start) / (max_aperture - taper_start)
                        amp *= 0.5 * (1 + np.cos(t * np.pi))

                    output.image[ix, iy, it] += amp

        # Update fold (once per trace)
        if ix_min <= ix_max and iy_min <= iy_max:
            output.fold[ix_min, iy_min] += 1

        if (tr + 1) % 1000 == 0:
            debug_logger.debug(f"[CPU] Processed {tr+1:,}/{n_traces:,} traces")

    compute_time = time.time() - start_time

    return KernelMetrics(
        n_traces_processed=n_traces,
        n_samples_output=nx * ny * nt,
        compute_time_s=compute_time,
    )


def estimate_trace_overlap(
    spatial_index,
    tile_plan,
    max_aperture: float,
    sample_tiles: int = 10,
) -> float:
    """
    Estimate average trace overlap between tiles.

    Used to decide whether trace-centric approach is beneficial.

    Args:
        spatial_index: Spatial index for trace queries
        tile_plan: Tile plan with tile specifications
        max_aperture: Maximum aperture in meters
        sample_tiles: Number of tiles to sample

    Returns:
        Estimated overlap ratio (0-1). Higher = more overlap = trace-centric better.
    """
    from pstm.pipeline.spatial import query_traces_for_tile

    # Sample some tiles
    n_tiles = tile_plan.n_tiles
    sample_indices = np.linspace(0, n_tiles - 1, min(sample_tiles, n_tiles), dtype=int)

    total_traces = 0
    unique_traces = set()

    for tile_idx in sample_indices:
        tile = tile_plan.tiles[tile_idx]
        result = query_traces_for_tile(
            spatial_index,
            tile.x_min, tile.x_max,
            tile.y_min, tile.y_max,
            max_aperture,
        )
        total_traces += result.n_traces
        unique_traces.update(result.trace_indices)

    if len(unique_traces) == 0:
        return 0.0

    # Overlap ratio: how many times each trace is loaded on average
    overlap_ratio = total_traces / len(unique_traces)

    # Normalize to 0-1 range (1 = no overlap, higher = more overlap)
    # If overlap_ratio > 2, trace-centric is likely beneficial
    benefit_score = min(1.0, (overlap_ratio - 1) / 10)

    debug_logger.info(f"[OVERLAP] Sampled {len(sample_indices)} tiles")
    debug_logger.info(f"[OVERLAP] Total trace loads: {total_traces:,}")
    debug_logger.info(f"[OVERLAP] Unique traces: {len(unique_traces):,}")
    debug_logger.info(f"[OVERLAP] Overlap ratio: {overlap_ratio:.1f}x")
    debug_logger.info(f"[OVERLAP] Trace-centric benefit score: {benefit_score:.2f}")

    return benefit_score
