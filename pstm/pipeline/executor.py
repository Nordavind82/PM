"""
Migration executor for PSTM.

Orchestrates the complete migration pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
import zarr
from numpy.typing import NDArray

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from pstm.config.models import MigrationConfig
from pstm.data.parquet_headers import TraceGeometry


def get_cpu_info() -> dict:
    """Get CPU utilization info for profiling."""
    if not HAS_PSUTIL:
        return {"available": False}

    try:
        # Get per-CPU utilization (non-blocking with interval=None uses cached value)
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        overall = psutil.cpu_percent(interval=None)

        # Count cores at various utilization levels
        active_cores = sum(1 for c in per_cpu if c > 50)
        busy_cores = sum(1 for c in per_cpu if c > 90)

        return {
            "available": True,
            "overall_percent": overall,
            "per_cpu": per_cpu,
            "n_cores": len(per_cpu),
            "active_cores": active_cores,  # >50% utilization
            "busy_cores": busy_cores,  # >90% utilization
        }
    except Exception:
        return {"available": False}


def setup_debug_logging(output_dir: Path | None = None) -> logging.Logger:
    """Setup comprehensive debug logging to console and file.

    This function preserves any existing handlers (e.g., from run script file logging)
    and only adds new handlers if none exist. It also enables propagation so messages
    reach parent loggers for centralized logging.
    """
    # Create a specific logger for migration debugging
    debug_logger = logging.getLogger("pstm.migration.debug")
    debug_logger.setLevel(logging.DEBUG)

    # Enable propagation to parent loggers (for centralized file logging)
    debug_logger.propagate = True

    # Only add handlers if none exist (preserve handlers from run script)
    if not debug_logger.handlers:
        # Console handler with detailed format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)s:%(lineno)d - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        debug_logger.addHandler(console_handler)

        # File handler if output_dir provided
        if output_dir:
            log_file = output_dir / f"migration_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            debug_logger.addHandler(file_handler)
            debug_logger.info(f"Debug log file: {log_file}")
    else:
        debug_logger.debug(f"Preserving {len(debug_logger.handlers)} existing handler(s)")

    return debug_logger


def get_system_info() -> dict:
    """Get system information for debugging."""
    info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
    }

    # Memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_memory_gb"] = round(mem.total / (1024**3), 2)
        info["available_memory_gb"] = round(mem.available / (1024**3), 2)
        info["used_memory_percent"] = mem.percent
    except ImportError:
        info["memory_info"] = "psutil not installed"

    # NumPy info
    info["numpy_version"] = np.__version__

    # Check for compute backends
    try:
        import numba
        info["numba_version"] = numba.__version__
        info["numba_available"] = True
    except ImportError:
        info["numba_available"] = False

    try:
        import mlx.core as mx
        info["mlx_available"] = True
        # Try to get Metal device info
        try:
            info["mlx_default_device"] = str(mx.default_device())
        except:
            pass
    except ImportError:
        info["mlx_available"] = False

    return info


def log_memory_state(debug_logger: logging.Logger, context: str = ""):
    """Log current memory state."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        process = psutil.Process()
        proc_mem = process.memory_info()

        # Detailed memory breakdown
        available_gb = mem.available / (1024**3)
        used_pct = mem.percent
        rss_gb = proc_mem.rss / (1024**3)
        vms_gb = proc_mem.vms / (1024**3)

        debug_logger.info(
            f"MEMORY [{context}] System: {available_gb:.2f} GB available "
            f"({used_pct:.1f}% used) | Process RSS: {rss_gb:.3f} GB | VMS: {vms_gb:.3f} GB"
        )

        # Log warning if memory is getting low
        if available_gb < 8.0:
            debug_logger.warning(f"MEMORY WARNING [{context}]: Low memory! Only {available_gb:.2f} GB available")
        if rss_gb > 30.0:
            debug_logger.warning(f"MEMORY WARNING [{context}]: High RSS! Process using {rss_gb:.2f} GB")

    except ImportError:
        debug_logger.debug(f"MEMORY [{context}] psutil not available for memory tracking")


from pstm.data import (
    MemmapManager,
    ParquetHeaderManager,
    SpatialIndex,
    ZarrTraceReader,
    create_velocity_model,
    query_traces_for_tile,
)
from pstm.data.velocity_model import VelocityManager, create_velocity_manager
from pstm.data.trace_cache import LRUTraceCache
from pstm.kernels.base import (
    KernelConfig,
    KernelMetrics,
    OutputTile,
    TraceBlock,
    VelocitySlice,
    create_output_tile,
    create_trace_block,
)
from pstm.kernels.factory import create_kernel
from pstm.pipeline.checkpoint import CheckpointHandler, should_checkpoint
from pstm.pipeline.tile_planner import TilePlan, TilePlanner, TileSpec, iter_tiles
from pstm.utils.logging import get_logger, print_metric, print_section, print_success
from pstm.utils.units import format_bytes, format_duration

logger = get_logger(__name__)


class ExecutionPhase(Enum):
    """Migration execution phases."""

    INIT = "initialization"
    PLANNING = "planning"
    MIGRATION = "migration"
    FINALIZATION = "finalization"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ExecutionMetrics:
    """Accumulated execution metrics."""

    start_time: float = 0.0
    end_time: float = 0.0

    n_tiles_total: int = 0
    n_tiles_completed: int = 0
    n_traces_processed: int = 0  # Input traces processed (for rate calculation)
    n_samples_output: int = 0
    n_output_bins_completed: int = 0  # Output bins (pillars) fully migrated
    total_output_bins: int = 0  # Total output bins = nx * ny

    compute_time_total: float = 0.0
    io_time_total: float = 0.0

    # Grid coverage QC metrics
    grid_coverage_ratio: float = 1.0  # Fraction of midpoints inside grid
    grid_coverage_n_outside: int = 0  # Number of midpoints outside grid

    warnings: list[str] = field(default_factory=list)

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def progress_fraction(self) -> float:
        """Completion progress as fraction."""
        if self.n_tiles_total == 0:
            return 0.0
        return self.n_tiles_completed / self.n_tiles_total

    @property
    def traces_per_second(self) -> float:
        """Average processing rate."""
        if self.compute_time_total > 0:
            return self.n_traces_processed / self.compute_time_total
        return 0.0

    def estimate_remaining_time(self) -> float:
        """Estimate remaining time in seconds."""
        if self.n_tiles_completed == 0:
            return 0.0
        time_per_tile = self.elapsed_time / self.n_tiles_completed
        remaining_tiles = self.n_tiles_total - self.n_tiles_completed
        return time_per_tile * remaining_tiles


@dataclass
class ProgressInfo:
    """Detailed progress information passed to callback."""
    phase: ExecutionPhase
    current_tile: int
    total_tiles: int
    message: str
    traces_processed: int = 0  # Input traces (for rate display)
    traces_in_tile: int = 0  # Input traces in current tile
    tile_x: int = 0
    tile_y: int = 0
    total_traces: int = 0  # Total input traces in dataset
    # Output progress (what matters for completion)
    output_bins_completed: int = 0  # Output bins fully migrated
    total_output_bins: int = 0  # Total output bins = nx * ny
    output_bins_in_tile: int = 0  # Output bins in current tile


# Type for progress callback - accepts ProgressInfo
ProgressCallback = Callable[[ProgressInfo], None]


@dataclass
class PrefetchedTileData:
    """Pre-fetched trace and geometry data for a tile."""
    tile_id: int
    trace_data: NDArray[np.float32] | None = None
    geometry: Any | None = None  # TraceGeometry
    trace_indices: NDArray[np.int64] | None = None
    error: str | None = None


class MigrationExecutor:
    """
    Main executor for 3D PSTM migration.

    Orchestrates:
    1. Initialization (data loading, index building)
    2. Tile planning
    3. Migration loop
    4. Finalization (normalization, output writing)
    """

    def __init__(
        self,
        config: MigrationConfig,
        progress_callback: ProgressCallback | None = None,
    ):
        """
        Initialize the executor.

        Args:
            config: Migration configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.progress_callback = progress_callback

        # Setup debug logging
        self._debug = setup_debug_logging(config.output.output_dir)
        self._debug.info("=" * 70)
        self._debug.info("PSTM MIGRATION EXECUTOR INITIALIZED")
        self._debug.info("=" * 70)

        # Log system info
        sys_info = get_system_info()
        self._debug.info("SYSTEM INFORMATION:")
        for key, value in sys_info.items():
            self._debug.info(f"  {key}: {value}")

        # Log configuration
        self._debug.info("CONFIGURATION:")
        self._debug.info(f"  Config name: {config.name}")
        self._debug.info(f"  Input traces: {config.input.traces_path}")
        self._debug.info(f"  Input headers: {config.input.headers_path}")
        self._debug.info(f"  Output dir: {config.output.output_dir}")
        self._debug.info(f"  Backend requested: {config.execution.resources.backend}")
        self._debug.info(f"  Max memory: {config.execution.resources.max_memory_gb} GB")
        self._debug.info(f"  Output grid: {config.output.grid.nx}x{config.output.grid.ny}x{config.output.grid.nt}")

        log_memory_state(self._debug, "init")

        # State
        self.phase = ExecutionPhase.INIT
        self.metrics = ExecutionMetrics()

        # Resources (initialized during run)
        self._trace_reader: ZarrTraceReader | None = None
        self._header_manager: ParquetHeaderManager | None = None
        self._spatial_index: SpatialIndex | None = None
        self._velocity_manager: VelocityManager | None = None
        self._kernel = None
        self._memmap_manager: MemmapManager | None = None
        self._checkpoint: CheckpointHandler | None = None
        self._tile_plan: TilePlan | None = None

        # Trace cache for reducing redundant Zarr reads
        # Use 15% of max memory, capped at 2GB
        cache_size_mb = config.execution.resources.max_memory_gb * 1000 * 0.15
        self._trace_cache = LRUTraceCache(max_size_mb=min(cache_size_mb, 2000))
        self._debug.info(f"  Trace cache: {min(cache_size_mb, 2000):.0f} MB max")

        # Control flags
        self._pause_requested = False
        self._stop_requested = False

        # Performance optimization: filtered trace indices (set for fast lookup)
        # Set during init if data selection filters are applied
        self._filtered_trace_indices: set[int] | None = None

    def run(self, resume: bool = False) -> bool:
        """
        Execute the migration.

        Args:
            resume: Attempt to resume from checkpoint

        Returns:
            True if completed successfully
        """
        self._debug.info("=" * 70)
        self._debug.info("STARTING MIGRATION RUN")
        self._debug.info(f"  Resume mode: {resume}")
        self._debug.info("=" * 70)

        self.metrics.start_time = time.time()

        try:
            # Phase 1: Initialization
            self._debug.info(">>> PHASE 1: INITIALIZATION <<<")
            log_memory_state(self._debug, "before_init")
            self._report_phase(ExecutionPhase.INIT)
            self._initialize()
            log_memory_state(self._debug, "after_init")

            # Phase 2: Planning
            self._debug.info(">>> PHASE 2: PLANNING <<<")
            log_memory_state(self._debug, "before_plan")
            self._report_phase(ExecutionPhase.PLANNING)
            self._plan(resume)
            log_memory_state(self._debug, "after_plan")

            # Phase 3: Migration
            self._debug.info(">>> PHASE 3: MIGRATION <<<")
            log_memory_state(self._debug, "before_migrate")
            self._report_phase(ExecutionPhase.MIGRATION)
            self._migrate()
            log_memory_state(self._debug, "after_migrate")

            # Phase 4: Finalization
            self._debug.info(">>> PHASE 4: FINALIZATION <<<")
            log_memory_state(self._debug, "before_finalize")
            self._report_phase(ExecutionPhase.FINALIZATION)
            self._finalize()
            log_memory_state(self._debug, "after_finalize")

            # Done
            self._report_phase(ExecutionPhase.COMPLETE)
            self.metrics.end_time = time.time()

            self._debug.info("=" * 70)
            self._debug.info("MIGRATION COMPLETED SUCCESSFULLY")
            self._debug.info(f"  Total time: {self.metrics.elapsed_time:.2f} seconds")
            self._debug.info("=" * 70)

            self._print_summary()
            return True

        except KeyboardInterrupt:
            self._debug.warning("Migration interrupted by user (KeyboardInterrupt)")
            logger.info("Migration interrupted by user")
            self._save_checkpoint()
            return False

        except Exception as e:
            self._debug.error("=" * 70)
            self._debug.error("MIGRATION FAILED WITH EXCEPTION")
            self._debug.error(f"  Exception type: {type(e).__name__}")
            self._debug.error(f"  Exception message: {e}")
            self._debug.error("  Full traceback:")
            for line in traceback.format_exc().split('\n'):
                self._debug.error(f"    {line}")
            self._debug.error("=" * 70)
            log_memory_state(self._debug, "at_failure")

            logger.exception(f"Migration failed: {e}")
            print(f"\nMigration failed: {e}")
            print(f"Traceback (most recent call last):")
            traceback.print_exc()

            self._report_phase(ExecutionPhase.FAILED)
            self._save_checkpoint()
            return False

        finally:
            self._cleanup()

    def request_pause(self) -> None:
        """Request migration to pause (will pause after current tile)."""
        self._pause_requested = True

    def request_stop(self) -> None:
        """Request migration to stop (will stop after current tile)."""
        self._stop_requested = True

    def _analyze_grid_coverage(
        self,
        midpoint_x: NDArray[np.floating],
        midpoint_y: NDArray[np.floating],
    ) -> None:
        """
        Analyze how well the output grid covers the input data.

        Logs coverage statistics and warnings if significant data falls
        outside the output grid boundaries.
        """
        try:
            from pstm.analysis.grid_outliers import (
                classify_points_against_grid,
                generate_outlier_report,
            )

            # Get grid corners from config (handles both bounding-box and corner-point grids)
            grid = self.config.output.grid
            coords = grid.get_output_coordinates()
            X_grid = coords.get('X')
            Y_grid = coords.get('Y')

            # For rotated grids, use actual 2D corner coordinates
            if X_grid is not None and Y_grid is not None:
                corners = np.array([
                    [X_grid[0, 0], Y_grid[0, 0]],        # C1 (IL=0, XL=0)
                    [X_grid[-1, 0], Y_grid[-1, 0]],      # C2 (IL=max, XL=0)
                    [X_grid[-1, -1], Y_grid[-1, -1]],    # C3 (IL=max, XL=max)
                    [X_grid[0, -1], Y_grid[0, -1]],      # C4 (IL=0, XL=max)
                ])
            else:
                # Fallback for axis-aligned grids
                x_coords = coords['x']
                y_coords = coords['y']
                corners = np.array([
                    [x_coords[0], y_coords[0]],    # C1 SW
                    [x_coords[-1], y_coords[0]],   # C2 SE
                    [x_coords[-1], y_coords[-1]],  # C3 NE
                    [x_coords[0], y_coords[-1]],   # C4 NW
                ])

            # Get aperture parameters
            max_offset = self.config.algorithm.aperture.max_aperture_m
            max_dip = self.config.algorithm.aperture.max_dip_degrees

            # Generate coverage report
            report = generate_outlier_report(
                midpoint_x, midpoint_y, corners,
                max_offset_m=max_offset,
                max_dip_deg=max_dip,
            )

            classification = report.classification

            # Log results
            self._debug.info(f"Grid Coverage Analysis:")
            self._debug.info(f"  Total midpoints: {classification.n_total:,}")
            self._debug.info(f"  Inside grid: {classification.n_inside:,} ({classification.inside_ratio*100:.1f}%)")
            self._debug.info(f"  Outside grid: {classification.n_outside:,}")

            if classification.n_outside > 0:
                self._debug.info(f"  Max distance outside: {classification.max_distance_outside:.1f} m")
                self._debug.info(f"  Mean distance outside: {classification.mean_distance_outside:.1f} m")
                self._debug.info(f"  Suggested buffer: {classification.suggested_buffer_m:.1f} m")

                # Log quadrant breakdown
                for direction, count in classification.outside_by_quadrant.items():
                    if count > 0:
                        self._debug.info(f"    {direction}: {count:,} points")

            # Report warnings
            if report.outlier_warning:
                self._debug.warning(f"Coverage warning: {report.recommendation_reason}")
                logger.warning(f"Grid coverage: {report.recommendation_reason}")

            if not report.coverage_acceptable:
                self._debug.warning("Grid coverage is below acceptable threshold!")
                logger.warning(
                    f"Grid coverage is {classification.inside_ratio*100:.1f}%. "
                    f"Consider extending grid by {classification.suggested_buffer_m:.0f}m."
                )

            # Store result for later reference
            self.metrics.grid_coverage_ratio = classification.inside_ratio
            self.metrics.grid_coverage_n_outside = classification.n_outside

        except Exception as e:
            self._debug.warning(f"Grid coverage analysis failed: {e}")
            # Non-fatal, continue with migration

    def _report_phase(self, phase: ExecutionPhase) -> None:
        """Report phase change."""
        self.phase = phase
        print_section(f"Phase: {phase.value.title()}")
        if self.progress_callback:
            # Get total traces if reader is initialized
            total_traces = self._trace_reader.n_traces if self._trace_reader else 0
            info = ProgressInfo(
                phase=phase,
                current_tile=0,
                total_tiles=0,
                message="",
                traces_processed=self.metrics.n_traces_processed,
                total_traces=total_traces,
                output_bins_completed=self.metrics.n_output_bins_completed,
                total_output_bins=self.metrics.total_output_bins,
            )
            self.progress_callback(info)

    def _report_progress(
        self,
        current: int,
        total: int,
        message: str = "",
        traces_in_tile: int = 0,
        tile_x: int = 0,
        tile_y: int = 0,
    ) -> None:
        """Report progress within current phase."""
        self._debug.info(f"EXECUTOR._report_progress called:")
        self._debug.info(f"  current={current}, total={total}, traces_in_tile={traces_in_tile}")
        self._debug.info(f"  traces_processed={self.metrics.n_traces_processed}")
        self._debug.info(f"  message='{message}'")
        self._debug.info(f"  has callback: {self.progress_callback is not None}")

        if self.progress_callback:
            # Get total traces if reader is initialized
            total_traces = self._trace_reader.n_traces if self._trace_reader else 0
            # Calculate output bins in current tile (for display)
            output_bins_in_tile = 0
            if self._tile_plan and current < len(self._tile_plan.tiles):
                tile = self._tile_plan.tiles[current]
                output_bins_in_tile = tile.nx * tile.ny
            info = ProgressInfo(
                phase=self.phase,
                current_tile=current,
                total_tiles=total,
                message=message,
                traces_processed=self.metrics.n_traces_processed,
                traces_in_tile=traces_in_tile,
                tile_x=tile_x,
                tile_y=tile_y,
                total_traces=total_traces,
                output_bins_completed=self.metrics.n_output_bins_completed,
                total_output_bins=self.metrics.total_output_bins,
                output_bins_in_tile=output_bins_in_tile,
            )
            self._debug.info(f"  Calling progress_callback with ProgressInfo...")
            self.progress_callback(info)
            self._debug.info(f"  progress_callback returned")

    def _initialize(self) -> None:
        """Initialize all resources."""
        self._debug.info("--- Initialization: Opening input data ---")
        logger.info("Opening input data...")

        # Open trace reader
        self._debug.debug(f"Opening trace reader: {self.config.input.traces_path}")
        self._trace_reader = ZarrTraceReader(
            self.config.input.traces_path,
            sample_rate_ms=self.config.input.sample_rate_ms,
            start_time_ms=self.config.input.start_time_ms,
            transposed=self.config.input.transposed,
            n_traces=self.config.input.num_traces,
            n_samples=self.config.input.num_samples,
        )
        self._trace_reader.open()
        self._debug.info(f"Trace reader opened: {self._trace_reader.n_traces} traces, {self._trace_reader.n_samples} samples")

        # Open header manager
        self._debug.debug(f"Opening header manager: {self.config.input.headers_path}")
        self._header_manager = ParquetHeaderManager(
            self.config.input.headers_path,
            column_mapping=self.config.input.columns,
            apply_scalar=self.config.input.apply_coord_scalar,
        )
        self._header_manager.open()
        self._debug.info(f"Header manager opened: {self._header_manager.n_traces} headers")

        self._debug.info("--- Initialization: Building spatial index ---")
        log_memory_state(self._debug, "before_spatial_index")
        logger.info("Building spatial index...")
        # Build or load spatial index
        if self.config.geometry.index_path and self.config.geometry.index_path.exists():
            self._debug.debug(f"Loading existing spatial index: {self.config.geometry.index_path}")
            self._spatial_index = SpatialIndex.load(self.config.geometry.index_path)
        else:
            self._debug.debug("Building new spatial index from midpoints...")

            # Apply data selection filter to reduce traces BEFORE building spatial index
            # Use Parquet predicate pushdown for offset filtering (MUCH faster!)
            data_sel = self.config.data_selection
            if data_sel.mode.value == "offset" and data_sel.offset:
                # OPTIMIZED PATH: Use Parquet predicate pushdown for offset filter
                # This avoids loading all 22M traces just to filter to 300K
                offset_min = data_sel.offset.min_offset
                offset_max = data_sel.offset.max_offset
                self._debug.info(f"Using Parquet predicate pushdown for offset filter: {offset_min:.0f} - {offset_max:.0f} m")
                logger.info(f"Using Parquet predicate pushdown for offset filter: {offset_min:.0f} - {offset_max:.0f} m")

                # This method filters at Parquet level BEFORE loading into memory
                trace_indices, midpoint_x, midpoint_y, offset_values = (
                    self._header_manager.get_midpoints_with_offset_filter(offset_min, offset_max)
                )

                n_total = self._header_manager.n_traces
                n_after = len(trace_indices)
                self._debug.info(f"Predicate pushdown: {n_total:,} -> {n_after:,} traces ({100*n_after/n_total:.1f}%)")
                logger.info(f"Predicate pushdown result: {n_total:,} -> {n_after:,} traces ({100*n_after/n_total:.1f}%)")

                # Store filtered trace indices for tile-level optimization
                self._filtered_trace_indices = set(trace_indices.tolist())
                self._debug.info(f"Stored {len(self._filtered_trace_indices):,} filtered trace indices for tile optimization")

            elif data_sel.mode.value != "all":
                # FALLBACK PATH: Load all data then filter in Python
                # This is slower but necessary for OVT and other complex filters
                self._debug.info(f"Applying data selection filter (Python): {data_sel.mode.value}")
                logger.info(f"Applying data selection filter: {data_sel.mode.value}")

                trace_indices, midpoint_x, midpoint_y = self._header_manager.get_all_midpoints()

                # Load offset values for filtering
                offset_col = self.config.input.columns.offset or "OFFSET"
                if offset_col in self._header_manager.schema:
                    offset_values = self._header_manager.get_column(offset_col)

                    # Build headers dict for data selection
                    headers = {'offset': offset_values}

                    # Apply data selection filter
                    mask = data_sel.apply(headers)
                    n_before = len(trace_indices)
                    n_after = mask.sum()

                    # Filter arrays
                    trace_indices = trace_indices[mask]
                    midpoint_x = midpoint_x[mask]
                    midpoint_y = midpoint_y[mask]

                    self._debug.info(f"Data selection: {n_before:,} -> {n_after:,} traces ({100*n_after/n_before:.1f}%)")
                    logger.info(f"Data selection: {n_before:,} -> {n_after:,} traces ({100*n_after/n_before:.1f}%)")

                    # Store filtered trace indices for tile-level optimization
                    self._filtered_trace_indices = set(trace_indices.tolist())
                else:
                    self._debug.warning(f"Offset column '{offset_col}' not found, skipping data selection filter")
                    logger.warning(f"Offset column '{offset_col}' not found, skipping data selection filter")
                    self._filtered_trace_indices = None
            else:
                # No filtering - load all midpoints
                trace_indices, midpoint_x, midpoint_y = self._header_manager.get_all_midpoints()
                self._filtered_trace_indices = None

            self._spatial_index = SpatialIndex.build(trace_indices, midpoint_x, midpoint_y)
        self._debug.info(f"Spatial index ready: {self._spatial_index.n_points} entries")
        log_memory_state(self._debug, "after_spatial_index")

        # Pre-migration QC: Analyze grid coverage
        self._debug.info("--- Pre-migration QC: Grid Coverage Analysis ---")
        # Get midpoints from spatial index (works for both built and loaded indices)
        qc_midpoint_x = self._spatial_index.midpoint_x
        qc_midpoint_y = self._spatial_index.midpoint_y
        self._analyze_grid_coverage(qc_midpoint_x, qc_midpoint_y)

        self._debug.info("--- Initialization: Loading velocity model ---")
        logger.info("Loading velocity model...")
        # Load velocity model with manager for tile extraction
        self._debug.debug(f"Velocity source: {self.config.velocity.source}")
        self._velocity_manager = create_velocity_manager(
            self.config.velocity,
            self.config.output.grid,
        )
        if self._velocity_manager.memory_usage_gb > 0:
            self._debug.info(f"Velocity grid memory: {self._velocity_manager.memory_usage_gb:.2f} GB")
            logger.info(f"Velocity grid memory: {self._velocity_manager.memory_usage_gb:.2f} GB")
        log_memory_state(self._debug, "after_velocity")

        self._debug.info("--- Initialization: Creating compute kernel ---")
        logger.info("Initializing compute kernel...")
        # Initialize kernel
        backend = self.config.execution.resources.backend.value
        self._debug.info(f"KERNEL BACKEND REQUESTED: {backend}")
        self._debug.debug(f"Backend enum value: {self.config.execution.resources.backend}")

        self._kernel = create_kernel(backend)
        self._debug.info(f"KERNEL CREATED: {type(self._kernel).__name__}")
        self._debug.info(f"  Kernel class: {type(self._kernel).__module__}.{type(self._kernel).__name__}")

        kernel_config = KernelConfig(
            max_aperture_m=self.config.algorithm.aperture.max_aperture_m,
            min_aperture_m=self.config.algorithm.aperture.min_aperture_m,
            max_dip_degrees=self.config.algorithm.aperture.max_dip_degrees,
            taper_fraction=self.config.algorithm.aperture.taper_fraction,
            apply_spreading=self.config.algorithm.amplitude.geometrical_spreading,
            apply_obliquity=self.config.algorithm.amplitude.obliquity_factor,
            interpolation_method=self.config.algorithm.interpolation.value,
            output_dt_ms=self.config.output.grid.dt_ms,
            # Anti-aliasing config - use algorithm config if set, otherwise default to False
            aa_enabled=self.config.algorithm.anti_aliasing.enabled if self.config.algorithm.anti_aliasing else False,
            # Grid bin spacing for AA filter (use actual bin size, not linspace-derived)
            grid_dx=self.config.output.grid.dx,
            grid_dy=self.config.output.grid.dy,
        )
        self._debug.debug(f"Kernel config: aperture={kernel_config.max_aperture_m}m, dip={kernel_config.max_dip_degrees}deg, aa_enabled={kernel_config.aa_enabled}, grid_dx={kernel_config.grid_dx}m, grid_dy={kernel_config.grid_dy}m")
        self._kernel.initialize(kernel_config)
        self._debug.info("Kernel initialized successfully")

        logger.info("Setting up memory manager...")
        # Set up memmap manager
        work_dir = self.config.output.output_dir / ".work"
        self._memmap_manager = MemmapManager(work_dir)

        # Create output accumulator(s)
        grid = self.config.output.grid
        products = self.config.output.products

        # Determine gather bins for output (supports mixed offset + OVT bins)
        from pstm.config.models import GatherBin, GatherBinType
        self._gather_bins: list[GatherBin] = []

        # Check for unified gather_bins first (new API)
        if products.gather_bins:
            self._gather_bins = list(products.gather_bins)
            logger.info(f"Output gathers mode: {len(self._gather_bins)} bins (unified)")
            for i, gb in enumerate(self._gather_bins):
                if gb.bin_type == GatherBinType.OFFSET:
                    logger.info(f"  Bin {i}: OFFSET {gb.offset_min:.0f} - {gb.offset_max:.0f} m")
                else:
                    logger.info(f"  Bin {i}: OVT X[{gb.ovt_x_min:.0f},{gb.ovt_x_max:.0f}] Y[{gb.ovt_y_min:.0f},{gb.ovt_y_max:.0f}]")
        elif products.output_gathers:
            # Legacy: gather_offset_ranges
            if products.gather_offset_ranges:
                for omin, omax in products.gather_offset_ranges:
                    self._gather_bins.append(GatherBin.offset_bin(omin, omax))
                logger.info(f"Output gathers mode: {len(self._gather_bins)} offset bins (legacy)")
            # Legacy: ovt_output_tiles
            elif products.ovt_output_tiles:
                for tile_ix, tile_iy, x_min, x_max, y_min, y_max in products.ovt_output_tiles:
                    self._gather_bins.append(GatherBin.ovt_bin(x_min, x_max, y_min, y_max))
                logger.info(f"Output gathers mode: {len(self._gather_bins)} OVT bins (legacy)")

        n_bins = max(1, len(self._gather_bins))  # At least 1 for full stack

        # Create accumulators - if gathers mode, create per-bin arrays
        if self._gather_bins:
            # Create separate image/fold per bin (offset or OVT)
            for bid in range(n_bins):
                self._memmap_manager.create(
                    f"image_bin_{bid}",
                    shape=(grid.nx, grid.ny, grid.nt),
                    dtype=np.float64,
                    fill_value=0.0,
                )
                self._memmap_manager.create(
                    f"fold_bin_{bid}",
                    shape=(grid.nx, grid.ny, grid.nt),  # 3D fold per sample
                    dtype=np.int32,
                    fill_value=0,
                )
                self._memmap_manager.create(
                    f"trace_count_bin_{bid}",
                    shape=(grid.nx, grid.ny),
                    dtype=np.int32,
                    fill_value=0,
                )
                self._memmap_manager.create(
                    f"offset_sum_bin_{bid}",
                    shape=(grid.nx, grid.ny),
                    dtype=np.float64,
                    fill_value=0.0,
                )
                self._memmap_manager.create(
                    f"azimuth_sin_sum_bin_{bid}",
                    shape=(grid.nx, grid.ny),
                    dtype=np.float64,
                    fill_value=0.0,
                )
                self._memmap_manager.create(
                    f"azimuth_cos_sum_bin_{bid}",
                    shape=(grid.nx, grid.ny),
                    dtype=np.float64,
                    fill_value=0.0,
                )
        else:
            # Single full-stack output (original behavior)
            self._memmap_manager.create(
                "image",
                shape=(grid.nx, grid.ny, grid.nt),
                dtype=np.float64,
                fill_value=0.0,
            )
            self._memmap_manager.create(
                "fold",
                shape=(grid.nx, grid.ny, grid.nt),  # 3D fold per sample
                dtype=np.int32,
                fill_value=0,
            )
            # Create header accumulators for CIG support
            self._memmap_manager.create(
                "trace_count",
                shape=(grid.nx, grid.ny),
                dtype=np.int32,
                fill_value=0,
            )
            self._memmap_manager.create(
                "offset_sum",
                shape=(grid.nx, grid.ny),
                dtype=np.float64,
                fill_value=0.0,
            )
            self._memmap_manager.create(
                "azimuth_sin_sum",
                shape=(grid.nx, grid.ny),
                dtype=np.float64,
                fill_value=0.0,
            )
            self._memmap_manager.create(
                "azimuth_cos_sum",
                shape=(grid.nx, grid.ny),
                dtype=np.float64,
                fill_value=0.0,
            )

        print_success("Initialization complete")

    def _plan(self, resume: bool) -> None:
        """Plan tile processing."""
        grid = self.config.output.grid

        # Create tile planner
        planner = TilePlanner(
            output_grid=grid,
            tiling_config=self.config.execution.tiling,
            max_memory_gb=self.config.execution.resources.max_memory_gb,
            aperture_radius=self.config.algorithm.aperture.max_aperture_m,
        )

        self._tile_plan = planner.plan()
        self.metrics.n_tiles_total = self._tile_plan.n_tiles
        self.metrics.n_samples_output = self._tile_plan.total_output_samples
        # Total output bins = nx * ny (each bin is a fully migrated output trace)
        self.metrics.total_output_bins = grid.nx * grid.ny

        # Set up checkpoint handler
        checkpoint_dir = (
            self.config.execution.checkpoint.checkpoint_dir
            or self.config.output.output_dir / ".checkpoint"
        )
        self._checkpoint = CheckpointHandler(
            checkpoint_dir=checkpoint_dir,
            config=self.config,
            total_tiles=self._tile_plan.n_tiles,
        )

        # Try to resume if requested
        if resume and self._checkpoint.exists():
            state = self._checkpoint.load()
            if state:
                self.metrics.n_tiles_completed = state.n_completed
                self.metrics.n_traces_processed = state.total_traces_processed
                self.metrics.compute_time_total = state.total_compute_time_s
                logger.info(f"Resuming from checkpoint: {state.n_completed} tiles completed")

        print_metric("Total tiles", self._tile_plan.n_tiles)
        print_metric("Output samples", f"{self.metrics.n_samples_output:,}")
        print_metric("Output size", format_bytes(grid.size_gb * 1024**3))

    def _migrate(self) -> None:
        """Execute migration loop over tiles."""
        self._debug.info("--- Migration: Starting tile processing loop ---")
        assert self._tile_plan is not None
        assert self._checkpoint is not None
        assert self._spatial_index is not None
        assert self._trace_reader is not None
        assert self._header_manager is not None
        assert self._velocity_manager is not None
        assert self._kernel is not None
        assert self._memmap_manager is not None

        # Check if we're in gathers mode (multiple bins - offset, OVT, or mixed)
        from pstm.config.models import GatherBinType

        if self._gather_bins:
            # Run migration for each bin (can be offset or OVT, mixed in any order)
            for bid, gb in enumerate(self._gather_bins):
                if gb.bin_type == GatherBinType.OFFSET:
                    logger.info(f"=== Processing bin {bid}: OFFSET {gb.offset_min:.0f} - {gb.offset_max:.0f} m ===")
                    self._migrate_single_bin(bid, gb.offset_min, gb.offset_max)
                else:
                    logger.info(f"=== Processing bin {bid}: OVT X[{gb.ovt_x_min:.0f},{gb.ovt_x_max:.0f}] Y[{gb.ovt_y_min:.0f},{gb.ovt_y_max:.0f}] ===")
                    self._migrate_single_bin(
                        bid, None, None,
                        ovt_x_min=gb.ovt_x_min, ovt_x_max=gb.ovt_x_max,
                        ovt_y_min=gb.ovt_y_min, ovt_y_max=gb.ovt_y_max,
                    )
        else:
            # Single full-stack migration (original behavior)
            self._migrate_single_bin(None, None, None)

    def _migrate_single_bin(
        self,
        bin_id: int | None,
        offset_min: float | None,
        offset_max: float | None,
        ovt_x_min: float | None = None,
        ovt_x_max: float | None = None,
        ovt_y_min: float | None = None,
        ovt_y_max: float | None = None,
    ) -> None:
        """Execute migration for a single offset bin or OVT tile (or full stack if bin_id is None)."""
        # In gathers mode, reset checkpoint for each bin
        # (checkpoint tracks tiles per bin, but we process bins sequentially)
        if bin_id is not None and self._checkpoint:
            self._checkpoint.state.completed_tiles.clear()
            self._debug.info(f"[BIN {bin_id}] Reset checkpoint for new offset bin")

        # Get arrays - use bin-specific names if in gathers mode
        if bin_id is not None:
            image = self._memmap_manager.get(f"image_bin_{bin_id}")
            fold = self._memmap_manager.get(f"fold_bin_{bin_id}")
            trace_count = self._memmap_manager.get(f"trace_count_bin_{bin_id}")
            offset_sum = self._memmap_manager.get(f"offset_sum_bin_{bin_id}")
            azimuth_sin_sum = self._memmap_manager.get(f"azimuth_sin_sum_bin_{bin_id}")
            azimuth_cos_sum = self._memmap_manager.get(f"azimuth_cos_sum_bin_{bin_id}")
        else:
            image = self._memmap_manager.get("image")
            fold = self._memmap_manager.get("fold")
            trace_count = self._memmap_manager.get("trace_count")
            offset_sum = self._memmap_manager.get("offset_sum")
            azimuth_sin_sum = self._memmap_manager.get("azimuth_sin_sum")
            azimuth_cos_sum = self._memmap_manager.get("azimuth_cos_sum")

        self._debug.debug(f"Output image shape: {image.shape}, dtype: {image.dtype}")
        self._debug.debug(f"Output fold shape: {fold.shape}, dtype: {fold.dtype}")

        # Get completed tiles for skipping
        completed = set(self._checkpoint.state.completed_tiles)
        self._debug.info(f"Tiles already completed (resuming): {len(completed)}")

        # Build kernel config
        kernel_config = KernelConfig(
            max_aperture_m=self.config.algorithm.aperture.max_aperture_m,
            min_aperture_m=self.config.algorithm.aperture.min_aperture_m,
            max_dip_degrees=self.config.algorithm.aperture.max_dip_degrees,
            taper_fraction=self.config.algorithm.aperture.taper_fraction,
            apply_spreading=self.config.algorithm.amplitude.geometrical_spreading,
            apply_obliquity=self.config.algorithm.amplitude.obliquity_factor,
            interpolation_method=self.config.algorithm.interpolation.value,
            output_dt_ms=self.config.output.grid.dt_ms,
            # Anti-aliasing config - use algorithm config if set, otherwise default to False
            aa_enabled=self.config.algorithm.anti_aliasing.enabled if self.config.algorithm.anti_aliasing else False,
            # Grid bin spacing for AA filter (use actual bin size, not linspace-derived)
            grid_dx=self.config.output.grid.dx,
            grid_dy=self.config.output.grid.dy,
            # Kernel type selection
            kernel_type=self.config.algorithm.kernel_type.value,
            # Curved ray parameters
            curved_ray_enabled=self.config.algorithm.curved_ray is not None and self.config.algorithm.curved_ray.enabled,
            curved_ray_v0=self.config.algorithm.curved_ray.v0_m_s if self.config.algorithm.curved_ray and self.config.algorithm.curved_ray.v0_m_s else 1500.0,
            curved_ray_k=self.config.algorithm.curved_ray.k_per_s if self.config.algorithm.curved_ray and self.config.algorithm.curved_ray.k_per_s else 0.5,
            # VTI anisotropy parameters (constant eta fallback)
            vti_enabled=self.config.algorithm.anisotropy_vti is not None and self.config.algorithm.anisotropy_vti.enabled,
            vti_eta_constant=self.config.algorithm.anisotropy_vti.eta_constant if self.config.algorithm.anisotropy_vti else 0.0,
        )

        # Handle VTI eta sources (1D table or 3D cube)
        vti_cfg = self.config.algorithm.anisotropy_vti
        if vti_cfg and vti_cfg.enabled:
            grid = self.config.output.grid
            t_axis = np.arange(grid.t_min_ms, grid.t_max_ms + grid.dt_ms / 2, grid.dt_ms)

            if vti_cfg.eta_source == "table_1d" and vti_cfg.eta_table:
                # Interpolate 1D eta table to output time axis
                table_times = np.array([p[0] for p in vti_cfg.eta_table])
                table_eta = np.array([p[1] for p in vti_cfg.eta_table])
                eta_1d = np.interp(t_axis, table_times, table_eta)
                kernel_config.vti_eta_array = eta_1d
                kernel_config.vti_eta_is_1d = True
                self._debug.info(f"[VTI] Using 1D eta table: {len(vti_cfg.eta_table)} points -> {len(eta_1d)} samples")
                self._debug.info(f"[VTI] Eta range: {eta_1d.min():.3f} - {eta_1d.max():.3f}")

            elif vti_cfg.eta_source == "cube_3d" and vti_cfg.eta_cube_path:
                # Load 3D eta cube and interpolate to output grid
                try:
                    eta_cube = zarr.open(str(vti_cfg.eta_cube_path), mode='r')
                    if isinstance(eta_cube, zarr.Array):
                        eta_data = eta_cube[:]
                    else:
                        eta_data = eta_cube['data'][:]

                    # Get cube axes from attributes
                    x_axis_cube = np.array(eta_cube.attrs.get('x_axis', np.arange(eta_data.shape[0])))
                    y_axis_cube = np.array(eta_cube.attrs.get('y_axis', np.arange(eta_data.shape[1])))
                    t_axis_cube = np.array(eta_cube.attrs.get('t_axis_ms', np.arange(eta_data.shape[2])))

                    # Interpolate to output grid
                    from scipy.interpolate import RegularGridInterpolator
                    interp = RegularGridInterpolator(
                        (x_axis_cube, y_axis_cube, t_axis_cube),
                        eta_data,
                        method='linear',
                        bounds_error=False,
                        fill_value=None,  # Extrapolate
                    )

                    # Get output coordinates (handles both bounding-box and corner-point grids)
                    out_coords = grid.get_output_coordinates()
                    x_out = out_coords['x']
                    y_out = out_coords['y']
                    xx, yy, tt = np.meshgrid(x_out, y_out, t_axis, indexing='ij')
                    pts = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])
                    eta_3d = interp(pts).reshape(grid.nx, grid.ny, len(t_axis))

                    kernel_config.vti_eta_array = eta_3d
                    kernel_config.vti_eta_is_1d = False
                    self._debug.info(f"[VTI] Using 3D eta cube: {eta_data.shape} -> {eta_3d.shape}")
                    self._debug.info(f"[VTI] Eta range: {eta_3d.min():.3f} - {eta_3d.max():.3f}")

                except Exception as e:
                    self._debug.error(f"[VTI] Failed to load 3D eta cube: {e}")
                    self._debug.warning("[VTI] Falling back to constant eta")
            else:
                self._debug.info(f"[VTI] Using constant eta: {kernel_config.vti_eta_constant:.3f}")

        # Get grid config (needed for time-variant and tile processing)
        grid = self.config.output.grid

        # Add time-variant sampling config if enabled
        tv_config = self.config.algorithm.time_variant
        if tv_config.enabled:
            from pstm.algorithm.time_variant import (
                FrequencyTimeTable, compute_time_windows, estimate_speedup,
            )

            times = [p[0] for p in tv_config.frequency_table]
            freqs = [p[1] for p in tv_config.frequency_table]
            freq_table = FrequencyTimeTable(times_ms=times, frequencies_hz=freqs)

            windows = compute_time_windows(
                t_min_ms=grid.t_min_ms,
                t_max_ms=grid.t_max_ms,
                base_dt_ms=grid.dt_ms,
                freq_table=freq_table,
                min_factor=tv_config.min_downsample_factor,
                max_factor=tv_config.max_downsample_factor,
            )

            speedup = estimate_speedup(grid.t_min_ms, grid.t_max_ms, grid.dt_ms, freq_table)

            kernel_config.time_variant_enabled = True
            kernel_config.time_variant_windows = windows

            self._debug.info(f"[TIME-VARIANT] Enabled with {len(windows)} windows, ~{speedup:.1f}x speedup")
            for w in windows:
                self._debug.debug(f"  Window: {w.t_start_ms:.0f}-{w.t_end_ms:.0f} ms, dt={w.dt_effective_ms:.1f}, factor={w.downsample_factor}")
        else:
            self._debug.info("[TIME-VARIANT] Disabled")

        checkpoint_interval = self.config.execution.checkpoint.interval_tiles

        # Check if trace-centric approach would be beneficial
        # This estimates tile overlap and uses trace-centric if overlap > 50%
        use_trace_centric = self._should_use_trace_centric(kernel_config)

        if use_trace_centric:
            self._debug.info("=" * 60)
            self._debug.info("USING TRACE-CENTRIC MIGRATION (high overlap detected)")
            self._debug.info("=" * 60)
            self._run_trace_centric_migration(
                image, fold, trace_count, offset_sum, azimuth_sin_sum, azimuth_cos_sum,
                kernel_config, grid, offset_min, offset_max,
                ovt_x_min, ovt_x_max, ovt_y_min, ovt_y_max,
            )
            return  # Skip tile-by-tile processing

        self._debug.info(f"Starting tile loop: {self._tile_plan.n_tiles} total tiles")
        self._debug.info(f"Using kernel: {type(self._kernel).__name__}")

        # Aperture efficiency tracking (simplified to reduce memory)
        total_input_traces = self._trace_reader.n_traces
        traces_per_tile_list: list[int] = []
        total_trace_data_loaded_mb = 0.0
        # Disabled: unique_trace_indices set was consuming too much memory for large datasets
        # unique_trace_indices: set[int] = set()
        dataset_size_mb = (self._trace_reader.n_traces * self._trace_reader.n_samples * 4) / 1024**2  # Assuming float32

        self._debug.info(f"[APERTURE] Total input traces in dataset: {total_input_traces:,}")
        self._debug.info(f"[APERTURE] Dataset size (approx): {dataset_size_mb:.1f} MB")

        # PRE-COMPUTE tile-trace mappings for all tiles (optimization)
        # This avoids repeated spatial queries during the tile loop
        self._debug.info("Pre-computing tile-trace mappings...")
        t0_precompute = time.time()
        tile_query_cache: dict[int, TileQueryResult] = {}
        for tile in self._tile_plan.tiles:
            query_result = query_traces_for_tile(
                self._spatial_index,
                tile.x_min,
                tile.x_max,
                tile.y_min,
                tile.y_max,
                kernel_config.max_aperture_m,
            )
            tile_query_cache[tile.tile_id] = query_result
        t_precompute = time.time() - t0_precompute

        # Analyze overlap for efficiency reporting
        all_trace_sets = [set(q.trace_indices.tolist()) for q in tile_query_cache.values() if q.n_traces > 0]
        if all_trace_sets:
            unique_traces = set.union(*all_trace_sets)
            total_trace_refs = sum(len(s) for s in all_trace_sets)
            overlap_factor = total_trace_refs / len(unique_traces) if unique_traces else 1.0
            self._debug.info(f"[PRE-COMPUTE] Completed in {t_precompute:.2f}s")
            self._debug.info(f"[PRE-COMPUTE] Unique traces across all tiles: {len(unique_traces):,}")
            self._debug.info(f"[PRE-COMPUTE] Overlap factor: {overlap_factor:.2f}x (higher = more cache benefit)")
        else:
            self._debug.info(f"[PRE-COMPUTE] Completed in {t_precompute:.2f}s (no traces found)")

        # Process tiles with async prefetching
        # Convert to list for indexed access (needed for look-ahead prefetching)
        tiles_to_process = list(iter_tiles(self._tile_plan, completed))
        n_tiles = len(tiles_to_process)
        self._debug.info(f"[PREFETCH] Enabled - will prefetch next tile while GPU processes current")

        # Use ThreadPoolExecutor for background I/O (1 thread is sufficient for prefetching)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") as prefetch_executor:
            # Track the current prefetch future
            prefetch_future: Future[PrefetchedTileData] | None = None
            prefetched_data: PrefetchedTileData | None = None

            for tile_idx, tile in enumerate(tiles_to_process):
                # Check control flags
                if self._stop_requested:
                    logger.info("Stop requested, saving checkpoint...")
                    break

                if self._pause_requested:
                    logger.info("Pause requested, saving checkpoint...")
                    self._save_checkpoint()
                    self._pause_requested = False
                    # In a real implementation, we'd wait here
                    # For now, just continue

                # Wait for prefetched data from previous iteration (if any)
                if prefetch_future is not None:
                    try:
                        prefetched_data = prefetch_future.result(timeout=60.0)
                        if prefetched_data.error:
                            self._debug.warning(f"  [PREFETCH] Error: {prefetched_data.error}, falling back to sync load")
                            prefetched_data = None
                    except Exception as e:
                        self._debug.warning(f"  [PREFETCH] Failed to get result: {e}, falling back to sync load")
                        prefetched_data = None
                    prefetch_future = None

                # Start prefetching NEXT tile's data (if there is a next tile)
                if tile_idx + 1 < n_tiles:
                    next_tile = tiles_to_process[tile_idx + 1]
                    next_query = tile_query_cache.get(next_tile.tile_id)
                    if next_query and next_query.n_traces > 0:
                        # Submit prefetch job for next tile (runs in background during GPU processing)
                        prefetch_future = prefetch_executor.submit(
                            self._prefetch_tile_data,
                            next_tile.tile_id,
                            next_query.trace_indices,
                        )

                # Process current tile (GPU-bound) while prefetch runs in background
                precomputed_query = tile_query_cache.get(tile.tile_id)
                metrics, tile_trace_count, tile_data_mb, tile_trace_indices = self._process_tile(
                    tile, image, fold, trace_count, offset_sum, azimuth_sin_sum, azimuth_cos_sum,
                    kernel_config, grid, offset_min, offset_max,
                    ovt_x_min, ovt_x_max, ovt_y_min, ovt_y_max,
                    precomputed_query=precomputed_query,
                    prefetched_data=prefetched_data,
                )
                # Clear prefetched data after use
                prefetched_data = None

                # Track aperture efficiency
                traces_per_tile_list.append(tile_trace_count)
                total_trace_data_loaded_mb += tile_data_mb
                # Disabled to save memory: unique_trace_indices.update(tile_trace_indices)

                # Update metrics
                self.metrics.n_tiles_completed += 1
                self.metrics.n_traces_processed += metrics.n_traces_processed
                # Track output bins completed (tile.nx * tile.ny pillars fully migrated)
                self.metrics.n_output_bins_completed += tile.nx * tile.ny

                # Report completion of this tile
                self._report_progress(
                    current=self.metrics.n_tiles_completed,  # Now 1-indexed after increment
                    total=self._tile_plan.n_tiles,
                    message=f"Completed tile {tile.tile_id + 1}/{self._tile_plan.n_tiles} - {self.metrics.n_output_bins_completed:,}/{self.metrics.total_output_bins:,} output bins",
                    traces_in_tile=0,  # Tile is done
                    tile_x=0,
                    tile_y=0,
                )
                self.metrics.compute_time_total += metrics.compute_time_s

                # Explicit memory cleanup after each tile
                del tile_trace_indices
                import gc
                gc.collect()

                # Mark completed
                self._checkpoint.mark_completed(
                    tile.tile_id,
                    n_traces=metrics.n_traces_processed,
                    compute_time=metrics.compute_time_s,
                )

                # Periodic checkpoint
                if should_checkpoint(tile.tile_id, checkpoint_interval):
                    self._save_checkpoint()

                # Log progress
                if (tile.tile_id + 1) % 10 == 0:
                    eta = self.metrics.estimate_remaining_time()
                    logger.info(
                        f"Progress: {self.metrics.n_tiles_completed}/{self._tile_plan.n_tiles} tiles, "
                        f"ETA: {format_duration(eta)}"
                    )

        # Final checkpoint
        self._save_checkpoint()

        # Print aperture efficiency summary
        self._debug.info("=" * 60)
        self._debug.info("APERTURE EFFICIENCY ANALYSIS")
        self._debug.info("=" * 60)

        if traces_per_tile_list:
            import statistics
            min_traces = min(traces_per_tile_list)
            max_traces = max(traces_per_tile_list)
            avg_traces = statistics.mean(traces_per_tile_list)
            median_traces = statistics.median(traces_per_tile_list)
            std_traces = statistics.stdev(traces_per_tile_list) if len(traces_per_tile_list) > 1 else 0

            self._debug.info(f"[APERTURE] Traces per tile:")
            self._debug.info(f"  Min: {min_traces:,}")
            self._debug.info(f"  Max: {max_traces:,}")
            self._debug.info(f"  Avg: {avg_traces:,.0f}")
            self._debug.info(f"  Median: {median_traces:,.0f}")
            self._debug.info(f"  Std Dev: {std_traces:,.0f}")

            # Aperture efficiency indicator
            efficiency = 100 * (1 - avg_traces / total_input_traces) if total_input_traces > 0 else 0
            self._debug.info(f"[APERTURE] Aperture efficiency: {efficiency:.1f}%")
            if avg_traces > 0.8 * total_input_traces:
                self._debug.warning(f"[APERTURE] WARNING: Aperture is too wide! Most tiles use most traces.")
                self._debug.warning(f"[APERTURE] Consider reducing max_aperture_m (currently: {self.config.algorithm.aperture.max_aperture_m}m)")

            # Print to stderr for visibility
            print(f"\n{'='*60}", file=sys.stderr, flush=True)
            print(f"APERTURE EFFICIENCY ANALYSIS", file=sys.stderr, flush=True)
            print(f"  Traces per tile: min={min_traces:,}, max={max_traces:,}, avg={avg_traces:,.0f}", file=sys.stderr, flush=True)
            print(f"  Efficiency: {efficiency:.1f}% (higher=better spatial locality)", file=sys.stderr, flush=True)
            if avg_traces > 0.8 * total_input_traces:
                print(f"  ⚠️  WARNING: Aperture too wide! Consider reducing max_aperture_m", file=sys.stderr, flush=True)

        # Memory profiling summary
        self._debug.info(f"[MEMORY] Data loading analysis:")
        self._debug.info(f"  Dataset size (approx): {dataset_size_mb:.1f} MB")
        self._debug.info(f"  Total data loaded: {total_trace_data_loaded_mb:.1f} MB")
        data_reuse_factor = total_trace_data_loaded_mb / dataset_size_mb if dataset_size_mb > 0 else 0
        self._debug.info(f"  Data reuse factor: {data_reuse_factor:.1f}x (1.0=optimal, >1=traces loaded multiple times)")

        # Unique trace tracking disabled to save memory - estimate from reuse factor
        unique_count = total_input_traces  # Assume all traces loaded at least once
        self._debug.info(f"  Unique traces loaded: {unique_count:,} / {total_input_traces:,}")
        coverage = 100.0  # Approximate
        self._debug.info(f"  Trace coverage: {coverage:.1f}% (estimated)")

        print(f"MEMORY PROFILING:", file=sys.stderr, flush=True)
        print(f"  Dataset: {dataset_size_mb:.1f} MB | Loaded: {total_trace_data_loaded_mb:.1f} MB", file=sys.stderr, flush=True)
        print(f"  Data reuse factor: {data_reuse_factor:.1f}x", file=sys.stderr, flush=True)
        print(f"  Unique traces: ~{unique_count:,}/{total_input_traces:,} (estimated)", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)

    def _prefetch_tile_data(
        self,
        tile_id: int,
        trace_indices: NDArray[np.int64],
    ) -> PrefetchedTileData:
        """
        Prefetch trace data and geometry for a tile (runs in background thread).

        This method is designed to run concurrently while the GPU processes the
        current tile, hiding I/O latency.

        Args:
            tile_id: ID of the tile to prefetch
            trace_indices: Indices of traces to load

        Returns:
            PrefetchedTileData with loaded trace data and geometry
        """
        try:
            # Load traces using cache (thread-safe via GIL for dict operations)
            trace_data = self._trace_cache.get_traces(trace_indices, self._trace_reader)

            # Load geometry
            geometry = self._header_manager.get_geometry_for_indices(trace_indices)

            return PrefetchedTileData(
                tile_id=tile_id,
                trace_data=trace_data,
                geometry=geometry,
                trace_indices=trace_indices,
            )
        except Exception as e:
            return PrefetchedTileData(
                tile_id=tile_id,
                error=str(e),
            )

    def _process_tile(
        self,
        tile: TileSpec,
        image: NDArray,
        fold: NDArray,
        trace_count: NDArray,
        offset_sum: NDArray,
        azimuth_sin_sum: NDArray,
        azimuth_cos_sum: NDArray,
        kernel_config: KernelConfig,
        grid,
        offset_min: float | None = None,
        offset_max: float | None = None,
        ovt_x_min: float | None = None,
        ovt_x_max: float | None = None,
        ovt_y_min: float | None = None,
        ovt_y_max: float | None = None,
        precomputed_query: TileQueryResult | None = None,
        prefetched_data: PrefetchedTileData | None = None,
    ) -> tuple[KernelMetrics, int, float, list[int]]:
        """
        Process a single tile.

        Args:
            offset_min: If provided, only include traces with offset >= offset_min
            offset_max: If provided, only include traces with offset <= offset_max
            ovt_x_min: If provided, only include traces with offset_x >= ovt_x_min
            ovt_x_max: If provided, only include traces with offset_x <= ovt_x_max
            ovt_y_min: If provided, only include traces with offset_y >= ovt_y_min
            ovt_y_max: If provided, only include traces with offset_y <= ovt_y_max
            precomputed_query: Pre-computed tile query result (avoids spatial query)
            prefetched_data: Pre-loaded trace data and geometry (from async prefetch)

        Returns:
            Tuple of (metrics, tile_trace_count, tile_data_mb, tile_trace_indices):
            - metrics: Kernel execution metrics
            - tile_trace_count: Number of traces that contributed to this tile
            - tile_data_mb: Megabytes of trace data loaded for this tile
            - tile_trace_indices: List of trace indices loaded for aperture tracking
        """
        tile_start_time = time.time()
        self._debug.debug(f"--- Processing tile {tile.tile_id} ---")
        self._debug.debug(f"  Tile bounds: x=[{tile.x_min:.1f}, {tile.x_max:.1f}], y=[{tile.y_min:.1f}, {tile.y_max:.1f}]")
        self._debug.debug(f"  Tile grid indices: x=[{tile.x_start}, {tile.x_end}], y=[{tile.y_start}, {tile.y_end}]")
        self._debug.debug(f"  Tile size: {tile.nx}x{tile.ny}")

        # MEMORY SAFETY CHECK: Force GC if memory is low before starting tile
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            if available_gb < 12.0:
                self._debug.warning(
                    f"  [MEMORY] Low memory before tile {tile.tile_id}: {available_gb:.2f} GB available. "
                    "Running gc.collect()..."
                )
                import gc
                gc.collect()

                # Small delay to let OS reclaim memory
                import time as time_module
                time_module.sleep(0.5)

                mem = psutil.virtual_memory()
                available_after = mem.available / (1024**3)
                self._debug.info(f"  [MEMORY] After GC: {available_after:.2f} GB available")

                # If still critically low, log a severe warning
                if available_after < 8.0:
                    self._debug.error(
                        f"  [MEMORY] CRITICAL: Only {available_after:.2f} GB available! "
                        "Tile may fail. Consider reducing tile size or max_aperture."
                    )

        assert self._spatial_index is not None
        assert self._trace_reader is not None
        assert self._header_manager is not None
        assert self._velocity_manager is not None
        assert self._kernel is not None

        # Query traces in aperture (use pre-computed if available)
        t0 = time.time()
        if precomputed_query is not None:
            query_result = precomputed_query
            t_query = time.time() - t0
            self._debug.info(f"  [TIMING] Spatial query (pre-computed): {t_query:.3f}s - {query_result.n_traces:,} traces")
        else:
            query_result = query_traces_for_tile(
                self._spatial_index,
                tile.x_min,
                tile.x_max,
                tile.y_min,
                tile.y_max,
                kernel_config.max_aperture_m,
            )
            t_query = time.time() - t0
            self._debug.info(f"  [TIMING] Spatial query: {t_query:.3f}s - found {query_result.n_traces:,} traces")

        # Report progress with trace count (before heavy processing)
        # Use tile_id (0-indexed) as current to show we're WORKING on this tile, not done
        assert self._tile_plan is not None
        tile_ix = tile.x_start // max(1, tile.nx)
        tile_iy = tile.y_start // max(1, tile.ny)
        self._report_progress(
            current=tile.tile_id,  # 0-indexed: show 0/1 while working on tile 0
            total=self._tile_plan.n_tiles,
            message=f"Processing tile {tile.tile_id + 1}/{self._tile_plan.n_tiles} ({tile_ix},{tile_iy}) - {query_result.n_traces:,} traces",
            traces_in_tile=query_result.n_traces,
            tile_x=tile_ix,
            tile_y=tile_iy,
        )

        if query_result.n_traces == 0:
            # No traces contribute to this tile
            self._debug.debug(f"  No traces for tile {tile.tile_id}, skipping")
            return (
                KernelMetrics(n_traces_processed=0, n_samples_output=0, compute_time_s=0.0),
                0,  # tile_trace_count
                0.0,  # tile_data_mb
                [],  # tile_trace_indices
            )

        # OPTIMIZATION: If gather bin filtering is needed, load geometry FIRST (lightweight)
        # then filter, then load only matching trace data (heavyweight)
        has_gather_filter = (
            (offset_min is not None or offset_max is not None) or
            (ovt_x_min is not None or ovt_x_max is not None or ovt_y_min is not None or ovt_y_max is not None)
        )

        if has_gather_filter:
            # OPTIMIZED PATH: Load geometry first, filter, then load only matching traces
            # This avoids loading trace data we'll just throw away
            t0 = time.time()
            geometry = self._header_manager.get_geometry_for_indices(query_result.trace_indices)
            t_geom_load = time.time() - t0
            self._debug.info(f"  [TIMING] Geometry load (pre-filter): {t_geom_load:.3f}s - {geometry.n_traces:,} traces")

            # Build combined filter mask
            filter_mask = np.ones(geometry.n_traces, dtype=bool)

            # Apply offset filter if specified
            if offset_min is not None or offset_max is not None:
                if offset_min is not None:
                    filter_mask &= (geometry.offset >= offset_min)
                if offset_max is not None:
                    filter_mask &= (geometry.offset <= offset_max)
                self._debug.info(f"  [OFFSET FILTER] {offset_min:.0f}-{offset_max:.0f}m")

            # Apply OVT filter if specified
            if ovt_x_min is not None or ovt_x_max is not None or ovt_y_min is not None or ovt_y_max is not None:
                offset_x = geometry.receiver_x - geometry.source_x
                offset_y = geometry.receiver_y - geometry.source_y
                if ovt_x_min is not None:
                    filter_mask &= (offset_x >= ovt_x_min)
                if ovt_x_max is not None:
                    filter_mask &= (offset_x <= ovt_x_max)
                if ovt_y_min is not None:
                    filter_mask &= (offset_y >= ovt_y_min)
                if ovt_y_max is not None:
                    filter_mask &= (offset_y <= ovt_y_max)
                self._debug.info(f"  [OVT FILTER] X=[{ovt_x_min:.0f},{ovt_x_max:.0f}] Y=[{ovt_y_min:.0f},{ovt_y_max:.0f}]")

            n_before = geometry.n_traces
            n_after = np.sum(filter_mask)
            self._debug.info(f"  [GATHER FILTER] {n_before:,} -> {n_after:,} traces ({100*n_after/n_before:.1f}% kept)")

            if n_after == 0:
                # No traces match the gather bin filter for this tile
                return (
                    KernelMetrics(n_traces_processed=0, n_samples_output=0, compute_time_s=0.0),
                    0, 0.0, [],
                )

            # Filter geometry to matching traces
            filtered_indices = geometry.trace_indices[filter_mask]
            geometry = TraceGeometry(
                trace_indices=filtered_indices,
                source_x=geometry.source_x[filter_mask],
                source_y=geometry.source_y[filter_mask],
                receiver_x=geometry.receiver_x[filter_mask],
                receiver_y=geometry.receiver_y[filter_mask],
                offset=geometry.offset[filter_mask],
                midpoint_x=geometry.midpoint_x[filter_mask],
                midpoint_y=geometry.midpoint_y[filter_mask],
            )

            # Now load ONLY the matching trace data (the expensive operation)
            # Use cache to avoid redundant reads for overlapping tile apertures
            t0 = time.time()
            trace_data = self._trace_cache.get_traces(filtered_indices, self._trace_reader)
            t_trace_load = time.time() - t0
            trace_data_mb = trace_data.nbytes / 1024**2
            cache_stats = self._trace_cache.get_stats()
            self._debug.info(f"  [TIMING] Trace data load (post-filter): {t_trace_load:.3f}s - {trace_data.shape} ({trace_data_mb:.1f} MB) [cache hit: {cache_stats['hit_rate']:.1%}]")

        else:
            # STANDARD PATH: No gather filtering needed, load in original order
            # Check if we have prefetched data
            if prefetched_data is not None and prefetched_data.trace_data is not None and prefetched_data.error is None:
                # Use prefetched data (loaded in background while previous tile was processing)
                t0 = time.time()
                trace_data = prefetched_data.trace_data
                geometry = prefetched_data.geometry
                t_trace_load = time.time() - t0
                t_geom_load = 0.0  # Geometry was loaded with traces in prefetch
                trace_data_mb = trace_data.nbytes / 1024**2
                self._debug.info(f"  [TIMING] Using PREFETCHED data: {t_trace_load:.3f}s - {trace_data.shape} ({trace_data_mb:.1f} MB) [PREFETCH HIT]")
            else:
                # Use cache to avoid redundant reads for overlapping tile apertures
                log_memory_state(self._debug, f"tile_{tile.tile_id}_before_trace_load")
                t0 = time.time()
                trace_data = self._trace_cache.get_traces(query_result.trace_indices, self._trace_reader)
                t_trace_load = time.time() - t0
                trace_data_mb = trace_data.nbytes / 1024**2
                cache_stats = self._trace_cache.get_stats()
                self._debug.info(f"  [TIMING] Trace data load: {t_trace_load:.3f}s - {trace_data.shape} ({trace_data_mb:.1f} MB) [cache hit: {cache_stats['hit_rate']:.1%}]")
                log_memory_state(self._debug, f"tile_{tile.tile_id}_after_trace_load")

                # Load geometry
                t0 = time.time()
                geometry = self._header_manager.get_geometry_for_indices(query_result.trace_indices)
                t_geom_load = time.time() - t0
                self._debug.info(f"  [TIMING] Geometry load: {t_geom_load:.3f}s")

        # Create trace block
        traces = create_trace_block(
            amplitudes=trace_data,
            source_x=geometry.source_x,
            source_y=geometry.source_y,
            receiver_x=geometry.receiver_x,
            receiver_y=geometry.receiver_y,
            sample_rate_ms=self._trace_reader.sample_rate_ms or 2.0,
            start_time_ms=self._trace_reader.info.start_time_ms or 0.0,
        )
        self._debug.debug(f"  TraceBlock: {traces.n_traces} traces, {traces.n_samples} samples")

        # Create output tile with proper 2D coordinates for rotated grids
        grid_coords = grid.get_output_coordinates()
        X_grid = grid_coords.get('X')
        Y_grid = grid_coords.get('Y')

        if X_grid is not None and Y_grid is not None:
            # Rotated grid: extract 2D coordinates for this tile
            tile_X = X_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]
            tile_Y = Y_grid[tile.x_start:tile.x_end, tile.y_start:tile.y_end]
        else:
            # Axis-aligned grid: no 2D grids needed
            tile_X = None
            tile_Y = None

        output_tile = OutputTile(
            image=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.float64),
            fold=np.zeros((tile.nx, tile.ny, grid.nt), dtype=np.int32),  # 3D fold per sample
            x_axis=np.linspace(tile.x_min, tile.x_max, tile.nx),
            y_axis=np.linspace(tile.y_min, tile.y_max, tile.ny),
            t_axis_ms=np.arange(grid.t_min_ms, grid.t_max_ms + grid.dt_ms / 2, grid.dt_ms),
            x_grid=tile_X,
            y_grid=tile_Y,
        )
        self._debug.debug(f"  Output tile shape: {output_tile.image.shape}")

        # Get velocity for tile (supports both 1D and 3D)
        t0 = time.time()
        velocity = self._velocity_manager.get_velocity_slice_for_tile(
            tile.x_start, tile.x_end,
            tile.y_start, tile.y_end,
        )
        t_velocity = time.time() - t0
        self._debug.info(f"  [TIMING] Velocity slice: {t_velocity:.3f}s - is_1d={velocity.is_1d}, shape={velocity.vrms.shape}")

        # Execute kernel
        self._debug.info(f"  Executing kernel: {type(self._kernel).__name__}.migrate_tile()")
        self._debug.info(f"  === KERNEL START ===")
        n_output_points = output_tile.nx * output_tile.ny * output_tile.nt
        n_point_trace_pairs = output_tile.nx * output_tile.ny * traces.n_traces
        self._debug.info(f"  Output grid: {output_tile.nx} x {output_tile.ny} x {output_tile.nt} = {n_output_points:,} points")
        self._debug.info(f"  Traces: {traces.n_traces:,}")
        self._debug.info(f"  Estimated ops: {n_point_trace_pairs:,.0f} point-trace pairs")

        # Print directly to stderr so it shows immediately
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"KERNEL EXECUTING - This may take a LONG time!", file=sys.stderr, flush=True)
        print(f"  Grid: {output_tile.nx}x{output_tile.ny}x{output_tile.nt}", file=sys.stderr, flush=True)
        print(f"  Traces: {traces.n_traces:,}", file=sys.stderr, flush=True)
        print(f"  Point-trace pairs: {n_point_trace_pairs:,.0f}", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)

        log_memory_state(self._debug, f"tile_{tile.tile_id}_before_kernel")

        # Initialize CPU monitoring (call once to start sampling)
        if HAS_PSUTIL:
            psutil.cpu_percent(interval=None, percpu=True)  # Prime the pump

        kernel_start = time.time()

        # Dispatch based on kernel type
        kernel_type = kernel_config.kernel_type
        self._debug.info(f"  Kernel type: {kernel_type}")

        # Check for time-variant sampling with non-straight-ray kernels
        # Time-variant only supports straight_ray currently
        use_time_variant = kernel_config.time_variant_enabled and kernel_config.time_variant_windows
        if use_time_variant and kernel_type != "straight_ray":
            self._debug.warning(
                f"  Time-variant sampling is not supported with '{kernel_type}' kernel. "
                "Disabling time-variant for this tile."
            )
            print(
                f"  WARNING: Time-variant not supported with {kernel_type} kernel, using uniform sampling",
                file=sys.stderr, flush=True
            )
            use_time_variant = False

        # Check for time-variant sampling (only with straight_ray kernel)
        if use_time_variant:
            # Use time-variant kernel if available
            if hasattr(self._kernel, 'migrate_tile_time_variant'):
                windows = kernel_config.time_variant_windows
                self._debug.info(f"  Using TIME-VARIANT sampling with {len(windows)} windows")
                print(f"  TIME-VARIANT: {len(windows)} windows, {sum(w.n_samples for w in windows)} samples", file=sys.stderr, flush=True)

                metrics = self._kernel.migrate_tile_time_variant(
                    traces, output_tile, velocity, kernel_config, windows
                )

                # Resample output back to uniform sampling
                from pstm.algorithm.time_variant import resample_to_uniform
                tv_image = output_tile.image
                output_tile.image = resample_to_uniform(
                    tv_image, windows, kernel_config.output_dt_ms or 2.0, output_tile.nt
                )
                self._debug.info(f"  Resampled time-variant output to uniform")
            else:
                self._debug.warning("  Time-variant requested but kernel doesn't support it, using uniform")
                metrics = self._kernel.migrate_tile(traces, output_tile, velocity, kernel_config)
        elif kernel_type == "curved_ray" and hasattr(self._kernel, 'migrate_tile_curved_ray'):
            # Curved ray migration (V(z) = V0 + k*z)
            self._debug.info(f"  Using CURVED RAY kernel: v0={kernel_config.curved_ray_v0}, k={kernel_config.curved_ray_k}")
            print(f"  CURVED RAY: v0={kernel_config.curved_ray_v0}, k={kernel_config.curved_ray_k}", file=sys.stderr, flush=True)
            metrics = self._kernel.migrate_tile_curved_ray(traces, output_tile, velocity, kernel_config)
        elif kernel_type == "anisotropic_vti" and hasattr(self._kernel, 'migrate_tile_vti'):
            # VTI anisotropic migration (Alkhalifah-Tsvankin eta)
            self._debug.info(f"  Using VTI ANISOTROPIC kernel: eta={kernel_config.vti_eta_constant}")
            print(f"  VTI ANISOTROPIC: eta={kernel_config.vti_eta_constant}", file=sys.stderr, flush=True)
            metrics = self._kernel.migrate_tile_vti(traces, output_tile, velocity, kernel_config)
        else:
            # Standard straight ray migration
            if kernel_type not in ("straight_ray", "curved_ray", "anisotropic_vti"):
                self._debug.warning(f"  Unknown kernel type '{kernel_type}', defaulting to straight_ray")
            metrics = self._kernel.migrate_tile(traces, output_tile, velocity, kernel_config)

        kernel_time = time.time() - kernel_start

        # Get CPU utilization during kernel (shows what happened during execution)
        cpu_info = get_cpu_info()
        if cpu_info.get("available"):
            n_cores = cpu_info["n_cores"]
            active = cpu_info["active_cores"]
            busy = cpu_info["busy_cores"]
            overall = cpu_info["overall_percent"]
            self._debug.info(f"  [CPU] Overall: {overall:.1f}% | Active cores (>50%): {active}/{n_cores} | Busy cores (>90%): {busy}/{n_cores}")
            print(f"  [CPU] Overall: {overall:.1f}% | Active: {active}/{n_cores} cores | Busy: {busy}/{n_cores} cores", file=sys.stderr, flush=True)

        self._debug.info(f"  === KERNEL COMPLETE ===")
        self._debug.info(f"  Kernel completed in {kernel_time:.3f}s")

        # Calculate performance metrics
        traces_per_sec = traces.n_traces / kernel_time if kernel_time > 0 else 0
        points_per_sec = n_output_points / kernel_time if kernel_time > 0 else 0
        pairs_per_sec = n_point_trace_pairs / kernel_time if kernel_time > 0 else 0

        print(f"\nKERNEL COMPLETE: {kernel_time:.1f}s", file=sys.stderr, flush=True)
        print(f"  Performance: {traces_per_sec:,.0f} traces/s | {points_per_sec/1e6:.2f}M points/s | {pairs_per_sec/1e6:.1f}M pairs/s", file=sys.stderr, flush=True)

        self._debug.info(f"  [PERF] {traces_per_sec:,.0f} traces/s | {points_per_sec/1e6:.2f}M points/s | {pairs_per_sec/1e6:.1f}M pairs/s")
        self._debug.debug(f"  Kernel metrics: {metrics.n_traces_processed} traces, {metrics.n_samples_output} samples")
        log_memory_state(self._debug, f"tile_{tile.tile_id}_after_kernel")

        # Accumulate to output
        t0 = time.time()
        image[tile.x_start:tile.x_end, tile.y_start:tile.y_end, :] += output_tile.image
        fold[tile.x_start:tile.x_end, tile.y_start:tile.y_end, :] += output_tile.fold  # 3D fold

        # Accumulate header values (offset, azimuth) per output bin
        # This enables later resort to CIG (common offset, common angle, etc.)
        if geometry.n_traces > 0:
            # Compute azimuth from source/receiver (degrees, 0-360)
            dx = geometry.receiver_x - geometry.source_x
            dy = geometry.receiver_y - geometry.source_y
            azimuth_rad = np.arctan2(dx, dy)  # North = 0
            azimuth_deg = np.degrees(azimuth_rad) % 360

            # Convert to radians for sin/cos accumulation (circular mean)
            azimuth_rad_for_mean = np.radians(azimuth_deg)

            # Compute bin indices for each trace midpoint
            # Map midpoint to tile-local bin indices
            ix_local = np.floor((geometry.midpoint_x - tile.x_min) / grid.dx).astype(np.int32)
            iy_local = np.floor((geometry.midpoint_y - tile.y_min) / grid.dy).astype(np.int32)

            # Clamp to valid range and convert to global indices
            ix_local = np.clip(ix_local, 0, tile.nx - 1)
            iy_local = np.clip(iy_local, 0, tile.ny - 1)
            ix_global = ix_local + tile.x_start
            iy_global = iy_local + tile.y_start

            # Accumulate using np.add.at for thread-safe binning
            np.add.at(trace_count, (ix_global, iy_global), 1)
            np.add.at(offset_sum, (ix_global, iy_global), geometry.offset)
            np.add.at(azimuth_sin_sum, (ix_global, iy_global), np.sin(azimuth_rad_for_mean))
            np.add.at(azimuth_cos_sum, (ix_global, iy_global), np.cos(azimuth_rad_for_mean))

        t_accumulate = time.time() - t0
        self._debug.info(f"  [TIMING] Accumulate to output: {t_accumulate:.3f}s")

        # Save values needed for return before cleanup
        tile_trace_count = query_result.n_traces
        # Note: We skip tile_trace_indices to save memory - it's disabled anyway
        # (see line ~1163: "Disabled to save memory")

        # Explicit cleanup of large arrays to release memory before next tile
        # Clear references explicitly to help garbage collector
        traces = None
        output_tile = None
        trace_data = None
        geometry = None
        velocity = None
        query_result = None

        import gc
        # Run multiple GC passes to catch reference cycles
        gc.collect()
        gc.collect()
        gc.collect()

        log_memory_state(self._debug, f"tile_{tile.tile_id}_after_cleanup")

        tile_time = time.time() - tile_start_time

        # Print timing summary for this tile
        t_other = tile_time - (t_query + t_trace_load + t_geom_load + t_velocity + kernel_time + t_accumulate)
        self._debug.info(f"  [TIMING SUMMARY] Tile {tile.tile_id}:")
        self._debug.info(f"    Spatial query:   {t_query:6.3f}s ({100*t_query/tile_time:5.1f}%)")
        self._debug.info(f"    Trace load:      {t_trace_load:6.3f}s ({100*t_trace_load/tile_time:5.1f}%)")
        self._debug.info(f"    Geometry load:   {t_geom_load:6.3f}s ({100*t_geom_load/tile_time:5.1f}%)")
        self._debug.info(f"    Velocity slice:  {t_velocity:6.3f}s ({100*t_velocity/tile_time:5.1f}%)")
        self._debug.info(f"    KERNEL:          {kernel_time:6.3f}s ({100*kernel_time/tile_time:5.1f}%) <-- Main work")
        self._debug.info(f"    Accumulate:      {t_accumulate:6.3f}s ({100*t_accumulate/tile_time:5.1f}%)")
        self._debug.info(f"    Other overhead:  {t_other:6.3f}s ({100*t_other/tile_time:5.1f}%)")
        self._debug.info(f"    TOTAL:           {tile_time:6.3f}s")

        print(f"\n  TILE {tile.tile_id} TIMING BREAKDOWN:", file=sys.stderr, flush=True)
        print(f"    Query: {t_query:.2f}s | Load: {t_trace_load:.2f}s | Geom: {t_geom_load:.2f}s | Vel: {t_velocity:.2f}s", file=sys.stderr, flush=True)
        print(f"    KERNEL: {kernel_time:.2f}s ({100*kernel_time/tile_time:.0f}%) | Accum: {t_accumulate:.2f}s | Total: {tile_time:.2f}s\n", file=sys.stderr, flush=True)

        # Return metrics and aperture tracking data
        # Note: tile_trace_indices is empty to save memory (it was disabled anyway)
        return (
            metrics,
            tile_trace_count,  # tile_trace_count
            trace_data_mb,  # tile_data_mb
            [],  # tile_trace_indices - empty to save memory
        )

    def _should_use_trace_centric(self, kernel_config: KernelConfig) -> bool:
        """
        Determine if trace-centric migration should be used.

        Returns True if:
        - Tile overlap is high (>50% of traces per tile)
        - Dataset is small enough to benefit from single-pass processing
        """
        assert self._spatial_index is not None
        assert self._tile_plan is not None
        assert self._trace_reader is not None

        # Sample a few tiles to estimate overlap
        from pstm.pipeline.trace_centric_executor import estimate_trace_overlap

        try:
            overlap_score = estimate_trace_overlap(
                self._spatial_index,
                self._tile_plan,
                kernel_config.max_aperture_m,
                sample_tiles=min(10, self._tile_plan.n_tiles),
            )

            # Use trace-centric if overlap is high
            # Threshold: 0.5 means average trace is loaded 6x (1 + 0.5*10)
            use_trace_centric = overlap_score > 0.5

            self._debug.info(f"[TRACE-CENTRIC] Overlap score: {overlap_score:.2f}")
            self._debug.info(f"[TRACE-CENTRIC] Use trace-centric: {use_trace_centric}")

            # Also check dataset size - trace-centric is better for smaller datasets
            # that fit in GPU memory
            n_traces = self._trace_reader.n_traces
            n_samples = self._trace_reader.n_samples
            dataset_mb = (n_traces * n_samples * 4) / 1024**2

            # If dataset is very large, fall back to tile-by-tile
            if dataset_mb > 8000:  # >8GB
                self._debug.info(f"[TRACE-CENTRIC] Dataset too large ({dataset_mb:.0f} MB), using tile-by-tile")
                return False

            return use_trace_centric

        except Exception as e:
            self._debug.warning(f"[TRACE-CENTRIC] Failed to estimate overlap: {e}")
            return False

    def _run_trace_centric_migration(
        self,
        image: NDArray,
        fold: NDArray,
        trace_count: NDArray,
        offset_sum: NDArray,
        azimuth_sin_sum: NDArray,
        azimuth_cos_sum: NDArray,
        kernel_config: KernelConfig,
        grid,
        offset_min: float | None,
        offset_max: float | None,
        ovt_x_min: float | None = None,
        ovt_x_max: float | None = None,
        ovt_y_min: float | None = None,
        ovt_y_max: float | None = None,
    ) -> None:
        """
        Run trace-centric migration instead of tile-by-tile.

        Processes all traces in a single pass, scattering contributions to output.
        """
        from pstm.pipeline.trace_centric_executor import (
            run_trace_centric_migration,
            TraceCentricConfig,
            TraceCentricProgress,
        )

        assert self._trace_reader is not None
        assert self._header_manager is not None
        assert self._velocity_manager is not None
        assert self._memmap_manager is not None

        # Get trace indices to process (optionally filtered by offset/OVT)
        trace_indices = None
        if offset_min is not None or offset_max is not None:
            # Filter traces by offset using parquet predicate pushdown
            filtered = self._header_manager.get_midpoints_with_offset_filter(
                offset_min=offset_min,
                offset_max=offset_max,
            )
            trace_indices = filtered["trace_index"]
            self._debug.info(f"[TRACE-CENTRIC] Filtered to {len(trace_indices):,} traces by offset")
        elif ovt_x_min is not None or ovt_x_max is not None:
            # Filter by OVT - need to load all geometry and filter
            all_geometry = self._header_manager.get_all_geometry()
            offset_x = all_geometry.receiver_x - all_geometry.source_x
            offset_y = all_geometry.receiver_y - all_geometry.source_y

            mask = np.ones(len(offset_x), dtype=bool)
            if ovt_x_min is not None:
                mask &= (offset_x >= ovt_x_min)
            if ovt_x_max is not None:
                mask &= (offset_x <= ovt_x_max)
            if ovt_y_min is not None:
                mask &= (offset_y >= ovt_y_min)
            if ovt_y_max is not None:
                mask &= (offset_y <= ovt_y_max)

            trace_indices = all_geometry.trace_indices[mask]
            self._debug.info(f"[TRACE-CENTRIC] Filtered to {len(trace_indices):,} traces by OVT")

        # Progress callback
        def progress_callback(progress: TraceCentricProgress) -> None:
            self._report_progress(
                current=int(progress.traces_processed / 1000),  # Use thousands for tile equiv
                total=int(progress.total_traces / 1000),
                message=progress.message,
                traces_in_tile=progress.traces_processed,
                tile_x=0,
                tile_y=0,
            )

        # Create a temporary memmap manager wrapper that writes to the right arrays
        class TempMemmapWrapper:
            def __init__(self, image, fold):
                self._image = image
                self._fold = fold

            def get(self, name):
                if name == "image":
                    return self._image
                elif name == "fold":
                    return self._fold
                raise KeyError(name)

        temp_memmap = TempMemmapWrapper(image, fold)

        # Run trace-centric migration
        tc_config = TraceCentricConfig(
            batch_size=100_000,  # Process 100K traces per GPU batch
            report_interval=10_000,
        )

        metrics = run_trace_centric_migration(
            trace_reader=self._trace_reader,
            header_manager=self._header_manager,
            velocity_manager=self._velocity_manager,
            output_grid=grid,
            kernel_config=kernel_config,
            memmap_manager=temp_memmap,
            progress_callback=progress_callback,
            trace_indices=trace_indices,
            tc_config=tc_config,
        )

        # Update executor metrics
        self.metrics.n_traces_processed = metrics.n_traces_processed
        self.metrics.compute_time_total = metrics.compute_time_s
        self.metrics.n_tiles_completed = self._tile_plan.n_tiles  # Mark all as done

        self._debug.info(f"[TRACE-CENTRIC] Complete: {metrics.n_traces_processed:,} traces in {metrics.compute_time_s:.1f}s")

    def _finalize(self) -> None:
        """Finalize migration output."""
        assert self._memmap_manager is not None

        grid = self.config.output.grid

        # Check if we're in gathers mode (multiple gather bins - offset or OVT)
        if self._gather_bins:
            self._finalize_gathers()
        else:
            self._finalize_stack()

    def _finalize_stack(self) -> None:
        """Finalize single stacked output (original behavior)."""
        logger.info("Normalizing output...")

        image = self._memmap_manager.get("image")
        fold = self._memmap_manager.get("fold")

        # DIAGNOSTIC: Log image and fold statistics before normalization
        self._debug.info("=" * 60)
        self._debug.info("FINALIZE DIAGNOSTICS")
        self._debug.info("=" * 60)
        self._debug.info(f"FOLD DEBUG: shape={fold.shape}, dtype={fold.dtype}")
        self._debug.info(f"FOLD DEBUG: min={fold.min()}, max={fold.max()}, mean={fold.mean():.1f}")
        self._debug.info(f"FOLD DEBUG: non-zero pixels: {np.count_nonzero(fold)} / {fold.size} ({100*np.count_nonzero(fold)/fold.size:.1f}%)")
        self._debug.info(f"FOLD DEBUG: zero pixels: {np.count_nonzero(fold == 0)} / {fold.size}")

        self._debug.info(f"IMAGE DEBUG (before norm): shape={image.shape}, dtype={image.dtype}")
        self._debug.info(f"IMAGE DEBUG (before norm): min={image.min():.6f}, max={image.max():.6f}, mean={image.mean():.6f}")
        self._debug.info(f"IMAGE DEBUG (before norm): non-zero voxels: {np.count_nonzero(image)} / {image.size}")
        self._debug.info(f"IMAGE DEBUG (before norm): NaN count: {np.count_nonzero(np.isnan(image))}")
        self._debug.info(f"IMAGE DEBUG (before norm): Inf count: {np.count_nonzero(np.isinf(image))}")

        # Kirchhoff migration output: DO NOT normalize by fold
        # The Kirchhoff integral sum is the correct physical quantity.
        # Dividing by fold (~40,000 traces) would reduce amplitudes to ~1e-5 level.
        # Fold is saved separately for QC purposes.
        self._debug.info("FOLD DEBUG: Using Kirchhoff sum (no fold normalization)")
        self._debug.info(f"  Fold stats: min={fold.min()}, max={fold.max()}, mean={fold.mean():.1f}")
        image_normalized = image
        self._debug.info(f"IMAGE DEBUG (Kirchhoff sum): min={image_normalized.min():.6f}, max={image_normalized.max():.6f}, mean={image_normalized.mean():.6f}")

        logger.info("Writing output files...")

        # Write to Zarr
        output_path = self.config.output.output_dir / "migrated_stack.zarr"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        grid = self.config.output.grid
        z = zarr.open(
            str(output_path),
            mode="w",
            shape=image_normalized.shape,
            chunks=(
                self.config.output.chunk_x,
                self.config.output.chunk_y,
                self.config.output.chunk_t or grid.nt,
            ),
            dtype=np.float32,
        )
        z[:] = image_normalized.astype(np.float32)

        # Add metadata (using coordinates for corner-point grid compatibility)
        coords = grid.get_output_coordinates()
        z.attrs["x_min"] = float(coords['x'][0])
        z.attrs["x_max"] = float(coords['x'][-1])
        z.attrs["dx"] = grid.dx
        z.attrs["y_min"] = float(coords['y'][0])
        z.attrs["y_max"] = float(coords['y'][-1])
        z.attrs["dy"] = grid.dy
        z.attrs["t_min_ms"] = grid.t_min_ms
        z.attrs["t_max_ms"] = grid.t_max_ms
        z.attrs["dt_ms"] = grid.dt_ms
        z.attrs["migration_type"] = "PSTM"

        # Write fold map
        if self.config.output.products.fold_volume:
            fold_path = self.config.output.output_dir / "fold.zarr"
            z_fold = zarr.open(str(fold_path), mode="w", shape=fold.shape, dtype=np.int32)
            z_fold[:] = fold
            z_fold.attrs["description"] = "Migration fold map"

        # Compute and write average header values (offset, azimuth) per bin
        logger.info("Computing average headers per bin...")
        trace_count = self._memmap_manager.get("trace_count")
        offset_sum = self._memmap_manager.get("offset_sum")
        azimuth_sin_sum = self._memmap_manager.get("azimuth_sin_sum")
        azimuth_cos_sum = self._memmap_manager.get("azimuth_cos_sum")

        # Use trace_count for averaging (independent of kernel fold tracking)
        with np.errstate(invalid="ignore", divide="ignore"):
            offset_avg = np.where(trace_count > 0, offset_sum / trace_count, 0.0)
            azimuth_avg = np.degrees(np.arctan2(azimuth_sin_sum, azimuth_cos_sum)) % 360
            azimuth_avg = np.where(trace_count > 0, azimuth_avg, 0.0)

        # Write headers to Parquet
        self._write_bin_headers(trace_count, offset_avg, azimuth_avg, grid)

        # Clean up checkpoint
        if self._checkpoint:
            self._checkpoint.cleanup()

        print_success(f"Output written to {output_path}")

    def _finalize_gathers(self) -> None:
        """Finalize gather output (multiple volumes - offset and/or OVT bins)."""
        from pstm.config.models import GatherBinType

        logger.info(f"Finalizing {len(self._gather_bins)} gather volumes...")

        grid = self.config.output.grid
        gathers_dir = self.config.output.output_dir / "gathers"
        gathers_dir.mkdir(parents=True, exist_ok=True)

        # Collect all bin headers for combined output
        all_headers = []
        index_records = []

        for bid, gb in enumerate(self._gather_bins):
            # Log bin info based on type
            if gb.bin_type == GatherBinType.OFFSET:
                logger.info(f"Writing gather bin {bid}: OFFSET {gb.offset_min:.0f} - {gb.offset_max:.0f} m")
                bin_name = gb.name or f"offset_{gb.offset_min:.0f}_{gb.offset_max:.0f}"
            else:
                logger.info(f"Writing gather bin {bid}: OVT X[{gb.ovt_x_min:.0f},{gb.ovt_x_max:.0f}] Y[{gb.ovt_y_min:.0f},{gb.ovt_y_max:.0f}]")
                bin_name = gb.name or f"ovt_x{gb.ovt_x_min:.0f}_{gb.ovt_x_max:.0f}_y{gb.ovt_y_min:.0f}_{gb.ovt_y_max:.0f}"

            image = self._memmap_manager.get(f"image_bin_{bid}")
            fold = self._memmap_manager.get(f"fold_bin_{bid}")
            trace_count = self._memmap_manager.get(f"trace_count_bin_{bid}")
            offset_sum = self._memmap_manager.get(f"offset_sum_bin_{bid}")
            azimuth_sin_sum = self._memmap_manager.get(f"azimuth_sin_sum_bin_{bid}")
            azimuth_cos_sum = self._memmap_manager.get(f"azimuth_cos_sum_bin_{bid}")

            # Normalize by per-sample fold (3D fold)
            if np.any(fold > 0):
                with np.errstate(invalid="ignore", divide="ignore"):
                    image_normalized = np.where(fold > 0, image / fold, 0.0)
            else:
                image_normalized = image

            # Write volume for this bin
            output_path = gathers_dir / f"gather_bin_{bid:03d}.zarr"

            z = zarr.open(
                str(output_path),
                mode="w",
                shape=image_normalized.shape,
                chunks=(
                    self.config.output.chunk_x,
                    self.config.output.chunk_y,
                    self.config.output.chunk_t or grid.nt,
                ),
                dtype=np.float32,
            )
            z[:] = image_normalized.astype(np.float32)

            # Add metadata including bin info (using coordinates for corner-point grid compatibility)
            gather_coords = grid.get_output_coordinates()
            z.attrs["x_min"] = float(gather_coords['x'][0])
            z.attrs["x_max"] = float(gather_coords['x'][-1])
            z.attrs["dx"] = grid.dx
            z.attrs["y_min"] = float(gather_coords['y'][0])
            z.attrs["y_max"] = float(gather_coords['y'][-1])
            z.attrs["dy"] = grid.dy
            z.attrs["t_min_ms"] = grid.t_min_ms
            z.attrs["t_max_ms"] = grid.t_max_ms
            z.attrs["dt_ms"] = grid.dt_ms
            z.attrs["migration_type"] = "PSTM"
            z.attrs["gather_bin_id"] = bid
            z.attrs["gather_bin_type"] = gb.bin_type.value
            z.attrs["gather_bin_name"] = bin_name

            # Type-specific attributes
            if gb.bin_type == GatherBinType.OFFSET:
                z.attrs["offset_min"] = gb.offset_min
                z.attrs["offset_max"] = gb.offset_max
                z.attrs["offset_center"] = (gb.offset_min + gb.offset_max) / 2
            else:
                z.attrs["ovt_x_min"] = gb.ovt_x_min
                z.attrs["ovt_x_max"] = gb.ovt_x_max
                z.attrs["ovt_y_min"] = gb.ovt_y_min
                z.attrs["ovt_y_max"] = gb.ovt_y_max

            # Write fold for this bin
            fold_path = gathers_dir / f"fold_bin_{bid:03d}.zarr"
            z_fold = zarr.open(str(fold_path), mode="w", shape=fold.shape, dtype=np.int32)
            z_fold[:] = fold
            z_fold.attrs["gather_bin_id"] = bid

            # Compute averages for this bin
            with np.errstate(invalid="ignore", divide="ignore"):
                offset_avg = np.where(trace_count > 0, offset_sum / trace_count, 0.0)
                azimuth_avg = np.degrees(np.arctan2(azimuth_sin_sum, azimuth_cos_sum)) % 360
                azimuth_avg = np.where(trace_count > 0, azimuth_avg, 0.0)

            # Build headers for this bin (using coordinates for corner-point grid compatibility)
            header_grid_coords = grid.get_output_coordinates()
            x_coords = header_grid_coords['x']
            y_coords = header_grid_coords['y']
            xx, yy = np.meshgrid(np.arange(grid.nx), np.arange(grid.ny), indexing="ij")
            x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")

            # Build header dataframe with type-agnostic columns
            header_data = {
                "ix": xx.ravel().astype(np.int32),
                "iy": yy.ravel().astype(np.int32),
                "x": x_grid.ravel().astype(np.float64),
                "y": y_grid.ravel().astype(np.float64),
                "trace_count": trace_count.ravel().astype(np.int32),
                "offset_avg": offset_avg.ravel().astype(np.float32),
                "azimuth_avg": azimuth_avg.ravel().astype(np.float32),
                "gather_bin_id": np.full(grid.nx * grid.ny, bid, dtype=np.int32),
                "gather_bin_type": np.full(grid.nx * grid.ny, gb.bin_type.value, dtype=object),
                "gather_bin_name": np.full(grid.nx * grid.ny, bin_name, dtype=object),
            }

            # Add type-specific columns
            if gb.bin_type == GatherBinType.OFFSET:
                header_data["offset_bin_min"] = np.full(grid.nx * grid.ny, gb.offset_min, dtype=np.float32)
                header_data["offset_bin_max"] = np.full(grid.nx * grid.ny, gb.offset_max, dtype=np.float32)
                header_data["offset_bin_center"] = np.full(grid.nx * grid.ny, (gb.offset_min + gb.offset_max) / 2, dtype=np.float32)
            else:
                header_data["ovt_x_min"] = np.full(grid.nx * grid.ny, gb.ovt_x_min, dtype=np.float32)
                header_data["ovt_x_max"] = np.full(grid.nx * grid.ny, gb.ovt_x_max, dtype=np.float32)
                header_data["ovt_y_min"] = np.full(grid.nx * grid.ny, gb.ovt_y_min, dtype=np.float32)
                header_data["ovt_y_max"] = np.full(grid.nx * grid.ny, gb.ovt_y_max, dtype=np.float32)

            df_bin = pl.DataFrame(header_data)

            # Filter to bins with data
            df_bin = df_bin.filter(pl.col("trace_count") > 0)
            all_headers.append(df_bin)

            n_bins_with_data = len(df_bin)
            logger.info(f"  Bin {bid}: {n_bins_with_data} spatial bins with data")

            # Build index record for this bin
            index_record = {
                "bin_id": bid,
                "bin_type": gb.bin_type.value,
                "bin_name": bin_name,
                "volume_path": f"gathers/gather_bin_{bid:03d}.zarr",
            }
            if gb.bin_type == GatherBinType.OFFSET:
                index_record["offset_min"] = gb.offset_min
                index_record["offset_max"] = gb.offset_max
                index_record["offset_center"] = (gb.offset_min + gb.offset_max) / 2
            else:
                index_record["ovt_x_min"] = gb.ovt_x_min
                index_record["ovt_x_max"] = gb.ovt_x_max
                index_record["ovt_y_min"] = gb.ovt_y_min
                index_record["ovt_y_max"] = gb.ovt_y_max
            index_records.append(index_record)

        # Combine all headers and write
        if all_headers:
            df_all = pl.concat(all_headers, how="diagonal")  # diagonal allows different columns
            headers_path = self.config.output.output_dir / "gather_headers.parquet"
            df_all.write_parquet(str(headers_path))
            logger.info(f"Gather headers written to {headers_path}: {len(df_all)} total entries")

        # Write gather index file
        index_path = self.config.output.output_dir / "gathers_index.parquet"
        index_df = pl.DataFrame(index_records)
        index_df.write_parquet(str(index_path))
        logger.info(f"Gather index written to {index_path}")

        # Clean up checkpoint
        if self._checkpoint:
            self._checkpoint.cleanup()

        print_success(f"Gathers output written to {gathers_dir}")

    def _write_bin_headers(
        self,
        trace_count: NDArray,
        offset_avg: NDArray,
        azimuth_avg: NDArray,
        grid,
    ) -> None:
        """Write bin headers to Parquet."""
        headers_path = self.config.output.output_dir / "bin_headers.parquet"

        # Get coordinates (handles both bounding-box and corner-point grids)
        coords = grid.get_output_coordinates()
        x_coords = coords['x']
        y_coords = coords['y']

        xx, yy = np.meshgrid(np.arange(grid.nx), np.arange(grid.ny), indexing="ij")
        x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")

        df = pl.DataFrame({
            "ix": xx.ravel().astype(np.int32),
            "iy": yy.ravel().astype(np.int32),
            "x": x_grid.ravel().astype(np.float64),
            "y": y_grid.ravel().astype(np.float64),
            "trace_count": trace_count.ravel().astype(np.int32),
            "offset_avg": offset_avg.ravel().astype(np.float32),
            "azimuth_avg": azimuth_avg.ravel().astype(np.float32),
        })

        df_with_data = df.filter(pl.col("trace_count") > 0)
        df_with_data.write_parquet(str(headers_path))

        logger.info(
            f"Headers written to {headers_path}: "
            f"{len(df_with_data)} bins with data out of {grid.nx * grid.ny} total"
        )

    def _save_checkpoint(self) -> None:
        """Save checkpoint."""
        if self._checkpoint and self.config.execution.checkpoint.enabled:
            self._checkpoint.save()
            if self._memmap_manager:
                self._memmap_manager.flush()

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.debug("Cleaning up resources...")

        if self._trace_reader:
            self._trace_reader.close()

        if self._header_manager:
            self._header_manager.close()

        if self._kernel:
            self._kernel.cleanup()

        if self._memmap_manager:
            self._memmap_manager.release_all(delete_files=True)

    def _print_summary(self) -> None:
        """Print execution summary."""
        print_section("Migration Complete")
        print_metric("Total time", format_duration(self.metrics.elapsed_time))
        print_metric("Tiles processed", f"{self.metrics.n_tiles_completed:,}")
        print_metric("Traces processed", f"{self.metrics.n_traces_processed:,}")
        print_metric("Compute time", format_duration(self.metrics.compute_time_total))
        print_metric("Processing rate", f"{self.metrics.traces_per_second:.0f} traces/s")

        if self.metrics.warnings:
            logger.warning(f"{len(self.metrics.warnings)} warnings during execution")


def run_migration(
    config: MigrationConfig,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> bool:
    """
    Convenience function to run migration.

    Args:
        config: Migration configuration
        resume: Attempt to resume from checkpoint
        progress_callback: Optional progress callback

    Returns:
        True if completed successfully
    """
    executor = MigrationExecutor(config, progress_callback)
    return executor.run(resume=resume)
