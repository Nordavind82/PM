"""
MLX Metal kernel for PSTM.

GPU-accelerated migration kernel using Apple's MLX framework.
Optimized for Apple Silicon unified memory architecture.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pstm.kernels.base import (
    KernelCapability,
    KernelConfig,
    KernelMetrics,
    MigrationKernel,
    OutputTile,
    TraceBlock,
    VelocitySlice,
)
from pstm.utils.logging import get_logger

logger = get_logger(__name__)
debug_logger = logging.getLogger("pstm.migration.debug")

# Try to import MLX
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


def _log_memory_state(context: str) -> None:
    """Log current memory state for debugging MLX memory issues."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()

        debug_logger.info(
            f"MLX_MEMORY [{context}]: "
            f"Process RSS={mem_info.rss / 1e9:.2f}GB, "
            f"VMS={mem_info.vms / 1e9:.2f}GB, "
            f"System available={system_mem.available / 1e9:.2f}GB "
            f"({system_mem.percent}% used)"
        )
    except ImportError:
        debug_logger.info(f"MLX_MEMORY [{context}]: psutil not available for memory tracking")
    except Exception as e:
        debug_logger.warning(f"MLX_MEMORY [{context}]: Error getting memory info: {e}")


def check_mlx_available() -> bool:
    """Check if MLX is available and functional."""
    if not MLX_AVAILABLE:
        return False
    try:
        # Quick test
        x = mx.array([1.0, 2.0, 3.0])
        _ = mx.sum(x)
        return True
    except Exception:
        return False


class MLXKernel:
    """
    MLX-accelerated GPU migration kernel.

    Uses Apple's MLX framework for Metal GPU acceleration.
    Leverages unified memory for zero-copy data transfer.

    Key optimizations:
    - Vectorized travel time computation
    - Batched trace processing
    - Efficient gather operations for interpolation
    - Chunked accumulation to manage memory
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize MLX kernel.

        Args:
            chunk_size: Number of traces to process per GPU batch
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available. Install with: pip install mlx")

        self._initialized = False
        self._config: KernelConfig | None = None
        self._chunk_size = chunk_size

    @property
    def name(self) -> str:
        return "mlx_metal"

    @property
    def capabilities(self) -> set[KernelCapability]:
        return {KernelCapability.FP32, KernelCapability.BATCH, KernelCapability.ASYNC}

    def initialize(self, config: KernelConfig) -> None:
        """Initialize kernel and warm up MLX."""
        self._config = config

        logger.info("Initializing MLX kernel...")

        # Warm up MLX with small computation
        test = mx.array([1.0, 2.0, 3.0])
        _ = mx.sum(test * test)
        mx.eval(_)

        self._initialized = True
        logger.info("MLX kernel initialized")

    def migrate_tile(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: KernelConfig,
    ) -> KernelMetrics:
        """
        Migrate traces to output tile using MLX GPU.

        Args:
            traces: Input trace data
            output: Output tile (modified in place)
            velocity: Velocity model
            config: Algorithm configuration

        Returns:
            Execution metrics
        """
        if not self._initialized:
            self.initialize(config)

        start_time = time.perf_counter()

        n_traces = traces.n_traces
        nx, ny, nt = output.nx, output.ny, output.nt

        debug_logger.info(f"MLX migrate_tile START: n_traces={n_traces}, output_shape=({nx}, {ny}, {nt})")
        debug_logger.info(f"MLX migrate_tile: amplitudes_shape={traces.amplitudes.shape}, dtype={traces.amplitudes.dtype}")
        debug_logger.info(f"MLX migrate_tile: chunk_size={self._chunk_size}")

        _log_memory_state("before_array_conversion")

        # Convert inputs to MLX arrays (zero-copy on unified memory)
        debug_logger.info("MLX: Converting amplitudes to MLX array...")
        amplitudes_mx = mx.array(traces.amplitudes)  # (n_traces, n_samples)
        _log_memory_state("after_amplitudes_conversion")

        debug_logger.info("MLX: Converting coordinates to MLX arrays...")
        source_x_mx = mx.array(traces.source_x)
        source_y_mx = mx.array(traces.source_y)
        receiver_x_mx = mx.array(traces.receiver_x)
        receiver_y_mx = mx.array(traces.receiver_y)
        midpoint_x_mx = mx.array(traces.midpoint_x)
        midpoint_y_mx = mx.array(traces.midpoint_y)
        _log_memory_state("after_coordinates_conversion")

        # Output grid
        debug_logger.info("MLX: Converting output grid to MLX arrays...")
        x_axis_mx = mx.array(output.x_axis)
        y_axis_mx = mx.array(output.y_axis)
        t_axis_mx = mx.array(output.t_axis_ms / 1000.0)  # Convert to seconds

        # Velocity
        debug_logger.info("MLX: Converting velocity to MLX array...")
        if velocity.is_1d:
            vrms_mx = mx.array(velocity.vrms)  # (nt,)
        else:
            vrms_mx = mx.array(velocity.vrms)  # (nx, ny, nt)
        _log_memory_state("after_velocity_conversion")

        # Initialize output accumulator on GPU
        debug_logger.info(f"MLX: Initializing output arrays ({nx}x{ny}x{nt})...")
        image_mx = mx.zeros((nx, ny, nt), dtype=mx.float32)
        fold_mx = mx.zeros((nx, ny), dtype=mx.int32)
        _log_memory_state("after_output_init")

        dt_s = traces.sample_rate_ms / 1000.0
        t0_s = traces.start_time_ms / 1000.0
        n_input_samples = traces.n_samples

        max_aperture = config.max_aperture_m
        taper_fraction = config.taper_fraction

        total_chunks = (n_traces + self._chunk_size - 1) // self._chunk_size
        debug_logger.info(f"MLX: Processing {total_chunks} chunks...")

        # Process in chunks to manage memory
        for chunk_idx, chunk_start in enumerate(range(0, n_traces, self._chunk_size)):
            chunk_end = min(chunk_start + self._chunk_size, n_traces)
            debug_logger.info(f"MLX CHUNK {chunk_idx + 1}/{total_chunks}: traces {chunk_start}-{chunk_end}")

            # Slice chunk
            amp_chunk = amplitudes_mx[chunk_start:chunk_end]
            sx_chunk = source_x_mx[chunk_start:chunk_end]
            sy_chunk = source_y_mx[chunk_start:chunk_end]
            rx_chunk = receiver_x_mx[chunk_start:chunk_end]
            ry_chunk = receiver_y_mx[chunk_start:chunk_end]
            mx_chunk = midpoint_x_mx[chunk_start:chunk_end]
            my_chunk = midpoint_y_mx[chunk_start:chunk_end]

            # Migrate chunk
            image_chunk, fold_chunk = self._migrate_chunk(
                amp_chunk, sx_chunk, sy_chunk, rx_chunk, ry_chunk,
                mx_chunk, my_chunk,
                x_axis_mx, y_axis_mx, t_axis_mx,
                vrms_mx, velocity.is_1d,
                dt_s, t0_s, n_input_samples,
                max_aperture, taper_fraction,
                config.apply_spreading, config.apply_obliquity,
            )

            # Accumulate
            image_mx = image_mx + image_chunk
            fold_mx = fold_mx + fold_chunk

            # Evaluate to free intermediate memory
            debug_logger.info(f"MLX CHUNK {chunk_idx + 1}: Evaluating to free intermediate memory...")
            mx.eval(image_mx, fold_mx)
            _log_memory_state(f"after_chunk_{chunk_idx + 1}_eval")

        # Synchronize and convert back to NumPy
        debug_logger.info("MLX: Final synchronization...")
        mx.eval(image_mx, fold_mx)
        _log_memory_state("after_final_eval")

        # Copy to output (back to NumPy)
        output.image[:] = np.array(image_mx, dtype=np.float64)
        output.fold[:] = np.array(fold_mx, dtype=np.int32)

        compute_time = time.perf_counter() - start_time

        return KernelMetrics(
            n_traces_processed=n_traces,
            n_samples_output=nx * ny * nt,
            compute_time_s=compute_time,
        )

    def _migrate_chunk(
        self,
        amplitudes: "mx.array",
        source_x: "mx.array",
        source_y: "mx.array",
        receiver_x: "mx.array",
        receiver_y: "mx.array",
        midpoint_x: "mx.array",
        midpoint_y: "mx.array",
        x_axis: "mx.array",
        y_axis: "mx.array",
        t_axis: "mx.array",
        vrms: "mx.array",
        vrms_is_1d: bool,
        dt_s: float,
        t0_s: float,
        n_input_samples: int,
        max_aperture: float,
        taper_fraction: float,
        apply_spreading: bool,
        apply_obliquity: bool,
    ) -> tuple["mx.array", "mx.array"]:
        """
        Migrate a chunk of traces to output grid.

        Fully vectorized MLX implementation.
        """
        n_traces = amplitudes.shape[0]
        nx = x_axis.shape[0]
        ny = y_axis.shape[0]
        nt = t_axis.shape[0]

        debug_logger.debug(f"MLX _migrate_chunk: n_traces={n_traces}, output=({nx}x{ny}x{nt})")

        # Create output point meshgrid
        # Shape: (nx, ny)
        ox, oy = mx.meshgrid(x_axis, y_axis, indexing='ij')

        # Initialize accumulators
        image = mx.zeros((nx, ny, nt), dtype=mx.float32)
        fold = mx.zeros((nx, ny), dtype=mx.int32)

        # Process each trace (vectorized over output grid)
        # Track last yield time for periodic GIL release
        import time as _time
        last_yield_time = _time.time()
        yield_interval = 0.1  # Release GIL every 100ms to allow UI updates

        for i in range(n_traces):
            # Periodically release GIL to allow Qt event processing
            now = _time.time()
            if now - last_yield_time > yield_interval:
                mx.eval(image, fold)  # Force evaluation
                _time.sleep(0)  # Yield GIL
                last_yield_time = now
                if i % 1000 == 0:
                    debug_logger.debug(f"MLX _migrate_chunk: trace {i}/{n_traces} (GIL yielded)")

            sx = source_x[i]
            sy = source_y[i]
            rx = receiver_x[i]
            ry = receiver_y[i]
            mx_i = midpoint_x[i]
            my_i = midpoint_y[i]
            trace = amplitudes[i]  # (n_samples,)

            # Compute distance from each output point to midpoint
            # Shape: (nx, ny)
            dist = mx.sqrt((ox - mx_i) ** 2 + (oy - my_i) ** 2)

            # Aperture mask
            in_aperture = dist < max_aperture

            # Skip if no output points in aperture
            if not mx.any(in_aperture):
                continue

            # Compute aperture weights
            # Shape: (nx, ny)
            taper_start = max_aperture * (1.0 - taper_fraction)
            weights = mx.where(
                dist <= taper_start,
                mx.ones_like(dist),
                0.5 * (1.0 + mx.cos(mx.pi * (dist - taper_start) / (max_aperture - taper_start)))
            )
            weights = mx.where(in_aperture, weights, mx.zeros_like(weights))

            # Loop over output times
            for it in range(nt):
                t0_out = t_axis[it]

                if t0_out <= 0:
                    continue

                # Get velocity
                if vrms_is_1d:
                    v = vrms[it]
                else:
                    v = vrms[:, :, it]  # (nx, ny)

                v2 = v * v
                t0_sq = t0_out * t0_out

                # Compute DSR travel time for all output points
                # t = sqrt(t0^2 + d_s^2/v^2) + sqrt(t0^2 + d_r^2/v^2)
                dx_s = ox - sx
                dy_s = oy - sy
                dist_sq_s = dx_s * dx_s + dy_s * dy_s

                dx_r = ox - rx
                dy_r = oy - ry
                dist_sq_r = dx_r * dx_r + dy_r * dy_r

                t_s = mx.sqrt(t0_sq + dist_sq_s / v2)
                t_r = mx.sqrt(t0_sq + dist_sq_r / v2)
                t_travel = t_s + t_r  # (nx, ny)

                # Convert to sample index
                sample_idx = (t_travel - t0_s) / dt_s  # (nx, ny)

                # Bounds check
                valid = (sample_idx >= 0) & (sample_idx < n_input_samples - 1) & in_aperture

                # Linear interpolation indices
                idx0 = mx.floor(sample_idx).astype(mx.int32)
                idx0 = mx.clip(idx0, 0, n_input_samples - 2)
                idx1 = idx0 + 1
                frac = sample_idx - idx0.astype(mx.float32)

                # Gather amplitudes (linear interpolation)
                amp0 = trace[idx0]  # Broadcasting gather
                amp1 = trace[idx1]
                amp_interp = amp0 * (1.0 - frac) + amp1 * frac

                # Apply weights
                w = weights

                if apply_obliquity:
                    t_half = t_travel / 2.0
                    obliq = mx.minimum(t_half / t_travel, mx.ones_like(t_travel))
                    w = w * obliq

                if apply_spreading:
                    spread = mx.where(t_travel > 0, (2.0 * t0_out) / t_travel, mx.ones_like(t_travel))
                    w = w * spread

                # Accumulate with validity mask
                contribution = mx.where(valid, amp_interp * w, mx.zeros_like(amp_interp))
                image = image.at[:, :, it].add(contribution)

            # Update fold for this trace
            fold = fold + in_aperture.astype(mx.int32)

        return image, fold

    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        if MLX_AVAILABLE:
            mx.eval()

    def cleanup(self) -> None:
        """Clean up kernel resources."""
        self._initialized = False
        self._config = None


class MLXKernelOptimized:
    """
    Highly optimized MLX kernel using batched operations.

    This version processes multiple output time samples simultaneously
    for better GPU utilization.
    """

    def __init__(self, chunk_size: int = 500, time_batch: int = 50):
        """
        Initialize optimized MLX kernel.

        Args:
            chunk_size: Traces per batch
            time_batch: Output time samples per batch
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available")

        self._initialized = False
        self._config: KernelConfig | None = None
        self._chunk_size = chunk_size
        self._time_batch = time_batch

    @property
    def name(self) -> str:
        return "mlx_metal_optimized"

    @property
    def capabilities(self) -> set[KernelCapability]:
        return {KernelCapability.FP32, KernelCapability.BATCH, KernelCapability.ASYNC}

    def initialize(self, config: KernelConfig) -> None:
        """Initialize kernel."""
        self._config = config
        self._initialized = True
        logger.info("MLX optimized kernel initialized")

    def migrate_tile(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: KernelConfig,
    ) -> KernelMetrics:
        """Migrate with batched time processing."""
        if not self._initialized:
            self.initialize(config)

        start_time = time.perf_counter()

        nx, ny, nt = output.nx, output.ny, output.nt
        n_traces = traces.n_traces

        debug_logger.info(f"MLX_OPT migrate_tile START: n_traces={n_traces}, output=({nx}x{ny}x{nt})")
        debug_logger.info(f"MLX_OPT: chunk_size={self._chunk_size}, time_batch={self._time_batch}")
        _log_memory_state("opt_before_conversion")

        # Convert to MLX
        debug_logger.info("MLX_OPT: Converting input arrays...")
        amplitudes_mx = mx.array(traces.amplitudes)
        source_x_mx = mx.array(traces.source_x)
        source_y_mx = mx.array(traces.source_y)
        receiver_x_mx = mx.array(traces.receiver_x)
        receiver_y_mx = mx.array(traces.receiver_y)
        midpoint_x_mx = mx.array(traces.midpoint_x)
        midpoint_y_mx = mx.array(traces.midpoint_y)
        _log_memory_state("opt_after_input_conversion")

        x_axis_mx = mx.array(output.x_axis)
        y_axis_mx = mx.array(output.y_axis)
        t_axis_s = mx.array(output.t_axis_ms / 1000.0)

        if velocity.is_1d:
            vrms_mx = mx.array(velocity.vrms)
        else:
            vrms_mx = mx.array(velocity.vrms)

        dt_s = traces.sample_rate_ms / 1000.0
        t0_s = traces.start_time_ms / 1000.0
        n_input_samples = traces.n_samples

        # Create meshgrid once
        debug_logger.info("MLX_OPT: Creating meshgrid...")
        ox, oy = mx.meshgrid(x_axis_mx, y_axis_mx, indexing='ij')
        ox = ox.reshape(-1)  # Flatten to (nx*ny,)
        oy = oy.reshape(-1)

        n_pillars = nx * ny
        debug_logger.info(f"MLX_OPT: n_pillars={n_pillars}")

        # Initialize output
        debug_logger.info("MLX_OPT: Initializing output arrays...")
        image_flat = mx.zeros((n_pillars, nt), dtype=mx.float32)
        fold_flat = mx.zeros((n_pillars,), dtype=mx.int32)
        _log_memory_state("opt_after_output_init")

        max_aperture = config.max_aperture_m
        taper_fraction = config.taper_fraction
        taper_start = max_aperture * (1.0 - taper_fraction)

        total_trace_chunks = (n_traces + self._chunk_size - 1) // self._chunk_size
        debug_logger.info(f"MLX_OPT: Processing {total_trace_chunks} trace chunks...")

        # Track time for periodic GIL release
        import time as _time
        last_yield_time = _time.time()
        yield_interval = 0.1  # Release GIL every 100ms to allow UI updates

        # Process traces in chunks
        for chunk_idx, t_start in enumerate(range(0, n_traces, self._chunk_size)):
            t_end = min(t_start + self._chunk_size, n_traces)
            n_chunk = t_end - t_start
            debug_logger.info(f"MLX_OPT TRACE_CHUNK {chunk_idx + 1}/{total_trace_chunks}: traces {t_start}-{t_end}")

            # Get chunk data
            sx = source_x_mx[t_start:t_end]  # (n_chunk,)
            sy = source_y_mx[t_start:t_end]
            rx = receiver_x_mx[t_start:t_end]
            ry = receiver_y_mx[t_start:t_end]
            mx_c = midpoint_x_mx[t_start:t_end]
            my_c = midpoint_y_mx[t_start:t_end]
            amp = amplitudes_mx[t_start:t_end]  # (n_chunk, n_samples)

            # Compute distances: (n_pillars, n_chunk)
            dist = mx.sqrt(
                (ox[:, None] - mx_c[None, :]) ** 2 +
                (oy[:, None] - my_c[None, :]) ** 2
            )

            # Aperture mask and weights
            in_aperture = dist < max_aperture
            weights = mx.where(
                dist <= taper_start,
                mx.ones_like(dist),
                0.5 * (1.0 + mx.cos(mx.pi * (dist - taper_start) / (max_aperture - taper_start)))
            )
            weights = mx.where(in_aperture, weights, mx.zeros_like(weights))

            # Update fold
            fold_contribution = mx.sum(in_aperture.astype(mx.int32), axis=1)
            fold_flat = fold_flat + fold_contribution

            total_time_batches = (nt + self._time_batch - 1) // self._time_batch
            _log_memory_state(f"opt_chunk_{chunk_idx + 1}_before_time_loop")

            # Process time in batches
            for time_idx, it_start in enumerate(range(0, nt, self._time_batch)):
                # Periodically release GIL to allow Qt event processing
                now = _time.time()
                if now - last_yield_time > yield_interval:
                    mx.eval(image_flat, fold_flat)  # Force evaluation
                    _time.sleep(0)  # Yield GIL
                    last_yield_time = now

                it_end = min(it_start + self._time_batch, nt)
                t_batch = t_axis_s[it_start:it_end]  # (n_time_batch,)
                n_time_batch = it_end - it_start

                if time_idx % 10 == 0:  # Log every 10th time batch to avoid spam
                    debug_logger.debug(f"MLX_OPT TIME_BATCH {time_idx + 1}/{total_time_batches}: t[{it_start}:{it_end}]")

                # Get velocity for time batch
                if velocity.is_1d:
                    v_batch = vrms_mx[it_start:it_end]  # (n_time_batch,)
                    v2 = v_batch * v_batch
                    # Broadcast: (n_pillars, n_chunk, n_time_batch)
                    v2 = v2[None, None, :]
                else:
                    v_batch = vrms_mx[:, :, it_start:it_end].reshape(n_pillars, n_time_batch)
                    v2 = v_batch[:, None, :] ** 2

                t0_sq = t_batch[None, None, :] ** 2  # (1, 1, n_time_batch)

                # Compute distances for DSR
                dx_s = ox[:, None, None] - sx[None, :, None]  # (n_pillars, n_chunk, 1)
                dy_s = oy[:, None, None] - sy[None, :, None]
                dist_sq_s = dx_s ** 2 + dy_s ** 2

                dx_r = ox[:, None, None] - rx[None, :, None]
                dy_r = oy[:, None, None] - ry[None, :, None]
                dist_sq_r = dx_r ** 2 + dy_r ** 2

                # DSR travel time: (n_pillars, n_chunk, n_time_batch)
                t_s = mx.sqrt(t0_sq + dist_sq_s / v2)
                t_r = mx.sqrt(t0_sq + dist_sq_r / v2)
                t_travel = t_s + t_r

                # Sample indices
                sample_idx = (t_travel - t0_s) / dt_s
                valid = (sample_idx >= 0) & (sample_idx < n_input_samples - 1)
                valid = valid & in_aperture[:, :, None]

                # Interpolation
                idx0 = mx.floor(sample_idx).astype(mx.int32)
                idx0 = mx.clip(idx0, 0, n_input_samples - 2)
                frac = sample_idx - idx0.astype(mx.float32)

                # Gather - need to handle batched indexing
                # amp shape: (n_chunk, n_samples)
                # idx0 shape: (n_pillars, n_chunk, n_time_batch)
                # Result shape: (n_pillars, n_chunk, n_time_batch)
                amp_shape = amp.shape
                idx0_flat = idx0.reshape(-1)
                idx1_flat = (idx0 + 1).reshape(-1)

                # Batch index for traces
                trace_idx = mx.arange(n_chunk)[None, :, None]
                trace_idx = mx.broadcast_to(trace_idx, idx0.shape).reshape(-1)

                # Gather
                amp0 = amp[trace_idx, idx0_flat].reshape(idx0.shape)
                amp1 = amp[trace_idx, idx1_flat].reshape(idx0.shape)
                amp_interp = amp0 * (1.0 - frac) + amp1 * frac

                # Apply weights
                w = weights[:, :, None]

                if config.apply_obliquity:
                    obliq = mx.where(t_travel > 0, t_batch[None, None, :] / t_travel, mx.ones_like(t_travel))
                    w = w * obliq

                if config.apply_spreading:
                    spread = mx.where(t_travel > 0, (2.0 * t_batch[None, None, :]) / t_travel, mx.ones_like(t_travel))
                    w = w * spread

                # Accumulate
                contribution = mx.where(valid, amp_interp * w, mx.zeros_like(amp_interp))
                contribution = mx.sum(contribution, axis=1)  # Sum over traces: (n_pillars, n_time_batch)

                image_flat = image_flat.at[:, it_start:it_end].add(contribution)

            debug_logger.info(f"MLX_OPT TRACE_CHUNK {chunk_idx + 1}: Evaluating to free memory...")
            mx.eval(image_flat, fold_flat)
            _log_memory_state(f"opt_chunk_{chunk_idx + 1}_after_eval")

        # Reshape and copy to output
        debug_logger.info("MLX_OPT: Final synchronization...")
        mx.eval(image_flat, fold_flat)
        _log_memory_state("opt_after_final_eval")
        output.image[:] = np.array(image_flat.reshape(nx, ny, nt), dtype=np.float64)
        output.fold[:] = np.array(fold_flat.reshape(nx, ny), dtype=np.int32)

        compute_time = time.perf_counter() - start_time

        return KernelMetrics(
            n_traces_processed=n_traces,
            n_samples_output=nx * ny * nt,
            compute_time_s=compute_time,
        )

    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        if MLX_AVAILABLE:
            mx.eval()

    def cleanup(self) -> None:
        """Clean up resources."""
        self._initialized = False
        self._config = None


def get_mlx_kernel(optimized: bool = True) -> MLXKernel | MLXKernelOptimized:
    """
    Factory function to get MLX kernel.

    Args:
        optimized: Use optimized batched version

    Returns:
        MLX kernel instance
    """
    if not check_mlx_available():
        raise RuntimeError("MLX not available on this system")

    if optimized:
        return MLXKernelOptimized()
    return MLXKernel()
