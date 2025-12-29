"""
Trace-Centric Metal Kernel for PSTM Migration.

This kernel processes each trace ONCE and scatters contributions to output points,
eliminating redundant trace processing when traces overlap multiple tiles.

Key advantages:
- Each trace loaded exactly once (vs. ~100x in tile-centric approach)
- O(traces) complexity instead of O(tiles × traces)
- Better for large apertures with high trace overlap

Trade-offs:
- Requires atomic operations for output accumulation
- Less cache-friendly for output access pattern
- Best when trace overlap is high (>50% per tile)
"""

from __future__ import annotations

import ctypes
import logging
import time
from pathlib import Path
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

# Try to import Metal via PyObjC
try:
    import Metal
    import Foundation
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    Metal = None
    Foundation = None


class TraceCentricParams(ctypes.Structure):
    """Migration parameters matching Metal struct layout."""
    _fields_ = [
        # Grid parameters
        ("dx", ctypes.c_float),
        ("dy", ctypes.c_float),
        ("dt_ms", ctypes.c_float),
        ("t_start_ms", ctypes.c_float),

        # Output grid bounds
        ("x_min", ctypes.c_float),
        ("y_min", ctypes.c_float),
        ("t_min_ms", ctypes.c_float),

        # Aperture parameters
        ("max_aperture", ctypes.c_float),
        ("min_aperture", ctypes.c_float),
        ("taper_fraction", ctypes.c_float),
        ("max_dip_deg", ctypes.c_float),

        # Amplitude correction flags
        ("apply_spreading", ctypes.c_int),
        ("apply_obliquity", ctypes.c_int),
        ("apply_aa", ctypes.c_int),
        ("aa_dominant_freq", ctypes.c_float),

        # Dimensions
        ("n_traces", ctypes.c_int),
        ("n_samples", ctypes.c_int),
        ("nx", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("nt", ctypes.c_int),
    ]


class TraceCache:
    """
    Cache for trace data to avoid redundant loading.

    Keeps trace data in GPU memory across multiple kernel invocations.
    Uses LRU eviction when cache is full.
    """

    def __init__(self, max_traces: int = 500_000, device=None):
        """
        Initialize trace cache.

        Args:
            max_traces: Maximum number of traces to cache
            device: Metal device
        """
        self.max_traces = max_traces
        self._device = device

        # Cache state
        self._cached_trace_indices: set[int] = set()
        self._trace_data: NDArray | None = None
        self._geometry_data: dict[str, NDArray] = {}
        self._trace_index_map: dict[int, int] = {}  # original_idx -> cache_idx

        # GPU buffers (persistent)
        self._amplitudes_buffer = None
        self._source_x_buffer = None
        self._source_y_buffer = None
        self._receiver_x_buffer = None
        self._receiver_y_buffer = None
        self._midpoint_x_buffer = None
        self._midpoint_y_buffer = None

        self._buffer_capacity = 0
        self._buffer_used = 0

    @property
    def n_cached(self) -> int:
        """Number of traces currently cached."""
        return len(self._cached_trace_indices)

    def contains(self, trace_indices: NDArray) -> tuple[NDArray, NDArray]:
        """
        Check which traces are in cache.

        Returns:
            Tuple of (cached_mask, missing_indices)
        """
        cached_mask = np.array([idx in self._cached_trace_indices for idx in trace_indices])
        missing_indices = trace_indices[~cached_mask]
        return cached_mask, missing_indices

    def add_traces(
        self,
        trace_indices: NDArray,
        amplitudes: NDArray,
        source_x: NDArray,
        source_y: NDArray,
        receiver_x: NDArray,
        receiver_y: NDArray,
    ) -> None:
        """
        Add traces to cache.

        Args:
            trace_indices: Original trace indices
            amplitudes: Trace amplitude data [n_traces, n_samples]
            source_x, source_y: Source coordinates
            receiver_x, receiver_y: Receiver coordinates
        """
        n_new = len(trace_indices)
        if n_new == 0:
            return

        # Check if we need to evict
        if self.n_cached + n_new > self.max_traces:
            n_to_evict = self.n_cached + n_new - self.max_traces
            self._evict_oldest(n_to_evict)

        # Add to cache
        start_idx = self._buffer_used
        for i, orig_idx in enumerate(trace_indices):
            self._cached_trace_indices.add(orig_idx)
            self._trace_index_map[orig_idx] = start_idx + i

        # Store data (expand arrays if needed)
        if self._trace_data is None:
            self._trace_data = amplitudes.astype(np.float32)
            self._geometry_data = {
                'source_x': source_x.astype(np.float32),
                'source_y': source_y.astype(np.float32),
                'receiver_x': receiver_x.astype(np.float32),
                'receiver_y': receiver_y.astype(np.float32),
                'midpoint_x': ((source_x + receiver_x) / 2).astype(np.float32),
                'midpoint_y': ((source_y + receiver_y) / 2).astype(np.float32),
            }
        else:
            self._trace_data = np.vstack([self._trace_data, amplitudes.astype(np.float32)])
            for key, new_data in [
                ('source_x', source_x),
                ('source_y', source_y),
                ('receiver_x', receiver_x),
                ('receiver_y', receiver_y),
            ]:
                self._geometry_data[key] = np.concatenate([
                    self._geometry_data[key],
                    new_data.astype(np.float32)
                ])
            self._geometry_data['midpoint_x'] = np.concatenate([
                self._geometry_data['midpoint_x'],
                ((source_x + receiver_x) / 2).astype(np.float32)
            ])
            self._geometry_data['midpoint_y'] = np.concatenate([
                self._geometry_data['midpoint_y'],
                ((source_y + receiver_y) / 2).astype(np.float32)
            ])

        self._buffer_used += n_new

        # Invalidate GPU buffers (will be recreated on next use)
        self._amplitudes_buffer = None

    def _evict_oldest(self, n_to_evict: int) -> None:
        """Evict oldest traces from cache."""
        # Simple strategy: evict from the beginning
        indices_to_evict = list(self._cached_trace_indices)[:n_to_evict]
        for idx in indices_to_evict:
            self._cached_trace_indices.discard(idx)
            self._trace_index_map.pop(idx, None)

        # Note: This is a simplified eviction that doesn't compact memory
        # A more sophisticated implementation would rebuild the cache periodically

    def get_gpu_buffers(self, device) -> dict:
        """
        Get GPU buffers for cached traces.

        Returns dict with buffer objects.
        """
        if self._amplitudes_buffer is None and self._trace_data is not None:
            self._create_gpu_buffers(device)

        return {
            'amplitudes': self._amplitudes_buffer,
            'source_x': self._source_x_buffer,
            'source_y': self._source_y_buffer,
            'receiver_x': self._receiver_x_buffer,
            'receiver_y': self._receiver_y_buffer,
            'midpoint_x': self._midpoint_x_buffer,
            'midpoint_y': self._midpoint_y_buffer,
        }

    def _create_gpu_buffers(self, device) -> None:
        """Create Metal buffers from cached data."""
        if self._trace_data is None:
            return

        def create_buffer(arr):
            arr = np.ascontiguousarray(arr)
            return device.newBufferWithBytes_length_options_(
                arr.tobytes(),
                arr.nbytes,
                Metal.MTLResourceStorageModeShared
            )

        self._amplitudes_buffer = create_buffer(self._trace_data)
        self._source_x_buffer = create_buffer(self._geometry_data['source_x'])
        self._source_y_buffer = create_buffer(self._geometry_data['source_y'])
        self._receiver_x_buffer = create_buffer(self._geometry_data['receiver_x'])
        self._receiver_y_buffer = create_buffer(self._geometry_data['receiver_y'])
        self._midpoint_x_buffer = create_buffer(self._geometry_data['midpoint_x'])
        self._midpoint_y_buffer = create_buffer(self._geometry_data['midpoint_y'])

    def clear(self) -> None:
        """Clear the cache."""
        self._cached_trace_indices.clear()
        self._trace_index_map.clear()
        self._trace_data = None
        self._geometry_data.clear()
        self._amplitudes_buffer = None
        self._buffer_used = 0


class TraceCentricKernel(MigrationKernel):
    """
    Trace-centric PSTM migration kernel.

    Processes all traces once and scatters contributions to output grid.
    """

    def __init__(self):
        self._device = None
        self._command_queue = None
        self._library = None
        self._pipeline_state = None
        self._initialized = False
        self._trace_cache = None

    @property
    def name(self) -> str:
        return "TraceCentricMetal"

    @property
    def capabilities(self) -> set[KernelCapability]:
        return {
            KernelCapability.GPU,
            KernelCapability.STRAIGHT_RAY,
        }

    def initialize(self, config: KernelConfig) -> None:
        """Initialize the Metal kernel."""
        if self._initialized:
            return

        if not METAL_AVAILABLE:
            raise RuntimeError("Metal is not available")

        # Get Metal device
        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("No Metal device found")

        logger.info(f"Using Metal device: {self._device.name()}")

        # Create command queue
        self._command_queue = self._device.newCommandQueue()

        # Try to load pre-compiled metallib first, fall back to source compilation
        metallib_path = Path(__file__).parent.parent / "metal" / "pstm_kernels.metallib"
        shader_path = Path(__file__).parent.parent / "metal" / "shaders" / "pstm_trace_centric.metal"

        if metallib_path.exists():
            # Use pre-compiled library
            url = Foundation.NSURL.fileURLWithPath_(str(metallib_path))
            self._library, error = self._device.newLibraryWithURL_error_(url, None)

            if self._library is None or error is not None:
                logger.warning(f"Failed to load metallib: {error}, falling back to source compilation")
                self._library = None
            else:
                logger.info(f"Loaded pre-compiled Metal library: {metallib_path}")

        if self._library is None:
            # Fall back to source compilation
            if not shader_path.exists():
                raise RuntimeError(f"Shader file not found: {shader_path}")

            with open(shader_path, 'r') as f:
                shader_source = f.read()

            options = Metal.MTLCompileOptions.alloc().init()
            self._library = self._device.newLibraryWithSource_options_error_(
                shader_source, options, None
            )

            if self._library is None:
                raise RuntimeError(f"Failed to compile Metal shader")

        # Get compute function
        function = self._library.newFunctionWithName_("pstm_migrate_trace_centric")
        if function is None:
            raise RuntimeError("Failed to find kernel function 'pstm_migrate_trace_centric'")

        # Create pipeline state
        self._pipeline_state, error = self._device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if self._pipeline_state is None:
            raise RuntimeError(f"Failed to create pipeline state: {error}")

        # Initialize trace cache
        self._trace_cache = TraceCache(max_traces=500_000, device=self._device)

        self._initialized = True
        logger.info("Trace-centric Metal kernel initialized successfully")

    def migrate_tile(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: KernelConfig,
    ) -> KernelMetrics:
        """
        Migrate traces to output using trace-centric approach.

        Instead of tile-by-tile processing, this processes all traces once.
        """
        if not self._initialized:
            self.initialize(config)

        start_time = time.perf_counter()

        n_traces = traces.n_traces
        nx, ny, nt = output.nx, output.ny, output.nt
        n_samples = traces.n_samples

        debug_logger.info(f"TRACE-CENTRIC migrate: {n_traces} traces -> {nx}x{ny}x{nt} output")

        # Prepare input arrays
        amplitudes = np.ascontiguousarray(traces.amplitudes.astype(np.float32))
        source_x = np.ascontiguousarray(traces.source_x.astype(np.float32))
        source_y = np.ascontiguousarray(traces.source_y.astype(np.float32))
        receiver_x = np.ascontiguousarray(traces.receiver_x.astype(np.float32))
        receiver_y = np.ascontiguousarray(traces.receiver_y.astype(np.float32))
        midpoint_x = np.ascontiguousarray(traces.midpoint_x.astype(np.float32))
        midpoint_y = np.ascontiguousarray(traces.midpoint_y.astype(np.float32))

        # Pre-compute time-dependent values
        t_axis_s = output.t_axis_ms / 1000.0
        if velocity.is_1d:
            vrms = velocity.vrms
        else:
            vrms = velocity.vrms[nx//2, ny//2, :]

        t0_half_sq = ((t_axis_s / 2.0) ** 2).astype(np.float32)
        inv_v_sq = (1.0 / (vrms ** 2)).astype(np.float32)
        t0_s = t_axis_s.astype(np.float32)
        apertures = np.full(nt, config.max_aperture_m, dtype=np.float32)

        # Output arrays (initialized to zero)
        image_out = np.zeros((nx, ny, nt), dtype=np.float32)
        fold_out = np.zeros((nx, ny), dtype=np.int32)

        # Create Metal buffers
        def create_buffer(arr):
            arr = np.ascontiguousarray(arr)
            return self._device.newBufferWithBytes_length_options_(
                arr.tobytes(),
                arr.nbytes,
                Metal.MTLResourceStorageModeShared
            )

        amp_buf = create_buffer(amplitudes)
        sx_buf = create_buffer(source_x)
        sy_buf = create_buffer(source_y)
        rx_buf = create_buffer(receiver_x)
        ry_buf = create_buffer(receiver_y)
        mx_buf = create_buffer(midpoint_x)
        my_buf = create_buffer(midpoint_y)

        # Output buffers
        img_buf = create_buffer(image_out)
        fold_buf = create_buffer(fold_out)

        # Time-dependent value buffers
        t0_buf = create_buffer(t0_half_sq)
        inv_v_buf = create_buffer(inv_v_sq)
        t0s_buf = create_buffer(t0_s)
        ap_buf = create_buffer(apertures)

        # Create parameters
        params = TraceCentricParams()
        params.dx = output.x_axis[1] - output.x_axis[0] if nx > 1 else 25.0
        params.dy = output.y_axis[1] - output.y_axis[0] if ny > 1 else 25.0
        params.dt_ms = traces.sample_rate_ms
        params.t_start_ms = traces.start_time_ms
        params.x_min = float(output.x_axis[0])
        params.y_min = float(output.y_axis[0])
        params.t_min_ms = float(output.t_axis_ms[0])
        params.max_aperture = config.max_aperture_m
        params.min_aperture = config.min_aperture_m
        params.taper_fraction = config.taper_fraction
        params.max_dip_deg = config.max_dip_degrees
        params.apply_spreading = 1 if config.apply_spreading else 0
        params.apply_obliquity = 1 if config.apply_obliquity else 0
        params.apply_aa = 1 if config.aa_enabled else 0
        params.aa_dominant_freq = getattr(config, 'aa_dominant_freq', 30.0)
        params.n_traces = n_traces
        params.n_samples = n_samples
        params.nx = nx
        params.ny = ny
        params.nt = nt

        params_bytes = bytes(params)
        params_buf = self._device.newBufferWithBytes_length_options_(
            params_bytes,
            len(params_bytes),
            Metal.MTLResourceStorageModeShared
        )

        # Create command buffer and encoder
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline_state)

        # Set buffers in order matching shader
        buffers = [
            amp_buf, sx_buf, sy_buf, rx_buf, ry_buf, mx_buf, my_buf,
            img_buf, fold_buf,
            t0_buf, inv_v_buf, t0s_buf, ap_buf,
            params_buf
        ]
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        # Dispatch - one thread per trace
        threads_per_grid = Metal.MTLSize()
        threads_per_grid.width = n_traces
        threads_per_grid.height = 1
        threads_per_grid.depth = 1

        max_threads = self._pipeline_state.maxTotalThreadsPerThreadgroup()
        tg_width = min(256, n_traces, max_threads)

        threads_per_threadgroup = Metal.MTLSize()
        threads_per_threadgroup.width = tg_width
        threads_per_threadgroup.height = 1
        threads_per_threadgroup.depth = 1

        debug_logger.info(f"TRACE-CENTRIC dispatch: {n_traces} threads, threadgroup={tg_width}")

        encoder.dispatchThreads_threadsPerThreadgroup_(threads_per_grid, threads_per_threadgroup)
        encoder.endEncoding()

        # Execute and wait
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        if command_buffer.error():
            raise RuntimeError(f"Metal execution error: {command_buffer.error()}")

        # Read results
        image_data = img_buf.contents().as_buffer(image_out.nbytes)
        fold_data = fold_buf.contents().as_buffer(fold_out.nbytes)

        image_result = np.frombuffer(image_data, dtype=np.float32).reshape(nx, ny, nt)
        fold_result = np.frombuffer(fold_data, dtype=np.int32).reshape(nx, ny)

        output.image[:] = image_result.astype(np.float64)
        output.fold[:] = fold_result

        compute_time = time.perf_counter() - start_time

        debug_logger.info(f"TRACE-CENTRIC completed in {compute_time:.3f}s")
        debug_logger.info(f"  Throughput: {n_traces / compute_time:,.0f} traces/s")

        return KernelMetrics(
            n_traces_processed=n_traces,
            n_samples_output=nx * ny * nt,
            compute_time_s=compute_time,
        )

    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        pass

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._trace_cache:
            self._trace_cache.clear()
        self._initialized = False
        self._pipeline_state = None
        self._library = None
        self._command_queue = None
        self._device = None
