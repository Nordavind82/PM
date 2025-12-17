"""
Compiled Metal kernel for PSTM migration.

Uses PyObjC to directly call compiled Metal shaders for maximum GPU performance.
This bypasses MLX's Python overhead and runs native Metal compute kernels.
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


def check_metal_available() -> bool:
    """Check if Metal is available."""
    if not METAL_AVAILABLE:
        return False
    try:
        device = Metal.MTLCreateSystemDefaultDevice()
        return device is not None
    except Exception:
        return False


# Migration parameters struct - must match Metal shader exactly
class MigrationParams(ctypes.Structure):
    """Migration parameters matching Metal struct layout."""
    _fields_ = [
        # Grid parameters
        ("dx", ctypes.c_float),
        ("dy", ctypes.c_float),
        ("dt_ms", ctypes.c_float),
        ("t_start_ms", ctypes.c_float),

        # Aperture parameters
        ("max_aperture", ctypes.c_float),
        ("min_aperture", ctypes.c_float),
        ("taper_fraction", ctypes.c_float),
        ("max_dip_deg", ctypes.c_float),

        # Amplitude correction flags
        ("apply_spreading", ctypes.c_int),
        ("apply_obliquity", ctypes.c_int),

        # Anti-aliasing parameters
        ("apply_aa", ctypes.c_int),
        ("aa_dominant_freq", ctypes.c_float),

        # Dimensions
        ("n_traces", ctypes.c_int),
        ("n_samples", ctypes.c_int),
        ("nx", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("nt", ctypes.c_int),
    ]


# Time window struct for time-variant sampling - must match Metal
class TimeWindowStruct(ctypes.Structure):
    """Time window matching Metal struct layout."""
    _fields_ = [
        ("t_start_ms", ctypes.c_float),
        ("t_end_ms", ctypes.c_float),
        ("dt_effective_ms", ctypes.c_float),
        ("downsample_factor", ctypes.c_int),
        ("sample_offset", ctypes.c_int),
        ("n_samples", ctypes.c_int),
    ]


# Time-variant parameters struct - must match Metal
class TimeVariantParamsStruct(ctypes.Structure):
    """Time-variant migration parameters matching Metal struct layout."""
    _fields_ = [
        # Grid parameters
        ("dx", ctypes.c_float),
        ("dy", ctypes.c_float),
        ("dt_base_ms", ctypes.c_float),
        ("t_start_ms", ctypes.c_float),

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
        ("n_windows", ctypes.c_int),
        ("total_output_samples", ctypes.c_int),
    ]


class CompiledMetalKernel:
    """
    High-performance Metal kernel using compiled shaders.
    
    Directly calls compiled .metallib for maximum GPU performance.
    """
    
    # Path to compiled Metal library
    METALLIB_PATH = Path(__file__).parent.parent / "metal" / "pstm_kernels.metallib"
    
    def __init__(self, use_simd: bool = True):
        """
        Initialize compiled Metal kernel.
        
        Args:
            use_simd: Use SIMD-optimized kernel variant
        """
        if not METAL_AVAILABLE:
            raise RuntimeError("Metal/PyObjC not available. Install with: pip install pyobjc-framework-Metal")
        
        self._initialized = False
        self._use_simd = use_simd
        self._config: KernelConfig | None = None
        
        # Metal objects
        self._device = None
        self._command_queue = None
        self._library = None
        self._pipeline_state = None
        self._kernel_name = "pstm_migrate_3d_simd" if use_simd else "pstm_migrate_3d"
    
    @property
    def name(self) -> str:
        return "metal_compiled"
    
    @property
    def capabilities(self) -> set[KernelCapability]:
        return {KernelCapability.FP32, KernelCapability.BATCH, KernelCapability.ASYNC}
    
    def initialize(self, config: KernelConfig) -> None:
        """Initialize Metal device and load compiled library."""
        self._config = config
        
        logger.info("Initializing compiled Metal kernel...")
        debug_logger.info(f"Metal library path: {self.METALLIB_PATH}")
        
        # Create Metal device
        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("Failed to create Metal device")
        
        logger.info(f"Metal device: {self._device.name()}")
        debug_logger.info(f"Metal device max threads per threadgroup: {self._device.maxThreadsPerThreadgroup()}")
        
        # Create command queue
        self._command_queue = self._device.newCommandQueue()
        if self._command_queue is None:
            raise RuntimeError("Failed to create command queue")
        
        # Load compiled Metal library
        if not self.METALLIB_PATH.exists():
            raise RuntimeError(f"Metal library not found: {self.METALLIB_PATH}. Run scripts/build_metal.sh first.")
        
        error = None
        self._library, error = self._device.newLibraryWithURL_error_(
            Foundation.NSURL.fileURLWithPath_(str(self.METALLIB_PATH)),
            None
        )
        if error:
            raise RuntimeError(f"Failed to load Metal library: {error}")
        
        logger.info(f"Loaded Metal library: {self.METALLIB_PATH}")
        
        # Get kernel function
        kernel_func = self._library.newFunctionWithName_(self._kernel_name)
        if kernel_func is None:
            raise RuntimeError(f"Kernel function '{self._kernel_name}' not found in library")
        
        # Create pipeline state
        self._pipeline_state, error = self._device.newComputePipelineStateWithFunction_error_(
            kernel_func, None
        )
        if error:
            raise RuntimeError(f"Failed to create pipeline state: {error}")
        
        logger.info(f"Pipeline state created for kernel: {self._kernel_name}")
        debug_logger.info(f"Max total threads per threadgroup: {self._pipeline_state.maxTotalThreadsPerThreadgroup()}")
        
        self._initialized = True
        logger.info("Compiled Metal kernel initialized successfully")
    
    def migrate_tile(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: KernelConfig,
    ) -> KernelMetrics:
        """
        Migrate traces to output tile using compiled Metal kernel.
        """
        if not self._initialized:
            self.initialize(config)
        
        start_time = time.perf_counter()
        
        n_traces = traces.n_traces
        nx, ny, nt = output.nx, output.ny, output.nt
        n_samples = traces.n_samples
        
        debug_logger.info(f"METAL migrate_tile: {n_traces} traces -> {nx}x{ny}x{nt} output")
        
        # Prepare input arrays as float32
        amplitudes = traces.amplitudes.astype(np.float32)
        source_x = traces.source_x.astype(np.float32)
        source_y = traces.source_y.astype(np.float32)
        receiver_x = traces.receiver_x.astype(np.float32)
        receiver_y = traces.receiver_y.astype(np.float32)
        midpoint_x = traces.midpoint_x.astype(np.float32)
        midpoint_y = traces.midpoint_y.astype(np.float32)
        x_coords = output.x_axis.astype(np.float32)
        y_coords = output.y_axis.astype(np.float32)
        
        # Pre-compute time-dependent values
        t_axis_s = output.t_axis_ms / 1000.0  # Convert to seconds
        if velocity.is_1d:
            vrms = velocity.vrms  # (nt,)
        else:
            # For 3D velocity, take center pillar
            vrms = velocity.vrms[nx//2, ny//2, :]
        
        t0_half_sq = ((t_axis_s / 2.0) ** 2).astype(np.float32)
        inv_v_sq = (1.0 / (vrms ** 2)).astype(np.float32)
        t0_s = t_axis_s.astype(np.float32)
        
        # Compute apertures for each time sample
        apertures = np.full(nt, config.max_aperture_m, dtype=np.float32)
        
        # Output arrays
        image_out = np.zeros((nx, ny, nt), dtype=np.float32)
        fold_out = np.zeros((nx, ny), dtype=np.int32)
        
        # Create Metal buffers
        buffers = self._create_buffers(
            amplitudes, source_x, source_y, receiver_x, receiver_y,
            midpoint_x, midpoint_y, image_out, fold_out,
            x_coords, y_coords, t0_half_sq, inv_v_sq, t0_s, apertures
        )
        
        # Create parameters
        params = MigrationParams()
        params.dx = output.x_axis[1] - output.x_axis[0] if nx > 1 else 25.0
        params.dy = output.y_axis[1] - output.y_axis[0] if ny > 1 else 25.0
        params.dt_ms = traces.sample_rate_ms
        params.t_start_ms = traces.start_time_ms
        params.max_aperture = config.max_aperture_m
        params.min_aperture = config.min_aperture_m
        params.taper_fraction = config.taper_fraction
        params.max_dip_deg = config.max_dip_degrees
        params.apply_spreading = 1 if config.apply_spreading else 0
        params.apply_obliquity = 1 if config.apply_obliquity else 0
        params.apply_aa = 1 if config.apply_aa else 0
        params.aa_dominant_freq = config.aa_dominant_freq if hasattr(config, 'aa_dominant_freq') else 30.0
        params.n_traces = n_traces
        params.n_samples = n_samples
        params.nx = nx
        params.ny = ny
        params.nt = nt
        
        # Create params buffer - convert struct to bytes
        params_bytes = bytes(params)
        params_buffer = self._device.newBufferWithBytes_length_options_(
            params_bytes,
            len(params_bytes),
            Metal.MTLResourceStorageModeShared
        )
        
        # Create command buffer
        command_buffer = self._command_queue.commandBuffer()
        
        # Create compute encoder
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline_state)
        
        # Set buffers
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 15)
        
        # Calculate thread grid
        # 3D grid: (nx, ny, nt) threads
        threads_per_grid = Metal.MTLSize()
        threads_per_grid.width = nx
        threads_per_grid.height = ny
        threads_per_grid.depth = nt
        
        # Threadgroup size - optimize for GPU
        max_threads = self._pipeline_state.maxTotalThreadsPerThreadgroup()
        tg_width = min(8, nx)
        tg_height = min(8, ny)
        tg_depth = min(max_threads // (tg_width * tg_height), nt)
        
        threads_per_threadgroup = Metal.MTLSize()
        threads_per_threadgroup.width = tg_width
        threads_per_threadgroup.height = tg_height
        threads_per_threadgroup.depth = tg_depth
        
        debug_logger.info(f"METAL dispatch: grid=({nx},{ny},{nt}), threadgroup=({tg_width},{tg_height},{tg_depth})")
        
        # Dispatch
        encoder.dispatchThreads_threadsPerThreadgroup_(threads_per_grid, threads_per_threadgroup)
        encoder.endEncoding()
        
        # Execute and wait
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # Check for errors
        if command_buffer.error():
            raise RuntimeError(f"Metal execution error: {command_buffer.error()}")
        
        # Read results back using as_buffer() to get memoryview
        image_buf = buffers[7].contents().as_buffer(image_out.nbytes)
        fold_buf = buffers[8].contents().as_buffer(fold_out.nbytes)

        # Copy to numpy arrays
        image_result = np.frombuffer(image_buf, dtype=np.float32).reshape(nx, ny, nt)
        fold_result = np.frombuffer(fold_buf, dtype=np.int32).reshape(nx, ny)

        # Copy to output
        output.image[:] = image_result.astype(np.float64)
        output.fold[:] = fold_result
        
        compute_time = time.perf_counter() - start_time
        
        debug_logger.info(f"METAL completed in {compute_time:.3f}s")
        
        return KernelMetrics(
            n_traces_processed=n_traces,
            n_samples_output=nx * ny * nt,
            compute_time_s=compute_time,
        )
    
    def _create_buffer(self, arr: np.ndarray) -> "Metal.MTLBuffer":
        """Create a Metal buffer from a numpy array."""
        arr = np.ascontiguousarray(arr)
        # Create buffer with data directly using tobytes()
        buf = self._device.newBufferWithBytes_length_options_(
            arr.tobytes(),
            arr.nbytes,
            Metal.MTLResourceStorageModeShared
        )
        return buf

    def _create_buffers(
        self,
        amplitudes, source_x, source_y, receiver_x, receiver_y,
        midpoint_x, midpoint_y, image, fold,
        x_coords, y_coords, t0_half_sq, inv_v_sq, t0_s, apertures
    ):
        """Create Metal buffers from numpy arrays."""
        arrays = [
            amplitudes, source_x, source_y, receiver_x, receiver_y,
            midpoint_x, midpoint_y, image, fold,
            x_coords, y_coords, t0_half_sq, inv_v_sq, t0_s, apertures
        ]

        buffers = []
        for arr in arrays:
            buf = self._create_buffer(arr)
            buffers.append(buf)

        return buffers
    
    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        pass  # Metal commits are synchronous after waitUntilCompleted
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._initialized = False
        self._pipeline_state = None
        self._library = None
        self._command_queue = None
        self._device = None

    def migrate_tile_time_variant(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: KernelConfig,
        windows: list,
    ) -> KernelMetrics:
        """
        Migrate traces using time-variant sampling.

        Args:
            traces: Input trace block
            output: Output tile
            velocity: Velocity model
            config: Kernel configuration
            windows: List of TimeWindow objects from time_variant module

        Returns:
            Kernel metrics
        """
        if not self._initialized:
            self.initialize(config)

        start_time = time.perf_counter()

        n_traces = traces.n_traces
        nx, ny = output.nx, output.ny
        n_samples = traces.n_samples

        # Compute total output samples across all windows
        total_output_samples = sum(w.n_samples for w in windows)

        debug_logger.info(
            f"METAL TV migrate_tile: {n_traces} traces -> {nx}x{ny}x{total_output_samples} output "
            f"({len(windows)} windows)"
        )

        # Get time-variant kernel
        tv_kernel_func = self._library.newFunctionWithName_("pstm_migrate_time_variant")
        if tv_kernel_func is None:
            raise RuntimeError("Time-variant kernel function not found in library")

        tv_pipeline, error = self._device.newComputePipelineStateWithFunction_error_(
            tv_kernel_func, None
        )
        if error:
            raise RuntimeError(f"Failed to create time-variant pipeline: {error}")

        # Prepare input arrays as float32
        amplitudes = traces.amplitudes.astype(np.float32)
        source_x = traces.source_x.astype(np.float32)
        source_y = traces.source_y.astype(np.float32)
        receiver_x = traces.receiver_x.astype(np.float32)
        receiver_y = traces.receiver_y.astype(np.float32)
        midpoint_x = traces.midpoint_x.astype(np.float32)
        midpoint_y = traces.midpoint_y.astype(np.float32)
        x_coords = output.x_axis.astype(np.float32)
        y_coords = output.y_axis.astype(np.float32)

        # Velocity array
        if velocity.is_1d:
            vrms = velocity.vrms.astype(np.float32)
        else:
            vrms = velocity.vrms[nx//2, ny//2, :].astype(np.float32)

        # Output arrays (time-variant sized)
        image_out = np.zeros((nx, ny, total_output_samples), dtype=np.float32)
        fold_out = np.zeros((nx, ny), dtype=np.int32)

        # Create buffers
        buffers = [
            self._create_buffer(amplitudes),
            self._create_buffer(source_x),
            self._create_buffer(source_y),
            self._create_buffer(receiver_x),
            self._create_buffer(receiver_y),
            self._create_buffer(midpoint_x),
            self._create_buffer(midpoint_y),
            self._create_buffer(image_out),
            self._create_buffer(fold_out),
            self._create_buffer(x_coords),
            self._create_buffer(y_coords),
            self._create_buffer(vrms),
        ]

        # Create windows buffer
        n_windows = len(windows)
        windows_array = (TimeWindowStruct * n_windows)()
        for i, w in enumerate(windows):
            windows_array[i].t_start_ms = w.t_start_ms
            windows_array[i].t_end_ms = w.t_end_ms
            windows_array[i].dt_effective_ms = w.dt_effective_ms
            windows_array[i].downsample_factor = w.downsample_factor
            windows_array[i].sample_offset = w.sample_start
            windows_array[i].n_samples = w.n_samples

        windows_bytes = bytes(windows_array)
        windows_buffer = self._device.newBufferWithBytes_length_options_(
            windows_bytes, len(windows_bytes), Metal.MTLResourceStorageModeShared
        )
        buffers.append(windows_buffer)

        # Create params
        params = TimeVariantParamsStruct()
        params.dx = output.x_axis[1] - output.x_axis[0] if nx > 1 else 25.0
        params.dy = output.y_axis[1] - output.y_axis[0] if ny > 1 else 25.0
        params.dt_base_ms = traces.sample_rate_ms
        params.t_start_ms = traces.start_time_ms
        params.max_aperture = config.max_aperture_m
        params.min_aperture = config.min_aperture_m
        params.taper_fraction = config.taper_fraction
        params.max_dip_deg = config.max_dip_degrees
        params.apply_spreading = 1 if config.apply_spreading else 0
        params.apply_obliquity = 1 if config.apply_obliquity else 0
        params.apply_aa = 1 if config.apply_aa else 0
        params.aa_dominant_freq = config.aa_dominant_freq if hasattr(config, 'aa_dominant_freq') else 30.0
        params.n_traces = n_traces
        params.n_samples = n_samples
        params.nx = nx
        params.ny = ny
        params.n_windows = n_windows
        params.total_output_samples = total_output_samples

        params_bytes = bytes(params)
        params_buffer = self._device.newBufferWithBytes_length_options_(
            params_bytes, len(params_bytes), Metal.MTLResourceStorageModeShared
        )

        # Create command buffer
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(tv_pipeline)

        # Set buffers (0-11 trace/output, 12 windows, 13 params)
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 13)

        # Thread grid
        threads_per_grid = Metal.MTLSize()
        threads_per_grid.width = nx
        threads_per_grid.height = ny
        threads_per_grid.depth = total_output_samples

        max_threads = tv_pipeline.maxTotalThreadsPerThreadgroup()
        tg_width = min(8, nx)
        tg_height = min(8, ny)
        tg_depth = min(max_threads // (tg_width * tg_height), total_output_samples)

        threads_per_threadgroup = Metal.MTLSize()
        threads_per_threadgroup.width = tg_width
        threads_per_threadgroup.height = tg_height
        threads_per_threadgroup.depth = max(1, tg_depth)

        debug_logger.info(
            f"METAL TV dispatch: grid=({nx},{ny},{total_output_samples}), "
            f"threadgroup=({tg_width},{tg_height},{tg_depth})"
        )

        encoder.dispatchThreads_threadsPerThreadgroup_(threads_per_grid, threads_per_threadgroup)
        encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        if command_buffer.error():
            raise RuntimeError(f"Metal execution error: {command_buffer.error()}")

        # Read results
        image_buf = buffers[7].contents().as_buffer(image_out.nbytes)
        fold_buf = buffers[8].contents().as_buffer(fold_out.nbytes)

        image_result = np.frombuffer(image_buf, dtype=np.float32).reshape(nx, ny, total_output_samples)
        fold_result = np.frombuffer(fold_buf, dtype=np.int32).reshape(nx, ny)

        # Store time-variant result (caller must resample to uniform)
        # We return the raw TV output - resampling is done by caller
        output.image = image_result.astype(np.float64)
        output.fold[:] = fold_result

        compute_time = time.perf_counter() - start_time
        debug_logger.info(f"METAL TV completed in {compute_time:.3f}s")

        return KernelMetrics(
            n_traces_processed=n_traces,
            n_samples_output=nx * ny * total_output_samples,
            compute_time_s=compute_time,
        )


def get_compiled_metal_kernel(use_simd: bool = True) -> CompiledMetalKernel:
    """
    Factory function to get compiled Metal kernel.
    
    Args:
        use_simd: Use SIMD-optimized kernel
    
    Returns:
        Compiled Metal kernel instance
    """
    if not check_metal_available():
        raise RuntimeError("Metal not available on this system")
    return CompiledMetalKernel(use_simd=use_simd)
