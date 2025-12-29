"""
Compiled Metal kernel for PSTM migration.

Uses PyObjC to directly call compiled Metal shaders for maximum GPU performance.
This bypasses MLX's Python overhead and runs native Metal compute kernels.
"""

from __future__ import annotations

import ctypes
import gc
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
    import objc
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    Metal = None
    Foundation = None
    objc = None


def drain_autorelease_pool():
    """Create and drain an autorelease pool to release Objective-C objects."""
    if Foundation is not None:
        try:
            pool = Foundation.NSAutoreleasePool.alloc().init()
            del pool
        except Exception as e:
            debug_logger.warning(f"Failed to drain autorelease pool: {e}")


def check_metal_available() -> bool:
    """Check if Metal is available."""
    if not METAL_AVAILABLE:
        return False
    try:
        device = Metal.MTLCreateSystemDefaultDevice()
        return device is not None
    except Exception:
        return False


def log_metal_memory(context: str = "") -> tuple[float, float, float]:
    """Log memory state for Metal debugging.

    Returns:
        Tuple of (available_gb, rss_gb, vms_gb) or (0, 0, 0) on error.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        process = psutil.Process()
        proc_mem = process.memory_info()

        available_gb = mem.available / (1024**3)
        rss_gb = proc_mem.rss / (1024**3)
        vms_gb = proc_mem.vms / (1024**3)

        debug_logger.info(
            f"METAL_MEM [{context}] Avail: {available_gb:.2f} GB | RSS: {rss_gb:.3f} GB | VMS: {vms_gb:.1f} GB"
        )

        if available_gb < 8.0:
            debug_logger.warning(f"METAL_MEM WARNING [{context}]: LOW MEMORY! {available_gb:.2f} GB available")

        return available_gb, rss_gb, vms_gb

    except ImportError:
        debug_logger.warning(f"METAL_MEM [{context}] psutil not installed - cannot get memory info")
        return 0.0, 0.0, 0.0
    except Exception as e:
        debug_logger.warning(f"METAL_MEM [{context}] Error getting memory: {e}")
        return 0.0, 0.0, 0.0


def check_memory_and_gc(min_available_gb: float = 10.0, context: str = "") -> float:
    """Check memory and run gc.collect() if below threshold.

    Args:
        min_available_gb: Minimum available memory before triggering GC
        context: Logging context string

    Returns:
        Available memory in GB after any GC
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)

        if available_gb < min_available_gb:
            debug_logger.warning(
                f"METAL_MEM [{context}] Low memory ({available_gb:.2f} GB), forcing gc.collect()..."
            )
            gc.collect()

            # Check again after GC
            mem = psutil.virtual_memory()
            available_after = mem.available / (1024**3)
            freed = available_after - available_gb
            debug_logger.info(
                f"METAL_MEM [{context}] After GC: {available_after:.2f} GB (freed {freed:.2f} GB)"
            )
            return available_after

        return available_gb
    except Exception as e:
        debug_logger.warning(f"METAL_MEM [{context}] check_memory_and_gc error: {e}")
        return 0.0


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

        # 3D velocity flag
        ("use_3d_velocity", ctypes.c_int),  # If 1, inv_v_sq is [nx, ny, nt]
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

        # 3D velocity support
        ("use_3d_velocity", ctypes.c_int),
        ("nt_base", ctypes.c_int),
    ]


# Curved ray parameters struct - must match pstm_curved_ray.metal
class CurvedRayParams(ctypes.Structure):
    """Curved ray migration parameters matching Metal struct layout."""
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
        ("apply_aa", ctypes.c_int),
        ("aa_dominant_freq", ctypes.c_float),

        # Curved ray parameters (V(z) = V0 + k*z)
        ("v0", ctypes.c_float),  # Surface velocity (m/s)
        ("k", ctypes.c_float),   # Velocity gradient (1/s)

        # Dimensions
        ("n_traces", ctypes.c_int),
        ("n_samples", ctypes.c_int),
        ("nx", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("nt", ctypes.c_int),
    ]


# VTI anisotropy parameters struct - must match pstm_anisotropic_vti.metal
class VTIParams(ctypes.Structure):
    """VTI anisotropic migration parameters matching Metal struct layout."""
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
        ("apply_aa", ctypes.c_int),
        ("aa_dominant_freq", ctypes.c_float),

        # VTI anisotropy parameters
        ("eta_constant", ctypes.c_float),  # Constant eta value
        ("eta_is_1d", ctypes.c_int),       # 1 if eta_array is 1D (nt,), 0 if 3D (nx,ny,nt)

        # Dimensions
        ("n_traces", ctypes.c_int),
        ("n_samples", ctypes.c_int),
        ("nx", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("nt", ctypes.c_int),
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
        log_metal_memory("migrate_tile_start")

        # DIAGNOSTIC: Log trace geometry and amplitude stats
        debug_logger.info(f"TRACE DEBUG: n_traces={n_traces}, n_samples={n_samples}")
        debug_logger.info(f"TRACE DEBUG: amplitudes shape={traces.amplitudes.shape}, dtype={traces.amplitudes.dtype}")
        debug_logger.info(f"TRACE DEBUG: amplitudes min={traces.amplitudes.min():.6f}, max={traces.amplitudes.max():.6f}, mean={traces.amplitudes.mean():.6f}")
        debug_logger.info(f"TRACE DEBUG: source_x range=[{traces.source_x.min():.1f}, {traces.source_x.max():.1f}]")
        debug_logger.info(f"TRACE DEBUG: source_y range=[{traces.source_y.min():.1f}, {traces.source_y.max():.1f}]")
        debug_logger.info(f"TRACE DEBUG: midpoint_x range=[{traces.midpoint_x.min():.1f}, {traces.midpoint_x.max():.1f}]")
        debug_logger.info(f"TRACE DEBUG: midpoint_y range=[{traces.midpoint_y.min():.1f}, {traces.midpoint_y.max():.1f}]")

        # DIAGNOSTIC: Log output grid
        debug_logger.info(f"OUTPUT DEBUG: x_axis range=[{output.x_axis.min():.1f}, {output.x_axis.max():.1f}]")
        debug_logger.info(f"OUTPUT DEBUG: y_axis range=[{output.y_axis.min():.1f}, {output.y_axis.max():.1f}]")
        debug_logger.info(f"OUTPUT DEBUG: t_axis range=[{output.t_axis_ms.min():.1f}, {output.t_axis_ms.max():.1f}] ms")

        # Prepare input arrays as float32
        amplitudes = traces.amplitudes.astype(np.float32)
        source_x = traces.source_x.astype(np.float32)
        source_y = traces.source_y.astype(np.float32)
        receiver_x = traces.receiver_x.astype(np.float32)
        receiver_y = traces.receiver_y.astype(np.float32)
        midpoint_x = traces.midpoint_x.astype(np.float32)
        midpoint_y = traces.midpoint_y.astype(np.float32)

        # Create 2D coordinate grids for rotated grid support
        # For rotated grids, x_grid and y_grid contain the actual 2D coordinates
        # For axis-aligned grids, we create meshgrid from 1D axes
        if output.x_grid is not None and output.y_grid is not None:
            # Rotated grid: use provided 2D coordinates
            x_coords = output.x_grid.flatten().astype(np.float32)
            y_coords = output.y_grid.flatten().astype(np.float32)
        else:
            # Axis-aligned grid: create meshgrid and flatten
            xx, yy = np.meshgrid(output.x_axis, output.y_axis, indexing='ij')
            x_coords = xx.flatten().astype(np.float32)
            y_coords = yy.flatten().astype(np.float32)

        # Pre-compute time-dependent values
        t_axis_s = output.t_axis_ms / 1000.0  # Convert to seconds

        # DIAGNOSTIC: Log velocity details for debugging
        debug_logger.info(f"VELOCITY DEBUG (migrate_tile): is_1d={velocity.is_1d}, vrms shape={velocity.vrms.shape}")

        # Determine if we should use 3D velocity mode
        use_3d_velocity = False
        if velocity.is_1d:
            vrms = velocity.vrms  # (nt,)
            debug_logger.info(f"VELOCITY DEBUG: 1D velocity - min={vrms.min():.0f}, max={vrms.max():.0f}, mean={vrms.mean():.0f} m/s")
            inv_v_sq = (1.0 / (vrms ** 2)).astype(np.float32)
        else:
            # 3D velocity - check if lateral variation is significant
            full_vrms = velocity.vrms  # (nx, ny, nt)
            debug_logger.info(f"VELOCITY DEBUG: 3D velocity cube - shape={full_vrms.shape}")
            debug_logger.info(f"VELOCITY DEBUG: 3D velocity - min={full_vrms.min():.0f}, max={full_vrms.max():.0f}, mean={full_vrms.mean():.0f} m/s")

            # Check lateral variation - if max variation > 5%, use 3D mode
            center_ix, center_iy = nx//2, ny//2
            center_pillar = full_vrms[center_ix, center_iy, :]
            max_lateral_var = np.abs(full_vrms - center_pillar[np.newaxis, np.newaxis, :]).max()
            rel_variation = max_lateral_var / center_pillar.mean()

            if rel_variation > 0.05:  # More than 5% lateral variation
                use_3d_velocity = True
                debug_logger.info(f"VELOCITY DEBUG: Using FULL 3D velocity (lateral variation: {rel_variation*100:.1f}%)")

                # Compute 3D inv_v_sq array - shape (nx, ny, nt)
                # Flatten to (nx * ny * nt) in C-order matching Metal indexing
                inv_v_sq_3d = (1.0 / (full_vrms ** 2)).astype(np.float32)
                inv_v_sq = inv_v_sq_3d.flatten()  # Flattens in C-order (row-major)

                vel_size_mb = inv_v_sq.nbytes / 1024**2
                debug_logger.info(f"VELOCITY DEBUG: 3D inv_v_sq buffer size: {vel_size_mb:.1f} MB")
            else:
                # Lateral variation is small, use center pillar for efficiency
                debug_logger.info(f"VELOCITY DEBUG: Using center pillar (lateral variation only {rel_variation*100:.1f}%)")
                vrms = center_pillar
                inv_v_sq = (1.0 / (vrms ** 2)).astype(np.float32)

        t0_half_sq = ((t_axis_s / 2.0) ** 2).astype(np.float32)
        t0_s = t_axis_s.astype(np.float32)
        
        # Compute apertures for each time sample
        apertures = np.full(nt, config.max_aperture_m, dtype=np.float32)
        
        # Output arrays
        image_out = np.zeros((nx, ny, nt), dtype=np.float32)
        fold_out = np.zeros((nx, ny), dtype=np.int32)
        
        # Create Metal buffers
        total_buffer_bytes = (amplitudes.nbytes + source_x.nbytes + source_y.nbytes +
                              receiver_x.nbytes + receiver_y.nbytes + midpoint_x.nbytes +
                              midpoint_y.nbytes + image_out.nbytes + fold_out.nbytes +
                              x_coords.nbytes + y_coords.nbytes + t0_half_sq.nbytes +
                              inv_v_sq.nbytes + t0_s.nbytes + apertures.nbytes)
        debug_logger.info(f"METAL migrate_tile buffers: {total_buffer_bytes / 1024**2:.1f} MB total")
        log_metal_memory("migrate_tile_before_buffers")

        buffers = self._create_buffers(
            amplitudes, source_x, source_y, receiver_x, receiver_y,
            midpoint_x, midpoint_y, image_out, fold_out,
            x_coords, y_coords, t0_half_sq, inv_v_sq, t0_s, apertures
        )
        log_metal_memory("migrate_tile_after_buffers")

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
        params.apply_aa = 1 if config.aa_enabled else 0
        params.aa_dominant_freq = config.aa_dominant_freq if hasattr(config, 'aa_dominant_freq') else 30.0
        params.n_traces = n_traces
        params.n_samples = n_samples
        params.nx = nx
        params.ny = ny
        params.nt = nt
        params.use_3d_velocity = 1 if use_3d_velocity else 0

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
        log_metal_memory("migrate_tile_before_dispatch")

        # Dispatch
        encoder.dispatchThreads_threadsPerThreadgroup_(threads_per_grid, threads_per_threadgroup)
        encoder.endEncoding()

        # Execute and wait
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        log_metal_memory("migrate_tile_after_dispatch")

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

        # Explicit cleanup of Metal buffers to release unified memory
        del buffers, params_buffer, command_buffer, encoder
        del image_buf, fold_buf, image_result, fold_result
        gc.collect()
        log_metal_memory("migrate_tile_after_cleanup")

        return KernelMetrics(
            n_traces_processed=n_traces,
            n_samples_output=nx * ny * nt,
            compute_time_s=compute_time,
        )

    def _create_buffer(self, arr: np.ndarray, name: str = "") -> "Metal.MTLBuffer":
        """Create a Metal buffer from a numpy array.

        Args:
            arr: NumPy array to create buffer from
            name: Optional name for logging

        Note: This creates a copy via tobytes(). For large arrays (>100MB),
        consider using memory-mapped buffers or pre-allocated pools.
        """
        # Ensure contiguous - may create copy
        arr_contig = np.ascontiguousarray(arr)
        arr_bytes = arr.nbytes

        # Log memory for large buffers
        if arr_bytes > 100 * 1024 * 1024:  # > 100 MB
            debug_logger.debug(
                f"METAL_BUF [{name}] Creating {arr_bytes / 1024**2:.1f} MB buffer "
                f"(shape={arr.shape}, dtype={arr.dtype})"
            )

        # Create buffer - tobytes() creates another copy
        buf = self._device.newBufferWithBytes_length_options_(
            arr_contig.tobytes(),
            arr_bytes,
            Metal.MTLResourceStorageModeShared
        )

        # Explicit cleanup of the bytes copy (helps GC)
        del arr_contig

        return buf

    def _create_buffers(
        self,
        amplitudes, source_x, source_y, receiver_x, receiver_y,
        midpoint_x, midpoint_y, image, fold,
        x_coords, y_coords, t0_half_sq, inv_v_sq, t0_s, apertures
    ):
        """Create Metal buffers from numpy arrays."""
        names = [
            "amplitudes", "source_x", "source_y", "receiver_x", "receiver_y",
            "midpoint_x", "midpoint_y", "image", "fold",
            "x_coords", "y_coords", "t0_half_sq", "inv_v_sq", "t0_s", "apertures"
        ]
        arrays = [
            amplitudes, source_x, source_y, receiver_x, receiver_y,
            midpoint_x, midpoint_y, image, fold,
            x_coords, y_coords, t0_half_sq, inv_v_sq, t0_s, apertures
        ]

        # Check memory before creating large buffers
        total_bytes = sum(a.nbytes for a in arrays)
        total_mb = total_bytes / 1024**2
        debug_logger.info(f"METAL_BUF Creating {len(arrays)} buffers totaling {total_mb:.1f} MB")

        # Force GC before large allocation if memory is low
        check_memory_and_gc(min_available_gb=12.0, context="before_buffer_creation")

        buffers = []
        for arr, name in zip(arrays, names):
            buf = self._create_buffer(arr, name)
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

        # Log via both loggers to ensure visibility
        logger.info(
            f"METAL TV migrate_tile: {n_traces} traces -> {nx}x{ny}x{total_output_samples} output "
            f"({len(windows)} windows)"
        )
        debug_logger.info(
            f"METAL TV migrate_tile: {n_traces} traces -> {nx}x{ny}x{total_output_samples} output "
            f"({len(windows)} windows)"
        )
        log_metal_memory("tv_kernel_start")

        # DIAGNOSTIC: Log trace geometry and amplitude stats
        debug_logger.info(f"TRACE DEBUG (TV): n_traces={n_traces}, n_samples={n_samples}")
        debug_logger.info(f"TRACE DEBUG (TV): amplitudes shape={traces.amplitudes.shape}, dtype={traces.amplitudes.dtype}")
        debug_logger.info(f"TRACE DEBUG (TV): amplitudes min={traces.amplitudes.min():.6f}, max={traces.amplitudes.max():.6f}, mean={traces.amplitudes.mean():.6f}")
        debug_logger.info(f"TRACE DEBUG (TV): source_x range=[{traces.source_x.min():.1f}, {traces.source_x.max():.1f}]")
        debug_logger.info(f"TRACE DEBUG (TV): source_y range=[{traces.source_y.min():.1f}, {traces.source_y.max():.1f}]")
        debug_logger.info(f"TRACE DEBUG (TV): midpoint_x range=[{traces.midpoint_x.min():.1f}, {traces.midpoint_x.max():.1f}]")
        debug_logger.info(f"TRACE DEBUG (TV): midpoint_y range=[{traces.midpoint_y.min():.1f}, {traces.midpoint_y.max():.1f}]")

        # DIAGNOSTIC: Log output grid
        debug_logger.info(f"OUTPUT DEBUG (TV): x_axis range=[{output.x_axis.min():.1f}, {output.x_axis.max():.1f}]")
        debug_logger.info(f"OUTPUT DEBUG (TV): y_axis range=[{output.y_axis.min():.1f}, {output.y_axis.max():.1f}]")

        # DIAGNOSTIC: Log time windows
        for i, w in enumerate(windows):
            debug_logger.info(f"WINDOW DEBUG: [{i}] t={w.t_start_ms:.0f}-{w.t_end_ms:.0f}ms, dt_eff={w.dt_effective_ms:.1f}ms, ds_factor={w.downsample_factor}, n_samples={w.n_samples}")

        # Check memory and potentially force GC before heavy allocations
        check_memory_and_gc(min_available_gb=12.0, context="tv_kernel_pre_alloc")

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
        # Note: astype() creates copies which can be expensive for large arrays
        amp_size_mb = traces.amplitudes.nbytes / 1024**2
        debug_logger.debug(f"METAL TV converting amplitudes: {amp_size_mb:.1f} MB, dtype={traces.amplitudes.dtype}")

        # Use copy=False to avoid copy if already float32
        amplitudes = np.asarray(traces.amplitudes, dtype=np.float32)
        source_x = np.asarray(traces.source_x, dtype=np.float32)
        source_y = np.asarray(traces.source_y, dtype=np.float32)
        receiver_x = np.asarray(traces.receiver_x, dtype=np.float32)
        receiver_y = np.asarray(traces.receiver_y, dtype=np.float32)
        midpoint_x = np.asarray(traces.midpoint_x, dtype=np.float32)
        midpoint_y = np.asarray(traces.midpoint_y, dtype=np.float32)

        # Create 2D coordinate grids for rotated grid support
        if output.x_grid is not None and output.y_grid is not None:
            x_coords = output.x_grid.flatten().astype(np.float32)
            y_coords = output.y_grid.flatten().astype(np.float32)
        else:
            xx, yy = np.meshgrid(output.x_axis, output.y_axis, indexing='ij')
            x_coords = xx.flatten().astype(np.float32)
            y_coords = yy.flatten().astype(np.float32)

        # Velocity array - support both 1D and 3D velocity
        # DIAGNOSTIC: Log velocity details for debugging
        debug_logger.info(f"VELOCITY DEBUG (TV): is_1d={velocity.is_1d}, vrms shape={velocity.vrms.shape}")

        use_3d_velocity = False
        nt_base = velocity.vrms.shape[-1]  # Time dimension is always last

        if velocity.is_1d:
            vrms = velocity.vrms.astype(np.float32)
            debug_logger.info(f"VELOCITY DEBUG (TV): 1D velocity - min={vrms.min():.0f}, max={vrms.max():.0f}, mean={vrms.mean():.0f} m/s")
        else:
            # 3D velocity - check if lateral variation is significant
            full_vrms = velocity.vrms  # (nx, ny, nt)
            debug_logger.info(f"VELOCITY DEBUG (TV): 3D velocity cube - shape={full_vrms.shape}")
            debug_logger.info(f"VELOCITY DEBUG (TV): 3D velocity - min={full_vrms.min():.0f}, max={full_vrms.max():.0f}, mean={full_vrms.mean():.0f} m/s")

            # Check lateral variation - if max variation > 5%, use 3D mode
            center_ix, center_iy = nx//2, ny//2
            center_pillar = full_vrms[center_ix, center_iy, :]
            max_lateral_var = np.abs(full_vrms - center_pillar[np.newaxis, np.newaxis, :]).max()
            rel_variation = max_lateral_var / center_pillar.mean()

            if rel_variation > 0.05:  # More than 5% lateral variation
                use_3d_velocity = True
                debug_logger.info(f"VELOCITY DEBUG (TV): Using FULL 3D velocity (lateral variation: {rel_variation*100:.1f}%)")

                # Flatten 3D velocity to match Metal indexing: [nx * ny * nt]
                vrms = full_vrms.flatten().astype(np.float32)

                vel_size_mb = vrms.nbytes / 1024**2
                debug_logger.info(f"VELOCITY DEBUG (TV): 3D vrms buffer size: {vel_size_mb:.1f} MB")
            else:
                # Lateral variation is small, use center pillar for efficiency
                debug_logger.info(f"VELOCITY DEBUG (TV): Using center pillar (lateral variation only {rel_variation*100:.1f}%)")
                vrms = center_pillar.astype(np.float32)

        # Output arrays (time-variant sized)
        image_out = np.zeros((nx, ny, total_output_samples), dtype=np.float32)
        fold_out = np.zeros((nx, ny), dtype=np.int32)

        # Store sizes for later use when reading results (allows deleting arrays earlier)
        image_out_nbytes = image_out.nbytes
        fold_out_nbytes = fold_out.nbytes

        # Create buffers
        log_metal_memory("tv_before_buffers")

        # Calculate buffer sizes for logging
        buffer_sizes_mb = {
            'amplitudes': amplitudes.nbytes / 1024**2,
            'geometry': (source_x.nbytes + source_y.nbytes + receiver_x.nbytes + receiver_y.nbytes + midpoint_x.nbytes + midpoint_y.nbytes) / 1024**2,
            'image_out': image_out.nbytes / 1024**2,
            'coords': (x_coords.nbytes + y_coords.nbytes) / 1024**2,
            'vrms': vrms.nbytes / 1024**2,
        }
        total_buffer_mb = sum(buffer_sizes_mb.values())
        debug_logger.info(f"METAL TV buffers: {total_buffer_mb:.1f} MB total "
                         f"(amp: {buffer_sizes_mb['amplitudes']:.1f}, geom: {buffer_sizes_mb['geometry']:.1f}, "
                         f"img: {buffer_sizes_mb['image_out']:.1f})")

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

        # CRITICAL: Delete source numpy arrays AFTER buffers are created
        # This releases ~2 GB of amplitudes array immediately
        del amplitudes, source_x, source_y, receiver_x, receiver_y
        del midpoint_x, midpoint_y, x_coords, y_coords, vrms
        del image_out, fold_out  # Sizes stored earlier for reading results

        log_metal_memory("tv_after_buffers")

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
        params.apply_aa = 1 if config.aa_enabled else 0
        params.aa_dominant_freq = config.aa_dominant_freq if hasattr(config, 'aa_dominant_freq') else 30.0
        params.n_traces = n_traces
        params.n_samples = n_samples
        params.nx = nx
        params.ny = ny
        params.n_windows = n_windows
        params.total_output_samples = total_output_samples
        params.use_3d_velocity = 1 if use_3d_velocity else 0
        params.nt_base = nt_base

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

        # Read results (using stored sizes since source arrays were deleted)
        image_buf = buffers[7].contents().as_buffer(image_out_nbytes)
        fold_buf = buffers[8].contents().as_buffer(fold_out_nbytes)

        image_result = np.frombuffer(image_buf, dtype=np.float32).reshape(nx, ny, total_output_samples)
        fold_result = np.frombuffer(fold_buf, dtype=np.int32).reshape(nx, ny)

        # Store time-variant result (caller must resample to uniform)
        # We return the raw TV output - resampling is done by caller
        # Make copies before clearing buffer references
        output.image = image_result.astype(np.float64)
        output.fold[:] = fold_result

        compute_time = time.perf_counter() - start_time
        debug_logger.info(f"METAL TV completed in {compute_time:.3f}s")

        # Explicit cleanup of Metal buffers to release unified memory
        # First, break any references to buffer contents
        del image_buf, fold_buf, image_result, fold_result

        # Log memory before buffer cleanup
        avail_before, rss_before, vms_before = log_metal_memory("tv_before_buffer_cleanup")

        # Try to mark buffers as purgeable before releasing (allows OS to reclaim)
        try:
            for buf in buffers:
                if buf is not None and hasattr(buf, 'setPurgeableState_'):
                    buf.setPurgeableState_(Metal.MTLPurgeableStateEmpty)
        except Exception as e:
            debug_logger.debug(f"Could not set purgeable state: {e}")

        # Clear each buffer explicitly
        for i in range(len(buffers)):
            buffers[i] = None
        buffers.clear()
        del buffers

        # Clear other Metal objects
        if params_buffer is not None:
            try:
                if hasattr(params_buffer, 'setPurgeableState_'):
                    params_buffer.setPurgeableState_(Metal.MTLPurgeableStateEmpty)
            except:
                pass
        params_buffer = None
        windows_buffer = None
        command_buffer = None
        encoder = None
        tv_pipeline = None
        tv_kernel_func = None

        # Drain autorelease pool to release Objective-C objects
        drain_autorelease_pool()

        # Force garbage collection multiple times
        gc.collect()
        gc.collect()

        # Drain again after GC
        drain_autorelease_pool()

        # Log memory after cleanup
        avail_after, rss_after, vms_after = log_metal_memory("tv_after_buffer_cleanup")
        freed_gb = avail_after - avail_before
        debug_logger.info(f"METAL TV cleanup freed: {freed_gb:.2f} GB (VMS change: {vms_after - vms_before:.1f} GB)")

        return KernelMetrics(
            n_traces_processed=n_traces,
            n_samples_output=nx * ny * total_output_samples,
            compute_time_s=compute_time,
        )

    def migrate_tile_curved_ray(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: KernelConfig,
    ) -> KernelMetrics:
        """
        Migrate traces using curved ray traveltime (V(z) = V0 + k*z).

        Args:
            traces: Input trace block
            output: Output tile
            velocity: Velocity model (used for aperture, not traveltime)
            config: Kernel configuration with curved_ray_v0 and curved_ray_k

        Returns:
            Kernel metrics
        """
        if not self._initialized:
            self.initialize(config)

        start_time = time.perf_counter()

        n_traces = traces.n_traces
        nx, ny, nt = output.nx, output.ny, output.nt
        n_samples = traces.n_samples

        debug_logger.info(
            f"METAL curved_ray migrate: {n_traces} traces -> {nx}x{ny}x{nt} output "
            f"(v0={config.curved_ray_v0}, k={config.curved_ray_k})"
        )
        log_metal_memory("curved_ray_start")

        # Get curved ray kernel
        kernel_name = "pstm_migrate_curved_ray_simd" if self._use_simd else "pstm_migrate_curved_ray"
        cr_kernel_func = self._library.newFunctionWithName_(kernel_name)
        if cr_kernel_func is None:
            raise RuntimeError(f"Curved ray kernel function '{kernel_name}' not found in library")

        cr_pipeline, error = self._device.newComputePipelineStateWithFunction_error_(
            cr_kernel_func, None
        )
        if error:
            raise RuntimeError(f"Failed to create curved ray pipeline: {error}")

        # Prepare input arrays as float32
        amplitudes = traces.amplitudes.astype(np.float32)
        source_x = traces.source_x.astype(np.float32)
        source_y = traces.source_y.astype(np.float32)
        receiver_x = traces.receiver_x.astype(np.float32)
        receiver_y = traces.receiver_y.astype(np.float32)
        midpoint_x = traces.midpoint_x.astype(np.float32)
        midpoint_y = traces.midpoint_y.astype(np.float32)

        # Create 2D coordinate grids for rotated grid support
        if output.x_grid is not None and output.y_grid is not None:
            x_coords = output.x_grid.flatten().astype(np.float32)
            y_coords = output.y_grid.flatten().astype(np.float32)
        else:
            xx, yy = np.meshgrid(output.x_axis, output.y_axis, indexing='ij')
            x_coords = xx.flatten().astype(np.float32)
            y_coords = yy.flatten().astype(np.float32)

        # Time axis
        t0_s = (output.t_axis_ms / 1000.0).astype(np.float32)

        # Apertures (can still use velocity for aperture calculation)
        apertures = np.full(nt, config.max_aperture_m, dtype=np.float32)

        # Output arrays
        image_out = np.zeros((nx, ny, nt), dtype=np.float32)
        fold_out = np.zeros((nx, ny), dtype=np.int32)

        # Create buffers
        buffers = [
            self._create_buffer(amplitudes),       # 0
            self._create_buffer(source_x),         # 1
            self._create_buffer(source_y),         # 2
            self._create_buffer(receiver_x),       # 3
            self._create_buffer(receiver_y),       # 4
            self._create_buffer(midpoint_x),       # 5
            self._create_buffer(midpoint_y),       # 6
            self._create_buffer(image_out),        # 7
            self._create_buffer(fold_out),         # 8
            self._create_buffer(x_coords),         # 9
            self._create_buffer(y_coords),         # 10
            self._create_buffer(t0_s),             # 11
            self._create_buffer(apertures),        # 12
        ]

        # Create params
        params = CurvedRayParams()
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
        params.apply_aa = 1 if config.aa_enabled else 0
        params.aa_dominant_freq = getattr(config, 'aa_dominant_freq', 30.0)
        params.v0 = config.curved_ray_v0
        params.k = config.curved_ray_k
        params.n_traces = n_traces
        params.n_samples = n_samples
        params.nx = nx
        params.ny = ny
        params.nt = nt

        params_bytes = bytes(params)
        params_buffer = self._device.newBufferWithBytes_length_options_(
            params_bytes, len(params_bytes), Metal.MTLResourceStorageModeShared
        )

        # Create command buffer
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(cr_pipeline)

        # Set buffers
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 13)

        # Thread grid
        threads_per_grid = Metal.MTLSize()
        threads_per_grid.width = nx
        threads_per_grid.height = ny
        threads_per_grid.depth = nt

        max_threads = cr_pipeline.maxTotalThreadsPerThreadgroup()
        tg_width = min(8, nx)
        tg_height = min(8, ny)
        tg_depth = min(max_threads // (tg_width * tg_height), nt)

        threads_per_threadgroup = Metal.MTLSize()
        threads_per_threadgroup.width = tg_width
        threads_per_threadgroup.height = tg_height
        threads_per_threadgroup.depth = max(1, tg_depth)

        debug_logger.info(
            f"METAL curved_ray dispatch: grid=({nx},{ny},{nt}), "
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

        image_result = np.frombuffer(image_buf, dtype=np.float32).reshape(nx, ny, nt)
        fold_result = np.frombuffer(fold_buf, dtype=np.int32).reshape(nx, ny)

        output.image[:] = image_result.astype(np.float64)
        output.fold[:] = fold_result

        compute_time = time.perf_counter() - start_time
        debug_logger.info(f"METAL curved_ray completed in {compute_time:.3f}s")

        # Explicit cleanup of Metal buffers
        del buffers, params_buffer, command_buffer, encoder
        del image_buf, fold_buf, image_result, fold_result
        gc.collect()
        log_metal_memory("curved_ray_after_cleanup")

        return KernelMetrics(
            n_traces_processed=n_traces,
            n_samples_output=nx * ny * nt,
            compute_time_s=compute_time,
        )

    def migrate_tile_vti(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: KernelConfig,
    ) -> KernelMetrics:
        """
        Migrate traces using VTI anisotropic traveltime (Alkhalifah-Tsvankin eta).

        Args:
            traces: Input trace block
            output: Output tile
            velocity: Velocity model (V_nmo)
            config: Kernel configuration with VTI eta parameters

        Returns:
            Kernel metrics
        """
        if not self._initialized:
            self.initialize(config)

        start_time = time.perf_counter()

        n_traces = traces.n_traces
        nx, ny, nt = output.nx, output.ny, output.nt
        n_samples = traces.n_samples

        # Determine eta usage
        use_eta_array = config.vti_eta_array is not None
        eta_is_1d = config.vti_eta_is_1d

        debug_logger.info(
            f"METAL VTI migrate: {n_traces} traces -> {nx}x{ny}x{nt} output "
            f"(eta_constant={config.vti_eta_constant}, eta_array={'yes' if use_eta_array else 'no'})"
        )
        log_metal_memory("vti_start")

        # Get VTI kernel
        kernel_name = "pstm_migrate_vti_simd" if self._use_simd else "pstm_migrate_vti"
        vti_kernel_func = self._library.newFunctionWithName_(kernel_name)
        if vti_kernel_func is None:
            raise RuntimeError(f"VTI kernel function '{kernel_name}' not found in library")

        vti_pipeline, error = self._device.newComputePipelineStateWithFunction_error_(
            vti_kernel_func, None
        )
        if error:
            raise RuntimeError(f"Failed to create VTI pipeline: {error}")

        # Prepare input arrays as float32
        amplitudes = traces.amplitudes.astype(np.float32)
        source_x = traces.source_x.astype(np.float32)
        source_y = traces.source_y.astype(np.float32)
        receiver_x = traces.receiver_x.astype(np.float32)
        receiver_y = traces.receiver_y.astype(np.float32)
        midpoint_x = traces.midpoint_x.astype(np.float32)
        midpoint_y = traces.midpoint_y.astype(np.float32)

        # Create 2D coordinate grids for rotated grid support
        if output.x_grid is not None and output.y_grid is not None:
            x_coords = output.x_grid.flatten().astype(np.float32)
            y_coords = output.y_grid.flatten().astype(np.float32)
        else:
            xx, yy = np.meshgrid(output.x_axis, output.y_axis, indexing='ij')
            x_coords = xx.flatten().astype(np.float32)
            y_coords = yy.flatten().astype(np.float32)

        # Velocity and traveltime arrays
        t_axis_s = output.t_axis_ms / 1000.0
        if velocity.is_1d:
            vrms = velocity.vrms
        else:
            vrms = velocity.vrms[nx//2, ny//2, :]

        t0_half_sq = ((t_axis_s / 2.0) ** 2).astype(np.float32)
        inv_v_sq = (1.0 / (vrms ** 2)).astype(np.float32)
        t0_s = t_axis_s.astype(np.float32)

        # Eta array - if provided, use it; otherwise create constant array
        if use_eta_array:
            eta_array = config.vti_eta_array.astype(np.float32)
        else:
            # Create constant eta array of size nt (1D)
            eta_array = np.full(nt, config.vti_eta_constant, dtype=np.float32)

        # Apertures
        apertures = np.full(nt, config.max_aperture_m, dtype=np.float32)

        # Output arrays
        image_out = np.zeros((nx, ny, nt), dtype=np.float32)
        fold_out = np.zeros((nx, ny), dtype=np.int32)

        # Create buffers
        buffers = [
            self._create_buffer(amplitudes),       # 0
            self._create_buffer(source_x),         # 1
            self._create_buffer(source_y),         # 2
            self._create_buffer(receiver_x),       # 3
            self._create_buffer(receiver_y),       # 4
            self._create_buffer(midpoint_x),       # 5
            self._create_buffer(midpoint_y),       # 6
            self._create_buffer(image_out),        # 7
            self._create_buffer(fold_out),         # 8
            self._create_buffer(x_coords),         # 9
            self._create_buffer(y_coords),         # 10
            self._create_buffer(t0_half_sq),       # 11
            self._create_buffer(inv_v_sq),         # 12
            self._create_buffer(t0_s),             # 13
            self._create_buffer(apertures),        # 14
            self._create_buffer(eta_array),        # 15
        ]

        # Create params
        params = VTIParams()
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
        params.apply_aa = 1 if config.aa_enabled else 0
        params.aa_dominant_freq = getattr(config, 'aa_dominant_freq', 30.0)
        params.eta_constant = config.vti_eta_constant
        params.eta_is_1d = 1 if eta_is_1d else 0
        params.n_traces = n_traces
        params.n_samples = n_samples
        params.nx = nx
        params.ny = ny
        params.nt = nt

        params_bytes = bytes(params)
        params_buffer = self._device.newBufferWithBytes_length_options_(
            params_bytes, len(params_bytes), Metal.MTLResourceStorageModeShared
        )

        # Create command buffer
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(vti_pipeline)

        # Set buffers
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 16)

        # Thread grid
        threads_per_grid = Metal.MTLSize()
        threads_per_grid.width = nx
        threads_per_grid.height = ny
        threads_per_grid.depth = nt

        max_threads = vti_pipeline.maxTotalThreadsPerThreadgroup()
        tg_width = min(8, nx)
        tg_height = min(8, ny)
        tg_depth = min(max_threads // (tg_width * tg_height), nt)

        threads_per_threadgroup = Metal.MTLSize()
        threads_per_threadgroup.width = tg_width
        threads_per_threadgroup.height = tg_height
        threads_per_threadgroup.depth = max(1, tg_depth)

        debug_logger.info(
            f"METAL VTI dispatch: grid=({nx},{ny},{nt}), "
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

        image_result = np.frombuffer(image_buf, dtype=np.float32).reshape(nx, ny, nt)
        fold_result = np.frombuffer(fold_buf, dtype=np.int32).reshape(nx, ny)

        output.image[:] = image_result.astype(np.float64)
        output.fold[:] = fold_result

        compute_time = time.perf_counter() - start_time
        debug_logger.info(f"METAL VTI completed in {compute_time:.3f}s")

        # Explicit cleanup of Metal buffers
        del buffers, params_buffer, command_buffer, encoder
        del image_buf, fold_buf, image_result, fold_result
        gc.collect()
        log_metal_memory("vti_after_cleanup")

        return KernelMetrics(
            n_traces_processed=n_traces,
            n_samples_output=nx * ny * nt,
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
