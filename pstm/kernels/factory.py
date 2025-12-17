"""
Kernel factory for PSTM.

Handles kernel discovery, selection, and instantiation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from pstm.config.models import ComputeBackend
from pstm.kernels.base import (
    KernelConfig,
    KernelMetrics,
    MigrationKernel,
    OutputTile,
    TraceBlock,
    VelocitySlice,
)
from pstm.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# Registry of available backends
_BACKEND_REGISTRY: dict[ComputeBackend, type] = {}


def _register_backends() -> None:
    """Register all available backends."""
    global _BACKEND_REGISTRY

    # Always available: NumPy reference
    try:
        from pstm.kernels.numpy_reference import NumpyReferenceKernel

        _BACKEND_REGISTRY[ComputeBackend.NUMPY] = NumpyReferenceKernel
        logger.debug("Registered NumPy reference kernel")
    except ImportError as e:
        logger.warning(f"NumPy kernel not available: {e}")

    # Numba CPU - use optimized version (40% faster)
    try:
        from pstm.kernels.numba_cpu_optimized import OptimizedNumbaKernel

        _BACKEND_REGISTRY[ComputeBackend.NUMBA_CPU] = OptimizedNumbaKernel
        logger.debug("Registered Optimized Numba CPU kernel")
    except ImportError as e:
        # Fall back to original if optimized not available
        logger.debug(f"Optimized kernel not available, trying original: {e}")
        try:
            from pstm.kernels.numba_cpu import NumbaKernel

            _BACKEND_REGISTRY[ComputeBackend.NUMBA_CPU] = NumbaKernel
            logger.debug("Registered Numba CPU kernel (original)")
        except ImportError as e2:
            logger.warning(f"Numba kernel not available: {e2}")

    # MLX Metal (optional, Apple Silicon only)
    try:
        import mlx  # noqa: F401

        from pstm.kernels.mlx_metal import MLXKernel

        _BACKEND_REGISTRY[ComputeBackend.MLX_METAL] = MLXKernel
        logger.debug("Registered MLX Metal kernel")
    except ImportError:
        logger.debug("MLX kernel not available (mlx not installed)")
    except Exception as e:
        logger.debug(f"MLX kernel not available: {e}")

    # Metal C++ (optional, Apple Silicon, requires compiled module)
    try:
        from pstm.kernels.metal_cpp import MetalCppKernel, is_metal_cpp_available

        if is_metal_cpp_available():
            _BACKEND_REGISTRY[ComputeBackend.METAL_CPP] = MetalCppKernel
            logger.debug("Registered Metal C++ kernel")
        else:
            logger.debug("Metal C++ kernel: module available but Metal not supported")
    except ImportError:
        logger.debug("Metal C++ kernel not available (module not built)")
    except Exception as e:
        logger.debug(f"Metal C++ kernel not available: {e}")


def get_available_backends() -> list[ComputeBackend]:
    """
    Get list of available compute backends.

    Returns:
        List of available backends
    """
    if not _BACKEND_REGISTRY:
        _register_backends()

    return list(_BACKEND_REGISTRY.keys())


def is_backend_available(backend: ComputeBackend) -> bool:
    """
    Check if a specific backend is available.

    Args:
        backend: Backend to check

    Returns:
        True if available
    """
    if not _BACKEND_REGISTRY:
        _register_backends()

    return backend in _BACKEND_REGISTRY


def create_kernel(backend: ComputeBackend | str) -> MigrationKernel:
    """
    Create a kernel instance for the specified backend.

    Args:
        backend: Compute backend to use (enum or string)

    Returns:
        Kernel instance

    Raises:
        ValueError: If backend is not available
    """
    if not _BACKEND_REGISTRY:
        _register_backends()

    # Handle string input - fail fast if invalid
    if isinstance(backend, str):
        logger.info(f"KERNEL FACTORY: Received backend string '{backend}'")
        from pstm.config.backends import parse_backend
        backend = parse_backend(backend)
        logger.info(f"KERNEL FACTORY: Mapped to enum {backend}")

    logger.info(f"KERNEL FACTORY: Requested backend = {backend.value}")
    logger.info(f"KERNEL FACTORY: Available backends = {[b.value for b in _BACKEND_REGISTRY.keys()]}")

    if backend == ComputeBackend.AUTO:
        logger.info("KERNEL FACTORY: AUTO mode - selecting best backend...")
        backend = select_best_backend()
        logger.info(f"KERNEL FACTORY: AUTO selected = {backend.value}")

    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Backend '{backend.value}' not available. "
            f"Available: {[b.value for b in get_available_backends()]}"
        )

    kernel_class = _BACKEND_REGISTRY[backend]
    logger.info(f"KERNEL FACTORY: Creating kernel class = {kernel_class.__name__}")
    logger.info(f"KERNEL FACTORY: Kernel module = {kernel_class.__module__}")

    kernel = kernel_class()
    logger.info(f"KERNEL FACTORY: Kernel instance created = {type(kernel).__name__}")

    return kernel


def select_best_backend() -> ComputeBackend:
    """
    Select the best available backend.

    Priority: Metal C++ > Numba CPU > MLX Metal > NumPy

    Note: Metal C++ is preferred on Apple Silicon for best GPU performance.
    Numba CPU is preferred over MLX Metal because:
    - Numba's parallel JIT releases the GIL, allowing UI updates
    - MLX kernel currently uses Python for loops which are slow and block the GIL
    - Numba achieves ~50k+ traces/s vs MLX's ~40 traces/s in current implementation

    Returns:
        Best available backend
    """
    if not _BACKEND_REGISTRY:
        _register_backends()

    # Priority order - Metal C++ first for GPU, then Numba for CPU
    priority = [
        ComputeBackend.METAL_CPP,
        ComputeBackend.NUMBA_CPU,
        ComputeBackend.MLX_METAL,
        ComputeBackend.NUMPY,
    ]

    for backend in priority:
        if backend in _BACKEND_REGISTRY:
            logger.info(f"Auto-selected backend: {backend.value}")
            return backend

    raise RuntimeError("No compute backends available")


def benchmark_kernel(
    kernel: MigrationKernel,
    config: KernelConfig,
    n_traces: int = 1000,
    n_samples: int = 2000,
    tile_size: int = 20,
    n_iterations: int = 3,
) -> float:
    """
    Benchmark a kernel with synthetic data.

    Args:
        kernel: Kernel to benchmark
        config: Kernel configuration
        n_traces: Number of synthetic traces
        n_samples: Samples per trace
        tile_size: Output tile dimensions
        n_iterations: Number of benchmark iterations

    Returns:
        Average samples per second
    """
    logger.info(f"Benchmarking {kernel.name}...")

    # Initialize kernel
    kernel.initialize(config)

    # Create synthetic data
    rng = np.random.default_rng(42)

    # Random geometry in 1km x 1km area
    traces = TraceBlock(
        amplitudes=rng.standard_normal((n_traces, n_samples)).astype(np.float32),
        source_x=rng.uniform(0, 1000, n_traces),
        source_y=rng.uniform(0, 1000, n_traces),
        receiver_x=rng.uniform(0, 1000, n_traces),
        receiver_y=rng.uniform(0, 1000, n_traces),
        offset=rng.uniform(100, 3000, n_traces),
        midpoint_x=rng.uniform(0, 1000, n_traces),
        midpoint_y=rng.uniform(0, 1000, n_traces),
        sample_rate_ms=2.0,
        start_time_ms=0.0,
    )

    # Output tile
    ox = np.linspace(200, 800, tile_size)
    oy = np.linspace(200, 800, tile_size)
    ot = np.linspace(0, 4000, n_samples // 2)

    output = OutputTile(
        image=np.zeros((tile_size, tile_size, len(ot)), dtype=np.float64),
        fold=np.zeros((tile_size, tile_size), dtype=np.int32),
        x_axis=ox,
        y_axis=oy,
        t_axis_ms=ot,
    )

    # Velocity
    velocity = VelocitySlice(
        vrms=np.linspace(1500, 3000, len(ot)),
        is_1d=True,
        t_axis_ms=ot,
    )

    # Warm-up run
    output.reset()
    kernel.migrate_tile(traces, output, velocity)
    kernel.synchronize()

    # Benchmark runs
    times = []
    for i in range(n_iterations):
        output.reset()
        start = time.perf_counter()
        stats = kernel.migrate_tile(traces, output, velocity)
        kernel.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    samples_per_second = output.image.size / avg_time

    logger.info(
        f"  {kernel.name}: {samples_per_second:.2e} samples/s "
        f"(avg={avg_time:.3f}s, std={std_time:.3f}s)"
    )

    kernel.cleanup()

    return samples_per_second


def benchmark_all_backends(
    config: KernelConfig | None = None,
    **kwargs,
) -> dict[ComputeBackend, float]:
    """
    Benchmark all available backends.

    Args:
        config: Kernel configuration (uses defaults if None)
        **kwargs: Additional arguments passed to benchmark_kernel

    Returns:
        Dictionary of backend -> samples per second
    """
    if config is None:
        config = KernelConfig()

    results = {}

    for backend in get_available_backends():
        try:
            kernel = create_kernel(backend)
            perf = benchmark_kernel(kernel, config, **kwargs)
            results[backend] = perf
        except Exception as e:
            logger.warning(f"Failed to benchmark {backend.value}: {e}")

    return results


def auto_select_with_benchmark(
    config: KernelConfig | None = None,
) -> tuple[ComputeBackend, MigrationKernel]:
    """
    Auto-select best backend using benchmark.

    Args:
        config: Kernel configuration

    Returns:
        Tuple of (selected backend, initialized kernel)
    """
    if config is None:
        config = KernelConfig()

    logger.info("Running backend benchmarks for auto-selection...")

    results = benchmark_all_backends(
        config,
        n_traces=500,
        n_samples=1000,
        tile_size=10,
        n_iterations=2,
    )

    if not results:
        raise RuntimeError("No backends available")

    # Select fastest
    best_backend = max(results, key=results.get)
    logger.info(f"Selected backend: {best_backend.value}")

    # Create and initialize kernel
    kernel = create_kernel(best_backend)
    kernel.initialize(config)

    return best_backend, kernel


# Initialize registry on import
_register_backends()
