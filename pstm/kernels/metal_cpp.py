"""
Metal C++ kernel for PSTM migration.

High-performance GPU kernel using Apple Metal via C++ and pybind11.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from pstm.kernels.base import (
    KernelCapability,
    KernelConfig,
    KernelMetrics,
    OutputTile,
    TraceBlock,
    VelocitySlice,
)
from pstm.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class MetalCppKernel:
    """
    Metal C++ GPU kernel for PSTM migration.

    Uses native Metal compute shaders via C++ bindings for maximum performance
    on Apple Silicon.
    """

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._initialized = False
        self._metal_available: bool | None = None

    @property
    def name(self) -> str:
        return "Metal C++"

    @property
    def capabilities(self) -> set[KernelCapability]:
        return {KernelCapability.FP64, KernelCapability.ASYNC}

    def _check_availability(self) -> bool:
        """Check if Metal module is available."""
        if self._metal_available is not None:
            return self._metal_available

        try:
            from pstm.metal.python import is_available
            self._metal_available = is_available()
            if self._metal_available:
                from pstm.metal.python import get_device_info
                info = get_device_info()
                logger.info(f"Metal device: {info.get('device_name', 'Unknown')}")
                logger.info(f"Metal memory: {info.get('device_memory_gb', 0):.1f} GB")
            else:
                logger.warning("Metal not available on this system")
        except ImportError as e:
            logger.warning(f"Metal module not built: {e}")
            self._metal_available = False
        except Exception as e:
            logger.warning(f"Metal initialization error: {e}")
            self._metal_available = False

        return self._metal_available

    def initialize(self, config: KernelConfig) -> None:
        """Initialize the Metal kernel."""
        if not self._check_availability():
            raise RuntimeError(
                "Metal C++ kernel not available. "
                "Build with: ./scripts/build_metal.sh"
            )

        self._config = config
        self._initialized = True
        logger.info("Metal C++ kernel initialized")

    def migrate_tile(
        self,
        traces: TraceBlock,
        output: OutputTile,
        velocity: VelocitySlice,
        config: KernelConfig | None = None,
    ) -> KernelMetrics:
        """Execute migration on GPU."""
        if not self._initialized:
            raise RuntimeError("Kernel not initialized")

        cfg = config or self._config
        if cfg is None:
            cfg = KernelConfig()

        start_time = time.perf_counter()

        try:
            from pstm.metal.python import migrate_tile as metal_migrate
        except ImportError as e:
            raise RuntimeError(f"Metal module not available: {e}")

        # Ensure contiguous arrays
        traces = traces.ensure_contiguous()

        # Get velocity array (1D only for now)
        if not velocity.is_1d:
            raise ValueError("Metal kernel currently only supports 1D velocity models")

        vrms = np.ascontiguousarray(velocity.vrms, dtype=np.float64)

        # Build config dict for Metal kernel
        metal_config = {
            "max_dip_deg": float(cfg.max_dip_degrees or 45.0),
            "min_aperture": float(cfg.min_aperture_m or 100.0),
            "max_aperture": float(cfg.max_aperture_m or 2500.0),
            "taper_fraction": float(cfg.taper_fraction or 0.1),
            "dt_ms": float(traces.sample_rate_ms),
            "t_start_ms": float(traces.start_time_ms),
            "apply_spreading": bool(cfg.apply_spreading),
            "apply_obliquity": bool(cfg.apply_obliquity),
        }

        # Call Metal kernel
        metrics = metal_migrate(
            amplitudes=traces.amplitudes,
            source_x=traces.source_x,
            source_y=traces.source_y,
            receiver_x=traces.receiver_x,
            receiver_y=traces.receiver_y,
            midpoint_x=traces.midpoint_x,
            midpoint_y=traces.midpoint_y,
            image=output.image,
            fold=output.fold,
            x_coords=output.x_axis,
            y_coords=output.y_axis,
            t_coords_ms=output.t_axis_ms,
            vrms=vrms,
            config=metal_config,
        )

        elapsed = time.perf_counter() - start_time

        return KernelMetrics(
            n_traces_processed=traces.n_traces,
            n_samples_output=output.image.size,
            compute_time_s=elapsed,
            memory_peak_mb=0.0,  # TODO: track GPU memory
        )

    def synchronize(self) -> None:
        """Wait for GPU operations to complete."""
        # Metal operations are synchronous in current implementation
        pass

    def cleanup(self) -> None:
        """Release GPU resources."""
        try:
            from pstm.metal.python import cleanup
            cleanup()
        except ImportError:
            pass
        self._initialized = False
        logger.debug("Metal C++ kernel cleaned up")


def is_metal_cpp_available() -> bool:
    """Check if Metal C++ kernel is available."""
    try:
        from pstm.metal.python import is_available
        return is_available()
    except ImportError:
        return False


def get_metal_device_info() -> dict:
    """Get Metal device information."""
    try:
        from pstm.metal.python import get_device_info
        return get_device_info()
    except ImportError:
        return {
            "available": False,
            "device_name": "Module not built",
            "device_memory_gb": 0,
        }
