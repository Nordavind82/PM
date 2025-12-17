"""Migration kernels for PSTM."""

from pstm.kernels.base import (
    KernelCapability,
    KernelConfig,
    KernelMetrics,
    MigrationKernel,
    OutputTile,
    TraceBlock,
    VelocitySlice,
    create_output_tile,
    create_trace_block,
)
from pstm.kernels.factory import (
    create_kernel,
    get_available_backends,
    benchmark_all_backends,
    auto_select_with_benchmark,
)
from pstm.kernels.interpolation import (
    InterpolationMethod,
    get_available_methods as get_interpolation_methods,
    get_method_code,
    get_method_info,
    recommend_method as recommend_interpolation,
)

__all__ = [
    # Base classes
    "MigrationKernel",
    "TraceBlock",
    "OutputTile",
    "VelocitySlice",
    "KernelConfig",
    "KernelMetrics",
    "KernelCapability",
    # Factory functions
    "create_trace_block",
    "create_output_tile",
    "create_kernel",
    "get_available_backends",
    "benchmark_all_backends",
    "auto_select_with_benchmark",
    # Interpolation
    "InterpolationMethod",
    "get_interpolation_methods",
    "get_method_code",
    "get_method_info",
    "recommend_interpolation",
]
