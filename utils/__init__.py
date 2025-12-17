"""Utility functions for PSTM."""

from pstm.utils.logging import (
    setup_logging,
    get_logger,
    print_banner,
    print_section,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_metric,
)
from pstm.utils.units import (
    meters_to_feet,
    feet_to_meters,
    ms_to_s,
    s_to_ms,
    samples_to_time,
    time_to_samples,
    time_to_sample_indices,
    offset_to_midpoint,
    compute_offset,
    compute_azimuth,
    apply_seg_y_scalar,
    format_bytes,
    format_duration,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "print_banner",
    "print_section",
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
    "print_metric",
    # Units
    "meters_to_feet",
    "feet_to_meters",
    "ms_to_s",
    "s_to_ms",
    "samples_to_time",
    "time_to_samples",
    "time_to_sample_indices",
    "offset_to_midpoint",
    "compute_offset",
    "compute_azimuth",
    "apply_seg_y_scalar",
    "format_bytes",
    "format_duration",
]
