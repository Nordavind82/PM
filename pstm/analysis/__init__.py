"""
PSTM Analysis Module.

This module provides analysis utilities for seismic data including:
- Bin size auto-calculation from midpoint distribution
- Grid outlier detection and classification
- Pre-migration QC utilities
"""

from pstm.analysis.bin_size import (
    BinSizeMethod,
    BinSizeResult,
    STANDARD_BIN_SIZES,
    auto_calculate_bin_size,
    auto_calculate_bin_size_ensemble,
    calculate_bin_size_histogram,
    calculate_bin_size_nearest_neighbor,
    calculate_bin_size_fft,
    round_to_standard_bin_size,
)

from pstm.analysis.grid_outliers import (
    OutlierHandling,
    GridClassificationResult,
    OutlierReport,
    classify_points_against_grid,
    compute_extended_corners,
    compute_aperture_extended_corners,
    compute_boundary_taper_weights,
    generate_outlier_report,
)

__all__ = [
    # Bin size
    "BinSizeMethod",
    "BinSizeResult",
    "STANDARD_BIN_SIZES",
    "auto_calculate_bin_size",
    "auto_calculate_bin_size_ensemble",
    "calculate_bin_size_histogram",
    "calculate_bin_size_nearest_neighbor",
    "calculate_bin_size_fft",
    "round_to_standard_bin_size",
    # Grid outliers
    "OutlierHandling",
    "GridClassificationResult",
    "OutlierReport",
    "classify_points_against_grid",
    "compute_extended_corners",
    "compute_aperture_extended_corners",
    "compute_boundary_taper_weights",
    "generate_outlier_report",
]
