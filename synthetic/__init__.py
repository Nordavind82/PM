"""Synthetic data generation for PSTM testing."""

from pstm.synthetic.common_offset_gathers import (
    # Configuration classes
    SyntheticConfig,
    SurveyGeometry,
    DiffractorLocation,
    OffsetAzimuthPlane,
    TraceParameters,
    WaveletParameters,
    OffsetAzimuthMode,
    
    # Result class
    SyntheticGatherResult,
    
    # Generation functions
    generate_synthetic_gathers,
    generate_ricker_wavelet,
    generate_ormsby_wavelet,
    compute_dsr_travel_time,
    
    # Convenience functions
    create_simple_synthetic,
    create_multi_diffractor_synthetic,
    
    # Export functions
    export_to_zarr_parquet,
    export_to_segy,
)

__all__ = [
    # Config
    "SyntheticConfig",
    "SurveyGeometry",
    "DiffractorLocation",
    "OffsetAzimuthPlane",
    "TraceParameters",
    "WaveletParameters",
    "OffsetAzimuthMode",
    # Result
    "SyntheticGatherResult",
    # Generation
    "generate_synthetic_gathers",
    "generate_ricker_wavelet",
    "generate_ormsby_wavelet",
    "compute_dsr_travel_time",
    # Convenience
    "create_simple_synthetic",
    "create_multi_diffractor_synthetic",
    # Export
    "export_to_zarr_parquet",
    "export_to_segy",
]
