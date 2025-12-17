"""Configuration models and validation for PSTM."""

from pstm.config.models import (
    AlgorithmConfig,
    AmplitudeConfig,
    AntiAliasingConfig,
    AntiAliasingMethod,
    ApertureConfig,
    CheckpointConfig,
    ColumnMapping,
    ComputeBackend,
    CoordinateUnit,
    ExecutionConfig,
    GeometryConfig,
    InputConfig,
    InterpolationMethod,
    MigrationConfig,
    MuteConfig,
    OutputConfig,
    OutputFormat,
    OutputProductsConfig,
    ResourceConfig,
    SpatialIndexType,
    TaperType,
    TilingConfig,
    VelocityConfig,
    VelocitySource,
    VelocityType,
    create_minimal_config,
)

# Enhanced output grid with corner-point support
from pstm.config.output_grid import (
    CornerPoints,
    GridDefinitionMethod,
    OutputGridConfig,
    Point2D,
)

# Flexible data selection
from pstm.config.data_selection import (
    AzimuthConvention,
    AzimuthSector,
    CustomExpressionSelector,
    DataSelectionConfig,
    OffsetAzimuthSelector,
    OffsetRange,
    OffsetRangeSelector,
    OffsetVectorSelector,
    RangeMode,
    SelectionMode,
    # Preset factories
    east_west_azimuth_selection,
    far_offset_selection,
    near_offset_selection,
    north_south_azimuth_selection,
    quadrant_selection,
)

__all__ = [
    # Main config
    "MigrationConfig",
    # Sub-configs
    "InputConfig",
    "GeometryConfig",
    "VelocityConfig",
    "AlgorithmConfig",
    "OutputConfig",
    "ExecutionConfig",
    # Data selection (NEW)
    "DataSelectionConfig",
    "SelectionMode",
    "OffsetRangeSelector",
    "OffsetRange",
    "RangeMode",
    "OffsetAzimuthSelector",
    "AzimuthSector",
    "AzimuthConvention",
    "OffsetVectorSelector",
    "CustomExpressionSelector",
    # Data selection presets
    "near_offset_selection",
    "far_offset_selection",
    "north_south_azimuth_selection",
    "east_west_azimuth_selection",
    "quadrant_selection",
    # Output grid (enhanced)
    "OutputGridConfig",
    "GridDefinitionMethod",
    "CornerPoints",
    "Point2D",
    # Other nested configs
    "ColumnMapping",
    "ApertureConfig",
    "AntiAliasingConfig",
    "AmplitudeConfig",
    "MuteConfig",
    "OutputProductsConfig",
    "ResourceConfig",
    "TilingConfig",
    "CheckpointConfig",
    # Enums
    "CoordinateUnit",
    "VelocityType",
    "VelocitySource",
    "InterpolationMethod",
    "AntiAliasingMethod",
    "TaperType",
    "ComputeBackend",
    "OutputFormat",
    "SpatialIndexType",
    # Factory functions
    "create_minimal_config",
]
