"""Data access layer for PSTM."""

from pstm.data.memmap_manager import (
    BufferInfo,
    MemmapManager,
    OutputTileBuffer,
    TraceBuffer,
)
from pstm.data.parquet_headers import (
    HeaderStatistics,
    ParquetHeaderManager,
    TraceGeometry,
    create_parquet_headers,
)
from pstm.data.spatial_index import (
    SpatialIndex,
    SpatialIndexInfo,
    TileQueryResult,
    query_traces_for_tile,
)
from pstm.data.velocity_model import (
    ConstantVelocityModel,
    CubeVelocityModel,
    LinearVelocityModel,
    TableVelocityModel,
    VelocityManager,
    VelocityModel,
    create_velocity_manager,
    create_velocity_model,
    validate_velocity_range,
)
from pstm.data.zarr_reader import (
    TraceDataInfo,
    ZarrTraceReader,
    create_zarr_traces,
)

__all__ = [
    # Zarr
    "ZarrTraceReader",
    "TraceDataInfo",
    "create_zarr_traces",
    # Parquet
    "ParquetHeaderManager",
    "HeaderStatistics",
    "TraceGeometry",
    "create_parquet_headers",
    # Spatial Index
    "SpatialIndex",
    "SpatialIndexInfo",
    "TileQueryResult",
    "query_traces_for_tile",
    # Velocity
    "VelocityModel",
    "ConstantVelocityModel",
    "LinearVelocityModel",
    "TableVelocityModel",
    "CubeVelocityModel",
    "VelocityManager",
    "create_velocity_model",
    "create_velocity_manager",
    "validate_velocity_range",
    # Memmap
    "MemmapManager",
    "BufferInfo",
    "TraceBuffer",
    "OutputTileBuffer",
]
