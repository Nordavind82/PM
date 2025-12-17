"""
Comprehensive tests for configuration module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pstm.config.models import (
    MigrationConfig,
    InputConfig,
    GeometryConfig,
    VelocityConfig,
    AlgorithmConfig,
    OutputConfig,
    OutputGridConfig,
    ExecutionConfig,
    ResourceConfig,
    TilingConfig,
    CheckpointConfig,
    ApertureConfig,
    AntiAliasingConfig,
    AmplitudeConfig,
    ColumnMapping,
    VelocitySource,
    VelocityType,
    InterpolationMethod,
    AntiAliasingMethod,
    ComputeBackend,
    CoordinateUnit,
    TaperType,
    SpatialIndexType,
    OutputFormat,
    create_minimal_config,
)


class TestColumnMapping:
    """Tests for ColumnMapping model."""

    def test_default_values(self):
        mapping = ColumnMapping()
        assert mapping.source_x == "SOU_X"
        assert mapping.source_y == "SOU_Y"
        assert mapping.receiver_x == "REC_X"
        assert mapping.receiver_y == "REC_Y"

    def test_custom_values(self):
        mapping = ColumnMapping(
            source_x="src_x",
            source_y="src_y",
            receiver_x="rec_x",
            receiver_y="rec_y",
        )
        assert mapping.source_x == "src_x"


class TestInputConfig:
    """Tests for InputConfig model."""

    def test_basic_creation(self):
        config = InputConfig(
            traces_path=Path("/data/traces.zarr"),
            headers_path=Path("/data/headers.parquet"),
        )
        assert config.traces_path == Path("/data/traces.zarr")
        assert config.coordinate_unit == CoordinateUnit.METERS

    def test_string_paths_converted(self):
        config = InputConfig(
            traces_path="/data/traces.zarr",
            headers_path="/data/headers.parquet",
        )
        assert isinstance(config.traces_path, Path)


class TestVelocityConfig:
    """Tests for VelocityConfig model."""

    def test_constant_velocity(self):
        config = VelocityConfig(
            source=VelocitySource.CONSTANT,
            constant_velocity=2500.0,
        )
        assert config.source == VelocitySource.CONSTANT
        assert config.constant_velocity == 2500.0

    def test_linear_velocity(self):
        config = VelocityConfig(
            source=VelocitySource.LINEAR_V0K,
            v0=1500.0,
            k=0.5,
        )
        assert config.v0 == 1500.0
        assert config.k == 0.5


class TestOutputGridConfig:
    """Tests for OutputGridConfig model."""

    def test_basic_grid(self):
        grid = OutputGridConfig(
            x_min=0, x_max=1000, dx=25,
            y_min=0, y_max=1000, dy=25,
            t_min_ms=0, t_max_ms=2000, dt_ms=2,
        )
        assert grid.nx == 41
        assert grid.ny == 41
        assert grid.nt == 1001

    def test_grid_shape(self):
        grid = OutputGridConfig(
            x_min=0, x_max=100, dx=10,
            y_min=0, y_max=100, dy=10,
            t_min_ms=0, t_max_ms=100, dt_ms=1,
        )
        assert grid.shape == (11, 11, 101)


class TestMigrationConfig:
    """Tests for complete MigrationConfig model."""

    def test_minimal_config(self):
        config = create_minimal_config(
            traces_path="/data/traces.zarr",
            headers_path="/data/headers.parquet",
            output_dir="/output",
            velocity=2000.0,
            x_range=(0, 1000),
            y_range=(0, 1000),
            t_range_ms=(0, 2000),
        )
        assert config.name == "unnamed_migration"
        assert config.velocity.constant_velocity == 2000.0

    def test_json_serialization(self):
        config = create_minimal_config(
            traces_path="/data/traces.zarr",
            headers_path="/data/headers.parquet",
            output_dir="/output",
            velocity=2500.0,
            x_range=(0, 5000),
            y_range=(0, 5000),
            t_range_ms=(0, 3000),
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config.to_json(f.name)
            loaded = MigrationConfig.from_json(f.name)

        assert loaded.velocity.constant_velocity == 2500.0


class TestEnumerations:
    """Tests for enumeration types."""

    def test_velocity_source(self):
        assert VelocitySource.CONSTANT.value == "constant"
        assert VelocitySource.LINEAR_V0K.value == "linear_v0k"

    def test_compute_backend(self):
        assert ComputeBackend.AUTO.value == "auto"
        assert ComputeBackend.NUMBA_CPU.value == "numba_cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
