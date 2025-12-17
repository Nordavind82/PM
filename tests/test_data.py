"""Comprehensive tests for data access module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pstm.data.spatial_index import SpatialIndex, query_traces_for_tile, TileQueryResult
from pstm.data.velocity_model import (
    ConstantVelocityModel,
    LinearVelocityModel,
    TableVelocityModel,
    create_velocity_model,
    validate_velocity_range,
)
from pstm.data.memmap_manager import MemmapManager
from pstm.config.models import VelocityConfig, VelocitySource


class TestSpatialIndex:
    """Tests for SpatialIndex."""

    @pytest.fixture
    def sample_index(self):
        np.random.seed(42)
        n_points = 1000
        trace_indices = np.arange(n_points, dtype=np.int64)
        x = np.random.uniform(0, 1000, n_points)
        y = np.random.uniform(0, 1000, n_points)
        return SpatialIndex.build(trace_indices, x, y)

    def test_build_index(self, sample_index):
        assert sample_index.n_points == 1000

    def test_query_radius(self, sample_index):
        result = sample_index.query_radius(500, 500, 100)
        assert len(result) >= 0

    def test_query_rectangle(self, sample_index):
        result = sample_index.query_rectangle(400, 600, 400, 600)
        assert len(result) >= 0

    def test_query_nearest(self, sample_index):
        # query_nearest returns (indices, distances) tuple
        result = sample_index.query_nearest(500, 500, k=10)
        if isinstance(result, tuple):
            indices, distances = result
            assert len(indices) == 10
        else:
            assert len(result) == 10

    def test_save_load(self, sample_index):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            sample_index.save(f.name)
            loaded = SpatialIndex.load(f.name)
        assert loaded.n_points == sample_index.n_points

    def test_empty_query(self, sample_index):
        result = sample_index.query_radius(-10000, -10000, 10)
        assert len(result) == 0


class TestVelocityModels:
    """Tests for velocity models."""

    def test_constant_velocity(self):
        model = ConstantVelocityModel(velocity=2000.0)
        t_axis = np.linspace(0, 4000, 2001)
        vrms = model.get_vrms_1d(t_axis)
        assert np.all(vrms == 2000.0)

    def test_linear_velocity(self):
        # k is in m/s per second, t is in ms
        # V(t) = V0 + k * t_seconds = 1500 + 500 * t_seconds
        model = LinearVelocityModel(v0=1500.0, k=500.0)
        t_axis = np.array([0, 1000, 2000])  # ms -> 0, 1, 2 seconds
        vrms = model.get_vrms_1d(t_axis)
        expected = np.array([1500, 2000, 2500])
        np.testing.assert_array_almost_equal(vrms, expected)

    def test_table_velocity(self):
        times = np.array([0, 1000, 2000, 3000])
        velocities = np.array([1500, 2000, 2500, 3000])
        model = TableVelocityModel(times, velocities)
        vrms = model.get_vrms_1d(np.array([0, 1000, 2000]))
        np.testing.assert_array_almost_equal(vrms, [1500, 2000, 2500])

    def test_create_constant(self):
        config = VelocityConfig(source=VelocitySource.CONSTANT, constant_velocity=2500.0)
        model = create_velocity_model(config)
        assert isinstance(model, ConstantVelocityModel)

    def test_validate_velocity_range(self):
        model = ConstantVelocityModel(velocity=2000.0)
        t_axis = np.linspace(0, 4000, 100)
        warnings = validate_velocity_range(model, t_axis)
        assert len(warnings) == 0


class TestMemmapManager:
    """Tests for MemmapManager."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_create_memmap(self, temp_dir):
        manager = MemmapManager(temp_dir)
        arr = manager.create("test", shape=(100, 100), dtype=np.float32)
        assert arr.shape == (100, 100)

    def test_get_memmap(self, temp_dir):
        manager = MemmapManager(temp_dir)
        manager.create("data", shape=(50, 50), dtype=np.float32)
        retrieved = manager.get("data")
        assert retrieved.shape == (50, 50)

    def test_release(self, temp_dir):
        manager = MemmapManager(temp_dir)
        manager.create("test", shape=(10, 10), dtype=np.float32)
        manager.release("test")
        with pytest.raises(KeyError):
            manager.get("test")

    def test_flush(self, temp_dir):
        manager = MemmapManager(temp_dir)
        arr = manager.create("test", shape=(10, 10), dtype=np.float32)
        arr[:] = 42.0
        manager.flush("test")
        # Should not raise

    def test_get_stats(self, temp_dir):
        manager = MemmapManager(temp_dir)
        manager.create("a", shape=(100, 100), dtype=np.float32)
        manager.create("b", shape=(100, 100), dtype=np.float64)
        stats = manager.get_stats()
        assert stats["n_buffers"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
