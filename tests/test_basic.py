"""
Basic tests for PSTM.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestConfig:
    """Test configuration models."""

    def test_minimal_config(self):
        """Test creating minimal configuration."""
        from pstm.config import create_minimal_config

        config = create_minimal_config(
            traces_path="/tmp/traces.zarr",
            headers_path="/tmp/headers.parquet",
            output_dir="/tmp/output",
            velocity=2000.0,
            x_range=(0, 1000),
            y_range=(0, 1000),
            t_range_ms=(0, 2000),
        )

        assert config.name == "unnamed_migration"
        assert config.output.grid.nx > 0
        assert config.output.grid.ny > 0
        assert config.output.grid.nt > 0

    def test_config_serialization(self):
        """Test config save/load."""
        from pstm.config import MigrationConfig, create_minimal_config

        config = create_minimal_config(
            traces_path="/tmp/traces.zarr",
            headers_path="/tmp/headers.parquet",
            output_dir="/tmp/output",
            velocity=2000.0,
            x_range=(0, 1000),
            y_range=(0, 1000),
            t_range_ms=(0, 2000),
        )
        config.name = "test_migration"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config.to_json(f.name)
            loaded = MigrationConfig.from_json(f.name)

        assert loaded.name == "test_migration"
        assert loaded.output.grid.shape == config.output.grid.shape


class TestKernels:
    """Test kernel functionality."""

    def test_trace_block_creation(self):
        """Test creating trace blocks."""
        from pstm.kernels.base import create_trace_block

        n_traces, n_samples = 10, 100

        block = create_trace_block(
            amplitudes=np.random.randn(n_traces, n_samples).astype(np.float32),
            source_x=np.zeros(n_traces),
            source_y=np.zeros(n_traces),
            receiver_x=np.linspace(0, 500, n_traces),
            receiver_y=np.zeros(n_traces),
            sample_rate_ms=2.0,
        )

        assert block.n_traces == n_traces
        assert block.n_samples == n_samples
        assert len(block.offset) == n_traces
        assert block.offset[0] == 0  # First receiver at same position as source

    def test_output_tile_creation(self):
        """Test creating output tiles."""
        from pstm.kernels.base import create_output_tile

        tile = create_output_tile(
            x_min=0, x_max=100, dx=25,
            y_min=0, y_max=100, dy=25,
            t_min_ms=0, t_max_ms=1000, dt_ms=2,
        )

        assert tile.nx == 5
        assert tile.ny == 5
        assert tile.nt == 501
        assert tile.image.shape == (5, 5, 501)

    def test_numba_kernel_initialization(self):
        """Test Numba kernel can be initialized."""
        from pstm.kernels.numba_cpu import NumbaKernel
        from pstm.kernels.base import KernelConfig

        kernel = NumbaKernel()
        config = KernelConfig()

        kernel.initialize(config)
        assert kernel._initialized

        kernel.cleanup()
        assert not kernel._initialized


class TestData:
    """Test data handling."""

    def test_spatial_index(self):
        """Test spatial index queries."""
        from pstm.data.spatial_index import SpatialIndex

        n_points = 1000
        trace_indices = np.arange(n_points, dtype=np.int64)
        x = np.random.uniform(0, 1000, n_points)
        y = np.random.uniform(0, 1000, n_points)

        index = SpatialIndex.build(trace_indices, x, y)

        # Query center
        result = index.query_radius(500, 500, 100)
        assert len(result) > 0

        # Query rectangle
        result = index.query_rectangle(400, 600, 400, 600)
        assert len(result) > 0


class TestSynthetic:
    """Test synthetic data generation."""

    def test_ricker_wavelet(self):
        """Test Ricker wavelet generation."""
        from tests.fixtures.synthetic import generate_ricker_wavelet

        wavelet = generate_ricker_wavelet(30.0, 2.0, 200.0)

        assert len(wavelet) > 0
        assert wavelet.max() > 0
        assert np.isfinite(wavelet).all()

    def test_synthetic_geometry_2d(self):
        """Test 2D geometry generation."""
        from tests.fixtures.synthetic import generate_synthetic_geometry_2d

        (trace_idx, sx, sy, rx, ry, shot_ids) = generate_synthetic_geometry_2d(
            n_shots=5,
            n_receivers_per_shot=10,
            shot_spacing=100.0,
            receiver_spacing=25.0,
        )

        assert len(trace_idx) == 50  # 5 * 10
        assert len(np.unique(shot_ids)) == 5


class TestTilePlanner:
    """Test tile planning."""

    def test_tile_plan_creation(self):
        """Test creating a tile plan."""
        from pstm.config.models import OutputGridConfig, TilingConfig
        from pstm.pipeline.tile_planner import TilePlanner

        grid = OutputGridConfig(
            x_min=0, x_max=1000, dx=25,
            y_min=0, y_max=1000, dy=25,
            t_min_ms=0, t_max_ms=2000, dt_ms=2,
        )

        tiling = TilingConfig(auto_tile_size=False, tile_nx=10, tile_ny=10)

        planner = TilePlanner(grid, tiling)
        plan = planner.plan()

        assert plan.n_tiles > 0
        assert len(plan.tiles) == plan.n_tiles

        # Check tiles cover entire grid
        covered_x = set()
        covered_y = set()
        for tile in plan.tiles:
            for i in range(tile.x_start, tile.x_end):
                covered_x.add(i)
            for i in range(tile.y_start, tile.y_end):
                covered_y.add(i)

        assert len(covered_x) == grid.nx
        assert len(covered_y) == grid.ny


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
