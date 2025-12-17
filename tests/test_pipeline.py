"""Comprehensive tests for pipeline module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pstm.config.models import OutputGridConfig, TilingConfig
from pstm.pipeline.tile_planner import TilePlanner, TilePlan, TileSpec, iter_tiles
from pstm.pipeline.checkpoint import CheckpointHandler, CheckpointState, should_checkpoint


class TestTilePlanner:
    def test_create_plan(self):
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

    def test_tiles_cover_grid(self):
        grid = OutputGridConfig(
            x_min=0, x_max=100, dx=10,
            y_min=0, y_max=100, dy=10,
            t_min_ms=0, t_max_ms=100, dt_ms=2,
        )
        tiling = TilingConfig(auto_tile_size=False, tile_nx=5, tile_ny=5)
        planner = TilePlanner(grid, tiling)
        plan = planner.plan()
        
        covered_x = set()
        covered_y = set()
        for tile in plan.tiles:
            for i in range(tile.x_start, tile.x_end):
                covered_x.add(i)
            for i in range(tile.y_start, tile.y_end):
                covered_y.add(i)
        
        assert len(covered_x) == grid.nx
        assert len(covered_y) == grid.ny

    def test_auto_tile_size(self):
        grid = OutputGridConfig(
            x_min=0, x_max=10000, dx=25,
            y_min=0, y_max=10000, dy=25,
            t_min_ms=0, t_max_ms=4000, dt_ms=2,
        )
        tiling = TilingConfig(auto_tile_size=True)
        planner = TilePlanner(grid, tiling, max_memory_gb=4.0)
        plan = planner.plan()
        
        assert plan.tile_nx > 0
        assert plan.tile_ny > 0

    def test_snake_ordering(self):
        grid = OutputGridConfig(
            x_min=0, x_max=100, dx=25,
            y_min=0, y_max=100, dy=25,
            t_min_ms=0, t_max_ms=100, dt_ms=2,
        )
        tiling = TilingConfig(auto_tile_size=False, tile_nx=2, tile_ny=2, ordering="snake")
        planner = TilePlanner(grid, tiling)
        plan = planner.plan()
        
        # Snake ordering should alternate directions
        assert plan.n_tiles > 0


class TestTileSpec:
    def test_tile_properties(self):
        tile = TileSpec(
            tile_id=0, ix=0, iy=0,
            x_start=0, x_end=10, y_start=0, y_end=10,
            x_min=0.0, x_max=250.0, y_min=0.0, y_max=250.0,
        )
        
        assert tile.nx == 10
        assert tile.ny == 10
        assert tile.center_x == 125.0
        assert tile.center_y == 125.0


class TestIterTiles:
    def test_iter_all_tiles(self):
        grid = OutputGridConfig(
            x_min=0, x_max=100, dx=25,
            y_min=0, y_max=100, dy=25,
            t_min_ms=0, t_max_ms=100, dt_ms=2,
        )
        tiling = TilingConfig(auto_tile_size=False, tile_nx=3, tile_ny=3)
        planner = TilePlanner(grid, tiling)
        plan = planner.plan()
        
        tiles = list(iter_tiles(plan))
        assert len(tiles) == plan.n_tiles

    def test_iter_skip_completed(self):
        grid = OutputGridConfig(
            x_min=0, x_max=100, dx=25,
            y_min=0, y_max=100, dy=25,
            t_min_ms=0, t_max_ms=100, dt_ms=2,
        )
        tiling = TilingConfig(auto_tile_size=False, tile_nx=2, tile_ny=2)
        planner = TilePlanner(grid, tiling)
        plan = planner.plan()
        
        completed = {0, 1}  # Skip first two tiles
        tiles = list(iter_tiles(plan, completed))
        
        assert len(tiles) == plan.n_tiles - 2
        assert all(t.tile_id not in completed for t in tiles)


class TestCheckpointHandler:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def mock_config(self):
        from pstm.config import create_minimal_config
        return create_minimal_config(
            traces_path="/data/traces.zarr",
            headers_path="/data/headers.parquet",
            output_dir="/output",
            velocity=2000.0,
            x_range=(0, 1000),
            y_range=(0, 1000),
            t_range_ms=(0, 2000),
        )

    def test_create_handler(self, temp_dir, mock_config):
        handler = CheckpointHandler(temp_dir / "checkpoint", mock_config, total_tiles=100)
        
        assert handler.total_tiles == 100
        assert not handler.exists()

    def test_save_and_load(self, temp_dir, mock_config):
        handler = CheckpointHandler(temp_dir / "checkpoint", mock_config, total_tiles=100)
        
        handler.mark_completed(0, n_traces=1000, compute_time=1.5)
        handler.mark_completed(1, n_traces=1000, compute_time=1.5)
        handler.save()
        
        # Create new handler and load
        handler2 = CheckpointHandler(temp_dir / "checkpoint", mock_config, total_tiles=100)
        state = handler2.load()
        
        assert state is not None
        assert state.n_completed == 2

    def test_get_remaining_tiles(self, temp_dir, mock_config):
        handler = CheckpointHandler(temp_dir / "checkpoint", mock_config, total_tiles=10)
        
        handler.mark_completed(0)
        handler.mark_completed(2)
        handler.mark_completed(5)
        
        remaining = handler.get_remaining_tiles()
        
        assert 0 not in remaining
        assert 2 not in remaining
        assert 5 not in remaining
        assert 1 in remaining
        assert len(remaining) == 7


class TestCheckpointState:
    def test_state_properties(self):
        state = CheckpointState(
            config_hash="abc123",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            total_tiles=100,
            completed_tiles=[0, 1, 2, 3, 4],
        )
        
        assert state.n_completed == 5
        assert state.progress_fraction == 0.05

    def test_mark_completed(self):
        state = CheckpointState(
            config_hash="abc123",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            total_tiles=100,
        )
        
        state.mark_tile_completed(0)
        state.mark_tile_completed(1)
        
        assert state.is_tile_completed(0)
        assert state.is_tile_completed(1)
        assert not state.is_tile_completed(2)


class TestShouldCheckpoint:
    def test_interval_checkpoint(self):
        assert should_checkpoint(99, interval=100)  # 100th tile (0-indexed)
        assert not should_checkpoint(98, interval=100)

    def test_force_checkpoint(self):
        assert should_checkpoint(5, interval=100, force=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
