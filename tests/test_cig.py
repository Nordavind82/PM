"""Tests for Common Image Gathers module."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from pstm.pipeline.cig import (
    CIGConfig,
    CIGAccumulator,
    create_cig_accumulator,
    save_cig_to_zarr,
    load_cig_from_zarr,
    analyze_cig_flatness,
)


class TestCIGConfig:
    def test_default_config(self):
        config = CIGConfig()
        assert not config.enabled
        # Default from settings is 20
        from pstm.settings import get_settings
        s = get_settings()
        assert config.n_offset_bins == s.cig.n_offset_bins

    def test_get_offset_bins(self):
        config = CIGConfig(n_offset_bins=5, min_offset=0, max_offset=1000)
        bins = config.get_offset_bins()
        assert len(bins) == 5
        np.testing.assert_array_almost_equal(bins, [0, 250, 500, 750, 1000])

    def test_get_offset_edges(self):
        config = CIGConfig(n_offset_bins=5, min_offset=0, max_offset=1000)
        edges = config.get_offset_edges()
        assert len(edges) == 6  # n_bins + 1


class TestCIGAccumulator:
    def test_create_accumulator(self):
        acc = CIGAccumulator(
            nx=10, ny=10, nt=100, n_offset_bins=5,
            offset_edges=np.array([0, 200, 400, 600, 800, 1000]),
            offset_centers=np.array([100, 300, 500, 700, 900]),
        )
        assert acc.image.shape == (10, 10, 100, 5)
        assert acc.fold.shape == (10, 10, 5)

    def test_get_offset_bin(self):
        acc = CIGAccumulator(
            nx=10, ny=10, nt=100, n_offset_bins=5,
            offset_edges=np.array([0, 200, 400, 600, 800, 1000]),
            offset_centers=np.array([100, 300, 500, 700, 900]),
        )
        assert acc.get_offset_bin(150) == 0
        assert acc.get_offset_bin(350) == 1
        assert acc.get_offset_bin(950) == 4
        assert acc.get_offset_bin(-100) == -1  # Out of range
        assert acc.get_offset_bin(1500) == -1  # Out of range

    def test_accumulate(self):
        acc = CIGAccumulator(
            nx=10, ny=10, nt=100, n_offset_bins=5,
            offset_edges=np.array([0, 200, 400, 600, 800, 1000]),
            offset_centers=np.array([100, 300, 500, 700, 900]),
        )
        acc.accumulate(5, 5, 50, offset=150, amplitude=1.0)
        assert acc.image[5, 5, 50, 0] == 1.0

    def test_get_stacked_image(self):
        acc = CIGAccumulator(
            nx=5, ny=5, nt=10, n_offset_bins=3,
            offset_edges=np.array([0, 500, 1000, 1500]),
            offset_centers=np.array([250, 750, 1250]),
        )
        acc.image[:] = 1.0
        stacked = acc.get_stacked_image()
        assert stacked.shape == (5, 5, 10)
        assert np.all(stacked == 3.0)

    def test_get_gather_at_location(self):
        acc = CIGAccumulator(
            nx=5, ny=5, nt=10, n_offset_bins=3,
            offset_edges=np.array([0, 500, 1000, 1500]),
            offset_centers=np.array([250, 750, 1250]),
        )
        acc.image[2, 3, :, :] = np.random.randn(10, 3)
        gather = acc.get_gather_at_location(2, 3)
        assert gather.shape == (10, 3)


class TestCIGFactory:
    def test_create_cig_accumulator(self):
        config = CIGConfig(n_offset_bins=5, min_offset=0, max_offset=1000)
        acc = create_cig_accumulator(10, 10, 100, config)
        assert acc.n_offset_bins == 5


class TestCIGIO:
    def test_save_and_load(self):
        config = CIGConfig(n_offset_bins=3, min_offset=0, max_offset=1500)
        acc = create_cig_accumulator(5, 5, 50, config)
        acc.image[:] = np.random.randn(*acc.image.shape)
        acc.fold[:] = np.random.randint(1, 10, acc.fold.shape)

        x_axis = np.arange(5) * 25
        y_axis = np.arange(5) * 25
        t_axis_ms = np.arange(50) * 2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cig.zarr"
            save_cig_to_zarr(acc, path, x_axis, y_axis, t_axis_ms, normalize=False)

            cig, fold, coords = load_cig_from_zarr(path)
            assert cig.shape == acc.image.shape
            assert fold.shape == acc.fold.shape


class TestBinHeaders:
    """Test bin header output to Parquet."""

    def test_bin_headers_parquet_output(self):
        """Test that bin headers are correctly saved to Parquet."""
        import polars as pl

        # Simulate grid dimensions
        nx, ny = 10, 8

        # Create sample header data (as would be computed in executor._finalize)
        trace_count = np.random.randint(0, 100, (nx, ny)).astype(np.int32)
        offset_avg = np.random.uniform(100, 2000, (nx, ny)).astype(np.float32)
        azimuth_avg = np.random.uniform(0, 360, (nx, ny)).astype(np.float32)

        # Set some bins to zero (no data)
        trace_count[0, :] = 0
        trace_count[:, 0] = 0

        # Create grid coordinates
        x_min, x_max = 1000.0, 1900.0
        y_min, y_max = 2000.0, 2700.0
        x_coords = np.linspace(x_min, x_max, nx)
        y_coords = np.linspace(y_min, y_max, ny)

        # Create meshgrid (same as executor.py)
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")

        # Build DataFrame
        df = pl.DataFrame({
            "ix": xx.ravel().astype(np.int32),
            "iy": yy.ravel().astype(np.int32),
            "x": x_grid.ravel().astype(np.float64),
            "y": y_grid.ravel().astype(np.float64),
            "trace_count": trace_count.ravel().astype(np.int32),
            "offset_avg": offset_avg.ravel().astype(np.float32),
            "azimuth_avg": azimuth_avg.ravel().astype(np.float32),
        })

        # Filter to bins with data
        df_with_data = df.filter(pl.col("trace_count") > 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bin_headers.parquet"
            df_with_data.write_parquet(str(path))

            # Read back and verify
            df_read = pl.read_parquet(str(path))

            # Check schema
            assert "ix" in df_read.columns
            assert "iy" in df_read.columns
            assert "x" in df_read.columns
            assert "y" in df_read.columns
            assert "trace_count" in df_read.columns
            assert "offset_avg" in df_read.columns
            assert "azimuth_avg" in df_read.columns

            # All read entries should have trace_count > 0
            assert (df_read["trace_count"] > 0).all()

            # Check row count matches non-zero bins
            expected_count = np.sum(trace_count > 0)
            assert len(df_read) == expected_count

            # Verify coordinate ranges
            assert df_read["x"].min() >= x_min
            assert df_read["x"].max() <= x_max
            assert df_read["y"].min() >= y_min
            assert df_read["y"].max() <= y_max


class TestCIGAnalysis:
    def test_analyze_flatness(self):
        # Create synthetic CIG with flat events
        nt, n_offsets = 100, 5
        cig = np.zeros((nt, n_offsets))

        # Add flat event at t=50
        cig[48:52, :] = 1.0

        t_axis_ms = np.arange(nt) * 4
        offset_centers = np.array([100, 300, 500, 700, 900])

        result = analyze_cig_flatness(cig, t_axis_ms, offset_centers)
        assert 'flatness_score' in result
        assert result['flatness_score'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
