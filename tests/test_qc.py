"""Comprehensive tests for QC module."""

import numpy as np
import pytest

from pstm.qc.analysis import (
    GeometryQCReport,
    VelocityQCReport,
    OutputQCReport,
    MigrationVerificationReport,
    compute_fold_map,
    analyze_offsets,
    analyze_azimuths,
    run_geometry_qc,
    run_velocity_qc,
    run_output_qc,
    extract_slice,
    verify_diffractor_focus,
    verify_flat_reflector_depth,
)


class TestFoldMap:
    def test_compute_fold_map(self):
        np.random.seed(42)
        n = 1000
        midpoint_x = np.random.uniform(0, 1000, n)
        midpoint_y = np.random.uniform(0, 1000, n)
        
        fold_map, x_bins, y_bins = compute_fold_map(
            midpoint_x, midpoint_y, bin_size=100
        )
        
        assert fold_map.shape[0] == len(x_bins) - 1
        assert fold_map.shape[1] == len(y_bins) - 1
        assert np.sum(fold_map) == n

    def test_fold_map_with_range(self):
        n = 500
        midpoint_x = np.random.uniform(0, 1000, n)
        midpoint_y = np.random.uniform(0, 1000, n)
        
        fold_map, x_bins, y_bins = compute_fold_map(
            midpoint_x, midpoint_y, bin_size=50,
            x_range=(0, 1000), y_range=(0, 1000),
        )
        
        assert x_bins[0] == 0
        assert y_bins[0] == 0


class TestAnalyzeOffsets:
    def test_basic_offset_analysis(self):
        offsets = np.random.uniform(100, 5000, 1000)
        
        result = analyze_offsets(offsets)
        
        assert "min" in result
        assert "max" in result
        assert "mean" in result
        assert "histogram" in result

    def test_uniform_offsets(self):
        offsets = np.linspace(0, 1000, 100)
        
        result = analyze_offsets(offsets)
        
        assert result["min"] == 0
        assert result["max"] == 1000
        assert 490 < result["mean"] < 510


class TestAnalyzeAzimuths:
    def test_basic_azimuth_analysis(self):
        n = 100
        source_x = np.zeros(n)
        source_y = np.zeros(n)
        receiver_x = np.random.uniform(-500, 500, n)
        receiver_y = np.random.uniform(-500, 500, n)
        
        result = analyze_azimuths(source_x, source_y, receiver_x, receiver_y)
        
        assert "histogram" in result
        assert "mean" in result

    def test_single_azimuth(self):
        n = 100
        source_x = np.zeros(n)
        source_y = np.zeros(n)
        receiver_x = np.ones(n) * 100  # East
        receiver_y = np.zeros(n)
        
        result = analyze_azimuths(source_x, source_y, receiver_x, receiver_y)
        
        # East should be around 90 degrees
        assert 80 < result["mean"] < 100


class TestRunGeometryQC:
    @pytest.fixture
    def sample_geometry(self):
        np.random.seed(42)
        n = 500
        source_x = np.random.uniform(0, 1000, n)
        source_y = np.random.uniform(0, 1000, n)
        receiver_x = source_x + np.random.uniform(100, 500, n)
        receiver_y = source_y + np.random.uniform(-50, 50, n)
        midpoint_x = (source_x + receiver_x) / 2
        midpoint_y = (source_y + receiver_y) / 2
        offset = np.sqrt((receiver_x - source_x)**2 + (receiver_y - source_y)**2)
        shot_ids = np.repeat(np.arange(50), 10).astype(np.int32)
        
        return (midpoint_x, midpoint_y, source_x, source_y, 
                receiver_x, receiver_y, offset, shot_ids)

    def test_run_geometry_qc(self, sample_geometry):
        (mx, my, sx, sy, rx, ry, off, shots) = sample_geometry
        
        report = run_geometry_qc(mx, my, sx, sy, rx, ry, off, shots)
        
        assert isinstance(report, GeometryQCReport)
        assert report.n_traces == 500
        assert report.n_shots == 50
        assert report.max_fold > 0


class TestRunVelocityQC:
    def test_valid_velocity(self):
        vrms = np.linspace(1500, 3500, 1000)
        t_axis_ms = np.linspace(0, 4000, 1000)
        
        report = run_velocity_qc(vrms, t_axis_ms)
        
        assert isinstance(report, VelocityQCReport)
        assert report.v_min == 1500
        assert report.v_max == 3500
        assert len(report.warnings) == 0

    def test_velocity_with_inversions(self):
        t_axis_ms = np.linspace(0, 4000, 1000)
        vrms = np.linspace(1500, 3500, 1000)
        vrms[500:600] = 1500  # Inversion
        
        report = run_velocity_qc(vrms, t_axis_ms)
        
        assert report.n_inversions > 0

    def test_out_of_range_velocity(self):
        vrms = np.array([500, 600, 700])  # Too low
        t_axis_ms = np.array([0, 1000, 2000])
        
        report = run_velocity_qc(vrms, t_axis_ms, min_valid=1000.0)
        
        assert len(report.warnings) > 0


class TestRunOutputQC:
    def test_valid_output(self):
        image = np.random.randn(50, 50, 500).astype(np.float32)
        fold = np.random.randint(1, 100, (50, 50)).astype(np.int32)
        
        report = run_output_qc(image, fold)
        
        assert isinstance(report, OutputQCReport)
        assert report.shape == (50, 50, 500)
        assert report.n_nan == 0
        assert report.fold_max > 0

    def test_output_with_nan(self):
        image = np.random.randn(10, 10, 100).astype(np.float32)
        image[0, 0, 0] = np.nan
        
        report = run_output_qc(image)
        
        assert report.n_nan == 1
        assert len(report.warnings) > 0

    def test_mostly_zero_output(self):
        image = np.zeros((10, 10, 100), dtype=np.float32)
        image[5, 5, 50] = 1.0  # Only one non-zero
        
        report = run_output_qc(image)
        
        assert report.zero_percent > 99


class TestExtractSlice:
    def test_time_slice(self):
        volume = np.random.randn(50, 50, 100)
        
        time_slice = extract_slice(volume, axis=2, index=50)
        
        assert time_slice.shape == (50, 50)

    def test_inline_slice(self):
        volume = np.random.randn(50, 50, 100)
        
        inline_slice = extract_slice(volume, axis=0, index=25)
        
        assert inline_slice.shape == (50, 100)

    def test_crossline_slice(self):
        volume = np.random.randn(50, 50, 100)
        
        crossline_slice = extract_slice(volume, axis=1, index=25)
        
        assert crossline_slice.shape == (50, 100)


class TestVerifyDiffractorFocus:
    def test_correct_focus(self):
        # Create image with peak at known location
        nx, ny, nt = 50, 50, 500
        image = np.zeros((nx, ny, nt))
        
        # Put peak at center
        image[25, 25, 250] = 10.0
        
        x_axis = np.linspace(0, 1000, nx)
        y_axis = np.linspace(0, 1000, ny)
        t_axis_ms = np.linspace(0, 2000, nt)
        
        result = verify_diffractor_focus(
            image, x_axis, y_axis, t_axis_ms,
            expected_x=500, expected_y=500, expected_t_ms=1000,
            tolerance_xy=50, tolerance_t_ms=50,
        )
        
        assert result.passed

    def test_wrong_focus(self):
        nx, ny, nt = 50, 50, 500
        image = np.zeros((nx, ny, nt))
        image[25, 25, 250] = 10.0  # Peak at center
        
        x_axis = np.linspace(0, 1000, nx)
        y_axis = np.linspace(0, 1000, ny)
        t_axis_ms = np.linspace(0, 2000, nt)
        
        result = verify_diffractor_focus(
            image, x_axis, y_axis, t_axis_ms,
            expected_x=0, expected_y=0, expected_t_ms=0,  # Wrong location
            tolerance_xy=50, tolerance_t_ms=50,
        )
        
        assert not result.passed


class TestVerifyFlatReflectorDepth:
    def test_correct_depth(self):
        nx, ny, nt = 50, 50, 500
        image = np.zeros((nx, ny, nt))
        
        # Flat reflector at t=1000ms (index 250)
        image[:, :, 250] = 1.0
        
        t_axis_ms = np.linspace(0, 2000, nt)
        
        result = verify_flat_reflector_depth(
            image, t_axis_ms,
            expected_t_ms=1000, tolerance_ms=20,
        )
        
        assert result.passed

    def test_wrong_depth(self):
        nx, ny, nt = 50, 50, 500
        image = np.zeros((nx, ny, nt))
        image[:, :, 250] = 1.0  # At 1000ms
        
        t_axis_ms = np.linspace(0, 2000, nt)
        
        result = verify_flat_reflector_depth(
            image, t_axis_ms,
            expected_t_ms=500, tolerance_ms=20,  # Wrong depth
        )
        
        assert not result.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
