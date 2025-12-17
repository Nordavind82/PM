"""Tests for synthetic common offset gather generation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pstm.synthetic.common_offset_gathers import (
    SyntheticConfig,
    SurveyGeometry,
    DiffractorLocation,
    OffsetAzimuthPlane,
    TraceParameters,
    WaveletParameters,
    generate_synthetic_gathers,
    generate_ricker_wavelet,
    generate_ormsby_wavelet,
    compute_dsr_travel_time,
    create_simple_synthetic,
    create_multi_diffractor_synthetic,
    export_to_zarr_parquet,
)


class TestSurveyGeometry:
    def test_basic_geometry(self):
        survey = SurveyGeometry(
            x_min=0, x_max=1000,
            y_min=0, y_max=1000,
            dx=25, dy=25,
        )
        assert survey.nx == 41
        assert survey.ny == 41
        assert survey.n_positions == 1681

    def test_get_midpoint_grid(self):
        survey = SurveyGeometry(x_min=0, x_max=100, y_min=0, y_max=100, dx=50, dy=50)
        mx, my = survey.get_midpoint_grid()
        
        assert mx.shape == (3, 3)
        assert my.shape == (3, 3)
        assert mx[0, 0] == 0
        assert mx[2, 0] == 100


class TestOffsetAzimuthPlane:
    def test_from_offset_azimuth(self):
        plane = OffsetAzimuthPlane(offset=1000, azimuth_deg=90)  # East
        
        assert plane.offset == 1000
        assert plane.azimuth_deg == 90
        assert abs(plane.offset_x - 1000) < 0.01
        assert abs(plane.offset_y) < 0.01

    def test_from_offset_xy(self):
        plane = OffsetAzimuthPlane(offset_x=500, offset_y=500)
        
        assert abs(plane.offset - 707.1) < 1
        assert abs(plane.azimuth_deg - 45) < 0.1

    def test_source_receiver_offsets(self):
        plane = OffsetAzimuthPlane(offset_x=100, offset_y=0)
        src_dx, src_dy, rec_dx, rec_dy = plane.get_source_receiver_offsets()
        
        assert src_dx == -50
        assert src_dy == 0
        assert rec_dx == 50
        assert rec_dy == 0


class TestTraceParameters:
    def test_time_axis(self):
        params = TraceParameters(n_samples=501, dt_ms=4.0, t_start_ms=0)
        
        t = params.time_axis_ms
        assert len(t) == 501
        assert t[0] == 0
        assert t[-1] == 2000

    def test_end_time(self):
        params = TraceParameters(n_samples=1001, dt_ms=2.0, t_start_ms=100)
        assert params.t_end_ms == 2100


class TestWavelets:
    def test_ricker_wavelet(self):
        wavelet = generate_ricker_wavelet(freq_hz=30, dt_ms=2.0, length_ms=100)
        
        assert len(wavelet) > 0
        # Peak should be at center
        peak_idx = np.argmax(np.abs(wavelet))
        center = len(wavelet) // 2
        assert abs(peak_idx - center) < 3

    def test_ricker_normalized(self):
        wavelet = generate_ricker_wavelet(freq_hz=25, dt_ms=1.0)
        # Peak amplitude should be 1
        assert abs(np.max(np.abs(wavelet)) - 1.0) < 0.01

    def test_ormsby_wavelet(self):
        wavelet = generate_ormsby_wavelet(f1=5, f2=15, f3=45, f4=60, dt_ms=2.0)
        
        assert len(wavelet) > 0
        assert np.max(np.abs(wavelet)) <= 1.01


class TestDSRTravelTime:
    def test_zero_offset(self):
        # Zero offset should give 2*t0
        t0_s = 1.0
        velocity = 2000.0
        
        t_travel = compute_dsr_travel_time(
            source_x=0, source_y=0,
            receiver_x=0, receiver_y=0,
            diffractor_x=0, diffractor_y=0,
            t0_s=t0_s,
            velocity=velocity,
        )
        
        assert abs(t_travel - 2*t0_s) < 0.001

    def test_offset_increases_time(self):
        t0_s = 1.0
        velocity = 2000.0
        
        t_zero = compute_dsr_travel_time(
            0, 0, 0, 0, 0, 0, t0_s, velocity
        )
        
        t_offset = compute_dsr_travel_time(
            -500, 0, 500, 0, 0, 0, t0_s, velocity
        )
        
        assert t_offset > t_zero

    def test_symmetric_offset(self):
        t0_s = 1.0
        velocity = 2000.0
        
        # Source and receiver equidistant from diffractor
        t1 = compute_dsr_travel_time(
            source_x=-500, source_y=0,
            receiver_x=500, receiver_y=0,
            diffractor_x=0, diffractor_y=0,
            t0_s=t0_s, velocity=velocity,
        )
        
        t2 = compute_dsr_travel_time(
            source_x=500, source_y=0,
            receiver_x=-500, receiver_y=0,
            diffractor_x=0, diffractor_y=0,
            t0_s=t0_s, velocity=velocity,
        )
        
        assert abs(t1 - t2) < 0.001


class TestSyntheticConfig:
    def test_add_diffractor(self):
        config = SyntheticConfig()
        config.add_diffractor(100, 200, 500, amplitude=2.0)
        
        assert len(config.diffractors) == 1
        assert config.diffractors[0].x == 100
        assert config.diffractors[0].amplitude == 2.0

    def test_add_offset_azimuth_plane(self):
        config = SyntheticConfig()
        config.add_offset_azimuth_plane(offset=1000, azimuth_deg=45)
        
        assert config.n_planes == 1
        assert config.offset_azimuth_planes[0].offset == 1000

    def test_add_offset_azimuth_range(self):
        config = SyntheticConfig()
        config.add_offset_azimuth_range(
            offsets=[500, 1000],
            azimuths=[0, 90],
        )
        
        assert config.n_planes == 4  # 2 offsets × 2 azimuths


class TestGenerateSyntheticGathers:
    def test_simple_generation(self):
        config = SyntheticConfig(
            survey=SurveyGeometry(
                x_min=0, x_max=500, y_min=0, y_max=500, dx=50, dy=50
            ),
            trace_params=TraceParameters(n_samples=501, dt_ms=2.0),
            velocity_ms=2000.0,
        )
        config.add_diffractor(250, 250, 500)
        config.add_offset_azimuth_plane(offset=200, azimuth_deg=0)
        
        result = generate_synthetic_gathers(config)
        
        assert result.n_traces == config.survey.n_positions
        assert result.traces.shape[1] == 501
        assert len(result.source_x) == result.n_traces

    def test_multiple_planes(self):
        config = SyntheticConfig(
            survey=SurveyGeometry(x_min=0, x_max=200, y_min=0, y_max=200, dx=50, dy=50),
            trace_params=TraceParameters(n_samples=251, dt_ms=4.0),
        )
        config.add_diffractor(100, 100, 400)
        config.add_offset_azimuth_range([100, 200], [0, 90])
        
        result = generate_synthetic_gathers(config)
        
        expected_traces = config.survey.n_positions * 4  # 4 planes
        assert result.n_traces == expected_traces

    def test_diffractor_response_present(self):
        config = SyntheticConfig(
            survey=SurveyGeometry(x_min=0, x_max=500, y_min=0, y_max=500, dx=50, dy=50),
            trace_params=TraceParameters(n_samples=1001, dt_ms=2.0),
            velocity_ms=2000.0,
        )
        config.add_diffractor(250, 250, 500)  # 500m depth -> t0=0.5s
        config.add_offset_azimuth_plane(offset=200, azimuth_deg=0)  # 200m offset
        
        result = generate_synthetic_gathers(config)
        
        # Check signal is present
        assert np.max(np.abs(result.traces)) > 0

    def test_headers_consistency(self):
        config = SyntheticConfig(
            survey=SurveyGeometry(x_min=0, x_max=100, y_min=0, y_max=100, dx=50, dy=50),
        )
        config.add_diffractor(50, 50, 500)
        config.add_offset_azimuth_plane(offset=100, azimuth_deg=90)
        
        result = generate_synthetic_gathers(config)
        
        # Midpoint should be average of source and receiver
        for i in range(result.n_traces):
            expected_mx = (result.source_x[i] + result.receiver_x[i]) / 2
            expected_my = (result.source_y[i] + result.receiver_y[i]) / 2
            assert abs(result.midpoint_x[i] - expected_mx) < 0.01
            assert abs(result.midpoint_y[i] - expected_my) < 0.01


class TestConvenienceFunctions:
    def test_create_simple_synthetic(self):
        result = create_simple_synthetic(
            diffractor_x=500, diffractor_y=500, diffractor_z=500,
            survey_extent=1000,
            grid_spacing=50,
            offsets=[200],
            azimuths=[0],
            n_samples=501,
        )
        
        assert result.n_traces > 0
        assert result.n_samples == 501

    def test_create_multi_diffractor_synthetic(self):
        result = create_multi_diffractor_synthetic(
            diffractor_locations=[
                (500, 500, 500),
                (1500, 1500, 1000),
            ],
            survey_x_range=(0, 2000),
            survey_y_range=(0, 2000),
            grid_spacing=100,
            offset_x_values=[0, 200],
            offset_y_values=[0, 200],
        )
        
        assert result.n_traces > 0
        # Check multiple diffractors contribute
        assert np.max(np.abs(result.traces)) > 0


class TestExport:
    @pytest.fixture
    def sample_result(self):
        return create_simple_synthetic(
            diffractor_x=500, diffractor_y=500, diffractor_z=500,
            survey_extent=500,
            grid_spacing=100,
            offsets=[200],
            azimuths=[0],
            n_samples=251,
        )

    def test_export_zarr_parquet(self, sample_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            traces_path, headers_path = export_to_zarr_parquet(
                sample_result, tmpdir
            )
            
            assert traces_path.exists()
            assert headers_path.exists()
            
            # Verify we can read back
            import zarr
            from zarr.storage import LocalStore
            store = LocalStore(str(traces_path))
            z = zarr.open_array(store, mode='r')
            assert z.shape == sample_result.traces.shape
            
            import polars as pl
            df = pl.read_parquet(headers_path)
            assert len(df) == sample_result.n_traces


class TestNoiseAddition:
    def test_noise_increases_variance(self):
        config = SyntheticConfig(
            survey=SurveyGeometry(x_min=0, x_max=200, y_min=0, y_max=200, dx=50, dy=50),
            noise_level=0.0,
        )
        config.add_diffractor(100, 100, 500)
        config.add_offset_azimuth_plane(offset=100, azimuth_deg=0)
        
        result_clean = generate_synthetic_gathers(config)
        
        config.noise_level = 0.5
        result_noisy = generate_synthetic_gathers(config)
        
        std_clean = np.std(result_clean.traces)
        std_noisy = np.std(result_noisy.traces)
        
        # Noisy data should have higher variance
        assert std_noisy > std_clean


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
