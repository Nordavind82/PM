"""Unit tests for time-variant sampling module."""

import numpy as np
import pytest

from pstm.algorithm.time_variant import (
    FrequencyTimeTable,
    TimeWindow,
    interpolate_fmax,
    nearest_power_of_2,
    compute_downsample_factor,
    compute_time_windows,
    estimate_speedup,
    create_output_sample_map,
    resample_to_uniform,
    get_window_info_string,
)


class TestFrequencyTimeTable:
    """Tests for FrequencyTimeTable dataclass."""

    def test_creation_valid(self):
        """Test creating valid table."""
        table = FrequencyTimeTable(
            times_ms=[0, 1000, 2000],
            frequencies_hz=[80, 50, 30],
        )
        assert table.n_entries == 3
        assert table.t_min == 0
        assert table.t_max == 2000

    def test_from_list(self):
        """Test creating from list of tuples."""
        pairs = [(1000, 50), (0, 80), (2000, 30)]  # Unsorted
        table = FrequencyTimeTable.from_list(pairs)

        # Should be sorted by time
        assert table.times_ms == [0, 1000, 2000]
        assert table.frequencies_hz == [80, 50, 30]

    def test_to_list(self):
        """Test converting to list of tuples."""
        table = FrequencyTimeTable(
            times_ms=[0, 1000],
            frequencies_hz=[80, 50],
        )
        result = table.to_list()
        assert result == [(0, 80), (1000, 50)]

    def test_default_table(self):
        """Test default table creation."""
        table = FrequencyTimeTable.default()
        assert table.n_entries == 5
        assert table.t_min == 0
        assert table.frequencies_hz[0] == 80  # High freq at shallow

    def test_invalid_length_mismatch(self):
        """Test error on mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            FrequencyTimeTable(times_ms=[0, 1000], frequencies_hz=[80])

    def test_invalid_too_few_entries(self):
        """Test error on single entry."""
        with pytest.raises(ValueError, match="at least 2"):
            FrequencyTimeTable(times_ms=[0], frequencies_hz=[80])

    def test_invalid_non_increasing_times(self):
        """Test error on non-increasing times."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            FrequencyTimeTable(
                times_ms=[0, 1000, 500],  # Not increasing
                frequencies_hz=[80, 50, 60],
            )

    def test_invalid_negative_frequency(self):
        """Test error on negative frequency."""
        with pytest.raises(ValueError, match="positive"):
            FrequencyTimeTable(
                times_ms=[0, 1000],
                frequencies_hz=[80, -10],
            )


class TestInterpolateFmax:
    """Tests for frequency interpolation."""

    @pytest.fixture
    def table(self):
        return FrequencyTimeTable(
            times_ms=[0, 1000, 2000, 3000],
            frequencies_hz=[80, 60, 40, 20],
        )

    def test_at_table_points(self, table):
        """Test interpolation at exact table points."""
        assert interpolate_fmax(0, table) == 80
        assert interpolate_fmax(1000, table) == 60
        assert interpolate_fmax(2000, table) == 40
        assert interpolate_fmax(3000, table) == 20

    def test_between_points(self, table):
        """Test interpolation between table points."""
        f_500 = interpolate_fmax(500, table)
        assert 60 < f_500 < 80  # Between first two values

        f_1500 = interpolate_fmax(1500, table)
        assert 40 < f_1500 < 60  # Between second and third

    def test_clamp_below_min(self, table):
        """Test clamping below minimum time."""
        assert interpolate_fmax(-100, table) == 80

    def test_clamp_above_max(self, table):
        """Test clamping above maximum time."""
        assert interpolate_fmax(5000, table) == 20


class TestNearestPowerOf2:
    """Tests for power of 2 rounding."""

    def test_exact_powers(self):
        """Test exact powers of 2."""
        assert nearest_power_of_2(1) == 1
        assert nearest_power_of_2(2) == 2
        assert nearest_power_of_2(4) == 4
        assert nearest_power_of_2(8) == 8

    def test_round_down(self):
        """Test rounding down to nearest power."""
        assert nearest_power_of_2(3) == 2  # Closer to 2 than 4
        assert nearest_power_of_2(5) == 4  # Closer to 4 than 8
        assert nearest_power_of_2(6) == 4  # Closer to 4 than 8

    def test_round_up(self):
        """Test rounding up to nearest power."""
        assert nearest_power_of_2(7) == 8  # Closer to 8 than 4

    def test_zero_and_negative(self):
        """Test edge cases."""
        assert nearest_power_of_2(0) == 1
        assert nearest_power_of_2(-1) == 1


class TestComputeDownsampleFactor:
    """Tests for downsample factor computation."""

    def test_high_frequency_no_downsample(self):
        """High frequency requires fine sampling."""
        # f_max=80Hz, dt_base=2ms
        # Nyquist dt = 1000/(2*80) = 6.25ms -> factor = 3 -> round to 2
        factor = compute_downsample_factor(80, 2.0)
        assert factor in [1, 2]  # Can't downsample much

    def test_low_frequency_more_downsample(self):
        """Low frequency allows coarser sampling."""
        # f_max=20Hz, dt_base=2ms
        # Nyquist dt = 1000/(2*20) = 25ms -> factor = 12 -> round to 8
        factor = compute_downsample_factor(20, 2.0, max_factor=8)
        assert factor == 8

    def test_respects_min_factor(self):
        """Test minimum factor constraint."""
        factor = compute_downsample_factor(80, 2.0, min_factor=2)
        assert factor >= 2

    def test_respects_max_factor(self):
        """Test maximum factor constraint."""
        factor = compute_downsample_factor(10, 2.0, max_factor=4)
        assert factor <= 4

    def test_result_is_power_of_2(self):
        """Result should always be power of 2."""
        for f_max in [10, 20, 30, 50, 80, 100]:
            factor = compute_downsample_factor(f_max, 2.0)
            assert factor & (factor - 1) == 0, f"Factor {factor} is not power of 2"


class TestComputeTimeWindows:
    """Tests for time window computation."""

    @pytest.fixture
    def table(self):
        return FrequencyTimeTable(
            times_ms=[0, 1000, 3000],
            frequencies_hz=[80, 40, 20],
        )

    def test_creates_windows(self, table):
        """Test that windows are created."""
        windows = compute_time_windows(0, 3000, 2.0, table)
        assert len(windows) > 0

    def test_windows_cover_range(self, table):
        """Test windows cover entire time range."""
        windows = compute_time_windows(0, 3000, 2.0, table)

        # First window starts at 0
        assert windows[0].t_start_ms == 0

        # Last window ends at or near t_max
        assert windows[-1].t_end_ms >= 2900  # Close to 3000

    def test_windows_are_contiguous(self, table):
        """Test windows don't overlap or have gaps."""
        windows = compute_time_windows(0, 3000, 2.0, table)

        for i in range(len(windows) - 1):
            # End of window i should equal start of window i+1
            assert abs(windows[i].t_end_ms - windows[i+1].t_start_ms) < 0.01

    def test_downsample_factors_are_valid(self, table):
        """Test all downsample factors are powers of 2."""
        windows = compute_time_windows(0, 3000, 2.0, table)

        for win in windows:
            factor = win.downsample_factor
            assert factor & (factor - 1) == 0, f"Factor {factor} not power of 2"
            assert 1 <= factor <= 8

    def test_effective_dt_matches_factor(self, table):
        """Test effective dt = base_dt * factor."""
        base_dt = 2.0
        windows = compute_time_windows(0, 3000, base_dt, table)

        for win in windows:
            expected_dt = base_dt * win.downsample_factor
            assert abs(win.dt_effective_ms - expected_dt) < 0.01

    def test_sample_indices_are_contiguous(self, table):
        """Test sample indices form contiguous sequence."""
        windows = compute_time_windows(0, 3000, 2.0, table)

        for i in range(len(windows) - 1):
            assert windows[i].sample_end == windows[i+1].sample_start

        # First window starts at 0
        assert windows[0].sample_start == 0


class TestEstimateSpeedup:
    """Tests for speedup estimation."""

    def test_uniform_sampling_minimal_speedup(self):
        """Uniform very high frequency gives minimal speedup."""
        table = FrequencyTimeTable(
            times_ms=[0, 3000],
            frequencies_hz=[200, 200],  # Very high freq - needs fine sampling
        )
        speedup = estimate_speedup(0, 3000, 2.0, table)
        # At 200Hz, Nyquist dt = 2.5ms, so factor=1 throughout
        assert 0.9 <= speedup <= 1.5  # Minimal speedup

    def test_decreasing_frequency_gives_speedup(self):
        """Decreasing frequency gives speedup."""
        table = FrequencyTimeTable(
            times_ms=[0, 1000, 3000],
            frequencies_hz=[80, 40, 20],
        )
        speedup = estimate_speedup(0, 3000, 2.0, table)
        assert speedup > 1.5  # Should be significant speedup


class TestCreateOutputSampleMap:
    """Tests for output sample mapping."""

    def test_single_window_identity(self):
        """Single window with factor 1 gives identity mapping."""
        windows = [TimeWindow(
            t_start_ms=0, t_end_ms=100, dt_effective_ms=2.0,
            downsample_factor=1, sample_start=0, sample_end=50, f_max_hz=80
        )]
        sample_map = create_output_sample_map(windows, 2.0, 50)

        # Should be 0, 1, 2, ..., 49
        expected = np.arange(50, dtype=np.int32)
        np.testing.assert_array_equal(sample_map, expected)

    def test_downsampled_window(self):
        """Downsampled window maps correctly."""
        windows = [TimeWindow(
            t_start_ms=0, t_end_ms=100, dt_effective_ms=4.0,  # 2x downsample
            downsample_factor=2, sample_start=0, sample_end=25, f_max_hz=40
        )]
        sample_map = create_output_sample_map(windows, 2.0, 50)

        # Should be 0, 2, 4, ..., 48
        expected = np.arange(0, 50, 2, dtype=np.int32)
        np.testing.assert_array_equal(sample_map, expected)


class TestResampleToUniform:
    """Tests for resampling to uniform grid."""

    def test_preserves_shape(self):
        """Output has correct shape."""
        nx, ny = 10, 10
        windows = [
            TimeWindow(0, 50, 2.0, 1, 0, 25, 80),
            TimeWindow(50, 100, 4.0, 2, 25, 38, 40),
        ]
        n_tv_samples = sum(w.n_samples for w in windows)
        tv_image = np.random.randn(nx, ny, n_tv_samples)

        output = resample_to_uniform(tv_image, windows, 2.0, 50)

        assert output.shape == (nx, ny, 50)

    def test_preserves_values_no_downsample(self):
        """No downsampling preserves values."""
        nx, ny = 2, 2
        windows = [TimeWindow(0, 100, 2.0, 1, 0, 50, 80)]

        # Create simple ramp
        tv_image = np.zeros((nx, ny, 50))
        tv_image[0, 0, :] = np.arange(50)

        output = resample_to_uniform(tv_image, windows, 2.0, 50)

        # Should be identical
        np.testing.assert_array_almost_equal(output[0, 0, :], np.arange(50))


class TestWindowInfoString:
    """Tests for window info string generation."""

    def test_generates_string(self):
        """Test string is generated."""
        windows = [
            TimeWindow(0, 1000, 2.0, 1, 0, 500, 80),
            TimeWindow(1000, 3000, 4.0, 2, 500, 1000, 40),
        ]
        info = get_window_info_string(windows)

        assert "2 windows" in info
        assert "80Hz" in info
        assert "40Hz" in info
        assert "2x" in info


class TestTimeVariantConfig:
    """Tests for configuration model."""

    def test_default_config(self):
        """Test default configuration."""
        from pstm.config.models import TimeVariantConfig

        config = TimeVariantConfig()
        assert config.enabled is False
        assert len(config.frequency_table) == 4
        assert config.min_downsample_factor == 1
        assert config.max_downsample_factor == 8

    def test_validation_too_few_entries(self):
        """Test validation rejects too few entries."""
        from pstm.config.models import TimeVariantConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TimeVariantConfig(frequency_table=[(0, 80)])

    def test_validation_negative_frequency(self):
        """Test validation rejects negative frequency."""
        from pstm.config.models import TimeVariantConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TimeVariantConfig(frequency_table=[(0, 80), (1000, -10)])

    def test_validation_non_power_of_2(self):
        """Test validation rejects non-power-of-2 max factor."""
        from pstm.config.models import TimeVariantConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TimeVariantConfig(max_downsample_factor=5)
