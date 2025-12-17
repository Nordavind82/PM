"""Comprehensive tests for utilities and edge case handling."""

import numpy as np
import pytest

from pstm.utils.units import (
    meters_to_feet, feet_to_meters,
    ms_to_s, s_to_ms,
    samples_to_time, time_to_samples,
    offset_to_midpoint, compute_offset, compute_azimuth,
    format_bytes, format_duration,
)
from pstm.utils.edge_cases import (
    ValidationResult, ValidationSeverity, ValidationIssue,
    validate_trace_data, validate_coordinates, validate_velocity,
    validate_output_grid,
    safe_divide, clip_to_valid_range, handle_empty_input,
    handle_zero_traces_in_aperture, handle_invalid_travel_time,
    handle_interpolation_bounds, handle_memory_overflow,
)


class TestUnitConversions:
    def test_meters_to_feet(self):
        assert meters_to_feet(1.0) == pytest.approx(3.28084, rel=1e-4)
        assert meters_to_feet(0) == 0
        
    def test_feet_to_meters(self):
        assert feet_to_meters(3.28084) == pytest.approx(1.0, rel=1e-4)
        
    def test_round_trip(self):
        original = 1000.0
        assert feet_to_meters(meters_to_feet(original)) == pytest.approx(original)

    def test_ms_to_s(self):
        assert ms_to_s(1000) == 1.0
        assert ms_to_s(2500) == 2.5

    def test_s_to_ms(self):
        assert s_to_ms(1.0) == 1000
        assert s_to_ms(2.5) == 2500

    def test_samples_to_time(self):
        result = samples_to_time(100, sample_rate_ms=2.0, start_time_ms=0)
        assert result == 200.0

    def test_time_to_samples(self):
        result = time_to_samples(200.0, sample_rate_ms=2.0, start_time_ms=0)
        assert result == 100


class TestGeometryCalculations:
    def test_compute_offset(self):
        offset = compute_offset(
            source_x=0, source_y=0,
            receiver_x=100, receiver_y=0,
        )
        assert offset == 100

    def test_compute_offset_2d(self):
        offset = compute_offset(
            source_x=0, source_y=0,
            receiver_x=30, receiver_y=40,
        )
        assert offset == 50  # 3-4-5 triangle

    def test_compute_azimuth_east(self):
        azimuth = compute_azimuth(
            source_x=0, source_y=0,
            receiver_x=100, receiver_y=0,
        )
        assert 85 < azimuth < 95  # Should be ~90 (East)

    def test_compute_azimuth_north(self):
        azimuth = compute_azimuth(
            source_x=0, source_y=0,
            receiver_x=0, receiver_y=100,
        )
        assert azimuth < 5 or azimuth > 355  # Should be ~0 (North)

    def test_offset_to_midpoint(self):
        mx, my = offset_to_midpoint(
            source_x=0, source_y=0,
            receiver_x=100, receiver_y=100,
        )
        assert mx == 50
        assert my == 50


class TestFormatting:
    def test_format_bytes_bytes(self):
        result = format_bytes(100)
        assert "100" in result and "B" in result
        
    def test_format_bytes_kb(self):
        assert "KB" in format_bytes(2000)
        
    def test_format_bytes_mb(self):
        assert "MB" in format_bytes(2_000_000)
        
    def test_format_bytes_gb(self):
        assert "GB" in format_bytes(2_000_000_000)

    def test_format_duration_seconds(self):
        result = format_duration(45)
        assert "45" in result or "s" in result.lower()
        
    def test_format_duration_minutes(self):
        result = format_duration(125)
        assert "m" in result.lower() or "2" in result
        
    def test_format_duration_hours(self):
        result = format_duration(3700)
        assert "h" in result.lower() or "1" in result


class TestValidationResult:
    def test_empty_result_is_valid(self):
        result = ValidationResult()
        assert result.is_valid
        assert not result.has_warnings

    def test_warning_still_valid(self):
        result = ValidationResult()
        result.add("field", "warning message", ValidationSeverity.WARNING)
        assert result.is_valid
        assert result.has_warnings

    def test_error_makes_invalid(self):
        result = ValidationResult()
        result.add("field", "error message", ValidationSeverity.ERROR)
        assert not result.is_valid

    def test_raise_if_invalid(self):
        result = ValidationResult()
        result.add("field", "error", ValidationSeverity.ERROR)
        with pytest.raises(ValueError):
            result.raise_if_invalid()

    def test_merge_results(self):
        result1 = ValidationResult()
        result1.add("a", "msg1", ValidationSeverity.WARNING)
        result2 = ValidationResult()
        result2.add("b", "msg2", ValidationSeverity.ERROR)
        result1.merge(result2)
        assert len(result1.issues) == 2
        assert not result1.is_valid


class TestValidateTraceData:
    def test_valid_data(self):
        data = np.random.randn(100, 500).astype(np.float32)
        result = validate_trace_data(data)
        assert result.is_valid

    def test_wrong_dimensions(self):
        data = np.random.randn(100).astype(np.float32)  # 1D
        result = validate_trace_data(data)
        assert not result.is_valid

    def test_empty_data(self):
        data = np.zeros((0, 100), dtype=np.float32)
        result = validate_trace_data(data)
        assert not result.is_valid

    def test_nan_values(self):
        data = np.random.randn(100, 500).astype(np.float32)
        data[50, 250] = np.nan
        result = validate_trace_data(data, max_nan_fraction=0.0)
        assert not result.is_valid


class TestValidateCoordinates:
    def test_valid_coordinates(self):
        x = np.random.uniform(0, 10000, 100)
        y = np.random.uniform(0, 10000, 100)
        result = validate_coordinates(x, y)
        assert result.is_valid

    def test_mismatched_lengths(self):
        x = np.zeros(100)
        y = np.zeros(50)
        result = validate_coordinates(x, y)
        assert not result.is_valid

    def test_nan_coordinates(self):
        x = np.zeros(100)
        x[0] = np.nan
        y = np.zeros(100)
        result = validate_coordinates(x, y)
        assert not result.is_valid


class TestValidateVelocity:
    def test_valid_velocity(self):
        velocity = np.linspace(1500, 3500, 100)
        result = validate_velocity(velocity)
        assert result.is_valid

    def test_low_velocity(self):
        velocity = np.array([500])
        result = validate_velocity(velocity, min_valid=1000)
        assert not result.is_valid

    def test_high_velocity(self):
        velocity = np.array([15000])
        result = validate_velocity(velocity, max_valid=8000)
        assert not result.is_valid


class TestValidateOutputGrid:
    def test_valid_grid(self):
        result = validate_output_grid(
            x_min=0, x_max=1000, dx=25,
            y_min=0, y_max=1000, dy=25,
            t_min_ms=0, t_max_ms=2000, dt_ms=2,
        )
        assert result.is_valid

    def test_invalid_x_range(self):
        result = validate_output_grid(
            x_min=1000, x_max=0, dx=25,
            y_min=0, y_max=1000, dy=25,
            t_min_ms=0, t_max_ms=2000, dt_ms=2,
        )
        assert not result.is_valid

    def test_negative_spacing(self):
        result = validate_output_grid(
            x_min=0, x_max=1000, dx=-25,
            y_min=0, y_max=1000, dy=25,
            t_min_ms=0, t_max_ms=2000, dt_ms=2,
        )
        assert not result.is_valid


class TestSafeDivide:
    def test_normal_division(self):
        result = safe_divide(np.array([10, 20]), np.array([2, 4]))
        np.testing.assert_array_equal(result, [5, 5])

    def test_zero_denominator(self):
        result = safe_divide(np.array([10, 20]), np.array([2, 0]), fill_value=0)
        np.testing.assert_array_equal(result, [5, 0])

    def test_custom_fill(self):
        result = safe_divide(np.array([10]), np.array([0]), fill_value=-999)
        assert result[0] == -999


class TestClipToValidRange:
    def test_basic_clip(self):
        data = np.array([-10, 0, 50, 150])
        result = clip_to_valid_range(data, min_val=0, max_val=100)
        np.testing.assert_array_equal(result, [0, 0, 50, 100])

    def test_replace_nan(self):
        data = np.array([1, np.nan, 3])
        result = clip_to_valid_range(data, replace_nan=0)
        np.testing.assert_array_equal(result, [1, 0, 3])

    def test_replace_inf(self):
        data = np.array([1, np.inf, 3])
        result = clip_to_valid_range(data, replace_inf=999)
        np.testing.assert_array_equal(result, [1, 999, 3])


class TestHandleEmptyInput:
    def test_none_input(self):
        result = handle_empty_input(None, default_shape=(10, 10))
        assert result.shape == (10, 10)

    def test_empty_array(self):
        result = handle_empty_input(np.array([]), default_shape=(5,))
        assert result.shape == (5,)

    def test_valid_input_unchanged(self):
        original = np.array([1, 2, 3])
        result = handle_empty_input(original, default_shape=(10,))
        np.testing.assert_array_equal(result, original)


class TestHandleZeroTracesInAperture:
    def test_returns_skip(self):
        result = handle_zero_traces_in_aperture(0, tile_id=5)
        assert result['skip'] is True
        assert result['n_traces'] == 0


class TestHandleInvalidTravelTime:
    def test_valid_time(self):
        is_valid, time = handle_invalid_travel_time(1.5)
        assert is_valid
        assert time == 1.5

    def test_negative_time(self):
        is_valid, time = handle_invalid_travel_time(-1.0)
        assert not is_valid

    def test_nan_time(self):
        is_valid, time = handle_invalid_travel_time(np.nan)
        assert not is_valid

    def test_excessive_time(self):
        is_valid, time = handle_invalid_travel_time(100.0, max_valid_time_s=10.0)
        assert not is_valid


class TestHandleInterpolationBounds:
    def test_valid_index(self):
        is_valid, idx = handle_interpolation_bounds(50.5, n_samples=100)
        assert is_valid
        assert idx == 50.5

    def test_out_of_bounds_skip(self):
        is_valid, idx = handle_interpolation_bounds(-5, n_samples=100, method="skip")
        assert not is_valid

    def test_out_of_bounds_clamp(self):
        is_valid, idx = handle_interpolation_bounds(-5, n_samples=100, method="clamp")
        assert is_valid
        assert idx >= 0


class TestHandleMemoryOverflow:
    def test_sufficient_memory(self):
        result = handle_memory_overflow(
            requested_bytes=int(1e9),
            available_bytes=int(2e9),
        )
        assert result['action'] == 'proceed'

    def test_reduce_strategy(self):
        result = handle_memory_overflow(
            requested_bytes=int(2e9),
            available_bytes=int(1e9),
            strategy="reduce",
        )
        assert result['action'] == 'reduce'
        assert result['reduction_factor'] < 1.0

    def test_error_strategy(self):
        with pytest.raises(MemoryError):
            handle_memory_overflow(
                requested_bytes=int(2e9),
                available_bytes=int(1e9),
                strategy="error",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
