"""Tests for interpolation module."""

import numpy as np
import pytest

from pstm.kernels.interpolation import (
    InterpolationMethod,
    get_available_methods,
    get_method_code,
    get_method_info,
    recommend_method,
    interpolate_sample,
    interpolate_traces_batch,
    _linear_interp,
    _cubic_interp,
    _sinc8_interp,
    _lanczos3_interp,
)


class TestInterpolationMethod:
    def test_enum_values(self):
        assert InterpolationMethod.LINEAR.value == "linear"
        assert InterpolationMethod.SINC8.value == "sinc8"
        assert InterpolationMethod.CUBIC.value == "cubic"

    def test_from_string(self):
        assert InterpolationMethod.from_string("linear") == InterpolationMethod.LINEAR
        assert InterpolationMethod.from_string("sinc") == InterpolationMethod.SINC8
        assert InterpolationMethod.from_string("CUBIC") == InterpolationMethod.CUBIC


class TestGetMethodCode:
    def test_known_methods(self):
        assert get_method_code("nearest") == 0
        assert get_method_code("linear") == 1
        assert get_method_code("cubic") == 2
        assert get_method_code("sinc4") == 3
        assert get_method_code("sinc8") == 4
        assert get_method_code("sinc16") == 5
        assert get_method_code("lanczos3") == 6
        assert get_method_code("lanczos5") == 7

    def test_aliases(self):
        assert get_method_code("sinc") == 4  # Default sinc = sinc8
        assert get_method_code("spline") == 2  # Alias for cubic
        assert get_method_code("lanczos") == 6  # Default lanczos = lanczos3

    def test_unknown_defaults_to_linear(self):
        assert get_method_code("unknown") == 1


class TestGetMethodInfo:
    def test_linear_info(self):
        info = get_method_info("linear")
        assert info["name"] == "Linear"
        assert info["points"] == 2
        assert info["quality"] == "medium"
        assert info["speed"] == "fast"

    def test_sinc8_info(self):
        info = get_method_info("sinc8")
        assert info["name"] == "8-point Sinc"
        assert info["points"] == 8
        assert info["quality"] == "high"

    def test_all_methods_have_info(self):
        for method in get_available_methods():
            info = get_method_info(method)
            assert "name" in info
            assert "points" in info
            assert "quality" in info
            assert "speed" in info


class TestRecommendMethod:
    def test_speed_priority(self):
        assert recommend_method("speed") == "linear"

    def test_quality_priority(self):
        assert recommend_method("quality") == "sinc8"

    def test_balanced_default(self):
        method = recommend_method("balanced")
        assert method in ["linear", "cubic", "sinc4", "sinc8"]


class TestLinearInterp:
    def test_exact_sample(self):
        trace = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert _linear_interp(trace, 2.0) == 2.0

    def test_midpoint(self):
        trace = np.array([0.0, 2.0, 4.0])
        assert _linear_interp(trace, 0.5) == 1.0

    def test_out_of_bounds(self):
        trace = np.array([0.0, 1.0, 2.0])
        assert _linear_interp(trace, -1.0) == 0.0
        assert _linear_interp(trace, 5.0) == 0.0


class TestCubicInterp:
    def test_smooth_interpolation(self):
        # Cubic should give smoother results than linear
        trace = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        linear = _linear_interp(trace, 2.5)
        cubic = _cubic_interp(trace, 2.5)
        # Both should be between 0 and 1
        assert 0 <= linear <= 1
        assert 0 <= cubic <= 1

    def test_fallback_at_edges(self):
        trace = np.array([1.0, 2.0, 3.0])
        # Should fall back to linear at edges
        result = _cubic_interp(trace, 0.5)
        assert result > 0


class TestSinc8Interp:
    def test_preserves_samples(self):
        trace = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # At exact sample, should return sample value
        result = _sinc8_interp(trace, 4.0)
        assert abs(result - 1.0) < 0.1  # Allow some tolerance

    def test_near_edge_uses_linear(self):
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Near edge should fall back to linear
        result = _sinc8_interp(trace, 0.5)
        assert result > 0


class TestLanczos3Interp:
    def test_basic_interpolation(self):
        trace = np.linspace(0, 10, 20)
        result = _lanczos3_interp(trace, 10.5)
        # Should be close to linear interpolation value
        assert 5 < result < 6


class TestInterpolateSample:
    def test_method_selection(self):
        trace = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        t = 4.5
        
        # All methods should give reasonable results
        for method_code in range(8):
            result = interpolate_sample(trace, t, method_code)
            assert 4 < result < 5.5


class TestInterpolateTracesBatch:
    def test_batch_processing(self):
        n_traces = 100
        n_samples = 50
        traces = np.random.randn(n_traces, n_samples).astype(np.float64)
        t_samples = np.random.uniform(5, n_samples - 10, n_traces)
        
        results = interpolate_traces_batch(traces, t_samples, 1)  # Linear
        assert results.shape == (n_traces,)
        assert not np.any(np.isnan(results))


class TestInterpolationAccuracy:
    """Test interpolation accuracy on known functions."""

    def test_linear_func_with_linear_interp(self):
        """Linear interpolation should be exact for linear functions."""
        x = np.arange(100, dtype=np.float64)
        trace = 2.0 * x + 1.0  # Linear function
        
        for t in [10.3, 25.7, 50.5]:
            expected = 2.0 * t + 1.0
            result = _linear_interp(trace, t)
            assert abs(result - expected) < 1e-10

    def test_sinusoid_interpolation(self):
        """Test interpolation of sinusoidal signal."""
        n_samples = 200
        x = np.arange(n_samples, dtype=np.float64)
        trace = np.sin(2 * np.pi * x / 20)  # Period of 20 samples
        
        # Test at various points
        for t in [50.25, 75.5, 100.75]:
            expected = np.sin(2 * np.pi * t / 20)
            
            linear = _linear_interp(trace, t)
            cubic = _cubic_interp(trace, t)
            sinc8 = _sinc8_interp(trace, t)
            
            # Sinc should be most accurate for band-limited signals
            assert abs(sinc8 - expected) < abs(linear - expected) or abs(sinc8 - expected) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
