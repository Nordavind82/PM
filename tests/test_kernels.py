"""Comprehensive tests for migration kernels."""

import numpy as np
import pytest

from pstm.kernels.base import (
    TraceBlock, OutputTile, VelocitySlice, KernelConfig, KernelMetrics,
    KernelCapability, create_trace_block, create_output_tile,
)
from pstm.kernels.numba_cpu import NumbaKernel
from pstm.kernels.numpy_reference import NumpyReferenceKernel
from pstm.kernels.factory import create_kernel, get_available_backends


class TestTraceBlock:
    def test_create_trace_block(self):
        block = create_trace_block(
            amplitudes=np.random.randn(100, 500).astype(np.float32),
            source_x=np.zeros(100), source_y=np.zeros(100),
            receiver_x=np.linspace(0, 1000, 100), receiver_y=np.zeros(100),
            sample_rate_ms=2.0,
        )
        assert block.n_traces == 100
        assert block.n_samples == 500

    def test_offset_calculation(self):
        block = create_trace_block(
            amplitudes=np.zeros((10, 100), dtype=np.float32),
            source_x=np.zeros(10), source_y=np.zeros(10),
            receiver_x=np.ones(10) * 100, receiver_y=np.zeros(10),
            sample_rate_ms=2.0,
        )
        np.testing.assert_array_almost_equal(block.offset, 100.0)

    def test_validate_valid_block(self):
        block = create_trace_block(
            amplitudes=np.random.randn(10, 100).astype(np.float32),
            source_x=np.zeros(10), source_y=np.zeros(10),
            receiver_x=np.zeros(10), receiver_y=np.zeros(10),
            sample_rate_ms=2.0,
        )
        assert len(block.validate()) == 0


class TestOutputTile:
    def test_create_output_tile(self):
        tile = create_output_tile(
            x_min=0, x_max=100, dx=10,
            y_min=0, y_max=100, dy=10,
            t_min_ms=0, t_max_ms=1000, dt_ms=2,
        )
        assert tile.nx == 11
        assert tile.ny == 11
        assert tile.nt == 501

    def test_reset(self):
        tile = create_output_tile(0, 100, 25, 0, 100, 25, 0, 100, 2)
        tile.image[:] = 1.0
        tile.fold[:] = 10
        tile.reset()
        assert np.all(tile.image == 0.0)
        assert np.all(tile.fold == 0)


class TestKernelConfig:
    def test_default_config(self):
        config = KernelConfig()
        assert config.max_aperture_m == 5000.0
        assert config.apply_spreading is True


class TestNumbaKernel:
    @pytest.fixture
    def kernel(self):
        config = KernelConfig(max_aperture_m=200.0)
        k = NumbaKernel()
        k.initialize(config)
        return k

    def test_initialization(self, kernel):
        assert kernel._initialized
        assert kernel.name == "numba_cpu"

    def test_migrate_basic(self, kernel):
        traces = create_trace_block(
            amplitudes=np.random.randn(10, 100).astype(np.float32),
            source_x=np.zeros(10), source_y=np.zeros(10),
            receiver_x=np.linspace(0, 100, 10), receiver_y=np.zeros(10),
            sample_rate_ms=2.0,
        )
        output = create_output_tile(0, 50, 10, -25, 25, 10, 0, 100, 2)
        velocity = VelocitySlice(vrms=np.full(output.nt, 2000.0), is_1d=True)
        
        # NumbaKernel.migrate_tile takes only 3 args (traces, output, velocity)
        # config is set during initialize()
        metrics = kernel.migrate_tile(traces, output, velocity)
        assert metrics.n_traces_processed == 10


class TestKernelFactory:
    def test_get_available_backends(self):
        backends = get_available_backends()
        assert len(backends) > 0

    def test_create_auto_kernel(self):
        from pstm.config.models import ComputeBackend
        kernel = create_kernel(ComputeBackend.AUTO)
        assert kernel is not None


class TestKernelEdgeCases:
    """Edge case tests for kernels."""
    
    def test_single_trace(self):
        """Test with single trace."""
        config = KernelConfig(max_aperture_m=500.0)
        kernel = NumbaKernel()
        kernel.initialize(config)
        
        traces = create_trace_block(
            amplitudes=np.random.randn(1, 100).astype(np.float32),
            source_x=np.array([0.0]), source_y=np.array([0.0]),
            receiver_x=np.array([100.0]), receiver_y=np.array([0.0]),
            sample_rate_ms=2.0,
        )
        output = create_output_tile(0, 100, 25, -50, 50, 25, 0, 100, 2)
        velocity = VelocitySlice(vrms=np.full(output.nt, 2000.0), is_1d=True)
        
        metrics = kernel.migrate_tile(traces, output, velocity)
        assert metrics.n_traces_processed == 1

    def test_traces_far_from_output(self):
        """Test when traces are far from output grid."""
        config = KernelConfig(max_aperture_m=100.0)  # Small aperture
        kernel = NumbaKernel()
        kernel.initialize(config)
        
        # Traces far away
        traces = create_trace_block(
            amplitudes=np.random.randn(10, 100).astype(np.float32),
            source_x=np.full(10, 10000.0), source_y=np.full(10, 10000.0),
            receiver_x=np.full(10, 10100.0), receiver_y=np.full(10, 10000.0),
            sample_rate_ms=2.0,
        )
        output = create_output_tile(0, 100, 25, 0, 100, 25, 0, 100, 2)
        velocity = VelocitySlice(vrms=np.full(output.nt, 2000.0), is_1d=True)
        
        metrics = kernel.migrate_tile(traces, output, velocity)
        # Should complete without error, fold should be low/zero
        assert np.sum(output.fold) == 0 or np.max(output.fold) < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
