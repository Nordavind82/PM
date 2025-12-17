"""
Integration test for full PSTM migration pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np


def test_full_migration():
    """Test complete migration pipeline with synthetic data."""
    from tests.fixtures.synthetic import create_synthetic_dataset
    from pstm.config import create_minimal_config, MigrationConfig
    from pstm.data import ZarrTraceReader, ParquetHeaderManager, SpatialIndex
    from pstm.kernels import create_kernel
    from pstm.kernels.base import KernelConfig, VelocitySlice, create_trace_block, create_output_tile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic data
        print("Creating synthetic dataset...")
        traces_path, headers_path = create_synthetic_dataset(
            output_dir=tmpdir / "data",
            n_shots=5,
            n_receivers=20,
            n_samples=500,
            sample_rate_ms=2.0,
            geometry_type="2d",
        )

        # Load data
        print("Loading data...")
        trace_reader = ZarrTraceReader(traces_path)
        trace_reader.open()

        header_manager = ParquetHeaderManager(headers_path)
        header_manager.open()

        # Build spatial index
        print("Building spatial index...")
        trace_indices, midpoint_x, midpoint_y = header_manager.get_all_midpoints()
        spatial_index = SpatialIndex.build(trace_indices, midpoint_x, midpoint_y)

        # Create output tile
        x_min = midpoint_x.min() - 200
        x_max = midpoint_x.max() + 200
        y_min = midpoint_y.min() - 200
        y_max = midpoint_y.max() + 200

        print(f"Output extent: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]")

        output_tile = create_output_tile(
            x_min=x_min, x_max=x_max, dx=50.0,
            y_min=y_min, y_max=y_max, dy=50.0,
            t_min_ms=0, t_max_ms=1000, dt_ms=2.0,
        )

        print(f"Output tile shape: {output_tile.shape}")

        # Get all traces
        print("Loading traces...")
        all_trace_data = trace_reader.get_traces(trace_indices)
        geometry = header_manager.get_geometry_for_indices(trace_indices)

        # Create trace block
        traces = create_trace_block(
            amplitudes=all_trace_data,
            source_x=geometry.source_x,
            source_y=geometry.source_y,
            receiver_x=geometry.receiver_x,
            receiver_y=geometry.receiver_y,
            sample_rate_ms=trace_reader.sample_rate_ms or 2.0,
            start_time_ms=0.0,
        )

        print(f"Traces: {traces.n_traces} traces × {traces.n_samples} samples")

        # Create velocity model (constant)
        vrms = np.full(output_tile.nt, 2000.0, dtype=np.float64)
        velocity = VelocitySlice(vrms=vrms, is_1d=True)

        # Create kernel
        print("Initializing kernel...")
        kernel = create_kernel("numba_cpu")
        kernel_config = KernelConfig(
            max_aperture_m=2000.0,
            apply_spreading=True,
            apply_obliquity=True,
        )
        kernel.initialize(kernel_config)

        # Run migration
        print("Running migration...")
        metrics = kernel.migrate_tile(traces, output_tile, velocity)

        print(f"Migration complete:")
        print(f"  - Traces processed: {metrics.n_traces_processed}")
        print(f"  - Compute time: {metrics.compute_time_s:.2f}s")
        print(f"  - Rate: {metrics.traces_per_second:.0f} traces/s")

        # Check results
        image = output_tile.image
        fold = output_tile.fold

        print(f"Output image stats:")
        print(f"  - Shape: {image.shape}")
        print(f"  - Min: {image.min():.2e}")
        print(f"  - Max: {image.max():.2e}")
        print(f"  - Non-zero: {(np.abs(image) > 1e-10).sum()}")

        print(f"Fold stats:")
        print(f"  - Max fold: {fold.max()}")
        print(f"  - Mean fold: {fold.mean():.1f}")

        # Verify we got some signal
        assert image.max() > 0, "No signal in output!"
        assert fold.max() > 0, "No fold accumulated!"

        # Find peak location (should be near diffractor)
        # The diffractor is at the center of the survey at z=1000m
        # With v=2000m/s, t0 = 2 * 1000 / 2000 = 1.0s = 1000ms
        it_peak = np.unravel_index(np.argmax(np.abs(image)), image.shape)[2]
        t_peak_ms = output_tile.t_axis_ms[it_peak]

        print(f"Peak at t={t_peak_ms:.0f}ms (expected ~1000ms for diffractor)")

        # Should be within 100ms of expected (allowing for aperture/smearing)
        assert 800 < t_peak_ms < 1200, f"Peak at unexpected time: {t_peak_ms}ms"

        print("\n✓ Integration test PASSED!")

        # Cleanup
        trace_reader.close()
        header_manager.close()
        kernel.cleanup()


if __name__ == "__main__":
    test_full_migration()
