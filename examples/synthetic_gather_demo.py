#!/usr/bin/env python3
"""
Demonstration: Synthetic Common Offset Gather Generation with Point Diffractor

This script shows how to generate synthetic prestack seismic data with:
- Single or multiple point diffractors
- Multiple offset-azimuth configurations
- Export to Zarr/Parquet or SEG-Y formats
"""

import numpy as np
from pathlib import Path

from pstm.synthetic import (
    SyntheticConfig,
    SurveyGeometry,
    TraceParameters,
    WaveletParameters,
    create_simple_synthetic,
    create_multi_diffractor_synthetic,
    generate_synthetic_gathers,
    export_to_zarr_parquet,
)


def example_1_basic_usage():
    """Basic usage with convenience function."""
    print("=" * 60)
    print("Example 1: Basic Usage - Single Diffractor")
    print("=" * 60)
    
    # Generate simple synthetic with single diffractor
    result = create_simple_synthetic(
        diffractor_x=1000.0,      # Diffractor X position (m)
        diffractor_y=1000.0,      # Diffractor Y position (m)
        diffractor_z=800.0,       # Diffractor depth (m)
        survey_extent=2000.0,     # Survey area: 2km x 2km
        grid_spacing=25.0,        # Midpoint spacing: 25m
        offsets=[500, 1000],      # Two offset values (m)
        azimuths=[0, 90],         # Two azimuths (North, East)
        velocity=2500.0,          # Constant velocity (m/s)
        n_samples=1501,           # 3 seconds @ 2ms
        dt_ms=2.0,
        wavelet_freq=25.0,        # Ricker wavelet frequency
        noise_level=0.1,          # 10% noise
    )
    
    print(f"Generated {result.n_traces} traces")
    print(f"Trace length: {result.n_samples} samples ({result.dt_ms}ms)")
    print(f"Survey: {result.config.survey.nx} x {result.config.survey.ny} midpoints")
    print(f"Offset-azimuth planes: {result.config.n_planes}")
    print(f"Max amplitude: {np.max(np.abs(result.traces)):.4f}")
    
    return result


def example_2_custom_configuration():
    """Custom configuration with full control."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    # Create configuration step by step
    config = SyntheticConfig(
        # Survey geometry
        survey=SurveyGeometry(
            x_min=0,
            x_max=3000,
            y_min=0,
            y_max=3000,
            dx=50,   # Coarser grid for faster computation
            dy=50,
        ),
        
        # Trace parameters
        trace_params=TraceParameters(
            n_samples=2001,    # 4 seconds
            dt_ms=2.0,
            t_start_ms=0.0,
        ),
        
        # Wavelet
        wavelet=WaveletParameters(
            type="ricker",
            dominant_freq_hz=30.0,
        ),
        
        # Velocity
        velocity_ms=2200.0,
        
        # Noise
        noise_level=0.05,
    )
    
    # Add single diffractor at center
    config.add_diffractor(
        x=1500,
        y=1500,
        z=1000,       # 1km depth
        amplitude=1.0
    )
    
    # Add offset-azimuth planes using offset/azimuth specification
    # This creates a star pattern of azimuths at each offset
    for offset in [300, 600, 1000, 1500]:
        for azimuth in [0, 45, 90, 135, 180, 225, 270, 315]:
            config.add_offset_azimuth_plane(offset=offset, azimuth_deg=azimuth)
    
    print(f"Configuration summary:")
    print(f"  Survey: {config.survey.nx} x {config.survey.ny} midpoints")
    print(f"  Diffractors: {len(config.diffractors)}")
    print(f"  Offset-azimuth planes: {config.n_planes}")
    print(f"  Total traces: {config.n_traces_total}")
    
    # Generate data
    result = generate_synthetic_gathers(config)
    
    print(f"\nGenerated data:")
    print(f"  Traces shape: {result.traces.shape}")
    print(f"  Memory: {result.traces.nbytes / 1e6:.1f} MB")
    
    return result


def example_3_offset_xy_specification():
    """Using offset_x/offset_y instead of offset/azimuth."""
    print("\n" + "=" * 60)
    print("Example 3: Offset X/Y Specification")
    print("=" * 60)
    
    # Use convenience function with offset_x/offset_y grid
    result = create_multi_diffractor_synthetic(
        diffractor_locations=[
            (1000, 1000, 500),   # Shallow diffractor
            (2000, 2000, 1200),  # Deeper diffractor
        ],
        survey_x_range=(0, 3000),
        survey_y_range=(0, 3000),
        grid_spacing=50,
        
        # Offset specified as (offset_x, offset_y) pairs
        # This creates a rectangular grid of offset vectors
        offset_x_values=[-600, -300, 0, 300, 600],
        offset_y_values=[-600, -300, 0, 300, 600],
        
        velocity=2000.0,
        n_samples=1501,
        dt_ms=2.0,
    )
    
    print(f"Generated {result.n_traces} traces")
    print(f"Unique offsets: {len(np.unique(result.offset))}")
    print(f"Offset range: {result.offset.min():.0f} - {result.offset.max():.0f} m")
    
    # Show offset-azimuth distribution
    print("\nOffset-Azimuth distribution:")
    for plane_id in np.unique(result.plane_id):
        mask = result.plane_id == plane_id
        off = result.offset[mask][0]
        az = result.azimuth[mask][0]
        ox = result.offset_x[mask][0]
        oy = result.offset_y[mask][0]
        print(f"  Plane {plane_id}: offset={off:.0f}m, azimuth={az:.1f}°, "
              f"offset_x={ox:.0f}m, offset_y={oy:.0f}m")
    
    return result


def example_4_export_data():
    """Export synthetic data to files."""
    print("\n" + "=" * 60)
    print("Example 4: Export to Zarr/Parquet")
    print("=" * 60)
    
    # Generate small synthetic dataset
    result = create_simple_synthetic(
        diffractor_x=500, diffractor_y=500, diffractor_z=600,
        survey_extent=1000,
        grid_spacing=50,
        offsets=[200, 400],
        azimuths=[0, 90],
        n_samples=501,
    )
    
    # Export to Zarr/Parquet
    output_dir = Path("./synthetic_output")
    traces_path, headers_path = export_to_zarr_parquet(
        result,
        output_dir,
        traces_name="diffractor_traces.zarr",
        headers_name="diffractor_headers.parquet",
    )
    
    print(f"\nExported files:")
    print(f"  Traces: {traces_path}")
    print(f"  Headers: {headers_path}")
    
    # Show header columns
    import polars as pl
    df = pl.read_parquet(headers_path)
    print(f"\nHeader columns: {df.columns}")
    print(f"\nSample headers (first 5 traces):")
    print(df.head(5))
    
    return result


def example_5_header_inspection():
    """Inspect generated headers for PSTM compatibility."""
    print("\n" + "=" * 60)
    print("Example 5: Header Inspection for PSTM")
    print("=" * 60)
    
    result = create_simple_synthetic(
        diffractor_x=500, diffractor_y=500, diffractor_z=500,
        survey_extent=1000,
        grid_spacing=100,
        offsets=[200],
        azimuths=[45],
    )
    
    print("Headers generated for PSTM:")
    headers = result.get_headers_dict()
    for key, arr in headers.items():
        print(f"  {key}: dtype={arr.dtype}, range=[{arr.min():.2f}, {arr.max():.2f}]")
    
    # Verify midpoint calculation
    print("\nMidpoint verification (first 3 traces):")
    for i in range(min(3, result.n_traces)):
        mx_calc = (result.source_x[i] + result.receiver_x[i]) / 2
        my_calc = (result.source_y[i] + result.receiver_y[i]) / 2
        print(f"  Trace {i}: midpoint=({result.midpoint_x[i]:.1f}, {result.midpoint_y[i]:.1f}), "
              f"calculated=({mx_calc:.1f}, {my_calc:.1f})")
    
    # Verify offset calculation
    print("\nOffset verification (first 3 traces):")
    for i in range(min(3, result.n_traces)):
        off_calc = np.sqrt(
            (result.receiver_x[i] - result.source_x[i])**2 +
            (result.receiver_y[i] - result.source_y[i])**2
        )
        print(f"  Trace {i}: stored offset={result.offset[i]:.1f}m, "
              f"calculated={off_calc:.1f}m")


def example_6_visualize_gather():
    """Visualize a synthetic gather (if matplotlib available)."""
    print("\n" + "=" * 60)
    print("Example 6: Visualization")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    # Generate data for single offset plane
    result = create_simple_synthetic(
        diffractor_x=500, diffractor_y=500, diffractor_z=500,
        survey_extent=1000,
        grid_spacing=25,
        offsets=[300],
        azimuths=[0],
        n_samples=751,
        dt_ms=2.0,
    )
    
    # Extract inline through diffractor
    inline_idx = result.config.survey.nx // 2
    crossline_traces = result.inline == inline_idx
    
    # Get traces and sort by crossline
    trace_data = result.traces[crossline_traces]
    crosslines = result.crossline[crossline_traces]
    sort_idx = np.argsort(crosslines)
    trace_data = trace_data[sort_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    t_axis = result.config.trace_params.time_axis_ms
    extent = [0, trace_data.shape[0], t_axis[-1], t_axis[0]]
    
    clip = np.percentile(np.abs(trace_data), 99)
    ax.imshow(
        trace_data.T,
        aspect='auto',
        extent=extent,
        cmap='seismic',
        vmin=-clip,
        vmax=clip,
    )
    
    ax.set_xlabel('Trace number (crossline)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Synthetic Gather - Inline {inline_idx}\n'
                 f'Offset={result.offset[0]:.0f}m, Azimuth={result.azimuth[0]:.0f}°')
    
    plt.tight_layout()
    plt.savefig('./synthetic_output/gather_example.png', dpi=150)
    print(f"Saved visualization to ./synthetic_output/gather_example.png")
    plt.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SYNTHETIC COMMON OFFSET GATHER GENERATOR")
    print("Point Diffractor Response for PSTM Testing")
    print("=" * 60)
    
    # Run examples
    example_1_basic_usage()
    example_2_custom_configuration()
    example_3_offset_xy_specification()
    example_4_export_data()
    example_5_header_inspection()
    example_6_visualize_gather()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
