#!/usr/bin/env python3
"""
PSTM Diagnostic Test Suite - Offset Bin 10

Step-by-step diagnostic tests to trace internal mechanics and visualize
QC results at every stage of the migration pipeline.

Tests:
1. Input Data Validation
2. Coordinate Transformation & Scalar Application
3. Spatial Index & Trace Selection
4. DSR Traveltime Calculation
5. Sample Interpolation Accuracy
6. Velocity Sampling
7. Aperture & Taper Weights
8. Single-Trace Migration Test
9. Multi-Trace Coherency Test
10. Full Tile Migration Comparison

Usage:
    python diagnostic_pstm_bin10.py --test all
    python diagnostic_pstm_bin10.py --test 1,2,3
    python diagnostic_pstm_bin10.py --test dsr
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import polars as pl
import zarr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add pstm to path
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# Configuration
# =============================================================================

BIN_NUM = 10
COMMON_OFFSET_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")
VELOCITY_PATH = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/velocity_pstm_ilxl.zarr")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/diagnostic_qc")
MIGRATION_OUTPUT = Path(f"/Users/olegadamovich/SeismicData/PSTM_common_offset/migration_bin_{BIN_NUM:02d}")

# Grid parameters
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),  # Origin (IL=1, XL=1)
    'c2': (627094.02, 5106803.16),  # Inline end (IL=511, XL=1)
    'c3': (631143.35, 5110261.43),  # Far corner (IL=511, XL=427)
    'c4': (622862.92, 5119956.77),  # Crossline end (IL=1, XL=427)
}
NX, NY, NT = 511, 427, 1001
DX, DY, DT_MS = 25.0, 12.5, 2.0

# Test point for detailed diagnostics (IL=256, XL=214 - center of grid)
TEST_IL, TEST_XL = 256, 214
TEST_IX, TEST_IY = TEST_IL - 1, TEST_XL - 1  # 0-based indices


@dataclass
class DiagnosticResult:
    """Result of a diagnostic test."""
    name: str
    passed: bool
    message: str
    figures: list[Path] = None
    data: dict = None


# =============================================================================
# Test 1: Input Data Validation
# =============================================================================

def test_1_input_data_validation() -> DiagnosticResult:
    """Validate input trace data and headers."""
    print("\n" + "="*60)
    print("TEST 1: Input Data Validation")
    print("="*60)

    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{BIN_NUM:02d}"
    traces_path = bin_dir / "traces.zarr"
    headers_path = bin_dir / "headers.parquet"

    issues = []
    data = {}

    # Check traces
    print(f"\n[1.1] Loading traces from {traces_path}")
    try:
        traces_store = zarr.open_array(traces_path, mode='r')
        traces_shape = traces_store.shape
        traces_attrs = dict(traces_store.attrs)
        print(f"      Shape: {traces_shape}")
        print(f"      Attrs: {traces_attrs}")

        # Check for transposed storage
        if traces_shape[0] < traces_shape[1]:
            print(f"      WARNING: Data appears transposed (n_samples, n_traces)")
            data['transposed'] = True
            n_samples, n_traces = traces_shape
        else:
            data['transposed'] = False
            n_traces, n_samples = traces_shape

        data['n_traces'] = n_traces
        data['n_samples'] = n_samples

        # Sample some traces for statistics
        sample_indices = np.linspace(0, n_traces-1, min(1000, n_traces), dtype=int)
        if data['transposed']:
            sample_traces = np.asarray(traces_store[:, sample_indices]).T
        else:
            sample_traces = np.asarray(traces_store[sample_indices, :])

        print(f"\n[1.2] Trace statistics (from {len(sample_indices)} samples):")
        print(f"      Min: {sample_traces.min():.6f}")
        print(f"      Max: {sample_traces.max():.6f}")
        print(f"      Mean: {sample_traces.mean():.6f}")
        print(f"      Std: {sample_traces.std():.6f}")
        print(f"      RMS: {np.sqrt(np.mean(sample_traces**2)):.6f}")
        print(f"      Non-zero: {np.count_nonzero(sample_traces)} / {sample_traces.size}")

        data['trace_stats'] = {
            'min': float(sample_traces.min()),
            'max': float(sample_traces.max()),
            'rms': float(np.sqrt(np.mean(sample_traces**2))),
        }

        if sample_traces.max() == 0 and sample_traces.min() == 0:
            issues.append("All sampled traces are zero!")

    except Exception as e:
        issues.append(f"Failed to load traces: {e}")
        return DiagnosticResult("Input Data Validation", False, str(issues), data=data)

    # Check headers
    print(f"\n[1.3] Loading headers from {headers_path}")
    try:
        df = pl.read_parquet(headers_path)
        print(f"      N headers: {len(df)}")
        print(f"      Columns: {df.columns}")

        # Validate required columns
        required_cols = ['source_x', 'source_y', 'receiver_x', 'receiver_y',
                        'offset', 'scalar_coord', 'bin_trace_idx']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

        # Check trace index alignment
        max_trace_idx = df['bin_trace_idx'].max()
        print(f"      Max trace index: {max_trace_idx}")
        if max_trace_idx >= n_traces:
            issues.append(f"Header trace indices exceed trace count: {max_trace_idx} >= {n_traces}")

        data['headers_df'] = df

    except Exception as e:
        issues.append(f"Failed to load headers: {e}")

    # Create QC figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Test 1: Input Data Validation - Bin {BIN_NUM}', fontsize=14, fontweight='bold')

    # Plot sample traces
    ax = axes[0, 0]
    t_axis = np.arange(n_samples) * DT_MS
    for i in range(min(10, len(sample_traces))):
        ax.plot(t_axis, sample_traces[i] + i*0.1, 'k-', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trace (offset for visibility)')
    ax.set_title('Sample Traces (first 10)')

    # Plot trace amplitude histogram
    ax = axes[0, 1]
    ax.hist(sample_traces.flatten(), bins=100, density=True, alpha=0.7)
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Density')
    ax.set_title('Amplitude Distribution')
    ax.set_yscale('log')

    # Plot offset distribution
    if 'headers_df' in data:
        ax = axes[1, 0]
        offsets = data['headers_df']['offset'].to_numpy()
        ax.hist(offsets, bins=50, alpha=0.7)
        ax.set_xlabel('Offset (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'Offset Distribution (mean={offsets.mean():.0f}m)')

        # Plot azimuth distribution
        ax = axes[1, 1]
        if 'sr_azim' in data['headers_df'].columns:
            azimuths = data['headers_df']['sr_azim'].to_numpy()
            ax.hist(azimuths, bins=36, alpha=0.7)
            ax.set_xlabel('Azimuth (degrees)')
            ax.set_ylabel('Count')
            ax.set_title('Azimuth Distribution')
        else:
            ax.text(0.5, 0.5, 'No azimuth column', ha='center', va='center',
                   transform=ax.transAxes)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "test_1_input_validation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n      Saved: {fig_path}")

    passed = len(issues) == 0
    return DiagnosticResult(
        "Input Data Validation",
        passed,
        "PASSED" if passed else f"FAILED: {issues}",
        figures=[fig_path],
        data=data
    )


# =============================================================================
# Test 2: Coordinate Transformation & Scalar Application
# =============================================================================

def test_2_coordinate_transformation(prev_data: dict = None) -> DiagnosticResult:
    """Test coordinate scalar application and transformation."""
    print("\n" + "="*60)
    print("TEST 2: Coordinate Transformation & Scalar Application")
    print("="*60)

    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{BIN_NUM:02d}"
    headers_path = bin_dir / "headers.parquet"

    df = pl.read_parquet(headers_path)
    issues = []
    data = {}

    # Get raw coordinates
    raw_sx = df['source_x'].to_numpy().astype(np.float64)
    raw_sy = df['source_y'].to_numpy().astype(np.float64)
    raw_rx = df['receiver_x'].to_numpy().astype(np.float64)
    raw_ry = df['receiver_y'].to_numpy().astype(np.float64)

    # Get scalar
    scalar = int(df['scalar_coord'][0])
    print(f"\n[2.1] Coordinate scalar: {scalar}")

    # Apply scalar (SEG-Y convention: negative = divide)
    if scalar < 0:
        scale_factor = 1.0 / abs(scalar)
    elif scalar > 0:
        scale_factor = float(scalar)
    else:
        scale_factor = 1.0

    scaled_sx = raw_sx * scale_factor
    scaled_sy = raw_sy * scale_factor
    scaled_rx = raw_rx * scale_factor
    scaled_ry = raw_ry * scale_factor

    print(f"      Scale factor: {scale_factor}")
    print(f"\n[2.2] Raw coordinates:")
    print(f"      Source X: {raw_sx.min():.0f} - {raw_sx.max():.0f}")
    print(f"      Source Y: {raw_sy.min():.0f} - {raw_sy.max():.0f}")
    print(f"\n[2.3] Scaled coordinates (meters):")
    print(f"      Source X: {scaled_sx.min():.1f} - {scaled_sx.max():.1f}")
    print(f"      Source Y: {scaled_sy.min():.1f} - {scaled_sy.max():.1f}")

    # Compute offset from scaled coordinates
    computed_offset = np.sqrt((scaled_rx - scaled_sx)**2 + (scaled_ry - scaled_sy)**2)
    header_offset = df['offset'].to_numpy()

    print(f"\n[2.4] Offset comparison:")
    print(f"      Header offset: {header_offset.min():.1f} - {header_offset.max():.1f}")
    print(f"      Computed offset: {computed_offset.min():.1f} - {computed_offset.max():.1f}")

    offset_ratio = computed_offset / (header_offset + 1e-10)
    print(f"      Ratio (computed/header): {offset_ratio.mean():.4f} +/- {offset_ratio.std():.4f}")

    if abs(offset_ratio.mean() - 1.0) > 0.01:
        issues.append(f"Offset mismatch: ratio = {offset_ratio.mean():.4f}")

    # Compute midpoints
    midpoint_x = (scaled_sx + scaled_rx) / 2.0
    midpoint_y = (scaled_sy + scaled_ry) / 2.0

    print(f"\n[2.5] Midpoint range (meters):")
    print(f"      X: {midpoint_x.min():.1f} - {midpoint_x.max():.1f}")
    print(f"      Y: {midpoint_y.min():.1f} - {midpoint_y.max():.1f}")

    # Compare with output grid extent
    grid_x_min, grid_y_min = GRID_CORNERS['c1']
    grid_x_max = max(c[0] for c in GRID_CORNERS.values())
    grid_y_max = max(c[1] for c in GRID_CORNERS.values())

    print(f"\n[2.6] Output grid extent:")
    print(f"      X: {grid_x_min:.1f} - {grid_x_max:.1f}")
    print(f"      Y: {grid_y_min:.1f} - {grid_y_max:.1f}")

    # Check coverage
    midpoints_in_grid = (
        (midpoint_x >= grid_x_min) & (midpoint_x <= grid_x_max) &
        (midpoint_y >= grid_y_min) & (midpoint_y <= grid_y_max)
    )
    coverage_pct = 100 * midpoints_in_grid.sum() / len(midpoint_x)
    print(f"\n[2.7] Midpoints within grid: {coverage_pct:.1f}%")

    if coverage_pct < 50:
        issues.append(f"Low midpoint coverage: {coverage_pct:.1f}%")

    data['scaled_coords'] = {
        'source_x': scaled_sx,
        'source_y': scaled_sy,
        'receiver_x': scaled_rx,
        'receiver_y': scaled_ry,
        'midpoint_x': midpoint_x,
        'midpoint_y': midpoint_y,
        'offset': computed_offset,
    }

    # Create QC figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Test 2: Coordinate Transformation - Bin {BIN_NUM}', fontsize=14, fontweight='bold')

    # Plot source/receiver positions
    ax = axes[0, 0]
    sample_idx = np.random.choice(len(scaled_sx), min(5000, len(scaled_sx)), replace=False)
    ax.scatter(scaled_sx[sample_idx], scaled_sy[sample_idx], s=1, alpha=0.3, label='Sources')
    ax.scatter(scaled_rx[sample_idx], scaled_ry[sample_idx], s=1, alpha=0.3, label='Receivers')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Source/Receiver Positions (sample)')
    ax.legend()
    ax.axis('equal')

    # Plot midpoints with grid overlay
    ax = axes[0, 1]
    ax.scatter(midpoint_x[sample_idx], midpoint_y[sample_idx], s=1, alpha=0.3, c='blue', label='Midpoints')
    # Plot grid corners
    corners = np.array([GRID_CORNERS['c1'], GRID_CORNERS['c2'], GRID_CORNERS['c3'],
                       GRID_CORNERS['c4'], GRID_CORNERS['c1']])
    ax.plot(corners[:, 0], corners[:, 1], 'r-', linewidth=2, label='Output Grid')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Midpoints vs Output Grid')
    ax.legend()
    ax.axis('equal')

    # Offset comparison
    ax = axes[1, 0]
    ax.scatter(header_offset, computed_offset, s=1, alpha=0.1)
    ax.plot([header_offset.min(), header_offset.max()],
            [header_offset.min(), header_offset.max()], 'r--', label='1:1 line')
    ax.set_xlabel('Header Offset (m)')
    ax.set_ylabel('Computed Offset (m)')
    ax.set_title('Offset Comparison')
    ax.legend()

    # Offset ratio histogram
    ax = axes[1, 1]
    ax.hist(offset_ratio, bins=100, density=True)
    ax.axvline(1.0, color='r', linestyle='--', label='Expected (1.0)')
    ax.set_xlabel('Computed / Header Offset Ratio')
    ax.set_ylabel('Density')
    ax.set_title(f'Offset Ratio Distribution (mean={offset_ratio.mean():.4f})')
    ax.legend()

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "test_2_coordinate_transformation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n      Saved: {fig_path}")

    passed = len(issues) == 0
    return DiagnosticResult(
        "Coordinate Transformation",
        passed,
        "PASSED" if passed else f"FAILED: {issues}",
        figures=[fig_path],
        data=data
    )


# =============================================================================
# Test 3: Spatial Index & Trace Selection
# =============================================================================

def test_3_spatial_index(prev_data: dict = None) -> DiagnosticResult:
    """Test spatial index construction and trace selection."""
    print("\n" + "="*60)
    print("TEST 3: Spatial Index & Trace Selection")
    print("="*60)

    from pstm.data.spatial_index import SpatialIndex

    # Get coordinates from previous test or reload
    if prev_data and 'scaled_coords' in prev_data:
        coords = prev_data['scaled_coords']
        midpoint_x = coords['midpoint_x']
        midpoint_y = coords['midpoint_y']
    else:
        bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{BIN_NUM:02d}"
        df = pl.read_parquet(bin_dir / "headers.parquet")
        scalar = int(df['scalar_coord'][0])
        scale_factor = 1.0 / abs(scalar) if scalar < 0 else float(scalar) if scalar > 0 else 1.0

        sx = df['source_x'].to_numpy().astype(np.float64) * scale_factor
        sy = df['source_y'].to_numpy().astype(np.float64) * scale_factor
        rx = df['receiver_x'].to_numpy().astype(np.float64) * scale_factor
        ry = df['receiver_y'].to_numpy().astype(np.float64) * scale_factor
        midpoint_x = (sx + rx) / 2.0
        midpoint_y = (sy + ry) / 2.0

    trace_indices = np.arange(len(midpoint_x), dtype=np.int64)

    issues = []
    data = {}

    # Build spatial index
    print(f"\n[3.1] Building spatial index for {len(trace_indices):,} traces...")
    spatial_index = SpatialIndex.build(trace_indices, midpoint_x, midpoint_y)
    print(f"      Index built: {spatial_index.n_points:,} points")

    # Test query at grid center
    # Compute grid center from corners
    center_x = np.mean([c[0] for c in GRID_CORNERS.values()])
    center_y = np.mean([c[1] for c in GRID_CORNERS.values()])

    print(f"\n[3.2] Test query at grid center ({center_x:.1f}, {center_y:.1f}):")

    apertures = [500, 1000, 1500, 2000]
    for aperture in apertures:
        selected = spatial_index.query_radius(center_x, center_y, aperture)
        print(f"      Aperture {aperture}m: {len(selected):,} traces")
        data[f'aperture_{aperture}'] = len(selected)

    # Test query at multiple points across grid
    print(f"\n[3.3] Query coverage across grid (aperture=2000m):")
    test_points_il = [50, 150, 256, 350, 450]
    test_points_xl = [50, 150, 214, 300, 380]

    # Compute coordinates for test points using rotated grid
    c1 = np.array(GRID_CORNERS['c1'])
    c2 = np.array(GRID_CORNERS['c2'])
    c4 = np.array(GRID_CORNERS['c4'])

    # Direction vectors
    il_dir = (c2 - c1) / (NX - 1)
    xl_dir = (c4 - c1) / (NY - 1)

    query_results = []
    for il in test_points_il:
        for xl in test_points_xl:
            pt = c1 + (il - 1) * il_dir + (xl - 1) * xl_dir
            selected = spatial_index.query_radius(pt[0], pt[1], 2000)
            query_results.append({
                'il': il, 'xl': xl, 'x': pt[0], 'y': pt[1], 'n_traces': len(selected)
            })

    # Print grid of results
    print("      IL\\XL  ", end="")
    for xl in test_points_xl:
        print(f"{xl:>8}", end="")
    print()

    for il in test_points_il:
        print(f"      {il:>4}  ", end="")
        for xl in test_points_xl:
            result = [r for r in query_results if r['il'] == il and r['xl'] == xl][0]
            print(f"{result['n_traces']:>8}", end="")
        print()

    data['query_results'] = query_results

    # Check for sparse regions
    min_traces = min(r['n_traces'] for r in query_results)
    if min_traces < 100:
        issues.append(f"Sparse region detected: min traces = {min_traces}")

    # Create QC figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    fig.suptitle(f'Test 3: Spatial Index & Trace Selection - Bin {BIN_NUM}',
                fontsize=14, fontweight='bold')

    # Plot midpoints and query points
    ax = fig.add_subplot(gs[0, 0])
    sample_idx = np.random.choice(len(midpoint_x), min(10000, len(midpoint_x)), replace=False)
    ax.scatter(midpoint_x[sample_idx], midpoint_y[sample_idx], s=1, alpha=0.2, c='blue')

    # Add query points
    for r in query_results:
        color = 'green' if r['n_traces'] > 1000 else 'orange' if r['n_traces'] > 100 else 'red'
        ax.scatter(r['x'], r['y'], s=50, c=color, edgecolors='black', zorder=5)

    # Plot grid outline
    corners = np.array([GRID_CORNERS['c1'], GRID_CORNERS['c2'], GRID_CORNERS['c3'],
                       GRID_CORNERS['c4'], GRID_CORNERS['c1']])
    ax.plot(corners[:, 0], corners[:, 1], 'r-', linewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Midpoints and Query Points')
    ax.axis('equal')

    # Plot trace count heatmap
    ax = fig.add_subplot(gs[0, 1])
    # Create denser grid for heatmap
    heatmap_il = np.linspace(1, NX, 20, dtype=int)
    heatmap_xl = np.linspace(1, NY, 20, dtype=int)
    heatmap_counts = np.zeros((len(heatmap_il), len(heatmap_xl)))

    for i, il in enumerate(heatmap_il):
        for j, xl in enumerate(heatmap_xl):
            pt = c1 + (il - 1) * il_dir + (xl - 1) * xl_dir
            selected = spatial_index.query_radius(pt[0], pt[1], 2000)
            heatmap_counts[i, j] = len(selected)

    im = ax.imshow(heatmap_counts.T, origin='lower', aspect='auto',
                   extent=[1, NX, 1, NY], cmap='viridis')
    plt.colorbar(im, ax=ax, label='Trace Count')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title('Trace Count per Location (aperture=2000m)')

    # Plot aperture vs trace count
    ax = fig.add_subplot(gs[1, 0])
    aperture_test = np.linspace(100, 3000, 30)
    counts = []
    for ap in aperture_test:
        selected = spatial_index.query_radius(center_x, center_y, ap)
        counts.append(len(selected))
    ax.plot(aperture_test, counts, 'b-o')
    ax.set_xlabel('Aperture (m)')
    ax.set_ylabel('Trace Count')
    ax.set_title('Aperture vs Trace Count at Grid Center')
    ax.grid(True, alpha=0.3)

    # Plot azimuth distribution of selected traces
    ax = fig.add_subplot(gs[1, 1])
    selected = spatial_index.query_radius(center_x, center_y, 2000)
    if prev_data and 'scaled_coords' in prev_data:
        coords = prev_data['scaled_coords']
        dx = coords['receiver_x'][selected] - coords['source_x'][selected]
        dy = coords['receiver_y'][selected] - coords['source_y'][selected]
        az = np.degrees(np.arctan2(dx, dy)) % 360
        ax.hist(az, bins=36, alpha=0.7)
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Count')
        ax.set_title(f'Azimuth Distribution of Selected Traces (n={len(selected):,})')
    else:
        ax.text(0.5, 0.5, 'Need prev_data for azimuth', ha='center', va='center',
               transform=ax.transAxes)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "test_3_spatial_index.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n      Saved: {fig_path}")

    passed = len(issues) == 0
    return DiagnosticResult(
        "Spatial Index",
        passed,
        "PASSED" if passed else f"FAILED: {issues}",
        figures=[fig_path],
        data=data
    )


# =============================================================================
# Test 4: DSR Traveltime Calculation
# =============================================================================

def test_4_dsr_traveltime(prev_data: dict = None) -> DiagnosticResult:
    """Test Double Square Root traveltime calculation."""
    print("\n" + "="*60)
    print("TEST 4: DSR Traveltime Calculation")
    print("="*60)

    issues = []
    data = {}

    # Load velocity
    print(f"\n[4.1] Loading velocity model from {VELOCITY_PATH}")
    vel_store = zarr.open(VELOCITY_PATH, mode='r')
    velocity = np.asarray(vel_store)
    print(f"      Shape: {velocity.shape}")
    print(f"      Range: {velocity.min():.0f} - {velocity.max():.0f} m/s")

    # Get velocity at test point
    vel_at_test = velocity[TEST_IX, TEST_IY, :]
    t_axis = np.arange(NT) * DT_MS

    print(f"\n[4.2] Velocity at test point (IL={TEST_IL}, XL={TEST_XL}):")
    print(f"      At t=500ms: {vel_at_test[250]:.0f} m/s")
    print(f"      At t=1000ms: {vel_at_test[500]:.0f} m/s")
    print(f"      At t=1500ms: {vel_at_test[750]:.0f} m/s")

    # Compute output point coordinates
    c1 = np.array(GRID_CORNERS['c1'])
    c2 = np.array(GRID_CORNERS['c2'])
    c4 = np.array(GRID_CORNERS['c4'])
    il_dir = (c2 - c1) / (NX - 1)
    xl_dir = (c4 - c1) / (NY - 1)

    output_pt = c1 + TEST_IX * il_dir + TEST_IY * xl_dir
    ox, oy = output_pt
    print(f"\n[4.3] Output point coordinates: ({ox:.1f}, {oy:.1f})")

    # Create synthetic test traces with known geometry
    print(f"\n[4.4] Testing DSR formula with synthetic geometry:")

    # Test case: Source and receiver symmetric about output point (zero offset)
    offset_test = 500  # 500m offset
    azimuth_test = 45  # 45 degrees

    # Source and receiver positions
    dx = offset_test / 2 * np.sin(np.radians(azimuth_test))
    dy = offset_test / 2 * np.cos(np.radians(azimuth_test))

    sx, sy = ox - dx, oy - dy
    rx, ry = ox + dx, oy + dy

    print(f"      Source: ({sx:.1f}, {sy:.1f})")
    print(f"      Receiver: ({rx:.1f}, {ry:.1f})")
    print(f"      Offset: {offset_test}m, Azimuth: {azimuth_test}deg")

    # Compute DSR traveltimes for different output times
    t0_values = np.array([0.5, 0.75, 1.0, 1.25, 1.5])  # seconds

    print(f"\n      DSR Traveltimes:")
    print(f"      {'t0 (s)':<10} {'V (m/s)':<10} {'t_travel (s)':<12} {'Expected NMO':<12}")

    dsr_results = []
    for t0 in t0_values:
        t_idx = int(t0 * 1000 / DT_MS)
        v = vel_at_test[t_idx]

        # DSR formula: t = sqrt((t0/2)^2 + ds^2/v^2) + sqrt((t0/2)^2 + dr^2/v^2)
        ds2 = (ox - sx)**2 + (oy - sy)**2
        dr2 = (ox - rx)**2 + (oy - ry)**2
        t0_half_sq = (t0 / 2)**2
        inv_v_sq = 1 / (v * v)

        t_travel = np.sqrt(t0_half_sq + ds2 * inv_v_sq) + np.sqrt(t0_half_sq + dr2 * inv_v_sq)

        # NMO formula for comparison: t = sqrt(t0^2 + (offset/v)^2)
        t_nmo = np.sqrt(t0**2 + (offset_test / v)**2)

        print(f"      {t0:<10.2f} {v:<10.0f} {t_travel:<12.4f} {t_nmo:<12.4f}")
        dsr_results.append({'t0': t0, 'v': v, 't_travel': t_travel, 't_nmo': t_nmo})

    data['dsr_results'] = dsr_results

    # Test DSR for different azimuths (should give same result for isotropic velocity)
    print(f"\n[4.5] Testing azimuth invariance (isotropic velocity):")
    t0 = 1.0  # 1 second
    v = vel_at_test[500]
    azimuths = np.arange(0, 360, 45)

    print(f"      Azimuth (deg)    t_travel (s)")
    azimuth_results = []
    for az in azimuths:
        dx = offset_test / 2 * np.sin(np.radians(az))
        dy = offset_test / 2 * np.cos(np.radians(az))
        sx, sy = ox - dx, oy - dy
        rx, ry = ox + dx, oy + dy

        ds2 = (ox - sx)**2 + (oy - sy)**2
        dr2 = (ox - rx)**2 + (oy - ry)**2
        t0_half_sq = (t0 / 2)**2
        inv_v_sq = 1 / (v * v)

        t_travel = np.sqrt(t0_half_sq + ds2 * inv_v_sq) + np.sqrt(t0_half_sq + dr2 * inv_v_sq)
        print(f"      {az:>12.0f}    {t_travel:.6f}")
        azimuth_results.append({'azimuth': az, 't_travel': t_travel})

    # Check azimuth invariance
    t_travels = [r['t_travel'] for r in azimuth_results]
    if max(t_travels) - min(t_travels) > 1e-10:
        issues.append("DSR not azimuth-invariant for symmetric offset!")

    data['azimuth_results'] = azimuth_results

    # Create QC figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Test 4: DSR Traveltime Calculation - Bin {BIN_NUM}',
                fontsize=14, fontweight='bold')

    # Plot velocity profile at test point
    ax = axes[0, 0]
    ax.plot(t_axis, vel_at_test, 'b-')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'Velocity Profile at IL={TEST_IL}, XL={TEST_XL}')
    ax.grid(True, alpha=0.3)

    # Plot DSR vs NMO comparison
    ax = axes[0, 1]
    t0_arr = np.array([r['t0'] for r in dsr_results])
    t_travel = np.array([r['t_travel'] for r in dsr_results])
    t_nmo = np.array([r['t_nmo'] for r in dsr_results])
    ax.plot(t0_arr * 1000, t_travel * 1000, 'b-o', label='DSR')
    ax.plot(t0_arr * 1000, t_nmo * 1000, 'r--s', label='NMO')
    ax.plot(t0_arr * 1000, t0_arr * 1000, 'k:', label='t0 (zero offset)')
    ax.set_xlabel('t0 (ms)')
    ax.set_ylabel('Travel time (ms)')
    ax.set_title(f'DSR vs NMO (offset={offset_test}m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot t_travel - t0 (moveout)
    ax = axes[1, 0]
    moveout_dsr = (t_travel - t0_arr) * 1000
    moveout_nmo = (t_nmo - t0_arr) * 1000
    ax.plot(t0_arr * 1000, moveout_dsr, 'b-o', label='DSR')
    ax.plot(t0_arr * 1000, moveout_nmo, 'r--s', label='NMO')
    ax.set_xlabel('t0 (ms)')
    ax.set_ylabel('Moveout (ms)')
    ax.set_title('Moveout Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot azimuth invariance
    ax = axes[1, 1]
    az_arr = np.array([r['azimuth'] for r in azimuth_results])
    t_arr = np.array([r['t_travel'] for r in azimuth_results])
    ax.plot(az_arr, t_arr * 1000, 'b-o')
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Travel time (ms)')
    ax.set_title(f'Azimuth Invariance Test (offset={offset_test}m, t0=1000ms)')
    ax.set_ylim([t_arr.min() * 1000 - 0.1, t_arr.max() * 1000 + 0.1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "test_4_dsr_traveltime.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n      Saved: {fig_path}")

    passed = len(issues) == 0
    return DiagnosticResult(
        "DSR Traveltime",
        passed,
        "PASSED" if passed else f"FAILED: {issues}",
        figures=[fig_path],
        data=data
    )


# =============================================================================
# Test 5: Sample Interpolation
# =============================================================================

def test_5_sample_interpolation(prev_data: dict = None) -> DiagnosticResult:
    """Test sample interpolation accuracy."""
    print("\n" + "="*60)
    print("TEST 5: Sample Interpolation Accuracy")
    print("="*60)

    issues = []
    data = {}

    # Create synthetic trace with known frequency content
    print(f"\n[5.1] Creating synthetic test traces:")
    n_samples = 1001
    dt_s = DT_MS / 1000.0
    t_axis = np.arange(n_samples) * dt_s

    # Ricker wavelet at different times
    def ricker_wavelet(t, t_peak, f_peak):
        tau = t - t_peak
        return (1 - 2 * (np.pi * f_peak * tau)**2) * np.exp(-(np.pi * f_peak * tau)**2)

    # Create trace with wavelets at specific times
    trace = np.zeros(n_samples, dtype=np.float32)
    wavelet_times = [0.5, 0.75, 1.0, 1.25, 1.5]  # seconds
    f_peak = 30  # Hz

    for t_peak in wavelet_times:
        trace += ricker_wavelet(t_axis, t_peak, f_peak)

    print(f"      Sample rate: {DT_MS} ms")
    print(f"      Wavelet frequency: {f_peak} Hz")
    print(f"      Wavelet times: {wavelet_times} s")

    # Test interpolation at fractional sample positions
    print(f"\n[5.2] Testing linear interpolation:")

    # Sample index for t=1.0s
    t_target = 1.0  # seconds
    sample_exact = t_target / dt_s
    print(f"      Target time: {t_target}s -> sample index: {sample_exact}")

    # Test offsets from exact sample
    offsets = np.array([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

    print(f"\n      Sample offset    Interp value    Error vs exact")
    interp_results = []
    for offset in offsets:
        sample_idx = sample_exact + offset
        idx0 = int(sample_idx)
        frac = sample_idx - idx0

        # Linear interpolation
        interp_value = trace[idx0] * (1 - frac) + trace[idx0 + 1] * frac
        exact_value = trace[int(sample_exact)]

        print(f"      {offset:+.2f}            {interp_value:+.6f}    {(interp_value - exact_value):+.6f}")
        interp_results.append({
            'offset': offset,
            'sample_idx': sample_idx,
            'interp_value': interp_value,
            'exact_value': exact_value
        })

    data['interp_results'] = interp_results

    # Test sample index precision
    print(f"\n[5.3] Sample index precision test:")

    # Simulate DSR calculation with float32 (as in Metal kernel)
    t_travel_f64 = 1.0  # double precision
    t_travel_f32 = np.float32(t_travel_f64)  # single precision

    sample_idx_f64 = (t_travel_f64 * 1000.0 - 0.0) / DT_MS
    sample_idx_f32 = (t_travel_f32 * 1000.0 - 0.0) / DT_MS

    print(f"      Float64 sample index: {sample_idx_f64}")
    print(f"      Float32 sample index: {sample_idx_f32}")
    print(f"      Difference: {abs(sample_idx_f64 - sample_idx_f32):.10f} samples")

    # Test edge cases
    print(f"\n[5.4] Edge case tests:")
    edge_cases = [
        (0.0, "First sample"),
        (0.001, "Near first sample"),
        ((n_samples - 1) * dt_s - 0.001, "Near last sample"),
        ((n_samples - 1) * dt_s, "Last sample"),
    ]

    for t_test, desc in edge_cases:
        sample_idx = (t_test * 1000.0) / DT_MS
        in_bounds = 0 <= sample_idx < n_samples - 1
        print(f"      {desc}: t={t_test:.4f}s, idx={sample_idx:.4f}, in_bounds={in_bounds}")

    # Create QC figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Test 5: Sample Interpolation Accuracy', fontsize=14, fontweight='bold')

    # Plot synthetic trace
    ax = axes[0, 0]
    ax.plot(t_axis * 1000, trace, 'b-', linewidth=0.5)
    for t_peak in wavelet_times:
        ax.axvline(t_peak * 1000, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Synthetic Trace with Ricker Wavelets')
    ax.set_xlim([400, 1600])

    # Zoom on one wavelet
    ax = axes[0, 1]
    zoom_start, zoom_end = 950, 1050
    zoom_mask = (t_axis * 1000 >= zoom_start) & (t_axis * 1000 <= zoom_end)
    ax.plot(t_axis[zoom_mask] * 1000, trace[zoom_mask], 'b-o', markersize=3)
    ax.axvline(1000, color='r', linestyle='--', alpha=0.5, label='t=1000ms')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Zoom on t=1000ms Wavelet')
    ax.legend()

    # Plot interpolation error
    ax = axes[1, 0]
    offs = np.array([r['offset'] for r in interp_results])
    vals = np.array([r['interp_value'] for r in interp_results])
    exact = interp_results[2]['exact_value']  # offset=0
    ax.plot(offs, vals, 'b-o')
    ax.axhline(exact, color='r', linestyle='--', label='Exact')
    ax.set_xlabel('Sample Offset')
    ax.set_ylabel('Interpolated Value')
    ax.set_title('Linear Interpolation Test')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot frequency response of interpolation
    ax = axes[1, 1]
    # Create high-resolution version for comparison
    t_fine = np.linspace(0, t_axis[-1], 10001)
    trace_fine = np.zeros_like(t_fine)
    for t_peak in wavelet_times:
        trace_fine += ricker_wavelet(t_fine, t_peak, f_peak)

    # Interpolate the sampled trace
    from scipy.interpolate import interp1d
    interp_func = interp1d(t_axis, trace, kind='linear', bounds_error=False, fill_value=0)
    trace_interp = interp_func(t_fine)

    # Compare around one wavelet
    zoom_mask_fine = (t_fine * 1000 >= 950) & (t_fine * 1000 <= 1050)
    ax.plot(t_fine[zoom_mask_fine] * 1000, trace_fine[zoom_mask_fine], 'b-', label='Continuous', linewidth=2)
    ax.plot(t_fine[zoom_mask_fine] * 1000, trace_interp[zoom_mask_fine], 'r--', label='Linear interp')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Interpolation vs Continuous')
    ax.legend()

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "test_5_interpolation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n      Saved: {fig_path}")

    passed = len(issues) == 0
    return DiagnosticResult(
        "Sample Interpolation",
        passed,
        "PASSED" if passed else f"FAILED: {issues}",
        figures=[fig_path],
        data=data
    )


# =============================================================================
# Test 6: Velocity Sampling
# =============================================================================

def test_6_velocity_sampling(prev_data: dict = None) -> DiagnosticResult:
    """Test velocity sampling and lateral variation impact."""
    print("\n" + "="*60)
    print("TEST 6: Velocity Sampling & Lateral Variation")
    print("="*60)

    issues = []
    data = {}

    # Load velocity
    vel_store = zarr.open(VELOCITY_PATH, mode='r')
    velocity = np.asarray(vel_store)

    # Analyze velocity at different times
    test_times_ms = [500, 750, 1000, 1250, 1500]

    print(f"\n[6.1] Lateral velocity variation:")
    print(f"      Time (ms)    Mean V    Std V    Range V    Var %")

    variation_data = []
    for t_ms in test_times_ms:
        t_idx = int(t_ms / DT_MS)
        v_slice = velocity[:, :, t_idx]
        mean_v = v_slice.mean()
        std_v = v_slice.std()
        range_v = v_slice.max() - v_slice.min()
        var_pct = 100 * range_v / mean_v

        print(f"      {t_ms:>8}    {mean_v:>6.0f}    {std_v:>5.0f}    {range_v:>7.0f}    {var_pct:>5.1f}%")
        variation_data.append({
            't_ms': t_ms, 'mean_v': mean_v, 'std_v': std_v,
            'range_v': range_v, 'var_pct': var_pct
        })

    data['variation'] = variation_data

    # Impact analysis: how much traveltime error for different azimuths?
    print(f"\n[6.2] Traveltime error from lateral velocity variation:")

    # At test point, compute traveltime for different azimuth traces
    # Using velocity at output point vs velocity along ray path
    c1 = np.array(GRID_CORNERS['c1'])
    c2 = np.array(GRID_CORNERS['c2'])
    c4 = np.array(GRID_CORNERS['c4'])
    il_dir = (c2 - c1) / (NX - 1)
    xl_dir = (c4 - c1) / (NY - 1)

    output_pt = c1 + TEST_IX * il_dir + TEST_IY * xl_dir
    ox, oy = output_pt

    t0 = 1.0  # 1 second output time
    offset = 500  # meters
    t_idx = int(t0 * 1000 / DT_MS)

    v_at_output = velocity[TEST_IX, TEST_IY, t_idx]

    print(f"\n      Output point: IL={TEST_IL}, XL={TEST_XL}")
    print(f"      Velocity at output: {v_at_output:.0f} m/s")
    print(f"      t0 = {t0*1000:.0f} ms, offset = {offset} m")

    azimuths = np.arange(0, 360, 30)
    print(f"\n      Azimuth    Midpoint IL/XL    V_mid    t_output    t_midpoint    Error (ms)")

    azimuth_errors = []
    for az in azimuths:
        # Source/receiver for this azimuth
        dx = offset / 2 * np.sin(np.radians(az))
        dy = offset / 2 * np.cos(np.radians(az))

        sx, sy = ox - dx, oy - dy
        rx, ry = ox + dx, oy + dy
        mx, my = ox, oy  # midpoint is at output point for symmetric offset

        # But the ray paths go through different velocity regions
        # Approximate by checking velocity at midpoint of each ray leg
        mid_s = ((ox + sx) / 2, (oy + sy) / 2)
        mid_r = ((ox + rx) / 2, (oy + ry) / 2)

        # Convert to grid indices
        def xy_to_ilxl(x, y):
            # Solve: pt = c1 + il * il_dir + xl * xl_dir
            delta = np.array([x, y]) - c1
            # Use least squares since grid may not be perfectly orthogonal
            A = np.column_stack([il_dir, xl_dir])
            coeffs = np.linalg.lstsq(A, delta, rcond=None)[0]
            return int(np.clip(coeffs[0], 0, NX-1)), int(np.clip(coeffs[1], 0, NY-1))

        il_s, xl_s = xy_to_ilxl(*mid_s)
        il_r, xl_r = xy_to_ilxl(*mid_r)

        v_s = velocity[il_s, xl_s, t_idx]
        v_r = velocity[il_r, xl_r, t_idx]
        v_avg = (v_s + v_r) / 2

        # Traveltime with output velocity
        ds2 = (ox - sx)**2 + (oy - sy)**2
        dr2 = (ox - rx)**2 + (oy - ry)**2
        t0_half_sq = (t0 / 2)**2

        t_output = np.sqrt(t0_half_sq + ds2 / v_at_output**2) + np.sqrt(t0_half_sq + dr2 / v_at_output**2)

        # Traveltime with path-average velocity
        t_path = np.sqrt(t0_half_sq + ds2 / v_s**2) + np.sqrt(t0_half_sq + dr2 / v_r**2)

        error_ms = (t_output - t_path) * 1000

        print(f"      {az:>6.0f}    ({il_s:>3},{xl_s:>3})-({il_r:>3},{xl_r:>3})    {v_avg:>5.0f}    "
              f"{t_output*1000:>10.2f}    {t_path*1000:>12.2f}    {error_ms:>+9.2f}")

        azimuth_errors.append({
            'azimuth': az, 'v_avg': v_avg, 't_output': t_output,
            't_path': t_path, 'error_ms': error_ms
        })

    data['azimuth_errors'] = azimuth_errors

    max_error = max(abs(e['error_ms']) for e in azimuth_errors)
    if max_error > 1.0:
        issues.append(f"Significant traveltime error from lateral velocity: {max_error:.2f} ms")
        print(f"\n      WARNING: Max traveltime error = {max_error:.2f} ms")

    # Create QC figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Test 6: Velocity Sampling - Bin {BIN_NUM}', fontsize=14, fontweight='bold')

    # Velocity slice at t=1000ms
    ax = axes[0, 0]
    v_slice = velocity[:, :, 500]
    im = ax.imshow(v_slice.T, origin='lower', aspect='auto', cmap='jet',
                   extent=[1, NX, 1, NY])
    ax.plot(TEST_IL, TEST_XL, 'ko', markersize=10, markerfacecolor='white')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title('Velocity at t=1000ms')
    plt.colorbar(im, ax=ax, label='V (m/s)')

    # Velocity variation with time
    ax = axes[0, 1]
    times = [v['t_ms'] for v in variation_data]
    var_pcts = [v['var_pct'] for v in variation_data]
    ax.plot(times, var_pcts, 'b-o')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity Range / Mean (%)')
    ax.set_title('Lateral Velocity Variation vs Time')
    ax.grid(True, alpha=0.3)

    # Traveltime error by azimuth
    ax = axes[1, 0]
    azs = [e['azimuth'] for e in azimuth_errors]
    errs = [e['error_ms'] for e in azimuth_errors]
    ax.bar(azs, errs, width=25)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Traveltime Error (ms)')
    ax.set_title('Traveltime Error from Output-Point Velocity Approximation')
    ax.grid(True, alpha=0.3)

    # Polar plot of error
    ax = axes[1, 1]
    ax = fig.add_subplot(2, 2, 4, projection='polar')
    azs_rad = np.radians(azs)
    ax.plot(azs_rad, np.abs(errs), 'b-o')
    ax.set_title('|Traveltime Error| by Azimuth (ms)')

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "test_6_velocity_sampling.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n      Saved: {fig_path}")

    passed = len(issues) == 0
    return DiagnosticResult(
        "Velocity Sampling",
        passed,
        "PASSED" if passed else f"FAILED: {issues}",
        figures=[fig_path],
        data=data
    )


# =============================================================================
# Test 7: Single Trace Migration (most critical test)
# =============================================================================

def test_7_single_trace_migration(prev_data: dict = None) -> DiagnosticResult:
    """Test migration of a single synthetic trace with known geometry."""
    print("\n" + "="*60)
    print("TEST 7: Single Trace Migration Test")
    print("="*60)

    issues = []
    data = {}

    # Load actual trace data
    bin_dir = COMMON_OFFSET_DIR / f"offset_bin_{BIN_NUM:02d}"
    traces_store = zarr.open_array(bin_dir / "traces.zarr", mode='r')
    df = pl.read_parquet(bin_dir / "headers.parquet")

    # Get scalar and apply
    scalar = int(df['scalar_coord'][0])
    scale_factor = 1.0 / abs(scalar) if scalar < 0 else float(scalar) if scalar > 0 else 1.0

    # Find a trace near the test point
    c1 = np.array(GRID_CORNERS['c1'])
    c2 = np.array(GRID_CORNERS['c2'])
    c4 = np.array(GRID_CORNERS['c4'])
    il_dir = (c2 - c1) / (NX - 1)
    xl_dir = (c4 - c1) / (NY - 1)

    output_pt = c1 + TEST_IX * il_dir + TEST_IY * xl_dir
    ox, oy = output_pt

    # Compute midpoints
    sx = df['source_x'].to_numpy().astype(np.float64) * scale_factor
    sy = df['source_y'].to_numpy().astype(np.float64) * scale_factor
    rx = df['receiver_x'].to_numpy().astype(np.float64) * scale_factor
    ry = df['receiver_y'].to_numpy().astype(np.float64) * scale_factor
    mx = (sx + rx) / 2.0
    my = (sy + ry) / 2.0

    # Find closest trace to test point
    distances = np.sqrt((mx - ox)**2 + (my - oy)**2)
    nearest_idx = np.argmin(distances)

    print(f"\n[7.1] Found nearest trace to test point:")
    print(f"      Trace index: {nearest_idx}")
    print(f"      Distance to output: {distances[nearest_idx]:.1f} m")
    print(f"      Source: ({sx[nearest_idx]:.1f}, {sy[nearest_idx]:.1f})")
    print(f"      Receiver: ({rx[nearest_idx]:.1f}, {ry[nearest_idx]:.1f})")
    print(f"      Midpoint: ({mx[nearest_idx]:.1f}, {my[nearest_idx]:.1f})")
    print(f"      Output point: ({ox:.1f}, {oy:.1f})")

    # Get trace data
    bin_trace_idx = df['bin_trace_idx'].to_numpy()
    trace_storage_idx = bin_trace_idx[nearest_idx]

    # Handle transposed storage
    if traces_store.shape[0] < traces_store.shape[1]:
        trace_data = np.asarray(traces_store[:, trace_storage_idx])
    else:
        trace_data = np.asarray(traces_store[trace_storage_idx, :])

    trace_data = trace_data.astype(np.float32)
    print(f"\n[7.2] Trace data:")
    print(f"      Shape: {trace_data.shape}")
    print(f"      Min: {trace_data.min():.6f}")
    print(f"      Max: {trace_data.max():.6f}")
    print(f"      RMS: {np.sqrt(np.mean(trace_data**2)):.6f}")

    # Load velocity
    vel_store = zarr.open(VELOCITY_PATH, mode='r')
    velocity = np.asarray(vel_store)

    # Manually perform migration for this single trace
    print(f"\n[7.3] Manual single-trace migration:")

    n_samples = len(trace_data)
    dt_s = DT_MS / 1000.0

    # Get trace geometry
    trace_sx, trace_sy = sx[nearest_idx], sy[nearest_idx]
    trace_rx, trace_ry = rx[nearest_idx], ry[nearest_idx]

    # Compute distances
    ds2 = (ox - trace_sx)**2 + (oy - trace_sy)**2
    dr2 = (ox - trace_rx)**2 + (oy - trace_ry)**2

    print(f"      ds (to source): {np.sqrt(ds2):.1f} m")
    print(f"      dr (to receiver): {np.sqrt(dr2):.1f} m")

    # Migrate to output times
    t_axis_ms = np.arange(NT) * DT_MS
    migrated_trace = np.zeros(NT, dtype=np.float32)

    t_start_ms = 0.0

    for it in range(NT):
        t0_s = t_axis_ms[it] / 1000.0
        if t0_s < 0.1:  # Skip very shallow
            continue

        v = velocity[TEST_IX, TEST_IY, it]
        t0_half_sq = (t0_s / 2)**2
        inv_v_sq = 1 / (v * v)

        # DSR traveltime
        t_travel = np.sqrt(t0_half_sq + ds2 * inv_v_sq) + np.sqrt(t0_half_sq + dr2 * inv_v_sq)

        # Sample index
        sample_idx = (t_travel * 1000.0 - t_start_ms) / DT_MS

        # Interpolate
        if 0 <= sample_idx < n_samples - 1:
            idx0 = int(sample_idx)
            frac = sample_idx - idx0
            amp = trace_data[idx0] * (1 - frac) + trace_data[idx0 + 1] * frac
            migrated_trace[it] = amp

    print(f"\n[7.4] Migrated trace statistics:")
    print(f"      Non-zero samples: {np.count_nonzero(migrated_trace)}")
    print(f"      Min: {migrated_trace.min():.6f}")
    print(f"      Max: {migrated_trace.max():.6f}")
    print(f"      RMS: {np.sqrt(np.mean(migrated_trace**2)):.6f}")

    data['input_trace'] = trace_data
    data['migrated_trace'] = migrated_trace
    data['trace_geometry'] = {
        'sx': trace_sx, 'sy': trace_sy,
        'rx': trace_rx, 'ry': trace_ry,
        'ox': ox, 'oy': oy,
    }

    # Compare with actual migration output
    print(f"\n[7.5] Comparing with actual migration output:")
    migration_path = MIGRATION_OUTPUT / "migrated_stack.zarr"
    if migration_path.exists():
        mig_store = zarr.open_array(migration_path, mode='r')
        actual_migrated = np.asarray(mig_store[TEST_IX, TEST_IY, :])
        print(f"      Actual migration shape: {mig_store.shape}")
        print(f"      Actual at test point - Min: {actual_migrated.min():.6f}, Max: {actual_migrated.max():.6f}")
        data['actual_migrated'] = actual_migrated
    else:
        print(f"      Migration output not found: {migration_path}")

    # Create QC figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Test 7: Single Trace Migration - Bin {BIN_NUM}', fontsize=14, fontweight='bold')

    # Input trace
    ax = axes[0, 0]
    t_input = np.arange(n_samples) * DT_MS
    ax.plot(t_input, trace_data, 'b-', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Input Trace')
    ax.set_xlim([0, 2000])

    # Migrated trace
    ax = axes[0, 1]
    ax.plot(t_axis_ms, migrated_trace, 'r-', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Migrated Trace (Manual)')

    # Comparison
    ax = axes[1, 0]
    if 'actual_migrated' in data:
        ax.plot(t_axis_ms, actual_migrated, 'b-', linewidth=0.5, label='Actual (full migration)')
        ax.plot(t_axis_ms, migrated_trace, 'r--', linewidth=0.5, label='Manual (single trace)')
        ax.legend()
    else:
        ax.plot(t_axis_ms, migrated_trace, 'r-', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Comparison: Manual vs Actual Migration')

    # Geometry plot
    ax = axes[1, 1]
    ax.scatter([trace_sx], [trace_sy], s=100, c='blue', marker='*', label='Source')
    ax.scatter([trace_rx], [trace_ry], s=100, c='green', marker='v', label='Receiver')
    ax.scatter([ox], [oy], s=100, c='red', marker='o', label='Output Point')
    ax.scatter([(trace_sx+trace_rx)/2], [(trace_sy+trace_ry)/2], s=50, c='purple', marker='x', label='Midpoint')
    ax.plot([trace_sx, trace_rx], [trace_sy, trace_ry], 'k--', alpha=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trace Geometry')
    ax.legend()
    ax.axis('equal')

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "test_7_single_trace_migration.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n      Saved: {fig_path}")

    passed = len(issues) == 0
    return DiagnosticResult(
        "Single Trace Migration",
        passed,
        "PASSED" if passed else f"FAILED: {issues}",
        figures=[fig_path],
        data=data
    )


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all diagnostic tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("PSTM DIAGNOSTIC TEST SUITE")
    print(f"Offset Bin: {BIN_NUM}")
    print(f"Test Point: IL={TEST_IL}, XL={TEST_XL}")
    print("="*60)

    results = []
    data = {}

    # Test 1: Input Data
    result = test_1_input_data_validation()
    results.append(result)
    data.update(result.data or {})

    # Test 2: Coordinates
    result = test_2_coordinate_transformation(data)
    results.append(result)
    data.update(result.data or {})

    # Test 3: Spatial Index
    result = test_3_spatial_index(data)
    results.append(result)
    data.update(result.data or {})

    # Test 4: DSR
    result = test_4_dsr_traveltime(data)
    results.append(result)
    data.update(result.data or {})

    # Test 5: Interpolation
    result = test_5_sample_interpolation(data)
    results.append(result)
    data.update(result.data or {})

    # Test 6: Velocity
    result = test_6_velocity_sampling(data)
    results.append(result)
    data.update(result.data or {})

    # Test 7: Single Trace Migration
    result = test_7_single_trace_migration(data)
    results.append(result)
    data.update(result.data or {})

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    failed = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.message}")
        if r.passed:
            passed += 1
        else:
            failed += 1

    print(f"\n  Total: {passed} passed, {failed} failed")
    print(f"\n  QC figures saved to: {OUTPUT_DIR}/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSTM Diagnostic Test Suite")
    parser.add_argument("--test", type=str, default="all",
                       help="Test(s) to run: 'all', or comma-separated list (1,2,3)")
    args = parser.parse_args()

    if args.test.lower() == "all":
        run_all_tests()
    else:
        print("Specific test selection not yet implemented. Running all tests.")
        run_all_tests()
