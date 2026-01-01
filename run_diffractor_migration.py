#!/usr/bin/env python3
"""
Run PSTM Migration on Synthetic Diffractor Data.

Migrates the synthetic 3D common offset data with point diffractors
and creates QC images to verify migration quality.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import (
    MigrationConfig,
    InputConfig,
    VelocityConfig,
    AlgorithmConfig,
    OutputConfig,
    ExecutionConfig,
    OutputGridConfig,
    ApertureConfig,
    AmplitudeConfig,
    ResourceConfig,
    TilingConfig,
    CheckpointConfig,
    VelocitySource,
    InterpolationMethod,
    ComputeBackend,
    OutputFormat,
    TimeVariantConfig,
    AntiAliasingConfig,
    AntiAliasingMethod,
    ColumnMapping,
)
from pstm.pipeline.executor import run_migration

# =============================================================================
# Configuration
# =============================================================================

SYNTHETIC_BASE = Path("/Users/olegadamovich/SeismicData/synthetic_diffractor_v2")
OUTPUT_BASE = SYNTHETIC_BASE / "migration_output"

# Algorithm parameters (matching real data script)
MAX_APERTURE_M = 2000.0
MIN_APERTURE_M = 500.0  # Match real data
MAX_DIP_DEGREES = 65.0  # Match real data

# Tile size
TILE_NX = 64
TILE_NY = 64


def load_grid_config(base_dir: Path) -> dict:
    """Load grid configuration."""
    config_path = base_dir / "grid_info.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def run_migration_task(grid_config: dict) -> bool:
    """Run migration for the synthetic diffractor data."""
    print(f"\n{'='*60}")
    print("Migrating Synthetic Diffractor Data")
    print(f"{'='*60}")

    input_dir = SYNTHETIC_BASE / "offset_bin_15"
    traces_path = input_dir / "traces.zarr"
    headers_path = input_dir / "headers.parquet"

    if not traces_path.exists() or not headers_path.exists():
        print(f"ERROR: Input data not found at {input_dir}")
        return False

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    velocity_path = SYNTHETIC_BASE / "velocity.zarr"

    # Build configuration
    input_config = InputConfig(
        traces_path=traces_path,
        headers_path=headers_path,
        columns=ColumnMapping(
            source_x="source_x",
            source_y="source_y",
            receiver_x="receiver_x",
            receiver_y="receiver_y",
            offset="offset",
            azimuth="sr_azim",
            trace_index="bin_trace_idx",
            coord_scalar="scalar_coord",
        ),
        apply_coord_scalar=True,
        sample_rate_ms=grid_config['dt_ms'],
        transposed=True,
    )

    velocity_config = VelocityConfig(
        source=VelocitySource.CUBE_3D,
        velocity_path=velocity_path,
        precompute_to_output_grid=True,
    )

    aperture_config = ApertureConfig(
        max_dip_degrees=MAX_DIP_DEGREES,
        min_aperture_m=MIN_APERTURE_M,
        max_aperture_m=MAX_APERTURE_M,
        taper_fraction=0.1,
    )

    amplitude_config = AmplitudeConfig(
        geometrical_spreading=False,  # Match real data
        obliquity_factor=False,
    )

    time_variant_config = TimeVariantConfig(enabled=False)
    anti_aliasing_config = AntiAliasingConfig(enabled=False, method=AntiAliasingMethod.NONE)

    algorithm_config = AlgorithmConfig(
        interpolation=InterpolationMethod.LINEAR,
        aperture=aperture_config,
        amplitude=amplitude_config,
        time_variant=time_variant_config,
        anti_aliasing=anti_aliasing_config,
    )

    # Grid from corners
    output_grid = OutputGridConfig.from_corners(
        corner1=tuple(grid_config['c1']),
        corner2=tuple(grid_config['c2']),
        corner3=tuple(grid_config['c3']),
        corner4=tuple(grid_config['c4']),
        t_min_ms=0.0,
        t_max_ms=grid_config['nt'] * grid_config['dt_ms'],
        dx=grid_config['dx'],
        dy=grid_config['dy'],
        dt_ms=grid_config['dt_ms'],
    )

    output_config = OutputConfig(
        output_dir=OUTPUT_BASE,
        grid=output_grid,
        format=OutputFormat.ZARR,
    )

    resource_config = ResourceConfig(
        backend=ComputeBackend.METAL_COMPILED,
        max_memory_gb=16.0,
    )

    tiling_config = TilingConfig(
        auto_tile_size=False,
        tile_nx=TILE_NX,
        tile_ny=TILE_NY,
        ordering='snake',
    )

    checkpoint_config = CheckpointConfig(
        enabled=True,
        interval_tiles=5,
    )

    execution_config = ExecutionConfig(
        resources=resource_config,
        tiling=tiling_config,
        checkpoint=checkpoint_config,
    )

    migration_config = MigrationConfig(
        name="Synthetic_Diffractor_Migration",
        input=input_config,
        velocity=velocity_config,
        algorithm=algorithm_config,
        output=output_config,
        execution=execution_config,
    )

    # Run migration
    import time
    start_time = time.time()
    try:
        run_migration(migration_config, resume=False)
        elapsed = time.time() - start_time
        print(f"Migration completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"Migration FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_qc_images(grid_config: dict):
    """Create QC images for the migrated diffractor data."""
    print("\n" + "="*60)
    print("Creating QC Images")
    print("="*60)

    # Load migrated data
    stack_path = OUTPUT_BASE / "migrated_stack.zarr"
    fold_path = OUTPUT_BASE / "fold.zarr"

    if not stack_path.exists():
        print(f"ERROR: Migrated data not found at {stack_path}")
        return

    print("Loading migrated volume...")
    z = zarr.open_array(str(stack_path), mode='r')
    image = np.array(z[:])
    print(f"  Shape: {image.shape}")

    if fold_path.exists():
        fold = np.array(zarr.open_array(str(fold_path), mode='r'))
    else:
        fold = np.ones_like(image, dtype=np.int32)

    nx, ny, nt = image.shape
    dx = grid_config['dx']
    dy = grid_config['dy']
    dt_ms = grid_config['dt_ms']

    # Compute amplitude clip
    vmax = np.percentile(np.abs(image), 99)
    if vmax == 0:
        vmax = 1.0

    # Diffractor expected locations (after migration, should be at t0)
    diffractors = grid_config.get('diffractors', [])
    print(f"Expected diffractors: {len(diffractors)}")
    for i, diff in enumerate(diffractors):
        print(f"  {i+1}. IL={diff[0]+1}, XL={diff[1]+1}, t0={diff[2]}ms")

    # === Figure 1: Inline sections ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PSTM Migration - Inline Sections\n'
                 '3 Point Diffractors at t=300ms, 500ms, 700ms', fontsize=14, fontweight='bold')

    # Plot at diffractor IL positions
    il_positions = [d[0] for d in diffractors]
    t_axis_ms = np.arange(nt) * dt_ms

    for i, il_idx in enumerate(il_positions):
        ax = axes[0, i]
        section = image[il_idx, :, :].T
        im = ax.imshow(section, aspect='auto', cmap='gray',
                       extent=[1, ny, t_axis_ms[-1], 0],
                       vmin=-vmax, vmax=vmax, interpolation='bilinear')
        ax.set_xlabel('Crossline')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Inline {il_idx+1}')

        # Mark expected diffractor time
        diff_t = diffractors[i][2]
        ax.axhline(y=diff_t, color='r', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot fold at same positions
    for i, il_idx in enumerate(il_positions):
        ax = axes[1, i]
        fold_section = fold[il_idx, :, :].T
        im = ax.imshow(fold_section, aspect='auto', cmap='viridis',
                       extent=[1, ny, t_axis_ms[-1], 0], interpolation='bilinear')
        ax.set_xlabel('Crossline')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Fold - Inline {il_idx+1}')
        plt.colorbar(im, ax=ax, label='Fold', shrink=0.8)

    plt.tight_layout()
    fig_path = OUTPUT_BASE / "qc_inline_sections.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")

    # === Figure 2: Crossline at center ===
    xl_center = ny // 2

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'PSTM Migration - Crossline {xl_center+1} (Center)\n'
                 'Should show 3 diffractor collapses', fontsize=14, fontweight='bold')

    # Image
    ax = axes[0]
    section = image[:, xl_center, :].T
    im = ax.imshow(section, aspect='auto', cmap='gray',
                   extent=[1, nx, t_axis_ms[-1], 0],
                   vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Migrated Amplitude')

    # Mark diffractor positions
    for diff in diffractors:
        ax.axhline(y=diff[2], color='r', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.plot(diff[0]+1, diff[2], 'r+', markersize=10, markeredgewidth=2)

    plt.colorbar(im, ax=ax, label='Amplitude', shrink=0.8)

    # Fold
    ax = axes[1]
    fold_section = fold[:, xl_center, :].T
    im = ax.imshow(fold_section, aspect='auto', cmap='viridis',
                   extent=[1, nx, t_axis_ms[-1], 0], interpolation='bilinear')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Fold')
    plt.colorbar(im, ax=ax, label='Fold', shrink=0.8)

    plt.tight_layout()
    fig_path = OUTPUT_BASE / "qc_crossline_center.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")

    # === Figure 3: Time slices at diffractor times ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PSTM Migration - Time Slices at Diffractor Times\n'
                 'Should show focused energy at diffractor locations', fontsize=14, fontweight='bold')

    for i, diff in enumerate(diffractors):
        t_ms = diff[2]
        t_idx = int(t_ms / dt_ms)
        t_idx = min(max(t_idx, 0), nt - 1)

        # Amplitude
        ax = axes[0, i]
        slice_data = image[:, :, t_idx]
        im = ax.imshow(slice_data.T, aspect='auto', cmap='gray',
                       extent=[1, nx, ny, 1],
                       vmin=-vmax, vmax=vmax, interpolation='bilinear', origin='lower')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        ax.set_title(f'Time = {t_ms:.0f} ms')

        # Mark expected position
        ax.plot(diff[0]+1, diff[1]+1, 'r+', markersize=15, markeredgewidth=2)

        # Fold
        ax = axes[1, i]
        fold_slice = fold[:, :, t_idx]
        im = ax.imshow(fold_slice.T, aspect='auto', cmap='viridis',
                       extent=[1, nx, ny, 1],
                       interpolation='bilinear', origin='lower')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        ax.set_title(f'Fold at {t_ms:.0f} ms')
        plt.colorbar(im, ax=ax, label='Fold', shrink=0.8)

    plt.tight_layout()
    fig_path = OUTPUT_BASE / "qc_time_slices.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")

    # === Figure 4: Summary with before/after comparison ===
    # Load input traces for comparison
    input_traces_path = SYNTHETIC_BASE / "offset_bin_15" / "traces.zarr"
    input_traces = np.array(zarr.open_array(str(input_traces_path), mode='r'))

    # Reshape input to grid (nt, n_traces) -> (nx, ny, nt)
    input_cube = input_traces.T.reshape(nx, ny, nt)
    input_vmax = np.percentile(np.abs(input_cube), 99)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('PSTM Migration Summary: Before vs After\n'
                 'Input (diffraction hyperbolas) → Output (collapsed diffractors)',
                 fontsize=14, fontweight='bold')

    # Input crossline
    ax = axes[0, 0]
    section = input_cube[:, xl_center, :].T
    ax.imshow(section, aspect='auto', cmap='gray',
              extent=[1, nx, t_axis_ms[-1], 0],
              vmin=-input_vmax, vmax=input_vmax, interpolation='bilinear')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Time (ms)')
    ax.set_title('INPUT: Crossline (Diffraction Hyperbolas)')

    # Output crossline
    ax = axes[0, 1]
    section = image[:, xl_center, :].T
    ax.imshow(section, aspect='auto', cmap='gray',
              extent=[1, nx, t_axis_ms[-1], 0],
              vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Time (ms)')
    ax.set_title('OUTPUT: Crossline (Collapsed Diffractors)')
    for diff in diffractors:
        ax.plot(diff[0]+1, diff[2], 'r+', markersize=10, markeredgewidth=2)

    # Input time slice at 500ms
    ax = axes[1, 0]
    t_idx = int(500 / dt_ms)
    slice_data = input_cube[:, :, t_idx]
    ax.imshow(slice_data.T, aspect='auto', cmap='gray',
              extent=[1, nx, ny, 1],
              vmin=-input_vmax, vmax=input_vmax, interpolation='bilinear', origin='lower')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title('INPUT: Time Slice @ 500ms')

    # Output time slice at 500ms
    ax = axes[1, 1]
    slice_data = image[:, :, t_idx]
    ax.imshow(slice_data.T, aspect='auto', cmap='gray',
              extent=[1, nx, ny, 1],
              vmin=-vmax, vmax=vmax, interpolation='bilinear', origin='lower')
    ax.set_xlabel('Inline')
    ax.set_ylabel('Crossline')
    ax.set_title('OUTPUT: Time Slice @ 500ms')
    # Mark the middle diffractor
    ax.plot(nx//2 + 1, ny//2 + 1, 'r+', markersize=15, markeredgewidth=2)

    plt.tight_layout()
    fig_path = OUTPUT_BASE / "qc_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")

    print(f"\nAll QC images saved to: {OUTPUT_BASE}")


def main():
    print("=" * 70)
    print("PSTM MIGRATION - SYNTHETIC DIFFRACTOR TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check input data exists
    if not SYNTHETIC_BASE.exists():
        print(f"ERROR: Synthetic data not found at {SYNTHETIC_BASE}")
        print("Run create_diffractor_synthetic.py first")
        return 1

    # Load grid config
    grid_config = load_grid_config(SYNTHETIC_BASE)
    print(f"\nGrid configuration:")
    print(f"  Grid: {grid_config['nx']} x {grid_config['ny']} x {grid_config['nt']}")
    print(f"  Spacing: dx={grid_config['dx']}m, dy={grid_config['dy']}m, dt={grid_config['dt_ms']}ms")
    print(f"  Offset: {grid_config.get('offset_m', grid_config.get('nominal_offset_m', 'N/A'))}m")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Run migration
    success = run_migration_task(grid_config)

    if not success:
        print("\nMigration failed!")
        return 1

    # Create QC images
    create_qc_images(grid_config)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_BASE}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_BASE.glob("*.png")):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
