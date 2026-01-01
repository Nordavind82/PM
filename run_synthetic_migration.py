#!/usr/bin/env python3
"""
Run PSTM Migration on Synthetic Azimuth Data

Runs the Metal shader-based migration on the synthetic common offset gathers
and creates QC images for comparison.

This script:
1. Migrates Offset 1 (Az 0-90) separately
2. Migrates Offset 2 (Az 90-180) separately
3. Migrates combined (Az 0-180) jointly
4. Creates comparison QC images (inline, crossline, time slices)
"""

import argparse
import json
import sys
import time
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

SYNTHETIC_BASE = Path("/Users/olegadamovich/SeismicData/synthetic_azimuth_test")
OUTPUT_BASE = SYNTHETIC_BASE / "migration_output"

# Algorithm parameters
MAX_APERTURE_M = 800.0
MIN_APERTURE_M = 200.0
MAX_DIP_DEGREES = 45.0

# Tile size
TILE_NX = 64
TILE_NY = 64


def load_grid_config(base_dir: Path) -> dict:
    """Load grid configuration."""
    config_path = base_dir / "grid_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def run_single_migration(
    input_dir: Path,
    output_dir: Path,
    velocity_path: Path,
    grid_config: dict,
    name: str
) -> bool:
    """Run migration for a single offset gather."""
    print(f"\n{'='*60}")
    print(f"Migrating: {name}")
    print(f"{'='*60}")

    traces_path = input_dir / "traces.zarr"
    headers_path = input_dir / "headers.parquet"

    if not traces_path.exists() or not headers_path.exists():
        print(f"ERROR: Input data not found at {input_dir}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

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
        apply_coord_scalar=False,  # Our synthetic data doesn't need scaling
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
        geometrical_spreading=False,
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
    corners = grid_config['corners']
    output_grid = OutputGridConfig.from_corners(
        corner1=tuple(corners['c1']),
        corner2=tuple(corners['c2']),
        corner3=tuple(corners['c3']),
        corner4=tuple(corners['c4']),
        t_min_ms=0.0,
        t_max_ms=grid_config['nt'] * grid_config['dt_ms'],
        dx=grid_config['dx'],
        dy=grid_config['dy'],
        dt_ms=grid_config['dt_ms'],
    )

    output_config = OutputConfig(
        output_dir=output_dir,
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
        name=name,
        input=input_config,
        velocity=velocity_config,
        algorithm=algorithm_config,
        output=output_config,
        execution=execution_config,
    )

    # Run migration
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


def create_qc_images(output_base: Path, grid_config: dict):
    """Create QC comparison images for all three migrations."""
    print("\n" + "="*60)
    print("Creating QC Images")
    print("="*60)

    # Load migrated images
    results = {}
    for name, subdir in [
        ('Offset 1 (Az 0-90)', 'migration_offset_01'),
        ('Offset 2 (Az 90-180)', 'migration_offset_02'),
        ('Joint (Az 0-180)', 'migration_offset_joint'),
    ]:
        stack_path = output_base / subdir / "migrated_stack.zarr"
        fold_path = output_base / subdir / "fold.zarr"

        if not stack_path.exists():
            print(f"WARNING: {stack_path} not found, skipping {name}")
            continue

        print(f"Loading {name}...")
        image = np.array(zarr.open_array(str(stack_path), mode='r'))
        if fold_path.exists():
            fold = np.array(zarr.open_array(str(fold_path), mode='r'))
        else:
            fold = np.ones_like(image, dtype=np.int32)

        results[name] = {'image': image, 'fold': fold}
        print(f"  Shape: {image.shape}")
        print(f"  Max fold: {fold.max()}")

    if len(results) < 3:
        print("ERROR: Not all migrations completed successfully")
        return

    # Grid params
    nx = grid_config['nx']
    ny = grid_config['ny']
    nt = grid_config['nt']
    dx = grid_config['dx']
    dy = grid_config['dy']
    dt_ms = grid_config['dt_ms']

    # Key positions
    il_idx = nx // 2
    xl_idx = ny // 2
    time_slices_ms = [200, 400, 600]

    # Create comparison figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('PSTM Migration Comparison: Azimuth Test\n'
                 'Offset 1 (Az 0-90) vs Offset 2 (Az 90-180) vs Joint',
                 fontsize=14, fontweight='bold')

    # Get global amplitude scale
    all_images = [r['image'] for r in results.values()]
    vmax = max(np.percentile(np.abs(img), 99) for img in all_images)
    if vmax == 0:
        vmax = 1.0
    vmin = -vmax

    for row, (name, data) in enumerate(results.items()):
        image = data['image']
        fold = data['fold']

        # Inline section
        ax = axes[row, 0]
        il_section = image[il_idx, :, :].T
        ax.imshow(il_section, aspect='auto', cmap='gray',
                  extent=[0, ny * dy, nt * dt_ms, 0],
                  vmin=vmin, vmax=vmax)
        ax.set_ylabel('Time (ms)')
        if row == 2:
            ax.set_xlabel('Crossline (m)')
        ax.set_title(f'{name}\nInline {il_idx}')

        # Inline fold
        ax = axes[row, 1]
        il_fold = fold[il_idx, :, :].T
        ax.imshow(il_fold, aspect='auto', cmap='viridis',
                  extent=[0, ny * dy, nt * dt_ms, 0])
        if row == 2:
            ax.set_xlabel('Crossline (m)')
        ax.set_title('Fold')

        # Time slice at 400ms
        t_idx = int(400 / dt_ms)
        ax = axes[row, 2]
        t_slice = image[:, :, t_idx].T
        ax.imshow(t_slice, aspect='equal', cmap='gray',
                  extent=[0, nx * dx, ny * dy, 0],
                  vmin=vmin, vmax=vmax)
        if row == 2:
            ax.set_xlabel('Inline (m)')
        ax.set_ylabel('Crossline (m)')
        ax.set_title('Time Slice 400 ms')

        # Fold at 400ms
        ax = axes[row, 3]
        f_slice = fold[:, :, t_idx].T
        ax.imshow(f_slice, aspect='equal', cmap='viridis',
                  extent=[0, nx * dx, ny * dy, 0])
        if row == 2:
            ax.set_xlabel('Inline (m)')
        ax.set_title('Fold at 400 ms')

    plt.tight_layout()
    comparison_path = output_base / "migration_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {comparison_path}")

    # Create individual QC figures for each migration
    for name, data in results.items():
        create_single_qc(data['image'], data['fold'], grid_config, name, output_base)

    # Create fold comparison
    create_fold_comparison(results, grid_config, output_base)


def create_single_qc(image: np.ndarray, fold: np.ndarray,
                     grid_config: dict, name: str, output_base: Path):
    """Create detailed QC figure for a single migration."""
    nx, ny, nt = image.shape
    dx = grid_config['dx']
    dy = grid_config['dy']
    dt_ms = grid_config['dt_ms']

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'PSTM Migration QC: {name}', fontsize=14, fontweight='bold')

    vmax = np.percentile(np.abs(image), 99)
    if vmax == 0:
        vmax = 1.0
    vmin = -vmax

    # Inline sections at different positions
    for i, il_frac in enumerate([0.25, 0.5, 0.75]):
        il_idx = int(nx * il_frac)
        ax = fig.add_subplot(3, 4, 1 + i)
        section = image[il_idx, :, :].T
        ax.imshow(section, aspect='auto', cmap='gray',
                  extent=[0, ny * dy, nt * dt_ms, 0],
                  vmin=vmin, vmax=vmax)
        ax.set_xlabel('Crossline (m)')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Inline {il_idx}')

    # Crossline sections
    for i, xl_frac in enumerate([0.25, 0.5, 0.75]):
        xl_idx = int(ny * xl_frac)
        ax = fig.add_subplot(3, 4, 5 + i)
        section = image[:, xl_idx, :].T
        ax.imshow(section, aspect='auto', cmap='gray',
                  extent=[0, nx * dx, nt * dt_ms, 0],
                  vmin=vmin, vmax=vmax)
        ax.set_xlabel('Inline (m)')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Crossline {xl_idx}')

    # Time slices
    for i, t_ms in enumerate([200, 400, 600]):
        t_idx = int(t_ms / dt_ms)
        ax = fig.add_subplot(3, 4, 9 + i)
        t_slice = image[:, :, t_idx].T
        ax.imshow(t_slice, aspect='equal', cmap='gray',
                  extent=[0, nx * dx, ny * dy, 0],
                  vmin=vmin, vmax=vmax)
        ax.set_xlabel('Inline (m)')
        ax.set_ylabel('Crossline (m)')
        ax.set_title(f'Time Slice {t_ms} ms')

    # Fold at center
    ax = fig.add_subplot(3, 4, 4)
    il_fold = fold[nx//2, :, :].T
    ax.imshow(il_fold, aspect='auto', cmap='viridis',
              extent=[0, ny * dy, nt * dt_ms, 0])
    ax.set_xlabel('Crossline (m)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Fold (Inline {nx//2})')
    ax.text(0.02, 0.98, f'Max: {fold.max()}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    # Fold time slice
    ax = fig.add_subplot(3, 4, 8)
    t_idx = int(400 / dt_ms)
    f_slice = fold[:, :, t_idx].T
    im = ax.imshow(f_slice, aspect='equal', cmap='viridis',
                   extent=[0, nx * dx, ny * dy, 0])
    ax.set_xlabel('Inline (m)')
    ax.set_ylabel('Crossline (m)')
    ax.set_title(f'Fold at 400 ms')
    plt.colorbar(im, ax=ax, label='Fold')

    # Amplitude histogram
    ax = fig.add_subplot(3, 4, 12)
    valid_amps = image[fold > 0].flatten()
    if len(valid_amps) > 0:
        ax.hist(valid_amps, bins=100, alpha=0.7, color='blue')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Count')
    ax.set_title('Amplitude Distribution')
    ax.set_yscale('log')

    plt.tight_layout()
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    fig_path = output_base / f"qc_{safe_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


def create_fold_comparison(results: dict, grid_config: dict, output_base: Path):
    """Create fold comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('3D Fold Comparison Across Migrations', fontsize=14, fontweight='bold')

    # Get actual dimensions from the data
    first_result = list(results.values())[0]
    nx, ny, nt = first_result['fold'].shape
    dt_ms = grid_config['dt_ms']

    t_idx = int(400 / dt_ms)  # Time slice at 400ms

    names = list(results.keys())

    # Top row: fold at time slice 400ms
    for i, name in enumerate(names):
        ax = axes[0, i]
        fold = results[name]['fold']
        f_slice = fold[:, :, t_idx].T
        im = ax.imshow(f_slice, aspect='equal', cmap='viridis')
        ax.set_title(f'{name}\nFold at 400ms')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        plt.colorbar(im, ax=ax)

    # Bottom row: fold vs time (center trace)
    ax = axes[1, 0]
    t_axis = np.arange(nt) * dt_ms
    colors = ['blue', 'red', 'green']
    for i, name in enumerate(names):
        fold = results[name]['fold']
        fold_center = fold[nx//2, ny//2, :]
        ax.plot(t_axis, fold_center, color=colors[i], label=name, linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Fold')
    ax.set_title('Fold vs Time (center location)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean fold profile
    ax = axes[1, 1]
    for i, name in enumerate(names):
        fold = results[name]['fold']
        mean_fold = np.mean(fold, axis=(0, 1))
        ax.plot(t_axis, mean_fold, color=colors[i], label=name, linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mean Fold')
    ax.set_title('Mean Fold vs Time (spatial average)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fold statistics
    ax = axes[1, 2]
    stats_text = "Fold Statistics:\n" + "-"*30 + "\n"
    for name in names:
        fold = results[name]['fold']
        valid_fold = fold[fold > 0]
        stats_text += f"\n{name}:\n"
        stats_text += f"  Max: {fold.max()}\n"
        stats_text += f"  Mean: {valid_fold.mean():.1f}\n"
        stats_text += f"  Median: {np.median(valid_fold):.0f}\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    ax.set_title('Summary Statistics')

    plt.tight_layout()
    fig_path = output_base / "fold_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Run PSTM on synthetic azimuth data")
    parser.add_argument("--skip-migration", action="store_true",
                        help="Skip migration, only create QC images")
    args = parser.parse_args()

    print("=" * 70)
    print("PSTM MIGRATION - SYNTHETIC AZIMUTH TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check input data exists
    if not SYNTHETIC_BASE.exists():
        print(f"ERROR: Synthetic data not found at {SYNTHETIC_BASE}")
        print("Run create_synthetic_azimuth_data.py first")
        return 1

    # Load grid config
    grid_config = load_grid_config(SYNTHETIC_BASE)
    velocity_path = SYNTHETIC_BASE / "velocity_synthetic.zarr"

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not args.skip_migration:
        # Run migrations
        success = True

        # Offset 1
        success &= run_single_migration(
            input_dir=SYNTHETIC_BASE / "synthetic_offset_01",
            output_dir=OUTPUT_BASE / "migration_offset_01",
            velocity_path=velocity_path,
            grid_config=grid_config,
            name="Synthetic_Offset_01_Az_0_90"
        )

        # Offset 2
        success &= run_single_migration(
            input_dir=SYNTHETIC_BASE / "synthetic_offset_02",
            output_dir=OUTPUT_BASE / "migration_offset_02",
            velocity_path=velocity_path,
            grid_config=grid_config,
            name="Synthetic_Offset_02_Az_90_180"
        )

        # Joint
        success &= run_single_migration(
            input_dir=SYNTHETIC_BASE / "synthetic_offset_joint",
            output_dir=OUTPUT_BASE / "migration_offset_joint",
            velocity_path=velocity_path,
            grid_config=grid_config,
            name="Synthetic_Offset_Joint_Az_0_180"
        )

        if not success:
            print("\nWARNING: Some migrations failed")

    # Create QC images
    create_qc_images(OUTPUT_BASE, grid_config)

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
