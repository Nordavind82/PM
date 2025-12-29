#!/usr/bin/env python3
"""
Run grid migration for PSTM Velocity Analysis (pstm_va).

Creates a velocity analysis project with:
- Grid locations: every 20 inlines, every 40 crosslines
- Velocity scan: 1500-4500 m/s in 25 m/s steps
- Output: Zarr cubes + Parquet locations for pstm_va

Usage:
    python run_grid_migration_va.py [--dry-run]
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add pstm_va to path
sys.path.insert(0, '/Users/olegadamovich/pstm_va')
sys.path.insert(0, '/Users/olegadamovich/pstm')

import numpy as np
import polars as pl
import zarr


def create_va_project_structure(project_dir: Path) -> dict:
    """Create pstm_va compatible project structure."""

    # Create directories
    (project_dir / "config").mkdir(parents=True, exist_ok=True)
    (project_dir / "locations").mkdir(exist_ok=True)
    (project_dir / "derived").mkdir(exist_ok=True)
    (project_dir / "exports").mkdir(exist_ok=True)

    return {
        "config_dir": project_dir / "config",
        "locations_dir": project_dir / "locations",
        "derived_dir": project_dir / "derived",
        "exports_dir": project_dir / "exports",
    }


def define_grid_locations(
    il_range: tuple[int, int, int],  # (start, end, step)
    xl_range: tuple[int, int, int],
    headers_df: pl.DataFrame,
) -> list[dict]:
    """Define grid locations from inline/crossline ranges.

    Args:
        il_range: (start, end, step) for inlines
        xl_range: (start, end, step) for crosslines
        headers_df: Headers DataFrame with ix, iy, x, y columns

    Returns:
        List of location dictionaries
    """
    locations = []
    location_id = 0

    # Generate grid
    inlines = list(range(il_range[0], il_range[1] + 1, il_range[2]))
    crosslines = list(range(xl_range[0], xl_range[1] + 1, xl_range[2]))

    print(f"Grid: {len(inlines)} IL x {len(crosslines)} XL = {len(inlines) * len(crosslines)} locations")

    for il in inlines:
        for xl in crosslines:
            # Find nearest bin that exists
            candidates = headers_df.filter(
                (pl.col('ix') >= il - 5) & (pl.col('ix') <= il + 5) &
                (pl.col('iy') >= xl - 5) & (pl.col('iy') <= xl + 5)
            )

            if candidates.is_empty():
                continue

            # Get center coordinates (average of nearby bins)
            x_center = candidates['x'].mean()
            y_center = candidates['y'].mean()
            actual_il = int(candidates['ix'].mean())
            actual_xl = int(candidates['iy'].mean())

            locations.append({
                "location_id": location_id,
                "inline_number": actual_il,
                "crossline_center": actual_xl,
                "cdp_x_center": float(x_center),
                "cdp_y_center": float(y_center),
                "super_gather_extent": {
                    "inline_half_width": 2,
                    "crossline_half_width": 5,
                },
            })
            location_id += 1

    return locations


def create_velocity_scan_config() -> dict:
    """Create velocity scan configuration.

    Returns velocity scan parameters for pstm_va.
    """
    return {
        "v_min": 1500.0,      # Minimum velocity (m/s)
        "v_max": 4500.0,      # Maximum velocity (m/s)
        "v_step": 25.0,       # Velocity step (m/s)
        "velocity_type": "rms",
        "t_min_ms": 0.0,
        "t_max_ms": 2000.0,
        "dt_ms": 2.0,
    }


def create_offset_bin_config() -> dict:
    """Create offset binning configuration."""
    return {
        "min_offset": 0.0,
        "max_offset": 2000.0,  # Based on data offset range
        "n_bins": 40,          # 50m bins
        "bin_method": "linear",
    }


def create_project_config(
    project_name: str,
    scan_config: dict,
    offset_config: dict,
    n_locations: int,
    source_data_path: str,
) -> dict:
    """Create complete project configuration."""
    return {
        "metadata": {
            "name": project_name,
            "description": "Grid velocity analysis for PSTM - 20 IL x 40 XL",
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat(),
            "software_version": "0.1.0",
            "author": "",
            "survey_name": "SCD_XSD_Survey",
            "client": "",
        },
        "scan_config": scan_config,
        "offset_config": offset_config,
        "source_data": {
            "traces_path": f"{source_data_path}/traces.zarr",
            "headers_path": f"{source_data_path}/headers.parquet",
            "format": "zarr",
        },
        "n_locations": n_locations,
        "precompute_complete": False,
        "locations_defined": True,
    }


def save_locations_parquet(locations: list[dict], output_path: Path) -> None:
    """Save locations to Parquet format for pstm_va."""

    # Flatten for DataFrame
    rows = []
    for loc in locations:
        rows.append({
            "location_id": loc["location_id"],
            "inline_number": loc["inline_number"],
            "crossline_center": loc["crossline_center"],
            "cdp_x_center": loc["cdp_x_center"],
            "cdp_y_center": loc["cdp_y_center"],
            "inline_half_width": loc["super_gather_extent"]["inline_half_width"],
            "crossline_half_width": loc["super_gather_extent"]["crossline_half_width"],
        })

    df = pl.DataFrame(rows)
    df.write_parquet(output_path)
    print(f"Saved {len(rows)} locations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run grid migration for velocity analysis")
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")
    parser.add_argument("--il-step", type=int, default=20, help="Inline step (default: 20)")
    parser.add_argument("--xl-step", type=int, default=40, help="Crossline step (default: 40)")
    parser.add_argument("--output", type=str,
                       default="/Users/olegadamovich/SeismicData/VA_grid_20x40",
                       help="Output directory")

    args = parser.parse_args()

    print("=" * 60)
    print("PSTM Velocity Analysis - Grid Migration Setup")
    print("=" * 60)

    # Load existing bin headers to get survey geometry
    print("\n[1] Loading survey geometry...")
    headers_path = Path("/Users/olegadamovich/SeismicData/PSTM/bin_headers.parquet")
    headers_df = pl.read_parquet(headers_path)

    il_min, il_max = int(headers_df['ix'].min()), int(headers_df['ix'].max())
    xl_min, xl_max = int(headers_df['iy'].min()), int(headers_df['iy'].max())

    print(f"    Survey range: IL {il_min}-{il_max}, XL {xl_min}-{xl_max}")

    # Define grid locations
    print(f"\n[2] Defining grid locations (every {args.il_step} IL, {args.xl_step} XL)...")

    # Start from rounded values for clean grid
    il_start = ((il_min // args.il_step) + 1) * args.il_step
    xl_start = ((xl_min // args.xl_step) + 1) * args.xl_step

    locations = define_grid_locations(
        il_range=(il_start, il_max, args.il_step),
        xl_range=(xl_start, xl_max, args.xl_step),
        headers_df=headers_df,
    )

    print(f"    Created {len(locations)} locations")

    # Create velocity scan config
    print("\n[3] Creating velocity scan configuration...")
    scan_config = create_velocity_scan_config()

    n_velocities = int((scan_config["v_max"] - scan_config["v_min"]) / scan_config["v_step"]) + 1
    print(f"    Velocity range: {scan_config['v_min']:.0f} - {scan_config['v_max']:.0f} m/s")
    print(f"    Velocity step: {scan_config['v_step']:.0f} m/s ({n_velocities} velocities)")
    print(f"    Time range: {scan_config['t_min_ms']:.0f} - {scan_config['t_max_ms']:.0f} ms")

    # Create offset config
    offset_config = create_offset_bin_config()
    print(f"\n    Offset bins: {offset_config['n_bins']} bins (0-{offset_config['max_offset']:.0f}m)")

    # Estimate storage
    n_time = int((scan_config["t_max_ms"] - scan_config["t_min_ms"]) / scan_config["dt_ms"]) + 1
    bytes_per_location = (
        n_velocities * offset_config["n_bins"] * n_time * 4 +  # gathers
        n_velocities * n_time * 4 +  # stacks
        n_velocities * n_time * 4 +  # semblance
        n_velocities * n_time * 4    # stack power
    )
    total_gb = (bytes_per_location * len(locations)) / (1024**3)
    print(f"\n    Estimated storage: {total_gb:.2f} GB")

    # Create project structure
    print(f"\n[4] Creating project at: {args.output}")
    project_dir = Path(args.output)

    if args.dry_run:
        print("\n[DRY RUN] Would create:")
        print(f"    - {project_dir}/config/project_config.json")
        print(f"    - {project_dir}/config/locations.parquet")
        print(f"    - {project_dir}/locations/ (directory)")
        print(f"    - {project_dir}/derived/ (directory)")
        print("\n    Sample location:")
        print(f"    {locations[0]}")
        return 0

    # Create directories
    paths = create_va_project_structure(project_dir)

    # Save project config
    source_data_path = "/Users/olegadamovich/SeismicData/processing/processed_scd_xsd_data_new_20251221_225219_20251221_232357/output"
    project_config = create_project_config(
        project_name="VA_Grid_20x40",
        scan_config=scan_config,
        offset_config=offset_config,
        n_locations=len(locations),
        source_data_path=source_data_path,
    )

    config_path = paths["config_dir"] / "project_config.json"
    with open(config_path, 'w') as f:
        json.dump(project_config, f, indent=2, default=str)
    print(f"    Saved config: {config_path}")

    # Save locations
    locations_path = paths["config_dir"] / "locations.parquet"
    save_locations_parquet(locations, locations_path)

    # Also save as JSON for reference
    locations_json_path = paths["config_dir"] / "locations.json"
    with open(locations_json_path, 'w') as f:
        json.dump(locations, f, indent=2)
    print(f"    Saved locations JSON: {locations_json_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Grid Migration Project Created Successfully!")
    print("=" * 60)
    print(f"\nProject: {project_dir}")
    print(f"Locations: {len(locations)}")
    print(f"Velocity scan: {scan_config['v_min']:.0f}-{scan_config['v_max']:.0f} m/s @ {scan_config['v_step']:.0f} m/s")
    print(f"Estimated storage: {total_gb:.2f} GB")

    print("\nNext steps:")
    print("1. Run precomputation: pstm_va --project", str(project_dir))
    print("2. Or use Python API:")
    print("""
    from pstm_velocity_analysis.precompute import VelocityScanPrecomputer
    from pstm_velocity_analysis.storage import VelocityAnalysisProject

    project = VelocityAnalysisProject.load(Path("{project_dir}"))
    precomputer = VelocityScanPrecomputer(project, extractor)
    precomputer.compute_all(n_workers=8)
    """.format(project_dir=project_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())
