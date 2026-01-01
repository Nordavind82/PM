#!/usr/bin/env python3
"""
Compare kernel configurations between diagnostic and executor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pstm.kernels.base import KernelConfig
from pstm.kernels.metal_compiled import CompiledMetalKernel


def print_config(name, config):
    print(f"\n{name}:")
    print(f"  max_aperture_m: {config.max_aperture_m}")
    print(f"  min_aperture_m: {config.min_aperture_m}")
    print(f"  max_dip_degrees: {config.max_dip_degrees}")
    print(f"  taper_fraction: {config.taper_fraction}")
    print(f"  apply_spreading: {config.apply_spreading}")
    print(f"  apply_obliquity: {config.apply_obliquity}")
    print(f"  aa_enabled: {config.aa_enabled}")
    print(f"  interpolation_method: {config.interpolation_method}")
    print(f"  kernel_type: {config.kernel_type}")


def main():
    # Diagnostic-style config (from diagnostic_tile_by_tile.py)
    diag_config = KernelConfig(
        max_aperture_m=2000.0,
        min_aperture_m=500.0,
        taper_fraction=0.1,
        max_dip_degrees=65.0,
        apply_spreading=True,
        apply_obliquity=True,
        aa_enabled=False,
    )

    # Executor-style config (approximation based on run_pstm_all_offsets.py)
    exec_config = KernelConfig(
        max_aperture_m=2000.0,
        min_aperture_m=500.0,
        max_dip_degrees=65.0,
        taper_fraction=0.1,
        apply_spreading=True,
        apply_obliquity=True,
        interpolation_method="linear",
        output_dt_ms=2.0,
    )

    print_config("DIAGNOSTIC CONFIG", diag_config)
    print_config("EXECUTOR CONFIG", exec_config)

    # Check if they match
    print("\n--- Comparison ---")
    fields_to_compare = [
        'max_aperture_m', 'min_aperture_m', 'max_dip_degrees', 'taper_fraction',
        'apply_spreading', 'apply_obliquity', 'aa_enabled'
    ]

    all_match = True
    for field in fields_to_compare:
        diag_val = getattr(diag_config, field)
        exec_val = getattr(exec_config, field)
        match = diag_val == exec_val
        status = "MATCH" if match else "DIFFER"
        print(f"  {field}: diag={diag_val}, exec={exec_val} [{status}]")
        if not match:
            all_match = False

    if all_match:
        print("\nALL IMPORTANT FIELDS MATCH")
    else:
        print("\nCONFIGS DIFFER!")


if __name__ == "__main__":
    main()
