"""
PSTM Metal GPU Kernel

Provides high-performance GPU acceleration for PSTM migration using Apple Metal.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Try to import the compiled module
_metal_module = None
_import_error = None


def _find_and_import_module():
    """Find and import the compiled pstm_metal module."""
    global _metal_module, _import_error

    if _metal_module is not None:
        return _metal_module

    # Possible locations for the compiled module
    module_dir = Path(__file__).parent.parent
    search_paths = [
        module_dir / "build",  # Standard CMake build directory
        module_dir / "lib",
        module_dir,
        Path.cwd() / "build",
    ]

    # Add to sys.path temporarily and try to import
    original_path = sys.path.copy()

    for path in search_paths:
        if path.exists():
            sys.path.insert(0, str(path))

    try:
        import pstm_metal as metal
        _metal_module = metal
        return metal
    except ImportError as e:
        _import_error = str(e)
        return None
    finally:
        sys.path = original_path


def is_available() -> bool:
    """Check if Metal GPU acceleration is available."""
    module = _find_and_import_module()
    if module is None:
        return False
    try:
        return module.is_available()
    except Exception:
        return False


def get_device_info() -> dict:
    """Get information about the Metal GPU device."""
    module = _find_and_import_module()
    if module is None:
        return {
            "available": False,
            "device_name": "Not available",
            "device_memory_gb": 0,
            "error": _import_error,
        }
    return module.get_device_info()


def migrate_tile(
    amplitudes,
    source_x,
    source_y,
    receiver_x,
    receiver_y,
    midpoint_x,
    midpoint_y,
    image,
    fold,
    x_coords,
    y_coords,
    t_coords_ms,
    vrms,
    config,
):
    """
    Execute PSTM migration on GPU.

    See pstm_metal.migrate_tile for full documentation.
    """
    module = _find_and_import_module()
    if module is None:
        raise RuntimeError(
            f"Metal module not available. Build with: cd pstm/metal && mkdir build && cd build && cmake .. && make\n"
            f"Import error: {_import_error}"
        )

    # Find metallib path if not specified
    if "shader_path" not in config:
        module_dir = Path(__file__).parent.parent
        possible_paths = [
            module_dir / "build" / "migrate_tile.metallib",
            module_dir / "lib" / "migrate_tile.metallib",
            module_dir / "migrate_tile.metallib",
        ]
        for path in possible_paths:
            if path.exists():
                config = dict(config)  # Copy to avoid modifying original
                config["shader_path"] = str(path)
                break

    return module.migrate_tile(
        amplitudes,
        source_x,
        source_y,
        receiver_x,
        receiver_y,
        midpoint_x,
        midpoint_y,
        image,
        fold,
        x_coords,
        y_coords,
        t_coords_ms,
        vrms,
        config,
    )


def cleanup():
    """Release Metal GPU resources."""
    module = _find_and_import_module()
    if module is not None:
        module.cleanup()


__all__ = [
    "is_available",
    "get_device_info",
    "migrate_tile",
    "cleanup",
]
