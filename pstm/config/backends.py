"""
Centralized backend mapping and validation.

This module provides the single source of truth for backend string-to-enum
mapping, eliminating duplicate code and silent fallbacks.
"""

from __future__ import annotations

from pstm.config.models import ComputeBackend


# Single source of truth for backend string mapping
BACKEND_STRING_MAP: dict[str, ComputeBackend] = {
    "auto": ComputeBackend.AUTO,
    "numpy": ComputeBackend.NUMPY,
    "numba_cpu": ComputeBackend.NUMBA_CPU,
    "mlx_metal": ComputeBackend.MLX_METAL,
    "metal_cpp": ComputeBackend.METAL_CPP,
}

# Valid backend strings for validation messages
VALID_BACKEND_STRINGS = list(BACKEND_STRING_MAP.keys())


def parse_backend(backend: str | ComputeBackend) -> ComputeBackend:
    """
    Parse backend string or enum to ComputeBackend enum.

    Args:
        backend: Backend as string or enum

    Returns:
        ComputeBackend enum value

    Raises:
        ValueError: If backend string is not recognized
    """
    if isinstance(backend, ComputeBackend):
        return backend

    if not isinstance(backend, str):
        raise TypeError(f"Backend must be string or ComputeBackend, got {type(backend).__name__}")

    backend_lower = backend.lower().strip()

    if backend_lower not in BACKEND_STRING_MAP:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Valid backends: {', '.join(VALID_BACKEND_STRINGS)}"
        )

    return BACKEND_STRING_MAP[backend_lower]


def validate_backend_string(backend: str) -> None:
    """
    Validate that a backend string is recognized.

    Args:
        backend: Backend string to validate

    Raises:
        ValueError: If backend string is not recognized
    """
    if backend.lower().strip() not in BACKEND_STRING_MAP:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Valid backends: {', '.join(VALID_BACKEND_STRINGS)}"
        )
