"""
PSTM - 3D Prestack Kirchhoff Time Migration

A high-performance seismic migration package optimized for Apple Silicon.
"""

__version__ = "0.1.0"
__author__ = "Oleg"

from pstm.config.models import MigrationConfig
from pstm.settings import (
    get_settings,
    load_settings,
    save_settings,
    reset_settings,
    settings,
    ApplicationSettings,
)

__all__ = [
    "MigrationConfig",
    "__version__",
    # Settings
    "get_settings",
    "load_settings",
    "save_settings",
    "reset_settings",
    "settings",
    "ApplicationSettings",
]
