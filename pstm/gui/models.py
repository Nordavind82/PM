"""
GUI Models - Re-exports and aliases for wizard components.

This module provides type aliases and re-exports for GUI step components.
"""

from pstm.config.models import VelocitySource, InterpolationMethod, ComputeBackend
from pstm.config.data_selection import SelectionMode, AzimuthConvention, RangeMode
from pstm.gui.state import WizardState, WizardController

# Alias for backwards compatibility
ProjectModel = WizardState

__all__ = [
    "VelocitySource",
    "InterpolationMethod", 
    "ComputeBackend",
    "SelectionMode",
    "AzimuthConvention",
    "RangeMode",
    "ProjectModel",
    "WizardState",
    "WizardController",
]
