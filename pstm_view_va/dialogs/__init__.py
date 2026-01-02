"""Dialog classes for seismic viewer."""

from .settings import SemblanceSettingsDialog, VelocityGridDialog
from .startup import StartupDialog, NewProjectDialog
from .stacking import StackingDialog

__all__ = [
    'SemblanceSettingsDialog', 'VelocityGridDialog',
    'StartupDialog', 'NewProjectDialog',
    'StackingDialog',
]
