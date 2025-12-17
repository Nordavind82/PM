"""Wizard UI for PSTM."""

from pstm.wizard.app import check_textual_available, run_wizard

# Only import WizardApp if textual is available
if check_textual_available():
    from pstm.wizard.app import WizardApp
    __all__ = ["WizardApp", "check_textual_available", "run_wizard"]
else:
    WizardApp = None
    __all__ = ["check_textual_available", "run_wizard"]

# Alias for backward compatibility
MigrationWizard = WizardApp
