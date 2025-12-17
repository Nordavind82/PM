"""
PSTM Migration Wizard - PyQt6 GUI

Professional wizard interface for 3D Prestack Kirchhoff Time Migration.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pstm.gui.main_window import PSTMWizardWindow


def check_pyqt_available() -> bool:
    """Check if PyQt6 is available."""
    try:
        from PyQt6.QtWidgets import QApplication
        return True
    except ImportError:
        return False


def run_wizard(config_path: str | None = None) -> int:
    """
    Launch the PSTM Migration Wizard.
    
    Args:
        config_path: Optional path to load configuration from
        
    Returns:
        Application exit code
    """
    if not check_pyqt_available():
        print("Error: PyQt6 is required for the GUI wizard.")
        print("Install with: pip install PyQt6")
        return 1
    
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from pstm.gui.main_window import PSTMWizardWindow
    from pstm.gui.theme import apply_dark_theme
    
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("PSTM Migration Wizard")
    app.setOrganizationName("PSTM")
    app.setApplicationVersion("0.1.0")
    
    # Apply dark theme
    app.setStyle("Fusion")
    apply_dark_theme(app)
    
    # Create and show main window
    window = PSTMWizardWindow()
    
    if config_path:
        window.load_config(config_path)
    
    window.show()
    
    return app.exec()


__all__ = ["run_wizard", "check_pyqt_available"]
