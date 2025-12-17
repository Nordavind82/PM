"""
PSTM GUI Theme

Dark theme styling for the wizard interface.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


def apply_dark_theme(app: QApplication) -> None:
    """Apply dark theme to the application."""
    palette = QPalette()
    
    # Base colors
    dark = QColor(45, 45, 45)
    darker = QColor(35, 35, 35)
    darkest = QColor(25, 25, 25)
    light = QColor(200, 200, 200)
    lighter = QColor(220, 220, 220)
    accent = QColor(74, 158, 255)  # Blue accent
    
    # Window
    palette.setColor(QPalette.ColorRole.Window, dark)
    palette.setColor(QPalette.ColorRole.WindowText, light)
    
    # Base (text entry backgrounds)
    palette.setColor(QPalette.ColorRole.Base, darker)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark)
    
    # Text
    palette.setColor(QPalette.ColorRole.Text, light)
    palette.setColor(QPalette.ColorRole.BrightText, lighter)
    
    # Buttons
    palette.setColor(QPalette.ColorRole.Button, dark)
    palette.setColor(QPalette.ColorRole.ButtonText, light)
    
    # Highlights
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    
    # Tooltips
    palette.setColor(QPalette.ColorRole.ToolTipBase, darkest)
    palette.setColor(QPalette.ColorRole.ToolTipText, light)
    
    # Links
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(150, 120, 200))
    
    # Disabled
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    
    app.setPalette(palette)
    
    # Additional stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2d2d2d;
        }
        
        QToolTip {
            background-color: #1a1a1a;
            color: #cccccc;
            border: 1px solid #3d3d3d;
            padding: 5px;
        }
        
        QMenuBar {
            background-color: #252525;
            color: #cccccc;
        }
        
        QMenuBar::item:selected {
            background-color: #3d3d3d;
        }
        
        QMenu {
            background-color: #2d2d2d;
            color: #cccccc;
            border: 1px solid #3d3d3d;
        }
        
        QMenu::item:selected {
            background-color: #4a9eff;
        }
        
        QPushButton {
            background-color: #3d3d3d;
            border: 1px solid #4d4d4d;
            border-radius: 4px;
            padding: 8px 16px;
            color: #cccccc;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background-color: #4d4d4d;
            border-color: #5d5d5d;
        }
        
        QPushButton:pressed {
            background-color: #353535;
        }
        
        QPushButton:disabled {
            background-color: #2d2d2d;
            color: #666666;
        }
        
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #252525;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 6px;
            color: #cccccc;
            selection-background-color: #4a9eff;
        }
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
        QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #4a9eff;
        }
        
        QComboBox::drop-down {
            border: none;
            padding-right: 10px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #888888;
            margin-right: 5px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #2d2d2d;
            border: 1px solid #3d3d3d;
            selection-background-color: #4a9eff;
        }
        
        QTabWidget::pane {
            border: 1px solid #3d3d3d;
            background-color: #2d2d2d;
        }
        
        QTabBar::tab {
            background-color: #252525;
            border: 1px solid #3d3d3d;
            border-bottom: none;
            padding: 10px 20px;
            color: #888888;
        }
        
        QTabBar::tab:selected {
            background-color: #2d2d2d;
            color: #ffffff;
            border-bottom: 2px solid #4a9eff;
        }
        
        QTabBar::tab:hover:!selected {
            background-color: #353535;
        }
        
        QGroupBox {
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
            color: #cccccc;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #4a9eff;
        }
        
        QScrollBar:vertical {
            background-color: #252525;
            width: 12px;
            margin: 0;
        }
        
        QScrollBar::handle:vertical {
            background-color: #4d4d4d;
            border-radius: 6px;
            min-height: 20px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #5d5d5d;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        
        QScrollBar:horizontal {
            background-color: #252525;
            height: 12px;
            margin: 0;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #4d4d4d;
            border-radius: 6px;
            min-width: 20px;
            margin: 2px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #5d5d5d;
        }
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0;
        }
        
        QProgressBar {
            background-color: #252525;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            text-align: center;
            color: #cccccc;
        }
        
        QProgressBar::chunk {
            background-color: #4a9eff;
            border-radius: 3px;
        }
        
        QCheckBox {
            color: #cccccc;
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #4d4d4d;
            border-radius: 3px;
            background-color: #252525;
        }
        
        QCheckBox::indicator:checked {
            background-color: #4a9eff;
            border-color: #4a9eff;
        }
        
        QCheckBox::indicator:hover {
            border-color: #6d6d6d;
        }
        
        QRadioButton {
            color: #cccccc;
            spacing: 8px;
        }
        
        QRadioButton::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #4d4d4d;
            border-radius: 9px;
            background-color: #252525;
        }
        
        QRadioButton::indicator:checked {
            background-color: #4a9eff;
            border-color: #4a9eff;
        }
        
        QRadioButton::indicator:hover {
            border-color: #6d6d6d;
        }
        
        QSlider::groove:horizontal {
            height: 6px;
            background-color: #252525;
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal {
            width: 16px;
            height: 16px;
            margin: -5px 0;
            background-color: #4a9eff;
            border-radius: 8px;
        }
        
        QSlider::handle:horizontal:hover {
            background-color: #5aaeFF;
        }
        
        QTableWidget, QTableView {
            background-color: #252525;
            alternate-background-color: #2d2d2d;
            gridline-color: #3d3d3d;
            color: #cccccc;
        }
        
        QTableWidget::item:selected, QTableView::item:selected {
            background-color: #4a9eff;
        }
        
        QHeaderView::section {
            background-color: #353535;
            color: #cccccc;
            padding: 5px;
            border: none;
            border-right: 1px solid #3d3d3d;
            border-bottom: 1px solid #3d3d3d;
        }
        
        QSplitter::handle {
            background-color: #3d3d3d;
        }
        
        QSplitter::handle:hover {
            background-color: #4a9eff;
        }
        
        QFrame#sidebar {
            background-color: #252525;
            border-right: 1px solid #3d3d3d;
        }
        
        QFrame#headerBar {
            background-color: #1a1a2e;
            border-bottom: 1px solid #3d3d3d;
        }
        
        QFrame#actionBar {
            background-color: #252525;
            border-top: 1px solid #3d3d3d;
        }
    """)


def get_icon(name: str):
    """Get an icon by name (placeholder for actual icon loading)."""
    # In production, load actual icons from resources
    return None


# Color constants for use in custom widgets
class Colors:
    BACKGROUND = "#2d2d2d"
    DARKER = "#252525"
    DARKEST = "#1a1a1a"
    BORDER = "#3d3d3d"
    TEXT = "#cccccc"
    TEXT_LIGHT = "#ffffff"
    TEXT_DIM = "#888888"
    ACCENT = "#4a9eff"
    SUCCESS = "#4caf50"
    WARNING = "#ff9800"
    ERROR = "#f44336"


# Dict version for easy access
COLORS = {
    'background': "#2d2d2d",
    'background_alt': "#252525",
    'background_dark': "#1a1a1a",
    'border': "#3d3d3d",
    'text': "#cccccc",
    'text_light': "#ffffff",
    'text_dim': "#888888",
    'accent': "#4a9eff",
    'success': "#4caf50",
    'warning': "#ff9800",
    'error': "#f44336",
}
