"""
Base Step Widget

Abstract base class for wizard step panels.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame, 
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QPushButton, QCheckBox, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt

if TYPE_CHECKING:
    from pstm.gui.state import WizardController


class WizardStepWidget(QWidget):
    """Abstract base class for wizard step panels."""
    
    def __init__(self, controller: WizardController, parent=None):
        super().__init__(parent)
        self.controller = controller
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(30, 30, 30, 30)
        self.content_layout.setSpacing(20)
        
        scroll.setWidget(self.content)
        main_layout.addWidget(scroll)
        
        self._setup_ui()
    
    @property
    @abstractmethod
    def title(self) -> str:
        """Display title for this step."""
        pass
    
    @abstractmethod
    def _setup_ui(self) -> None:
        """Setup the step's UI components."""
        pass
    
    def on_enter(self) -> None:
        """Called when navigating to this step."""
        pass

    def on_leave(self) -> None:
        """Called when navigating away from this step."""
        pass

    def validate(self) -> bool:
        """Validate the step is complete/valid."""
        return True

    def refresh_from_state(self) -> None:
        """Refresh UI widgets from controller state.

        Called when state is loaded from file. Subclasses should override
        to update their widgets to match the loaded state.
        """
        pass
    
    def create_section(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        """Create a styled section with title."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                color: #4a9eff;
                font-size: 14px;
                font-weight: bold;
                border: none;
                background: transparent;
            }
        """)
        layout.addWidget(title_label)
        
        return frame, layout
    
    def create_form_row(self, label_text: str, widget: QWidget, 
                        tooltip: str = "") -> QHBoxLayout:
        """Create a labeled form row."""
        row = QHBoxLayout()
        row.setSpacing(10)
        
        label = QLabel(label_text)
        label.setFixedWidth(150)
        label.setStyleSheet("color: #cccccc; border: none; background: transparent;")
        if tooltip:
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        
        row.addWidget(label)
        row.addWidget(widget, 1)
        
        return row
    
    def create_info_box(self, text: str, style: str = "info") -> QFrame:
        """Create an info/warning/error box."""
        colors = {
            "info": ("#1a3a5c", "#4a9eff"),
            "warning": ("#5c4a1a", "#ff9800"),
            "error": ("#5c1a1a", "#f44336"),
            "success": ("#1a5c2e", "#4caf50"),
        }
        
        bg, border = colors.get(style, colors["info"])
        
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 4px;
            }}
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {border}; border: none; background: transparent;")
        layout.addWidget(label)
        
        return frame
    
    def create_header(self, title: str, description: str = "") -> QFrame:
        """Create a step header with title and optional description."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 20)
        layout.setSpacing(8)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 20px;
                font-weight: bold;
                border: none;
                background: transparent;
            }
        """)
        layout.addWidget(title_label)
        
        if description:
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("""
                QLabel {
                    color: #888888;
                    font-size: 13px;
                    border: none;
                    background: transparent;
                }
            """)
            layout.addWidget(desc_label)
        
        return frame
    
    def create_double_spinbox(self, min_val: float = 0, max_val: float = 99999,
                              decimals: int = 1, suffix: str = "") -> QDoubleSpinBox:
        """Create a styled double spinbox."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(f" {suffix}")
        spin.setMinimumWidth(120)
        return spin
    
    def create_spinbox(self, min_val: int = 0, max_val: int = 99999,
                       suffix: str = "") -> QSpinBox:
        """Create a styled spinbox."""
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        if suffix:
            spin.setSuffix(f" {suffix}")
        spin.setMinimumWidth(120)
        return spin
