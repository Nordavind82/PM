"""
PSTM Wizard Main Window

The main application window with sidebar navigation and step panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QPushButton, QFrame, QFileDialog, QMessageBox,
    QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QShortcut

from pstm.gui.state import WizardState, WizardController, StepStatus

if TYPE_CHECKING:
    from pstm.gui.steps.base import WizardStepWidget


class StepButton(QPushButton):
    """Sidebar button for a wizard step."""
    
    def __init__(self, step_number: int, title: str, parent=None):
        super().__init__(parent)
        self.step_number = step_number
        self.title = title
        self._status = StepStatus.NOT_VISITED
        self._is_current = False
        
        self.setCheckable(True)
        self.setMinimumHeight(50)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_display()
    
    def set_status(self, status: StepStatus) -> None:
        self._status = status
        self.update_display()
    
    def set_current(self, is_current: bool) -> None:
        self._is_current = is_current
        self.setChecked(is_current)
        self.update_display()
    
    def update_display(self) -> None:
        status_icons = {
            StepStatus.NOT_VISITED: "○",
            StepStatus.IN_PROGRESS: "◐",
            StepStatus.COMPLETE: "✓",
            StepStatus.ERROR: "✗",
            StepStatus.WARNING: "⚠",
        }
        
        status_colors = {
            StepStatus.NOT_VISITED: "#888888",
            StepStatus.IN_PROGRESS: "#4a9eff",
            StepStatus.COMPLETE: "#4caf50",
            StepStatus.ERROR: "#f44336",
            StepStatus.WARNING: "#ff9800",
        }
        
        icon = status_icons.get(self._status, "○")
        color = status_colors.get(self._status, "#888888")
        
        self.setText(f"  {icon}  {self.step_number}. {self.title}")
        
        if self._is_current:
            self.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding: 10px;
                    border: none;
                    border-left: 3px solid {color};
                    background-color: #3d3d3d;
                    color: white;
                    font-weight: bold;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding: 10px;
                    border: none;
                    border-left: 3px solid transparent;
                    background-color: transparent;
                    color: #cccccc;
                }}
                QPushButton:hover {{
                    background-color: #3d3d3d;
                }}
            """)


class SidebarWidget(QFrame):
    """Sidebar with step navigation."""
    
    step_clicked = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(200)
        
        self.steps: list[StepButton] = []
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Logo/Title
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: #1a1a2e;")
        title_layout = QVBoxLayout(title_frame)
        
        title_label = QLabel("PSTM")
        title_label.setStyleSheet("color: #4a9eff; font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(title_label)
        
        subtitle_label = QLabel("Migration Wizard")
        subtitle_label.setStyleSheet("color: #888888; font-size: 12px;")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(subtitle_label)
        
        layout.addWidget(title_frame)
        
        # Step buttons - NEW ORDER per requirements
        step_info = [
            (1, "Input Data"),
            (2, "Survey"),
            (3, "Output Grid"),      # OUTPUT GRID BEFORE VELOCITY
            (4, "Velocity"),          # VELOCITY AFTER OUTPUT GRID
            (5, "Data Selection"),    # NEW FLEXIBLE TAB
            (6, "Algorithm"),
            (7, "Execution"),
            (8, "Results"),           # NEW RESULTS STEP
            (9, "Visualization"),     # NEW VISUALIZATION STEP
        ]
        
        steps_frame = QFrame()
        steps_layout = QVBoxLayout(steps_frame)
        steps_layout.setContentsMargins(0, 10, 0, 0)
        steps_layout.setSpacing(2)
        
        for num, title in step_info:
            btn = StepButton(num, title)
            btn.clicked.connect(lambda checked, n=num: self.step_clicked.emit(n - 1))
            self.steps.append(btn)
            steps_layout.addWidget(btn)
        
        steps_layout.addStretch()
        layout.addWidget(steps_frame)
        
        # Progress
        self.progress_frame = QFrame()
        progress_layout = QVBoxLayout(self.progress_frame)
        
        self.progress_label = QLabel("Progress")
        self.progress_label.setStyleSheet("color: #888888; font-size: 11px;")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(len(step_info))
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(self.progress_frame)
    
    def set_current_step(self, step_index: int) -> None:
        for i, btn in enumerate(self.steps):
            btn.set_current(i == step_index)
    
    def set_step_status(self, step_index: int, status: StepStatus) -> None:
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].set_status(status)
    
    def update_progress(self, completed: int) -> None:
        self.progress_bar.setValue(completed)
        pct = int(100 * completed / max(1, self.progress_bar.maximum()))
        self.progress_label.setText(f"Progress: {pct}%")


class ActionBar(QFrame):
    """Bottom action bar with navigation buttons."""
    
    back_clicked = pyqtSignal()
    next_clicked = pyqtSignal()
    run_clicked = pyqtSignal()
    load_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("actionBar")
        self.setFixedHeight(60)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        
        self.load_btn = QPushButton("Load Config")
        self.load_btn.clicked.connect(self.load_clicked.emit)
        layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Save Config")
        self.save_btn.clicked.connect(self.save_clicked.emit)
        layout.addWidget(self.save_btn)
        
        layout.addStretch()
        
        self.back_btn = QPushButton("← Back")
        self.back_btn.clicked.connect(self.back_clicked.emit)
        self.back_btn.setMinimumWidth(100)
        layout.addWidget(self.back_btn)
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.next_clicked.emit)
        self.next_btn.setMinimumWidth(100)
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d8de8;
            }
        """)
        layout.addWidget(self.next_btn)
        
        self.run_btn = QPushButton("▶ Run Migration")
        self.run_btn.clicked.connect(self.run_clicked.emit)
        self.run_btn.setMinimumWidth(140)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_btn.hide()
        layout.addWidget(self.run_btn)
    
    def set_step(self, step_index: int, total_steps: int) -> None:
        self.back_btn.setEnabled(step_index > 0)
        is_last_step = step_index >= total_steps - 1
        self.next_btn.setVisible(not is_last_step)
        self.run_btn.setVisible(is_last_step)


class HeaderBar(QFrame):
    """Top header bar showing current status."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("headerBar")
        self.setFixedHeight(50)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        
        self.title_label = QLabel("Step 1: Input Data")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.status_label)
    
    def set_step(self, step_index: int, title: str) -> None:
        self.title_label.setText(f"Step {step_index + 1}: {title}")
    
    def set_status(self, status: str) -> None:
        self.status_label.setText(status)


class PSTMWizardWindow(QMainWindow):
    """Main wizard window."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSTM Migration Wizard")
        self.setMinimumSize(1200, 800)
        
        self.controller = WizardController()
        self.controller.add_change_callback(self._on_state_change)
        
        self.step_widgets: list[WizardStepWidget] = []
        self._last_save_path = None
        
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._create_steps()
        
        self._go_to_step(0)
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.sidebar = SidebarWidget()
        self.sidebar.step_clicked.connect(self._go_to_step)
        main_layout.addWidget(self.sidebar)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        self.header = HeaderBar()
        right_layout.addWidget(self.header)
        
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("background-color: #2d2d2d;")
        right_layout.addWidget(self.content_stack, 1)
        
        self.action_bar = ActionBar()
        self.action_bar.back_clicked.connect(self._go_back)
        self.action_bar.next_clicked.connect(self._go_next)
        self.action_bar.run_clicked.connect(self._run_migration)
        self.action_bar.load_clicked.connect(self._load_config)
        self.action_bar.save_clicked.connect(self._save_config)
        right_layout.addWidget(self.action_bar)
        
        main_layout.addWidget(right_widget)
    
    def _setup_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open Config...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._load_config)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save Config", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_config)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Left"), self, self._go_back)
        QShortcut(QKeySequence("Right"), self, self._go_next)

        for i in range(9):
            QShortcut(QKeySequence(f"Ctrl+{i+1}"), self, lambda idx=i: self._go_to_step(idx))
    
    def _create_steps(self):
        from pstm.gui.steps.input_step import InputDataStep
        from pstm.gui.steps.survey_step import SurveyStep
        from pstm.gui.steps.output_grid_step import OutputGridStep
        from pstm.gui.steps.velocity_step import VelocityStep
        from pstm.gui.steps.data_selection_step import DataSelectionStep
        from pstm.gui.steps.algorithm_step import AlgorithmStep
        from pstm.gui.steps.execution_step import ExecutionStep
        from pstm.gui.steps.results_step import ResultsStep
        from pstm.gui.steps.visualization_step import VisualizationStep

        step_classes = [
            InputDataStep,
            SurveyStep,
            OutputGridStep,
            VelocityStep,
            DataSelectionStep,
            AlgorithmStep,
            ExecutionStep,
            ResultsStep,
            VisualizationStep,
        ]

        for step_class in step_classes:
            step = step_class(self.controller)
            self.step_widgets.append(step)
            self.content_stack.addWidget(step)
    
    def _go_to_step(self, step_index: int) -> None:
        if 0 <= step_index < len(self.step_widgets):
            # Call on_leave() on current step before switching (if different step)
            current_idx = self.controller.state.current_step
            if current_idx != step_index and 0 <= current_idx < len(self.step_widgets):
                current_step = self.step_widgets[current_idx]
                current_step.on_leave()

            self.controller.state.current_step = step_index
            self.content_stack.setCurrentIndex(step_index)
            self.sidebar.set_current_step(step_index)

            step = self.step_widgets[step_index]
            self.header.set_step(step_index, step.title)
            self.action_bar.set_step(step_index, len(self.step_widgets))

            step.on_enter()

    def _go_back(self) -> None:
        current = self.controller.state.current_step
        if current > 0:
            self._go_to_step(current - 1)

    def _go_next(self) -> None:
        current = self.controller.state.current_step
        if current < len(self.step_widgets) - 1:
            step = self.step_widgets[current]
            if step.validate():
                self._go_to_step(current + 1)
            else:
                QMessageBox.warning(self, "Validation Error",
                    "Please complete the current step before proceeding.")
    
    def _run_migration(self) -> None:
        for i, step in enumerate(self.step_widgets):
            if not step.validate():
                QMessageBox.warning(self, "Validation Error",
                    f"Step {i+1} ({step.title}) has errors.")
                self._go_to_step(i)
                return
        
        og = self.controller.state.output_grid
        reply = QMessageBox.question(self, "Start Migration",
            f"Ready to start migration?\n\n"
            f"Output: {og.total_points:,} points\n"
            f"Estimated size: {og.estimated_size_gb:.2f} GB",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self._execute_migration()
    
    def _execute_migration(self) -> None:
        """Execute migration using the ExecutionStep's implementation."""
        # Navigate to Execution step (index 6) and trigger migration
        execution_step_index = 6
        self._go_to_step(execution_step_index)

        # Get the ExecutionStep and call its start migration method
        if len(self.step_widgets) > execution_step_index:
            execution_step = self.step_widgets[execution_step_index]
            if hasattr(execution_step, '_start_migration'):
                execution_step._start_migration()
    
    def _on_state_change(self) -> None:
        state = self.controller.state
        step_keys = ["input", "survey", "output_grid", "velocity",
                     "data_selection", "algorithm", "execution", "results", "visualization"]

        completed = 0
        for i, key in enumerate(step_keys):
            status = state.step_status.get(key, StepStatus.NOT_VISITED)
            self.sidebar.set_step_status(i, status)
            if status == StepStatus.COMPLETE:
                completed += 1

        self.sidebar.update_progress(completed)
    
    def _new_project(self) -> None:
        reply = QMessageBox.question(self, "New Project",
            "Create a new project? Current settings will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.controller.reset()
            self._go_to_step(0)
    
    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "",
            "JSON Files (*.json);;All Files (*)")
        if path:
            self.load_config(path)
    
    def load_config(self, path: str) -> None:
        try:
            self.controller.state = WizardState.load(path)
            self.controller.notify_change()
            self._go_to_step(self.controller.state.current_step)
            self.header.set_status(f"Loaded: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load:\n{e}")
    
    def _save_config(self) -> None:
        if self._last_save_path:
            self._do_save(self._last_save_path)
        else:
            self._save_config_as()
    
    def _save_config_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration",
            "pstm_config.json", "JSON Files (*.json);;All Files (*)")
        if path:
            self._do_save(path)
    
    def _do_save(self, path: str) -> None:
        try:
            self.controller.state.save(path)
            self._last_save_path = path
            self.header.set_status(f"Saved: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")
    
    def _show_about(self) -> None:
        QMessageBox.about(self, "About PSTM",
            "PSTM Migration Wizard\n\n"
            "3D Prestack Kirchhoff Time Migration\n"
            "Optimized for Apple Silicon\n\n"
            "Version 0.1.0")

    def closeEvent(self, event) -> None:
        """Handle window close - ensure any running migration is stopped safely."""
        # ExecutionStep is at index 6 (Step 7)
        if len(self.step_widgets) > 6:
            execution_step = self.step_widgets[6]
            if hasattr(execution_step, '_worker') and execution_step._worker:
                if execution_step._worker.isRunning():
                    reply = QMessageBox.question(
                        self, "Migration Running",
                        "A migration is currently running.\n\n"
                        "Do you want to stop it and save a checkpoint?\n"
                        "You can resume later from the checkpoint.",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )

                    if reply == QMessageBox.StandardButton.No:
                        event.ignore()
                        return

                    # Stop migration gracefully
                    execution_step.cleanup()

        event.accept()
