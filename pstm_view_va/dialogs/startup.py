"""Startup dialog for project selection."""

from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QWidget, QFrame
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont

from ..core import PSTMProject, get_recent_projects, add_recent_project


class StartupDialog(QDialog):
    """Startup dialog shown on application launch."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSTM Seismic Viewer")
        self.setMinimumSize(500, 400)
        self.setModal(True)

        self.selected_project: Optional[PSTMProject] = None
        self._setup_ui()
        self._load_recent_projects()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("PSTM Seismic Viewer")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Velocity Analysis & QC")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: gray;")
        layout.addWidget(subtitle)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #555;")
        layout.addWidget(line)

        # Buttons
        btn_layout = QHBoxLayout()

        self.new_btn = QPushButton("New Project")
        self.new_btn.setMinimumHeight(40)
        self.new_btn.clicked.connect(self._new_project)
        btn_layout.addWidget(self.new_btn)

        self.open_btn = QPushButton("Open Project...")
        self.open_btn.setMinimumHeight(40)
        self.open_btn.clicked.connect(self._open_project)
        btn_layout.addWidget(self.open_btn)

        layout.addLayout(btn_layout)

        # Recent projects section
        recent_label = QLabel("Recent Projects:")
        recent_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(recent_label)

        self.recent_list = QListWidget()
        self.recent_list.setMinimumHeight(150)
        self.recent_list.itemDoubleClicked.connect(self._open_recent)
        self.recent_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.recent_list)

        # Bottom buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(self.cancel_btn)

        self.ok_btn = QPushButton("Open")
        self.ok_btn.setEnabled(False)
        self.ok_btn.clicked.connect(self._open_selected)
        self.ok_btn.setDefault(True)
        bottom_layout.addWidget(self.ok_btn)

        layout.addLayout(bottom_layout)

    def _load_recent_projects(self):
        """Load recent projects into list."""
        self.recent_list.clear()

        recent = get_recent_projects(max_count=10)

        if not recent:
            item = QListWidgetItem("No recent projects")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            item.setForeground(Qt.GlobalColor.gray)
            self.recent_list.addItem(item)
            return

        for path, modified in recent:
            project_path = Path(path)
            name = project_path.stem

            # Format modified date
            date_str = ""
            if modified:
                try:
                    dt = datetime.fromisoformat(modified)
                    date_str = dt.strftime("%b %d, %Y %H:%M")
                except:
                    pass

            item = QListWidgetItem(f"{name}")
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, path)

            # Add date as second line or suffix
            if date_str:
                item.setText(f"{name}\n{date_str}")

            self.recent_list.addItem(item)

    def _on_selection_changed(self):
        """Handle selection change in recent list."""
        selected = self.recent_list.selectedItems()
        has_valid_selection = (
            len(selected) > 0 and
            selected[0].data(Qt.ItemDataRole.UserRole) is not None
        )
        self.ok_btn.setEnabled(has_valid_selection)

    def _new_project(self):
        """Create a new project."""
        # Ask for project location
        path = QFileDialog.getSaveFileName(
            self, "Create New Project",
            str(Path.home() / "SeismicData" / "new_project.pstm"),
            "PSTM Project (*.pstm)"
        )[0]

        if not path:
            return

        try:
            # Create project
            self.selected_project = PSTMProject.create(Path(path))
            add_recent_project(str(self.selected_project.path))
            self.accept()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{e}")

    def _open_project(self):
        """Open an existing project via file dialog."""
        path = QFileDialog.getExistingDirectory(
            self, "Open PSTM Project",
            str(Path.home() / "SeismicData")
        )

        if not path:
            return

        self._try_open_project(path)

    def _open_recent(self, item: QListWidgetItem):
        """Open a recent project by double-click."""
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._try_open_project(path)

    def _open_selected(self):
        """Open the selected recent project."""
        selected = self.recent_list.selectedItems()
        if selected:
            path = selected[0].data(Qt.ItemDataRole.UserRole)
            if path:
                self._try_open_project(path)

    def _try_open_project(self, path: str):
        """Try to open a project from path."""
        path = Path(path)

        if not PSTMProject.is_project(path):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Invalid Project",
                f"'{path.name}' is not a valid PSTM project.\n\n"
                "A project folder must contain a project.json file."
            )
            return

        try:
            self.selected_project = PSTMProject.open(path)
            add_recent_project(str(path))
            self.accept()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{e}")

    def get_project(self) -> Optional[PSTMProject]:
        """Get the selected/created project."""
        return self.selected_project


class NewProjectDialog(QDialog):
    """Dialog for creating a new project with initial settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(400)

        self.project_path: Optional[Path] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Project name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Project Name:"))
        from PyQt6.QtWidgets import QLineEdit
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("my_project")
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)

        # Location
        loc_layout = QHBoxLayout()
        loc_layout.addWidget(QLabel("Location:"))
        self.location_edit = QLineEdit()
        self.location_edit.setText(str(Path.home() / "SeismicData"))
        loc_layout.addWidget(self.location_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_location)
        loc_layout.addWidget(browse_btn)
        layout.addLayout(loc_layout)

        # Full path preview
        self.path_label = QLabel()
        self.path_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.path_label)

        self.name_edit.textChanged.connect(self._update_path_preview)
        self.location_edit.textChanged.connect(self._update_path_preview)
        self._update_path_preview()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        create_btn = QPushButton("Create")
        create_btn.clicked.connect(self._create_project)
        create_btn.setDefault(True)
        btn_layout.addWidget(create_btn)

        layout.addLayout(btn_layout)

    def _browse_location(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Project Location",
            self.location_edit.text()
        )
        if path:
            self.location_edit.setText(path)

    def _update_path_preview(self):
        name = self.name_edit.text() or "my_project"
        location = self.location_edit.text()
        full_path = Path(location) / f"{name}.pstm"
        self.path_label.setText(f"Will create: {full_path}")
        self.project_path = full_path

    def _create_project(self):
        if not self.name_edit.text():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Please enter a project name.")
            return

        self.accept()

    def get_project_path(self) -> Optional[Path]:
        """Get the project path to create."""
        return self.project_path
