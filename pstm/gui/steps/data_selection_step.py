"""
Step 5: Data Selection

Flexible data selection with offset/azimuth/OVT/custom expression filtering.
NO validation - user takes full responsibility for selections.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QGridLayout, QDoubleSpinBox, QRadioButton, QButtonGroup,
    QLineEdit, QCheckBox, QTabWidget, QWidget, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QGroupBox,
)
from PyQt6.QtCore import Qt

from pstm.gui.steps.base import WizardStepWidget
from pstm.gui.state import StepStatus, OffsetRange, AzimuthSector, GatherBinState


class DataSelectionStep(WizardStepWidget):
    """Step 5: Data Selection - Flexible filtering, NO validation."""
    
    @property
    def title(self) -> str:
        return "Data Selection"
    
    def _setup_ui(self) -> None:
        # Philosophy info box
        info = self.create_info_box(
            "Maximum flexibility - NO validation or hardcoded limits.\n"
            "You are responsible for ensuring adequate coverage and fold.",
            "warning"
        )
        self.content_layout.addWidget(info)
        
        # Create tabs FIRST before mode selection (so signal can reference it)
        self.tabs = QTabWidget()
        self.tabs.setVisible(False)
        
        # Selection Mode
        mode_frame, mode_layout = self.create_section("Selection Mode")
        
        self.mode_group = QButtonGroup(self)
        
        modes = [
            ("all", "Use All Data (no filtering)"),
            ("offset", "Filter by Offset Range"),
            ("azimuth", "Filter by Offset-Azimuth Sectors"),
            ("ovt", "Filter by Offset Vector (OVT Style)"),
            ("custom", "Custom Expression"),
        ]
        
        for key, label in modes:
            radio = QRadioButton(label)
            radio.setProperty("mode_key", key)
            radio.toggled.connect(self._on_mode_changed)
            self.mode_group.addButton(radio)
            mode_layout.addWidget(radio)
        
        self.mode_group.buttons()[0].setChecked(True)
        
        self.content_layout.addWidget(mode_frame)
        
        # Create tab content
        self._create_offset_tab()
        self._create_azimuth_tab()
        self._create_ovt_tab()
        self._create_custom_tab()
        self._create_gather_bins_tab()  # New: unified mixed bins

        self.content_layout.addWidget(self.tabs)

        # Selection Summary
        self._create_summary_section()
        
        # User Acknowledgment
        ack_frame, ack_layout = self.create_section("User Acknowledgment")
        
        self.ack_checkbox = QCheckBox(
            "I understand that restrictive selection may cause incomplete coverage, "
            "low fold, or artifacts. I take full responsibility for my selections."
        )
        self.ack_checkbox.setStyleSheet("color: #ff9800;")
        ack_layout.addWidget(self.ack_checkbox)
        
        self.content_layout.addWidget(ack_frame)
        self.content_layout.addStretch()
    
    def _create_offset_tab(self) -> None:
        """Create offset range tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.offset_table = QTableWidget(3, 2)
        self.offset_table.setHorizontalHeaderLabels(["Min Offset (m)", "Max Offset (m)"])
        self.offset_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.offset_table.setMaximumHeight(150)
        
        for row in range(3):
            self.offset_table.setItem(row, 0, QTableWidgetItem(""))
            self.offset_table.setItem(row, 1, QTableWidgetItem(""))
        
        layout.addWidget(QLabel("Offset Ranges (leave blank for no limit):"))
        layout.addWidget(self.offset_table)
        
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Range")
        add_btn.clicked.connect(lambda: self.offset_table.insertRow(self.offset_table.rowCount()))
        btn_row.addWidget(add_btn)
        
        remove_btn = QPushButton("- Remove Range")
        remove_btn.clicked.connect(lambda: self.offset_table.removeRow(max(0, self.offset_table.currentRow())))
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        
        self.offset_include = QCheckBox("Include mode (checked) / Exclude mode (unchecked)")
        self.offset_include.setChecked(True)
        layout.addWidget(self.offset_include)
        
        self.include_negative = QCheckBox("Include negative offsets")
        self.include_negative.setChecked(True)
        layout.addWidget(self.include_negative)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Offset Range")
    
    def _create_azimuth_tab(self) -> None:
        """Create azimuth sectors tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        conv_row = QHBoxLayout()
        conv_row.addWidget(QLabel("Azimuth Convention:"))
        self.azimuth_convention = QComboBox()
        self.azimuth_convention.addItems([
            "Receiver-relative (0-360°)",
            "North 0-360°",
            "North ±180°",
        ])
        conv_row.addWidget(self.azimuth_convention)
        conv_row.addStretch()
        layout.addLayout(conv_row)
        
        self.sector_table = QTableWidget(4, 5)
        self.sector_table.setHorizontalHeaderLabels([
            "Active", "Offset Min", "Offset Max", "Azimuth Min", "Azimuth Max"
        ])
        self.sector_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sector_table.setMaximumHeight(200)
        
        defaults = [
            (True, "", "", "0", "90"),
            (True, "", "", "90", "180"),
            (True, "", "", "180", "270"),
            (True, "", "", "270", "360"),
        ]
        for row, (active, o_min, o_max, a_min, a_max) in enumerate(defaults):
            cb = QCheckBox()
            cb.setChecked(active)
            self.sector_table.setCellWidget(row, 0, cb)
            self.sector_table.setItem(row, 1, QTableWidgetItem(o_min))
            self.sector_table.setItem(row, 2, QTableWidgetItem(o_max))
            self.sector_table.setItem(row, 3, QTableWidgetItem(a_min))
            self.sector_table.setItem(row, 4, QTableWidgetItem(a_max))
        
        layout.addWidget(QLabel("Azimuth Sectors:"))
        layout.addWidget(self.sector_table)
        
        rose_frame = QFrame()
        rose_frame.setMinimumHeight(150)
        rose_frame.setStyleSheet("background-color: #1a1a1a; border: 1px solid #3d3d3d; border-radius: 4px;")
        rose_layout = QVBoxLayout(rose_frame)
        rose_label = QLabel("Rose diagram preview (interactive coming soon)")
        rose_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rose_label.setStyleSheet("color: #666666;")
        rose_layout.addWidget(rose_label)
        layout.addWidget(rose_frame)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Offset-Azimuth")
    
    def _create_ovt_tab(self) -> None:
        """Create OVT tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        ovt_info = QLabel(
            "Offset Vector components:\n"
            "  Offset_X = Rx - Sx (positive = receiver East of source)\n"
            "  Offset_Y = Ry - Sy (positive = receiver North of source)"
        )
        ovt_info.setStyleSheet("color: #888888; background-color: #1a1a1a; padding: 10px; border-radius: 4px;")
        layout.addWidget(ovt_info)

        # OVT Mode selection
        mode_group = QGroupBox("OVT Definition Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.ovt_mode_range = QRadioButton("Range mode (min/max for X and Y)")
        self.ovt_mode_vector = QRadioButton("Vector mode (define tile edges from vectors)")
        self.ovt_mode_range.setChecked(True)
        self.ovt_mode_range.toggled.connect(self._on_ovt_mode_changed)

        mode_layout.addWidget(self.ovt_mode_range)
        mode_layout.addWidget(self.ovt_mode_vector)
        layout.addWidget(mode_group)

        # Range mode controls
        self.ovt_range_frame = QFrame()
        ovt_range_layout = QGridLayout(self.ovt_range_frame)

        ovt_range_layout.addWidget(QLabel("Offset_X range:"), 0, 0)
        self.ovt_x_min = QLineEdit()
        self.ovt_x_min.setPlaceholderText("min (blank=no limit)")
        ovt_range_layout.addWidget(self.ovt_x_min, 0, 1)

        self.ovt_x_max = QLineEdit()
        self.ovt_x_max.setPlaceholderText("max (blank=no limit)")
        ovt_range_layout.addWidget(self.ovt_x_max, 0, 2)

        ovt_range_layout.addWidget(QLabel("Offset_Y range:"), 1, 0)
        self.ovt_y_min = QLineEdit()
        self.ovt_y_min.setPlaceholderText("min (blank=no limit)")
        ovt_range_layout.addWidget(self.ovt_y_min, 1, 1)

        self.ovt_y_max = QLineEdit()
        self.ovt_y_max.setPlaceholderText("max (blank=no limit)")
        ovt_range_layout.addWidget(self.ovt_y_max, 1, 2)

        layout.addWidget(self.ovt_range_frame)

        # Vector mode controls
        self.ovt_vector_frame = QFrame()
        ovt_vector_layout = QVBoxLayout(self.ovt_vector_frame)

        vector_info = QLabel(
            "Enter comma-separated offset values to define tile edges.\n"
            "Example: -1000,-600,-200,200,600,1000 creates 5 tiles"
        )
        vector_info.setStyleSheet("color: #888888;")
        ovt_vector_layout.addWidget(vector_info)

        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("X Vector:"))
        self.ovt_vector_x = QLineEdit()
        self.ovt_vector_x.setPlaceholderText("-1000,-600,-200,200,600,1000")
        self.ovt_vector_x.textChanged.connect(self._update_ovt_tile_table)
        x_row.addWidget(self.ovt_vector_x)
        ovt_vector_layout.addLayout(x_row)

        y_row = QHBoxLayout()
        y_row.addWidget(QLabel("Y Vector:"))
        self.ovt_vector_y = QLineEdit()
        self.ovt_vector_y.setPlaceholderText("-1000,-600,-200,200,600,1000")
        self.ovt_vector_y.textChanged.connect(self._update_ovt_tile_table)
        y_row.addWidget(self.ovt_vector_y)
        ovt_vector_layout.addLayout(y_row)

        # Generated tiles table
        self.ovt_tiles_table = QTableWidget(0, 6)
        self.ovt_tiles_table.setHorizontalHeaderLabels([
            "Tile IX", "Tile IY", "X Min", "X Max", "Y Min", "Y Max"
        ])
        self.ovt_tiles_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ovt_tiles_table.setMaximumHeight(200)
        ovt_vector_layout.addWidget(QLabel("Generated OVT Tiles:"))
        ovt_vector_layout.addWidget(self.ovt_tiles_table)

        self.ovt_tiles_count_label = QLabel("0 tiles defined")
        ovt_vector_layout.addWidget(self.ovt_tiles_count_label)

        self.ovt_vector_frame.setVisible(False)
        layout.addWidget(self.ovt_vector_frame)

        # Tile size group (for range mode)
        tile_group = QGroupBox("OVT Tile Grid (optional)")
        tile_layout = QGridLayout(tile_group)

        tile_layout.addWidget(QLabel("Tile size X:"), 0, 0)
        self.tile_size_x = QDoubleSpinBox()
        self.tile_size_x.setRange(10, 10000)
        self.tile_size_x.setValue(500)
        self.tile_size_x.setSuffix(" m")
        tile_layout.addWidget(self.tile_size_x, 0, 1)

        tile_layout.addWidget(QLabel("Tile size Y:"), 0, 2)
        self.tile_size_y = QDoubleSpinBox()
        self.tile_size_y.setRange(10, 10000)
        self.tile_size_y.setValue(500)
        self.tile_size_y.setSuffix(" m")
        tile_layout.addWidget(self.tile_size_y, 0, 3)

        self.tile_size_group = tile_group
        layout.addWidget(tile_group)

        # Output gathers checkbox
        self.ovt_output_gathers = QCheckBox("Create separate output volume per OVT tile")
        self.ovt_output_gathers.setStyleSheet("color: #4a9eff;")
        layout.addWidget(self.ovt_output_gathers)

        # Quadrant diagram
        quad_frame = QFrame()
        quad_frame.setMinimumHeight(130)
        quad_frame.setStyleSheet("background-color: #1a1a1a; border: 1px solid #3d3d3d; border-radius: 4px;")
        quad_layout = QVBoxLayout(quad_frame)
        quad_label = QLabel(
            "       +Y (N)\n"
            "        ↑\n"
            "   Q2   │   Q1\n"
            "  (-,+) │  (+,+)\n"
            "────────┼────────→ +X (E)\n"
            "   Q3   │   Q4\n"
            "  (-,-) │  (+,-)"
        )
        quad_label.setStyleSheet("font-family: monospace; color: #4a9eff;")
        quad_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        quad_layout.addWidget(quad_label)
        layout.addWidget(quad_frame)

        layout.addStretch()
        self.tabs.addTab(tab, "Offset Vector (OVT)")

    def _on_ovt_mode_changed(self) -> None:
        """Handle OVT mode toggle between range and vector mode."""
        use_vectors = self.ovt_mode_vector.isChecked()
        self.ovt_range_frame.setVisible(not use_vectors)
        self.ovt_vector_frame.setVisible(use_vectors)
        self.tile_size_group.setVisible(not use_vectors)

    def _update_ovt_tile_table(self) -> None:
        """Update the OVT tiles table when vectors change."""
        x_text = self.ovt_vector_x.text().strip()
        y_text = self.ovt_vector_y.text().strip()

        try:
            x_edges = sorted([float(v.strip()) for v in x_text.split(",") if v.strip()])
            y_edges = sorted([float(v.strip()) for v in y_text.split(",") if v.strip()])
        except ValueError:
            self.ovt_tiles_table.setRowCount(0)
            self.ovt_tiles_count_label.setText("Invalid vector format")
            return

        if len(x_edges) < 2 or len(y_edges) < 2:
            self.ovt_tiles_table.setRowCount(0)
            self.ovt_tiles_count_label.setText("Need at least 2 values in each vector")
            return

        # Generate tiles
        tiles = []
        for ix in range(len(x_edges) - 1):
            for iy in range(len(y_edges) - 1):
                tiles.append((ix, iy, x_edges[ix], x_edges[ix + 1], y_edges[iy], y_edges[iy + 1]))

        # Update table
        self.ovt_tiles_table.setRowCount(len(tiles))
        for row, (tix, tiy, xmin, xmax, ymin, ymax) in enumerate(tiles):
            self.ovt_tiles_table.setItem(row, 0, QTableWidgetItem(str(tix)))
            self.ovt_tiles_table.setItem(row, 1, QTableWidgetItem(str(tiy)))
            self.ovt_tiles_table.setItem(row, 2, QTableWidgetItem(f"{xmin:.0f}"))
            self.ovt_tiles_table.setItem(row, 3, QTableWidgetItem(f"{xmax:.0f}"))
            self.ovt_tiles_table.setItem(row, 4, QTableWidgetItem(f"{ymin:.0f}"))
            self.ovt_tiles_table.setItem(row, 5, QTableWidgetItem(f"{ymax:.0f}"))

        self.ovt_tiles_count_label.setText(f"{len(tiles)} tiles defined ({len(x_edges)-1} x {len(y_edges)-1})")
    
    def _create_custom_tab(self) -> None:
        """Create custom expression tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        vars_info = QLabel(
            "Available variables:\n"
            "  offset, azimuth, offset_x, offset_y\n"
            "  sx, sy, rx, ry, mx, my\n\n"
            "Math/logic: np.abs, np.sqrt, np.sin, np.cos, &, |, ~"
        )
        vars_info.setStyleSheet("color: #888888; background-color: #1a1a1a; padding: 10px; border-radius: 4px;")
        layout.addWidget(vars_info)
        
        layout.addWidget(QLabel("Expression (returns boolean mask):"))
        
        self.custom_expr = QTextEdit()
        self.custom_expr.setMaximumHeight(100)
        self.custom_expr.setPlaceholderText("(offset >= 500) & (offset <= 3000) & (azimuth < 180)")
        layout.addWidget(self.custom_expr)
        
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        
        presets = [
            ("Near", "(offset <= 1500)"),
            ("Mid", "(offset >= 1000) & (offset <= 3000)"),
            ("Far", "(offset >= 2500)"),
            ("North", "(azimuth >= 315) | (azimuth <= 45)"),
        ]
        
        for name, expr in presets:
            btn = QPushButton(name)
            btn.setFixedWidth(80)
            btn.clicked.connect(lambda _, e=expr: self.custom_expr.setPlainText(e))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        layout.addLayout(preset_row)
        
        validate_row = QHBoxLayout()
        self.validate_expr_btn = QPushButton("Validate Expression")
        self.validate_expr_btn.clicked.connect(self._validate_expression)
        validate_row.addWidget(self.validate_expr_btn)
        
        self.expr_status = QLabel("")
        validate_row.addWidget(self.expr_status)
        validate_row.addStretch()
        layout.addLayout(validate_row)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Custom Expression")

    def _create_gather_bins_tab(self) -> None:
        """Create unified gather bins tab for mixed offset+OVT output."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QLabel(
            "Define output gather bins - each bin creates a separate migrated volume.\n"
            "You can mix common offset bins with OVT tiles in any order."
        )
        info.setStyleSheet("color: #4a9eff; background-color: #1a1a1a; padding: 10px; border-radius: 4px;")
        layout.addWidget(info)

        # Gather bins table
        self.gather_bins_table = QTableWidget(0, 7)
        self.gather_bins_table.setHorizontalHeaderLabels([
            "Type", "Name", "Offset Min", "Offset Max", "OVT X Min", "OVT X Max", "OVT Y Range"
        ])
        self.gather_bins_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.gather_bins_table.setMinimumHeight(200)
        layout.addWidget(self.gather_bins_table)

        # Add bin buttons
        btn_row = QHBoxLayout()

        add_offset_btn = QPushButton("+ Add Offset Bin")
        add_offset_btn.clicked.connect(self._add_offset_gather_bin)
        btn_row.addWidget(add_offset_btn)

        add_ovt_btn = QPushButton("+ Add OVT Bin")
        add_ovt_btn.clicked.connect(self._add_ovt_gather_bin)
        btn_row.addWidget(add_ovt_btn)

        remove_btn = QPushButton("- Remove Selected")
        remove_btn.clicked.connect(self._remove_gather_bin)
        btn_row.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(lambda: self.gather_bins_table.setRowCount(0))
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Quick presets
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Quick presets:"))

        preset_near_far = QPushButton("Near/Mid/Far Offsets")
        preset_near_far.clicked.connect(self._preset_near_mid_far)
        preset_row.addWidget(preset_near_far)

        preset_ovt_grid = QPushButton("OVT 3x3 Grid")
        preset_ovt_grid.clicked.connect(self._preset_ovt_grid)
        preset_row.addWidget(preset_ovt_grid)

        preset_mixed = QPushButton("Mixed Example")
        preset_mixed.clicked.connect(self._preset_mixed)
        preset_row.addWidget(preset_mixed)

        preset_row.addStretch()
        layout.addLayout(preset_row)

        # Status
        self.gather_bins_count = QLabel("0 bins defined")
        layout.addWidget(self.gather_bins_count)

        layout.addStretch()
        self.tabs.addTab(tab, "Gather Bins")

    def _add_offset_gather_bin(self) -> None:
        """Add an offset range bin to the table."""
        row = self.gather_bins_table.rowCount()
        self.gather_bins_table.insertRow(row)
        self.gather_bins_table.setItem(row, 0, QTableWidgetItem("offset"))
        self.gather_bins_table.setItem(row, 1, QTableWidgetItem(""))  # name
        self.gather_bins_table.setItem(row, 2, QTableWidgetItem("0"))  # offset min
        self.gather_bins_table.setItem(row, 3, QTableWidgetItem("500"))  # offset max
        self.gather_bins_table.setItem(row, 4, QTableWidgetItem(""))  # ovt x min (unused)
        self.gather_bins_table.setItem(row, 5, QTableWidgetItem(""))  # ovt x max (unused)
        self.gather_bins_table.setItem(row, 6, QTableWidgetItem(""))  # ovt y (unused)
        self._update_gather_bins_count()

    def _add_ovt_gather_bin(self) -> None:
        """Add an OVT tile bin to the table."""
        row = self.gather_bins_table.rowCount()
        self.gather_bins_table.insertRow(row)
        self.gather_bins_table.setItem(row, 0, QTableWidgetItem("ovt"))
        self.gather_bins_table.setItem(row, 1, QTableWidgetItem(""))  # name
        self.gather_bins_table.setItem(row, 2, QTableWidgetItem(""))  # offset min (unused)
        self.gather_bins_table.setItem(row, 3, QTableWidgetItem(""))  # offset max (unused)
        self.gather_bins_table.setItem(row, 4, QTableWidgetItem("-500"))  # ovt x min
        self.gather_bins_table.setItem(row, 5, QTableWidgetItem("500"))  # ovt x max
        self.gather_bins_table.setItem(row, 6, QTableWidgetItem("-500,500"))  # ovt y min,max
        self._update_gather_bins_count()

    def _remove_gather_bin(self) -> None:
        """Remove selected bin from table."""
        current = self.gather_bins_table.currentRow()
        if current >= 0:
            self.gather_bins_table.removeRow(current)
            self._update_gather_bins_count()

    def _update_gather_bins_count(self) -> None:
        """Update the gather bins count label."""
        count = self.gather_bins_table.rowCount()
        self.gather_bins_count.setText(f"{count} bins defined")

    def _preset_near_mid_far(self) -> None:
        """Add near/mid/far offset bins."""
        self.gather_bins_table.setRowCount(0)
        presets = [
            ("offset", "Near", "0", "1500", "", "", ""),
            ("offset", "Mid", "1000", "3000", "", "", ""),
            ("offset", "Far", "2500", "10000", "", "", ""),
        ]
        for row_data in presets:
            row = self.gather_bins_table.rowCount()
            self.gather_bins_table.insertRow(row)
            for col, val in enumerate(row_data):
                self.gather_bins_table.setItem(row, col, QTableWidgetItem(val))
        self._update_gather_bins_count()

    def _preset_ovt_grid(self) -> None:
        """Add 3x3 OVT grid bins."""
        self.gather_bins_table.setRowCount(0)
        edges = [-1500, -500, 500, 1500]
        for ix in range(len(edges) - 1):
            for iy in range(len(edges) - 1):
                row = self.gather_bins_table.rowCount()
                self.gather_bins_table.insertRow(row)
                self.gather_bins_table.setItem(row, 0, QTableWidgetItem("ovt"))
                self.gather_bins_table.setItem(row, 1, QTableWidgetItem(f"OVT_{ix}_{iy}"))
                self.gather_bins_table.setItem(row, 2, QTableWidgetItem(""))
                self.gather_bins_table.setItem(row, 3, QTableWidgetItem(""))
                self.gather_bins_table.setItem(row, 4, QTableWidgetItem(str(edges[ix])))
                self.gather_bins_table.setItem(row, 5, QTableWidgetItem(str(edges[ix + 1])))
                self.gather_bins_table.setItem(row, 6, QTableWidgetItem(f"{edges[iy]},{edges[iy + 1]}"))
        self._update_gather_bins_count()

    def _preset_mixed(self) -> None:
        """Add mixed offset + OVT bins example."""
        self.gather_bins_table.setRowCount(0)
        presets = [
            ("offset", "Near", "0", "1000", "", "", ""),
            ("ovt", "Q1_Near", "", "", "0", "1000", "0,1000"),
            ("ovt", "Q2_Near", "", "", "-1000", "0", "0,1000"),
            ("ovt", "Q3_Near", "", "", "-1000", "0", "-1000,0"),
            ("ovt", "Q4_Near", "", "", "0", "1000", "-1000,0"),
        ]
        for row_data in presets:
            row = self.gather_bins_table.rowCount()
            self.gather_bins_table.insertRow(row)
            for col, val in enumerate(row_data):
                self.gather_bins_table.setItem(row, col, QTableWidgetItem(val))
        self._update_gather_bins_count()

    def _create_summary_section(self) -> None:
        """Create selection summary section."""
        summary_frame, summary_layout = self.create_section("Selection Summary")
        
        summary_grid = QGridLayout()
        
        self.total_traces_label = QLabel("Total traces: --")
        summary_grid.addWidget(self.total_traces_label, 0, 0)
        
        self.selected_traces_label = QLabel("Selected traces: -- (--)")
        summary_grid.addWidget(self.selected_traces_label, 0, 1)
        
        summary_layout.addLayout(summary_grid)
        
        preview_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Selection")
        self.preview_btn.clicked.connect(self._preview_selection)
        preview_row.addWidget(self.preview_btn)
        
        self.export_btn = QPushButton("Export Selected Indices")
        self.export_btn.clicked.connect(self._export_selection)
        preview_row.addWidget(self.export_btn)
        preview_row.addStretch()
        summary_layout.addLayout(preview_row)
        
        self.preview_frame = QFrame()
        self.preview_frame.setMinimumHeight(150)
        self.preview_frame.setStyleSheet("background-color: #1a1a1a; border: 1px solid #3d3d3d; border-radius: 4px;")
        preview_inner = QVBoxLayout(self.preview_frame)
        self.preview_label = QLabel("Selection preview will be shown here")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("color: #666666;")
        preview_inner.addWidget(self.preview_label)
        summary_layout.addWidget(self.preview_frame)
        
        self.content_layout.addWidget(summary_frame)
    
    def _on_mode_changed(self) -> None:
        """Handle selection mode change."""
        checked = self.mode_group.checkedButton()
        if not checked:
            return
        
        key = checked.property("mode_key")
        
        if key == "all":
            self.tabs.setVisible(False)
        else:
            self.tabs.setVisible(True)
            tab_indices = {"offset": 0, "azimuth": 1, "ovt": 2, "custom": 3}
            if key in tab_indices:
                self.tabs.setCurrentIndex(tab_indices[key])
    
    def _validate_expression(self) -> None:
        """Validate custom expression."""
        expr = self.custom_expr.toPlainText().strip()
        if not expr:
            self.expr_status.setText("Empty expression")
            self.expr_status.setStyleSheet("color: #ff9800;")
            return
        
        try:
            compile(expr, "<expression>", "eval")
            self.expr_status.setText("✓ Valid syntax")
            self.expr_status.setStyleSheet("color: #4caf50;")
        except SyntaxError as e:
            self.expr_status.setText(f"✗ Syntax error: {e}")
            self.expr_status.setStyleSheet("color: #f44336;")
    
    def _preview_selection(self) -> None:
        """Preview selection results."""
        self._save_ui_to_state()
        
        mask = self.controller.compute_selection_mask()
        
        if mask is not None:
            sel = self.controller.state.data_selection
            pct = 100 * sel.selected_traces / max(1, sel.total_traces)
            
            self.total_traces_label.setText(f"Total traces: {sel.total_traces:,}")
            self.selected_traces_label.setText(f"Selected traces: {sel.selected_traces:,} ({pct:.1f}%)")
            
            if pct < 10:
                self.selected_traces_label.setStyleSheet("color: #f44336;")
                self.preview_label.setText(f"⚠ Very low selection: {pct:.1f}%")
            elif pct < 50:
                self.selected_traces_label.setStyleSheet("color: #ff9800;")
                self.preview_label.setText(f"Selection preview: {pct:.1f}%")
            else:
                self.selected_traces_label.setStyleSheet("color: #4caf50;")
                self.preview_label.setText(f"✓ Selection: {pct:.1f}%")
    
    def _export_selection(self) -> None:
        """Export selected trace indices."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        
        mask = self.controller.compute_selection_mask()
        if mask is None:
            QMessageBox.warning(self, "Export", "No selection to export")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Selection", "selected_indices.npy",
            "NumPy (*.npy);;CSV (*.csv)"
        )
        
        if path:
            import numpy as np
            indices = np.where(mask)[0]
            if path.endswith(".csv"):
                np.savetxt(path, indices, fmt="%d")
            else:
                np.save(path, indices)
            QMessageBox.information(self, "Export", f"Exported {len(indices)} indices")
    
    def _get_selected_mode(self) -> str:
        checked = self.mode_group.checkedButton()
        return checked.property("mode_key") if checked else "all"
    
    def _set_selected_mode(self, mode: str) -> None:
        for btn in self.mode_group.buttons():
            if btn.property("mode_key") == mode:
                btn.setChecked(True)
                break
    
    def _save_ui_to_state(self) -> None:
        """Save UI values to state."""
        state = self.controller.state.data_selection

        state.mode = self._get_selected_mode()

        # Offset ranges
        ranges = []
        for row in range(self.offset_table.rowCount()):
            min_item = self.offset_table.item(row, 0)
            max_item = self.offset_table.item(row, 1)

            min_val = None
            max_val = None

            if min_item and min_item.text().strip():
                try:
                    min_val = float(min_item.text())
                except ValueError:
                    pass

            if max_item and max_item.text().strip():
                try:
                    max_val = float(max_item.text())
                except ValueError:
                    pass

            if min_val is not None or max_val is not None:
                ranges.append(OffsetRange(min_offset=min_val, max_offset=max_val))

        state.offset_ranges = ranges
        state.offset_include_mode = self.offset_include.isChecked()
        state.include_negative_offsets = self.include_negative.isChecked()

        # OVT
        def parse_float(text: str) -> float | None:
            try:
                return float(text) if text.strip() else None
            except ValueError:
                return None

        state.offset_x_min = parse_float(self.ovt_x_min.text())
        state.offset_x_max = parse_float(self.ovt_x_max.text())
        state.offset_y_min = parse_float(self.ovt_y_min.text())
        state.offset_y_max = parse_float(self.ovt_y_max.text())
        state.ovt_tile_size_x = self.tile_size_x.value()
        state.ovt_tile_size_y = self.tile_size_y.value()

        # OVT vector mode
        state.ovt_use_vectors = self.ovt_mode_vector.isChecked()
        state.ovt_vector_x = self.ovt_vector_x.text().strip()
        state.ovt_vector_y = self.ovt_vector_y.text().strip()
        state.output_gathers_enabled = self.ovt_output_gathers.isChecked()

        # Gather bins (unified offset + OVT)
        gather_bins = []
        for row in range(self.gather_bins_table.rowCount()):
            bin_type = self.gather_bins_table.item(row, 0)
            name = self.gather_bins_table.item(row, 1)
            offset_min = self.gather_bins_table.item(row, 2)
            offset_max = self.gather_bins_table.item(row, 3)
            ovt_x_min = self.gather_bins_table.item(row, 4)
            ovt_x_max = self.gather_bins_table.item(row, 5)
            ovt_y = self.gather_bins_table.item(row, 6)

            if not bin_type:
                continue

            gb = GatherBinState(
                bin_type=bin_type.text().strip() if bin_type else "offset",
                name=name.text().strip() if name else "",
            )

            if gb.bin_type == "offset":
                gb.offset_min = parse_float(offset_min.text()) if offset_min else None
                gb.offset_max = parse_float(offset_max.text()) if offset_max else None
            else:  # ovt
                gb.ovt_x_min = parse_float(ovt_x_min.text()) if ovt_x_min else None
                gb.ovt_x_max = parse_float(ovt_x_max.text()) if ovt_x_max else None
                # Parse Y range as "min,max"
                if ovt_y and ovt_y.text().strip():
                    parts = ovt_y.text().strip().split(",")
                    if len(parts) >= 2:
                        gb.ovt_y_min = parse_float(parts[0])
                        gb.ovt_y_max = parse_float(parts[1])

            gather_bins.append(gb)

        state.gather_bins = gather_bins
        # Auto-enable output gathers if bins are defined
        if gather_bins:
            state.output_gathers_enabled = True

        state.custom_expression = self.custom_expr.toPlainText().strip()

        conv_map = {0: "receiver_relative", 1: "north_0_360", 2: "north_pm180"}
        state.azimuth_convention = conv_map.get(self.azimuth_convention.currentIndex(), "receiver_relative")

        self.controller.state.step_status["data_selection"] = StepStatus.COMPLETE
        self.controller.notify_change()
    
    def _load_state_to_ui(self) -> None:
        """Load state values into UI widgets."""
        state = self.controller.state.data_selection

        self._set_selected_mode(state.mode)

        self.offset_table.setRowCount(max(3, len(state.offset_ranges)))
        for row in range(self.offset_table.rowCount()):
            if row < len(state.offset_ranges):
                r = state.offset_ranges[row]
                self.offset_table.setItem(row, 0, QTableWidgetItem(str(r.min_offset) if r.min_offset else ""))
                self.offset_table.setItem(row, 1, QTableWidgetItem(str(r.max_offset) if r.max_offset else ""))
            else:
                self.offset_table.setItem(row, 0, QTableWidgetItem(""))
                self.offset_table.setItem(row, 1, QTableWidgetItem(""))

        self.offset_include.setChecked(state.offset_include_mode)
        self.include_negative.setChecked(state.include_negative_offsets)

        self.ovt_x_min.setText(str(state.offset_x_min) if state.offset_x_min is not None else "")
        self.ovt_x_max.setText(str(state.offset_x_max) if state.offset_x_max is not None else "")
        self.ovt_y_min.setText(str(state.offset_y_min) if state.offset_y_min is not None else "")
        self.ovt_y_max.setText(str(state.offset_y_max) if state.offset_y_max is not None else "")
        self.tile_size_x.setValue(state.ovt_tile_size_x)
        self.tile_size_y.setValue(state.ovt_tile_size_y)

        # OVT vector mode
        self.ovt_mode_vector.setChecked(state.ovt_use_vectors)
        self.ovt_mode_range.setChecked(not state.ovt_use_vectors)
        self.ovt_vector_x.setText(state.ovt_vector_x)
        self.ovt_vector_y.setText(state.ovt_vector_y)
        self.ovt_output_gathers.setChecked(state.output_gathers_enabled)
        self._on_ovt_mode_changed()
        self._update_ovt_tile_table()

        # Gather bins (unified offset + OVT)
        self.gather_bins_table.setRowCount(len(state.gather_bins))
        for row, gb in enumerate(state.gather_bins):
            self.gather_bins_table.setItem(row, 0, QTableWidgetItem(gb.bin_type))
            self.gather_bins_table.setItem(row, 1, QTableWidgetItem(gb.name or ""))
            if gb.bin_type == "offset":
                self.gather_bins_table.setItem(row, 2, QTableWidgetItem(str(gb.offset_min) if gb.offset_min is not None else ""))
                self.gather_bins_table.setItem(row, 3, QTableWidgetItem(str(gb.offset_max) if gb.offset_max is not None else ""))
                self.gather_bins_table.setItem(row, 4, QTableWidgetItem(""))
                self.gather_bins_table.setItem(row, 5, QTableWidgetItem(""))
                self.gather_bins_table.setItem(row, 6, QTableWidgetItem(""))
            else:  # ovt
                self.gather_bins_table.setItem(row, 2, QTableWidgetItem(""))
                self.gather_bins_table.setItem(row, 3, QTableWidgetItem(""))
                self.gather_bins_table.setItem(row, 4, QTableWidgetItem(str(gb.ovt_x_min) if gb.ovt_x_min is not None else ""))
                self.gather_bins_table.setItem(row, 5, QTableWidgetItem(str(gb.ovt_x_max) if gb.ovt_x_max is not None else ""))
                y_range = f"{gb.ovt_y_min},{gb.ovt_y_max}" if gb.ovt_y_min is not None else ""
                self.gather_bins_table.setItem(row, 6, QTableWidgetItem(y_range))
        self._update_gather_bins_count()

        self.custom_expr.setPlainText(state.custom_expression)

        conv_map = {"receiver_relative": 0, "north_0_360": 1, "north_pm180": 2}
        self.azimuth_convention.setCurrentIndex(conv_map.get(state.azimuth_convention, 0))

        self._on_mode_changed()
    
    def on_enter(self) -> None:
        self._load_state_to_ui()
        
        inp = self.controller.state.input_data
        if inp.is_loaded:
            self.total_traces_label.setText(f"Total traces: {inp.n_traces:,}")
    
    def on_leave(self) -> None:
        self._save_ui_to_state()
    
    def validate(self) -> bool:
        mode = self._get_selected_mode()
        if mode != "all":
            return self.ack_checkbox.isChecked()
        return True

    def refresh_from_state(self) -> None:
        """Refresh UI from loaded state."""
        self._load_state_to_ui()
