"""
Step 2: Survey Geometry

Analyze and display survey geometry from loaded headers.
Provides QC visualization of source and receiver locations.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QGridLayout, QCheckBox, QComboBox,
)
from PyQt6.QtCore import Qt

from pstm.gui.steps.base import WizardStepWidget
from pstm.gui.state import StepStatus


class SurveyStep(WizardStepWidget):
    """Step 2: Survey Geometry - Analyze input geometry."""

    @property
    def title(self) -> str:
        return "Survey Geometry"

    def _setup_ui(self) -> None:
        # Survey Statistics Section
        stats_frame, stats_layout = self.create_section("Survey Statistics")

        grid = QGridLayout()
        grid.setSpacing(10)

        labels = [
            ("X Range:", "x_range"),
            ("Y Range:", "y_range"),
            ("Offset Range:", "offset_range"),
            ("Mean Offset:", "offset_mean"),
            ("Unique Sources:", "n_sources"),
            ("Unique Receivers:", "n_receivers"),
            ("Number of Traces:", "n_traces"),
            ("Azimuth Range:", "azimuth_range"),
        ]

        self.stat_labels = {}
        for i, (text, key) in enumerate(labels):
            row, col = divmod(i, 2)
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #888888; border: none; background: transparent;")
            grid.addWidget(lbl, row, col * 2)

            val = QLabel("--")
            val.setStyleSheet("color: #ffffff; border: none; background: transparent;")
            self.stat_labels[key] = val
            grid.addWidget(val, row, col * 2 + 1)

        stats_layout.addLayout(grid)
        self.content_layout.addWidget(stats_frame)

        # Geometry Map Section
        map_frame, map_layout = self.create_section("Source/Receiver Geometry")

        # Controls row
        ctrl_row = QHBoxLayout()

        self.show_sources = QCheckBox("Sources")
        self.show_sources.setChecked(True)
        self.show_sources.setStyleSheet("color: #cccccc; background: transparent;")
        self.show_sources.stateChanged.connect(self._update_plot)
        ctrl_row.addWidget(self.show_sources)

        self.show_receivers = QCheckBox("Receivers")
        self.show_receivers.setChecked(True)
        self.show_receivers.setStyleSheet("color: #cccccc; background: transparent;")
        self.show_receivers.stateChanged.connect(self._update_plot)
        ctrl_row.addWidget(self.show_receivers)

        self.show_midpoints = QCheckBox("Midpoints")
        self.show_midpoints.setChecked(False)
        self.show_midpoints.setStyleSheet("color: #cccccc; background: transparent;")
        self.show_midpoints.stateChanged.connect(self._update_plot)
        ctrl_row.addWidget(self.show_midpoints)

        ctrl_row.addStretch()

        # Decimation for large datasets
        ctrl_row.addWidget(QLabel("Decimation:"))
        self.decimation_combo = QComboBox()
        self.decimation_combo.addItems(["1 (all)", "2", "5", "10", "20", "50"])
        self.decimation_combo.setCurrentIndex(0)
        self.decimation_combo.currentIndexChanged.connect(self._update_plot)
        ctrl_row.addWidget(self.decimation_combo)

        map_layout.addLayout(ctrl_row)

        # Plot frame
        self.plot_frame = QFrame()
        self.plot_frame.setMinimumHeight(450)
        self.plot_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        plot_layout = QVBoxLayout(self.plot_frame)
        self.plot_label = QLabel("Click 'Analyze & Plot' to visualize geometry")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setStyleSheet("color: #666666; border: none; background: transparent;")
        plot_layout.addWidget(self.plot_label)
        map_layout.addWidget(self.plot_frame)

        # Analyze button
        btn_row = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze && Plot")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
            }
        """)
        self.analyze_btn.clicked.connect(self._analyze_and_plot)
        btn_row.addWidget(self.analyze_btn)
        btn_row.addStretch()
        map_layout.addLayout(btn_row)

        self.content_layout.addWidget(map_frame)
        self.content_layout.addStretch()

        # Store geometry data for replotting
        self._sources = None
        self._receivers = None
        self._midpoints = None
        self._canvas = None

    def _analyze_and_plot(self) -> None:
        """Analyze survey and plot geometry."""
        success, message = self.controller.analyze_survey_geometry()

        if success:
            self._update_statistics()
            self._extract_geometry()
            self._update_plot()
        else:
            self.plot_label.setText(f"Error: {message}")
            self.plot_label.setStyleSheet("color: #f44336; border: none; background: transparent;")

    def _update_statistics(self) -> None:
        """Update statistics display from state."""
        state = self.controller.state.survey
        inp = self.controller.state.input_data

        self.stat_labels["x_range"].setText(f"{state.x_min:.1f} - {state.x_max:.1f} m")
        self.stat_labels["y_range"].setText(f"{state.y_min:.1f} - {state.y_max:.1f} m")
        self.stat_labels["offset_range"].setText(f"{state.offset_min:.1f} - {state.offset_max:.1f} m")
        self.stat_labels["offset_mean"].setText(f"{state.offset_mean:.1f} m")
        self.stat_labels["n_traces"].setText(f"{inp.n_traces:,}")

        # Azimuth range if available
        if hasattr(state, 'azimuth_min') and hasattr(state, 'azimuth_max'):
            self.stat_labels["azimuth_range"].setText(f"{state.azimuth_min:.1f} - {state.azimuth_max:.1f} deg")
        else:
            self.stat_labels["azimuth_range"].setText("--")

    def _extract_geometry(self) -> None:
        """Extract unique source, receiver, and midpoint locations."""
        import numpy as np

        df = self.controller.headers_df
        if df is None:
            return

        inp = self.controller.state.input_data

        # Get column names
        sx_col = inp.col_source_x or "SOU_X"
        sy_col = inp.col_source_y or "SOU_Y"
        rx_col = inp.col_receiver_x or "REC_X"
        ry_col = inp.col_receiver_y or "REC_Y"
        mx_col = inp.col_midpoint_x or "CDP_X"
        my_col = inp.col_midpoint_y or "CDP_Y"

        # Helper to apply coordinate scalar
        def apply_scalar(values):
            return self.controller._apply_coord_scalar(values.astype(np.float64))

        # Extract unique sources
        if sx_col in df.columns and sy_col in df.columns:
            sources = df[[sx_col, sy_col]].drop_duplicates()
            self._sources = (
                apply_scalar(sources[sx_col].values),
                apply_scalar(sources[sy_col].values)
            )
            self.stat_labels["n_sources"].setText(f"{len(sources):,}")
        else:
            self._sources = None
            self.stat_labels["n_sources"].setText("N/A")

        # Extract unique receivers
        if rx_col in df.columns and ry_col in df.columns:
            receivers = df[[rx_col, ry_col]].drop_duplicates()
            self._receivers = (
                apply_scalar(receivers[rx_col].values),
                apply_scalar(receivers[ry_col].values)
            )
            self.stat_labels["n_receivers"].setText(f"{len(receivers):,}")
        else:
            self._receivers = None
            self.stat_labels["n_receivers"].setText("N/A")

        # Extract unique midpoints (can be large - just sample)
        if mx_col in df.columns and my_col in df.columns:
            midpoints = df[[mx_col, my_col]].drop_duplicates()
            self._midpoints = (
                apply_scalar(midpoints[mx_col].values),
                apply_scalar(midpoints[my_col].values)
            )
        else:
            self._midpoints = None

    def _update_plot(self) -> None:
        """Update the geometry plot."""
        if self._sources is None and self._receivers is None:
            return

        try:
            import matplotlib
            matplotlib.use('QtAgg')
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            import numpy as np

            # Clear previous plot
            layout = self.plot_frame.layout()
            for i in reversed(range(layout.count())):
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()

            # Get decimation factor
            dec_text = self.decimation_combo.currentText()
            dec = int(dec_text.split()[0])

            # Create figure with dark background
            fig = Figure(figsize=(6, 5), dpi=100, facecolor='#1a1a1a')
            self._canvas = FigureCanvasQTAgg(fig)

            ax = fig.add_subplot(111, facecolor='#1a1a1a')

            # Plot survey extent box
            state = self.controller.state.survey
            if state.x_max > state.x_min:
                extent_x = [state.x_min, state.x_max, state.x_max, state.x_min, state.x_min]
                extent_y = [state.y_min, state.y_min, state.y_max, state.y_max, state.y_min]
                ax.plot(extent_x, extent_y, 'w--', linewidth=1, alpha=0.5, label='Survey Extent')

            # Plot sources
            if self.show_sources.isChecked() and self._sources is not None:
                sx, sy = self._sources
                ax.scatter(sx[::dec], sy[::dec], c='#ff6b6b', marker='^', s=30,
                          alpha=0.7, label=f'Sources ({len(sx):,})', edgecolors='white', linewidths=0.5)

            # Plot receivers
            if self.show_receivers.isChecked() and self._receivers is not None:
                rx, ry = self._receivers
                ax.scatter(rx[::dec], ry[::dec], c='#4ecdc4', marker='o', s=20,
                          alpha=0.7, label=f'Receivers ({len(rx):,})', edgecolors='white', linewidths=0.3)

            # Plot midpoints
            if self.show_midpoints.isChecked() and self._midpoints is not None:
                mx, my = self._midpoints
                ax.scatter(mx[::dec], my[::dec], c='#ffe66d', marker='.', s=5,
                          alpha=0.3, label=f'Midpoints ({len(mx):,})')

            # Style the plot
            ax.set_xlabel('X (m)', color='white')
            ax.set_ylabel('Y (m)', color='white')
            ax.set_title('Survey Geometry', color='white', fontsize=12)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#3d3d3d')
            ax.grid(True, alpha=0.2, color='white')
            ax.set_aspect('equal')

            # Legend with dark background
            legend = ax.legend(loc='upper right', fontsize=8, facecolor='#2d2d2d',
                             edgecolor='#3d3d3d', labelcolor='white')

            fig.tight_layout()

            layout.addWidget(self._canvas)

        except ImportError as e:
            self.plot_label = QLabel(f"Matplotlib not available: {e}")
            self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.plot_label.setStyleSheet("color: #f44336; border: none; background: transparent;")
            self.plot_frame.layout().addWidget(self.plot_label)

    def on_enter(self) -> None:
        """Called when navigating to this step."""
        # Auto-analyze if data is loaded but survey not analyzed
        if self.controller.state.input_data.is_loaded:
            state = self.controller.state.survey
            if state.x_max == 0 and state.y_max == 0:
                self._analyze_and_plot()
            else:
                # Just refresh display
                self._update_statistics()
                if self._sources is None:
                    self._extract_geometry()
                    self._update_plot()

    def validate(self) -> bool:
        """Validate this step is complete."""
        state = self.controller.state.survey
        return state.x_max > state.x_min and state.y_max > state.y_min

    def refresh_from_state(self) -> None:
        """Refresh UI from state after loading."""
        if self.controller.state.input_data.is_loaded:
            state = self.controller.state.survey
            if state.x_max > state.x_min:
                self._update_statistics()
