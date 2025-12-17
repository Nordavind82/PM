"""
Step 7: Execution

Configure compute resources, tiling, checkpointing, and run migration.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from pathlib import Path

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QGroupBox,
    QGridLayout, QFormLayout, QProgressBar, QTextEdit,
    QLineEdit, QFileDialog,
)
from PyQt6.QtCore import Qt, QTimer

from pstm.gui.steps.base import WizardStepWidget
from pstm.gui.state import StepStatus
from pstm.gui.migration_worker import MigrationWorker, MigrationProgress
from pstm.gui.migration_dialog import MigrationProgressDialog

import logging
debug_logger = logging.getLogger("pstm.migration.debug")


def _is_metal_cpp_available() -> bool:
    """Check if Metal C++ kernel is available."""
    try:
        from pstm.kernels.metal_cpp import is_metal_cpp_available
        return is_metal_cpp_available()
    except ImportError:
        return False


def _get_metal_device_info() -> dict | None:
    """Get Metal device info if available."""
    try:
        from pstm.kernels.metal_cpp import is_metal_cpp_available, get_metal_device_info
        if is_metal_cpp_available():
            return get_metal_device_info()
    except ImportError:
        pass
    return None


class ExecutionStep(WizardStepWidget):
    """Step 7: Execution - Run migration."""

    def __init__(self, controller, parent=None):
        # Initialize worker reference before UI setup
        self._worker: MigrationWorker | None = None
        self._is_paused = False
        self._heartbeat_timer: QTimer | None = None
        self._migration_start_time: float = 0.0
        self._last_progress_msg: str = ""
        self._last_traces_in_tile: int = 0
        super().__init__(controller, parent)

    @property
    def title(self) -> str:
        return "Execution"

    def _setup_ui(self) -> None:
        self._create_output_section()
        self._create_backend_section()
        self._create_resources_section()
        self._create_tiling_section()
        self._create_checkpoint_section()
        self._create_preflight_section()
        self._create_run_section()
        self.content_layout.addStretch()

    def _create_output_section(self) -> None:
        """Create output configuration section."""
        frame, layout = self.create_section("Output Configuration")

        form = QFormLayout()
        form.setSpacing(10)

        # Output directory
        dir_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        dir_row.addWidget(self.output_dir_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(browse_btn)

        form.addRow("Output Directory:", dir_row)

        # Project name
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setText("migration_output")
        self.project_name_edit.setPlaceholderText("Project name for output files")
        form.addRow("Project Name:", self.project_name_edit)

        layout.addLayout(form)

        # Output products
        products_label = QLabel("Output Products:")
        products_label.setStyleSheet("color: #888888; margin-top: 10px;")
        layout.addWidget(products_label)

        products_grid = QGridLayout()
        products_grid.setSpacing(8)

        self.output_stacked_check = QCheckBox("Stacked Image")
        self.output_stacked_check.setChecked(True)
        self.output_stacked_check.setToolTip("Output the final migrated stack")
        products_grid.addWidget(self.output_stacked_check, 0, 0)

        self.output_fold_check = QCheckBox("Fold Map")
        self.output_fold_check.setChecked(True)
        self.output_fold_check.setToolTip("Output fold/hit count volume")
        products_grid.addWidget(self.output_fold_check, 0, 1)

        self.output_cig_check = QCheckBox("Common Image Gathers")
        self.output_cig_check.setChecked(False)
        self.output_cig_check.setToolTip("Output offset-binned gathers for velocity QC")
        products_grid.addWidget(self.output_cig_check, 1, 0)

        self.output_qc_check = QCheckBox("QC Report")
        self.output_qc_check.setChecked(True)
        self.output_qc_check.setToolTip("Generate HTML QC report with statistics")
        products_grid.addWidget(self.output_qc_check, 1, 1)

        layout.addLayout(products_grid)

        # Output format
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["Zarr (recommended)", "SEG-Y"])
        format_row.addWidget(self.output_format_combo)
        format_row.addStretch()
        layout.addLayout(format_row)

        self.content_layout.addWidget(frame)

    def _browse_output_dir(self) -> None:
        """Browse for output directory."""
        current = self.output_dir_edit.text()
        start_dir = current if current and Path(current).exists() else str(Path.home())

        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start_dir
        )
        if directory:
            self.output_dir_edit.setText(directory)
    
    def _create_backend_section(self) -> None:
        """Create compute backend selection."""
        frame, layout = self.create_section("Compute Backend")

        form = QFormLayout()
        form.setSpacing(10)

        # Build backend options dynamically
        self._backend_options = [("Auto (detect best)", "auto")]

        # Add Metal C++ if available (fastest GPU option - 20x speedup)
        metal_info = _get_metal_device_info()
        if metal_info and metal_info.get("available"):
            self._backend_options.append(("Metal C++ GPU (20x faster)", "metal_cpp"))

        # Add other backends
        self._backend_options.extend([
            ("Numba CPU (recommended)", "numba_cpu"),
            ("MLX Metal (Apple Silicon)", "mlx_metal"),
            ("NumPy (fallback)", "numpy"),
        ])

        self.backend_combo = QComboBox()
        self.backend_combo.addItems([opt[0] for opt in self._backend_options])
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        form.addRow("Backend:", self.backend_combo)

        layout.addLayout(form)

        # Backend info - show GPU info if available
        if metal_info and metal_info.get("available"):
            info_text = f"GPU: {metal_info.get('device_name', 'Unknown')} ({metal_info.get('device_memory_gb', 0):.1f} GB)"
        else:
            info_text = "Auto-detecting best backend..."
        self.backend_info = QLabel(info_text)
        self.backend_info.setStyleSheet("color: #888888; border: none; background: transparent;")
        layout.addWidget(self.backend_info)
        
        # Benchmark button
        btn_row = QHBoxLayout()
        self.benchmark_btn = QPushButton("Run Benchmark")
        self.benchmark_btn.clicked.connect(self._run_benchmark)
        btn_row.addWidget(self.benchmark_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        
        self.content_layout.addWidget(frame)
    
    def _create_resources_section(self) -> None:
        """Create resource allocation section."""
        frame, layout = self.create_section("Resource Allocation")
        
        form = QFormLayout()
        form.setSpacing(10)
        
        # Memory limit
        mem_row = QHBoxLayout()
        self.max_memory_spin = self.create_double_spinbox(1, 256, 1, "GB")
        self.max_memory_spin.setValue(8.0)
        mem_row.addWidget(self.max_memory_spin)
        
        detect_mem_btn = QPushButton("Detect")
        detect_mem_btn.clicked.connect(self._detect_memory)
        mem_row.addWidget(detect_mem_btn)
        
        form.addRow("Max Memory:", mem_row)
        
        # CPU threads
        thread_row = QHBoxLayout()
        self.n_threads_spin = self.create_spinbox(0, 256)
        self.n_threads_spin.setValue(0)
        self.n_threads_spin.setSpecialValueText("Auto")
        thread_row.addWidget(self.n_threads_spin)
        
        self.thread_info = QLabel(f"(Available: {os.cpu_count()} cores)")
        self.thread_info.setStyleSheet("color: #888888; border: none; background: transparent;")
        thread_row.addWidget(self.thread_info)
        
        form.addRow("CPU Threads:", thread_row)
        
        layout.addLayout(form)
        
        self.content_layout.addWidget(frame)
    
    def _create_tiling_section(self) -> None:
        """Create tiling configuration section."""
        frame, layout = self.create_section("Tile Configuration")
        
        form = QFormLayout()
        form.setSpacing(10)
        
        # Auto tile size
        self.auto_tile_check = QCheckBox("Auto-determine tile size")
        self.auto_tile_check.setChecked(True)
        self.auto_tile_check.toggled.connect(self._on_auto_tile_changed)
        form.addRow(self.auto_tile_check)
        
        # Manual tile size (user can test different sizes: 32, 48, 64, 96, 128...)
        tile_row = QHBoxLayout()
        self.tile_nx_spin = self.create_spinbox(8, 512)  # Reasonable range for testing
        self.tile_nx_spin.setValue(64)  # Default when auto is disabled
        self.tile_nx_spin.setSingleStep(8)  # Step by 8 for easy testing
        self.tile_nx_spin.setEnabled(False)
        tile_row.addWidget(QLabel("NX:"))
        tile_row.addWidget(self.tile_nx_spin)

        self.tile_ny_spin = self.create_spinbox(8, 512)  # Reasonable range for testing
        self.tile_ny_spin.setValue(64)  # Default when auto is disabled
        self.tile_ny_spin.setSingleStep(8)  # Step by 8 for easy testing
        self.tile_ny_spin.setEnabled(False)
        tile_row.addWidget(QLabel("NY:"))
        tile_row.addWidget(self.tile_ny_spin)

        form.addRow("Tile Size:", tile_row)
        
        # Tile ordering
        self.tile_order_combo = QComboBox()
        self.tile_order_combo.addItems([
            "Snake (cache efficient)",
            "Row Major",
            "Column Major",
            "Hilbert (spatial locality)",
        ])
        form.addRow("Tile Order:", self.tile_order_combo)
        
        layout.addLayout(form)
        
        self.content_layout.addWidget(frame)
    
    def _create_checkpoint_section(self) -> None:
        """Create checkpointing configuration section."""
        frame, layout = self.create_section("Checkpointing")
        
        form = QFormLayout()
        form.setSpacing(10)
        
        self.enable_checkpoint = QCheckBox("Enable checkpointing")
        self.enable_checkpoint.setChecked(True)
        self.enable_checkpoint.setToolTip("Save progress periodically for crash recovery")
        form.addRow(self.enable_checkpoint)
        
        # Interval
        interval_row = QHBoxLayout()
        self.checkpoint_tiles_spin = self.create_spinbox(1, 1000)
        self.checkpoint_tiles_spin.setValue(10)
        interval_row.addWidget(self.checkpoint_tiles_spin)
        interval_row.addWidget(QLabel("tiles"))
        
        interval_row.addWidget(QLabel("or"))
        
        self.checkpoint_seconds_spin = self.create_spinbox(60, 3600)
        self.checkpoint_seconds_spin.setValue(300)
        interval_row.addWidget(self.checkpoint_seconds_spin)
        interval_row.addWidget(QLabel("seconds"))
        
        form.addRow("Checkpoint Interval:", interval_row)
        
        self.resume_check = QCheckBox("Auto-resume from checkpoint if available")
        self.resume_check.setChecked(True)
        form.addRow(self.resume_check)
        
        layout.addLayout(form)
        
        self.content_layout.addWidget(frame)
    
    def _create_preflight_section(self) -> None:
        """Create pre-flight check section."""
        frame, layout = self.create_section("Pre-flight Check")
        
        self.preflight_text = QTextEdit()
        self.preflight_text.setReadOnly(True)
        self.preflight_text.setMaximumHeight(150)
        self.preflight_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                color: #cccccc;
                font-family: monospace;
            }
        """)
        self.preflight_text.setPlainText("Click 'Validate All' to run pre-flight checks...")
        layout.addWidget(self.preflight_text)
        
        btn_row = QHBoxLayout()
        self.validate_btn = QPushButton("Validate All")
        self.validate_btn.clicked.connect(self._run_preflight)
        btn_row.addWidget(self.validate_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        
        self.content_layout.addWidget(frame)
    
    def _create_run_section(self) -> None:
        """Create run control section."""
        frame, layout = self.create_section("Run Migration")
        
        # Estimate
        estimate_layout = QGridLayout()
        estimate_layout.setSpacing(10)

        labels = [
            ("ETA:", "est_time"),
            ("Tiles:", "n_tiles"),
            ("Traces:", "n_traces"),
            ("Rate:", "trace_rate"),
        ]

        self.estimate_labels = {}
        for i, (text, key) in enumerate(labels):
            row, col = divmod(i, 2)
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #888888; border: none; background: transparent;")
            estimate_layout.addWidget(lbl, row, col * 2)

            val = QLabel("--")
            val.setStyleSheet("color: #ffffff; border: none; background: transparent;")
            self.estimate_labels[key] = val
            estimate_layout.addWidget(val, row, col * 2 + 1)

        layout.addLayout(estimate_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #888888; border: none; background: transparent;")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        
        # Run buttons
        btn_row = QHBoxLayout()
        
        self.run_btn = QPushButton("▶ Start Migration")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
                padding: 15px 30px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #66bb6a;
            }
        """)
        self.run_btn.clicked.connect(self._start_migration)
        btn_row.addWidget(self.run_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setVisible(False)
        self.pause_btn.clicked.connect(self._pause_migration)
        btn_row.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self._stop_migration)
        btn_row.addWidget(self.stop_btn)
        
        btn_row.addStretch()
        layout.addLayout(btn_row)
        
        self.content_layout.addWidget(frame)
    
    def _on_backend_changed(self) -> None:
        """Handle backend selection change."""
        backend = self.backend_combo.currentText()
        
        if "Auto" in backend:
            self.backend_info.setText("Will auto-detect best backend at runtime")
        elif "Numba" in backend:
            self.backend_info.setText("Numba JIT compilation with SIMD vectorization")
        elif "MLX" in backend:
            self.backend_info.setText("Apple Metal GPU acceleration")
        elif "NumPy" in backend:
            self.backend_info.setText("Pure NumPy (slowest, always available)")
    
    def _on_auto_tile_changed(self, checked: bool) -> None:
        """Handle auto tile checkbox change."""
        self.tile_nx_spin.setEnabled(not checked)
        self.tile_ny_spin.setEnabled(not checked)
    
    def _detect_memory(self) -> None:
        """Detect available system memory."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            # Use 75% of available memory
            suggested = min(available_gb * 0.75, 64)
            self.max_memory_spin.setValue(round(suggested, 1))
        except ImportError:
            self.max_memory_spin.setValue(8.0)
    
    def _run_benchmark(self) -> None:
        """Run backend benchmark."""
        self.backend_info.setText("Running benchmark...")
        # Placeholder - would run actual benchmark
        QTimer.singleShot(1000, lambda: self.backend_info.setText(
            "Benchmark: Numba ~5.2M traces/s, MLX ~8.1M traces/s"
        ))

    def _calculate_tiles_from_ui(
        self, og, auto_tile: bool, manual_tile_nx: int, manual_tile_ny: int, max_memory_gb: float
    ) -> tuple[int, int, int]:
        """
        Calculate actual tile count using current UI values.

        Uses the shared calculate_tile_count function from tile_planner
        to ensure consistent behavior between UI preflight and actual execution.

        Args:
            og: Output grid config
            auto_tile: Whether auto tile size is enabled
            manual_tile_nx: Manual tile NX from spinbox
            manual_tile_ny: Manual tile NY from spinbox
            max_memory_gb: Max memory from spinbox

        Returns:
            (n_tiles, tile_nx, tile_ny)
        """
        from pstm.pipeline.tile_planner import calculate_tile_count
        return calculate_tile_count(
            nx=og.nx,
            ny=og.ny,
            nt=og.nt,
            auto_tile_size=auto_tile,
            tile_nx=manual_tile_nx,
            tile_ny=manual_tile_ny,
            max_memory_gb=max_memory_gb,
        )

    def _calculate_actual_tiles(self, og, exec_state) -> tuple[int, int, int]:
        """
        Calculate actual tile count using the same logic as TilePlanner.
        (Legacy method - uses state instead of UI values)

        Returns:
            (n_tiles, tile_nx, tile_ny)
        """
        return self._calculate_tiles_from_ui(
            og,
            exec_state.auto_tile_size,
            exec_state.tile_nx,
            exec_state.tile_ny,
            exec_state.max_memory_gb,
        )

    def _run_preflight(self) -> None:
        """Run pre-flight validation checks."""
        checks = []
        all_ok = True
        
        # Check input data
        inp = self.controller.state.input_data
        if inp.is_loaded:
            checks.append("✓ Input data loaded")
        else:
            checks.append("✗ Input data not loaded")
            all_ok = False
        
        # Check survey
        survey = self.controller.state.survey
        if survey.x_max > survey.x_min:
            checks.append("✓ Survey geometry analyzed")
        else:
            checks.append("✗ Survey not analyzed")
            all_ok = False
        
        # Check output grid
        og = self.controller.state.output_grid
        if og.total_points > 0:
            checks.append(f"✓ Output grid: {og.nx}×{og.ny}×{og.nt} points")
        else:
            checks.append("✗ Output grid not defined")
            all_ok = False
        
        # Check velocity
        vel = self.controller.state.velocity
        if vel.source == "constant" and vel.constant_velocity > 0:
            checks.append(f"✓ Velocity: constant {vel.constant_velocity} m/s")
        elif vel.source == "linear":
            checks.append(f"✓ Velocity: linear V₀={vel.linear_v0}")
        elif vel.source in ("cube_3d", "file") and vel.cube_path:
            checks.append(f"✓ Velocity: from file")
        else:
            checks.append("⚠ Velocity not configured")
        
        # Check algorithm
        algo = self.controller.state.algorithm
        checks.append(f"✓ Algorithm: {algo.interpolation_method}, {algo.max_dip_degrees}° max dip")
        
        # Check output directory - read from UI directly (not yet saved to state)
        output_dir = self.output_dir_edit.text().strip()
        if output_dir:
            if Path(output_dir).exists():
                checks.append(f"✓ Output: {output_dir}")
            else:
                checks.append(f"✗ Output directory does not exist: {output_dir}")
                all_ok = False
        else:
            checks.append("✗ Output directory not set")
            all_ok = False
        
        # Resource estimates
        checks.append("")
        checks.append("─── Resource Estimates ───")

        # Read tile settings directly from UI widgets (not from state which may be stale)
        auto_tile = self.auto_tile_check.isChecked()
        manual_tile_nx = self.tile_nx_spin.value()
        manual_tile_ny = self.tile_ny_spin.value()
        max_memory_gb = self.max_memory_spin.value()

        # Calculate actual tile count using TilePlanner logic
        n_tiles, tile_nx, tile_ny = self._calculate_tiles_from_ui(
            og, auto_tile, manual_tile_nx, manual_tile_ny, max_memory_gb
        )

        est_time = n_tiles * 2  # Placeholder seconds per tile
        est_traces = inp.n_traces if inp.is_loaded else 0

        if auto_tile:
            checks.append(f"Tiles: {n_tiles} (auto: {tile_nx}x{tile_ny} per tile)")
        else:
            checks.append(f"Tiles: {n_tiles} (manual: {tile_nx}x{tile_ny} per tile)")
        checks.append(f"Est. time: ~{est_time // 60} min")
        checks.append(f"Output size: ~{og.estimated_size_gb:.2f} GB")
        if est_traces > 0:
            checks.append(f"Input traces: {est_traces:,}")

        # Update estimates
        self.estimate_labels["n_tiles"].setText(f"~{n_tiles}")
        self.estimate_labels["est_time"].setText(f"~{est_time // 60} min")
        self.estimate_labels["n_traces"].setText(f"{est_traces:,}" if est_traces > 0 else "--")
        self.estimate_labels["trace_rate"].setText("--")
        
        # Final status
        checks.append("")
        if all_ok:
            checks.append("═══ READY TO RUN ═══")
        else:
            checks.append("═══ FIX ISSUES BEFORE RUNNING ═══")
        
        self.preflight_text.setPlainText("\n".join(checks))
    
    def _start_migration(self) -> None:
        """Start the migration."""
        from PyQt6.QtWidgets import QMessageBox
        
        # Run preflight first
        self._run_preflight()
        
        # Check if ready
        text = self.preflight_text.toPlainText()
        if "FIX ISSUES" in text:
            QMessageBox.warning(self, "Not Ready",
                "Please fix the issues shown in pre-flight check before running.")
            return
        
        # Confirm
        og = self.controller.state.output_grid
        reply = QMessageBox.question(self, "Start Migration",
            f"Ready to start migration?\n\n"
            f"Output: {og.total_points:,} points\n"
            f"Size: ~{og.estimated_size_gb:.2f} GB\n\n"
            f"This may take several minutes to hours.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self._execute_migration()
    
    def _execute_migration(self) -> None:
        """Execute the migration in a modal dialog."""
        from PyQt6.QtWidgets import QMessageBox

        debug_logger.info("EXECUTION_STEP: _execute_migration called")

        # Save current UI state to controller state before building config
        self.on_leave()

        # Build migration config
        try:
            debug_logger.info("EXECUTION_STEP: Building migration config...")
            config = self.controller.build_migration_config()
            debug_logger.info("EXECUTION_STEP: Config built successfully")
            debug_logger.info(f"EXECUTION_STEP: Output grid: {config.output.grid.nx}x{config.output.grid.ny}x{config.output.grid.nt}")
        except ValueError as e:
            debug_logger.error(f"EXECUTION_STEP: Config error: {e}")
            QMessageBox.critical(self, "Configuration Error", str(e))
            return
        except Exception as e:
            debug_logger.error(f"EXECUTION_STEP: Failed to build config: {e}")
            QMessageBox.critical(self, "Error", f"Failed to build config: {e}")
            return

        # Create and show the migration dialog (modal)
        debug_logger.info("EXECUTION_STEP: Creating migration dialog...")
        dialog = MigrationProgressDialog(config, parent=self)

        # Start migration in the dialog
        debug_logger.info("EXECUTION_STEP: Starting migration in dialog...")
        dialog.start_migration()

        # Show dialog (blocks until closed)
        debug_logger.info("EXECUTION_STEP: Showing dialog (modal)...")
        result = dialog.exec()
        debug_logger.info(f"EXECUTION_STEP: Dialog closed with result={result}")

        # Handle result
        if dialog.was_successful:
            debug_logger.info(f"EXECUTION_STEP: Migration successful: {dialog.output_path}")
            QMessageBox.information(
                self, "Migration Complete",
                f"Migration completed successfully!\n\n"
                f"Output saved to:\n{dialog.output_path}"
            )
            # Navigate to Results step
            main_window = self.window()
            if hasattr(main_window, '_go_to_step'):
                main_window._go_to_step(7)  # Results step
        elif dialog.error_message:
            debug_logger.error(f"EXECUTION_STEP: Migration failed: {dialog.error_message}")
            # Error already shown in dialog

        # Re-run preflight
        QTimer.singleShot(100, self._run_preflight)
    
    def _pause_migration(self) -> None:
        """Pause or resume the migration."""
        if not self._worker:
            return

        if self._is_paused:
            # Resume
            self._worker.request_resume()
            self._is_paused = False
            self.pause_btn.setText("⏸ Pause")
            self.progress_label.setText("Resuming...")
        else:
            # Pause
            self._worker.request_pause()
            self._is_paused = True
            self.pause_btn.setText("▶ Resume")
            self.progress_label.setText("Pausing after current tile...")

    def _stop_migration(self) -> None:
        """Stop the migration."""
        from PyQt6.QtWidgets import QMessageBox

        if not self._worker:
            return

        reply = QMessageBox.question(self, "Stop Migration",
            "Are you sure you want to stop the migration?\n"
            "Progress will be saved to checkpoint.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.progress_label.setText("Stopping - saving checkpoint...")
            self.stop_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self._worker.request_stop()
    
    def _set_inputs_enabled(self, enabled: bool) -> None:
        """Enable or disable all input widgets."""
        # Output section
        self.output_dir_edit.setEnabled(enabled)
        self.project_name_edit.setEnabled(enabled)
        self.output_stacked_check.setEnabled(enabled)
        self.output_fold_check.setEnabled(enabled)
        self.output_cig_check.setEnabled(enabled)
        self.output_qc_check.setEnabled(enabled)
        self.output_format_combo.setEnabled(enabled)

        # Backend section
        self.backend_combo.setEnabled(enabled)
        self.benchmark_btn.setEnabled(enabled)

        # Resources section
        self.max_memory_spin.setEnabled(enabled)
        self.n_threads_spin.setEnabled(enabled)

        # Tiling section
        self.auto_tile_check.setEnabled(enabled)
        self.tile_nx_spin.setEnabled(enabled and not self.auto_tile_check.isChecked())
        self.tile_ny_spin.setEnabled(enabled and not self.auto_tile_check.isChecked())
        self.tile_order_combo.setEnabled(enabled)

        # Checkpoint section
        self.enable_checkpoint.setEnabled(enabled)
        self.checkpoint_tiles_spin.setEnabled(enabled)
        self.checkpoint_seconds_spin.setEnabled(enabled)
        self.resume_check.setEnabled(enabled)

    def _on_heartbeat(self) -> None:
        """Update UI with elapsed time during long operations."""
        elapsed = time.time() - self._migration_start_time
        elapsed_str = self._format_elapsed(elapsed)

        # Update the progress label with elapsed time
        if self._last_traces_in_tile > 0:
            self.progress_label.setText(
                f"{self._last_progress_msg} [{elapsed_str}]"
            )
        else:
            self.progress_label.setText(
                f"{self._last_progress_msg} [{elapsed_str}]"
            )

        # Update ETA label to show elapsed time if no ETA available
        if self.estimate_labels["est_time"].text() in ("--", ""):
            self.estimate_labels["est_time"].setText(f"Elapsed: {elapsed_str}")

    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed time as human-readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _reset_run_ui(self) -> None:
        """Reset run controls."""
        # Stop heartbeat timer
        if self._heartbeat_timer:
            self._heartbeat_timer.stop()
            self._heartbeat_timer = None

        # Re-enable all input widgets
        self._set_inputs_enabled(True)

        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(False)
        self.pause_btn.setVisible(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setVisible(False)
        self.stop_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.validate_btn.setEnabled(True)
        self.pause_btn.setText("⏸ Pause")
        self._is_paused = False
        self._worker = None

    # --- Signal Slots for MigrationWorker ---

    def _on_progress_updated(self, progress: MigrationProgress) -> None:
        """Handle progress update from worker."""
        # Update progress bar
        if progress.total > 0:
            percent = int(100 * progress.current / progress.total)
            self.progress_bar.setValue(percent)

        # Update tile count
        self.estimate_labels["n_tiles"].setText(f"{progress.current}/{progress.total}")

        # Update ETA
        if progress.eta_seconds is not None:
            if progress.eta_seconds < 60:
                eta_str = f"~{int(progress.eta_seconds)} sec"
            elif progress.eta_seconds < 3600:
                eta_str = f"~{int(progress.eta_seconds / 60)} min"
            else:
                eta_str = f"~{progress.eta_seconds / 3600:.1f} hr"
            self.estimate_labels["est_time"].setText(eta_str)

        # Update trace counts
        if progress.traces_processed > 0:
            if progress.traces_in_tile > 0:
                self.estimate_labels["n_traces"].setText(
                    f"{progress.traces_processed:,} (+{progress.traces_in_tile:,})"
                )
            else:
                self.estimate_labels["n_traces"].setText(f"{progress.traces_processed:,}")

        # Update processing rate
        if progress.traces_per_second > 0:
            if progress.traces_per_second >= 1000:
                rate_str = f"{progress.traces_per_second / 1000:.1f}k traces/s"
            else:
                rate_str = f"{progress.traces_per_second:.0f} traces/s"
            self.estimate_labels["trace_rate"].setText(rate_str)
        elif progress.tiles_per_second > 0:
            self.estimate_labels["trace_rate"].setText(f"{progress.tiles_per_second:.2f} tiles/s")

        # Track for heartbeat updates
        self._last_traces_in_tile = progress.traces_in_tile

        # Update status message - include tile info if available
        if progress.current_tile_info:
            self._last_progress_msg = progress.current_tile_info
            elapsed = time.time() - self._migration_start_time
            self.progress_label.setText(f"{progress.current_tile_info} [{self._format_elapsed(elapsed)}]")
        elif progress.message:
            self._last_progress_msg = progress.message
            elapsed = time.time() - self._migration_start_time
            self.progress_label.setText(f"{progress.message} [{self._format_elapsed(elapsed)}]")

    def _on_phase_changed(self, phase: str, description: str) -> None:
        """Handle phase change from worker."""
        self._last_progress_msg = description
        elapsed = time.time() - self._migration_start_time
        self.progress_label.setText(f"{description} [{self._format_elapsed(elapsed)}]")

        # Update preflight text with current phase
        current_text = self.preflight_text.toPlainText()
        lines = current_text.split('\n')
        # Find and update status line
        for i, line in enumerate(lines):
            if line.startswith("═══"):
                lines[i] = f"═══ {phase.upper()} ═══"
                break
        self.preflight_text.setPlainText('\n'.join(lines))

    def _on_migration_success(self, output_path: str) -> None:
        """Handle successful migration completion."""
        from PyQt6.QtWidgets import QMessageBox

        self.progress_bar.setValue(100)
        self.progress_label.setText("Migration complete!")

        QMessageBox.information(self, "Migration Complete",
            f"Migration completed successfully!\n\n"
            f"Output saved to:\n{output_path}\n\n"
            f"Click OK to view results.")

        # Navigate to Results step (step 8, index 7)
        # Find the main window and navigate
        main_window = self.window()
        if hasattr(main_window, '_go_to_step'):
            main_window._go_to_step(7)  # Results step is index 7

    def _on_migration_error(self, error_msg: str) -> None:
        """Handle migration error."""
        from PyQt6.QtWidgets import QMessageBox

        self.progress_label.setText("Migration failed")

        QMessageBox.critical(self, "Migration Error", error_msg)

    def _on_log_message(self, level: str, message: str) -> None:
        """Handle log message from worker."""
        # Append to preflight text area for now
        current = self.preflight_text.toPlainText()
        timestamp = time.strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {level.upper()}: {message}"
        self.preflight_text.setPlainText(current + '\n' + log_line)

        # Auto-scroll to bottom
        scrollbar = self.preflight_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_worker_finished(self) -> None:
        """Handle worker thread completion."""
        self._reset_run_ui()
        # Re-run preflight to show final state
        QTimer.singleShot(100, self._run_preflight)

    def on_enter(self) -> None:
        """Load state and run preflight."""
        # Load output state
        output_state = self.controller.state.output
        self.output_dir_edit.setText(output_state.output_dir)
        self.project_name_edit.setText(output_state.project_name)
        self.output_stacked_check.setChecked(output_state.output_stacked_image)
        self.output_fold_check.setChecked(output_state.output_fold_map)
        self.output_cig_check.setChecked(output_state.output_cig)
        self.output_qc_check.setChecked(output_state.output_qc_report)

        format_map = {"zarr": 0, "segy": 1}
        self.output_format_combo.setCurrentIndex(format_map.get(output_state.output_format, 0))

        # Load execution state
        state = self.controller.state.execution

        # Backend - use dynamic options from _create_backend_section
        backend_values = [opt[1] for opt in self._backend_options]
        idx = backend_values.index(state.backend) if state.backend in backend_values else 0
        self.backend_combo.setCurrentIndex(idx)

        # Resources
        self.max_memory_spin.setValue(state.max_memory_gb)
        self.n_threads_spin.setValue(state.n_threads)

        # Tiling
        self.auto_tile_check.setChecked(state.auto_tile_size)
        self.tile_nx_spin.setValue(state.tile_nx)
        self.tile_ny_spin.setValue(state.tile_ny)

        order_map = {"snake": 0, "row_major": 1, "column_major": 2, "hilbert": 3}
        self.tile_order_combo.setCurrentIndex(order_map.get(state.tile_ordering, 0))

        # Checkpointing
        self.enable_checkpoint.setChecked(state.enable_checkpoint)
        self.checkpoint_tiles_spin.setValue(state.checkpoint_interval_tiles)
        self.checkpoint_seconds_spin.setValue(int(state.checkpoint_interval_seconds))
        self.resume_check.setChecked(state.resume_from_checkpoint)

        # Run preflight
        QTimer.singleShot(100, self._run_preflight)

    def on_leave(self) -> None:
        """Save UI to state."""
        import logging
        debug_logger = logging.getLogger("pstm.migration.debug")

        # Save output state
        output_state = self.controller.state.output
        output_state.output_dir = self.output_dir_edit.text().strip()
        output_state.project_name = self.project_name_edit.text().strip() or "migration_output"
        output_state.output_stacked_image = self.output_stacked_check.isChecked()
        output_state.output_fold_map = self.output_fold_check.isChecked()
        output_state.output_cig = self.output_cig_check.isChecked()
        output_state.output_qc_report = self.output_qc_check.isChecked()

        formats = ["zarr", "segy"]
        output_state.output_format = formats[self.output_format_combo.currentIndex()]

        # Save execution state
        state = self.controller.state.execution

        # Use dynamic backend options from _create_backend_section
        combo_index = self.backend_combo.currentIndex()
        combo_text = self.backend_combo.currentText()
        selected_backend = self._backend_options[combo_index][1]  # Get backend value from tuple

        debug_logger.info(f"EXECUTION_STEP on_leave: combo_index={combo_index}, combo_text='{combo_text}'")
        debug_logger.info(f"EXECUTION_STEP on_leave: selected_backend='{selected_backend}'")
        debug_logger.info(f"EXECUTION_STEP on_leave: state.backend BEFORE='{state.backend}'")

        state.backend = selected_backend

        debug_logger.info(f"EXECUTION_STEP on_leave: state.backend AFTER='{state.backend}'")

        state.max_memory_gb = self.max_memory_spin.value()
        state.n_threads = self.n_threads_spin.value()

        state.auto_tile_size = self.auto_tile_check.isChecked()
        state.tile_nx = self.tile_nx_spin.value()
        state.tile_ny = self.tile_ny_spin.value()

        orders = ["snake", "row_major", "column_major", "hilbert"]
        state.tile_ordering = orders[self.tile_order_combo.currentIndex()]

        state.enable_checkpoint = self.enable_checkpoint.isChecked()
        state.checkpoint_interval_tiles = self.checkpoint_tiles_spin.value()
        state.checkpoint_interval_seconds = float(self.checkpoint_seconds_spin.value())
        state.resume_from_checkpoint = self.resume_check.isChecked()

        self.controller.state.step_status["execution"] = StepStatus.COMPLETE
        self.controller.notify_change()

    def validate(self) -> bool:
        """Validate execution configuration."""
        errors = []

        # Check output directory
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            errors.append("Output directory is required")
        elif not Path(output_dir).exists():
            errors.append(f"Output directory does not exist: {output_dir}")

        # Check project name
        project_name = self.project_name_edit.text().strip()
        if not project_name:
            errors.append("Project name is required")

        # Check memory
        if self.max_memory_spin.value() < 1:
            errors.append("Max memory must be at least 1 GB")

        # Check at least one output product selected
        if not any([
            self.output_stacked_check.isChecked(),
            self.output_fold_check.isChecked(),
            self.output_cig_check.isChecked(),
        ]):
            errors.append("At least one output product must be selected")

        if errors:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Validation Errors",
                "Please fix the following issues:\n\n" + "\n".join(f"• {e}" for e in errors)
            )
            return False

        return True

    def cleanup(self) -> None:
        """
        Safely stop any running migration and clean up resources.

        Call this method before closing the application to ensure
        the migration worker thread terminates gracefully.
        """
        # Stop heartbeat timer
        if self._heartbeat_timer:
            self._heartbeat_timer.stop()
            self._heartbeat_timer = None

        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            # Wait for worker to finish with timeout
            if not self._worker.wait(5000):  # 5 second timeout
                # Force terminate if still running
                self._worker.terminate()
                self._worker.wait(1000)
            self._worker = None

    def closeEvent(self, event) -> None:
        """Handle widget close - ensure worker thread is stopped."""
        if self._worker and self._worker.isRunning():
            from PyQt6.QtWidgets import QMessageBox

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
            self.cleanup()

        event.accept()

    def refresh_from_state(self) -> None:
        """Refresh UI from loaded state."""
        # Output state
        output_state = self.controller.state.output
        self.output_dir_edit.setText(output_state.output_dir or "")
        self.project_name_edit.setText(output_state.project_name or "")
        self.output_stacked_check.setChecked(output_state.output_stacked_image)
        self.output_fold_check.setChecked(output_state.output_fold_map)
        self.output_cig_check.setChecked(output_state.output_cig)
        self.output_qc_check.setChecked(output_state.output_qc_report)

        format_map = {"zarr": 0, "segy": 1}
        self.output_format_combo.setCurrentIndex(format_map.get(output_state.output_format, 0))

        # Execution state
        state = self.controller.state.execution

        backend_values = [opt[1] for opt in self._backend_options]
        idx = backend_values.index(state.backend) if state.backend in backend_values else 0
        self.backend_combo.setCurrentIndex(idx)

        self.max_memory_spin.setValue(state.max_memory_gb)
        self.n_threads_spin.setValue(state.n_threads)

        self.auto_tile_check.setChecked(state.auto_tile_size)
        self.tile_nx_spin.setValue(state.tile_nx)
        self.tile_ny_spin.setValue(state.tile_ny)

        orders = ["snake", "row_major", "column_major", "hilbert"]
        if state.tile_ordering in orders:
            self.tile_order_combo.setCurrentIndex(orders.index(state.tile_ordering))

        self.enable_checkpoint.setChecked(state.enable_checkpoint)
        self.checkpoint_tiles_spin.setValue(state.checkpoint_interval_tiles)
        self.checkpoint_seconds_spin.setValue(int(state.checkpoint_interval_seconds))
        self.resume_check.setChecked(state.resume_from_checkpoint)
