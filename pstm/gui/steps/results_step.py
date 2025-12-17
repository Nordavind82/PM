"""
Step 8: Results - View migration results and QC.

This step displays results after migration execution and provides
tools for quality control and export.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QFrame, QSlider, QSpinBox,
    QSplitter, QFileDialog, QTabWidget, QTextEdit,
)
from PyQt6.QtCore import Qt

from pstm.gui.steps.base import WizardStepWidget

if TYPE_CHECKING:
    from pstm.gui.state import WizardController


class ResultsStep(WizardStepWidget):
    """Step 8: Results viewing and QC."""
    
    @property
    def title(self) -> str:
        return "Results"
    
    def _setup_ui(self) -> None:
        """Set up the UI."""
        # Header
        header_frame, header_layout = self.create_section("Step 8: Results")
        desc = QLabel(
            "View migration results and quality control. "
            "After migration completes, this step shows output statistics, "
            "quick QC views, and export options."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888888; border: none; background: transparent;")
        header_layout.addWidget(desc)
        self.content_layout.addWidget(header_frame)
        
        # Summary section
        self._create_summary_section()
        
        # Main content - tabs
        self.results_tabs = QTabWidget()
        
        # Quick QC tab
        qc_widget = self._create_quick_qc_tab()
        self.results_tabs.addTab(qc_widget, "Quick QC")
        
        # Statistics tab
        stats_widget = self._create_statistics_tab()
        self.results_tabs.addTab(stats_widget, "Statistics")
        
        # Files tab
        files_widget = self._create_files_tab()
        self.results_tabs.addTab(files_widget, "Output Files")
        
        # Log tab
        log_widget = self._create_log_tab()
        self.results_tabs.addTab(log_widget, "Execution Log")
        
        self.content_layout.addWidget(self.results_tabs, 1)
        
        # Actions
        self._create_actions_section()
    
    def _create_summary_section(self) -> None:
        """Create summary section."""
        frame, layout = self.create_section("Summary")
        
        form = QFormLayout()
        form.setSpacing(8)
        
        self.status_label = QLabel("⏳ Waiting for migration to complete")
        self.status_label.setStyleSheet("color: #ff9800; font-weight: bold; border: none; background: transparent;")
        form.addRow("Status:", self.status_label)
        
        self.total_time_label = QLabel("--")
        self.total_time_label.setStyleSheet("border: none; background: transparent;")
        form.addRow("Total Time:", self.total_time_label)
        
        self.traces_label = QLabel("--")
        self.traces_label.setStyleSheet("border: none; background: transparent;")
        form.addRow("Traces Processed:", self.traces_label)
        
        self.output_size_label = QLabel("--")
        self.output_size_label.setStyleSheet("border: none; background: transparent;")
        form.addRow("Output Size:", self.output_size_label)
        
        layout.addLayout(form)
        self.content_layout.addWidget(frame)
    
    def _create_quick_qc_tab(self) -> QWidget:
        """Create quick QC tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Splitter for two views
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Inline slice view
        inline_group = QGroupBox("Inline Slice")
        inline_layout = QVBoxLayout(inline_group)
        
        self.inline_frame = QFrame()
        self.inline_frame.setMinimumSize(300, 250)
        self.inline_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        inline_frame_layout = QVBoxLayout(self.inline_frame)
        self.inline_label = QLabel("Run migration to see results")
        self.inline_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inline_label.setStyleSheet("color: #666666; border: none;")
        inline_frame_layout.addWidget(self.inline_label)
        
        inline_layout.addWidget(self.inline_frame, 1)
        
        # Inline slider
        inline_ctrl = QHBoxLayout()
        inline_ctrl.addWidget(QLabel("IL:"))
        self.inline_slider = QSlider(Qt.Orientation.Horizontal)
        self.inline_slider.setRange(0, 100)
        self.inline_slider.valueChanged.connect(self._update_inline_view)
        inline_ctrl.addWidget(self.inline_slider, 1)
        self.inline_num_label = QLabel("0")
        inline_ctrl.addWidget(self.inline_num_label)
        inline_layout.addLayout(inline_ctrl)
        
        splitter.addWidget(inline_group)
        
        # Time slice view
        time_group = QGroupBox("Time Slice")
        time_layout = QVBoxLayout(time_group)
        
        self.time_frame = QFrame()
        self.time_frame.setMinimumSize(300, 250)
        self.time_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        time_frame_layout = QVBoxLayout(self.time_frame)
        self.time_label = QLabel("Run migration to see results")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("color: #666666; border: none;")
        time_frame_layout.addWidget(self.time_label)
        
        time_layout.addWidget(self.time_frame, 1)
        
        # Time slider
        time_ctrl = QHBoxLayout()
        time_ctrl.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 4000)
        self.time_slider.setValue(2000)
        self.time_slider.valueChanged.connect(self._update_time_view)
        time_ctrl.addWidget(self.time_slider, 1)
        self.time_num_label = QLabel("2000 ms")
        time_ctrl.addWidget(self.time_num_label)
        time_layout.addLayout(time_ctrl)
        
        splitter.addWidget(time_group)
        
        layout.addWidget(splitter, 1)
        
        # View buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QPushButton("Open in External Viewer"))
        btn_layout.addWidget(QPushButton("Export to SEG-Y"))
        btn_layout.addWidget(QPushButton("View Full QC Report"))
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return widget
    
    def _create_statistics_tab(self) -> QWidget:
        """Create statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Amplitude statistics
        amp_group = QGroupBox("Amplitude Statistics")
        amp_layout = QFormLayout(amp_group)
        
        self.amp_min_label = QLabel("--")
        amp_layout.addRow("Min:", self.amp_min_label)
        
        self.amp_max_label = QLabel("--")
        amp_layout.addRow("Max:", self.amp_max_label)
        
        self.amp_mean_label = QLabel("--")
        amp_layout.addRow("Mean:", self.amp_mean_label)
        
        self.amp_rms_label = QLabel("--")
        amp_layout.addRow("RMS:", self.amp_rms_label)
        
        layout.addWidget(amp_group)
        
        # Data quality
        quality_group = QGroupBox("Data Quality")
        quality_layout = QFormLayout(quality_group)
        
        self.zero_samples_label = QLabel("--")
        quality_layout.addRow("Zero samples:", self.zero_samples_label)
        
        self.nan_samples_label = QLabel("--")
        quality_layout.addRow("NaN samples:", self.nan_samples_label)
        
        self.inf_samples_label = QLabel("--")
        quality_layout.addRow("Inf samples:", self.inf_samples_label)
        
        layout.addWidget(quality_group)
        
        # Fold statistics
        fold_group = QGroupBox("Fold Statistics")
        fold_layout = QFormLayout(fold_group)
        
        self.fold_min_label = QLabel("--")
        fold_layout.addRow("Min Fold:", self.fold_min_label)
        
        self.fold_max_label = QLabel("--")
        fold_layout.addRow("Max Fold:", self.fold_max_label)
        
        self.fold_mean_label = QLabel("--")
        fold_layout.addRow("Mean Fold:", self.fold_mean_label)
        
        layout.addWidget(fold_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_files_tab(self) -> QWidget:
        """Create output files tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # File list
        files_group = QGroupBox("Output Files")
        files_layout = QVBoxLayout(files_group)
        
        self.files_list_label = QLabel("No output files yet - run migration first")
        self.files_list_label.setWordWrap(True)
        self.files_list_label.setStyleSheet("color: #888888;")
        files_layout.addWidget(self.files_list_label)
        
        layout.addWidget(files_group)
        
        # Open folder button
        open_btn = QPushButton("Open Output Folder")
        open_btn.clicked.connect(self._open_output_folder)
        layout.addWidget(open_btn)
        
        layout.addStretch()
        
        return widget
    
    def _create_log_tab(self) -> QWidget:
        """Create execution log tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                font-family: monospace;
                font-size: 11px;
                border: 1px solid #3d3d3d;
            }
        """)
        self.log_text.setPlainText(
            "Migration log will appear here after execution starts.\n"
            "Go to Step 7 (Execution) to run the migration."
        )
        layout.addWidget(self.log_text)
        
        # Log controls
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QPushButton("Save Log"))
        btn_layout.addWidget(QPushButton("Clear"))
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return widget
    
    def _create_actions_section(self) -> None:
        """Create actions section."""
        frame, layout = self.create_section("Actions")
        
        # Convert to horizontal layout
        btn_layout = QHBoxLayout()
        
        new_btn = QPushButton("New Migration")
        new_btn.clicked.connect(self._new_migration)
        btn_layout.addWidget(new_btn)
        
        edit_btn = QPushButton("Edit && Re-run")
        edit_btn.clicked.connect(self._edit_and_rerun)
        btn_layout.addWidget(edit_btn)
        
        open_btn = QPushButton("Open Output Folder")
        open_btn.clicked.connect(self._open_output_folder)
        btn_layout.addWidget(open_btn)
        
        export_btn = QPushButton("Export Results...")
        btn_layout.addWidget(export_btn)
        
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        self.content_layout.addWidget(frame)
    
    def _update_inline_view(self, value: int) -> None:
        """Update inline slice view."""
        self.inline_num_label.setText(str(value))
        # TODO: Update visualization when results are available
    
    def _update_time_view(self, value: int) -> None:
        """Update time slice view."""
        self.time_num_label.setText(f"{value} ms")
        # TODO: Update visualization when results are available
    
    def _open_output_folder(self) -> None:
        """Open output folder in file manager."""
        output_dir = self.controller.state.output.output_dir
        if output_dir and Path(output_dir).exists():
            import subprocess
            import sys
            
            if sys.platform == 'darwin':
                subprocess.run(['open', output_dir])
            elif sys.platform == 'linux':
                subprocess.run(['xdg-open', output_dir])
            elif sys.platform == 'win32':
                subprocess.run(['explorer', output_dir])
    
    def _new_migration(self) -> None:
        """Start new migration."""
        # Reset to step 1
        self.controller.reset()
    
    def _edit_and_rerun(self) -> None:
        """Go back to edit and re-run."""
        # Go to execution step
        self.controller.state.current_step = 6  # Step 7 (Execution)
    
    def on_enter(self) -> None:
        """Called when navigating to this step."""
        # Update display based on current state
        self._refresh_display()
    
    def _refresh_display(self) -> None:
        """Refresh the display with current results."""
        output = self.controller.state.output
        output_grid = self.controller.state.output_grid

        # Update time slider range
        self.time_slider.setRange(
            int(output_grid.t_min_ms),
            int(output_grid.t_max_ms)
        )

        # Update inline slider range
        self.inline_slider.setRange(0, max(0, output_grid.nx - 1))

        # Check if results exist - look for migrated_stack.zarr directly in output_dir
        if output.output_dir and Path(output.output_dir).exists():
            output_path = Path(output.output_dir)
            stack_path = output_path / "migrated_stack.zarr"
            fold_path = output_path / "fold.zarr"

            if stack_path.exists():
                self._load_results(str(output_path))
    
    def _load_results(self, result_path: str) -> None:
        """Load results from output path."""
        try:
            import zarr
            import numpy as np

            path = Path(result_path)

            # Update status
            self.status_label.setText("✓ Completed Successfully")
            self.status_label.setStyleSheet(
                "color: #4caf50; font-weight: bold; border: none; background: transparent;"
            )

            # Update file list
            files = []
            total_size_bytes = 0
            if path.is_dir():
                for f in path.iterdir():
                    if f.name.startswith('.'):
                        continue  # Skip hidden files/dirs
                    if f.is_file():
                        size_bytes = f.stat().st_size
                        total_size_bytes += size_bytes
                        size_mb = size_bytes / 1e6
                        files.append(f"• {f.name} ({size_mb:.1f} MB)")
                    elif f.is_dir():
                        # Calculate directory size for zarr files
                        dir_size = sum(p.stat().st_size for p in f.rglob('*') if p.is_file())
                        total_size_bytes += dir_size
                        size_mb = dir_size / 1e6
                        files.append(f"• {f.name}/ ({size_mb:.1f} MB)")

            self.files_list_label.setText(
                "\n".join(sorted(files)) if files else "No files found"
            )
            self.files_list_label.setStyleSheet("color: #cccccc;")

            # Update output size label
            self.output_size_label.setText(f"{total_size_bytes / 1e6:.2f} MB")

            # Load statistics from migrated_stack.zarr
            stack_path = path / 'migrated_stack.zarr'
            if stack_path.exists():
                try:
                    z = zarr.open(str(stack_path), mode='r')
                    # Handle both array and group formats
                    if isinstance(z, zarr.Array):
                        arr = z
                    elif 'data' in z:
                        arr = z['data']
                    else:
                        arr = z[list(z.keys())[0]] if z.keys() else None

                    if arr is not None:
                        # Compute basic stats on a subsample for speed
                        step = max(1, arr.shape[0] // 10)
                        sample = arr[::step, ::step, ::step]
                        sample_np = np.array(sample)

                        # Filter out zeros/nans for statistics
                        valid = sample_np[~np.isnan(sample_np)]

                        self.amp_min_label.setText(f"{float(np.min(valid)):.4e}")
                        self.amp_max_label.setText(f"{float(np.max(valid)):.4e}")
                        self.amp_mean_label.setText(f"{float(np.mean(valid)):.4e}")
                        self.amp_rms_label.setText(f"{float(np.sqrt(np.mean(valid**2))):.4e}")

                        # Data quality stats
                        total_samples = sample_np.size
                        zero_count = np.sum(sample_np == 0)
                        nan_count = np.sum(np.isnan(sample_np))
                        inf_count = np.sum(np.isinf(sample_np))

                        self.zero_samples_label.setText(f"{zero_count:,} ({100*zero_count/total_samples:.1f}%)")
                        self.nan_samples_label.setText(f"{nan_count:,} ({100*nan_count/total_samples:.1f}%)")
                        self.inf_samples_label.setText(f"{inf_count:,} ({100*inf_count/total_samples:.1f}%)")

                except Exception as e:
                    self.amp_min_label.setText(f"Error: {e}")

            # Load fold statistics
            fold_path = path / 'fold.zarr'
            if fold_path.exists():
                try:
                    z_fold = zarr.open(str(fold_path), mode='r')
                    if isinstance(z_fold, zarr.Array):
                        fold_arr = z_fold
                    elif 'data' in z_fold:
                        fold_arr = z_fold['data']
                    else:
                        fold_arr = z_fold[list(z_fold.keys())[0]] if z_fold.keys() else None

                    if fold_arr is not None:
                        fold_np = np.array(fold_arr)
                        self.fold_min_label.setText(f"{int(np.min(fold_np)):,}")
                        self.fold_max_label.setText(f"{int(np.max(fold_np)):,}")
                        self.fold_mean_label.setText(f"{float(np.mean(fold_np)):.1f}")

                except Exception as e:
                    self.fold_min_label.setText(f"Error: {e}")

            # Load execution log if available
            log_files = list(path.glob("migration_debug_*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                try:
                    log_content = latest_log.read_text()
                    # Show last 500 lines
                    lines = log_content.splitlines()
                    if len(lines) > 500:
                        lines = lines[-500:]
                    self.log_text.setPlainText("\n".join(lines))
                except Exception:
                    pass

        except Exception as e:
            self.status_label.setText(f"⚠ Error loading results: {e}")
            self.status_label.setStyleSheet(
                "color: #f44336; font-weight: bold; border: none; background: transparent;"
            )
    
    def update_from_execution(self, result: dict) -> None:
        """Update display after execution completes."""
        if result.get("status") == "complete":
            self.status_label.setText("✓ Completed Successfully")
            self.status_label.setStyleSheet(
                "color: #4caf50; font-weight: bold; border: none; background: transparent;"
            )
            
            if "total_time" in result:
                minutes = int(result["total_time"] // 60)
                seconds = int(result["total_time"] % 60)
                self.total_time_label.setText(f"{minutes}m {seconds}s")
            
            if "n_traces_processed" in result:
                self.traces_label.setText(f"{result['n_traces_processed']:,}")
            
            if "output_size_gb" in result:
                self.output_size_label.setText(f"{result['output_size_gb']:.2f} GB")
        
        elif result.get("status") == "error":
            self.status_label.setText(f"✗ Error: {result.get('error', 'Unknown')}")
            self.status_label.setStyleSheet(
                "color: #f44336; font-weight: bold; border: none; background: transparent;"
            )
    
    def append_log(self, message: str) -> None:
        """Append message to execution log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def validate(self) -> bool:
        """Validate the step - always valid as it's read-only."""
        return True
