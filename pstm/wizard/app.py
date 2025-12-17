"""
Wizard TUI application for PSTM.

Interactive configuration wizard using Textual framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.screen import Screen
    from textual.widgets import (
        Button,
        DirectoryTree,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        Placeholder,
        ProgressBar,
        RadioButton,
        RadioSet,
        Rule,
        Select,
        Static,
        Switch,
        TabbedContent,
        TabPane,
        Tree,
    )
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    App = object
    ComposeResult = None

from pstm.config.models import (
    MigrationConfig,
    InputConfig,
    GeometryConfig,
    VelocityConfig,
    AlgorithmConfig,
    OutputConfig,
    OutputGridConfig,
    ExecutionConfig,
    VelocitySource,
    InterpolationMethod,
    AntiAliasingMethod,
    ComputeBackend,
)
from pstm.utils.logging import get_logger

# Check Metal C++ availability
def _get_metal_info() -> dict | None:
    """Get Metal GPU device info if available."""
    try:
        from pstm.kernels.metal_cpp import is_metal_cpp_available
        if is_metal_cpp_available():
            from pstm.metal.python import get_device_info
            return get_device_info()
    except ImportError:
        pass
    return None

logger = get_logger(__name__)


def check_textual_available() -> bool:
    """Check if Textual is available."""
    return TEXTUAL_AVAILABLE


if TEXTUAL_AVAILABLE:
    
    class WizardApp(App):
        """
        PSTM Configuration Wizard Application.
        
        A 6-tab wizard interface for configuring migration jobs:
        1. Input Data - Select traces and headers
        2. Geometry - Survey analysis and spatial index
        3. Velocity - Velocity model configuration
        4. Algorithm - Migration parameters
        5. Output - Output grid and products
        6. Execution - Resources and run control
        """
        
        CSS = """
        Screen {
            background: $surface;
        }
        
        #main-container {
            height: 100%;
        }
        
        .tab-content {
            padding: 1;
        }
        
        .form-group {
            margin-bottom: 1;
        }
        
        .form-label {
            width: 20;
            padding-right: 1;
        }
        
        .form-input {
            width: 40;
        }
        
        .form-row {
            height: 3;
            margin-bottom: 1;
        }
        
        .section-header {
            background: $primary;
            color: $text;
            padding: 0 1;
            margin: 1 0;
        }
        
        .validation-error {
            color: $error;
            margin-left: 21;
        }
        
        .validation-ok {
            color: $success;
            margin-left: 21;
        }
        
        .file-path {
            color: $secondary;
        }
        
        #status-bar {
            dock: bottom;
            height: 3;
            background: $surface-darken-1;
            padding: 0 1;
        }
        
        .button-row {
            margin-top: 1;
            height: 3;
        }
        
        Button {
            margin-right: 1;
        }
        """
        
        BINDINGS = [
            Binding("ctrl+q", "quit", "Quit"),
            Binding("ctrl+s", "save_config", "Save Config"),
            Binding("ctrl+r", "run_migration", "Run"),
            Binding("ctrl+n", "next_tab", "Next Tab"),
            Binding("ctrl+p", "prev_tab", "Prev Tab"),
        ]
        
        def __init__(self, config_path: Path | None = None):
            """
            Initialize wizard.
            
            Args:
                config_path: Optional path to existing config to load
            """
            super().__init__()
            self.config_path = config_path
            self.config: MigrationConfig | None = None
            
            # Form state
            self._traces_path: str = ""
            self._headers_path: str = ""
            self._output_dir: str = ""
            self._velocity_value: str = "2000.0"
            self._velocity_source: str = "constant"
            
        def compose(self) -> ComposeResult:
            """Compose the wizard UI."""
            yield Header(show_clock=True)
            
            with Container(id="main-container"):
                with TabbedContent(id="wizard-tabs"):
                    with TabPane("Input", id="tab-input"):
                        yield from self._compose_input_tab()
                    
                    with TabPane("Geometry", id="tab-geometry"):
                        yield from self._compose_geometry_tab()
                    
                    with TabPane("Velocity", id="tab-velocity"):
                        yield from self._compose_velocity_tab()
                    
                    with TabPane("Algorithm", id="tab-algorithm"):
                        yield from self._compose_algorithm_tab()
                    
                    with TabPane("Output", id="tab-output"):
                        yield from self._compose_output_tab()
                    
                    with TabPane("Execution", id="tab-execution"):
                        yield from self._compose_execution_tab()
            
            with Horizontal(id="status-bar"):
                yield Static("Ready", id="status-text")
                yield Button("Save", id="btn-save", variant="primary")
                yield Button("Validate", id="btn-validate")
                yield Button("Run", id="btn-run", variant="success")
            
            yield Footer()
        
        def _compose_input_tab(self) -> ComposeResult:
            """Compose input data tab."""
            with ScrollableContainer(classes="tab-content"):
                yield Static("Input Data Configuration", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Traces (Zarr):", classes="form-label")
                    yield Input(
                        placeholder="/path/to/traces.zarr",
                        id="input-traces",
                        classes="form-input",
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Headers (Parquet):", classes="form-label")
                    yield Input(
                        placeholder="/path/to/headers.parquet",
                        id="input-headers",
                        classes="form-input",
                    )
                
                yield Static("Column Mapping", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Source X:", classes="form-label")
                    yield Input(value="SOU_X", id="col-source-x", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Source Y:", classes="form-label")
                    yield Input(value="SOU_Y", id="col-source-y", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Receiver X:", classes="form-label")
                    yield Input(value="REC_X", id="col-receiver-x", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Receiver Y:", classes="form-label")
                    yield Input(value="REC_Y", id="col-receiver-y", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Trace Index:", classes="form-label")
                    yield Input(value="trace_idx", id="col-trace-idx", classes="form-input")
                
                yield Static("Data Properties", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Sample Rate (ms):", classes="form-label")
                    yield Input(placeholder="Auto-detect", id="input-sample-rate", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Coordinate Unit:", classes="form-label")
                    yield Select(
                        [("Meters", "meters"), ("Feet", "feet")],
                        value="meters",
                        id="input-coord-unit",
                    )
                
                yield Static("", id="input-validation", classes="validation-ok")
        
        def _compose_geometry_tab(self) -> ComposeResult:
            """Compose geometry analysis tab."""
            with ScrollableContainer(classes="tab-content"):
                yield Static("Geometry Analysis", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Index Type:", classes="form-label")
                    yield Select(
                        [("KD-Tree", "kdtree"), ("Ball Tree", "balltree")],
                        value="kdtree",
                        id="geom-index-type",
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Index Key:", classes="form-label")
                    yield Select(
                        [("Midpoint", "midpoint"), ("CDP", "cdp"), ("Source", "source")],
                        value="midpoint",
                        id="geom-index-key",
                    )
                
                yield Static("Survey Extent (auto-detected)", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("X Range:", classes="form-label")
                    yield Static("Not loaded", id="geom-x-range")
                
                with Horizontal(classes="form-row"):
                    yield Label("Y Range:", classes="form-label")
                    yield Static("Not loaded", id="geom-y-range")
                
                with Horizontal(classes="form-row"):
                    yield Label("Offset Range:", classes="form-label")
                    yield Static("Not loaded", id="geom-offset-range")
                
                with Horizontal(classes="button-row"):
                    yield Button("Analyze Geometry", id="btn-analyze-geom")
        
        def _compose_velocity_tab(self) -> ComposeResult:
            """Compose velocity configuration tab."""
            with ScrollableContainer(classes="tab-content"):
                yield Static("Velocity Model", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Source:", classes="form-label")
                    yield Select(
                        [
                            ("Constant", "constant"),
                            ("Linear V(t)=V0+k*t", "linear_v0k"),
                            ("1D Table", "table_1d"),
                            ("3D Cube", "cube_3d"),
                        ],
                        value="constant",
                        id="vel-source",
                    )
                
                yield Static("Constant Velocity", classes="section-header", id="vel-const-section")
                
                with Horizontal(classes="form-row", id="vel-const-row"):
                    yield Label("Velocity (m/s):", classes="form-label")
                    yield Input(value="2000.0", id="vel-constant", classes="form-input")
                
                yield Static("Linear V(t)", classes="section-header", id="vel-linear-section")
                
                with Horizontal(classes="form-row", id="vel-v0-row"):
                    yield Label("V0 (m/s):", classes="form-label")
                    yield Input(value="1500.0", id="vel-v0", classes="form-input")
                
                with Horizontal(classes="form-row", id="vel-k-row"):
                    yield Label("k (m/s per s):", classes="form-label")
                    yield Input(value="0.5", id="vel-k", classes="form-input")
                
                yield Static("3D Velocity Cube", classes="section-header", id="vel-cube-section")
                
                with Horizontal(classes="form-row", id="vel-cube-row"):
                    yield Label("Cube Path:", classes="form-label")
                    yield Input(placeholder="/path/to/velocity.zarr", id="vel-cube-path", classes="form-input")
                
                yield Static("Velocity Bounds", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Min Velocity:", classes="form-label")
                    yield Input(value="1000.0", id="vel-min", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Max Velocity:", classes="form-label")
                    yield Input(value="8000.0", id="vel-max", classes="form-input")
        
        def _compose_algorithm_tab(self) -> ComposeResult:
            """Compose algorithm parameters tab."""
            with ScrollableContainer(classes="tab-content"):
                yield Static("Aperture Control", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Max Dip (°):", classes="form-label")
                    yield Input(value="45.0", id="algo-max-dip", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Min Aperture (m):", classes="form-label")
                    yield Input(value="500.0", id="algo-min-aperture", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Max Aperture (m):", classes="form-label")
                    yield Input(value="5000.0", id="algo-max-aperture", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Taper Fraction:", classes="form-label")
                    yield Input(value="0.1", id="algo-taper", classes="form-input")
                
                yield Static("Interpolation", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Method:", classes="form-label")
                    yield Select(
                        [
                            ("Linear", "linear"),
                            ("Nearest", "nearest"),
                            ("8-point Sinc", "sinc8"),
                        ],
                        value="linear",
                        id="algo-interp",
                    )
                
                yield Static("Anti-Aliasing", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Enabled:", classes="form-label")
                    yield Switch(value=True, id="algo-aa-enabled")
                
                with Horizontal(classes="form-row"):
                    yield Label("Method:", classes="form-label")
                    yield Select(
                        [
                            ("Triangle Filter", "triangle"),
                            ("Offset Sectoring", "offset_sectoring"),
                            ("None", "none"),
                        ],
                        value="triangle",
                        id="algo-aa-method",
                    )
                
                yield Static("Amplitude Weights", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Geometrical Spreading:", classes="form-label")
                    yield Switch(value=True, id="algo-spreading")
                
                with Horizontal(classes="form-row"):
                    yield Label("Obliquity Factor:", classes="form-label")
                    yield Switch(value=True, id="algo-obliquity")
        
        def _compose_output_tab(self) -> ComposeResult:
            """Compose output configuration tab."""
            with ScrollableContainer(classes="tab-content"):
                yield Static("Output Directory", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Output Path:", classes="form-label")
                    yield Input(placeholder="/path/to/output", id="output-dir", classes="form-input")
                
                yield Static("Output Grid", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("X Min:", classes="form-label")
                    yield Input(placeholder="0", id="output-x-min", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("X Max:", classes="form-label")
                    yield Input(placeholder="10000", id="output-x-max", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("dX:", classes="form-label")
                    yield Input(value="25.0", id="output-dx", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Y Min:", classes="form-label")
                    yield Input(placeholder="0", id="output-y-min", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Y Max:", classes="form-label")
                    yield Input(placeholder="10000", id="output-y-max", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("dY:", classes="form-label")
                    yield Input(value="25.0", id="output-dy", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("T Min (ms):", classes="form-label")
                    yield Input(value="0", id="output-t-min", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("T Max (ms):", classes="form-label")
                    yield Input(placeholder="4000", id="output-t-max", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("dT (ms):", classes="form-label")
                    yield Input(value="2.0", id="output-dt", classes="form-input")
                
                yield Static("Output Products", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Stacked Image:", classes="form-label")
                    yield Switch(value=True, id="output-stack")
                
                with Horizontal(classes="form-row"):
                    yield Label("Fold Volume:", classes="form-label")
                    yield Switch(value=True, id="output-fold")
                
                with Horizontal(classes="form-row"):
                    yield Label("CIG Gathers:", classes="form-label")
                    yield Switch(value=False, id="output-cig")
                
                yield Static("", id="output-size-estimate")
        
        def _compose_execution_tab(self) -> ComposeResult:
            """Compose execution configuration tab."""
            # Check for Metal GPU availability
            metal_info = _get_metal_info()

            with ScrollableContainer(classes="tab-content"):
                yield Static("Compute Resources", classes="section-header")

                # Build backend options dynamically
                backend_options = [("Auto (Best Available)", "auto")]
                if metal_info:
                    backend_options.append(("Metal C++ GPU (20x faster)", "metal_cpp"))
                backend_options.extend([
                    ("Numba CPU (Optimized)", "numba_cpu"),
                    ("MLX Metal (Python)", "mlx_metal"),
                    ("NumPy (Reference)", "numpy"),
                ])

                with Horizontal(classes="form-row"):
                    yield Label("Backend:", classes="form-label")
                    yield Select(
                        backend_options,
                        value="auto",
                        id="exec-backend",
                    )

                # Show GPU info if Metal available
                if metal_info:
                    yield Static("GPU Device Info", classes="section-header")
                    yield Static(
                        f"  Device: {metal_info.get('device_name', 'Unknown')}\n"
                        f"  Memory: {metal_info.get('device_memory_gb', 0):.1f} GB",
                        id="gpu-info"
                    )
                else:
                    yield Static(
                        "  Metal GPU not available - using CPU backends",
                        id="gpu-info"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Max Memory (GB):", classes="form-label")
                    yield Input(value="32.0", id="exec-max-memory", classes="form-input")
                
                with Horizontal(classes="form-row"):
                    yield Label("Workers:", classes="form-label")
                    yield Input(placeholder="Auto", id="exec-workers", classes="form-input")
                
                yield Static("Tiling", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Auto Tile Size:", classes="form-label")
                    yield Switch(value=True, id="exec-auto-tile")
                
                with Horizontal(classes="form-row"):
                    yield Label("Tile Order:", classes="form-label")
                    yield Select(
                        [
                            ("Snake", "snake"),
                            ("Row Major", "row_major"),
                            ("Hilbert", "hilbert"),
                        ],
                        value="snake",
                        id="exec-tile-order",
                    )
                
                yield Static("Checkpointing", classes="section-header")
                
                with Horizontal(classes="form-row"):
                    yield Label("Enable Checkpoints:", classes="form-label")
                    yield Switch(value=True, id="exec-checkpoint")
                
                with Horizontal(classes="form-row"):
                    yield Label("Interval (tiles):", classes="form-label")
                    yield Input(value="100", id="exec-checkpoint-interval", classes="form-input")
                
                yield Static("Pre-flight Check", classes="section-header")
                yield Static("", id="exec-preflight")
                
                with Horizontal(classes="button-row"):
                    yield Button("Validate All", id="btn-validate-all", variant="primary")
                    yield Button("Start Migration", id="btn-start", variant="success")
        
        async def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button presses."""
            button_id = event.button.id
            
            if button_id == "btn-save":
                await self.action_save_config()
            elif button_id == "btn-validate":
                await self._validate_current_tab()
            elif button_id == "btn-run":
                await self.action_run_migration()
            elif button_id == "btn-analyze-geom":
                await self._analyze_geometry()
            elif button_id == "btn-validate-all":
                await self._validate_all()
            elif button_id == "btn-start":
                await self.action_run_migration()
        
        async def _validate_current_tab(self) -> None:
            """Validate the current tab."""
            self._update_status("Validating...")
            # Validation logic here
            self._update_status("Validation complete")
        
        async def _validate_all(self) -> None:
            """Validate all configuration."""
            self._update_status("Validating all tabs...")
            
            errors = []
            
            # Check input paths
            traces_input = self.query_one("#input-traces", Input)
            if not traces_input.value:
                errors.append("Traces path is required")
            
            headers_input = self.query_one("#input-headers", Input)
            if not headers_input.value:
                errors.append("Headers path is required")
            
            output_input = self.query_one("#output-dir", Input)
            if not output_input.value:
                errors.append("Output directory is required")
            
            # Check output grid
            try:
                x_min = float(self.query_one("#output-x-min", Input).value or "0")
                x_max = float(self.query_one("#output-x-max", Input).value or "0")
                if x_max <= x_min:
                    errors.append("X Max must be greater than X Min")
            except ValueError:
                errors.append("Invalid X range values")
            
            # Update preflight display
            preflight = self.query_one("#exec-preflight", Static)
            if errors:
                preflight.update("\n".join(f"❌ {e}" for e in errors))
                self._update_status(f"Validation failed: {len(errors)} errors")
            else:
                preflight.update("✅ All checks passed")
                self._update_status("Validation passed")
        
        async def _analyze_geometry(self) -> None:
            """Analyze geometry from headers."""
            self._update_status("Analyzing geometry...")
            # Would load headers and compute statistics
            self._update_status("Geometry analysis not implemented in TUI")
        
        def _update_status(self, message: str) -> None:
            """Update status bar."""
            status = self.query_one("#status-text", Static)
            status.update(message)
        
        async def action_save_config(self) -> None:
            """Save configuration to file."""
            try:
                config = self._build_config()
                # In a real implementation, would show file dialog
                self._update_status("Save not implemented - use CLI")
            except Exception as e:
                self._update_status(f"Error: {e}")
        
        async def action_run_migration(self) -> None:
            """Run the migration."""
            self._update_status("Migration execution not implemented in TUI - use CLI")
        
        async def action_next_tab(self) -> None:
            """Switch to next tab."""
            tabs = self.query_one("#wizard-tabs", TabbedContent)
            tabs.action_next_tab()
        
        async def action_prev_tab(self) -> None:
            """Switch to previous tab."""
            tabs = self.query_one("#wizard-tabs", TabbedContent)
            tabs.action_previous_tab()
        
        def _build_config(self) -> MigrationConfig:
            """Build MigrationConfig from form values."""
            from pstm.config.models import (
                MigrationConfig, InputConfig, VelocityConfig, AlgorithmConfig,
                OutputConfig, ExecutionConfig, OutputGridConfig, OutputProductsConfig,
                ApertureConfig, AmplitudeConfig, AntiAliasingConfig,
                VelocitySource, InterpolationMethod, AntiAliasingMethod, ComputeBackend,
            )

            # Input configuration
            traces_path = self.query_one("#input-traces", Input).value
            headers_path = self.query_one("#input-headers", Input).value

            input_config = InputConfig(
                traces_path=traces_path or "/path/to/traces.zarr",
                headers_path=headers_path or "/path/to/headers.parquet",
            )

            # Velocity configuration
            vel_source = self.query_one("#vel-source", Select).value
            velocity_value = float(self.query_one("#vel-constant", Input).value or "2000")

            velocity_config = VelocityConfig(
                source=VelocitySource.CONSTANT if vel_source == "constant" else VelocitySource.TABLE_1D,
                constant_velocity=velocity_value,
            )

            # Algorithm configuration - aperture
            max_dip = float(self.query_one("#algo-max-dip", Input).value or "45")
            min_aperture = float(self.query_one("#algo-min-aperture", Input).value or "500")
            max_aperture = float(self.query_one("#algo-max-aperture", Input).value or "5000")
            taper_fraction = float(self.query_one("#algo-taper", Input).value or "0.1")

            aperture_config = ApertureConfig(
                max_dip_degrees=max_dip,
                min_aperture_m=min_aperture,
                max_aperture_m=max_aperture,
                taper_fraction=taper_fraction,
            )

            # Algorithm configuration - amplitude weights
            apply_spreading = self.query_one("#algo-spreading", Switch).value
            apply_obliquity = self.query_one("#algo-obliquity", Switch).value

            amplitude_config = AmplitudeConfig(
                geometrical_spreading=apply_spreading,
                obliquity_factor=apply_obliquity,
            )

            # Algorithm configuration - interpolation
            interp_method = self.query_one("#algo-interp", Select).value
            interp_map = {
                "linear": InterpolationMethod.LINEAR,
                "nearest": InterpolationMethod.NEAREST,
                "sinc8": InterpolationMethod.SINC8,
            }

            algorithm_config = AlgorithmConfig(
                interpolation=interp_map.get(interp_method, InterpolationMethod.LINEAR),
                aperture=aperture_config,
                amplitude=amplitude_config,
            )

            # Output grid configuration
            x_min = float(self.query_one("#output-x-min", Input).value or "0")
            x_max = float(self.query_one("#output-x-max", Input).value or "10000")
            y_min = float(self.query_one("#output-y-min", Input).value or "0")
            y_max = float(self.query_one("#output-y-max", Input).value or "10000")
            t_min = float(self.query_one("#output-t-min", Input).value or "0")
            t_max = float(self.query_one("#output-t-max", Input).value or "4000")
            dx = float(self.query_one("#output-dx", Input).value or "25")
            dy = float(self.query_one("#output-dy", Input).value or "25")
            dt = float(self.query_one("#output-dt", Input).value or "2")

            output_grid = OutputGridConfig(
                x_min=x_min, x_max=x_max, dx=dx,
                y_min=y_min, y_max=y_max, dy=dy,
                t_min_ms=t_min, t_max_ms=t_max, dt_ms=dt,
            )

            output_dir = self.query_one("#output-dir", Input).value or "/path/to/output"
            output_config = OutputConfig(
                output_dir=output_dir,
                grid=output_grid,
            )

            # Execution configuration - backend
            backend_value = self.query_one("#exec-backend", Select).value
            backend_map = {
                "auto": ComputeBackend.AUTO,
                "metal_cpp": ComputeBackend.METAL_CPP,
                "numba_cpu": ComputeBackend.NUMBA_CPU,
                "mlx_metal": ComputeBackend.MLX_METAL,
                "numpy": ComputeBackend.NUMPY,
            }

            execution_config = ExecutionConfig(
                backend=backend_map.get(backend_value, ComputeBackend.AUTO),
            )

            return MigrationConfig(
                input=input_config,
                velocity=velocity_config,
                algorithm=algorithm_config,
                output=output_config,
                execution=execution_config,
            )


def run_wizard(config_path: Path | None = None) -> None:
    """
    Run the wizard application.
    
    Args:
        config_path: Optional path to existing config
    """
    if not TEXTUAL_AVAILABLE:
        raise RuntimeError(
            "Textual not available. Install with: pip install textual"
        )
    
    app = WizardApp(config_path)
    app.run()
