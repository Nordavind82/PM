"""
Synthetic Common Offset Gather Generator with Point Diffractor.

Generates synthetic prestack data for testing PSTM algorithms.
Supports multiple offset-azimuth configurations and export to SEG-Y or Zarr/Parquet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from pstm.utils.logging import get_logger

logger = get_logger(__name__)


class OffsetAzimuthMode(Enum):
    """Mode for specifying offset-azimuth planes."""
    OFFSET_AZIMUTH = "offset_azimuth"  # Specify offset and azimuth angle
    OFFSET_XY = "offset_xy"  # Specify offset_x and offset_y components


@dataclass
class DiffractorLocation:
    """Point diffractor location."""
    x: float  # X coordinate (m)
    y: float  # Y coordinate (m)
    z: float  # Depth (m) - will be converted to t0 using velocity
    amplitude: float = 1.0  # Relative amplitude


@dataclass
class SurveyGeometry:
    """Survey area geometry."""
    x_min: float  # Minimum X (m)
    x_max: float  # Maximum X (m)
    y_min: float  # Minimum Y (m)
    y_max: float  # Maximum Y (m)
    dx: float = 25.0  # Inline spacing (m)
    dy: float = 25.0  # Crossline spacing (m)
    
    @property
    def nx(self) -> int:
        """Number of inline positions."""
        return int((self.x_max - self.x_min) / self.dx) + 1
    
    @property
    def ny(self) -> int:
        """Number of crossline positions."""
        return int((self.y_max - self.y_min) / self.dy) + 1
    
    @property
    def n_positions(self) -> int:
        """Total number of midpoint positions."""
        return self.nx * self.ny
    
    def get_midpoint_grid(self) -> tuple[NDArray, NDArray]:
        """Get 2D grid of midpoint coordinates."""
        x = np.arange(self.x_min, self.x_max + self.dx/2, self.dx)
        y = np.arange(self.y_min, self.y_max + self.dy/2, self.dy)
        return np.meshgrid(x, y, indexing='ij')


@dataclass
class OffsetAzimuthPlane:
    """Definition of a single offset-azimuth plane."""
    
    # Option 1: Offset and azimuth
    offset: float | None = None  # Total offset (m)
    azimuth_deg: float | None = None  # Azimuth from North, clockwise (degrees)
    
    # Option 2: Offset components
    offset_x: float | None = None  # Offset in X direction (m)
    offset_y: float | None = None  # Offset in Y direction (m)
    
    # Metadata
    name: str = ""
    
    def __post_init__(self):
        """Validate and compute derived quantities."""
        if self.offset is not None and self.azimuth_deg is not None:
            # Compute offset_x, offset_y from offset and azimuth
            # Azimuth: 0=North(+Y), 90=East(+X)
            az_rad = np.radians(self.azimuth_deg)
            self.offset_x = self.offset * np.sin(az_rad)
            self.offset_y = self.offset * np.cos(az_rad)
        elif self.offset_x is not None and self.offset_y is not None:
            # Compute offset and azimuth from components
            self.offset = np.sqrt(self.offset_x**2 + self.offset_y**2)
            self.azimuth_deg = np.degrees(np.arctan2(self.offset_x, self.offset_y))
            if self.azimuth_deg < 0:
                self.azimuth_deg += 360
        else:
            raise ValueError("Must specify either (offset, azimuth_deg) or (offset_x, offset_y)")
        
        if not self.name:
            self.name = f"off{self.offset:.0f}_az{self.azimuth_deg:.0f}"
    
    def get_source_receiver_offsets(self) -> tuple[float, float, float, float]:
        """
        Get source and receiver offsets from midpoint.
        
        Returns:
            (src_dx, src_dy, rec_dx, rec_dy) - offsets from midpoint
        """
        # Source is at midpoint - half_offset
        # Receiver is at midpoint + half_offset
        half_x = self.offset_x / 2
        half_y = self.offset_y / 2
        return (-half_x, -half_y, half_x, half_y)


@dataclass
class TraceParameters:
    """Trace recording parameters."""
    n_samples: int = 2001  # Number of samples per trace
    dt_ms: float = 2.0  # Sample interval (ms)
    t_start_ms: float = 0.0  # Start time (ms)
    
    @property
    def t_end_ms(self) -> float:
        """End time in ms."""
        return self.t_start_ms + (self.n_samples - 1) * self.dt_ms
    
    @property
    def time_axis_ms(self) -> NDArray:
        """Time axis in ms."""
        return np.arange(self.n_samples) * self.dt_ms + self.t_start_ms
    
    @property
    def time_axis_s(self) -> NDArray:
        """Time axis in seconds."""
        return self.time_axis_ms / 1000.0


@dataclass
class WaveletParameters:
    """Source wavelet parameters."""
    type: str = "ricker"  # Wavelet type: "ricker", "ormsby", "klauder"
    dominant_freq_hz: float = 30.0  # Dominant frequency
    phase_deg: float = 0.0  # Phase rotation
    
    # Ormsby parameters
    f1: float = 5.0
    f2: float = 15.0
    f3: float = 45.0
    f4: float = 60.0


@dataclass
class SyntheticConfig:
    """Complete configuration for synthetic data generation."""
    
    # Diffractor(s)
    diffractors: list[DiffractorLocation] = field(default_factory=list)
    
    # Survey geometry
    survey: SurveyGeometry = field(default_factory=lambda: SurveyGeometry(
        x_min=0, x_max=2000, y_min=0, y_max=2000, dx=25, dy=25
    ))
    
    # Offset-azimuth planes
    offset_azimuth_planes: list[OffsetAzimuthPlane] = field(default_factory=list)
    
    # Trace parameters
    trace_params: TraceParameters = field(default_factory=TraceParameters)
    
    # Wavelet
    wavelet: WaveletParameters = field(default_factory=WaveletParameters)
    
    # Velocity model (constant for simplicity)
    velocity_ms: float = 2000.0  # m/s
    
    # Noise
    noise_level: float = 0.0  # RMS noise level (fraction of signal)
    
    # Random seed for reproducibility
    random_seed: int | None = 42
    
    def add_diffractor(self, x: float, y: float, z: float, amplitude: float = 1.0) -> None:
        """Add a point diffractor."""
        self.diffractors.append(DiffractorLocation(x, y, z, amplitude))
    
    def add_offset_azimuth_plane(
        self,
        offset: float | None = None,
        azimuth_deg: float | None = None,
        offset_x: float | None = None,
        offset_y: float | None = None,
        name: str = "",
    ) -> None:
        """Add an offset-azimuth plane."""
        plane = OffsetAzimuthPlane(
            offset=offset,
            azimuth_deg=azimuth_deg,
            offset_x=offset_x,
            offset_y=offset_y,
            name=name,
        )
        self.offset_azimuth_planes.append(plane)
    
    def add_offset_azimuth_range(
        self,
        offsets: list[float],
        azimuths: list[float],
    ) -> None:
        """Add multiple planes from offset and azimuth ranges."""
        for offset in offsets:
            for azimuth in azimuths:
                self.add_offset_azimuth_plane(offset=offset, azimuth_deg=azimuth)
    
    @property
    def n_planes(self) -> int:
        """Number of offset-azimuth planes."""
        return len(self.offset_azimuth_planes)
    
    @property
    def n_traces_total(self) -> int:
        """Total number of traces."""
        return self.survey.n_positions * self.n_planes


def generate_ricker_wavelet(
    freq_hz: float,
    dt_ms: float,
    length_ms: float = 200.0,
) -> NDArray:
    """
    Generate Ricker wavelet.
    
    Args:
        freq_hz: Dominant frequency
        dt_ms: Sample interval (ms)
        length_ms: Wavelet length (ms)
        
    Returns:
        Wavelet samples
    """
    dt_s = dt_ms / 1000.0
    n_samples = int(length_ms / dt_ms) + 1
    t = (np.arange(n_samples) - n_samples // 2) * dt_s
    
    # Ricker wavelet
    a = (np.pi * freq_hz) ** 2
    wavelet = (1 - 2 * a * t**2) * np.exp(-a * t**2)
    
    return wavelet


def generate_ormsby_wavelet(
    f1: float, f2: float, f3: float, f4: float,
    dt_ms: float,
    length_ms: float = 200.0,
) -> NDArray:
    """
    Generate Ormsby wavelet.
    
    Args:
        f1, f2, f3, f4: Corner frequencies (Hz)
        dt_ms: Sample interval (ms)
        length_ms: Wavelet length (ms)
        
    Returns:
        Wavelet samples
    """
    dt_s = dt_ms / 1000.0
    n_samples = int(length_ms / dt_ms) + 1
    t = (np.arange(n_samples) - n_samples // 2) * dt_s
    
    # Avoid division by zero
    t = np.where(t == 0, 1e-10, t)
    
    def sinc_sq(f, t):
        return (np.pi * f)**2 * np.sinc(f * t)**2
    
    wavelet = (
        (sinc_sq(f4, t) / (f4 - f3)) - (sinc_sq(f3, t) / (f4 - f3)) -
        (sinc_sq(f2, t) / (f2 - f1)) + (sinc_sq(f1, t) / (f2 - f1))
    )
    
    # Normalize
    wavelet = wavelet / np.max(np.abs(wavelet))
    
    return wavelet


def compute_dsr_travel_time(
    source_x: float, source_y: float,
    receiver_x: float, receiver_y: float,
    diffractor_x: float, diffractor_y: float,
    t0_s: float,
    velocity: float,
) -> float:
    """
    Compute double square root travel time.
    
    t = sqrt(t0^2 + (dist_s/v)^2) + sqrt(t0^2 + (dist_r/v)^2)
    
    Args:
        source_x, source_y: Source coordinates
        receiver_x, receiver_y: Receiver coordinates
        diffractor_x, diffractor_y: Diffractor coordinates
        t0_s: Zero-offset time (seconds)
        velocity: RMS velocity (m/s)
        
    Returns:
        Travel time in seconds
    """
    dist_s = np.sqrt((source_x - diffractor_x)**2 + (source_y - diffractor_y)**2)
    dist_r = np.sqrt((receiver_x - diffractor_x)**2 + (receiver_y - diffractor_y)**2)
    
    t_s = np.sqrt(t0_s**2 + (dist_s / velocity)**2)
    t_r = np.sqrt(t0_s**2 + (dist_r / velocity)**2)
    
    return t_s + t_r


def generate_diffractor_response(
    config: SyntheticConfig,
    plane: OffsetAzimuthPlane,
    midpoint_x: NDArray,
    midpoint_y: NDArray,
) -> NDArray:
    """
    Generate diffractor response for one offset-azimuth plane.
    
    Args:
        config: Synthetic configuration
        plane: Offset-azimuth plane
        midpoint_x: 2D array of midpoint X coordinates
        midpoint_y: 2D array of midpoint Y coordinates
        
    Returns:
        Trace data array (nx, ny, nt)
    """
    nx, ny = midpoint_x.shape
    nt = config.trace_params.n_samples
    dt_s = config.trace_params.dt_ms / 1000.0
    t_start_s = config.trace_params.t_start_ms / 1000.0
    
    # Initialize output
    traces = np.zeros((nx, ny, nt), dtype=np.float32)
    
    # Get source/receiver offsets from midpoint
    src_dx, src_dy, rec_dx, rec_dy = plane.get_source_receiver_offsets()
    
    # Generate wavelet
    if config.wavelet.type == "ricker":
        wavelet = generate_ricker_wavelet(
            config.wavelet.dominant_freq_hz,
            config.trace_params.dt_ms,
        )
    elif config.wavelet.type == "ormsby":
        wavelet = generate_ormsby_wavelet(
            config.wavelet.f1, config.wavelet.f2,
            config.wavelet.f3, config.wavelet.f4,
            config.trace_params.dt_ms,
        )
    else:
        wavelet = generate_ricker_wavelet(
            config.wavelet.dominant_freq_hz,
            config.trace_params.dt_ms,
        )
    
    wavelet_len = len(wavelet)
    wavelet_center = wavelet_len // 2
    
    # Process each diffractor
    for diff in config.diffractors:
        # Convert depth to t0 (two-way time at zero offset)
        t0_s = 2 * diff.z / config.velocity_ms
        
        # Compute travel times for all midpoints
        for ix in range(nx):
            for iy in range(ny):
                mx = midpoint_x[ix, iy]
                my = midpoint_y[ix, iy]
                
                sx = mx + src_dx
                sy = my + src_dy
                rx = mx + rec_dx
                ry = my + rec_dy
                
                # Compute travel time using DSR equation
                t_travel = compute_dsr_travel_time(
                    sx, sy, rx, ry,
                    diff.x, diff.y, t0_s,
                    config.velocity_ms,
                )
                
                # Convert to sample index (relative to trace start)
                sample_idx = (t_travel - t_start_s) / dt_s
                
                # Check if within valid range (with margin for wavelet)
                if wavelet_center <= sample_idx < nt - wavelet_center - 1:
                    # Add wavelet centered at travel time
                    idx_int = int(sample_idx)
                    frac = sample_idx - idx_int
                    
                    # Apply wavelet with linear interpolation for fractional sample
                    for iw in range(wavelet_len):
                        out_idx = idx_int + iw - wavelet_center
                        if 0 <= out_idx < nt - 1:
                            # Linear interpolation between samples
                            traces[ix, iy, out_idx] += diff.amplitude * wavelet[iw] * (1 - frac)
                            traces[ix, iy, out_idx + 1] += diff.amplitude * wavelet[iw] * frac
    
    return traces


@dataclass
class SyntheticGatherResult:
    """Result from synthetic gather generation."""
    
    # Trace data: (n_traces, n_samples)
    traces: NDArray
    
    # Headers
    trace_indices: NDArray  # Global trace index
    source_x: NDArray
    source_y: NDArray
    receiver_x: NDArray
    receiver_y: NDArray
    midpoint_x: NDArray
    midpoint_y: NDArray
    offset: NDArray
    azimuth: NDArray
    offset_x: NDArray
    offset_y: NDArray
    inline: NDArray
    crossline: NDArray
    plane_id: NDArray  # Offset-azimuth plane index
    
    # Metadata
    n_samples: int
    dt_ms: float
    t_start_ms: float
    velocity_ms: float
    
    # Config reference
    config: SyntheticConfig
    
    @property
    def n_traces(self) -> int:
        return self.traces.shape[0]
    
    def get_headers_dict(self) -> dict[str, NDArray]:
        """Get all headers as dictionary."""
        return {
            'trace_idx': self.trace_indices,
            'SOU_X': self.source_x,
            'SOU_Y': self.source_y,
            'REC_X': self.receiver_x,
            'REC_Y': self.receiver_y,
            'CDP_X': self.midpoint_x,
            'CDP_Y': self.midpoint_y,
            'OFFSET': self.offset,
            'AZIMUTH': self.azimuth,
            'OFFSET_X': self.offset_x,
            'OFFSET_Y': self.offset_y,
            'INLINE': self.inline,
            'CROSSLINE': self.crossline,
            'PLANE_ID': self.plane_id,
        }


def generate_synthetic_gathers(config: SyntheticConfig) -> SyntheticGatherResult:
    """
    Generate complete synthetic common offset gathers.
    
    Args:
        config: Synthetic configuration
        
    Returns:
        SyntheticGatherResult with traces and headers
    """
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
    
    logger.info(f"Generating synthetic gathers:")
    logger.info(f"  Survey: {config.survey.nx} x {config.survey.ny} midpoints")
    logger.info(f"  Planes: {config.n_planes}")
    logger.info(f"  Total traces: {config.n_traces_total}")
    logger.info(f"  Diffractors: {len(config.diffractors)}")
    
    # Get midpoint grid
    midpoint_x_grid, midpoint_y_grid = config.survey.get_midpoint_grid()
    nx, ny = midpoint_x_grid.shape
    n_positions = nx * ny
    
    # Initialize output arrays
    n_traces_total = config.n_traces_total
    nt = config.trace_params.n_samples
    
    all_traces = np.zeros((n_traces_total, nt), dtype=np.float32)
    
    # Header arrays
    trace_indices = np.arange(n_traces_total, dtype=np.int64)
    source_x = np.zeros(n_traces_total, dtype=np.float64)
    source_y = np.zeros(n_traces_total, dtype=np.float64)
    receiver_x = np.zeros(n_traces_total, dtype=np.float64)
    receiver_y = np.zeros(n_traces_total, dtype=np.float64)
    midpoint_x = np.zeros(n_traces_total, dtype=np.float64)
    midpoint_y = np.zeros(n_traces_total, dtype=np.float64)
    offset = np.zeros(n_traces_total, dtype=np.float64)
    azimuth = np.zeros(n_traces_total, dtype=np.float64)
    offset_x = np.zeros(n_traces_total, dtype=np.float64)
    offset_y = np.zeros(n_traces_total, dtype=np.float64)
    inline = np.zeros(n_traces_total, dtype=np.int32)
    crossline = np.zeros(n_traces_total, dtype=np.int32)
    plane_id = np.zeros(n_traces_total, dtype=np.int32)
    
    # Generate data for each offset-azimuth plane
    trace_idx = 0
    
    for ip, plane in enumerate(config.offset_azimuth_planes):
        logger.info(f"  Generating plane {ip+1}/{config.n_planes}: {plane.name}")
        
        # Generate diffractor response for this plane
        traces_3d = generate_diffractor_response(
            config, plane, midpoint_x_grid, midpoint_y_grid
        )
        
        # Get source/receiver offsets
        src_dx, src_dy, rec_dx, rec_dy = plane.get_source_receiver_offsets()
        
        # Flatten and store
        for ix in range(nx):
            for iy in range(ny):
                mx = midpoint_x_grid[ix, iy]
                my = midpoint_y_grid[ix, iy]
                
                all_traces[trace_idx, :] = traces_3d[ix, iy, :]
                
                source_x[trace_idx] = mx + src_dx
                source_y[trace_idx] = my + src_dy
                receiver_x[trace_idx] = mx + rec_dx
                receiver_y[trace_idx] = my + rec_dy
                midpoint_x[trace_idx] = mx
                midpoint_y[trace_idx] = my
                offset[trace_idx] = plane.offset
                azimuth[trace_idx] = plane.azimuth_deg
                offset_x[trace_idx] = plane.offset_x
                offset_y[trace_idx] = plane.offset_y
                inline[trace_idx] = ix
                crossline[trace_idx] = iy
                plane_id[trace_idx] = ip
                
                trace_idx += 1
    
    # Add noise if requested
    if config.noise_level > 0:
        signal_rms = np.sqrt(np.mean(all_traces**2))
        noise_rms = config.noise_level * signal_rms
        noise = np.random.randn(*all_traces.shape).astype(np.float32) * noise_rms
        all_traces += noise
    
    logger.info(f"  Generation complete: {trace_idx} traces")
    
    return SyntheticGatherResult(
        traces=all_traces,
        trace_indices=trace_indices,
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        midpoint_x=midpoint_x,
        midpoint_y=midpoint_y,
        offset=offset,
        azimuth=azimuth,
        offset_x=offset_x,
        offset_y=offset_y,
        inline=inline,
        crossline=crossline,
        plane_id=plane_id,
        n_samples=nt,
        dt_ms=config.trace_params.dt_ms,
        t_start_ms=config.trace_params.t_start_ms,
        velocity_ms=config.velocity_ms,
        config=config,
    )


# =============================================================================
# Export Functions
# =============================================================================


def export_to_zarr_parquet(
    result: SyntheticGatherResult,
    output_dir: Path | str,
    traces_name: str = "traces.zarr",
    headers_name: str = "headers.parquet",
    compression: str = "blosc",
) -> tuple[Path, Path]:
    """
    Export synthetic data to Zarr (traces) and Parquet (headers).

    Uses the standard PSTM data format compatible with the UI wizard:
    - Zarr: 2D array (n_traces, n_samples) with attributes for sample_rate_ms, start_time_ms
    - Parquet: Headers with columns trace_idx, SOU_X, SOU_Y, REC_X, REC_Y, CDP_X, CDP_Y, OFFSET, AZIMUTH

    Args:
        result: Synthetic gather result
        output_dir: Output directory
        traces_name: Name for traces Zarr file
        headers_name: Name for headers Parquet file
        compression: Compression for Zarr (default: blosc)

    Returns:
        Tuple of (traces_path, headers_path)
    """
    from pstm.data.zarr_reader import create_zarr_traces
    from pstm.data.parquet_headers import create_parquet_headers

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces_path = output_dir / traces_name
    headers_path = output_dir / headers_name

    logger.info(f"Exporting to Zarr/Parquet (PSTM format): {output_dir}")

    # Export traces to Zarr using PSTM utility
    z = create_zarr_traces(
        path=traces_path,
        n_traces=result.n_traces,
        n_samples=result.n_samples,
        sample_rate_ms=result.dt_ms,
        start_time_ms=result.t_start_ms,
        compressor=compression,
    )
    z[:] = result.traces

    # Add additional synthetic-specific attributes
    z.attrs["velocity_ms"] = result.velocity_ms
    z.attrs["synthetic"] = True

    # Export headers to Parquet using PSTM utility
    # Prepare additional columns for synthetic data
    additional_columns = {
        "OFFSET_X": result.offset_x,
        "OFFSET_Y": result.offset_y,
        "INLINE": result.inline,
        "CROSSLINE": result.crossline,
        "PLANE_ID": result.plane_id,
    }

    # For synthetic data, use plane_id as shot_id (each offset-azimuth plane is a "shot")
    shot_ids = result.plane_id.astype(np.int32)

    create_parquet_headers(
        path=headers_path,
        trace_indices=result.trace_indices,
        source_x=result.source_x,
        source_y=result.source_y,
        receiver_x=result.receiver_x,
        receiver_y=result.receiver_y,
        shot_ids=shot_ids,
        additional_columns=additional_columns,
    )

    # Optionally save supplementary metadata as JSON (for reference)
    metadata = {
        'n_traces': result.n_traces,
        'n_samples': result.n_samples,
        'dt_ms': result.dt_ms,
        't_start_ms': result.t_start_ms,
        'velocity_ms': result.velocity_ms,
        'format': 'pstm_zarr_parquet',
    }
    metadata_path = output_dir / f"{traces_name.replace('.zarr', '')}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Traces: {traces_path} ({result.traces.nbytes / 1e6:.1f} MB)")
    logger.info(f"  Headers: {headers_path}")
    logger.info(f"  Metadata: {metadata_path}")

    return traces_path, headers_path


def export_to_segy(
    result: SyntheticGatherResult,
    output_path: Path | str,
    format_code: int = 1,  # 1=IBM float, 5=IEEE float
) -> Path:
    """
    Export synthetic data to SEG-Y format.
    
    Args:
        result: Synthetic gather result
        output_path: Output SEG-Y file path
        format_code: SEG-Y data format code (1=IBM, 5=IEEE)
        
    Returns:
        Output path
    """
    try:
        import segyio
    except ImportError:
        raise ImportError("segyio required for SEG-Y export. Install with: pip install segyio")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting to SEG-Y: {output_path}")
    
    # Create SEG-Y spec
    spec = segyio.spec()
    spec.sorting = 1  # No particular sorting
    spec.format = format_code
    spec.samples = result.config.trace_params.time_axis_ms
    spec.tracecount = result.n_traces
    
    with segyio.create(str(output_path), spec) as f:
        # Write traces
        for i in range(result.n_traces):
            f.trace[i] = result.traces[i]
            
            # Write trace headers
            f.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                segyio.TraceField.SourceX: int(result.source_x[i]),
                segyio.TraceField.SourceY: int(result.source_y[i]),
                segyio.TraceField.GroupX: int(result.receiver_x[i]),
                segyio.TraceField.GroupY: int(result.receiver_y[i]),
                segyio.TraceField.CDP_X: int(result.midpoint_x[i]),
                segyio.TraceField.CDP_Y: int(result.midpoint_y[i]),
                segyio.TraceField.offset: int(result.offset[i]),
                segyio.TraceField.INLINE_3D: int(result.inline[i]),
                segyio.TraceField.CROSSLINE_3D: int(result.crossline[i]),
                segyio.TraceField.SourceDepth: 0,
                segyio.TraceField.ReceiverDatumElevation: 0,
                segyio.TraceField.SourceSurfaceElevation: 0,
                segyio.TraceField.ReceiverGroupElevation: 0,
                segyio.TraceField.CoordinateUnits: 1,  # Meters
            }
        
        # Update binary header
        f.bin[segyio.BinField.Interval] = int(result.dt_ms * 1000)  # microseconds
        f.bin[segyio.BinField.Samples] = result.n_samples
        f.bin[segyio.BinField.Format] = format_code
    
    logger.info(f"  Written: {result.n_traces} traces, {result.n_samples} samples")
    
    return output_path


# =============================================================================
# Convenience Functions
# =============================================================================


def create_simple_synthetic(
    diffractor_x: float = 1000.0,
    diffractor_y: float = 1000.0,
    diffractor_z: float = 1000.0,
    survey_extent: float = 2000.0,
    grid_spacing: float = 25.0,
    offsets: list[float] | None = None,
    azimuths: list[float] | None = None,
    velocity: float = 2000.0,
    n_samples: int = 2001,
    dt_ms: float = 2.0,
    wavelet_freq: float = 30.0,
    noise_level: float = 0.0,
) -> SyntheticGatherResult:
    """
    Create simple synthetic data with single diffractor.
    
    Args:
        diffractor_x, diffractor_y, diffractor_z: Diffractor location
        survey_extent: Survey area size (square)
        grid_spacing: Midpoint grid spacing
        offsets: List of offsets (m), default [500, 1000, 1500, 2000]
        azimuths: List of azimuths (degrees), default [0, 45, 90, 135]
        velocity: Constant velocity (m/s)
        n_samples: Samples per trace
        dt_ms: Sample interval
        wavelet_freq: Ricker wavelet frequency
        noise_level: Noise level (fraction of signal)
        
    Returns:
        SyntheticGatherResult
    """
    if offsets is None:
        offsets = [500, 1000, 1500, 2000]
    if azimuths is None:
        azimuths = [0, 45, 90, 135]
    
    config = SyntheticConfig(
        survey=SurveyGeometry(
            x_min=0, x_max=survey_extent,
            y_min=0, y_max=survey_extent,
            dx=grid_spacing, dy=grid_spacing,
        ),
        trace_params=TraceParameters(
            n_samples=n_samples,
            dt_ms=dt_ms,
        ),
        wavelet=WaveletParameters(
            type="ricker",
            dominant_freq_hz=wavelet_freq,
        ),
        velocity_ms=velocity,
        noise_level=noise_level,
    )
    
    # Add diffractor
    config.add_diffractor(diffractor_x, diffractor_y, diffractor_z)
    
    # Add offset-azimuth planes
    config.add_offset_azimuth_range(offsets, azimuths)
    
    return generate_synthetic_gathers(config)


def create_multi_diffractor_synthetic(
    diffractor_locations: list[tuple[float, float, float]],
    survey_x_range: tuple[float, float] = (0, 3000),
    survey_y_range: tuple[float, float] = (0, 3000),
    grid_spacing: float = 25.0,
    offset_x_values: list[float] | None = None,
    offset_y_values: list[float] | None = None,
    velocity: float = 2000.0,
    n_samples: int = 2501,
    dt_ms: float = 2.0,
) -> SyntheticGatherResult:
    """
    Create synthetic data with multiple diffractors using offset_x/offset_y specification.
    
    Args:
        diffractor_locations: List of (x, y, z) tuples
        survey_x_range: (x_min, x_max)
        survey_y_range: (y_min, y_max)
        grid_spacing: Midpoint spacing
        offset_x_values: List of offset X components
        offset_y_values: List of offset Y components
        velocity: Velocity (m/s)
        n_samples: Samples per trace
        dt_ms: Sample interval
        
    Returns:
        SyntheticGatherResult
    """
    if offset_x_values is None:
        offset_x_values = [-500, 0, 500]
    if offset_y_values is None:
        offset_y_values = [-500, 0, 500]
    
    config = SyntheticConfig(
        survey=SurveyGeometry(
            x_min=survey_x_range[0], x_max=survey_x_range[1],
            y_min=survey_y_range[0], y_max=survey_y_range[1],
            dx=grid_spacing, dy=grid_spacing,
        ),
        trace_params=TraceParameters(
            n_samples=n_samples,
            dt_ms=dt_ms,
        ),
        velocity_ms=velocity,
    )
    
    # Add diffractors
    for x, y, z in diffractor_locations:
        config.add_diffractor(x, y, z)
    
    # Add offset planes using offset_x/offset_y
    for ox in offset_x_values:
        for oy in offset_y_values:
            if ox == 0 and oy == 0:
                continue  # Skip zero offset
            config.add_offset_azimuth_plane(offset_x=ox, offset_y=oy)
    
    return generate_synthetic_gathers(config)


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """Command line interface for synthetic data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic common offset gathers with point diffractor"
    )
    
    # Diffractor
    parser.add_argument("--diffractor-x", type=float, default=1000.0,
                        help="Diffractor X coordinate (m)")
    parser.add_argument("--diffractor-y", type=float, default=1000.0,
                        help="Diffractor Y coordinate (m)")
    parser.add_argument("--diffractor-z", type=float, default=1000.0,
                        help="Diffractor depth (m)")
    
    # Survey
    parser.add_argument("--survey-extent", type=float, default=2000.0,
                        help="Survey extent (m)")
    parser.add_argument("--grid-spacing", type=float, default=25.0,
                        help="Midpoint grid spacing (m)")
    
    # Offsets and azimuths
    parser.add_argument("--offsets", type=float, nargs="+",
                        default=[500, 1000, 1500, 2000],
                        help="Offset values (m)")
    parser.add_argument("--azimuths", type=float, nargs="+",
                        default=[0, 45, 90, 135],
                        help="Azimuth values (degrees)")
    
    # Or offset_x/offset_y mode
    parser.add_argument("--offset-xy-mode", action="store_true",
                        help="Use offset_x/offset_y instead of offset/azimuth")
    parser.add_argument("--offset-x", type=float, nargs="+",
                        help="Offset X components (m)")
    parser.add_argument("--offset-y", type=float, nargs="+",
                        help="Offset Y components (m)")
    
    # Trace parameters
    parser.add_argument("--n-samples", type=int, default=2001,
                        help="Samples per trace")
    parser.add_argument("--dt-ms", type=float, default=2.0,
                        help="Sample interval (ms)")
    parser.add_argument("--velocity", type=float, default=2000.0,
                        help="Velocity (m/s)")
    parser.add_argument("--wavelet-freq", type=float, default=30.0,
                        help="Wavelet frequency (Hz)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Noise level (fraction)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./synthetic_data",
                        help="Output directory")
    parser.add_argument("--format", choices=["zarr", "segy", "both"],
                        default="zarr", help="Output format")
    parser.add_argument("--name", type=str, default="synthetic",
                        help="Output file base name")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SyntheticConfig(
        survey=SurveyGeometry(
            x_min=0, x_max=args.survey_extent,
            y_min=0, y_max=args.survey_extent,
            dx=args.grid_spacing, dy=args.grid_spacing,
        ),
        trace_params=TraceParameters(
            n_samples=args.n_samples,
            dt_ms=args.dt_ms,
        ),
        wavelet=WaveletParameters(
            dominant_freq_hz=args.wavelet_freq,
        ),
        velocity_ms=args.velocity,
        noise_level=args.noise,
    )
    
    # Add diffractor
    config.add_diffractor(args.diffractor_x, args.diffractor_y, args.diffractor_z)
    
    # Add offset-azimuth planes
    if args.offset_xy_mode and args.offset_x and args.offset_y:
        for ox in args.offset_x:
            for oy in args.offset_y:
                if ox != 0 or oy != 0:
                    config.add_offset_azimuth_plane(offset_x=ox, offset_y=oy)
    else:
        config.add_offset_azimuth_range(args.offsets, args.azimuths)
    
    # Generate data
    result = generate_synthetic_gathers(config)
    
    # Export
    output_dir = Path(args.output_dir)
    
    if args.format in ["zarr", "both"]:
        export_to_zarr_parquet(
            result, output_dir,
            traces_name=f"{args.name}_traces.zarr",
            headers_name=f"{args.name}_headers.parquet",
        )
    
    if args.format in ["segy", "both"]:
        export_to_segy(result, output_dir / f"{args.name}.sgy")
    
    print(f"\nSynthetic data generation complete!")
    print(f"  Total traces: {result.n_traces}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
