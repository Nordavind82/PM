"""
Common Image Gathers (CIG) module for PSTM.

Provides functionality for generating and outputting CIGs during migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numpy.typing import NDArray

from pstm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CIGConfig:
    """Configuration for CIG output."""

    enabled: bool = False

    # Offset binning
    offset_bin_centers: list[float] | None = None  # If None, auto-compute
    n_offset_bins: int | None = None
    min_offset: float | None = None
    max_offset: float | None = None

    # Output options
    output_path: Path | None = None
    compression: str = "blosc"
    
    def __post_init__(self):
        """Apply defaults from settings."""
        from pstm.settings import get_settings
        s = get_settings()
        
        if self.n_offset_bins is None:
            self.n_offset_bins = s.cig.n_offset_bins
        if self.min_offset is None:
            self.min_offset = s.cig.min_offset_m
        if self.max_offset is None:
            self.max_offset = s.cig.max_offset_m

    def get_offset_bins(self) -> NDArray[np.float64]:
        """Get offset bin centers."""
        if self.offset_bin_centers is not None:
            return np.array(self.offset_bin_centers)
        return np.linspace(self.min_offset, self.max_offset, self.n_offset_bins)

    def get_offset_edges(self) -> NDArray[np.float64]:
        """Get offset bin edges."""
        centers = self.get_offset_bins()
        if len(centers) < 2:
            return np.array([self.min_offset, self.max_offset])

        # Compute edges from centers
        half_width = (centers[1] - centers[0]) / 2
        edges = np.concatenate([
            [centers[0] - half_width],
            (centers[:-1] + centers[1:]) / 2,
            [centers[-1] + half_width],
        ])
        return edges


@dataclass
class CIGAccumulator:
    """
    Accumulator for Common Image Gathers.

    Accumulates migration output into offset bins.
    """

    # Grid dimensions
    nx: int
    ny: int
    nt: int
    n_offset_bins: int

    # Offset bins
    offset_edges: NDArray[np.float64]
    offset_centers: NDArray[np.float64]

    # Accumulators
    # Shape: (nx, ny, nt, n_offset_bins)
    image: NDArray[np.float64] = field(init=False)
    fold: NDArray[np.int32] = field(init=False)

    def __post_init__(self):
        """Initialize accumulators."""
        shape = (self.nx, self.ny, self.nt, self.n_offset_bins)
        self.image = np.zeros(shape, dtype=np.float64)
        self.fold = np.zeros((self.nx, self.ny, self.n_offset_bins), dtype=np.int32)

        logger.info(f"CIG accumulator: {shape}, {self.n_offset_bins} offset bins")

    def get_offset_bin(self, offset: float) -> int:
        """
        Get bin index for an offset value.

        Args:
            offset: Offset value

        Returns:
            Bin index (or -1 if out of range)
        """
        if offset < self.offset_edges[0] or offset >= self.offset_edges[-1]:
            return -1

        return int(np.searchsorted(self.offset_edges, offset, side='right') - 1)

    def accumulate(
        self,
        ix: int,
        iy: int,
        it: int,
        offset: float,
        amplitude: float,
        weight: float = 1.0,
    ) -> None:
        """
        Accumulate a sample into the appropriate offset bin.

        Args:
            ix, iy, it: Output grid indices
            offset: Trace offset
            amplitude: Weighted amplitude
            weight: Additional weight
        """
        offset_bin = self.get_offset_bin(offset)
        if offset_bin < 0:
            return

        self.image[ix, iy, it, offset_bin] += amplitude * weight
        if it == 0:  # Only count fold once per pillar
            self.fold[ix, iy, offset_bin] += 1

    def accumulate_batch(
        self,
        ix: NDArray[np.int32],
        iy: NDArray[np.int32],
        it: NDArray[np.int32],
        offsets: NDArray[np.float64],
        amplitudes: NDArray[np.float64],
    ) -> None:
        """
        Accumulate a batch of samples.

        Args:
            ix, iy, it: Arrays of grid indices
            offsets: Array of offset values
            amplitudes: Array of weighted amplitudes
        """
        # Get offset bins for all samples
        offset_bins = np.searchsorted(self.offset_edges, offsets, side='right') - 1

        # Mask valid bins
        valid = (offset_bins >= 0) & (offset_bins < self.n_offset_bins)

        # Accumulate using bincount
        for i in range(self.n_offset_bins):
            mask = valid & (offset_bins == i)
            if not np.any(mask):
                continue

            # Use flat indexing
            flat_idx = (
                ix[mask] * self.ny * self.nt +
                iy[mask] * self.nt +
                it[mask]
            )

            np.add.at(
                self.image[:, :, :, i].ravel(),
                flat_idx,
                amplitudes[mask],
            )

    def get_stacked_image(self) -> NDArray[np.float64]:
        """
        Get stack of all offset bins (equivalent to conventional migration).

        Returns:
            Stacked image (nx, ny, nt)
        """
        return np.sum(self.image, axis=3)

    def get_gather_at_location(
        self,
        ix: int,
        iy: int,
    ) -> NDArray[np.float64]:
        """
        Get CIG at a specific location.

        Args:
            ix, iy: Grid indices

        Returns:
            Gather array (nt, n_offset_bins)
        """
        return self.image[ix, iy, :, :]

    def normalize(self) -> None:
        """Normalize CIGs by fold."""
        # Expand fold to 4D for broadcasting
        fold_4d = self.fold[:, :, np.newaxis, :]
        with np.errstate(invalid='ignore', divide='ignore'):
            self.image = np.where(fold_4d > 0, self.image / fold_4d, 0.0)

    @property
    def size_gb(self) -> float:
        """Memory size in GB."""
        return (self.image.nbytes + self.fold.nbytes) / (1024**3)


def create_cig_accumulator(
    nx: int,
    ny: int,
    nt: int,
    config: CIGConfig,
) -> CIGAccumulator:
    """
    Create a CIG accumulator from configuration.

    Args:
        nx, ny, nt: Output grid dimensions
        config: CIG configuration

    Returns:
        CIGAccumulator instance
    """
    offset_centers = config.get_offset_bins()
    offset_edges = config.get_offset_edges()

    return CIGAccumulator(
        nx=nx,
        ny=ny,
        nt=nt,
        n_offset_bins=len(offset_centers),
        offset_edges=offset_edges,
        offset_centers=offset_centers,
    )


def save_cig_to_zarr(
    accumulator: CIGAccumulator,
    output_path: Path | str,
    x_axis: NDArray[np.float64],
    y_axis: NDArray[np.float64],
    t_axis_ms: NDArray[np.float64],
    normalize: bool = True,
    compression: str = "blosc",
) -> None:
    """
    Save CIG accumulator to Zarr file.

    Args:
        accumulator: CIG accumulator
        output_path: Output path
        x_axis, y_axis, t_axis_ms: Coordinate axes
        normalize: Normalize by fold before saving
        compression: Compression codec
    """
    from numcodecs import Blosc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving CIG to {output_path}")

    # Normalize if requested
    if normalize:
        accumulator.normalize()

    # Create Zarr group
    root = zarr.open_group(str(output_path), mode='w')

    # Set compression
    if compression == "blosc":
        compressor = Blosc(cname='zstd', clevel=3)
    else:
        compressor = None

    # Check zarr version for API compatibility
    zarr_major = int(zarr.__version__.split('.')[0])
    
    if zarr_major >= 3:
        # Zarr v3 API - use array creation then assignment
        cig_data = accumulator.image.astype(np.float32)
        cig_arr = root.create_array(
            'cig',
            shape=cig_data.shape,
            chunks=(min(64, accumulator.nx), min(64, accumulator.ny),
                    accumulator.nt, accumulator.n_offset_bins),
            dtype=cig_data.dtype,
        )
        cig_arr[:] = cig_data
        
        fold_arr = root.create_array('fold', shape=accumulator.fold.shape, dtype=accumulator.fold.dtype)
        fold_arr[:] = accumulator.fold
        
        # Save coordinates
        for name, data in [('x_axis', x_axis), ('y_axis', y_axis), 
                           ('t_axis_ms', t_axis_ms), 
                           ('offset_centers', accumulator.offset_centers),
                           ('offset_edges', accumulator.offset_edges)]:
            arr = root.create_array(name, shape=data.shape, dtype=data.dtype)
            arr[:] = data
    else:
        # Zarr v2 API
        root.create_dataset(
            'cig',
            data=accumulator.image.astype(np.float32),
            chunks=(min(64, accumulator.nx), min(64, accumulator.ny),
                    accumulator.nt, accumulator.n_offset_bins),
            compressor=compressor,
        )
        root.create_dataset('fold', data=accumulator.fold, compressor=compressor)
        root.create_dataset('x_axis', data=x_axis)
        root.create_dataset('y_axis', data=y_axis)
        root.create_dataset('t_axis_ms', data=t_axis_ms)
        root.create_dataset('offset_centers', data=accumulator.offset_centers)
        root.create_dataset('offset_edges', data=accumulator.offset_edges)

    # Save metadata
    root.attrs['nx'] = accumulator.nx
    root.attrs['ny'] = accumulator.ny
    root.attrs['nt'] = accumulator.nt
    root.attrs['n_offset_bins'] = accumulator.n_offset_bins
    root.attrs['normalized'] = normalize

    logger.info(f"CIG saved: {accumulator.image.shape}")


def load_cig_from_zarr(
    path: Path | str,
) -> tuple[NDArray, NDArray, dict[str, NDArray]]:
    """
    Load CIG from Zarr file.

    Args:
        path: Path to Zarr CIG file

    Returns:
        Tuple of (cig_data, fold, coordinates_dict)
    """
    root = zarr.open_group(str(path), mode='r')

    cig = root['cig'][:]
    fold = root['fold'][:]

    coords = {
        'x_axis': root['x_axis'][:],
        'y_axis': root['y_axis'][:],
        't_axis_ms': root['t_axis_ms'][:],
        'offset_centers': root['offset_centers'][:],
        'offset_edges': root['offset_edges'][:],
    }

    return cig, fold, coords


# =============================================================================
# CIG Analysis
# =============================================================================


def analyze_cig_flatness(
    cig: NDArray,
    t_axis_ms: NDArray,
    offset_centers: NDArray,
    window_ms: float = 100.0,
) -> dict[str, Any]:
    """
    Analyze CIG flatness for velocity QC.

    Flat CIGs indicate correct velocity, curved CIGs indicate errors.

    Args:
        cig: CIG gather (nt, n_offset_bins)
        t_axis_ms: Time axis
        offset_centers: Offset bin centers
        window_ms: Analysis window size

    Returns:
        Analysis results dictionary
    """
    nt, n_offsets = cig.shape

    # Find strong events
    rms_trace = np.sqrt(np.mean(cig ** 2, axis=1))
    threshold = 0.5 * rms_trace.max()
    strong_times = np.where(rms_trace > threshold)[0]

    if len(strong_times) == 0:
        return {'flatness_score': 0.0, 'n_events': 0, 'curvature': []}

    # Analyze curvature at each strong event
    curvatures = []

    for it in strong_times:
        # Get event pick at each offset (max amplitude)
        window_start = max(0, it - int(window_ms / 2 / (t_axis_ms[1] - t_axis_ms[0])))
        window_end = min(nt, it + int(window_ms / 2 / (t_axis_ms[1] - t_axis_ms[0])))

        picks = []
        for io in range(n_offsets):
            window = cig[window_start:window_end, io]
            if len(window) > 0:
                local_max = window_start + np.argmax(np.abs(window))
                picks.append(t_axis_ms[local_max])
            else:
                picks.append(np.nan)

        picks = np.array(picks)
        valid = ~np.isnan(picks)

        if np.sum(valid) < 3:
            continue

        # Fit parabola to picks
        coeffs = np.polyfit(
            offset_centers[valid] ** 2,
            picks[valid],
            1,
        )
        curvature = coeffs[0]  # Coefficient of h^2
        curvatures.append({
            'time_ms': t_axis_ms[it],
            'curvature': curvature,
        })

    # Compute flatness score (lower curvature = more flat = better)
    if curvatures:
        mean_curvature = np.mean([abs(c['curvature']) for c in curvatures])
        # Normalize to 0-1 score (1 = perfectly flat)
        flatness_score = 1.0 / (1.0 + mean_curvature * 1e6)
    else:
        flatness_score = 0.0

    return {
        'flatness_score': flatness_score,
        'n_events': len(curvatures),
        'curvature': curvatures,
    }


def compute_semblance(
    cig: NDArray,
    t_axis_ms: NDArray,
    offset_centers: NDArray,
    velocity_range: tuple[float, float] = (1500, 5000),
    n_velocities: int = 100,
    window_samples: int = 11,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Compute semblance panel from CIG for velocity analysis.

    Args:
        cig: CIG gather (nt, n_offset_bins)
        t_axis_ms: Time axis
        offset_centers: Offset bin centers
        velocity_range: Velocity scan range
        n_velocities: Number of velocities to scan
        window_samples: Semblance window size

    Returns:
        Tuple of (semblance, t_axis, v_axis)
    """
    nt, n_offsets = cig.shape
    dt_ms = t_axis_ms[1] - t_axis_ms[0]

    velocities = np.linspace(velocity_range[0], velocity_range[1], n_velocities)
    semblance = np.zeros((nt, n_velocities))

    half_window = window_samples // 2

    for iv, v in enumerate(velocities):
        for it in range(half_window, nt - half_window):
            t0_ms = t_axis_ms[it]
            t0_s = t0_ms / 1000.0

            # Compute NMO times for each offset
            nmo_times_ms = np.sqrt(t0_ms ** 2 + (offset_centers / v) ** 2 * 1e6)
            nmo_samples = ((nmo_times_ms - t_axis_ms[0]) / dt_ms).astype(int)

            # Gather values at NMO times
            gathered = np.zeros(n_offsets)
            valid_count = 0

            for io, it_nmo in enumerate(nmo_samples):
                if 0 <= it_nmo < nt:
                    gathered[io] = cig[it_nmo, io]
                    valid_count += 1

            if valid_count < 2:
                continue

            # Compute semblance in window
            num = 0.0
            den = 0.0

            for iw in range(-half_window, half_window + 1):
                it_w = it + iw
                if 0 <= it_w < nt:
                    stack = np.sum(gathered)
                    num += stack ** 2
                    den += np.sum(gathered ** 2)

            if den > 0:
                semblance[it, iv] = num / (n_offsets * den)

    return semblance, t_axis_ms, velocities
