"""
QC visualization module for PSTM.

Provides plotting functions for QC analysis results.
Requires matplotlib (optional dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pstm.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    Figure = None


def check_matplotlib_available() -> bool:
    """Check if matplotlib is available."""
    return MATPLOTLIB_AVAILABLE


def plot_fold_map(
    fold_map: NDArray,
    x_bins: NDArray,
    y_bins: NDArray,
    title: str = "Fold Map",
    save_path: Path | str | None = None,
    show: bool = True,
) -> "Figure | None":
    """
    Plot fold map.

    Args:
        fold_map: 2D fold array
        x_bins: X bin edges
        y_bins: Y bin edges
        title: Plot title
        save_path: Optional path to save figure
        show: Show plot interactively

    Returns:
        matplotlib Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available for plotting")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot fold map
    im = ax.pcolormesh(
        x_bins, y_bins, fold_map.T,
        cmap='viridis',
        shading='auto',
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect('equal')

    cbar = fig.colorbar(im, ax=ax, label="Fold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved fold map to {save_path}")

    if show:
        plt.show()

    return fig


def plot_offset_histogram(
    offset: NDArray,
    n_bins: int = 50,
    title: str = "Offset Distribution",
    save_path: Path | str | None = None,
    show: bool = True,
) -> "Figure | None":
    """
    Plot offset histogram.

    Args:
        offset: Offset values
        n_bins: Number of bins
        title: Plot title
        save_path: Optional save path
        show: Show interactively

    Returns:
        matplotlib Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available for plotting")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(offset, bins=n_bins, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Count")
    ax.set_title(title)

    # Add statistics
    stats_text = (
        f"Min: {offset.min():.0f} m\n"
        f"Max: {offset.max():.0f} m\n"
        f"Mean: {offset.mean():.0f} m\n"
        f"Std: {offset.std():.0f} m"
    )
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved offset histogram to {save_path}")

    if show:
        plt.show()

    return fig


def plot_azimuth_rose(
    azimuth_histogram: tuple[NDArray, NDArray],
    title: str = "Azimuth Distribution",
    save_path: Path | str | None = None,
    show: bool = True,
) -> "Figure | None":
    """
    Plot azimuth rose diagram.

    Args:
        azimuth_histogram: (counts, bin_edges) from histogram
        title: Plot title
        save_path: Optional save path
        show: Show interactively

    Returns:
        matplotlib Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available for plotting")
        return None

    counts, bin_edges = azimuth_histogram

    # Create polar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Convert to radians, adjust for North=0
    theta = np.radians((bin_edges[:-1] + bin_edges[1:]) / 2)
    width = np.radians(bin_edges[1] - bin_edges[0])

    # Rotate so 0 is at top (North)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # Clockwise

    ax.bar(theta, counts, width=width, bottom=0, alpha=0.7)
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved azimuth rose to {save_path}")

    if show:
        plt.show()

    return fig


def plot_velocity_profile(
    velocity: NDArray,
    t_axis_ms: NDArray,
    title: str = "Velocity Profile",
    save_path: Path | str | None = None,
    show: bool = True,
) -> "Figure | None":
    """
    Plot 1D velocity profile.

    Args:
        velocity: Velocity values (m/s)
        t_axis_ms: Time axis (ms)
        title: Plot title
        save_path: Optional save path
        show: Show interactively

    Returns:
        matplotlib Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available for plotting")
        return None

    fig, ax = plt.subplots(figsize=(8, 10))

    ax.plot(velocity, t_axis_ms, 'b-', linewidth=2)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.invert_yaxis()  # Time increases downward
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved velocity profile to {save_path}")

    if show:
        plt.show()

    return fig


def plot_velocity_slice(
    velocity_3d: NDArray,
    x_axis: NDArray,
    y_axis: NDArray,
    t_axis_ms: NDArray,
    slice_type: str = "time",
    slice_index: int | None = None,
    slice_value: float | None = None,
    title: str | None = None,
    save_path: Path | str | None = None,
    show: bool = True,
) -> "Figure | None":
    """
    Plot a slice through 3D velocity cube.

    Args:
        velocity_3d: 3D velocity array (nx, ny, nt)
        x_axis, y_axis, t_axis_ms: Coordinate axes
        slice_type: "time", "inline", or "crossline"
        slice_index: Index of slice (or auto-select middle)
        slice_value: Value to slice at (alternative to index)
        title: Plot title
        save_path: Optional save path
        show: Show interactively

    Returns:
        matplotlib Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available for plotting")
        return None

    # Determine slice
    if slice_type == "time":
        if slice_index is None:
            if slice_value is not None:
                slice_index = np.argmin(np.abs(t_axis_ms - slice_value))
            else:
                slice_index = len(t_axis_ms) // 2
        data = velocity_3d[:, :, slice_index]
        extent = [y_axis[0], y_axis[-1], x_axis[0], x_axis[-1]]
        xlabel, ylabel = "Y (m)", "X (m)"
        slice_label = f"t = {t_axis_ms[slice_index]:.0f} ms"

    elif slice_type == "inline":
        if slice_index is None:
            slice_index = len(x_axis) // 2
        data = velocity_3d[slice_index, :, :]
        extent = [t_axis_ms[0], t_axis_ms[-1], y_axis[0], y_axis[-1]]
        xlabel, ylabel = "Time (ms)", "Y (m)"
        slice_label = f"X = {x_axis[slice_index]:.0f} m"

    elif slice_type == "crossline":
        if slice_index is None:
            slice_index = len(y_axis) // 2
        data = velocity_3d[:, slice_index, :]
        extent = [t_axis_ms[0], t_axis_ms[-1], x_axis[0], x_axis[-1]]
        xlabel, ylabel = "Time (ms)", "X (m)"
        slice_label = f"Y = {y_axis[slice_index]:.0f} m"

    else:
        raise ValueError(f"Unknown slice_type: {slice_type}")

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        data,
        aspect='auto',
        extent=extent,
        origin='lower',
        cmap='jet',
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Velocity Slice ({slice_label})")

    cbar = fig.colorbar(im, ax=ax, label="Velocity (m/s)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved velocity slice to {save_path}")

    if show:
        plt.show()

    return fig


def plot_image_slice(
    image: NDArray,
    x_axis: NDArray,
    y_axis: NDArray,
    t_axis_ms: NDArray,
    slice_type: str = "time",
    slice_index: int | None = None,
    clip_percentile: float = 99,
    title: str | None = None,
    save_path: Path | str | None = None,
    show: bool = True,
) -> "Figure | None":
    """
    Plot a slice through migrated image.

    Args:
        image: 3D image array (nx, ny, nt)
        x_axis, y_axis, t_axis_ms: Coordinate axes
        slice_type: "time", "inline", or "crossline"
        slice_index: Index of slice
        clip_percentile: Percentile for color clipping
        title: Plot title
        save_path: Optional save path
        show: Show interactively

    Returns:
        matplotlib Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available for plotting")
        return None

    # Determine slice
    if slice_type == "time":
        if slice_index is None:
            slice_index = len(t_axis_ms) // 2
        data = image[:, :, slice_index]
        extent = [y_axis[0], y_axis[-1], x_axis[0], x_axis[-1]]
        xlabel, ylabel = "Y (m)", "X (m)"
        slice_label = f"t = {t_axis_ms[slice_index]:.0f} ms"

    elif slice_type == "inline":
        if slice_index is None:
            slice_index = len(x_axis) // 2
        data = image[slice_index, :, :].T
        extent = [y_axis[0], y_axis[-1], t_axis_ms[0], t_axis_ms[-1]]
        xlabel, ylabel = "Y (m)", "Time (ms)"
        slice_label = f"X = {x_axis[slice_index]:.0f} m"

    elif slice_type == "crossline":
        if slice_index is None:
            slice_index = len(y_axis) // 2
        data = image[:, slice_index, :].T
        extent = [x_axis[0], x_axis[-1], t_axis_ms[0], t_axis_ms[-1]]
        xlabel, ylabel = "X (m)", "Time (ms)"
        slice_label = f"Y = {y_axis[slice_index]:.0f} m"

    else:
        raise ValueError(f"Unknown slice_type: {slice_type}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Clip for display
    vmax = np.percentile(np.abs(data), clip_percentile)
    vmin = -vmax

    im = ax.imshow(
        data,
        aspect='auto',
        extent=extent,
        origin='lower' if slice_type == "time" else 'upper',
        cmap='seismic',
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Image Slice ({slice_label})")

    cbar = fig.colorbar(im, ax=ax, label="Amplitude")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved image slice to {save_path}")

    if show:
        plt.show()

    return fig


def create_qc_report_figures(
    output_dir: Path,
    fold_map: NDArray | None = None,
    x_bins: NDArray | None = None,
    y_bins: NDArray | None = None,
    offset_histogram: tuple | None = None,
    azimuth_histogram: tuple | None = None,
    velocity_1d: NDArray | None = None,
    t_axis_ms: NDArray | None = None,
    show: bool = False,
) -> list[Path]:
    """
    Create all QC report figures.

    Args:
        output_dir: Directory to save figures
        fold_map: Fold map array
        x_bins, y_bins: Fold map bin edges
        offset_histogram: Offset histogram data
        azimuth_histogram: Azimuth histogram data
        velocity_1d: 1D velocity profile
        t_axis_ms: Time axis
        show: Show figures interactively

    Returns:
        List of saved figure paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if fold_map is not None and x_bins is not None:
        path = output_dir / "fold_map.png"
        plot_fold_map(fold_map, x_bins, y_bins, save_path=path, show=show)
        saved_files.append(path)

    if offset_histogram is not None:
        path = output_dir / "offset_histogram.png"
        counts, bins = offset_histogram
        # Reconstruct offset values for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2
        offset_approx = np.repeat(bin_centers, counts.astype(int))
        if len(offset_approx) > 0:
            plot_offset_histogram(offset_approx, save_path=path, show=show)
            saved_files.append(path)

    if azimuth_histogram is not None:
        path = output_dir / "azimuth_rose.png"
        plot_azimuth_rose(azimuth_histogram, save_path=path, show=show)
        saved_files.append(path)

    if velocity_1d is not None and t_axis_ms is not None:
        path = output_dir / "velocity_profile.png"
        plot_velocity_profile(velocity_1d, t_axis_ms, save_path=path, show=show)
        saved_files.append(path)

    return saved_files
