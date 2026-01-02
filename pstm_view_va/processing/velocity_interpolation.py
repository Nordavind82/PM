"""Velocity interpolation and extrapolation for stacking."""

from typing import List, Tuple, Optional, Callable
import numpy as np
from scipy.interpolate import griddata, interp1d
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving images
import matplotlib.pyplot as plt


def interpolate_velocity_along_inline(
    vel_grid,  # VelocityOutputGrid
    il: int,
    xl_range: Tuple[int, int],  # (start, end) crossline range
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate/extrapolate velocities along an inline for all crosslines.

    Args:
        vel_grid: VelocityOutputGrid with sparse velocity picks
        il: Inline number
        xl_range: (start, end) crossline range to interpolate over
        method: Interpolation method ('linear', 'nearest', 'cubic')

    Returns:
        velocities: 2D array (n_xl, n_time) of interpolated velocities
        xl_coords: Array of crossline coordinates
    """
    if vel_grid.velocities is None:
        return None, None

    n_time = len(vel_grid.t_coords)
    xl_start, xl_end = xl_range
    xl_coords_out = np.arange(xl_start, xl_end + 1)
    n_xl = len(xl_coords_out)

    # Find all XL locations with valid velocity at this IL
    valid_xls = []
    valid_vels = []

    # Get the IL index
    il_idx = np.argmin(np.abs(vel_grid.il_coords - il))

    for j, xl in enumerate(vel_grid.xl_coords):
        vel = vel_grid.velocities[il_idx, j, :]
        if not np.all(np.isnan(vel)):
            valid_xls.append(xl)
            valid_vels.append(vel)

    if len(valid_xls) == 0:
        # No valid velocities at this inline - try neighboring inlines
        print(f"[VelInterp] No velocities at IL={il}, searching neighbors...")
        return _interpolate_from_neighbors_il(vel_grid, il, xl_range)

    valid_xls = np.array(valid_xls)
    valid_vels = np.array(valid_vels)  # (n_valid, n_time)

    print(f"[VelInterp] IL={il}: Found {len(valid_xls)} velocity locations at XL={valid_xls.tolist()}")

    # Interpolate for each time sample
    velocities_out = np.zeros((n_xl, n_time), dtype=np.float32)

    for t_idx in range(n_time):
        vel_at_t = valid_vels[:, t_idx]

        if len(valid_xls) == 1:
            # Only one point - use constant value
            velocities_out[:, t_idx] = vel_at_t[0]
        else:
            # Interpolate with extrapolation at edges
            f = interp1d(valid_xls, vel_at_t, kind=method,
                        bounds_error=False, fill_value='extrapolate')
            velocities_out[:, t_idx] = f(xl_coords_out)

    return velocities_out, xl_coords_out


def interpolate_velocity_along_crossline(
    vel_grid,  # VelocityOutputGrid
    xl: int,
    il_range: Tuple[int, int],  # (start, end) inline range
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate/extrapolate velocities along a crossline for all inlines.

    Args:
        vel_grid: VelocityOutputGrid with sparse velocity picks
        xl: Crossline number
        il_range: (start, end) inline range to interpolate over
        method: Interpolation method ('linear', 'nearest', 'cubic')

    Returns:
        velocities: 2D array (n_il, n_time) of interpolated velocities
        il_coords: Array of inline coordinates
    """
    if vel_grid.velocities is None:
        return None, None

    n_time = len(vel_grid.t_coords)
    il_start, il_end = il_range
    il_coords_out = np.arange(il_start, il_end + 1)
    n_il = len(il_coords_out)

    # Find all IL locations with valid velocity at this XL
    valid_ils = []
    valid_vels = []

    # Get the XL index
    xl_idx = np.argmin(np.abs(vel_grid.xl_coords - xl))

    for i, il in enumerate(vel_grid.il_coords):
        vel = vel_grid.velocities[i, xl_idx, :]
        if not np.all(np.isnan(vel)):
            valid_ils.append(il)
            valid_vels.append(vel)

    if len(valid_ils) == 0:
        # No valid velocities at this crossline - try neighboring crosslines
        print(f"[VelInterp] No velocities at XL={xl}, searching neighbors...")
        return _interpolate_from_neighbors_xl(vel_grid, xl, il_range)

    valid_ils = np.array(valid_ils)
    valid_vels = np.array(valid_vels)  # (n_valid, n_time)

    print(f"[VelInterp] XL={xl}: Found {len(valid_ils)} velocity locations at IL={valid_ils.tolist()}")

    # Interpolate for each time sample
    velocities_out = np.zeros((n_il, n_time), dtype=np.float32)

    for t_idx in range(n_time):
        vel_at_t = valid_vels[:, t_idx]

        if len(valid_ils) == 1:
            # Only one point - use constant value
            velocities_out[:, t_idx] = vel_at_t[0]
        else:
            # Interpolate with extrapolation at edges
            f = interp1d(valid_ils, vel_at_t, kind=method,
                        bounds_error=False, fill_value='extrapolate')
            velocities_out[:, t_idx] = f(il_coords_out)

    return velocities_out, il_coords_out


def _interpolate_from_neighbors_il(vel_grid, il: int, xl_range: Tuple[int, int]):
    """Fallback: interpolate from neighboring inlines when no velocity at target IL."""
    n_time = len(vel_grid.t_coords)
    xl_start, xl_end = xl_range
    xl_coords_out = np.arange(xl_start, xl_end + 1)
    n_xl = len(xl_coords_out)

    # Find nearest ILs with any velocity data
    valid_data = []  # List of (il, xl, vel_array)

    for i, src_il in enumerate(vel_grid.il_coords):
        for j, src_xl in enumerate(vel_grid.xl_coords):
            vel = vel_grid.velocities[i, j, :]
            if not np.all(np.isnan(vel)):
                valid_data.append((src_il, src_xl, vel))

    if len(valid_data) == 0:
        print(f"[VelInterp] ERROR: No velocity data found anywhere!")
        return None, None

    # Use 2D interpolation
    points = np.array([(d[0], d[1]) for d in valid_data])
    velocities_out = np.zeros((n_xl, n_time), dtype=np.float32)

    for t_idx in range(n_time):
        values = np.array([d[2][t_idx] for d in valid_data])

        # Create target points
        target_points = np.array([(il, xl) for xl in xl_coords_out])

        # Interpolate using griddata (nearest for extrapolation)
        interp_vals = griddata(points, values, target_points, method='linear')

        # Fill NaN with nearest neighbor
        nan_mask = np.isnan(interp_vals)
        if np.any(nan_mask):
            nearest_vals = griddata(points, values, target_points[nan_mask], method='nearest')
            interp_vals[nan_mask] = nearest_vals

        velocities_out[:, t_idx] = interp_vals

    print(f"[VelInterp] IL={il}: Interpolated from {len(valid_data)} 2D locations")
    return velocities_out, xl_coords_out


def _interpolate_from_neighbors_xl(vel_grid, xl: int, il_range: Tuple[int, int]):
    """Fallback: interpolate from neighboring crosslines when no velocity at target XL."""
    n_time = len(vel_grid.t_coords)
    il_start, il_end = il_range
    il_coords_out = np.arange(il_start, il_end + 1)
    n_il = len(il_coords_out)

    # Find all locations with velocity data
    valid_data = []  # List of (il, xl, vel_array)

    for i, src_il in enumerate(vel_grid.il_coords):
        for j, src_xl in enumerate(vel_grid.xl_coords):
            vel = vel_grid.velocities[i, j, :]
            if not np.all(np.isnan(vel)):
                valid_data.append((src_il, src_xl, vel))

    if len(valid_data) == 0:
        print(f"[VelInterp] ERROR: No velocity data found anywhere!")
        return None, None

    # Use 2D interpolation
    points = np.array([(d[0], d[1]) for d in valid_data])
    velocities_out = np.zeros((n_il, n_time), dtype=np.float32)

    for t_idx in range(n_time):
        values = np.array([d[2][t_idx] for d in valid_data])

        # Create target points
        target_points = np.array([(il, xl) for il in il_coords_out])

        # Interpolate using griddata
        interp_vals = griddata(points, values, target_points, method='linear')

        # Fill NaN with nearest neighbor
        nan_mask = np.isnan(interp_vals)
        if np.any(nan_mask):
            nearest_vals = griddata(points, values, target_points[nan_mask], method='nearest')
            interp_vals[nan_mask] = nearest_vals

        velocities_out[:, t_idx] = interp_vals

    print(f"[VelInterp] XL={xl}: Interpolated from {len(valid_data)} 2D locations")
    return velocities_out, il_coords_out


def generate_velocity_qc_image(
    velocities: np.ndarray,
    t_coords: np.ndarray,
    x_coords: np.ndarray,
    x_label: str,
    title: str,
    save_path: Path,
    pick_locations: Optional[List[int]] = None
) -> bool:
    """
    Generate QC image of interpolated velocity field.

    Args:
        velocities: 2D array (n_x, n_time) of velocities
        t_coords: Time coordinates in ms
        x_coords: Spatial coordinates (IL or XL)
        x_label: Label for x-axis ("Inline" or "Crossline")
        title: Plot title
        save_path: Path to save the image
        pick_locations: List of x coordinates where picks exist (for marking)

    Returns:
        True if successful
    """
    try:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot velocity field
        extent = [x_coords[0], x_coords[-1], t_coords[-1], t_coords[0]]
        im = ax.imshow(velocities.T, aspect='auto', extent=extent,
                       cmap='jet', vmin=1500, vmax=5000)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Velocity (m/s)')

        # Mark pick locations if provided
        if pick_locations is not None and len(pick_locations) > 0:
            for loc in pick_locations:
                if x_coords[0] <= loc <= x_coords[-1]:
                    ax.axvline(x=loc, color='white', linestyle='--',
                              linewidth=1, alpha=0.7)
            # Add legend entry
            ax.plot([], [], 'w--', label=f'Velocity picks ({len(pick_locations)})')
            ax.legend(loc='upper right')

        ax.set_xlabel(x_label)
        ax.set_ylabel('Time (ms)')
        ax.set_title(title)

        # Add grid
        ax.grid(True, alpha=0.3, color='white')

        # Save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"[VelInterp] QC image saved: {save_path}")
        return True

    except Exception as e:
        print(f"[VelInterp] Error saving QC image: {e}")
        return False


class InterpolatedVelocityModel:
    """
    Wrapper providing velocity lookups from interpolated 2D velocity field.
    Used during stacking to provide velocity at any location along a line.
    """

    def __init__(self, velocities: np.ndarray, coords: np.ndarray,
                 t_coords: np.ndarray, direction: str):
        """
        Args:
            velocities: 2D array (n_positions, n_time)
            coords: Position coordinates (XL for inline, IL for crossline)
            t_coords: Time coordinates
            direction: 'inline' or 'crossline'
        """
        self.velocities = velocities
        self.coords = coords
        self.t_coords = t_coords
        self.direction = direction

        # Build coord to index mapping
        self._coord_to_idx = {int(c): i for i, c in enumerate(coords)}

    def get_velocity_at(self, il: int, xl: int) -> Optional[np.ndarray]:
        """Get velocity function at given position."""
        if self.direction == 'inline':
            # For inline stacking, look up by XL
            coord = xl
        else:
            # For crossline stacking, look up by IL
            coord = il

        idx = self._coord_to_idx.get(int(coord))
        if idx is None:
            return None

        return self.velocities[idx, :]

    def has_velocity(self) -> bool:
        """Check if velocity data exists."""
        return self.velocities is not None and len(self.velocities) > 0
