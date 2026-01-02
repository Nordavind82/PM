"""Live stack preview for real-time velocity analysis feedback."""

from typing import List, Tuple, Optional, Callable
import numpy as np
import zarr

from .nmo import apply_nmo_with_velocity_model
from .mute import apply_velocity_mute
from .filters import apply_bandpass_filter, apply_agc


class GatherCache:
    """
    Cache for pre-processed gathers ready for fast NMO application.

    Stores gathers after:
    - Inverse NMO with initial velocity (removes existing NMO)
    - Bandpass filter
    - AGC
    - Velocity mutes

    This allows instant forward NMO with any velocity during preview.
    """

    def __init__(self):
        self.cached_gathers: Optional[np.ndarray] = None  # (n_traces, n_offsets, n_time)
        self.cached_il: Optional[int] = None
        self.cached_xl: Optional[int] = None
        self.cached_direction: Optional[str] = None  # 'inline' or 'crossline'
        self.trace_coords: Optional[np.ndarray] = None  # XL coords for inline, IL coords for crossline
        self.offset_values: Optional[np.ndarray] = None
        self.t_coords: Optional[np.ndarray] = None
        self.initial_vel_t_coords: Optional[np.ndarray] = None

    def cache_inline(self,
                     offset_bins: List[zarr.Array],
                     offset_values: np.ndarray,
                     t_coords: np.ndarray,
                     il: int,
                     initial_velocity_grid,
                     settings: dict) -> bool:
        """
        Cache all gathers for an inline with pre-processing applied.

        Args:
            offset_bins: List of zarr arrays for each offset bin
            offset_values: Array of offset values
            t_coords: Time coordinates
            il: Inline number
            initial_velocity_grid: VelocityOutputGrid with initial velocities
            settings: Processing settings (mutes, filters, etc.)

        Returns:
            True if successful
        """
        if len(offset_bins) == 0:
            return False

        n_il_data, n_xl_data, n_time = offset_bins[0].shape
        n_offsets = len(offset_bins)
        dt_ms = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

        if il < 0 or il >= n_il_data:
            return False

        print(f"[LiveStack] Caching inline {il}: {n_xl_data} traces x {n_offsets} offsets")

        # Get processing parameters
        top_mute_enabled = settings.get('top_mute_enabled', False)
        v_top = settings.get('v_top', 1500)
        bottom_mute_enabled = settings.get('bottom_mute_enabled', False)
        v_bottom = settings.get('v_bottom', 5000)
        apply_bandpass = settings.get('apply_bandpass', False)
        f_low = settings.get('f_low', 5)
        f_high = settings.get('f_high', 80)
        apply_agc_flag = settings.get('apply_agc', False)
        agc_window = settings.get('agc_window', 250)

        # Read all data for this inline
        inline_data = np.zeros((n_offsets, n_xl_data, n_time), dtype=np.float32)
        for k, offset_zarr in enumerate(offset_bins):
            inline_data[k, :, :] = offset_zarr[il, :, :]

        # Pre-process each gather
        cached = np.zeros((n_xl_data, n_offsets, n_time), dtype=np.float32)

        # Get interpolated initial velocities if available
        from .velocity_interpolation import interpolate_velocity_along_inline, InterpolatedVelocityModel

        interp_initial_model = None
        if initial_velocity_grid is not None and initial_velocity_grid.has_velocity():
            interp_vels, xl_coords = interpolate_velocity_along_inline(
                initial_velocity_grid, il, (0, n_xl_data - 1), method='linear'
            )
            if interp_vels is not None:
                interp_initial_model = InterpolatedVelocityModel(
                    interp_vels, xl_coords, initial_velocity_grid.t_coords, 'inline'
                )

        for xl in range(n_xl_data):
            gather = inline_data[:, xl, :].copy()  # (n_offsets, n_time)

            # 1. Inverse NMO with initial velocity
            if interp_initial_model is not None:
                initial_vel = interp_initial_model.get_velocity_at(il, xl)
                if initial_vel is not None:
                    gather = apply_nmo_with_velocity_model(
                        gather, offset_values, t_coords, initial_vel,
                        inverse=True, stretch_mute_percent=100,
                        vel_t_coords=initial_velocity_grid.t_coords
                    )

            # 2. Apply processing
            if apply_bandpass:
                gather = apply_bandpass_filter(gather, dt_ms, f_low, f_high)

            if apply_agc_flag:
                gather = apply_agc(gather, agc_window, dt_ms)

            if top_mute_enabled or bottom_mute_enabled:
                v_t = v_top if top_mute_enabled else None
                v_b = v_bottom if bottom_mute_enabled else None
                gather = apply_velocity_mute(gather, offset_values, t_coords, v_t, v_b)

            cached[xl, :, :] = gather

        # Store cache
        self.cached_gathers = cached
        self.cached_il = il
        self.cached_xl = None
        self.cached_direction = 'inline'
        self.trace_coords = np.arange(n_xl_data)
        self.offset_values = offset_values
        self.t_coords = t_coords
        if initial_velocity_grid is not None:
            self.initial_vel_t_coords = initial_velocity_grid.t_coords

        print(f"[LiveStack] Cached {n_xl_data} gathers for inline {il}")
        return True

    def cache_crossline(self,
                        offset_bins: List[zarr.Array],
                        offset_values: np.ndarray,
                        t_coords: np.ndarray,
                        xl: int,
                        initial_velocity_grid,
                        settings: dict) -> bool:
        """
        Cache all gathers for a crossline with pre-processing applied.
        """
        if len(offset_bins) == 0:
            return False

        n_il_data, n_xl_data, n_time = offset_bins[0].shape
        n_offsets = len(offset_bins)
        dt_ms = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

        if xl < 0 or xl >= n_xl_data:
            return False

        print(f"[LiveStack] Caching crossline {xl}: {n_il_data} traces x {n_offsets} offsets")

        # Get processing parameters
        top_mute_enabled = settings.get('top_mute_enabled', False)
        v_top = settings.get('v_top', 1500)
        bottom_mute_enabled = settings.get('bottom_mute_enabled', False)
        v_bottom = settings.get('v_bottom', 5000)
        apply_bandpass = settings.get('apply_bandpass', False)
        f_low = settings.get('f_low', 5)
        f_high = settings.get('f_high', 80)
        apply_agc_flag = settings.get('apply_agc', False)
        agc_window = settings.get('agc_window', 250)

        # Read all data for this crossline
        xline_data = np.zeros((n_offsets, n_il_data, n_time), dtype=np.float32)
        for k, offset_zarr in enumerate(offset_bins):
            xline_data[k, :, :] = offset_zarr[:, xl, :]

        # Pre-process each gather
        cached = np.zeros((n_il_data, n_offsets, n_time), dtype=np.float32)

        # Get interpolated initial velocities if available
        from .velocity_interpolation import interpolate_velocity_along_crossline, InterpolatedVelocityModel

        interp_initial_model = None
        if initial_velocity_grid is not None and initial_velocity_grid.has_velocity():
            interp_vels, il_coords = interpolate_velocity_along_crossline(
                initial_velocity_grid, xl, (0, n_il_data - 1), method='linear'
            )
            if interp_vels is not None:
                interp_initial_model = InterpolatedVelocityModel(
                    interp_vels, il_coords, initial_velocity_grid.t_coords, 'crossline'
                )

        for il in range(n_il_data):
            gather = xline_data[:, il, :].copy()  # (n_offsets, n_time)

            # 1. Inverse NMO with initial velocity
            if interp_initial_model is not None:
                initial_vel = interp_initial_model.get_velocity_at(il, xl)
                if initial_vel is not None:
                    gather = apply_nmo_with_velocity_model(
                        gather, offset_values, t_coords, initial_vel,
                        inverse=True, stretch_mute_percent=100,
                        vel_t_coords=initial_velocity_grid.t_coords
                    )

            # 2. Apply processing
            if apply_bandpass:
                gather = apply_bandpass_filter(gather, dt_ms, f_low, f_high)

            if apply_agc_flag:
                gather = apply_agc(gather, agc_window, dt_ms)

            if top_mute_enabled or bottom_mute_enabled:
                v_t = v_top if top_mute_enabled else None
                v_b = v_bottom if bottom_mute_enabled else None
                gather = apply_velocity_mute(gather, offset_values, t_coords, v_t, v_b)

            cached[il, :, :] = gather

        # Store cache
        self.cached_gathers = cached
        self.cached_il = None
        self.cached_xl = xl
        self.cached_direction = 'crossline'
        self.trace_coords = np.arange(n_il_data)
        self.offset_values = offset_values
        self.t_coords = t_coords
        if initial_velocity_grid is not None:
            self.initial_vel_t_coords = initial_velocity_grid.t_coords

        print(f"[LiveStack] Cached {n_il_data} gathers for crossline {xl}")
        return True

    def is_valid_for(self, il: int, xl: int, direction: str) -> bool:
        """Check if cache is valid for given position and direction."""
        if self.cached_gathers is None:
            return False
        if direction == 'inline':
            return self.cached_direction == 'inline' and self.cached_il == il
        else:
            return self.cached_direction == 'crossline' and self.cached_xl == xl

    def clear(self):
        """Clear the cache."""
        self.cached_gathers = None
        self.cached_il = None
        self.cached_xl = None
        self.cached_direction = None


class LiveStackUpdater:
    """
    Computes stack in real-time using cached pre-processed gathers.

    Only needs to apply forward NMO (fast) since gathers are pre-processed.
    """

    def __init__(self, gather_cache: GatherCache):
        self.cache = gather_cache

    def compute_stack(self,
                      velocity_func: np.ndarray,
                      vel_t_coords: np.ndarray,
                      stretch_percent: float = 30) -> Optional[np.ndarray]:
        """
        Compute stack using cached gathers and given velocity function.

        Args:
            velocity_func: Velocity function (1D array)
            vel_t_coords: Time coordinates for velocity function
            stretch_percent: Stretch mute percentage

        Returns:
            Stack array (n_traces, n_time) or None if cache invalid
        """
        if self.cache.cached_gathers is None:
            return None

        n_traces = self.cache.cached_gathers.shape[0]
        n_time = self.cache.cached_gathers.shape[2]

        stack = np.zeros((n_traces, n_time), dtype=np.float32)

        for i in range(n_traces):
            gather = self.cache.cached_gathers[i, :, :]  # (n_offsets, n_time)

            # Apply forward NMO with preview velocity
            gather_nmo = apply_nmo_with_velocity_model(
                gather, self.cache.offset_values, self.cache.t_coords,
                velocity_func, inverse=False, stretch_mute_percent=stretch_percent,
                vel_t_coords=vel_t_coords
            )

            # Stack
            stack[i, :] = np.nanmean(gather_nmo, axis=0)

        return stack

    def compute_stack_with_temp_pick(self,
                                     current_picks: List[Tuple[float, float]],
                                     temp_time: float,
                                     temp_velocity: float,
                                     vel_t_coords: np.ndarray,
                                     stretch_percent: float = 30) -> Optional[np.ndarray]:
        """
        Compute stack with a temporary pick added to current picks.

        Args:
            current_picks: List of (time_ms, velocity) tuples
            temp_time: Time of temporary pick
            temp_velocity: Velocity of temporary pick
            vel_t_coords: Time coordinates for output velocity function
            stretch_percent: Stretch mute percentage

        Returns:
            Stack array or None
        """
        # Create velocity function from picks + temp pick
        all_picks = list(current_picks) + [(temp_time, temp_velocity)]
        all_picks.sort(key=lambda p: p[0])

        if len(all_picks) < 2:
            # Need at least 2 points for interpolation
            # Use constant velocity
            velocity_func = np.full(len(vel_t_coords), temp_velocity, dtype=np.float32)
        else:
            # Interpolate picks to velocity function
            pick_times = np.array([p[0] for p in all_picks])
            pick_vels = np.array([p[1] for p in all_picks])
            velocity_func = np.interp(vel_t_coords, pick_times, pick_vels).astype(np.float32)

        return self.compute_stack(velocity_func, vel_t_coords, stretch_percent)

    def compute_stack_with_spatial_velocities(self,
                                               velocity_grid,
                                               current_il: int,
                                               current_xl: int,
                                               temp_velocity_func: np.ndarray,
                                               vel_t_coords: np.ndarray,
                                               stretch_percent: float = 30) -> Optional[np.ndarray]:
        """
        Compute stack with spatially interpolated velocities from velocity grid.

        For each trace in the cached line, interpolates velocity from:
        - The temporary velocity at current location (being edited)
        - Velocities at other locations from the velocity grid

        Args:
            velocity_grid: VelocityOutputGrid with velocities at multiple locations
            current_il: Current inline being edited
            current_xl: Current crossline being edited
            temp_velocity_func: Temporary velocity function at current location (1D array)
            vel_t_coords: Time coordinates for velocity function
            stretch_percent: Stretch mute percentage

        Returns:
            Stack array (n_traces, n_time) or None if cache invalid
        """
        if self.cache.cached_gathers is None:
            return None

        from .velocity_interpolation import InterpolatedVelocityModel
        from scipy.interpolate import interp1d

        n_traces = self.cache.cached_gathers.shape[0]
        n_time = self.cache.cached_gathers.shape[2]

        stack = np.zeros((n_traces, n_time), dtype=np.float32)

        direction = self.cache.cached_direction
        trace_coords = self.cache.trace_coords

        # Build velocity functions at each velocity grid location along this line
        # and override the current location with temp_velocity_func
        if direction == 'inline':
            # For inline, trace_coords are XL indices
            il = self.cache.cached_il
            xl_coords_grid = velocity_grid.xl_coords if velocity_grid.has_velocity() else np.array([])

            # Get velocities at grid locations
            vel_at_locations = []  # list of (xl, velocity_func)

            for xl_grid in xl_coords_grid:
                if abs(xl_grid - current_xl) < 1:
                    # This is the current location - use temp velocity
                    vel_at_locations.append((xl_grid, temp_velocity_func))
                else:
                    vel_func = velocity_grid.get_velocity_at(il, xl_grid)
                    if vel_func is not None:
                        # Resample to vel_t_coords if needed
                        if len(vel_func) != len(vel_t_coords):
                            f = interp1d(velocity_grid.t_coords, vel_func,
                                        kind='linear', bounds_error=False, fill_value='extrapolate')
                            vel_func = f(vel_t_coords).astype(np.float32)
                        vel_at_locations.append((xl_grid, vel_func))

            # Always add current location if not in grid
            current_in_grid = any(abs(xl - current_xl) < 1 for xl, _ in vel_at_locations)
            if not current_in_grid:
                vel_at_locations.append((current_xl, temp_velocity_func))

            vel_at_locations.sort(key=lambda x: x[0])

            if len(vel_at_locations) == 0:
                # No velocities - use temp for all
                for i in range(n_traces):
                    gather = self.cache.cached_gathers[i, :, :]
                    gather_nmo = apply_nmo_with_velocity_model(
                        gather, self.cache.offset_values, self.cache.t_coords,
                        temp_velocity_func, inverse=False, stretch_mute_percent=stretch_percent,
                        vel_t_coords=vel_t_coords
                    )
                    stack[i, :] = np.nanmean(gather_nmo, axis=0)
            elif len(vel_at_locations) == 1:
                # Single location - use it for all
                _, vel_func = vel_at_locations[0]
                for i in range(n_traces):
                    gather = self.cache.cached_gathers[i, :, :]
                    gather_nmo = apply_nmo_with_velocity_model(
                        gather, self.cache.offset_values, self.cache.t_coords,
                        vel_func, inverse=False, stretch_mute_percent=stretch_percent,
                        vel_t_coords=vel_t_coords
                    )
                    stack[i, :] = np.nanmean(gather_nmo, axis=0)
            else:
                # Multiple locations - interpolate for each trace
                xl_grid_locs = np.array([loc[0] for loc in vel_at_locations])
                vel_funcs = np.array([loc[1] for loc in vel_at_locations])  # (n_locs, n_time)

                for i in range(n_traces):
                    xl = trace_coords[i]

                    # Interpolate velocity function at this XL
                    # For each time sample, interpolate velocity across XL
                    vel_interp = np.zeros(len(vel_t_coords), dtype=np.float32)
                    for t_idx in range(len(vel_t_coords)):
                        vels_at_t = vel_funcs[:, t_idx]
                        vel_interp[t_idx] = np.interp(xl, xl_grid_locs, vels_at_t)

                    gather = self.cache.cached_gathers[i, :, :]
                    gather_nmo = apply_nmo_with_velocity_model(
                        gather, self.cache.offset_values, self.cache.t_coords,
                        vel_interp, inverse=False, stretch_mute_percent=stretch_percent,
                        vel_t_coords=vel_t_coords
                    )
                    stack[i, :] = np.nanmean(gather_nmo, axis=0)

        else:  # crossline
            # For crossline, trace_coords are IL indices
            xl = self.cache.cached_xl
            il_coords_grid = velocity_grid.il_coords if velocity_grid.has_velocity() else np.array([])

            # Get velocities at grid locations
            vel_at_locations = []  # list of (il, velocity_func)

            for il_grid in il_coords_grid:
                if abs(il_grid - current_il) < 1:
                    # This is the current location - use temp velocity
                    vel_at_locations.append((il_grid, temp_velocity_func))
                else:
                    vel_func = velocity_grid.get_velocity_at(il_grid, xl)
                    if vel_func is not None:
                        # Resample to vel_t_coords if needed
                        if len(vel_func) != len(vel_t_coords):
                            f = interp1d(velocity_grid.t_coords, vel_func,
                                        kind='linear', bounds_error=False, fill_value='extrapolate')
                            vel_func = f(vel_t_coords).astype(np.float32)
                        vel_at_locations.append((il_grid, vel_func))

            # Always add current location if not in grid
            current_in_grid = any(abs(il - current_il) < 1 for il, _ in vel_at_locations)
            if not current_in_grid:
                vel_at_locations.append((current_il, temp_velocity_func))

            vel_at_locations.sort(key=lambda x: x[0])

            if len(vel_at_locations) == 0:
                # No velocities - use temp for all
                for i in range(n_traces):
                    gather = self.cache.cached_gathers[i, :, :]
                    gather_nmo = apply_nmo_with_velocity_model(
                        gather, self.cache.offset_values, self.cache.t_coords,
                        temp_velocity_func, inverse=False, stretch_mute_percent=stretch_percent,
                        vel_t_coords=vel_t_coords
                    )
                    stack[i, :] = np.nanmean(gather_nmo, axis=0)
            elif len(vel_at_locations) == 1:
                # Single location - use it for all
                _, vel_func = vel_at_locations[0]
                for i in range(n_traces):
                    gather = self.cache.cached_gathers[i, :, :]
                    gather_nmo = apply_nmo_with_velocity_model(
                        gather, self.cache.offset_values, self.cache.t_coords,
                        vel_func, inverse=False, stretch_mute_percent=stretch_percent,
                        vel_t_coords=vel_t_coords
                    )
                    stack[i, :] = np.nanmean(gather_nmo, axis=0)
            else:
                # Multiple locations - interpolate for each trace
                il_grid_locs = np.array([loc[0] for loc in vel_at_locations])
                vel_funcs = np.array([loc[1] for loc in vel_at_locations])  # (n_locs, n_time)

                for i in range(n_traces):
                    il = trace_coords[i]

                    # Interpolate velocity function at this IL
                    vel_interp = np.zeros(len(vel_t_coords), dtype=np.float32)
                    for t_idx in range(len(vel_t_coords)):
                        vels_at_t = vel_funcs[:, t_idx]
                        vel_interp[t_idx] = np.interp(il, il_grid_locs, vels_at_t)

                    gather = self.cache.cached_gathers[i, :, :]
                    gather_nmo = apply_nmo_with_velocity_model(
                        gather, self.cache.offset_values, self.cache.t_coords,
                        vel_interp, inverse=False, stretch_mute_percent=stretch_percent,
                        vel_t_coords=vel_t_coords
                    )
                    stack[i, :] = np.nanmean(gather_nmo, axis=0)

        return stack

    def compute_stack_with_modified_pick(self,
                                          velocity_grid,
                                          current_il: int,
                                          current_xl: int,
                                          current_picks: List[Tuple[float, float]],
                                          pick_index: int,
                                          new_time: float,
                                          new_velocity: float,
                                          vel_t_coords: np.ndarray,
                                          stretch_percent: float = 30) -> Optional[np.ndarray]:
        """
        Compute stack with a specific pick modified at current location.

        Creates temporary velocity function with the modified pick, then
        computes stack with spatial interpolation.

        Args:
            velocity_grid: VelocityOutputGrid with velocities at multiple locations
            current_il: Current inline being edited
            current_xl: Current crossline being edited
            current_picks: List of (time_ms, velocity) tuples at current location
            pick_index: Index of pick being modified
            new_time: New time for the pick
            new_velocity: New velocity for the pick
            vel_t_coords: Time coordinates for velocity function
            stretch_percent: Stretch mute percentage

        Returns:
            Stack array or None
        """
        # Create modified picks list
        modified_picks = list(current_picks)
        if 0 <= pick_index < len(modified_picks):
            modified_picks[pick_index] = (new_time, new_velocity)
        modified_picks.sort(key=lambda p: p[0])

        # Create velocity function from modified picks
        if len(modified_picks) < 2:
            if len(modified_picks) == 1:
                temp_velocity_func = np.full(len(vel_t_coords), modified_picks[0][1], dtype=np.float32)
            else:
                return None
        else:
            pick_times = np.array([p[0] for p in modified_picks])
            pick_vels = np.array([p[1] for p in modified_picks])
            temp_velocity_func = np.interp(vel_t_coords, pick_times, pick_vels).astype(np.float32)

        return self.compute_stack_with_spatial_velocities(
            velocity_grid, current_il, current_xl,
            temp_velocity_func, vel_t_coords, stretch_percent
        )
