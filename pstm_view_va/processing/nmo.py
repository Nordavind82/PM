"""NMO correction functions."""

import numpy as np


def apply_nmo_correction(gather: np.ndarray, offsets: np.ndarray, t_coords: np.ndarray,
                         velocity: float, inverse: bool = False) -> np.ndarray:
    """
    Apply NMO correction to gather. Fully vectorized implementation.

    Args:
        gather: (n_offsets, n_time) array
        offsets: Offset values in meters
        t_coords: Time values in ms
        velocity: NMO velocity in m/s
        inverse: If True, apply inverse NMO (stretch flat to hyperbola)
                 If False, apply forward NMO (flatten hyperbola)

    Returns:
        NMO corrected gather
    """
    if gather is None or velocity <= 0:
        return gather

    n_offsets, n_time = gather.shape
    dt = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

    # Output time grid (1, n_time) - where we place values
    time_grid = t_coords[np.newaxis, :]
    off_grid = offsets[:, np.newaxis]   # (n_offsets, 1)

    # Compute travel time correction term: (x/v)^2 in ms^2
    stretch_term_sq = (off_grid / velocity * 1000.0) ** 2

    if inverse:
        # Inverse NMO: stretch flat events back to hyperbola
        # At output time t, sample from input at t0 = sqrt(t^2 - (x/v)^2)
        t_sq = time_grid ** 2
        src_time = np.sqrt(np.maximum(0, t_sq - stretch_term_sq))
    else:
        # Forward NMO: flatten hyperbolic events
        # At output time t0, sample from input at t = sqrt(t0^2 + (x/v)^2)
        src_time = np.sqrt(time_grid**2 + stretch_term_sq)

    # Convert to indices and clip to valid range
    idx_src = np.clip((src_time / dt).astype(np.int32), 0, n_time - 1)

    # Fully vectorized gather using advanced indexing
    row_idx = np.arange(n_offsets)[:, np.newaxis]
    corrected = gather[row_idx, idx_src]

    return corrected.astype(np.float32)


def apply_nmo_with_velocity_model(gather: np.ndarray, offsets: np.ndarray,
                                   t_coords: np.ndarray, velocity_func: np.ndarray,
                                   inverse: bool = False,
                                   stretch_mute_percent: float = None,
                                   vel_t_coords: np.ndarray = None) -> np.ndarray:
    """
    Apply NMO correction using time-varying velocity function.

    Args:
        gather: (n_offsets, n_time) array
        offsets: Offset values in meters
        t_coords: Time values in ms
        velocity_func: 1D array of velocities at each time sample (m/s)
        inverse: If True, apply inverse NMO
        stretch_mute_percent: If set, mute samples where NMO stretch exceeds
                              this percentage (only applies to forward NMO)
        vel_t_coords: Time coordinates for velocity_func. If None, assumes
                      velocity_func is sampled at t_coords.

    Returns:
        NMO corrected gather
    """
    if gather is None or velocity_func is None:
        return gather

    n_offsets, n_time = gather.shape
    dt = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

    # Ensure velocity array matches time samples
    if len(velocity_func) != n_time:
        # Interpolate velocity to match time samples
        if vel_t_coords is not None:
            # Use provided velocity time coordinates
            velocity_func = np.interp(t_coords, vel_t_coords, velocity_func)
        else:
            # Fallback: assume velocity spans from 0 to t_coords[-1]
            vel_t = np.linspace(0, t_coords[-1], len(velocity_func))
            velocity_func = np.interp(t_coords, vel_t, velocity_func)

    # Output time grid and offset grid
    time_grid = t_coords[np.newaxis, :]  # (1, n_time)
    off_grid = offsets[:, np.newaxis]    # (n_offsets, 1)
    vel_grid = velocity_func[np.newaxis, :]  # (1, n_time)

    if inverse:
        # Inverse NMO: stretch flat events to hyperbola
        # For time-varying velocity, we must use V(t0) - velocity at INPUT (zero-offset) time
        # NMO equation: t = sqrt(t0^2 + x^2/V(t0)^2)
        # For each input time t0, compute output time t and scatter the sample there
        #
        # Implementation: for each output time t, we need to find which input t0
        # would map to it. This requires using V(t0), not V(t).
        # Since t = sqrt(t0^2 + x^2/V(t0)^2), we compute output times for all input times
        # and then interpolate to get the correct mapping.

        # Compute output times for all input times: t = sqrt(t0^2 + (x/V(t0))^2)
        stretch_term_sq = (off_grid / vel_grid * 1000.0) ** 2  # V(t0) at each t0
        t0_sq = time_grid ** 2
        t_out = np.sqrt(t0_sq + stretch_term_sq)  # (n_offsets, n_time) output times

        # For each output time, find where to sample from by inverting the mapping
        # We need to resample: for each output time t, find the input time t0
        # This is done by interpolating: input times are t_coords, output times are t_out
        corrected = np.zeros_like(gather)
        for i in range(n_offsets):
            # t_out[i, :] gives output times for input times t_coords
            # We want to find: for each output time in t_coords, what input time?
            # Invert by interpolating: x=t_out[i,:], y=t_coords, xnew=t_coords
            # This finds input time for each output time
            src_times = np.interp(t_coords, t_out[i, :], t_coords)
            idx_src = np.clip((src_times / dt).astype(np.int32), 0, n_time - 1)
            corrected[i, :] = gather[i, idx_src]
    else:
        # Forward NMO: flatten hyperbolic events
        # At output time t0, sample from t = sqrt(t0^2 + (x/v(t0))^2)
        # Here V(t0) is correct - we use velocity at the output (zero-offset) time
        stretch_term_sq = (off_grid / vel_grid * 1000.0) ** 2
        src_time = np.sqrt(time_grid**2 + stretch_term_sq)

        # Convert to indices
        idx_src = np.clip((src_time / dt).astype(np.int32), 0, n_time - 1)

        # Vectorized gather
        row_idx = np.arange(n_offsets)[:, np.newaxis]
        corrected = gather[row_idx, idx_src]

        # Apply stretch mute for forward NMO
        if stretch_mute_percent is not None and stretch_mute_percent > 0:
            # Stretch factor = (t_nmo - t0) / t0 = (src_time - time_grid) / time_grid
            # Avoid division by zero at t=0
            with np.errstate(divide='ignore', invalid='ignore'):
                stretch = np.abs(src_time - time_grid) / np.maximum(time_grid, dt)
                stretch_threshold = stretch_mute_percent / 100.0
                mute_mask = stretch > stretch_threshold
                corrected = np.where(mute_mask, 0.0, corrected)

    return corrected.astype(np.float32)
