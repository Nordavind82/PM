"""Velocity mute functions."""

import numpy as np


def apply_velocity_mute(gather: np.ndarray, offsets: np.ndarray, t_coords: np.ndarray,
                        v_top: float = None, v_bottom: float = None) -> np.ndarray:
    """Apply velocity-based mute to gather. Fully vectorized implementation."""
    if gather is None:
        return None

    muted = gather.copy()
    n_offsets, n_time = gather.shape

    # Create time grid (1, n_time)
    time_grid = t_coords[np.newaxis, :]  # (1, n_time) in ms

    # Top mute: zero out samples before t = offset/v_top
    if v_top is not None and v_top > 0:
        t_top = np.abs(offsets[:, np.newaxis]) / v_top * 1000.0  # (n_offsets, 1) in ms
        mute_mask_top = time_grid < t_top
        muted[mute_mask_top] = 0

    # Bottom mute: zero out samples after t = offset/v_bottom
    if v_bottom is not None and v_bottom > 0:
        t_bottom = np.abs(offsets[:, np.newaxis]) / v_bottom * 1000.0  # (n_offsets, 1) in ms
        mute_mask_bottom = time_grid > t_bottom
        muted[mute_mask_bottom] = 0

    return muted
