"""Semblance computation for velocity analysis."""

from typing import Tuple
import numpy as np


def compute_semblance_fast(gather: np.ndarray, offsets: np.ndarray, t_coords: np.ndarray,
                           v_min: float = 1500, v_max: float = 5000, v_step: float = 50,
                           window_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute semblance for velocity analysis. Fast vectorized implementation.

    Args:
        gather: (n_offsets, n_time) array of seismic traces
        offsets: Offset values in meters
        t_coords: Time values in ms
        v_min: Minimum velocity to scan (m/s)
        v_max: Maximum velocity to scan (m/s)
        v_step: Velocity step size (m/s)
        window_samples: Number of samples for semblance window

    Returns:
        Tuple of (semblance, velocities) arrays
    """
    if gather is None:
        return None, None

    n_offsets, n_time = gather.shape
    dt = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

    velocities = np.arange(v_min, v_max + v_step, v_step, dtype=np.float32)
    n_vel = len(velocities)
    semblance = np.zeros((n_vel, n_time), dtype=np.float32)
    half_win = window_samples // 2

    # Pre-compute offset grid
    off_sq = (offsets[:, np.newaxis] / 1000.0) ** 2  # (n_offsets, 1), in km^2
    t0_sq = (t_coords[np.newaxis, :] / 1000.0) ** 2  # (1, n_time), in s^2

    for iv, vel in enumerate(velocities):
        # Vectorized NMO: t = sqrt(t0^2 + x^2/v^2)
        v_sq = (vel / 1000.0) ** 2  # km^2/s^2
        t_nmo = np.sqrt(t0_sq + off_sq / v_sq) * 1000.0  # Back to ms

        # Convert to indices
        idx_nmo = np.clip((t_nmo / dt).astype(np.int32), 0, n_time - 1)

        # Gather NMO-corrected traces using advanced indexing
        row_idx = np.arange(n_offsets)[:, np.newaxis]
        nmo_gather = gather[row_idx, idx_nmo]

        # Compute semblance using vectorized operations
        # Use cumsum for efficient window sums
        cumsum = np.zeros((n_offsets, n_time + 1))
        cumsum[:, 1:] = np.cumsum(nmo_gather, axis=1)
        cumsum_sq = np.zeros((n_offsets, n_time + 1))
        cumsum_sq[:, 1:] = np.cumsum(nmo_gather ** 2, axis=1)

        for it in range(half_win, n_time - half_win):
            left = it - half_win
            right = it + half_win + 1

            # Sum across traces for each time in window
            trace_sums = cumsum[:, right] - cumsum[:, left]  # (n_offsets,)
            stack_sum = trace_sums.sum()  # Stack amplitude

            trace_sq_sums = cumsum_sq[:, right] - cumsum_sq[:, left]
            energy = trace_sq_sums.sum()

            window_size = right - left
            numerator = stack_sum ** 2
            denominator = n_offsets * window_size * energy

            if denominator > 1e-10:
                semblance[iv, it] = numerator / denominator

    return semblance, velocities
