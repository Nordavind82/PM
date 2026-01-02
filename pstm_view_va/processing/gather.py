"""Super gather creation functions."""

from typing import List, Tuple
import numpy as np
import zarr


def create_super_gather(offset_bins: List[zarr.Array], il_center: int, xl_center: int,
                        il_half: int, xl_half: int, offset_values: np.ndarray,
                        offset_bin_size: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create super gather by stacking traces in spatial window and rebinning offsets.
    Vectorized implementation for speed.
    """
    if len(offset_bins) == 0:
        return None, None

    n_il, n_xl, n_time = offset_bins[0].shape

    # Clamp window to data bounds
    il_min = max(0, il_center - il_half)
    il_max = min(n_il - 1, il_center + il_half) + 1
    xl_min = max(0, xl_center - xl_half)
    xl_max = min(n_xl - 1, xl_center + xl_half) + 1

    # Determine new offset bins
    off_min = offset_values.min()
    off_max = offset_values.max()
    n_new_bins = max(1, int(np.ceil((off_max - off_min) / offset_bin_size)))
    new_offsets = np.arange(n_new_bins) * offset_bin_size + off_min + offset_bin_size / 2

    # Pre-compute bin assignments for all original offsets
    bin_indices = np.clip(((offset_values - off_min) / offset_bin_size).astype(int), 0, n_new_bins - 1)

    # Initialize super gather
    super_gather = np.zeros((n_new_bins, n_time), dtype=np.float64)
    fold = np.zeros(n_new_bins, dtype=np.int32)

    # Stack traces - read entire spatial window at once per offset bin
    for i, offset_zarr in enumerate(offset_bins):
        bin_idx = bin_indices[i]
        # Read entire spatial window at once (vectorized)
        window_data = np.asarray(offset_zarr[il_min:il_max, xl_min:xl_max, :])
        # Sum all traces in window
        super_gather[bin_idx, :] += window_data.sum(axis=(0, 1))
        fold[bin_idx] += window_data.shape[0] * window_data.shape[1]

    # Normalize by fold (vectorized)
    valid = fold > 0
    super_gather[valid, :] /= fold[valid, np.newaxis]

    return super_gather.astype(np.float32), new_offsets
