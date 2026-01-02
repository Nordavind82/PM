"""Seismic filtering functions (bandpass, AGC)."""

import numpy as np


def apply_bandpass_filter(data: np.ndarray, dt_ms: float,
                          f_low: float = 5.0, f_high: float = 80.0,
                          taper_width: float = 5.0) -> np.ndarray:
    """
    Apply bandpass filter using FFT. Vectorized for 2D arrays.

    Args:
        data: (n_traces, n_time) or (n_time,) array
        dt_ms: Sample interval in milliseconds
        f_low, f_high: Corner frequencies in Hz
        taper_width: Taper width in Hz for smooth rolloff
    """
    if data is None:
        return None

    was_1d = data.ndim == 1
    if was_1d:
        data = data[np.newaxis, :]

    n_traces, n_time = data.shape
    dt_s = dt_ms / 1000.0

    # FFT
    spectrum = np.fft.rfft(data, axis=1)
    freqs = np.fft.rfftfreq(n_time, dt_s)

    # Create bandpass filter with cosine taper
    filt = np.zeros_like(freqs)

    # Passband
    passband = (freqs >= f_low) & (freqs <= f_high)
    filt[passband] = 1.0

    # Low taper
    low_taper = (freqs >= f_low - taper_width) & (freqs < f_low)
    if np.any(low_taper):
        filt[low_taper] = 0.5 * (1 + np.cos(np.pi * (freqs[low_taper] - f_low) / taper_width))

    # High taper
    high_taper = (freqs > f_high) & (freqs <= f_high + taper_width)
    if np.any(high_taper):
        filt[high_taper] = 0.5 * (1 + np.cos(np.pi * (freqs[high_taper] - f_high) / taper_width))

    # Apply filter
    filtered = np.fft.irfft(spectrum * filt, n=n_time, axis=1)

    if was_1d:
        filtered = filtered[0]

    return filtered.astype(np.float32)


def apply_agc(data: np.ndarray, window_ms: float = 500.0, dt_ms: float = 2.0) -> np.ndarray:
    """
    Apply Automatic Gain Control. Fully vectorized implementation.

    Args:
        data: (n_traces, n_time) or (n_time,) array
        window_ms: AGC window length in milliseconds
        dt_ms: Sample interval in milliseconds
    """
    if data is None:
        return None

    was_1d = data.ndim == 1
    if was_1d:
        data = data[np.newaxis, :]

    n_traces, n_time = data.shape
    window_samples = max(1, int(window_ms / dt_ms))
    half_win = window_samples // 2

    # Compute envelope using sliding window RMS with cumsum
    data_sq = data ** 2

    # Pad data for edge handling
    padded = np.pad(data_sq, ((0, 0), (half_win, half_win)), mode='reflect')

    # Use cumsum for fast sliding window (fully vectorized)
    cumsum = np.zeros((n_traces, padded.shape[1] + 1))
    cumsum[:, 1:] = np.cumsum(padded, axis=1)

    # Compute RMS using vectorized slicing
    left_idx = np.arange(n_time)
    right_idx = left_idx + window_samples
    window_sums = cumsum[:, right_idx + 1] - cumsum[:, left_idx]
    rms = np.sqrt(window_sums / window_samples)

    # Apply gain (avoid division by zero)
    rms_max = rms.max()
    if rms_max > 0:
        rms = np.maximum(rms, rms_max * 1e-6)
        agc_data = data / rms
    else:
        agc_data = data.copy()

    if was_1d:
        agc_data = agc_data[0]

    return agc_data.astype(np.float32)
