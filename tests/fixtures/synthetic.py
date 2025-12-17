"""
Synthetic test data generator for PSTM.

Creates synthetic seismic data for testing and validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pstm.data.parquet_headers import create_parquet_headers
from pstm.data.zarr_reader import create_zarr_traces
from pstm.utils.logging import get_logger

logger = get_logger(__name__)


def generate_ricker_wavelet(
    freq_hz: float,
    sample_rate_ms: float,
    length_ms: float = 200.0,
) -> NDArray[np.float32]:
    """
    Generate a Ricker (Mexican hat) wavelet.

    Args:
        freq_hz: Dominant frequency in Hz
        sample_rate_ms: Sample rate in milliseconds
        length_ms: Wavelet length in milliseconds

    Returns:
        Wavelet amplitude array
    """
    n_samples = int(length_ms / sample_rate_ms)
    t = np.linspace(-length_ms / 2, length_ms / 2, n_samples) / 1000  # Convert to seconds

    # Ricker wavelet formula
    pi_f_t = np.pi * freq_hz * t
    wavelet = (1 - 2 * pi_f_t**2) * np.exp(-(pi_f_t**2))

    return wavelet.astype(np.float32)


def generate_synthetic_traces(
    n_traces: int,
    n_samples: int,
    sample_rate_ms: float,
    reflector_times_ms: list[float],
    reflector_amplitudes: list[float],
    wavelet_freq_hz: float = 30.0,
    noise_level: float = 0.1,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """
    Generate synthetic seismic traces with flat reflectors.

    Args:
        n_traces: Number of traces
        n_samples: Samples per trace
        sample_rate_ms: Sample rate in milliseconds
        reflector_times_ms: Times of flat reflectors (ms)
        reflector_amplitudes: Amplitudes of reflectors
        wavelet_freq_hz: Wavelet frequency
        noise_level: Random noise level (fraction of signal)
        seed: Random seed for reproducibility

    Returns:
        Trace data array (n_traces, n_samples)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate wavelet
    wavelet = generate_ricker_wavelet(wavelet_freq_hz, sample_rate_ms)
    wavelet_half = len(wavelet) // 2

    # Initialize traces
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)

    # Add reflectors
    for t_ms, amp in zip(reflector_times_ms, reflector_amplitudes):
        sample_idx = int(t_ms / sample_rate_ms)

        for i in range(n_traces):
            # Add wavelet at reflector time
            start = max(0, sample_idx - wavelet_half)
            end = min(n_samples, sample_idx + wavelet_half)
            wav_start = max(0, wavelet_half - sample_idx)
            wav_end = wav_start + (end - start)

            if end > start:
                traces[i, start:end] += amp * wavelet[wav_start:wav_end]

    # Add noise
    if noise_level > 0:
        noise = np.random.randn(n_traces, n_samples).astype(np.float32)
        traces += noise_level * noise * np.abs(traces).max()

    return traces


def generate_synthetic_geometry_2d(
    n_shots: int,
    n_receivers_per_shot: int,
    shot_spacing: float,
    receiver_spacing: float,
    first_shot_x: float = 0.0,
    first_receiver_offset: float = -500.0,
    line_y: float = 0.0,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
]:
    """
    Generate 2D synthetic geometry (single line).

    Args:
        n_shots: Number of shot points
        n_receivers_per_shot: Receivers per shot
        shot_spacing: Shot interval (meters)
        receiver_spacing: Receiver interval (meters)
        first_shot_x: X coordinate of first shot
        first_receiver_offset: Offset of first receiver from shot
        line_y: Y coordinate of the line

    Returns:
        Tuple of (trace_indices, source_x, source_y, receiver_x, receiver_y, shot_ids)
    """
    n_traces = n_shots * n_receivers_per_shot

    trace_indices = np.arange(n_traces, dtype=np.int64)
    source_x = np.zeros(n_traces, dtype=np.float64)
    source_y = np.full(n_traces, line_y, dtype=np.float64)
    receiver_x = np.zeros(n_traces, dtype=np.float64)
    receiver_y = np.full(n_traces, line_y, dtype=np.float64)
    shot_ids = np.zeros(n_traces, dtype=np.int32)

    idx = 0
    for ishot in range(n_shots):
        shot_x = first_shot_x + ishot * shot_spacing

        for irec in range(n_receivers_per_shot):
            rec_x = shot_x + first_receiver_offset + irec * receiver_spacing

            source_x[idx] = shot_x
            receiver_x[idx] = rec_x
            shot_ids[idx] = ishot
            idx += 1

    return trace_indices, source_x, source_y, receiver_x, receiver_y, shot_ids


def generate_synthetic_geometry_3d(
    n_shots_x: int,
    n_shots_y: int,
    n_receivers_x: int,
    n_receivers_y: int,
    shot_spacing_x: float,
    shot_spacing_y: float,
    receiver_spacing_x: float,
    receiver_spacing_y: float,
    shot_origin: tuple[float, float] = (0.0, 0.0),
    receiver_origin: tuple[float, float] = (-500.0, -500.0),
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
]:
    """
    Generate 3D synthetic geometry (grid of shots, fixed receiver patch).

    Args:
        n_shots_x, n_shots_y: Number of shots in X and Y
        n_receivers_x, n_receivers_y: Receivers per shot in X and Y
        shot_spacing_x, shot_spacing_y: Shot intervals
        receiver_spacing_x, receiver_spacing_y: Receiver intervals
        shot_origin: Origin of shot grid
        receiver_origin: Offset of receiver patch from shot

    Returns:
        Tuple of geometry arrays
    """
    n_shots = n_shots_x * n_shots_y
    n_receivers = n_receivers_x * n_receivers_y
    n_traces = n_shots * n_receivers

    trace_indices = np.arange(n_traces, dtype=np.int64)
    source_x = np.zeros(n_traces, dtype=np.float64)
    source_y = np.zeros(n_traces, dtype=np.float64)
    receiver_x = np.zeros(n_traces, dtype=np.float64)
    receiver_y = np.zeros(n_traces, dtype=np.float64)
    shot_ids = np.zeros(n_traces, dtype=np.int32)

    idx = 0
    shot_id = 0

    for isx in range(n_shots_x):
        for isy in range(n_shots_y):
            sx = shot_origin[0] + isx * shot_spacing_x
            sy = shot_origin[1] + isy * shot_spacing_y

            for irx in range(n_receivers_x):
                for iry in range(n_receivers_y):
                    rx = sx + receiver_origin[0] + irx * receiver_spacing_x
                    ry = sy + receiver_origin[1] + iry * receiver_spacing_y

                    source_x[idx] = sx
                    source_y[idx] = sy
                    receiver_x[idx] = rx
                    receiver_y[idx] = ry
                    shot_ids[idx] = shot_id

                    idx += 1

            shot_id += 1

    return trace_indices, source_x, source_y, receiver_x, receiver_y, shot_ids


def generate_diffraction_response(
    n_traces: int,
    n_samples: int,
    sample_rate_ms: float,
    source_x: NDArray[np.float64],
    source_y: NDArray[np.float64],
    receiver_x: NDArray[np.float64],
    receiver_y: NDArray[np.float64],
    diffractor_x: float,
    diffractor_y: float,
    diffractor_z: float,
    velocity: float,
    wavelet_freq_hz: float = 30.0,
    amplitude: float = 1.0,
) -> NDArray[np.float32]:
    """
    Generate synthetic traces from a point diffractor.

    The diffractor response follows the DSR equation, making it
    ideal for migration validation.

    Args:
        n_traces: Number of traces
        n_samples: Samples per trace
        sample_rate_ms: Sample rate in milliseconds
        source_x, source_y: Source coordinates
        receiver_x, receiver_y: Receiver coordinates
        diffractor_x, diffractor_y, diffractor_z: Diffractor position
        velocity: Constant velocity (m/s)
        wavelet_freq_hz: Wavelet frequency
        amplitude: Diffractor amplitude

    Returns:
        Trace data array with diffraction response
    """
    # Generate wavelet
    wavelet = generate_ricker_wavelet(wavelet_freq_hz, sample_rate_ms)
    wavelet_half = len(wavelet) // 2

    # Initialize traces
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)

    for i in range(n_traces):
        # Compute travel time: source to diffractor + diffractor to receiver
        dist_s = np.sqrt(
            (source_x[i] - diffractor_x) ** 2 +
            (source_y[i] - diffractor_y) ** 2 +
            diffractor_z ** 2
        )
        dist_r = np.sqrt(
            (receiver_x[i] - diffractor_x) ** 2 +
            (receiver_y[i] - diffractor_y) ** 2 +
            diffractor_z ** 2
        )

        t_travel_s = (dist_s + dist_r) / velocity
        t_travel_ms = t_travel_s * 1000.0

        sample_idx = int(t_travel_ms / sample_rate_ms)

        # Add wavelet at travel time
        start = max(0, sample_idx - wavelet_half)
        end = min(n_samples, sample_idx + wavelet_half)
        wav_start = max(0, wavelet_half - sample_idx)
        wav_end = wav_start + (end - start)

        if end > start:
            # Apply geometrical spreading
            spread = 1.0 / (dist_s * dist_r) if dist_s > 0 and dist_r > 0 else 1.0
            traces[i, start:end] = amplitude * spread * 1e6 * wavelet[wav_start:wav_end]

    return traces


def create_synthetic_dataset(
    output_dir: Path | str,
    n_shots: int = 10,
    n_receivers: int = 50,
    n_samples: int = 1000,
    sample_rate_ms: float = 2.0,
    geometry_type: str = "2d",
) -> tuple[Path, Path]:
    """
    Create a complete synthetic dataset for testing.

    Args:
        output_dir: Output directory
        n_shots: Number of shots
        n_receivers: Receivers per shot
        n_samples: Samples per trace
        sample_rate_ms: Sample rate
        geometry_type: "2d" or "3d"

    Returns:
        Tuple of (traces_path, headers_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic {geometry_type.upper()} dataset...")

    # Generate geometry
    if geometry_type == "2d":
        (trace_indices, source_x, source_y,
         receiver_x, receiver_y, shot_ids) = generate_synthetic_geometry_2d(
            n_shots=n_shots,
            n_receivers_per_shot=n_receivers,
            shot_spacing=100.0,
            receiver_spacing=25.0,
        )
    else:
        n_shots_xy = int(np.sqrt(n_shots))
        n_rec_xy = int(np.sqrt(n_receivers))
        (trace_indices, source_x, source_y,
         receiver_x, receiver_y, shot_ids) = generate_synthetic_geometry_3d(
            n_shots_x=n_shots_xy,
            n_shots_y=n_shots_xy,
            n_receivers_x=n_rec_xy,
            n_receivers_y=n_rec_xy,
            shot_spacing_x=100.0,
            shot_spacing_y=100.0,
            receiver_spacing_x=25.0,
            receiver_spacing_y=25.0,
        )

    n_traces = len(trace_indices)

    # Generate traces with a point diffractor
    midpoint_x = (source_x.mean() + receiver_x.mean()) / 2
    midpoint_y = (source_y.mean() + receiver_y.mean()) / 2

    traces = generate_diffraction_response(
        n_traces=n_traces,
        n_samples=n_samples,
        sample_rate_ms=sample_rate_ms,
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        diffractor_x=midpoint_x,
        diffractor_y=midpoint_y,
        diffractor_z=1000.0,  # 1km depth
        velocity=2000.0,
        wavelet_freq_hz=30.0,
    )

    # Save traces
    traces_path = output_dir / "traces.zarr"
    z = create_zarr_traces(
        traces_path,
        n_traces=n_traces,
        n_samples=n_samples,
        sample_rate_ms=sample_rate_ms,
    )
    z[:] = traces

    # Save headers
    headers_path = output_dir / "headers.parquet"
    create_parquet_headers(
        headers_path,
        trace_indices=trace_indices,
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        shot_ids=shot_ids,
    )

    logger.info(
        f"Created synthetic dataset: {n_traces} traces, {n_samples} samples, "
        f"at {output_dir}"
    )

    return traces_path, headers_path
