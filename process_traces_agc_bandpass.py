#!/usr/bin/env python3
"""
Process seismic traces with bandpass filter and AGC.

Applies:
- Bandpass filter: 8-80 Hz
- AGC: 500 ms window

Creates a new copy of the data in zarr/parquet format.
"""

import sys
from pathlib import Path
import shutil
import time

import numpy as np
import zarr
import polars as pl
from numba import njit, prange
from scipy.signal import butter, sosfiltfilt
from numpy.typing import NDArray


# ============================================================================
# Trace processing functions (from pstm_va)
# ============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def apply_agc_numba(
    traces: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    """Apply AGC to traces using sliding window RMS normalization."""
    n_traces, n_samples = traces.shape
    half_window = window_samples // 2

    for itrace in prange(n_traces):
        trace = traces[itrace, :]

        for i in range(n_samples):
            i_start = max(0, i - half_window)
            i_end = min(n_samples, i + half_window + 1)

            window_sum_sq = 0.0
            for j in range(i_start, i_end):
                window_sum_sq += trace[j] * trace[j]

            window_len = i_end - i_start
            rms = np.sqrt(window_sum_sq / window_len)

            if rms > 1e-10:
                traces[itrace, i] = trace[i] / rms

    return traces


def apply_agc(
    traces: NDArray[np.float32],
    window_ms: float,
    dt_ms: float,
) -> NDArray[np.float32]:
    """Apply AGC (Automatic Gain Control) to traces."""
    window_samples = max(3, int(window_ms / dt_ms))
    result = traces.copy()
    return apply_agc_numba(result, window_samples)


def design_bandpass_filter(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> NDArray[np.float64]:
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs

    low = lowcut / nyq if lowcut > 0 else None
    high = highcut / nyq if highcut < nyq else None

    if low is not None and high is not None:
        sos = butter(order, [low, high], btype='band', output='sos')
    elif low is not None:
        sos = butter(order, low, btype='high', output='sos')
    elif high is not None:
        sos = butter(order, high, btype='low', output='sos')
    else:
        return None

    return sos


def apply_bandpass(
    traces: NDArray[np.float32],
    lowcut_hz: float,
    highcut_hz: float,
    dt_ms: float,
    order: int = 4,
) -> NDArray[np.float32]:
    """Apply bandpass filter to traces."""
    fs = 1000.0 / dt_ms

    sos = design_bandpass_filter(lowcut_hz, highcut_hz, fs, order)

    if sos is None:
        return traces.copy()

    result = sosfiltfilt(sos, traces, axis=1).astype(np.float32)

    return result


def process_trace_chunk(
    traces: NDArray[np.float32],
    dt_ms: float,
    agc_window_ms: float | None = None,
    bandpass_low_hz: float | None = None,
    bandpass_high_hz: float | None = None,
    bandpass_order: int = 4,
) -> NDArray[np.float32]:
    """Process a chunk of traces with optional AGC and bandpass."""
    result = traces

    # Apply bandpass first
    if bandpass_low_hz is not None or bandpass_high_hz is not None:
        low = bandpass_low_hz if bandpass_low_hz is not None else 0.0
        high = bandpass_high_hz if bandpass_high_hz is not None else 0.0

        if low > 0 or high > 0:
            result = apply_bandpass(result, low, high, dt_ms, bandpass_order)

    # Apply AGC second
    if agc_window_ms is not None and agc_window_ms > 0:
        result = apply_agc(result, agc_window_ms, dt_ms)

    return result


# ============================================================================
# Main processing functions
# ============================================================================


def process_traces(
    input_zarr_path: str,
    output_zarr_path: str,
    dt_ms: float = 2.0,
    agc_window_ms: float = 500.0,
    bandpass_low_hz: float = 8.0,
    bandpass_high_hz: float = 80.0,
    chunk_size: int = 50000,
    transposed: bool = True,
) -> None:
    """Process traces with bandpass and AGC.

    Args:
        input_zarr_path: Path to input traces zarr
        output_zarr_path: Path to output traces zarr
        dt_ms: Sample interval in ms
        agc_window_ms: AGC window in ms
        bandpass_low_hz: Bandpass low cut in Hz
        bandpass_high_hz: Bandpass high cut in Hz
        chunk_size: Number of traces to process at once
        transposed: If True, data is stored as (samples, traces)
    """
    print(f"Opening input: {input_zarr_path}")
    input_zarr = zarr.open(input_zarr_path)

    if transposed:
        n_samples, n_traces = input_zarr.shape
        print(f"Input shape (transposed): ({n_samples}, {n_traces})")
    else:
        n_traces, n_samples = input_zarr.shape
        print(f"Input shape: ({n_traces}, {n_samples})")

    print(f"Sample rate: {dt_ms} ms")
    print(f"Processing parameters:")
    print(f"  Bandpass: {bandpass_low_hz}-{bandpass_high_hz} Hz")
    print(f"  AGC window: {agc_window_ms} ms")

    # Create output zarr with same layout
    print(f"\nCreating output: {output_zarr_path}")
    output_path = Path(output_zarr_path)
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create output array with same shape and chunking
    if transposed:
        output_shape = (n_samples, n_traces)
        chunks = (n_samples, min(chunk_size, n_traces))
    else:
        output_shape = (n_traces, n_samples)
        chunks = (min(chunk_size, n_traces), n_samples)

    store = zarr.storage.LocalStore(str(output_path))
    output_zarr = zarr.create_array(
        store=store,
        shape=output_shape,
        dtype=np.float32,
        chunks=chunks,
        overwrite=True,
    )

    # Process in chunks
    n_chunks = (n_traces + chunk_size - 1) // chunk_size
    print(f"\nProcessing {n_traces:,} traces in {n_chunks} chunks of {chunk_size:,}")

    start_time = time.time()

    for i in range(n_chunks):
        idx_start = i * chunk_size
        idx_end = min((i + 1) * chunk_size, n_traces)

        if i % 10 == 0 or i == n_chunks - 1:
            pct = 100 * (i + 1) / n_chunks
            print(f"  Processing chunk {i+1}/{n_chunks} ({pct:.1f}%) - traces {idx_start:,}-{idx_end:,}")

        # Load chunk
        if transposed:
            # Data is (samples, traces), load as [:, idx_start:idx_end]
            chunk_data = input_zarr[:, idx_start:idx_end]
            # Transpose to (traces, samples) for processing
            chunk_data = chunk_data.T.astype(np.float32)
        else:
            chunk_data = input_zarr[idx_start:idx_end, :].astype(np.float32)

        # Process: bandpass first, then AGC
        processed = process_trace_chunk(
            traces=chunk_data,
            dt_ms=dt_ms,
            agc_window_ms=agc_window_ms,
            bandpass_low_hz=bandpass_low_hz,
            bandpass_high_hz=bandpass_high_hz,
        )

        # Write to output
        if transposed:
            # Transpose back to (samples, traces)
            output_zarr[:, idx_start:idx_end] = processed.T
        else:
            output_zarr[idx_start:idx_end, :] = processed

    elapsed = time.time() - start_time
    traces_per_sec = n_traces / elapsed

    print(f"\nProcessing complete!")
    print(f"  Time: {elapsed:.1f}s ({traces_per_sec:.0f} traces/s)")
    print(f"  Output: {output_zarr_path}")


def copy_headers(input_path: str, output_path: str) -> None:
    """Copy headers parquet file."""
    print(f"\nCopying headers: {input_path} -> {output_path}")
    shutil.copy2(input_path, output_path)
    print("  Done")


def main():
    # Input paths
    input_dir = Path("/Users/olegadamovich/SeismicData/processing/processed_scd_xsd_data_new_20251221_225219_20251221_232357/output")
    input_traces = input_dir / "traces.zarr"
    input_headers = input_dir / "headers.parquet"

    # Output paths
    output_dir = Path("/Users/olegadamovich/SeismicData/processing/processed_agc500_bp8_80")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_traces = output_dir / "traces.zarr"
    output_headers = output_dir / "headers.parquet"

    print("=" * 60)
    print("Seismic Trace Processing")
    print("=" * 60)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Process traces
    process_traces(
        input_zarr_path=str(input_traces),
        output_zarr_path=str(output_traces),
        dt_ms=2.0,
        agc_window_ms=500.0,
        bandpass_low_hz=8.0,
        bandpass_high_hz=80.0,
        chunk_size=100000,  # Process 100k traces at a time
        transposed=True,
    )

    # Copy headers
    copy_headers(str(input_headers), str(output_headers))

    # Also copy other metadata files if present
    for extra_file in ["metadata.json", "ensemble_index.parquet", "trace_index.parquet"]:
        src = input_dir / extra_file
        if src.exists():
            dst = output_dir / extra_file
            print(f"Copying {extra_file}...")
            shutil.copy2(src, dst)

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nProcessed data saved to: {output_dir}")
    print("\nTo use in PSTM, update the config to point to:")
    print(f"  traces_path: {output_traces}")
    print(f"  headers_path: {output_headers}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
