#!/usr/bin/env python3
"""
Sort seismic data to common offset gathers with AGC and bandpass filtering.
Input: /Volumes/AO_DISK/trace_math_output/
Output: /Users/olegadamovich/SeismicData/common_offset_gathers/

Optimized: reads sequentially, scatters to bins. Much faster than random access.
"""

import numpy as np
import zarr
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from numba import njit, prange
import time
import gc

# Parameters
INPUT_DIR = Path("/Volumes/AO_DISK/trace_math_output")
OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers")
OFFSET_BIN_SIZE = 50  # meters
AGC_WINDOW_MS = 500   # ms
BANDPASS_LOW = 2      # Hz
BANDPASS_HIGH = 120   # Hz
SAMPLE_RATE_MS = 2    # ms

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

@njit(parallel=True)
def apply_agc_batch(traces, window_samples):
    """Apply AGC to batch of traces (n_samples, n_traces)."""
    n_samples, n_traces = traces.shape
    output = np.zeros_like(traces)
    half_win = window_samples // 2

    for tr in prange(n_traces):
        trace = traces[:, tr]
        for i in range(n_samples):
            start = max(0, i - half_win)
            end = min(n_samples, i + half_win + 1)
            window = trace[start:end]
            rms = np.sqrt(np.mean(window * window))
            if rms > 1e-10:
                output[i, tr] = trace[i] / rms
            else:
                output[i, tr] = trace[i]

    return output

def main():
    print("=" * 60)
    print("Common Offset Sorting with AGC and Bandpass")
    print("=" * 60)
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Offset bin size: {OFFSET_BIN_SIZE}m")
    print(f"AGC window: {AGC_WINDOW_MS}ms")
    print(f"Bandpass: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load headers
    print("Loading headers...")
    t0 = time.time()
    headers = pd.read_parquet(INPUT_DIR / "headers.parquet")
    n_traces_total = len(headers)
    print(f"  {n_traces_total:,} traces loaded in {time.time()-t0:.1f}s")

    # Get offsets and assign bins
    print("Calculating offset bins...")
    offsets = headers['offset'].values
    max_offset = offsets.max()
    min_offset = offsets.min()
    print(f"  Offset range: {min_offset:.1f} - {max_offset:.1f} m")

    n_bins = int(np.ceil(max_offset / OFFSET_BIN_SIZE))
    print(f"  Number of offset bins: {n_bins}")

    bin_indices = (offsets / OFFSET_BIN_SIZE).astype(np.int32)
    headers['offset_bin'] = bin_indices

    # Count traces per bin and prepare write counters
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    for b in bin_indices:
        bin_counts[b] += 1
    print(f"  Traces per bin: {bin_counts[bin_counts>0].min()} - {bin_counts.max()}")

    # Open input zarr
    print("\nOpening input data...")
    traces_zarr = zarr.open(INPUT_DIR / "traces.zarr", mode='r')
    n_samples, _ = traces_zarr.shape
    print(f"  Shape: {n_samples} samples x {n_traces_total:,} traces")

    # Sampling frequency and filter
    fs = 1000.0 / SAMPLE_RATE_MS  # Hz
    agc_window_samples = int(AGC_WINDOW_MS / SAMPLE_RATE_MS)
    b, a = butter_bandpass(BANDPASS_LOW, BANDPASS_HIGH, fs)
    print(f"  Sample rate: {SAMPLE_RATE_MS}ms ({fs:.0f} Hz)")
    print(f"  AGC window: {agc_window_samples} samples")

    # Create output zarr files for each bin
    print("\nCreating output zarr stores...")
    output_zarrs = []
    bin_write_idx = np.zeros(n_bins, dtype=np.int64)  # Track write position per bin
    bin_headers_list = [[] for _ in range(n_bins)]

    for bin_idx in range(n_bins):
        if bin_counts[bin_idx] == 0:
            output_zarrs.append(None)
            continue

        bin_dir = OUTPUT_DIR / f"offset_bin_{bin_idx:02d}"
        bin_dir.mkdir(exist_ok=True)

        output_zarr = zarr.open(
            bin_dir / "traces.zarr",
            mode='w',
            shape=(int(n_samples), int(bin_counts[bin_idx])),
            chunks=(int(n_samples), int(min(10000, bin_counts[bin_idx]))),
            dtype=np.float32
        )
        output_zarrs.append(output_zarr)

    # Process in sequential chunks - read from disk sequentially
    print("\nProcessing data...")
    total_start = time.time()

    chunk_size = 200000  # Read 200k traces at a time (~1.3GB)
    n_chunks = (n_traces_total + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, n_traces_total)
        n_chunk = chunk_end - chunk_start

        t0 = time.time()

        # Load chunk of traces
        traces_chunk = traces_zarr[:, chunk_start:chunk_end].astype(np.float32)

        # Apply bandpass filter to entire chunk
        traces_chunk = filtfilt(b, a, traces_chunk, axis=0, padlen=min(100, n_samples-1)).astype(np.float32)

        # Apply AGC
        traces_chunk = apply_agc_batch(traces_chunk, agc_window_samples)

        # Get bin assignments for this chunk
        chunk_bins = bin_indices[chunk_start:chunk_end]
        chunk_headers = headers.iloc[chunk_start:chunk_end]

        # Batch traces by bin for efficient writing
        for unique_bin in np.unique(chunk_bins):
            if output_zarrs[unique_bin] is None:
                continue

            # Get indices for this bin within chunk
            bin_mask = chunk_bins == unique_bin
            local_indices = np.where(bin_mask)[0]

            # Get traces for this bin
            bin_traces = traces_chunk[:, local_indices]

            # Write batch to output
            write_start = bin_write_idx[unique_bin]
            write_end = write_start + len(local_indices)
            output_zarrs[unique_bin][:, write_start:write_end] = bin_traces

            bin_write_idx[unique_bin] = write_end

            # Save header info
            for local_idx in local_indices:
                bin_headers_list[unique_bin].append(chunk_headers.iloc[local_idx].to_dict())

        elapsed = time.time() - t0
        progress = (chunk_idx + 1) / n_chunks * 100
        rate = n_chunk / elapsed
        eta = (n_traces_total - chunk_end) / rate if rate > 0 else 0

        print(f"  Chunk {chunk_idx+1}/{n_chunks} ({progress:.1f}%) - "
              f"{n_chunk:,} traces in {elapsed:.1f}s ({rate:,.0f} tr/s) - "
              f"ETA: {eta/60:.1f}min", flush=True)

        del traces_chunk
        gc.collect()

    # Save headers for each bin
    print("\nSaving headers...")
    all_headers_list = []
    global_idx = 0

    for bin_idx in range(n_bins):
        if bin_counts[bin_idx] == 0 or len(bin_headers_list[bin_idx]) == 0:
            continue

        bin_dir = OUTPUT_DIR / f"offset_bin_{bin_idx:02d}"

        # Create DataFrame from list of dicts
        bin_headers_df = pd.DataFrame(bin_headers_list[bin_idx])
        bin_headers_df['bin_trace_idx'] = np.arange(len(bin_headers_df))
        bin_headers_df['global_trace_idx'] = np.arange(global_idx, global_idx + len(bin_headers_df))
        global_idx += len(bin_headers_df)

        bin_headers_df.to_parquet(bin_dir / "headers.parquet", index=False)
        all_headers_list.append(bin_headers_df)

        print(f"  Bin {bin_idx:2d}: {len(bin_headers_df):,} traces")

    # Save combined headers
    if all_headers_list:
        all_headers = pd.concat(all_headers_list, ignore_index=True)
        all_headers.to_parquet(OUTPUT_DIR / "all_headers.parquet", index=False)

    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Total traces processed: {global_idx:,}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  - {sum(1 for c in bin_counts if c > 0)} offset bin directories")
    print(f"  - all_headers.parquet: {global_idx:,} traces")

if __name__ == "__main__":
    main()
