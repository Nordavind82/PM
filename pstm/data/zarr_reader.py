"""
Zarr trace data reader for PSTM.

Provides efficient access to seismic trace data stored in Zarr format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numpy.typing import NDArray

from pstm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TraceDataInfo:
    """Information about trace data array."""

    path: Path
    shape: tuple[int, int]  # (n_traces, n_samples)
    dtype: np.dtype
    chunks: tuple[int, int]
    compressor: str | None
    sample_rate_ms: float | None
    start_time_ms: float | None

    @property
    def n_traces(self) -> int:
        """Number of traces."""
        return self.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of samples per trace."""
        return self.shape[1]

    @property
    def size_bytes(self) -> int:
        """Uncompressed size in bytes."""
        return int(np.prod(self.shape) * self.dtype.itemsize)

    @property
    def size_gb(self) -> float:
        """Uncompressed size in GB."""
        return self.size_bytes / (1024**3)


class ZarrTraceReader:
    """
    Reader for seismic trace data stored in Zarr format.

    Expected Zarr array structure:
    - Shape: (n_traces, n_samples) [standard] or (n_samples, n_traces) [transposed]
    - Dtype: float32 or float64
    - Attributes: sample_rate_ms, start_time_ms (optional)

    Provides:
    - Metadata access
    - Selective trace loading by indices
    - Batch loading with pre-allocated buffers
    - Async prefetching (optional)
    """

    def __init__(
        self,
        path: Path | str,
        mode: str = "r",
        sample_rate_ms: float | None = None,
        start_time_ms: float | None = None,
        transposed: bool = False,
        n_traces: int | None = None,
        n_samples: int | None = None,
    ):
        """
        Initialize the Zarr trace reader.

        Args:
            path: Path to Zarr array
            mode: Open mode ('r' for read-only)
            sample_rate_ms: Override sample rate (auto-detected if None)
            start_time_ms: Override start time (auto-detected if None)
            transposed: If True, data is stored as (n_samples, n_traces) instead of (n_traces, n_samples)
            n_traces: Override number of traces (required if transposed)
            n_samples: Override number of samples (required if transposed)
        """
        self.path = Path(path)
        self._mode = mode
        self._zarr: zarr.Array | None = None
        self._info: TraceDataInfo | None = None

        # Override values
        self._sample_rate_override = sample_rate_ms
        self._start_time_override = start_time_ms
        self._transposed = transposed
        self._n_traces_override = n_traces
        self._n_samples_override = n_samples

        # Pre-allocated buffer for batch loading
        self._buffer: NDArray[np.float32] | None = None
        self._buffer_size: int = 0

    def open(self) -> "ZarrTraceReader":
        """Open the Zarr array."""
        if self._zarr is not None:
            return self

        if not self.path.exists():
            raise FileNotFoundError(f"Zarr array not found: {self.path}")

        logger.debug(f"Opening Zarr array: {self.path}")
        self._zarr = zarr.open(str(self.path), mode=self._mode)

        # Validate shape
        if self._zarr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {self._zarr.ndim}D")

        # Build info
        self._info = self._build_info()
        logger.info(
            f"Opened trace data: {self._info.n_traces:,} traces × "
            f"{self._info.n_samples:,} samples ({self._info.size_gb:.2f} GB)"
        )

        return self

    def close(self) -> None:
        """Close the Zarr array."""
        self._zarr = None
        self._buffer = None
        self._buffer_size = 0

    def __enter__(self) -> "ZarrTraceReader":
        """Context manager entry."""
        return self.open()

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()

    def _build_info(self) -> TraceDataInfo:
        """Build trace data info from Zarr array."""
        assert self._zarr is not None

        # Get sample rate from attributes or override
        sample_rate = self._sample_rate_override
        if sample_rate is None:
            sample_rate = self._zarr.attrs.get("sample_rate_ms")
            if sample_rate is None:
                sample_rate = self._zarr.attrs.get("dt")  # Alternative name
                if sample_rate is not None:
                    # Check if it's in seconds (small value) or ms
                    if sample_rate < 0.1:
                        sample_rate *= 1000  # Convert s to ms

        # Get start time from attributes or override
        start_time = self._start_time_override
        if start_time is None:
            start_time = self._zarr.attrs.get("start_time_ms")
            if start_time is None:
                start_time = self._zarr.attrs.get("t0", 0.0)

        # Get compressor name (handle Zarr v2 and v3 differences)
        compressor_name = None
        try:
            if hasattr(self._zarr, 'compressor') and self._zarr.compressor is not None:
                compressor_name = str(self._zarr.compressor)
        except (TypeError, AttributeError):
            # Zarr v3 doesn't have .compressor attribute
            pass

        # Determine shape - handle transposed data
        zarr_shape = self._zarr.shape
        if self._transposed:
            # Data is stored as (n_samples, n_traces)
            if self._n_traces_override is not None and self._n_samples_override is not None:
                shape = (self._n_traces_override, self._n_samples_override)
            else:
                # Infer: first dim is samples, second is traces
                shape = (zarr_shape[1], zarr_shape[0])
            logger.debug(f"Transposed data: zarr shape {zarr_shape} -> logical shape {shape}")
        else:
            shape = zarr_shape

        return TraceDataInfo(
            path=self.path,
            shape=shape,
            dtype=self._zarr.dtype,
            chunks=self._zarr.chunks,
            compressor=compressor_name,
            sample_rate_ms=sample_rate,
            start_time_ms=start_time,
        )

    @property
    def info(self) -> TraceDataInfo:
        """Get trace data information."""
        if self._info is None:
            self.open()
        assert self._info is not None
        return self._info

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape (n_traces, n_samples)."""
        return self.info.shape

    @property
    def n_traces(self) -> int:
        """Number of traces."""
        return self.info.n_traces

    @property
    def n_samples(self) -> int:
        """Number of samples per trace."""
        return self.info.n_samples

    @property
    def sample_rate_ms(self) -> float | None:
        """Sample rate in milliseconds."""
        return self.info.sample_rate_ms

    @property
    def time_axis(self) -> NDArray[np.float64] | None:
        """Time axis in milliseconds."""
        if self.sample_rate_ms is None:
            return None
        start = self.info.start_time_ms or 0.0
        return np.arange(self.n_samples) * self.sample_rate_ms + start

    def get_traces(
        self,
        indices: NDArray[np.int64] | list[int] | slice,
        out: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """
        Load traces by index.

        Args:
            indices: Trace indices to load (array, list, or slice)
            out: Optional pre-allocated output array

        Returns:
            Array of shape (n_selected, n_samples)
        """
        if self._zarr is None:
            self.open()
        assert self._zarr is not None

        # Handle different index types
        if isinstance(indices, slice):
            if self._transposed:
                # Data is (n_samples, n_traces) - need to select columns and transpose
                data = self._zarr[:, indices].T
            else:
                data = self._zarr[indices]
        else:
            indices = np.asarray(indices, dtype=np.int64)
            if len(indices) == 0:
                return np.empty((0, self.n_samples), dtype=np.float32)

            if self._transposed:
                # Data is stored as (n_samples, n_traces)
                # We need to read columns (trace indices) and transpose result
                try:
                    # Try orthogonal indexing for columns
                    data = self._zarr.oindex[:, indices].T  # (n_samples, n_selected) -> (n_selected, n_samples)
                except (AttributeError, TypeError):
                    # Fallback: load traces one by one
                    data = np.empty((len(indices), self.n_samples), dtype=self._zarr.dtype)
                    for i, idx in enumerate(indices):
                        data[i] = self._zarr[:, int(idx)]
            else:
                # Standard format: (n_traces, n_samples)
                # Zarr 3 requires different approach - use oindex for orthogonal indexing
                # or load traces individually for robustness
                try:
                    # Try orthogonal indexing (works in zarr 3)
                    data = self._zarr.oindex[indices, :]
                except (AttributeError, TypeError):
                    # Fallback: load traces one by one (slower but robust)
                    data = np.empty((len(indices), self.n_samples), dtype=self._zarr.dtype)
                    for i, idx in enumerate(indices):
                        data[i] = self._zarr[int(idx)]

        # Convert to float32 if needed
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Copy to output array if provided
        if out is not None:
            out[:] = data
            return out

        return data

    def get_trace(self, index: int) -> NDArray[np.float32]:
        """
        Load a single trace.

        Args:
            index: Trace index

        Returns:
            Array of shape (n_samples,)
        """
        if self._zarr is None:
            self.open()
        assert self._zarr is not None

        if self._transposed:
            # Data is (n_samples, n_traces) - read column
            data = self._zarr[:, index]
        else:
            data = self._zarr[index]

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        return data

    def allocate_buffer(self, n_traces: int) -> NDArray[np.float32]:
        """
        Allocate or resize the internal buffer for batch loading.

        Args:
            n_traces: Number of traces the buffer should hold

        Returns:
            Buffer array of shape (n_traces, n_samples)
        """
        if self._buffer is None or self._buffer_size < n_traces:
            self._buffer = np.empty((n_traces, self.n_samples), dtype=np.float32)
            self._buffer_size = n_traces
            logger.debug(f"Allocated trace buffer: {n_traces} × {self.n_samples}")

        return self._buffer[:n_traces]

    def get_traces_buffered(
        self,
        indices: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        """
        Load traces using internal buffer.

        The buffer is automatically resized if needed.

        Args:
            indices: Trace indices to load

        Returns:
            View into buffer with loaded traces
        """
        n_traces = len(indices)
        buffer = self.allocate_buffer(n_traces)
        return self.get_traces(indices, out=buffer)

    def iter_chunks(
        self,
        chunk_size: int = 1000,
    ):
        """
        Iterate over traces in chunks.

        Args:
            chunk_size: Number of traces per chunk

        Yields:
            Tuple of (start_index, traces_array)
        """
        for start in range(0, self.n_traces, chunk_size):
            end = min(start + chunk_size, self.n_traces)
            indices = np.arange(start, end)
            traces = self.get_traces(indices)
            yield start, traces

    def get_attributes(self) -> dict[str, Any]:
        """Get all Zarr attributes."""
        if self._zarr is None:
            self.open()
        assert self._zarr is not None
        return dict(self._zarr.attrs)


def create_zarr_traces(
    path: Path | str,
    n_traces: int,
    n_samples: int,
    sample_rate_ms: float,
    start_time_ms: float = 0.0,
    chunks: tuple[int, int] | None = None,
    dtype: np.dtype = np.float32,
    compressor: str = "blosc",
) -> zarr.Array:
    """
    Create a new Zarr array for trace data.

    Args:
        path: Output path
        n_traces: Number of traces
        n_samples: Samples per trace
        sample_rate_ms: Sample rate in milliseconds
        start_time_ms: Start time in milliseconds
        chunks: Chunk shape (default: auto)
        dtype: Data type
        compressor: Compression codec

    Returns:
        Opened Zarr array in write mode
    """
    if chunks is None:
        # Default: chunks of ~1000 traces, full time axis
        chunks = (min(1000, n_traces), n_samples)

    # Use zarr_format=2 for compatibility with numcodecs compressors
    try:
        from numcodecs import Blosc
        if compressor == "blosc":
            comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        else:
            comp = None

        z = zarr.open(
            str(path),
            mode="w",
            shape=(n_traces, n_samples),
            chunks=chunks,
            dtype=dtype,
            compressor=comp,
            zarr_format=2,
        )
    except TypeError:
        # Fallback for older zarr versions without zarr_format parameter
        from numcodecs import Blosc
        if compressor == "blosc":
            comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        else:
            comp = None

        z = zarr.open(
            str(path),
            mode="w",
            shape=(n_traces, n_samples),
            chunks=chunks,
            dtype=dtype,
            compressor=comp,
        )

    # Set attributes
    z.attrs["sample_rate_ms"] = sample_rate_ms
    z.attrs["start_time_ms"] = start_time_ms
    z.attrs["n_traces"] = n_traces
    z.attrs["n_samples"] = n_samples

    logger.info(f"Created Zarr trace array: {path}")

    return z
