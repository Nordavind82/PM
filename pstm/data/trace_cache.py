"""
LRU Trace Cache for PSTM.

Provides a Least Recently Used cache for trace data to reduce redundant
Zarr reads when processing tiles with overlapping apertures.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pstm.data.zarr_reader import ZarrTraceReader


class LRUTraceCache:
    """
    LRU cache for trace data.

    Caches individual traces by their index to avoid redundant reads when
    tiles have overlapping apertures. Uses an OrderedDict for O(1) LRU
    operations.

    Attributes:
        max_size_mb: Maximum cache size in megabytes
        _cache: OrderedDict mapping trace_index -> trace_data
        _current_size_bytes: Current cache size in bytes
        _hits: Number of cache hits
        _misses: Number of cache misses
    """

    def __init__(self, max_size_mb: float = 1000.0):
        """
        Initialize the trace cache.

        Args:
            max_size_mb: Maximum cache size in megabytes (default: 1GB)
        """
        self.max_size_mb = max_size_mb
        self._max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._cache: OrderedDict[int, NDArray[np.float32]] = OrderedDict()
        self._current_size_bytes = 0
        self._hits = 0
        self._misses = 0
        self._bytes_per_trace = 0  # Set on first access

    def get_traces(
        self,
        indices: NDArray[np.int64] | list[int],
        reader: "ZarrTraceReader",
    ) -> NDArray[np.float32]:
        """
        Get traces by indices, using cache when possible.

        Args:
            indices: Array of trace indices to retrieve
            reader: ZarrTraceReader to load uncached traces

        Returns:
            Array of trace data with shape (n_traces, n_samples)
        """
        indices = np.asarray(indices, dtype=np.int64)
        n_traces = len(indices)

        if n_traces == 0:
            return np.empty((0, reader.n_samples), dtype=np.float32)

        # Set bytes per trace on first access
        if self._bytes_per_trace == 0:
            self._bytes_per_trace = reader.n_samples * 4  # float32

        # Allocate output array
        result = np.empty((n_traces, reader.n_samples), dtype=np.float32)

        # Find which traces are cached and which need loading
        cached_mask = np.zeros(n_traces, dtype=bool)
        uncached_indices = []
        uncached_positions = []

        for i, idx in enumerate(indices):
            idx_int = int(idx)
            if idx_int in self._cache:
                # Cache hit - move to end (most recently used)
                self._cache.move_to_end(idx_int)
                result[i] = self._cache[idx_int]
                cached_mask[i] = True
                self._hits += 1
            else:
                uncached_indices.append(idx_int)
                uncached_positions.append(i)
                self._misses += 1

        # Load uncached traces if any
        if uncached_indices:
            uncached_data = reader.get_traces(np.array(uncached_indices, dtype=np.int64))

            # Store in result and cache
            for j, (idx, pos) in enumerate(zip(uncached_indices, uncached_positions)):
                trace_data = uncached_data[j]
                result[pos] = trace_data

                # Add to cache (may evict old entries)
                self._add_to_cache(idx, trace_data)

        return result

    def _add_to_cache(self, idx: int, trace_data: NDArray[np.float32]) -> None:
        """Add a trace to the cache, evicting old entries if needed."""
        trace_size = trace_data.nbytes

        # Don't cache if single trace is larger than max
        if trace_size > self._max_size_bytes:
            return

        # Evict old entries until there's room
        while self._current_size_bytes + trace_size > self._max_size_bytes and self._cache:
            # Remove oldest (first) item
            _, old_trace = self._cache.popitem(last=False)
            self._current_size_bytes -= old_trace.nbytes

        # Add new trace
        self._cache[idx] = trace_data.copy()  # Copy to avoid reference issues
        self._current_size_bytes += trace_size

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": hit_rate,
            "size_mb": self._current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "n_cached_traces": len(self._cache),
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._current_size_bytes = 0
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        """Number of traces in cache."""
        return len(self._cache)

    @property
    def size_mb(self) -> float:
        """Current cache size in MB."""
        return self._current_size_bytes / (1024 * 1024)
