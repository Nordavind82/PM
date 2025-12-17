"""
I/O optimization module for PSTM.

Provides asynchronous I/O, double buffering, and prefetching.
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Any, Callable, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from pstm.utils.logging import get_logger

logger = get_logger(__name__)


T = TypeVar('T')


@dataclass
class BufferSlot(Generic[T]):
    """A slot in a double buffer."""

    data: T | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    in_use: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        """Clear the slot."""
        self.data = None
        self.ready.clear()
        self.in_use = False
        self.metadata.clear()


class DoubleBuffer(Generic[T]):
    """
    Double buffer for overlapping I/O with computation.

    While one buffer is being processed, the other is being filled.

    Usage:
        buffer = DoubleBuffer()

        # Producer fills buffers
        buffer.begin_fill(0)
        buffer.slots[0].data = load_data()
        buffer.end_fill(0)

        # Consumer processes buffers
        slot_id = buffer.wait_for_ready()
        process(buffer.slots[slot_id].data)
        buffer.release(slot_id)
    """

    def __init__(self, n_slots: int = 2):
        """
        Initialize double buffer.

        Args:
            n_slots: Number of buffer slots (default 2)
        """
        self.n_slots = n_slots
        self.slots: list[BufferSlot[T]] = [BufferSlot() for _ in range(n_slots)]

        self._fill_lock = threading.Lock()
        self._ready_queue: Queue[int] = Queue()
        self._next_fill_slot = 0

    def get_fill_slot(self) -> int | None:
        """
        Get next available slot for filling.

        Returns:
            Slot index or None if all slots are in use
        """
        with self._fill_lock:
            for i in range(self.n_slots):
                slot_id = (self._next_fill_slot + i) % self.n_slots
                slot = self.slots[slot_id]
                if not slot.in_use and not slot.ready.is_set():
                    slot.in_use = True
                    self._next_fill_slot = (slot_id + 1) % self.n_slots
                    return slot_id
        return None

    def begin_fill(self, slot_id: int) -> BufferSlot[T]:
        """
        Begin filling a slot.

        Args:
            slot_id: Slot index

        Returns:
            The buffer slot
        """
        slot = self.slots[slot_id]
        slot.in_use = True
        slot.ready.clear()
        return slot

    def end_fill(self, slot_id: int) -> None:
        """
        Mark a slot as ready for processing.

        Args:
            slot_id: Slot index
        """
        slot = self.slots[slot_id]
        slot.ready.set()
        self._ready_queue.put(slot_id)

    def wait_for_ready(self, timeout: float | None = None) -> int | None:
        """
        Wait for a ready slot.

        Args:
            timeout: Timeout in seconds

        Returns:
            Slot index or None if timeout
        """
        try:
            return self._ready_queue.get(timeout=timeout)
        except Empty:
            return None

    def release(self, slot_id: int) -> None:
        """
        Release a slot after processing.

        Args:
            slot_id: Slot index
        """
        slot = self.slots[slot_id]
        slot.clear()

    def is_full(self) -> bool:
        """Check if all slots are in use or ready."""
        return all(s.in_use or s.ready.is_set() for s in self.slots)

    def is_empty(self) -> bool:
        """Check if all slots are available."""
        return all(not s.in_use and not s.ready.is_set() for s in self.slots)


class AsyncTraceLoader:
    """
    Asynchronous trace loader with prefetching.

    Loads trace data in background threads while migration proceeds.
    """

    def __init__(
        self,
        trace_reader,  # ZarrTraceReader
        header_manager,  # ParquetHeaderManager
        n_workers: int = 2,
        buffer_size: int = 3,
    ):
        """
        Initialize async loader.

        Args:
            trace_reader: Zarr trace reader
            header_manager: Parquet header manager
            n_workers: Number of background loading threads
            buffer_size: Number of buffer slots
        """
        self.trace_reader = trace_reader
        self.header_manager = header_manager
        self.n_workers = n_workers

        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        self._buffer = DoubleBuffer(n_slots=buffer_size)
        self._pending_futures: dict[int, Future] = {}
        self._shutdown = False

    def submit_load(
        self,
        tile_id: int,
        trace_indices: NDArray[np.int64],
    ) -> bool:
        """
        Submit a load request for background execution.

        Args:
            tile_id: Tile identifier
            trace_indices: Trace indices to load

        Returns:
            True if submitted, False if buffer full
        """
        slot_id = self._buffer.get_fill_slot()
        if slot_id is None:
            return False

        # Submit load task
        future = self._executor.submit(
            self._load_task,
            slot_id,
            tile_id,
            trace_indices,
        )
        self._pending_futures[slot_id] = future

        return True

    def _load_task(
        self,
        slot_id: int,
        tile_id: int,
        trace_indices: NDArray[np.int64],
    ) -> None:
        """Background load task."""
        try:
            slot = self._buffer.begin_fill(slot_id)

            # Load trace data
            trace_data = self.trace_reader.get_traces(trace_indices)

            # Load geometry
            geometry = self.header_manager.get_geometry_for_indices(trace_indices)

            # Store in slot
            slot.data = {
                'traces': trace_data,
                'geometry': geometry,
            }
            slot.metadata['tile_id'] = tile_id
            slot.metadata['n_traces'] = len(trace_indices)

            self._buffer.end_fill(slot_id)

        except Exception as e:
            logger.error(f"Error loading tile {tile_id}: {e}")
            self._buffer.slots[slot_id].clear()

    def get_ready_data(
        self,
        timeout: float = 5.0,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """
        Get next ready data batch.

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (data_dict, metadata) or None if timeout
        """
        slot_id = self._buffer.wait_for_ready(timeout=timeout)
        if slot_id is None:
            return None

        slot = self._buffer.slots[slot_id]
        data = slot.data
        metadata = slot.metadata.copy()

        self._buffer.release(slot_id)

        return data, metadata

    def shutdown(self) -> None:
        """Shutdown the loader."""
        self._shutdown = True
        self._executor.shutdown(wait=True)


class AsyncWriter:
    """
    Asynchronous output writer.

    Writes output data in background while migration continues.
    """

    def __init__(
        self,
        output_path,
        n_workers: int = 1,
        queue_size: int = 10,
    ):
        """
        Initialize async writer.

        Args:
            output_path: Output file path
            n_workers: Number of writer threads
            queue_size: Maximum queue size
        """
        self.output_path = output_path
        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        self._write_queue: Queue = Queue(maxsize=queue_size)
        self._pending_futures: list[Future] = []

    def submit_write(
        self,
        data: NDArray,
        slice_info: tuple[slice, ...],
        callback: Callable | None = None,
    ) -> Future:
        """
        Submit data for background writing.

        Args:
            data: Data to write
            slice_info: Slices specifying where to write
            callback: Optional callback after write

        Returns:
            Future for the write operation
        """
        future = self._executor.submit(
            self._write_task,
            data.copy(),  # Copy to avoid race conditions
            slice_info,
            callback,
        )
        self._pending_futures.append(future)
        return future

    def _write_task(
        self,
        data: NDArray,
        slice_info: tuple[slice, ...],
        callback: Callable | None,
    ) -> None:
        """Background write task."""
        try:
            # In a real implementation, would write to self.output_path
            # For now, just simulate the write
            pass

            if callback:
                callback()

        except Exception as e:
            logger.error(f"Error writing data: {e}")

    def wait_all(self) -> None:
        """Wait for all pending writes to complete."""
        for future in self._pending_futures:
            future.result()
        self._pending_futures.clear()

    def shutdown(self) -> None:
        """Shutdown the writer."""
        self.wait_all()
        self._executor.shutdown(wait=True)


class Prefetcher:
    """
    Generic prefetcher for sequential access patterns.

    Prefetches the next N items while current items are being processed.
    """

    def __init__(
        self,
        load_func: Callable[[int], T],
        n_prefetch: int = 2,
    ):
        """
        Initialize prefetcher.

        Args:
            load_func: Function to load item by index
            n_prefetch: Number of items to prefetch
        """
        self.load_func = load_func
        self.n_prefetch = n_prefetch

        self._executor = ThreadPoolExecutor(max_workers=n_prefetch)
        self._cache: dict[int, T] = {}
        self._futures: dict[int, Future] = {}
        self._lock = threading.Lock()

    def prefetch(self, indices: list[int]) -> None:
        """
        Start prefetching items.

        Args:
            indices: List of indices to prefetch
        """
        with self._lock:
            for idx in indices:
                if idx not in self._cache and idx not in self._futures:
                    future = self._executor.submit(self.load_func, idx)
                    self._futures[idx] = future

    def get(self, idx: int) -> T:
        """
        Get an item, waiting if necessary.

        Args:
            idx: Item index

        Returns:
            The loaded item
        """
        with self._lock:
            if idx in self._cache:
                return self._cache.pop(idx)

            if idx in self._futures:
                future = self._futures.pop(idx)
                return future.result()

        # Not prefetched, load synchronously
        return self.load_func(idx)

    def update_prefetch(self, current_idx: int, total: int) -> None:
        """
        Update prefetch based on current position.

        Args:
            current_idx: Current processing index
            total: Total number of items
        """
        # Prefetch next N items
        indices = [
            i for i in range(current_idx + 1, min(current_idx + 1 + self.n_prefetch, total))
        ]
        self.prefetch(indices)

        # Clean up old cached items
        with self._lock:
            old_keys = [k for k in self._cache.keys() if k < current_idx]
            for k in old_keys:
                del self._cache[k]

    def shutdown(self) -> None:
        """Shutdown the prefetcher."""
        self._executor.shutdown(wait=True)


class IOStats:
    """Track I/O statistics."""

    def __init__(self):
        self.reads = 0
        self.read_bytes = 0
        self.read_time_s = 0.0
        self.writes = 0
        self.write_bytes = 0
        self.write_time_s = 0.0
        self._lock = threading.Lock()

    def record_read(self, n_bytes: int, elapsed_s: float) -> None:
        """Record a read operation."""
        with self._lock:
            self.reads += 1
            self.read_bytes += n_bytes
            self.read_time_s += elapsed_s

    def record_write(self, n_bytes: int, elapsed_s: float) -> None:
        """Record a write operation."""
        with self._lock:
            self.writes += 1
            self.write_bytes += n_bytes
            self.write_time_s += elapsed_s

    @property
    def read_throughput_mbps(self) -> float:
        """Read throughput in MB/s."""
        if self.read_time_s > 0:
            return (self.read_bytes / (1024 ** 2)) / self.read_time_s
        return 0.0

    @property
    def write_throughput_mbps(self) -> float:
        """Write throughput in MB/s."""
        if self.write_time_s > 0:
            return (self.write_bytes / (1024 ** 2)) / self.write_time_s
        return 0.0

    def get_summary(self) -> dict[str, Any]:
        """Get statistics summary."""
        return {
            "reads": self.reads,
            "read_mb": self.read_bytes / (1024 ** 2),
            "read_time_s": self.read_time_s,
            "read_throughput_mbps": self.read_throughput_mbps,
            "writes": self.writes,
            "write_mb": self.write_bytes / (1024 ** 2),
            "write_time_s": self.write_time_s,
            "write_throughput_mbps": self.write_throughput_mbps,
        }
