"""
Memory-mapped file manager for PSTM.

Provides efficient memory-mapped I/O for large arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pstm.utils.logging import get_logger
from pstm.utils.units import format_bytes

logger = get_logger(__name__)


@dataclass
class BufferInfo:
    """Information about a memory-mapped buffer."""

    name: str
    path: Path
    shape: tuple[int, ...]
    dtype: np.dtype
    size_bytes: int
    mode: str

    @property
    def size_gb(self) -> float:
        """Size in GB."""
        return self.size_bytes / (1024**3)


class MemmapManager:
    """
    Manager for memory-mapped arrays.

    Handles creation, tracking, and cleanup of memory-mapped files.
    """

    def __init__(self, work_dir: Path | str):
        """
        Initialize memmap manager.

        Args:
            work_dir: Working directory for temporary files
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._buffers: dict[str, tuple[NDArray, BufferInfo]] = {}
        self._total_size: int = 0

        logger.debug(f"MemmapManager initialized: {self.work_dir}")

    def create(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype | type = np.float32,
        fill_value: float | None = 0.0,
        mode: str = "w+",
    ) -> NDArray:
        """
        Create a new memory-mapped array.

        Args:
            name: Buffer name (used for file naming)
            shape: Array shape
            dtype: Data type
            fill_value: Initial fill value (None = uninitialized)
            mode: File mode ('w+' = create/overwrite, 'r+' = read/write existing)

        Returns:
            Memory-mapped array
        """
        if name in self._buffers:
            logger.warning(f"Buffer '{name}' already exists, will be replaced")
            self.release(name)

        dtype = np.dtype(dtype)
        path = self.work_dir / f"{name}.dat"
        size_bytes = int(np.prod(shape)) * dtype.itemsize

        logger.debug(f"Creating memmap '{name}': {shape}, {dtype}, {format_bytes(size_bytes)}")

        # Create memory-mapped file
        mmap = np.memmap(
            str(path),
            dtype=dtype,
            mode=mode,
            shape=shape,
        )

        # Initialize if requested
        if fill_value is not None:
            mmap[:] = fill_value
            mmap.flush()

        # Track buffer
        info = BufferInfo(
            name=name,
            path=path,
            shape=shape,
            dtype=dtype,
            size_bytes=size_bytes,
            mode=mode,
        )
        self._buffers[name] = (mmap, info)
        self._total_size += size_bytes

        logger.info(f"Created buffer '{name}': {shape}, {format_bytes(size_bytes)}")

        return mmap

    def open(
        self,
        name: str,
        path: Path | str,
        shape: tuple[int, ...],
        dtype: np.dtype | type = np.float32,
        mode: str = "r",
    ) -> NDArray:
        """
        Open an existing memory-mapped file.

        Args:
            name: Buffer name
            path: File path
            shape: Array shape
            dtype: Data type
            mode: File mode ('r' = read-only, 'r+' = read/write)

        Returns:
            Memory-mapped array
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Memmap file not found: {path}")

        dtype = np.dtype(dtype)
        size_bytes = int(np.prod(shape)) * dtype.itemsize

        mmap = np.memmap(
            str(path),
            dtype=dtype,
            mode=mode,
            shape=shape,
        )

        info = BufferInfo(
            name=name,
            path=path,
            shape=shape,
            dtype=dtype,
            size_bytes=size_bytes,
            mode=mode,
        )
        self._buffers[name] = (mmap, info)
        self._total_size += size_bytes

        logger.info(f"Opened buffer '{name}': {shape}, {format_bytes(size_bytes)}")

        return mmap

    def get(self, name: str) -> NDArray:
        """
        Get a buffer by name.

        Args:
            name: Buffer name

        Returns:
            Memory-mapped array
        """
        if name not in self._buffers:
            raise KeyError(f"Buffer not found: {name}")
        return self._buffers[name][0]

    def get_info(self, name: str) -> BufferInfo:
        """
        Get buffer info by name.

        Args:
            name: Buffer name

        Returns:
            BufferInfo object
        """
        if name not in self._buffers:
            raise KeyError(f"Buffer not found: {name}")
        return self._buffers[name][1]

    def flush(self, name: str | None = None) -> None:
        """
        Flush buffer(s) to disk.

        Args:
            name: Buffer name (None = flush all)
        """
        if name is not None:
            if name in self._buffers:
                self._buffers[name][0].flush()
        else:
            for mmap, _ in self._buffers.values():
                mmap.flush()

    def release(self, name: str, delete_file: bool = False) -> None:
        """
        Release a buffer.

        Args:
            name: Buffer name
            delete_file: Also delete the backing file
        """
        if name not in self._buffers:
            return

        mmap, info = self._buffers.pop(name)
        self._total_size -= info.size_bytes

        # Flush and delete memmap
        mmap.flush()
        del mmap

        # Delete file if requested
        if delete_file and info.path.exists():
            info.path.unlink()
            logger.debug(f"Deleted memmap file: {info.path}")

        logger.info(f"Released buffer '{name}'")

    def release_all(self, delete_files: bool = False) -> None:
        """
        Release all buffers.

        Args:
            delete_files: Also delete backing files
        """
        names = list(self._buffers.keys())
        for name in names:
            self.release(name, delete_file=delete_files)

    def __contains__(self, name: str) -> bool:
        """Check if buffer exists."""
        return name in self._buffers

    def __getitem__(self, name: str) -> NDArray:
        """Get buffer by name (dict-style access)."""
        return self.get(name)

    @property
    def total_size_bytes(self) -> int:
        """Total size of all buffers in bytes."""
        return self._total_size

    @property
    def total_size_gb(self) -> float:
        """Total size of all buffers in GB."""
        return self._total_size / (1024**3)

    @property
    def buffer_names(self) -> list[str]:
        """List of buffer names."""
        return list(self._buffers.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about managed buffers."""
        return {
            "n_buffers": len(self._buffers),
            "total_size_bytes": self._total_size,
            "total_size_gb": self.total_size_gb,
            "buffers": {
                name: {
                    "shape": info.shape,
                    "dtype": str(info.dtype),
                    "size_gb": info.size_gb,
                }
                for name, (_, info) in self._buffers.items()
            },
        }

    def __enter__(self) -> "MemmapManager":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - release all buffers."""
        self.release_all(delete_files=True)


@dataclass
class TraceBuffer:
    """
    Buffer for loading and caching trace data.

    Provides a reusable buffer for batch trace loading.
    """

    max_traces: int
    n_samples: int
    dtype: np.dtype = field(default=np.float32)

    _data: NDArray | None = field(default=None, init=False, repr=False)
    _geometry: NDArray | None = field(default=None, init=False, repr=False)
    _current_size: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Allocate buffers."""
        self._data = np.empty((self.max_traces, self.n_samples), dtype=self.dtype)

        # Geometry buffer: (sx, sy, rx, ry, offset, midx, midy)
        self._geometry = np.empty((self.max_traces, 7), dtype=np.float64)

    @property
    def data(self) -> NDArray:
        """Get trace data (valid portion only)."""
        assert self._data is not None
        return self._data[: self._current_size]

    @property
    def geometry(self) -> NDArray:
        """Get geometry data (valid portion only)."""
        assert self._geometry is not None
        return self._geometry[: self._current_size]

    @property
    def n_traces(self) -> int:
        """Current number of traces in buffer."""
        return self._current_size

    def load(
        self,
        trace_data: NDArray,
        source_x: NDArray,
        source_y: NDArray,
        receiver_x: NDArray,
        receiver_y: NDArray,
        offset: NDArray,
        midpoint_x: NDArray,
        midpoint_y: NDArray,
    ) -> None:
        """
        Load traces and geometry into buffer.

        Args:
            trace_data: Trace amplitudes (n_traces, n_samples)
            source_x: Source X coordinates
            source_y: Source Y coordinates
            receiver_x: Receiver X coordinates
            receiver_y: Receiver Y coordinates
            offset: Offset values
            midpoint_x: Midpoint X coordinates
            midpoint_y: Midpoint Y coordinates
        """
        n = len(trace_data)
        if n > self.max_traces:
            raise ValueError(f"Too many traces: {n} > {self.max_traces}")

        assert self._data is not None
        assert self._geometry is not None

        self._data[:n] = trace_data
        self._geometry[:n, 0] = source_x
        self._geometry[:n, 1] = source_y
        self._geometry[:n, 2] = receiver_x
        self._geometry[:n, 3] = receiver_y
        self._geometry[:n, 4] = offset
        self._geometry[:n, 5] = midpoint_x
        self._geometry[:n, 6] = midpoint_y

        self._current_size = n

    def clear(self) -> None:
        """Clear buffer (reset size, no memory deallocation)."""
        self._current_size = 0

    @property
    def size_bytes(self) -> int:
        """Buffer size in bytes."""
        assert self._data is not None
        assert self._geometry is not None
        return self._data.nbytes + self._geometry.nbytes


@dataclass
class OutputTileBuffer:
    """
    Buffer for output tile accumulation.

    Holds image and fold data for a single tile.
    """

    nx: int
    ny: int
    nt: int

    # Use float64 for accumulation to avoid precision loss
    _image: NDArray | None = field(default=None, init=False, repr=False)
    _fold: NDArray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Allocate buffers."""
        self._image = np.zeros((self.nx, self.ny, self.nt), dtype=np.float64)
        self._fold = np.zeros((self.nx, self.ny), dtype=np.int32)

    @property
    def image(self) -> NDArray:
        """Get image accumulator."""
        assert self._image is not None
        return self._image

    @property
    def fold(self) -> NDArray:
        """Get fold counter."""
        assert self._fold is not None
        return self._fold

    @property
    def shape(self) -> tuple[int, int, int]:
        """Tile shape."""
        return (self.nx, self.ny, self.nt)

    def reset(self) -> None:
        """Reset buffers to zero."""
        assert self._image is not None
        assert self._fold is not None
        self._image[:] = 0.0
        self._fold[:] = 0

    def normalize(self) -> NDArray[np.float32]:
        """
        Normalize image by fold and return as float32.

        Returns:
            Normalized image (float32)
        """
        assert self._image is not None
        assert self._fold is not None

        # Avoid division by zero
        fold_3d = self._fold[:, :, np.newaxis]
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = np.where(fold_3d > 0, self._image / fold_3d, 0.0)

        return normalized.astype(np.float32)

    @property
    def size_bytes(self) -> int:
        """Buffer size in bytes."""
        assert self._image is not None
        assert self._fold is not None
        return self._image.nbytes + self._fold.nbytes
