"""
Spatial indexing for PSTM.

Provides efficient spatial queries for trace selection based on aperture.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from pstm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SpatialIndexInfo:
    """Information about the spatial index."""

    n_points: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    index_type: str


class SpatialIndex:
    """
    Spatial index for efficient aperture-based trace queries.

    Uses scipy's cKDTree for fast radius and rectangle queries.
    Indexes traces by their midpoint coordinates.
    """

    def __init__(self):
        """Initialize empty spatial index."""
        self._tree: cKDTree | None = None
        self._trace_indices: NDArray[np.int64] | None = None
        self._coordinates: NDArray[np.float64] | None = None
        self._info: SpatialIndexInfo | None = None

    @classmethod
    def build(
        cls,
        trace_indices: NDArray[np.int64],
        x_coords: NDArray[np.float64],
        y_coords: NDArray[np.float64],
        leafsize: int = 32,
    ) -> "SpatialIndex":
        """
        Build a spatial index from coordinates.

        Args:
            trace_indices: Array of trace indices
            x_coords: X coordinates (e.g., midpoint X)
            y_coords: Y coordinates (e.g., midpoint Y)
            leafsize: KDTree leaf size (affects query performance)

        Returns:
            Built SpatialIndex instance
        """
        instance = cls()

        n_points = len(trace_indices)
        if n_points == 0:
            raise ValueError("Cannot build index from empty coordinates")

        if len(x_coords) != n_points or len(y_coords) != n_points:
            raise ValueError("Coordinate arrays must have same length as trace_indices")

        logger.info(f"Building spatial index for {n_points:,} points...")

        # Store data
        instance._trace_indices = np.asarray(trace_indices, dtype=np.int64)
        instance._coordinates = np.column_stack([
            np.asarray(x_coords, dtype=np.float64),
            np.asarray(y_coords, dtype=np.float64),
        ])

        # Build KDTree
        instance._tree = cKDTree(instance._coordinates, leafsize=leafsize)

        # Build info
        instance._info = SpatialIndexInfo(
            n_points=n_points,
            x_range=(float(x_coords.min()), float(x_coords.max())),
            y_range=(float(y_coords.min()), float(y_coords.max())),
            index_type="cKDTree",
        )

        logger.info(
            f"Built spatial index: {n_points:,} points, "
            f"X: [{instance._info.x_range[0]:.1f}, {instance._info.x_range[1]:.1f}], "
            f"Y: [{instance._info.y_range[0]:.1f}, {instance._info.y_range[1]:.1f}]"
        )

        return instance

    @property
    def info(self) -> SpatialIndexInfo | None:
        """Get index information."""
        return self._info

    @property
    def n_points(self) -> int:
        """Number of indexed points."""
        return self._info.n_points if self._info else 0

    def query_radius(
        self,
        x: float,
        y: float,
        radius: float,
    ) -> NDArray[np.int64]:
        """
        Query all traces within radius of a point.

        Args:
            x: Query point X coordinate
            y: Query point Y coordinate
            radius: Search radius

        Returns:
            Array of trace indices within radius
        """
        if self._tree is None or self._trace_indices is None:
            raise RuntimeError("Spatial index not built")

        point = np.array([x, y])
        idx = self._tree.query_ball_point(point, radius)

        return self._trace_indices[idx]

    def query_radius_batch(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        radius: float,
    ) -> list[NDArray[np.int64]]:
        """
        Query all traces within radius for multiple points.

        Args:
            x: Query points X coordinates
            y: Query points Y coordinates
            radius: Search radius (same for all points)

        Returns:
            List of arrays of trace indices (one per query point)
        """
        if self._tree is None or self._trace_indices is None:
            raise RuntimeError("Spatial index not built")

        points = np.column_stack([x, y])
        idx_lists = self._tree.query_ball_point(points, radius)

        return [self._trace_indices[idx] for idx in idx_lists]

    def query_rectangle(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        padding: float = 0.0,
    ) -> NDArray[np.int64]:
        """
        Query all traces within a rectangle.

        Uses a bounding circle query followed by exact filtering.

        Args:
            x_min: Rectangle minimum X
            x_max: Rectangle maximum X
            y_min: Rectangle minimum Y
            y_max: Rectangle maximum Y
            padding: Extra padding around rectangle

        Returns:
            Array of trace indices within rectangle
        """
        if self._tree is None or self._trace_indices is None or self._coordinates is None:
            raise RuntimeError("Spatial index not built")

        # Apply padding
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding

        # Query using bounding circle
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        radius = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) / 2

        point = np.array([center_x, center_y])
        candidate_idx = self._tree.query_ball_point(point, radius)

        if len(candidate_idx) == 0:
            return np.array([], dtype=np.int64)

        # Exact filtering
        coords = self._coordinates[candidate_idx]
        mask = (
            (coords[:, 0] >= x_min)
            & (coords[:, 0] <= x_max)
            & (coords[:, 1] >= y_min)
            & (coords[:, 1] <= y_max)
        )

        return self._trace_indices[candidate_idx][mask]

    def query_nearest(
        self,
        x: float,
        y: float,
        k: int = 1,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Find k nearest traces to a point.

        Args:
            x: Query point X coordinate
            y: Query point Y coordinate
            k: Number of nearest neighbors

        Returns:
            Tuple of (trace_indices, distances)
        """
        if self._tree is None or self._trace_indices is None:
            raise RuntimeError("Spatial index not built")

        point = np.array([x, y])
        distances, indices = self._tree.query(point, k=k)

        # Handle single result case
        if k == 1:
            indices = np.array([indices])
            distances = np.array([distances])

        return self._trace_indices[indices], distances

    def get_coordinates(self, trace_indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Get coordinates for specific trace indices.

        Args:
            trace_indices: Trace indices to look up

        Returns:
            Array of shape (n, 2) with [x, y] coordinates
        """
        if self._trace_indices is None or self._coordinates is None:
            raise RuntimeError("Spatial index not built")

        # Create index lookup
        idx_map = {idx: i for i, idx in enumerate(self._trace_indices)}
        positions = [idx_map.get(idx) for idx in trace_indices]

        valid = [p for p in positions if p is not None]
        if len(valid) != len(trace_indices):
            raise ValueError("Some trace indices not found in index")

        return self._coordinates[valid]

    def save(self, path: Path | str) -> None:
        """
        Save spatial index to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "trace_indices": self._trace_indices,
            "coordinates": self._coordinates,
            "info": self._info,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved spatial index to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "SpatialIndex":
        """
        Load spatial index from file.

        Args:
            path: Input file path

        Returns:
            Loaded SpatialIndex instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Spatial index not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance._trace_indices = data["trace_indices"]
        instance._coordinates = data["coordinates"]
        instance._info = data["info"]

        # Rebuild tree
        instance._tree = cKDTree(instance._coordinates)

        logger.info(f"Loaded spatial index from {path}: {instance.n_points:,} points")

        return instance


class TileQueryResult:
    """Result of querying traces for a tile."""

    def __init__(
        self,
        tile_x_range: tuple[float, float],
        tile_y_range: tuple[float, float],
        aperture_radius: float,
        trace_indices: NDArray[np.int64],
    ):
        self.tile_x_range = tile_x_range
        self.tile_y_range = tile_y_range
        self.aperture_radius = aperture_radius
        self.trace_indices = trace_indices

    @property
    def n_traces(self) -> int:
        """Number of traces in result."""
        return len(self.trace_indices)

    @property
    def query_bounds(self) -> tuple[float, float, float, float]:
        """Get the query bounds (tile + aperture)."""
        return (
            self.tile_x_range[0] - self.aperture_radius,
            self.tile_x_range[1] + self.aperture_radius,
            self.tile_y_range[0] - self.aperture_radius,
            self.tile_y_range[1] + self.aperture_radius,
        )


def query_traces_for_tile(
    spatial_index: SpatialIndex,
    tile_x_min: float,
    tile_x_max: float,
    tile_y_min: float,
    tile_y_max: float,
    aperture_radius: float,
) -> TileQueryResult:
    """
    Query all traces that could contribute to a tile.

    The query includes all traces within aperture_radius of the tile boundary.

    Args:
        spatial_index: Built spatial index
        tile_x_min: Tile minimum X
        tile_x_max: Tile maximum X
        tile_y_min: Tile minimum Y
        tile_y_max: Tile maximum Y
        aperture_radius: Migration aperture radius

    Returns:
        TileQueryResult with trace indices
    """
    trace_indices = spatial_index.query_rectangle(
        tile_x_min,
        tile_x_max,
        tile_y_min,
        tile_y_max,
        padding=aperture_radius,
    )

    return TileQueryResult(
        tile_x_range=(tile_x_min, tile_x_max),
        tile_y_range=(tile_y_min, tile_y_max),
        aperture_radius=aperture_radius,
        trace_indices=trace_indices,
    )
