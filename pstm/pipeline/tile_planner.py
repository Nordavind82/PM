"""
Tile planner for PSTM.

Handles output grid tiling for memory-efficient processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from pstm.config.models import OutputGridConfig, TilingConfig
from pstm.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Shared tile calculation functions (used by UI preflight and TilePlanner)
# =============================================================================


def calculate_auto_tile_size(
    nt: int,
    nx: int,
    ny: int,
    max_memory_gb: float = 8.0,
    prefer_throughput: bool = False,
) -> tuple[int, int]:
    """
    Calculate optimal tile size based on memory budget and time depth.

    This is the single source of truth for auto-tile calculation,
    used by both TilePlanner and UI preflight validation.

    Args:
        nt: Number of time samples in output grid
        nx: Number of X bins in output grid
        ny: Number of Y bins in output grid
        max_memory_gb: Maximum memory budget in GB
        prefer_throughput: If True, use larger tiles for better performance
                          even if UI updates are less frequent

    Returns:
        (tile_nx, tile_ny) dimensions
    """
    # Memory per output pillar (float64 accumulator)
    bytes_per_pillar = nt * 8  # float64

    # Target: use ~1/4 of memory budget for output tile
    target_memory_bytes = max_memory_gb * 1024**3 / 4

    # Maximum pillars we can fit
    max_pillars = int(target_memory_bytes / bytes_per_pillar)

    # Make roughly square tiles
    tile_size = int(np.sqrt(max_pillars))

    if prefer_throughput:
        # Throughput mode: use much larger tiles to reduce overhead
        # Each trace is loaded fewer times with larger tiles
        # Max tile is limited only by memory, not UI responsiveness
        max_tile = 256  # Very large tiles
        logger.debug("Tile planner: prefer_throughput mode, using larger tiles")
    else:
        # Adjust max tile size based on time depth for reasonable progress updates
        # Deep data (high nt) takes longer per tile, so use smaller tiles
        # Target: ~60-90 seconds per tile for reasonable UI feedback
        if nt > 1000:
            # Very deep data: use smaller tiles for ~60-90s updates
            max_tile = 32
        elif nt > 500:
            # Medium depth: ~45-60s updates
            max_tile = 48
        else:
            # Shallow data: ~10-15s updates
            max_tile = 64

    tile_size = max(32, min(tile_size, max_tile))

    # Don't make tiles larger than output
    tile_nx = min(tile_size, nx)
    tile_ny = min(tile_size, ny)

    return tile_nx, tile_ny


def calculate_tile_count(
    nx: int,
    ny: int,
    nt: int,
    auto_tile_size: bool,
    tile_nx: int,
    tile_ny: int,
    max_memory_gb: float = 8.0,
) -> tuple[int, int, int]:
    """
    Calculate number of tiles and actual tile dimensions.

    This is the single source of truth for tile count calculation,
    used by both TilePlanner and UI preflight validation.

    Args:
        nx: Number of X bins in output grid
        ny: Number of Y bins in output grid
        nt: Number of time samples in output grid
        auto_tile_size: Whether to auto-determine tile size
        tile_nx: Manual tile NX (used if auto_tile_size=False)
        tile_ny: Manual tile NY (used if auto_tile_size=False)
        max_memory_gb: Max memory budget in GB

    Returns:
        (n_tiles, actual_tile_nx, actual_tile_ny)
    """
    if auto_tile_size:
        actual_tile_nx, actual_tile_ny = calculate_auto_tile_size(nt, nx, ny, max_memory_gb)
    else:
        actual_tile_nx = min(tile_nx, nx)
        actual_tile_ny = min(tile_ny, ny)

    n_tiles_x = (nx + actual_tile_nx - 1) // actual_tile_nx
    n_tiles_y = (ny + actual_tile_ny - 1) // actual_tile_ny
    n_tiles = n_tiles_x * n_tiles_y

    return n_tiles, actual_tile_nx, actual_tile_ny


@dataclass
class TileSpec:
    """Specification for a single output tile."""

    # Tile indices
    tile_id: int
    ix: int  # X tile index
    iy: int  # Y tile index

    # Grid indices (into full output)
    x_start: int
    x_end: int  # Exclusive
    y_start: int
    y_end: int  # Exclusive

    # Coordinates
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    # Derived
    @property
    def nx(self) -> int:
        """Number of X points in tile."""
        return self.x_end - self.x_start

    @property
    def ny(self) -> int:
        """Number of Y points in tile."""
        return self.y_end - self.y_start

    @property
    def center_x(self) -> float:
        """Tile center X coordinate."""
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self) -> float:
        """Tile center Y coordinate."""
        return (self.y_min + self.y_max) / 2


@dataclass
class TilePlan:
    """Complete tile plan for migration."""

    # Grid info
    output_grid: OutputGridConfig
    tile_nx: int  # Points per tile in X
    tile_ny: int  # Points per tile in Y
    n_tiles_x: int  # Number of tiles in X
    n_tiles_y: int  # Number of tiles in Y

    # Tiles
    tiles: list[TileSpec] = field(default_factory=list)

    # Estimated work
    estimated_traces_per_tile: NDArray[np.int64] | None = None

    @property
    def n_tiles(self) -> int:
        """Total number of tiles."""
        return self.n_tiles_x * self.n_tiles_y

    @property
    def total_output_samples(self) -> int:
        """Total output samples."""
        return self.output_grid.nx * self.output_grid.ny * self.output_grid.nt


class TilePlanner:
    """
    Plans output grid tiling for memory-efficient processing.

    Determines optimal tile sizes based on:
    - Available memory
    - Aperture size (traces needed per tile)
    - Cache efficiency (tile ordering)
    """

    def __init__(
        self,
        output_grid: OutputGridConfig,
        tiling_config: TilingConfig,
        max_memory_gb: float | None = None,
        aperture_radius: float | None = None,
        prefer_throughput: bool = False,
    ):
        """
        Initialize the tile planner.

        Args:
            output_grid: Output grid configuration
            tiling_config: Tiling configuration
            max_memory_gb: Maximum memory budget for tiling (default from settings)
            aperture_radius: Aperture radius (default from settings)
            prefer_throughput: If True, use larger tiles for better performance
                              at the expense of less frequent UI updates
        """
        from pstm.settings import get_settings
        s = get_settings()

        self.output_grid = output_grid
        self.tiling_config = tiling_config
        self.max_memory_gb = max_memory_gb if max_memory_gb is not None else s.tiling.max_memory_gb
        self.aperture_radius = aperture_radius if aperture_radius is not None else s.aperture.max_aperture_m
        self.prefer_throughput = prefer_throughput

    def plan(self) -> TilePlan:
        """
        Create a tile plan.

        Returns:
            TilePlan with all tiles
        """
        # Determine tile dimensions
        if self.tiling_config.auto_tile_size:
            tile_nx, tile_ny = self._auto_tile_size()
        else:
            # Validate manual tile sizes - fail fast if not set
            if not self.tiling_config.tile_nx or self.tiling_config.tile_nx <= 0:
                raise ValueError(
                    f"Manual tile_nx must be positive, got {self.tiling_config.tile_nx}. "
                    f"Either set auto_tile_size=True or provide valid tile_nx."
                )
            if not self.tiling_config.tile_ny or self.tiling_config.tile_ny <= 0:
                raise ValueError(
                    f"Manual tile_ny must be positive, got {self.tiling_config.tile_ny}. "
                    f"Either set auto_tile_size=True or provide valid tile_ny."
                )
            tile_nx = min(self.tiling_config.tile_nx, self.output_grid.nx)
            tile_ny = min(self.tiling_config.tile_ny, self.output_grid.ny)

        # Calculate number of tiles
        n_tiles_x = (self.output_grid.nx + tile_nx - 1) // tile_nx
        n_tiles_y = (self.output_grid.ny + tile_ny - 1) // tile_ny

        logger.info(
            f"Tile plan: {tile_nx}×{tile_ny} points per tile, "
            f"{n_tiles_x}×{n_tiles_y} = {n_tiles_x * n_tiles_y} tiles"
        )

        # Create tile specifications
        tiles = self._generate_tiles(tile_nx, tile_ny, n_tiles_x, n_tiles_y)

        # Order tiles
        tiles = self._order_tiles(tiles)

        return TilePlan(
            output_grid=self.output_grid,
            tile_nx=tile_nx,
            tile_ny=tile_ny,
            n_tiles_x=n_tiles_x,
            n_tiles_y=n_tiles_y,
            tiles=tiles,
        )

    def _auto_tile_size(self) -> tuple[int, int]:
        """
        Automatically determine tile size based on memory budget and time depth.

        Returns:
            (tile_nx, tile_ny) dimensions
        """
        tile_nx, tile_ny = calculate_auto_tile_size(
            nt=self.output_grid.nt,
            nx=self.output_grid.nx,
            ny=self.output_grid.ny,
            max_memory_gb=self.max_memory_gb,
            prefer_throughput=self.prefer_throughput,
        )

        bytes_per_pillar = self.output_grid.nt * 8  # float64
        mode = "throughput" if self.prefer_throughput else "responsive"
        logger.debug(
            f"Auto tile size: {tile_nx}×{tile_ny} "
            f"({tile_nx * tile_ny * bytes_per_pillar / 1024**2:.1f} MB per tile) "
            f"[nt={self.output_grid.nt}, mode={mode}]"
        )

        return tile_nx, tile_ny

    def _generate_tiles(
        self,
        tile_nx: int,
        tile_ny: int,
        n_tiles_x: int,
        n_tiles_y: int,
    ) -> list[TileSpec]:
        """Generate tile specifications."""
        tiles = []
        tile_id = 0

        # Get coordinate arrays (handles both bounding-box and corner-point grids)
        coords = self.output_grid.get_output_coordinates()

        # For rotated grids, we need to use the full 2D coordinate meshgrids
        # to get correct bounding boxes for spatial queries
        X_grid = coords.get('X')  # 2D meshgrid of X coordinates
        Y_grid = coords.get('Y')  # 2D meshgrid of Y coordinates

        for iy in range(n_tiles_y):
            for ix in range(n_tiles_x):
                # Grid indices
                x_start = ix * tile_nx
                x_end = min((ix + 1) * tile_nx, self.output_grid.nx)
                y_start = iy * tile_ny
                y_end = min((iy + 1) * tile_ny, self.output_grid.ny)

                # Get actual coordinate bounds from 2D grids
                # This correctly handles rotated grids where X,Y vary in both IL and XL
                if X_grid is not None and Y_grid is not None:
                    tile_X = X_grid[x_start:x_end, y_start:y_end]
                    tile_Y = Y_grid[x_start:x_end, y_start:y_end]
                    x_min = float(tile_X.min())
                    x_max = float(tile_X.max())
                    y_min = float(tile_Y.min())
                    y_max = float(tile_Y.max())
                else:
                    # Fallback for axis-aligned grids (1D coordinates)
                    x_coords = coords['x']
                    y_coords = coords['y']
                    x_min = x_coords[x_start]
                    x_max = x_coords[x_end - 1]
                    y_min = y_coords[y_start]
                    y_max = y_coords[y_end - 1]

                tiles.append(TileSpec(
                    tile_id=tile_id,
                    ix=ix,
                    iy=iy,
                    x_start=x_start,
                    x_end=x_end,
                    y_start=y_start,
                    y_end=y_end,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                ))
                tile_id += 1

        return tiles

    def _order_tiles(self, tiles: list[TileSpec]) -> list[TileSpec]:
        """
        Order tiles for cache-efficient processing.

        Args:
            tiles: List of tiles in row-major order

        Returns:
            Reordered tile list
        """
        ordering = self.tiling_config.ordering

        if ordering == "row_major":
            # Already in row-major order
            return tiles

        elif ordering == "column_major":
            # Sort by Y then X
            return sorted(tiles, key=lambda t: (t.ix, t.iy))

        elif ordering == "snake":
            # Alternate direction each row for cache locality
            result = []
            n_tiles_x = max(t.ix for t in tiles) + 1

            # Group by Y
            rows: dict[int, list[TileSpec]] = {}
            for tile in tiles:
                if tile.iy not in rows:
                    rows[tile.iy] = []
                rows[tile.iy].append(tile)

            # Sort rows by X, reverse every other row
            for iy in sorted(rows.keys()):
                row = sorted(rows[iy], key=lambda t: t.ix)
                if iy % 2 == 1:
                    row = row[::-1]
                result.extend(row)

            # Renumber tile IDs
            for i, tile in enumerate(result):
                tile.tile_id = i

            return result

        elif ordering == "hilbert":
            # Hilbert curve ordering (better locality than snake)
            # Simplified: use snake for now
            logger.warning("Hilbert ordering not implemented, using snake")
            return self._order_tiles_snake(tiles)

        else:
            logger.warning(f"Unknown ordering '{ordering}', using row_major")
            return tiles

    def _order_tiles_snake(self, tiles: list[TileSpec]) -> list[TileSpec]:
        """Snake ordering helper."""
        result = []
        rows: dict[int, list[TileSpec]] = {}

        for tile in tiles:
            if tile.iy not in rows:
                rows[tile.iy] = []
            rows[tile.iy].append(tile)

        for iy in sorted(rows.keys()):
            row = sorted(rows[iy], key=lambda t: t.ix)
            if iy % 2 == 1:
                row = row[::-1]
            result.extend(row)

        for i, tile in enumerate(result):
            tile.tile_id = i

        return result


def estimate_traces_per_tile(
    plan: TilePlan,
    midpoint_x: NDArray[np.float64],
    midpoint_y: NDArray[np.float64],
    aperture_radius: float,
) -> NDArray[np.int64]:
    """
    Estimate number of traces contributing to each tile.

    Args:
        plan: Tile plan
        midpoint_x: Trace midpoint X coordinates
        midpoint_y: Trace midpoint Y coordinates
        aperture_radius: Aperture radius

    Returns:
        Array of trace counts per tile
    """
    counts = np.zeros(plan.n_tiles, dtype=np.int64)

    for tile in plan.tiles:
        # Check which midpoints are within aperture of tile
        in_x = (
            (midpoint_x >= tile.x_min - aperture_radius) &
            (midpoint_x <= tile.x_max + aperture_radius)
        )
        in_y = (
            (midpoint_y >= tile.y_min - aperture_radius) &
            (midpoint_y <= tile.y_max + aperture_radius)
        )
        counts[tile.tile_id] = np.sum(in_x & in_y)

    return counts


def iter_tiles(plan: TilePlan, completed: set[int] | None = None) -> Iterator[TileSpec]:
    """
    Iterate over tiles, optionally skipping completed ones.

    Args:
        plan: Tile plan
        completed: Set of completed tile IDs to skip

    Yields:
        TileSpec for each remaining tile
    """
    for tile in plan.tiles:
        if completed is None or tile.tile_id not in completed:
            yield tile
