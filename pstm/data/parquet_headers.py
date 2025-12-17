"""
Parquet header manager for PSTM using Polars.

Provides efficient access to seismic trace headers stored in Parquet format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from pstm.config.models import ColumnMapping
from pstm.utils.logging import get_logger
from pstm.utils.units import apply_seg_y_scalar, compute_azimuth, compute_offset, offset_to_midpoint

logger = get_logger(__name__)


@dataclass
class HeaderStatistics:
    """Statistics about trace headers."""

    n_traces: int
    n_shots: int

    # Coordinate ranges
    source_x_range: tuple[float, float]
    source_y_range: tuple[float, float]
    receiver_x_range: tuple[float, float]
    receiver_y_range: tuple[float, float]
    midpoint_x_range: tuple[float, float]
    midpoint_y_range: tuple[float, float]

    # Offset and azimuth
    offset_range: tuple[float, float]
    offset_mean: float
    azimuth_range: tuple[float, float] | None

    def get_survey_extent(self) -> dict[str, tuple[float, float]]:
        """Get survey extent as dictionary."""
        return {
            "x": (
                min(self.source_x_range[0], self.receiver_x_range[0]),
                max(self.source_x_range[1], self.receiver_x_range[1]),
            ),
            "y": (
                min(self.source_y_range[0], self.receiver_y_range[0]),
                max(self.source_y_range[1], self.receiver_y_range[1]),
            ),
        }


@dataclass
class TraceGeometry:
    """Geometry data for a set of traces."""

    trace_indices: NDArray[np.int64]
    source_x: NDArray[np.float64]
    source_y: NDArray[np.float64]
    receiver_x: NDArray[np.float64]
    receiver_y: NDArray[np.float64]
    offset: NDArray[np.float64]
    midpoint_x: NDArray[np.float64]
    midpoint_y: NDArray[np.float64]

    @property
    def n_traces(self) -> int:
        """Number of traces."""
        return len(self.trace_indices)

    def to_structured_array(self) -> NDArray:
        """Convert to structured NumPy array for kernel consumption."""
        dtype = np.dtype([
            ("trace_idx", np.int64),
            ("src_x", np.float64),
            ("src_y", np.float64),
            ("rec_x", np.float64),
            ("rec_y", np.float64),
            ("offset", np.float64),
            ("mid_x", np.float64),
            ("mid_y", np.float64),
        ])
        arr = np.empty(self.n_traces, dtype=dtype)
        arr["trace_idx"] = self.trace_indices
        arr["src_x"] = self.source_x
        arr["src_y"] = self.source_y
        arr["rec_x"] = self.receiver_x
        arr["rec_y"] = self.receiver_y
        arr["offset"] = self.offset
        arr["mid_x"] = self.midpoint_x
        arr["mid_y"] = self.midpoint_y
        return arr


class ParquetHeaderManager:
    """
    Manager for seismic trace headers stored in Parquet format.

    Uses Polars for efficient lazy evaluation and predicate pushdown.

    Expected columns (configurable via ColumnMapping):
    - trace_idx: Global trace index (matches Zarr array index)
    - Source X/Y coordinates
    - Receiver X/Y coordinates
    - Optional: CDP X/Y, offset, azimuth, shot ID, etc.
    """

    def __init__(
        self,
        path: Path | str,
        column_mapping: ColumnMapping | None = None,
        apply_scalar: bool = False,
        scalar_column: str | None = None,
    ):
        """
        Initialize the header manager.

        Args:
            path: Path to Parquet file
            column_mapping: Mapping of column names
            apply_scalar: Apply SEG-Y coordinate scalar
            scalar_column: Column containing scalar values
        """
        self.path = Path(path)
        self.columns = column_mapping or ColumnMapping()
        self.apply_scalar = apply_scalar
        self.scalar_column = scalar_column or self.columns.coord_scalar

        self._lazy_frame: pl.LazyFrame | None = None
        self._schema: dict[str, pl.DataType] | None = None
        self._statistics: HeaderStatistics | None = None
        self._n_traces: int | None = None

    def open(self) -> "ParquetHeaderManager":
        """Open the Parquet file for lazy access."""
        if self._lazy_frame is not None:
            return self

        if not self.path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.path}")

        logger.debug(f"Opening Parquet headers: {self.path}")
        self._lazy_frame = pl.scan_parquet(self.path)
        self._schema = self._lazy_frame.collect_schema()

        # Validate required columns
        self._validate_columns()

        logger.info(f"Opened headers: {len(self._schema)} columns")

        return self

    def close(self) -> None:
        """Close the Parquet file."""
        self._lazy_frame = None
        self._schema = None
        self._statistics = None

    def __enter__(self) -> "ParquetHeaderManager":
        """Context manager entry."""
        return self.open()

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()

    def _validate_columns(self) -> None:
        """Validate that required columns exist."""
        assert self._schema is not None

        required = [
            self.columns.source_x,
            self.columns.source_y,
            self.columns.receiver_x,
            self.columns.receiver_y,
            self.columns.trace_index,
        ]

        missing = [col for col in required if col not in self._schema]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @property
    def schema(self) -> dict[str, pl.DataType]:
        """Get Parquet schema."""
        if self._schema is None:
            self.open()
        assert self._schema is not None
        return self._schema

    @property
    def n_traces(self) -> int:
        """Get number of traces."""
        if self._n_traces is None:
            if self._lazy_frame is None:
                self.open()
            assert self._lazy_frame is not None
            self._n_traces = self._lazy_frame.select(pl.len()).collect().item()
        return self._n_traces

    def _apply_scalar_if_needed(
        self,
        df: pl.DataFrame,
        coord_arrays: dict[str, NDArray[np.float64]] | None = None,
    ) -> pl.DataFrame | dict[str, NDArray[np.float64]]:
        """
        Apply coordinate scalar if configured.

        Args:
            df: DataFrame containing data (and optionally scalar column)
            coord_arrays: If provided, apply scalar to these arrays instead of df columns

        Returns:
            Modified DataFrame or dict of scaled arrays
        """
        if not self.apply_scalar:
            return coord_arrays if coord_arrays is not None else df

        # Determine scalar value - either from df column or from stored value
        scalar_value = None

        if self.scalar_column and self.scalar_column in df.columns:
            # Get first scalar value (typically constant for whole dataset)
            scalar_value = int(df[self.scalar_column][0])
            logger.debug(f"Using scalar from column {self.scalar_column}: {scalar_value}")
        elif hasattr(self, '_cached_scalar') and self._cached_scalar is not None:
            scalar_value = self._cached_scalar
            logger.debug(f"Using cached scalar: {scalar_value}")

        if scalar_value is None or scalar_value == 0 or scalar_value == 1:
            return coord_arrays if coord_arrays is not None else df

        # Cache the scalar for future use
        self._cached_scalar = scalar_value

        # If we have coord_arrays, apply directly to them
        if coord_arrays is not None:
            for key in coord_arrays:
                if scalar_value > 0:
                    coord_arrays[key] = coord_arrays[key] * scalar_value
                else:
                    coord_arrays[key] = coord_arrays[key] / abs(scalar_value)
            return coord_arrays

        # Apply to DataFrame columns (vectorized)
        coord_cols = [
            self.columns.source_x,
            self.columns.source_y,
            self.columns.receiver_x,
            self.columns.receiver_y,
        ]

        for col in coord_cols:
            if col in df.columns:
                if scalar_value > 0:
                    df = df.with_columns(
                        (pl.col(col).cast(pl.Float64) * scalar_value).alias(col)
                    )
                else:
                    df = df.with_columns(
                        (pl.col(col).cast(pl.Float64) / abs(scalar_value)).alias(col)
                    )

        return df

    def get_geometry_for_indices(
        self,
        indices: NDArray[np.int64] | list[int],
    ) -> TraceGeometry:
        """
        Get geometry for specific trace indices.

        Args:
            indices: Trace indices to retrieve

        Returns:
            TraceGeometry with coordinate arrays
        """
        if self._lazy_frame is None:
            self.open()
        assert self._lazy_frame is not None

        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return TraceGeometry(
                trace_indices=np.array([], dtype=np.int64),
                source_x=np.array([], dtype=np.float64),
                source_y=np.array([], dtype=np.float64),
                receiver_x=np.array([], dtype=np.float64),
                receiver_y=np.array([], dtype=np.float64),
                offset=np.array([], dtype=np.float64),
                midpoint_x=np.array([], dtype=np.float64),
                midpoint_y=np.array([], dtype=np.float64),
            )

        # Build filter expression
        idx_col = self.columns.trace_index
        filter_expr = pl.col(idx_col).is_in(indices.tolist())

        # Select required columns
        select_cols = [
            idx_col,
            self.columns.source_x,
            self.columns.source_y,
            self.columns.receiver_x,
            self.columns.receiver_y,
        ]

        # Include offset if available
        if self.columns.offset and self.columns.offset in self.schema:
            select_cols.append(self.columns.offset)

        # Include scalar column if we need to apply it
        if self.apply_scalar and self.scalar_column and self.scalar_column in self.schema:
            select_cols.append(self.scalar_column)

        # Execute query
        df = (
            self._lazy_frame
            .filter(filter_expr)
            .select(select_cols)
            .collect()
        )

        # Apply scalar if needed
        df = self._apply_scalar_if_needed(df)

        # Extract arrays
        trace_indices = df[idx_col].to_numpy().astype(np.int64)
        source_x = df[self.columns.source_x].to_numpy().astype(np.float64)
        source_y = df[self.columns.source_y].to_numpy().astype(np.float64)
        receiver_x = df[self.columns.receiver_x].to_numpy().astype(np.float64)
        receiver_y = df[self.columns.receiver_y].to_numpy().astype(np.float64)

        # Compute or extract offset
        if self.columns.offset and self.columns.offset in df.columns:
            offset = df[self.columns.offset].to_numpy().astype(np.float64)
        else:
            offset = compute_offset(source_x, source_y, receiver_x, receiver_y)

        # Compute midpoints
        midpoint_x, midpoint_y = offset_to_midpoint(source_x, source_y, receiver_x, receiver_y)

        # Reorder to match input indices order
        # (Polars may return in different order)
        if len(trace_indices) > 0:
            # Create mapping from trace_idx to position
            idx_to_pos = {idx: i for i, idx in enumerate(trace_indices)}
            order = [idx_to_pos.get(idx, 0) for idx in indices if idx in idx_to_pos]

            if len(order) == len(trace_indices):
                trace_indices = trace_indices[order]
                source_x = source_x[order]
                source_y = source_y[order]
                receiver_x = receiver_x[order]
                receiver_y = receiver_y[order]
                offset = offset[order]
                midpoint_x = midpoint_x[order]
                midpoint_y = midpoint_y[order]

        return TraceGeometry(
            trace_indices=trace_indices,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            offset=offset,
            midpoint_x=midpoint_x,
            midpoint_y=midpoint_y,
        )

    def get_all_midpoints(self) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Get midpoint coordinates for all traces.

        Returns:
            Tuple of (trace_indices, midpoint_x, midpoint_y)
        """
        if self._lazy_frame is None:
            self.open()
        assert self._lazy_frame is not None

        # Build list of columns to select
        select_cols = [
            self.columns.trace_index,
            self.columns.source_x,
            self.columns.source_y,
            self.columns.receiver_x,
            self.columns.receiver_y,
        ]

        # Include scalar column if we need to apply it
        if self.apply_scalar and self.scalar_column and self.scalar_column in self.schema:
            select_cols.append(self.scalar_column)

        df = self._lazy_frame.select(select_cols).collect()

        # Apply scalar if configured
        df = self._apply_scalar_if_needed(df)

        trace_indices = df[self.columns.trace_index].to_numpy().astype(np.int64)
        source_x = df[self.columns.source_x].to_numpy().astype(np.float64)
        source_y = df[self.columns.source_y].to_numpy().astype(np.float64)
        receiver_x = df[self.columns.receiver_x].to_numpy().astype(np.float64)
        receiver_y = df[self.columns.receiver_y].to_numpy().astype(np.float64)

        midpoint_x, midpoint_y = offset_to_midpoint(source_x, source_y, receiver_x, receiver_y)

        logger.debug(f"get_all_midpoints: apply_scalar={self.apply_scalar}, "
                    f"midpoint_x range: {midpoint_x.min():.1f} - {midpoint_x.max():.1f}")

        return trace_indices, midpoint_x, midpoint_y

    def get_offset_range(
        self,
        min_offset: float,
        max_offset: float,
    ) -> pl.LazyFrame:
        """
        Get lazy frame filtered by offset range.

        Args:
            min_offset: Minimum offset
            max_offset: Maximum offset

        Returns:
            Filtered LazyFrame (not yet collected)
        """
        if self._lazy_frame is None:
            self.open()
        assert self._lazy_frame is not None

        if self.columns.offset is None or self.columns.offset not in self.schema:
            raise ValueError("Offset column not available")

        return self._lazy_frame.filter(
            pl.col(self.columns.offset).is_between(min_offset, max_offset)
        )

    def compute_statistics(self) -> HeaderStatistics:
        """
        Compute statistics about the trace headers.

        Returns:
            HeaderStatistics object
        """
        if self._statistics is not None:
            return self._statistics

        if self._lazy_frame is None:
            self.open()
        assert self._lazy_frame is not None

        logger.info("Computing header statistics...")

        # Build aggregation expressions
        aggs = [
            pl.len().alias("n_traces"),
            pl.col(self.columns.shot_id).n_unique().alias("n_shots"),
            pl.col(self.columns.source_x).min().alias("sx_min"),
            pl.col(self.columns.source_x).max().alias("sx_max"),
            pl.col(self.columns.source_y).min().alias("sy_min"),
            pl.col(self.columns.source_y).max().alias("sy_max"),
            pl.col(self.columns.receiver_x).min().alias("rx_min"),
            pl.col(self.columns.receiver_x).max().alias("rx_max"),
            pl.col(self.columns.receiver_y).min().alias("ry_min"),
            pl.col(self.columns.receiver_y).max().alias("ry_max"),
        ]

        # Add offset stats if available
        if self.columns.offset and self.columns.offset in self.schema:
            aggs.extend([
                pl.col(self.columns.offset).min().alias("offset_min"),
                pl.col(self.columns.offset).max().alias("offset_max"),
                pl.col(self.columns.offset).mean().alias("offset_mean"),
            ])

        # Add azimuth stats if available
        if self.columns.azimuth and self.columns.azimuth in self.schema:
            aggs.extend([
                pl.col(self.columns.azimuth).min().alias("azimuth_min"),
                pl.col(self.columns.azimuth).max().alias("azimuth_max"),
            ])

        # Execute
        stats = self._lazy_frame.select(aggs).collect().row(0, named=True)

        # Compute midpoint ranges
        _, midpoint_x, midpoint_y = self.get_all_midpoints()

        # Build statistics object
        self._statistics = HeaderStatistics(
            n_traces=stats["n_traces"],
            n_shots=stats["n_shots"],
            source_x_range=(stats["sx_min"], stats["sx_max"]),
            source_y_range=(stats["sy_min"], stats["sy_max"]),
            receiver_x_range=(stats["rx_min"], stats["rx_max"]),
            receiver_y_range=(stats["ry_min"], stats["ry_max"]),
            midpoint_x_range=(float(midpoint_x.min()), float(midpoint_x.max())),
            midpoint_y_range=(float(midpoint_y.min()), float(midpoint_y.max())),
            offset_range=(
                stats.get("offset_min", 0),
                stats.get("offset_max", 0),
            ),
            offset_mean=stats.get("offset_mean", 0),
            azimuth_range=(
                (stats["azimuth_min"], stats["azimuth_max"])
                if "azimuth_min" in stats
                else None
            ),
        )

        return self._statistics

    def get_column(self, column_name: str) -> NDArray:
        """
        Get a single column as NumPy array.

        Args:
            column_name: Column name

        Returns:
            NumPy array
        """
        if self._lazy_frame is None:
            self.open()
        assert self._lazy_frame is not None

        if column_name not in self.schema:
            raise ValueError(f"Column not found: {column_name}")

        return self._lazy_frame.select(column_name).collect().to_numpy().flatten()


def create_parquet_headers(
    path: Path | str,
    trace_indices: NDArray[np.int64],
    source_x: NDArray[np.float64],
    source_y: NDArray[np.float64],
    receiver_x: NDArray[np.float64],
    receiver_y: NDArray[np.float64],
    shot_ids: NDArray[np.int32] | None = None,
    additional_columns: dict[str, NDArray] | None = None,
) -> None:
    """
    Create a Parquet header file from arrays.

    Args:
        path: Output path
        trace_indices: Global trace indices
        source_x: Source X coordinates
        source_y: Source Y coordinates
        receiver_x: Receiver X coordinates
        receiver_y: Receiver Y coordinates
        shot_ids: Shot IDs (optional)
        additional_columns: Additional columns to include
    """
    # Build dataframe
    data = {
        "trace_idx": trace_indices,
        "SOU_X": source_x,
        "SOU_Y": source_y,
        "REC_X": receiver_x,
        "REC_Y": receiver_y,
    }

    # Compute derived columns
    midpoint_x, midpoint_y = offset_to_midpoint(source_x, source_y, receiver_x, receiver_y)
    data["CDP_X"] = midpoint_x
    data["CDP_Y"] = midpoint_y

    offset = compute_offset(source_x, source_y, receiver_x, receiver_y)
    data["OFFSET"] = offset

    azimuth = compute_azimuth(source_x, source_y, receiver_x, receiver_y)
    data["AZIMUTH"] = azimuth

    if shot_ids is not None:
        data["FFID"] = shot_ids

    if additional_columns:
        data.update(additional_columns)

    # Create and write dataframe
    df = pl.DataFrame(data)
    df.write_parquet(str(path))

    logger.info(f"Created Parquet headers: {path}")
