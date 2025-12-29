"""
Grid outlier detection and classification.

This module provides methods to classify points relative to an output grid
defined by corner points, including:

- Point containment testing
- Distance to boundary calculation
- Outlier statistics and reporting
- Buffer zone extension suggestions

Used for pre-migration QC to identify traces outside the output grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class OutlierHandling(str, Enum):
    """Strategy for handling outliers."""
    EXCLUDE = "exclude"  # Simply exclude traces outside grid
    BUFFER_EXTEND = "buffer_extend"  # Extend grid by buffer
    APERTURE_EXTEND = "aperture_extend"  # Extend by migration aperture
    TAPER_WEIGHT = "taper_weight"  # Apply tapered weights near boundary


@dataclass
class GridClassificationResult:
    """Result of classifying points against grid boundary."""

    # Masks
    inside_mask: NDArray[np.bool_]  # True for points inside grid

    # Distances (negative = inside, positive = outside)
    signed_distances: NDArray[np.floating]

    # Edge information (0=C1-C2, 1=C2-C3, 2=C3-C4, 3=C4-C1)
    nearest_edge: NDArray[np.int32]

    # Statistics
    n_total: int
    n_inside: int
    n_outside: int
    inside_ratio: float

    # Distance statistics for outside points
    max_distance_outside: float
    mean_distance_outside: float

    # Quadrant breakdown for outside points (NE, SE, SW, NW)
    outside_by_quadrant: dict[str, int]

    # Suggested buffer to include all points
    suggested_buffer_m: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "n_total": self.n_total,
            "n_inside": self.n_inside,
            "n_outside": self.n_outside,
            "inside_ratio": self.inside_ratio,
            "max_distance_outside": self.max_distance_outside,
            "mean_distance_outside": self.mean_distance_outside,
            "outside_by_quadrant": self.outside_by_quadrant,
            "suggested_buffer_m": self.suggested_buffer_m,
        }

    def get_summary_text(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Grid Coverage Analysis:",
            f"  Total points: {self.n_total:,}",
            f"  Inside grid: {self.n_inside:,} ({self.inside_ratio*100:.1f}%)",
            f"  Outside grid: {self.n_outside:,} ({(1-self.inside_ratio)*100:.1f}%)",
        ]

        if self.n_outside > 0:
            lines.extend([
                f"  Max distance outside: {self.max_distance_outside:.1f} m",
                f"  Mean distance outside: {self.mean_distance_outside:.1f} m",
                f"  Suggested buffer: {self.suggested_buffer_m:.1f} m",
            ])

            # Quadrant breakdown
            if any(v > 0 for v in self.outside_by_quadrant.values()):
                lines.append("  Outside points by direction:")
                for direction, count in self.outside_by_quadrant.items():
                    if count > 0:
                        lines.append(f"    {direction}: {count:,}")

        return "\n".join(lines)


def compute_edge_signed_distance(
    px: NDArray[np.floating],
    py: NDArray[np.floating],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> NDArray[np.floating]:
    """
    Compute signed distance from points to a line segment (edge).

    Positive distance means point is on the right side of the edge
    (outside if edges are ordered clockwise), negative means left (inside).

    Args:
        px, py: Point coordinates
        p1, p2: Edge endpoints

    Returns:
        Signed distance array (positive = outside, negative = inside for CW ordering)
    """
    # Edge vector
    edge_x = p2[0] - p1[0]
    edge_y = p2[1] - p1[1]
    edge_length = np.sqrt(edge_x**2 + edge_y**2)

    if edge_length < 1e-10:
        return np.zeros(len(px))

    # Outward normal (perpendicular to edge, pointing right for CW ordering)
    normal_x = edge_y / edge_length
    normal_y = -edge_x / edge_length

    # Vector from p1 to each point
    dx = px - p1[0]
    dy = py - p1[1]

    # Signed distance = dot product with normal
    signed_dist = dx * normal_x + dy * normal_y

    return signed_dist


def classify_points_against_grid(
    px: NDArray[np.floating],
    py: NDArray[np.floating],
    corners: NDArray[np.floating],
    buffer_m: float = 0.0,
) -> GridClassificationResult:
    """
    Classify points relative to a grid defined by 4 corner points.

    The corners should be in order: C1 (SW), C2 (SE), C3 (NE), C4 (NW),
    forming a clockwise quadrilateral.

    Args:
        px: X coordinates of points to classify
        py: Y coordinates of points to classify
        corners: (4, 2) array of corner coordinates
        buffer_m: Optional buffer to extend grid (positive = expand)

    Returns:
        GridClassificationResult with masks and statistics
    """
    n_points = len(px)

    if n_points == 0:
        return GridClassificationResult(
            inside_mask=np.array([], dtype=bool),
            signed_distances=np.array([], dtype=np.float64),
            nearest_edge=np.array([], dtype=np.int32),
            n_total=0,
            n_inside=0,
            n_outside=0,
            inside_ratio=1.0,
            max_distance_outside=0.0,
            mean_distance_outside=0.0,
            outside_by_quadrant={"NE": 0, "SE": 0, "SW": 0, "NW": 0},
            suggested_buffer_m=0.0,
        )

    # Corners in order: C1(SW), C2(SE), C3(NE), C4(NW)
    # Edges: C1-C2 (bottom), C2-C3 (right), C3-C4 (top), C4-C1 (left)
    edges = [
        (tuple(corners[0]), tuple(corners[1])),  # Bottom edge
        (tuple(corners[1]), tuple(corners[2])),  # Right edge
        (tuple(corners[2]), tuple(corners[3])),  # Top edge
        (tuple(corners[3]), tuple(corners[0])),  # Left edge
    ]

    # Compute signed distance to each edge
    edge_distances = np.zeros((n_points, 4), dtype=np.float64)
    for i, (p1, p2) in enumerate(edges):
        edge_distances[:, i] = compute_edge_signed_distance(px, py, p1, p2)

    # For a convex quadrilateral with CW ordering:
    # Point is inside if all signed distances are negative (on left side of all edges)
    # Apply buffer by offsetting the threshold
    inside_mask = np.all(edge_distances <= buffer_m, axis=1)

    # Distance to boundary = minimum distance to any edge
    # For inside points, this is negative (take max of negatives)
    # For outside points, this is the max positive distance
    min_edge_dist = edge_distances.min(axis=1)  # Most negative = furthest inside
    max_edge_dist = edge_distances.max(axis=1)  # Most positive = furthest outside

    # Signed distance: negative inside, positive outside
    signed_distances = np.where(inside_mask, min_edge_dist, max_edge_dist)

    # Nearest edge
    nearest_edge = np.argmin(np.abs(edge_distances), axis=1).astype(np.int32)

    # Statistics
    n_inside = int(inside_mask.sum())
    n_outside = n_points - n_inside
    inside_ratio = n_inside / n_points if n_points > 0 else 1.0

    # Outside point statistics
    if n_outside > 0:
        outside_distances = signed_distances[~inside_mask]
        max_distance_outside = float(outside_distances.max())
        mean_distance_outside = float(outside_distances.mean())
    else:
        max_distance_outside = 0.0
        mean_distance_outside = 0.0

    # Classify outside points by direction (quadrant relative to grid center)
    center_x = corners[:, 0].mean()
    center_y = corners[:, 1].mean()

    outside_mask = ~inside_mask
    outside_px = px[outside_mask]
    outside_py = py[outside_mask]

    quadrant_counts = {"NE": 0, "SE": 0, "SW": 0, "NW": 0}
    if n_outside > 0:
        dx_from_center = outside_px - center_x
        dy_from_center = outside_py - center_y

        quadrant_counts["NE"] = int(((dx_from_center >= 0) & (dy_from_center >= 0)).sum())
        quadrant_counts["SE"] = int(((dx_from_center >= 0) & (dy_from_center < 0)).sum())
        quadrant_counts["SW"] = int(((dx_from_center < 0) & (dy_from_center < 0)).sum())
        quadrant_counts["NW"] = int(((dx_from_center < 0) & (dy_from_center >= 0)).sum())

    # Suggested buffer to include all points (with 5% margin)
    suggested_buffer = max_distance_outside * 1.05 if max_distance_outside > 0 else 0.0

    return GridClassificationResult(
        inside_mask=inside_mask,
        signed_distances=signed_distances,
        nearest_edge=nearest_edge,
        n_total=n_points,
        n_inside=n_inside,
        n_outside=n_outside,
        inside_ratio=inside_ratio,
        max_distance_outside=max_distance_outside,
        mean_distance_outside=mean_distance_outside,
        outside_by_quadrant=quadrant_counts,
        suggested_buffer_m=suggested_buffer,
    )


def compute_extended_corners(
    corners: NDArray[np.floating],
    buffer_m: float,
) -> NDArray[np.floating]:
    """
    Extend grid corners outward by a buffer distance.

    Each corner is moved outward along the diagonal direction from the center.

    Args:
        corners: (4, 2) array of corner coordinates
        buffer_m: Buffer distance in meters

    Returns:
        (4, 2) array of extended corner coordinates
    """
    center = corners.mean(axis=0)
    extended = corners.copy()

    for i in range(4):
        # Direction from center to corner
        direction = corners[i] - center
        length = np.linalg.norm(direction)
        if length > 0:
            unit_dir = direction / length
            # Extend outward by buffer
            extended[i] = corners[i] + unit_dir * buffer_m

    return extended


def compute_aperture_extended_corners(
    corners: NDArray[np.floating],
    max_offset_m: float,
    max_dip_deg: float,
    velocity_m_s: float = 2500.0,
) -> NDArray[np.floating]:
    """
    Extend grid corners by the migration aperture.

    Traces within the aperture distance outside the grid can contribute
    energy to output points near the edge.

    Args:
        corners: (4, 2) array of corner coordinates
        max_offset_m: Maximum offset in meters
        max_dip_deg: Maximum dip angle in degrees
        velocity_m_s: Representative velocity for aperture calculation

    Returns:
        (4, 2) array of extended corner coordinates
    """
    # Simple aperture estimate: horizontal distance covered by dipping reflector
    # For a dip of theta, horizontal aperture ~ max_offset * tan(theta)
    max_dip_rad = np.radians(max_dip_deg)
    aperture_m = max_offset_m * np.tan(max_dip_rad)

    # Clamp to reasonable range
    aperture_m = min(aperture_m, max_offset_m)

    return compute_extended_corners(corners, aperture_m)


def compute_boundary_taper_weights(
    px: NDArray[np.floating],
    py: NDArray[np.floating],
    corners: NDArray[np.floating],
    taper_width_m: float = 500.0,
) -> NDArray[np.floating]:
    """
    Compute taper weights for points near grid boundary.

    Points inside the grid but within taper_width of the boundary
    get a weight < 1, ramping from 0 at boundary to 1 at full taper distance.

    Args:
        px, py: Point coordinates
        corners: (4, 2) array of corner coordinates
        taper_width_m: Width of taper zone in meters

    Returns:
        Weight array (0-1) for each point
    """
    result = classify_points_against_grid(px, py, corners)

    # Start with all weights = 1
    weights = np.ones(len(px), dtype=np.float64)

    # Points outside get weight 0
    weights[~result.inside_mask] = 0.0

    # Points inside but near boundary get tapered weight
    inside_mask = result.inside_mask
    distances = -result.signed_distances[inside_mask]  # Make positive for inside

    # Taper: 0 at boundary, 1 at taper_width inside
    taper = np.clip(distances / taper_width_m, 0, 1)

    # Apply cosine taper for smooth transition
    taper = 0.5 * (1 - np.cos(np.pi * taper))

    weights[inside_mask] = taper

    return weights


@dataclass
class OutlierReport:
    """Complete outlier analysis report for a dataset."""

    # Classification result
    classification: GridClassificationResult

    # Extended grid suggestions
    buffer_extended_corners: NDArray[np.floating] | None = None
    aperture_extended_corners: NDArray[np.floating] | None = None

    # Handling recommendation
    recommended_handling: OutlierHandling = OutlierHandling.EXCLUDE
    recommendation_reason: str = ""

    # Quality flags
    coverage_acceptable: bool = True
    outlier_warning: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "classification": self.classification.to_dict(),
            "recommended_handling": self.recommended_handling.value,
            "recommendation_reason": self.recommendation_reason,
            "coverage_acceptable": self.coverage_acceptable,
            "outlier_warning": self.outlier_warning,
        }

        if self.buffer_extended_corners is not None:
            result["buffer_extended_corners"] = self.buffer_extended_corners.tolist()
        if self.aperture_extended_corners is not None:
            result["aperture_extended_corners"] = self.aperture_extended_corners.tolist()

        return result


def generate_outlier_report(
    px: NDArray[np.floating],
    py: NDArray[np.floating],
    corners: NDArray[np.floating],
    max_offset_m: float = 5000.0,
    max_dip_deg: float = 45.0,
    acceptable_outlier_ratio: float = 0.05,
) -> OutlierReport:
    """
    Generate a complete outlier analysis report.

    Args:
        px, py: Point coordinates (typically midpoints)
        corners: (4, 2) array of grid corner coordinates
        max_offset_m: Maximum offset for aperture calculation
        max_dip_deg: Maximum dip for aperture calculation
        acceptable_outlier_ratio: Ratio of outliers considered acceptable

    Returns:
        OutlierReport with analysis and recommendations
    """
    # Classify points
    classification = classify_points_against_grid(px, py, corners)

    # Compute extended corners
    buffer_extended = compute_extended_corners(
        corners, classification.suggested_buffer_m
    ) if classification.n_outside > 0 else None

    aperture_extended = compute_aperture_extended_corners(
        corners, max_offset_m, max_dip_deg
    )

    # Determine recommendation
    outlier_ratio = 1 - classification.inside_ratio

    if outlier_ratio <= 0.001:
        # <0.1% outliers: just exclude
        handling = OutlierHandling.EXCLUDE
        reason = "Very few outliers (<0.1%), safe to exclude"
        warning = False
        acceptable = True
    elif outlier_ratio <= acceptable_outlier_ratio:
        # Acceptable level: exclude but note
        handling = OutlierHandling.EXCLUDE
        reason = f"Outlier ratio ({outlier_ratio*100:.1f}%) within acceptable range"
        warning = False
        acceptable = True
    elif outlier_ratio <= 0.15:
        # Moderate outliers: suggest buffer
        handling = OutlierHandling.BUFFER_EXTEND
        reason = (
            f"Significant outliers ({outlier_ratio*100:.1f}%). "
            f"Consider extending grid by {classification.suggested_buffer_m:.0f}m"
        )
        warning = True
        acceptable = True
    else:
        # Too many outliers: grid may be misconfigured
        handling = OutlierHandling.BUFFER_EXTEND
        reason = (
            f"High outlier ratio ({outlier_ratio*100:.1f}%). "
            f"Grid corners may not match data extent. "
            f"Review grid definition or extend by {classification.suggested_buffer_m:.0f}m"
        )
        warning = True
        acceptable = False

    return OutlierReport(
        classification=classification,
        buffer_extended_corners=buffer_extended,
        aperture_extended_corners=aperture_extended,
        recommended_handling=handling,
        recommendation_reason=reason,
        coverage_acceptable=acceptable,
        outlier_warning=warning,
    )
