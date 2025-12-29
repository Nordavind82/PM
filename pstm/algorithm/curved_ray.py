"""
Curved ray traveltime algorithms for PSTM.

Implements traveltime computation for V(z) = V0 + k*z linear gradient media
where rays follow circular arc paths instead of straight lines.

References:
- Slotnick (1959) - Original curved ray theory
- "A Practical Approach of Curved Ray Prestack Kirchhoff Time Migration on GPGPU"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator


def curved_ray_traveltime(x: float, z: float, v0: float, k: float) -> float:
    """
    Compute one-way traveltime for curved ray in V(z) = V0 + k*z medium.

    In a medium with linear velocity gradient, rays follow circular arcs.
    This function computes the traveltime from surface (0,0) to point (x, z).

    Args:
        x: Horizontal distance (m)
        z: Depth (m)
        v0: Velocity at z=0 (m/s)
        k: Velocity gradient (1/s), typically 0.3-0.6

    Returns:
        One-way traveltime (s)
    """
    if abs(k) < 1e-10:
        # Near-zero gradient: use straight ray approximation
        r = np.sqrt(x * x + z * z)
        return r / v0 if r > 0 else 0.0

    # V(z) = v0 + k*z
    v_z = v0 + k * z

    # Curved ray traveltime formula
    # t = (1/k) * ln[(V_z + sqrt(V_z^2 + k^2*x^2)) / (V_0 + sqrt(V_0^2 + k^2*x^2))]
    kx = k * x
    term1 = v_z + np.sqrt(v_z * v_z + kx * kx)
    term2 = v0 + np.sqrt(v0 * v0 + kx * kx)

    if term2 <= 0:
        return 0.0

    return (1.0 / k) * np.log(term1 / term2)


def curved_ray_traveltime_vectorized(
    x: NDArray[np.float64],
    z: NDArray[np.float64],
    v0: float,
    k: float,
) -> NDArray[np.float64]:
    """
    Vectorized curved ray traveltime computation.

    Args:
        x: Horizontal distances (m)
        z: Depths (m)
        v0: Velocity at z=0 (m/s)
        k: Velocity gradient (1/s)

    Returns:
        Traveltimes (s)
    """
    if abs(k) < 1e-10:
        r = np.sqrt(x * x + z * z)
        return np.where(r > 0, r / v0, 0.0)

    v_z = v0 + k * z
    kx = k * x

    term1 = v_z + np.sqrt(v_z * v_z + kx * kx)
    term2 = v0 + np.sqrt(v0 * v0 + kx * kx)

    with np.errstate(divide='ignore', invalid='ignore'):
        t = (1.0 / k) * np.log(term1 / term2)
        t = np.where(term2 > 0, t, 0.0)

    return t


def straight_ray_traveltime(x: float, z: float, v: float) -> float:
    """
    Compute one-way traveltime for straight ray (constant velocity).

    Args:
        x: Horizontal distance (m)
        z: Depth (m)
        v: Velocity (m/s)

    Returns:
        One-way traveltime (s)
    """
    r = np.sqrt(x * x + z * z)
    return r / v if r > 0 and v > 0 else 0.0


def curved_vs_straight_correction(
    t_straight: float,
    offset: float,
    depth: float,
    v0: float,
    k: float,
) -> float:
    """
    Compute correction factor from straight to curved ray traveltime.

    Args:
        t_straight: Straight ray traveltime (s)
        offset: Source-receiver offset (m)
        depth: Image point depth (m)
        v0: Surface velocity (m/s)
        k: Velocity gradient (1/s)

    Returns:
        Ratio t_curved / t_straight
    """
    # Compute average velocity at depth
    v_avg = v0 + k * depth / 2

    # Straight ray traveltime
    h = offset / 2  # Half offset
    t_str = straight_ray_traveltime(h, depth, v_avg)

    if t_str <= 0:
        return 1.0

    # Curved ray traveltime
    t_cur = curved_ray_traveltime(h, depth, v0, k)

    if t_cur <= 0:
        return 1.0

    return t_cur / t_str


def estimate_gradient_from_vrms(
    vrms: NDArray[np.float64],
    t_ms: NDArray[np.float64],
    window_ms: float = 500.0,
) -> tuple[float, float]:
    """
    Estimate V0 and k from RMS velocity function.

    Uses linear regression on V_rms(t) to estimate the gradient.
    For a linear V(z) model, V_rms increases approximately linearly with t.

    Args:
        vrms: RMS velocity array (m/s)
        t_ms: Time axis (ms)
        window_ms: Time window for fitting (ms)

    Returns:
        Tuple of (v0_m_s, k_per_s)
    """
    if len(vrms) < 2:
        return float(vrms[0]) if len(vrms) > 0 else 1500.0, 0.0

    # Convert to seconds
    t_s = t_ms / 1000.0

    # Use early portion of velocity for more stable estimate
    mask = t_ms <= window_ms
    if np.sum(mask) < 2:
        mask = np.ones_like(t_ms, dtype=bool)
        mask[len(mask) // 2:] = False

    t_fit = t_s[mask]
    v_fit = vrms[mask]

    if len(t_fit) < 2:
        return float(vrms[0]), 0.0

    # Linear regression: V_rms(t) ≈ V0 + slope * t
    # For V(z) = V0 + k*z, the relationship is more complex,
    # but this gives a reasonable first-order approximation
    coeffs = np.polyfit(t_fit, v_fit, 1)
    slope = coeffs[0]  # dV/dt
    v0 = coeffs[1]  # Intercept

    # Convert dV/dt to k (1/s)
    # For V(z) = V0 + k*z and z ≈ V*t/2, we get dV/dt ≈ k*V/2
    # So k ≈ 2 * (dV/dt) / V0
    if v0 > 0:
        k = 2.0 * slope / v0
    else:
        k = 0.0

    # Clamp to reasonable range
    v0 = max(1000.0, min(v0, 6000.0))
    k = max(0.0, min(k, 2.0))

    return float(v0), float(k)


def estimate_gradient_from_vint(
    vint: NDArray[np.float64],
    t_ms: NDArray[np.float64],
) -> tuple[float, float]:
    """
    Estimate V0 and k from interval velocity function.

    More accurate than Vrms estimation since Vint directly represents V(z).

    Args:
        vint: Interval velocity array (m/s)
        t_ms: Time axis (ms)

    Returns:
        Tuple of (v0_m_s, k_per_s)
    """
    if len(vint) < 2:
        return float(vint[0]) if len(vint) > 0 else 1500.0, 0.0

    # Convert to depth (approximate)
    t_s = t_ms / 1000.0
    z_approx = np.cumsum(vint * np.gradient(t_s)) / 2  # Approximate depth

    # Linear fit: V(z) = V0 + k*z
    coeffs = np.polyfit(z_approx, vint, 1)
    k = coeffs[0]  # dV/dz
    v0 = coeffs[1]  # Intercept

    # Clamp to reasonable range
    v0 = max(1000.0, min(v0, 6000.0))
    k = max(0.0, min(k, 2.0))

    return float(v0), float(k)


def time_to_depth_curved(t_s: float, v0: float, k: float) -> float:
    """
    Convert one-way vertical time to depth for V(z) = V0 + k*z.

    For vertical rays (x=0), the traveltime simplifies and we can
    solve for depth analytically.

    Args:
        t_s: One-way traveltime (s)
        v0: Surface velocity (m/s)
        k: Velocity gradient (1/s)

    Returns:
        Depth (m)
    """
    if abs(k) < 1e-10:
        # Constant velocity
        return v0 * t_s

    # For vertical ray in V(z) = V0 + k*z:
    # t = (1/k) * ln((V0 + k*z) / V0)
    # Solving for z:
    # z = (V0 / k) * (exp(k*t) - 1)
    return (v0 / k) * (np.exp(k * t_s) - 1)


def depth_to_time_curved(z: float, v0: float, k: float) -> float:
    """
    Convert depth to one-way vertical time for V(z) = V0 + k*z.

    Args:
        z: Depth (m)
        v0: Surface velocity (m/s)
        k: Velocity gradient (1/s)

    Returns:
        One-way traveltime (s)
    """
    if abs(k) < 1e-10:
        return z / v0 if v0 > 0 else 0.0

    v_z = v0 + k * z
    if v_z <= 0:
        return 0.0

    return (1.0 / k) * np.log(v_z / v0)


def curved_ray_dsr_traveltime(
    t0_s: float,
    h_s: float,
    h_r: float,
    v0: float,
    k: float,
) -> float:
    """
    Compute DSR (Double Square Root) traveltime with curved rays.

    This is the curved-ray equivalent of the standard DSR equation.

    Args:
        t0_s: Zero-offset two-way time to image point (s)
        h_s: Source horizontal distance from image point (m)
        h_r: Receiver horizontal distance from image point (m)
        v0: Surface velocity (m/s)
        k: Velocity gradient (1/s)

    Returns:
        Total two-way traveltime (s)
    """
    # Convert t0 to depth
    z = time_to_depth_curved(t0_s / 2, v0, k)  # Half of two-way time

    # Compute curved ray traveltimes for each leg
    t_s = curved_ray_traveltime(h_s, z, v0, k)
    t_r = curved_ray_traveltime(h_r, z, v0, k)

    return t_s + t_r


def get_gradient_info_string(v0: float, k: float) -> str:
    """
    Generate informative string about the velocity gradient.

    Args:
        v0: Surface velocity (m/s)
        k: Velocity gradient (1/s)

    Returns:
        Formatted info string
    """
    lines = [
        f"Velocity Gradient Model: V(z) = {v0:.0f} + {k:.3f}z m/s",
        f"  Surface velocity (V0): {v0:.0f} m/s",
        f"  Gradient (k): {k:.3f} 1/s ({k*1000:.1f} m/s per km)",
    ]

    # Example velocities at depth
    depths = [1000, 2000, 3000, 5000]
    lines.append("  Velocity at depth:")
    for d in depths:
        v = v0 + k * d
        lines.append(f"    z={d}m: V={v:.0f} m/s")

    return "\n".join(lines)
