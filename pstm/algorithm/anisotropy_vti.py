"""
VTI anisotropic traveltime algorithms for PSTM.

Implements Alkhalifah-Tsvankin (1995) formulation for P-wave time processing
in vertically transversely isotropic (VTI) media using the eta parameter.

The key insight is that time processing (NMO, DMO, PSTM) depends only on:
1. V_nmo - NMO velocity for horizontal reflector
2. η (eta) - anisotropy parameter

η = (ε - δ) / (1 + 2δ) ≈ ε - δ for weak anisotropy

References:
- Alkhalifah, T., & Tsvankin, I. (1995). Velocity analysis for transversely
  isotropic media. Geophysics, 60(5), 1550-1566.
- Alkhalifah, T. (1997). Velocity analysis using nonhyperbolic moveout in
  transversely isotropic media. Geophysics, 62(6), 1839-1854.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator


def vti_nonhyperbolic_moveout(
    t0_s: float,
    x: float,
    v_nmo: float,
    eta: float,
) -> float:
    """
    Compute VTI non-hyperbolic moveout traveltime.

    This is the Alkhalifah (1998) approximation:
    t²(x) = t₀² + x²/V_nmo² - 2η·x⁴ / [V_nmo²·(t₀²·V_nmo² + (1+2η)·x²)]

    Args:
        t0_s: Zero-offset two-way time (s)
        x: Offset (m)
        v_nmo: NMO velocity (m/s)
        eta: Anisotropy parameter (dimensionless, typical 0.05-0.20)

    Returns:
        Non-hyperbolic traveltime (s)
    """
    if v_nmo <= 0 or t0_s <= 0:
        return t0_s

    x2 = x * x
    v2 = v_nmo * v_nmo
    t02 = t0_s * t0_s

    # Hyperbolic term
    t2_hyper = t02 + x2 / v2

    # Non-hyperbolic correction (fourth-order term)
    # C = 2η·x⁴ / [V²·(t₀²·V² + (1+2η)·x²)]
    denom = v2 * (t02 * v2 + (1.0 + 2.0 * eta) * x2)

    if abs(denom) < 1e-20:
        return np.sqrt(max(t2_hyper, 0.0))

    correction = (2.0 * eta * x2 * x2) / denom

    t2 = t2_hyper - correction

    return np.sqrt(max(t2, 0.0))


def vti_nonhyperbolic_moveout_vectorized(
    t0_s: NDArray[np.float64],
    x: NDArray[np.float64],
    v_nmo: NDArray[np.float64],
    eta: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """
    Vectorized VTI non-hyperbolic moveout computation.

    Args:
        t0_s: Zero-offset two-way times (s)
        x: Offsets (m)
        v_nmo: NMO velocities (m/s)
        eta: Anisotropy parameter(s)

    Returns:
        Non-hyperbolic traveltimes (s)
    """
    x2 = x * x
    v2 = v_nmo * v_nmo
    t02 = t0_s * t0_s

    # Hyperbolic term
    t2_hyper = t02 + x2 / v2

    # Non-hyperbolic correction
    eta_arr = np.asarray(eta)
    denom = v2 * (t02 * v2 + (1.0 + 2.0 * eta_arr) * x2)

    with np.errstate(divide='ignore', invalid='ignore'):
        correction = np.where(
            np.abs(denom) > 1e-20,
            (2.0 * eta_arr * x2 * x2) / denom,
            0.0
        )

    t2 = t2_hyper - correction

    return np.sqrt(np.maximum(t2, 0.0))


def vti_correction_term(
    t0_s: float,
    x: float,
    v_nmo: float,
    eta: float,
) -> float:
    """
    Compute only the VTI correction term (fourth-order).

    This is the difference between VTI and isotropic traveltime.

    Args:
        t0_s: Zero-offset two-way time (s)
        x: Offset (m)
        v_nmo: NMO velocity (m/s)
        eta: Anisotropy parameter

    Returns:
        Correction term (s²), to be subtracted from t²_hyperbolic
    """
    if v_nmo <= 0 or t0_s <= 0:
        return 0.0

    x2 = x * x
    v2 = v_nmo * v_nmo
    t02 = t0_s * t0_s

    denom = v2 * (t02 * v2 + (1.0 + 2.0 * eta) * x2)

    if abs(denom) < 1e-20:
        return 0.0

    return (2.0 * eta * x2 * x2) / denom


def vti_dsr_traveltime(
    t0_s: float,
    h_s: float,
    h_r: float,
    v_nmo: float,
    eta: float,
) -> float:
    """
    Compute DSR (Double Square Root) traveltime with VTI corrections.

    Applies non-hyperbolic correction to both source and receiver legs.

    Args:
        t0_s: Zero-offset two-way time to image point (s)
        h_s: Source horizontal distance from image point (m)
        h_r: Receiver horizontal distance from image point (m)
        v_nmo: NMO velocity (m/s)
        eta: Anisotropy parameter

    Returns:
        Total traveltime (s)
    """
    # One-way time for each leg
    t0_half = t0_s / 2.0

    # Apply VTI correction to each leg independently
    # Using the one-way equivalent of the non-hyperbolic formula
    t_s = vti_one_way_time(t0_half, h_s, v_nmo, eta)
    t_r = vti_one_way_time(t0_half, h_r, v_nmo, eta)

    return t_s + t_r


def vti_one_way_time(
    t0_one_way: float,
    h: float,
    v_nmo: float,
    eta: float,
) -> float:
    """
    Compute one-way VTI traveltime for a single ray leg.

    Args:
        t0_one_way: One-way zero-offset time (s)
        h: Horizontal distance (m)
        v_nmo: NMO velocity (m/s)
        eta: Anisotropy parameter

    Returns:
        One-way traveltime (s)
    """
    if v_nmo <= 0 or t0_one_way <= 0:
        return t0_one_way

    h2 = h * h
    v2 = v_nmo * v_nmo
    t02 = t0_one_way * t0_one_way

    # Hyperbolic term
    t2_hyper = t02 + h2 / v2

    # Scale eta for one-way (approximation)
    # The correction is applied with half the effect for one-way
    denom = v2 * (t02 * v2 + (1.0 + 2.0 * eta) * h2)

    if abs(denom) < 1e-20:
        return np.sqrt(max(t2_hyper, 0.0))

    correction = (2.0 * eta * h2 * h2) / denom

    t2 = t2_hyper - correction

    return np.sqrt(max(t2, 0.0))


def interpolate_eta_1d(
    t_ms: float,
    eta_table: list[tuple[float, float]],
) -> float:
    """
    Interpolate eta from 1D table using PCHIP.

    Args:
        t_ms: Time to interpolate at (ms)
        eta_table: List of (time_ms, eta) tuples

    Returns:
        Interpolated eta value
    """
    if not eta_table:
        return 0.0

    if len(eta_table) == 1:
        return eta_table[0][1]

    times = np.array([t for t, e in eta_table])
    etas = np.array([e for t, e in eta_table])

    # Clamp to table range
    if t_ms <= times[0]:
        return float(etas[0])
    if t_ms >= times[-1]:
        return float(etas[-1])

    # PCHIP interpolation for smooth curve
    interp = PchipInterpolator(times, etas)
    return float(interp(t_ms))


def interpolate_eta_to_axis(
    eta_table: list[tuple[float, float]],
    t_axis_ms: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Interpolate eta table to full time axis.

    Args:
        eta_table: List of (time_ms, eta) tuples
        t_axis_ms: Output time axis (ms)

    Returns:
        Eta values at each time sample
    """
    if not eta_table:
        return np.zeros_like(t_axis_ms)

    if len(eta_table) == 1:
        return np.full_like(t_axis_ms, eta_table[0][1])

    times = np.array([t for t, e in eta_table])
    etas = np.array([e for t, e in eta_table])

    # PCHIP interpolation
    interp = PchipInterpolator(times, etas, extrapolate=True)
    result = interp(t_axis_ms)

    # Clamp to reasonable range
    result = np.clip(result, -0.5, 0.5)

    return result


def thomsen_to_eta(epsilon: float, delta: float) -> float:
    """
    Convert Thomsen parameters to eta.

    η = (ε - δ) / (1 + 2δ)

    Args:
        epsilon: Thomsen ε parameter
        delta: Thomsen δ parameter

    Returns:
        Eta parameter
    """
    return (epsilon - delta) / (1.0 + 2.0 * delta)


def eta_to_thomsen_weak(eta: float, delta: float = 0.05) -> tuple[float, float]:
    """
    Estimate Thomsen parameters from eta (weak anisotropy approximation).

    For weak anisotropy: η ≈ ε - δ
    Given eta and assumed delta, compute epsilon.

    Args:
        eta: Eta parameter
        delta: Assumed delta value (default 0.05)

    Returns:
        Tuple of (epsilon, delta)
    """
    # η ≈ ε - δ for weak anisotropy
    epsilon = eta + delta
    return epsilon, delta


def estimate_eta_from_semblance(
    cmp_gather: NDArray[np.float64],
    offsets: NDArray[np.float64],
    t_axis_ms: NDArray[np.float64],
    v_nmo: NDArray[np.float64],
    eta_range: tuple[float, float] = (-0.1, 0.3),
    eta_step: float = 0.01,
) -> NDArray[np.float64]:
    """
    Estimate eta from CMP gather using semblance analysis.

    Scans eta values and picks maximum semblance at each time.

    Args:
        cmp_gather: CMP gather (n_traces, n_samples)
        offsets: Offset values (m)
        t_axis_ms: Time axis (ms)
        v_nmo: NMO velocity function (m/s)
        eta_range: Range of eta to scan
        eta_step: Step size for eta scan

    Returns:
        Estimated eta vs time
    """
    n_traces, n_samples = cmp_gather.shape
    t_s = t_axis_ms / 1000.0

    eta_values = np.arange(eta_range[0], eta_range[1] + eta_step, eta_step)
    n_eta = len(eta_values)

    # Semblance panel
    semblance = np.zeros((n_samples, n_eta))

    for i_eta, eta in enumerate(eta_values):
        for i_t in range(n_samples):
            t0 = t_s[i_t]
            v = v_nmo[i_t] if len(v_nmo) > 1 else v_nmo[0]

            # Compute NMO-corrected times for this eta
            t_nmo = np.array([
                vti_nonhyperbolic_moveout(t0, x, v, eta)
                for x in offsets
            ])

            # Convert to sample indices
            sample_idx = (t_nmo * 1000 - t_axis_ms[0]) / (t_axis_ms[1] - t_axis_ms[0])
            sample_idx = sample_idx.astype(int)

            # Extract samples and compute semblance
            valid = (sample_idx >= 0) & (sample_idx < n_samples)
            if np.sum(valid) < 2:
                continue

            samples = np.array([
                cmp_gather[i_tr, idx] if valid[i_tr] else 0.0
                for i_tr, idx in enumerate(sample_idx)
            ])

            # Semblance = (sum(samples))² / (n * sum(samples²))
            n_valid = np.sum(valid)
            sum_sq = np.sum(samples ** 2)
            if sum_sq > 0:
                semblance[i_t, i_eta] = (np.sum(samples) ** 2) / (n_valid * sum_sq)

    # Pick maximum semblance for each time
    eta_picked = np.array([
        eta_values[np.argmax(semblance[i_t, :])]
        for i_t in range(n_samples)
    ])

    return eta_picked


def vti_moveout_residual(
    t_observed: float,
    t0_s: float,
    x: float,
    v_nmo: float,
    eta: float,
) -> float:
    """
    Compute residual between observed and VTI-predicted traveltime.

    Useful for velocity/eta analysis and inversion.

    Args:
        t_observed: Observed traveltime (s)
        t0_s: Zero-offset time (s)
        x: Offset (m)
        v_nmo: NMO velocity (m/s)
        eta: Anisotropy parameter

    Returns:
        Residual t_observed - t_predicted (s)
    """
    t_predicted = vti_nonhyperbolic_moveout(t0_s, x, v_nmo, eta)
    return t_observed - t_predicted


def get_eta_info_string(eta: float) -> str:
    """
    Generate informative string about eta parameter.

    Args:
        eta: Anisotropy parameter

    Returns:
        Formatted info string
    """
    # Estimate Thomsen parameters for weak anisotropy
    eps, delta = eta_to_thomsen_weak(eta)

    lines = [
        f"VTI Anisotropy Parameter: η = {eta:.3f}",
        "",
        f"For weak anisotropy (assumed δ = 0.05):",
        f"  ε (epsilon) ≈ {eps:.3f}",
        f"  δ (delta) ≈ {delta:.3f}",
        "",
        "Physical interpretation:",
    ]

    if eta < 0:
        lines.append("  Negative η: unusual, may indicate HTI or measurement error")
    elif eta < 0.05:
        lines.append("  Low η (< 0.05): weak anisotropy, near-isotropic")
    elif eta < 0.15:
        lines.append("  Moderate η (0.05-0.15): typical for shales")
    elif eta < 0.25:
        lines.append("  High η (0.15-0.25): strong anisotropy, thick shale sequences")
    else:
        lines.append("  Very high η (> 0.25): extreme anisotropy, verify data quality")

    return "\n".join(lines)


def get_vti_table_info_string(eta_table: list[tuple[float, float]]) -> str:
    """
    Generate info string for eta table.

    Args:
        eta_table: List of (time_ms, eta) tuples

    Returns:
        Formatted info string
    """
    if not eta_table:
        return "Empty eta table"

    lines = [
        f"VTI Eta Table ({len(eta_table)} points):",
        f"{'Time (ms)':<12} {'Eta':<10}",
        "-" * 22,
    ]

    for t, eta in eta_table:
        lines.append(f"{t:<12.1f} {eta:<10.3f}")

    # Statistics
    etas = [e for _, e in eta_table]
    lines.extend([
        "-" * 22,
        f"Min eta: {min(etas):.3f}",
        f"Max eta: {max(etas):.3f}",
        f"Mean eta: {np.mean(etas):.3f}",
    ])

    return "\n".join(lines)
