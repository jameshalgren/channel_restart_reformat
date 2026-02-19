"""
Iteratively calculate depth from flow using Manning's equation
"""
import numpy as np
from scipy.optimize import brentq

def compound_geometry(
        h: float, bfd: float, bw: float, twcc: float, z: float
                      ) -> tuple[float, float, float, float, float]:
    """Calculates hydraulic properties for a compound channel."""
    h_lt_bf = min(h, bfd) # depth in main channel
    h_gt_bf = max(0.0, h - bfd) # depth in floodplain (if any)

    # Main Channel (Trapezoidal)
    area_main = (bw + h_lt_bf * z) * h_lt_bf
    wp_main = bw + 2 * h_lt_bf * np.sqrt(1 + z**2) # wetted perimeter

    # Floodplain (Rectangular/Simplified CC)
    # Based on your Fortran logic: AREA = twcc * h_gt_bf
    area_cc = twcc * h_gt_bf
    wp_cc = 2 * h_gt_bf + (2 * twcc) / 3 if h_gt_bf > 0 else 0.0

    total_area = area_main + area_cc
    total_wp = wp_main + wp_cc

    # Hydraulic Radius
    r = total_area / total_wp if total_wp > 0 else 0.0

    return wp_main, wp_cc, total_area, total_wp, r

def manning_residual(h: float, target_q: float, bfd: float, bw: float, twcc: float, z: float,
                     n: float, ncc: float, s0: float) -> float:

    """f(h) = Q_calc(h) - target_q. We want to find where this equals zero."""
    if h <= 0 or s0 <= 0:
        return -target_q

    wp_main, wp_cc, total_area, total_wp, r = compound_geometry(h, bfd, bw, twcc, z)

    # Composite Manning n (Perimeter weighted as per NWM logic)
    # n_eff = (P_channel * n + P_floodplain * ncc) / P_total

    n_eff = (wp_main * n + wp_cc * ncc) / total_wp if total_wp > 0 else n

    q_calc = (1.0 / n_eff) * total_area * (r**(2/3)) * np.sqrt(s0)
    return q_calc - target_q

def solve_depth(target_q: np.ndarray, bw: np.ndarray, tw: np.ndarray,
                cs: np.ndarray, n: np.ndarray, ncc: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """
    Solves for depth h using Brent's method.

    Parameters:
    - target_q: Target flow rate from CHRTOUT file. (m^3/s)
    - bw: Bottom width of the main channel. (m)
    - tw: Top width of the main channel. (m)
    - cs: Channel slope (dimensionless).
    - n: Manning's n for the main channel.
    - ncc: Manning's n for the floodplain (compound channel).
    - s0: Channel slope (dimensionless).

    Returns:
    - h: Depth that achieves the target flow rate, or 0 if no solution is found.
    """

    depths = np.zeros_like(target_q, dtype=float)

    # Pre-compute all parameters (vectorized)
    z = np.where(cs == 0, 1.0, 1.0 / cs)

    bfd = np.where(bw > tw, bw / 0.00001,
          np.where(bw == tw, bw / (2.0 * z), (tw - bw) / (2.0 * z)))

    twcc = 3 * tw

    for i, target in enumerate(target_q):
        # Skip if target flow is non-positive or NaN or if slope is non-positive
        if not np.isfinite(target) or target <= 0 or not np.isfinite(s0[i]) or s0[i] <= 0:
            depths[i] = 0.0
            continue
        try:
            # Search between 0.001m and 100m depth (adjust upper bound if needed)
            depths[i] = brentq(manning_residual, 0, 3000,
                        args=(target, bfd[i], bw[i], twcc[i], z[i], n[i], ncc[i], s0[i]))
        except (RuntimeError, ValueError):
            depths[i] = 0.0  # No solution found in range

    return depths
