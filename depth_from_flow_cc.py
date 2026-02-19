"""
Iteratively calculate depth from flow using Manning's equation
"""
import numpy as np
from scipy.optimize import brentq

def compound_geometry(h, bfd, bw, twcc, z):
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

def manning_residual(h, target_q, bfd, bw, twcc, z, n, ncc, s0):
    """f(h) = Q_calc(h) - target_q. We want to find where this equals zero."""
    if h <= 0:
        return -target_q

    wp_main, wp_cc, total_area, total_wp, r = compound_geometry(h, bfd, bw, twcc, z)

    # Composite Manning n (Perimeter weighted as per NWM logic)
    # n_eff = (P_channel * n + P_floodplain * ncc) / P_total

    n_eff = (wp_main * n + wp_cc * ncc) / total_wp if total_wp > 0 else n

    q_calc = (1.0 / n_eff) * total_area * (r**(2/3)) * np.sqrt(s0)
    return q_calc - target_q

def solve_depth(target_q, bw, tw, twcc, cs, n, ncc, s0):
    """Solves for depth h using Brent's method."""
    if target_q <= 0:
        return 0.0

    # calculate z
    if cs == 0:
        z = 1.0
    else:
        z = 1.0 / cs

    # calculate bankfull depth
    if bw > tw:
        bfd = bw/0.00001
    elif bw == tw:
        bfd = bw/(2.0 * z)
    else:
        bfd = (tw - bw) / (2.0 * z)

    try:
        # Search between 0m and 100m depth (adjust upper bound if needed)
        return brentq(manning_residual, 0, 100,
                      args=(target_q, bfd, bw, twcc, z, n, ncc, s0))
    except RuntimeError:
        return np.nan # No solution found in range
