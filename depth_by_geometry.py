"""
Solves for depth h using CHRTOUT file variables and channel geometry variables.
Used to estimate initial depth for t-route restart files.

Authored by: Quinn Lee
"""

import numpy as np

def solve_depth(
        streamflow: float, velocity: float, tw: float, bw: float, cs: float
        ) -> float:
    """
    Solves for depth h using CHRTOUT file variables and channel geometry variables.

    Parameters:
    - streamflow: Streamflow from CHRTOUT file. (m^3/s)
    - velocity: Velocity from CHRTOUT file. (m/s)
    - tw: Top width of the main channel. (m)
    - bw: Bottom width of the main channel. (m)
    - cs: Channel slope (dimensionless).

    Returns:
    - h: Initial depth that achieves the target flow rate, or NaN if no solution is found.
    """

    area = streamflow / velocity # cross-sectional area of initial flow

    db = (cs * (tw - bw)) / 2 # bankfull depth

    area_bankfull = (tw + bw) / 2 * db  # cross-sectional area at bankfull conditions
    # assume trapezoidal main channel

    if area >= area_bankfull:

        area_flood = area - area_bankfull # cross-sectional area of flow in floodplain only

        df = area_flood / (tw * 3) # depth of flood (assume rectangular floodplain)
        # Also assume compound channel width is 3x main channel top width

        h = db + df # total depth

        return h

    # executes if flow is less than bankfull
    # quadratic equation coefficients
    # h^2 + cs * bw * h - cs * area = 0
    a = 1
    b = cs * bw
    c = (-1) * cs * area
    coeffs = [a, b, c]

    all_roots = np.roots(coeffs)

    real_roots = all_roots[np.isclose(all_roots.imag, 0)].real

    try:
        positive_root = real_roots[real_roots > 0][0]
    except IndexError:
        positive_root = np.nan

    return positive_root
