"""
Solves for depth h using CHRTOUT file variables and channel geometry variables.
Used to estimate initial depth for t-route restart files.

Authored by: Quinn Lee
"""

import numpy as np

def solve_depth_geom(
        streamflow: np.ndarray, velocity: np.ndarray, tw: np.ndarray, bw: np.ndarray, cs: np.ndarray
        ) -> np.ndarray:
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
    area = np.where(np.isnan(area), 0, area) # set NaN areas to 0
    area = np.where(np.isinf(area), 0, area) # set infinite areas to 0

    db = (cs * (tw - bw)) / 2 # bankfull depth

    area_bankfull = (tw + bw) / 2 * db  # cross-sectional area at bankfull conditions
    # assume trapezoidal main channel

    depths = np.zeros_like(area) # initialize depths array with NaN values

    above_bankfull = area >= area_bankfull
    for i, above in enumerate(above_bankfull):
        if above:

            area_flood = area[i] - area_bankfull[i] # cross-sectional area of flow in floodplain only

            df = area_flood / (tw[i] * 3) # depth of flood (assume rectangular floodplain)
            # Also assume compound channel width is 3x main channel top width

            h = db[i] + df # total depth
            depths[i] = h

        else:
            # executes if flow is less than bankfull
            # quadratic equation coefficients
            # h^2 + cs * bw * h - cs * area = 0

            if area[i] <= 0:
                depths[i] = 0.0
                continue

            a = 1
            b = cs[i] * bw[i]
            c = (-1) * cs[i] * area[i]
            coeffs = [a, b, c]

            all_roots = np.roots(coeffs)
            real_roots = all_roots[np.isclose(all_roots.imag, 0)].real

            try:
                h = real_roots[real_roots > 0][0]
            except IndexError:
                h = np.nan # No positive roots found

            depths[i] = h

    return depths
