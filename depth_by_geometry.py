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

    depths = np.zeros_like(area) # initialize depths array with 0 values

    above_bankfull = area >= area_bankfull
    area_flood = area[above_bankfull] - area_bankfull[above_bankfull]
    df = area_flood / (tw[above_bankfull] * 3)
    depths[above_bankfull] = db[above_bankfull] + df

    # Below bankfull - solve quadratic formula directly (vectorized)
    below_bankfull = ~above_bankfull & (area > 0)

    # Quadratic: h^2 + cs*bw*h - cs*area = 0
    # Using formula: h = (-b + sqrt(b^2 + 4*c)) / 2, where a=1
    b_coeff = cs[below_bankfull] * bw[below_bankfull]
    c_coeff = -cs[below_bankfull] * area[below_bankfull]

    discriminant = b_coeff**2 - 4*c_coeff
    h_positive = (-b_coeff + np.sqrt(discriminant)) / 2
    depths[below_bankfull] = h_positive

    return depths
