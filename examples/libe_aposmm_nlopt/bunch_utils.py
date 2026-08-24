"""Defines utilities for generating particle bunches."""

import numpy as np
from scipy.constants import c


def gaussian_bunch(
    q_tot, n_part, gamma0, s_g, s_z, emit_x, s_x, zf=0.0, tf=0, x_c=0.0
):
    """Create a Gaussian particle bunch."""
    n_part = int(n_part)

    np.random.seed(42)
    z = zf + s_z * np.random.standard_normal(n_part)
    x = x_c + s_x * np.random.standard_normal(n_part)
    y = s_x * np.random.standard_normal(n_part)

    gamma = np.random.normal(gamma0, s_g, n_part)

    s_ux = emit_x / s_x
    ux = s_ux * np.random.standard_normal(n_part)
    uy = s_ux * np.random.standard_normal(n_part)

    uz = np.sqrt((gamma**2 - 1) - ux**2 - uy**2)

    if tf != 0.0:
        x = x - ux * c * tf / gamma
        y = y - uy * c * tf / gamma
        z = z - uz * c * tf / gamma

    q = np.ones(n_part) * q_tot / n_part

    return x, y, z, ux, uy, uz, q


def flattop_bunch(
    q_tot,
    n_part,
    gamma0,
    s_g,
    length,
    s_z,
    emit_x,
    s_x,
    emit_y,
    s_y,
    zf=0.0,
    tf=0,
    x_c=0.0,
    y_c=0,
):
    """Create a flat-top particle bunch."""
    n_part = int(n_part)

    norma = length + np.sqrt(2 * np.pi) * s_z
    n_plat = int(n_part * length / norma)
    n_gaus = int(n_part * np.sqrt(2 * np.pi) * s_z / norma)

    # Create flattop and gaussian profiles
    z_plat = np.random.uniform(0.0, length, n_plat)
    z_gaus = s_z * np.random.standard_normal(n_gaus)

    # Concatenate both profiles
    z = np.concatenate(
        (
            z_gaus[np.where(z_gaus <= 0)],
            z_plat,
            z_gaus[np.where(z_gaus > 0)] + length,
        )
    )

    z = z - length / 2.0 + zf  # shift to final position

    n_part = len(z)
    x = x_c + s_x * np.random.standard_normal(n_part)
    y = y_c + s_y * np.random.standard_normal(n_part)

    gamma = np.random.normal(gamma0, s_g, n_part)

    s_ux = emit_x / s_x
    ux = s_ux * np.random.standard_normal(n_part)

    s_uy = emit_y / s_y
    uy = s_uy * np.random.standard_normal(n_part)

    uz = np.sqrt((gamma**2 - 1) - ux**2 - uy**2)

    if tf != 0.0:
        x = x - ux * c * tf / gamma
        y = y - uy * c * tf / gamma
        z = z - uz * c * tf / gamma

    q = np.ones(n_part) * q_tot / n_part

    return x, y, z, ux, uy, uz, q


def trapezoidal_bunch(
    i0,  # Initial current (at the beginning of the trapezoid)
    i1,  # Final current (at the end of the trapezoid)
    n_part,  # Number of particles in the bunch
    gamma0,  # Central value of the relativistic gamma factor
    s_g,  # Standard deviation of the gamma factor
    length,  # Length of the trapezoidal bunch
    s_z,  # Standard deviation of the longitudinal distribution
    emit_x,  # Normalized emittance in the x-direction
    s_x,  # Standard deviation of the x-position distribution
    emit_y,  # Normalized emittance in the y-direction
    s_y,  # Standard deviation of the y-position distribution
    zf=0.0,  # Final z position
    tf=0,  # Final time
    x_c=0.0,  # x-center position
    y_c=0.0,  # y-center position
):
    """Create a trapezoidal particle bunch."""
    n_part = int(n_part)  # Ensure the number of particles is an integer

    # Calculate charges for the plateau, triangular, and Gaussian sections of the bunch
    q_plat = (min(i0, i1) / c) * length
    q_triag = ((max(i0, i1) - min(i0, i1)) / c) * length / 2.0
    q_gaus0 = (i0 / c) * np.sqrt(2 * np.pi) * s_z / 2.0
    q_gaus1 = (i1 / c) * np.sqrt(2 * np.pi) * s_z / 2.0
    q_tot = q_plat + q_triag + q_gaus0 + q_gaus1  # Total charge

    # Determine the number of particles in each section
    n_plat = int(n_part * q_plat / q_tot)
    n_triag = int(n_part * q_triag / q_tot)
    n_gaus0 = int(n_part * q_gaus0 / q_tot)
    n_gaus1 = int(n_part * q_gaus1 / q_tot)

    np.random.seed(42)  # Seed for reproducibility
    z_plat = np.random.uniform(
        0.0, length, n_plat
    )  # Uniform distribution for plateau
    if i0 <= i1:
        z_triag = np.random.triangular(
            0.0, length, length, n_triag
        )  # Triangular distribution (rising)
    else:
        z_triag = np.random.triangular(
            0.0, 0.0, length, n_triag
        )  # Triangular distribution (falling)
    z_gaus0 = s_z * np.random.standard_normal(
        2 * n_gaus0
    )  # Gaussian distribution for initial current
    z_gaus1 = s_z * np.random.standard_normal(
        2 * n_gaus1
    )  # Gaussian distribution for final current

    # Concatenate distributions and adjust positions
    z = np.concatenate(
        (
            z_gaus0[np.where(z_gaus0 < 0)],
            z_plat,
            z_triag,
            z_gaus1[np.where(z_gaus1 > 0)] + length,
        )
    )

    z = z - length / 2.0 + zf  # Shift to final position

    n_part = len(z)  # Recalculate the number of particles
    x = x_c + s_x * np.random.standard_normal(
        n_part
    )  # x-positions with Gaussian spread
    y = y_c + s_y * np.random.standard_normal(
        n_part
    )  # y-positions with Gaussian spread

    gamma = np.random.normal(gamma0, s_g, n_part)  # Gamma distribution

    s_ux = emit_x / s_x  # x-momentum spread
    ux = s_ux * np.random.standard_normal(n_part)  # x-momenta

    s_uy = emit_y / s_y  # y-momentum spread
    uy = s_uy * np.random.standard_normal(n_part)  # y-momenta

    uz = np.sqrt(
        (gamma**2 - 1) - ux**2 - uy**2
    )  # z-momenta from relativistic relation

    if tf != 0.0:  # Adjust positions and momenta if final time is specified
        x = x - ux * c * tf / gamma
        y = y - uy * c * tf / gamma
        z = z - uz * c * tf / gamma

    q = np.ones(n_part) * q_tot / n_part  # Charge per particle

    return x, y, z, ux, uy, uz, q  # Return particle properties
