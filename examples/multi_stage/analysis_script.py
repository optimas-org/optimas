"""Defines the analysis function that runs after the simulation."""

import os

import numpy as np
from openpmd_viewer.addons import LpaDiagnostics


def get_emittance(ts, t):
    """Calculate the beam emittance at the given time step."""
    w, x, ux = ts.get_particle(["w", "x", "ux"], t=t)
    x2 = np.average(x**2, weights=w)
    u2 = np.average(ux**2, weights=w)
    xu = np.average(x * ux, weights=w)
    return np.sqrt(x2 * u2 - xu**2)


def analyze_simulation(simulation_directory, output_params):
    """Analyze the output of the WarpX simulation.

    The function calculates the objective function 'f' as well as the
    diagnostic quantities listed as `analyzed_parameters` in the generator.
    """
    ts = LpaDiagnostics(os.path.join(simulation_directory, "diag"))
    t0 = 4.0e-11  # Time (boosted-frame) at which we calculate beam properties.

    charge_i = ts.get_charge(t=0)
    emittance_i = get_emittance(ts, t=0)
    charge_f = ts.get_charge(t=t0)
    emittance_f = get_emittance(ts, t=t0)
    energy_avg, energy_std = ts.get_mean_gamma(t=t0)

    # Here: Build a quantity to minimize (f) that encompasses
    # emittance AND charge loss 1% charge loss has the
    # same impact as doubling the initial emittance.
    # we minimize f!
    output_params["f"] = np.log(
        emittance_f + emittance_i * (1.0 - charge_f / charge_i) * 100
    )
    output_params["energy_std"] = energy_std
    output_params["energy_avg"] = energy_avg
    output_params["charge"] = charge_f
    output_params["emittance"] = emittance_f

    return output_params
