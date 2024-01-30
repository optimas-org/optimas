"""Defines the analysis function that runs after the simulation."""

import os

import numpy as np
import matplotlib.pyplot as plt
import visualpic as vp
from aptools.plotting.quick_diagnostics import (
    phase_space_overview,
    slice_analysis,
)


def analyze_simulation(simulation_directory, output_params):
    """Analyze the output of the simulation."""
    # Load data.
    diags_dir = os.path.join(simulation_directory, "diags/hdf5")
    dc = vp.DataContainer("openpmd", diags_dir)
    dc.load_data()

    # Get final bunch distribution.
    bunch = dc.get_species("bunch")
    ts = bunch.timesteps
    bunch_data = bunch.get_data(ts[-1])
    x = bunch_data["x"][0]
    y = bunch_data["y"][0]
    z = bunch_data["z"][0]
    px = bunch_data["px"][0]
    py = bunch_data["py"][0]
    pz = bunch_data["pz"][0]
    q = bunch_data["q"][0]

    # Remove particles with pz < 100
    pz_filter = np.where(pz >= 100)
    x = x[pz_filter]
    y = y[pz_filter]
    z = z[pz_filter]
    px = px[pz_filter]
    py = py[pz_filter]
    pz = pz[pz_filter]
    q = q[pz_filter]

    # Calculate relevant parameters.
    q_tot = np.abs(np.sum(q)) * 1e12  # pC
    q_ref = 10  # pC
    # ene = np.average(pz, weights=q) * 0.511  # MeV
    med, mad = weighted_mad(pz * 0.511, q)
    mad_rel = mad / med
    med *= 1e-3  # GeV
    mad_rel_ref = 1e-2

    # Calculate objective.
    f = np.log(med * q_tot / q_ref / (mad_rel / mad_rel_ref))

    # Store quantities in output.
    output_params["f"] = -f
    output_params["charge"] = q_tot
    output_params["energy_med"] = med
    output_params["energy_mad"] = mad

    # Save objective to file (for convenience).
    np.savetxt("f.txt", np.array([f]))

    # Make plots.
    try:
        plt.figure()
        slice_analysis(x, y, z, px, py, pz, q, show=False)
        plt.savefig("final_lon_phase_space.png")
        plt.figure()
        phase_space_overview(x, y, z, px, py, pz, q, show=False)
        plt.savefig("final_phase_space.png")
    except Exception:
        print("Failed to make plots.")

    # Remove all diagnostics except last file.
    try:
        for file in sorted(os.listdir(diags_dir))[:-1]:
            file_path = os.path.join(diags_dir, file)
            os.remove(file_path)
    except Exception:
        print("Could not remove diagnostics.")

    return output_params


def weighted_mad(x, w):
    """Calculate weighted median absolute deviation."""
    med = weighted_median(x, w)
    mad = weighted_median(np.abs(x - med), w)
    return med, mad


def weighted_median(data, weights):
    """Compute the weighted quantile of a 1D numpy array.

    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile_1D : float
        The output value.

    """
    quantile = 0.5
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if (quantile > 1.0) or (quantile < 0.0):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    # assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (Sn - 0.5 * sorted_weights) / Sn[-1]
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)
