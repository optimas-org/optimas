"""Defines the analysis function that runs after the simulation."""

import os

import numpy as np
import matplotlib.pyplot as plt
import visualpic as vp
from aptools.plotting.quick_diagnostics import (
    phase_space_overview,
    slice_analysis,
)


def bin_and_analyze_particles(z, pz, w, num_bins):
    """
    Bin particles based on their longitudinal positions and perform analysis on each bin.

    Parameters
    ----------
    z : array
        Array of longitudinal positions of the particles.
    pz : array
        Array of longitudinal momentum of the particles.
    w : array
        Array of weights of the particles.
    num_bins : int
        Number of bins to divide the particles into.

    Returns
    -------
    gamma_avgs : list
        List of dictionaries containing the averages for each bin.
    """
    # Create bins
    bin_nparts, bin_edges = np.histogram(z, bins=num_bins, weights=w)

    # Find the bin indices for each particle
    bin_indices = np.digitize(z, bin_edges) - 1

    # Calculate particle gamma from longitudinal momentum (almost irrelevant since ultra-relativistic)
    gamma = np.sqrt(pz**2 + 1)

    # Initialize list to hold the results
    bin_gammas = []

    # Analyze each bin
    for i in range(num_bins):
        # Select particles in the current bin
        mask = bin_indices == i
        bin_gamma = gamma[mask]
        bin_w = w[mask]

        try:
            # Compute average and particle number per bin
            avg_gamma = np.average(bin_gamma, weights=bin_w)
        except:
            avg_gamma = 0  # invalid for when all weights in the bin are zero

        # Store the results
        bin_gammas.append(avg_gamma)

    bin_gammas = np.array(bin_gammas)
    # Compute mean gamma among all beams

    try:
        # Compute mean gamma among all beams
        mean_gamma = np.average(bin_gammas, weights=bin_nparts)
    except:
        mean_gamma = 0  # invalid for when all weights in the bin are zero

    # Compute standard deviations
    try:
        std_gamma = np.sqrt(
            np.average((bin_gammas - mean_gamma) ** 2, weights=bin_nparts)
        )
    except:
        std_gamma = 9e9  # invalid for when all weights in the bin are zero

    return mean_gamma, std_gamma, bin_gammas, bin_nparts


def analyze_simulation(
    simulation_directory,
    output_params,
    ref_bunch_vals={"q_tot_pC": 200, "energy_spread": 5e-3},
    num_bins=10,
    make_plots=True,
    remove_all_diags_but_last=True,
):
    """Analyze the output of the simulation."""
    # Load data.
    diags_dir = os.path.join(simulation_directory, "diags/hdf5")
    dc = vp.DataContainer("openpmd", diags_dir)
    dc.load_data()

    # Get final bunch distribution.
    bunch = dc.get_species("bunch")
    ts = bunch.timesteps
    bunch_data = bunch.get_data(ts[-1])
    w = bunch_data["w"][0]
    x = bunch_data["x"][0]
    y = bunch_data["y"][0]
    z = bunch_data["z"][0]
    px = bunch_data["px"][0]
    py = bunch_data["py"][0]
    pz = bunch_data["pz"][0]
    q = bunch_data["q"][0]  # charge is already weighted, apparently

    # Remove particles with pz < 100
    pz_filter = np.where(pz >= 100)
    w = w[pz_filter]
    x = x[pz_filter]
    y = y[pz_filter]
    z = z[pz_filter]
    px = px[pz_filter]
    py = py[pz_filter]
    pz = pz[pz_filter]
    q = q[pz_filter]

    # Bin and analyze the particles
    mean_gamma, std_gamma, bin_averages, bin_nparts = bin_and_analyze_particles(
        z, pz, w, num_bins
    )

    # Calculate relevant parameters.
    q_tot = np.abs(np.sum(q)) * 1e12  # total charge (pC)
    q_ref = ref_bunch_vals["q_tot_pC"]  # bunch charge (pC) : 200 pC by default

    energy_spread_ref = ref_bunch_vals["energy_spread"]
    energy_spread = std_gamma / mean_gamma

    # Calculate objective.
    f = np.log(mean_gamma * q_tot / q_ref / (energy_spread / energy_spread_ref))

    # Store quantities in output
    output_params["f"] = -f
    output_params["charge"] = q_tot
    output_params["mean_gamma"] = mean_gamma
    output_params["std_gamma"] = std_gamma
    output_params["charge"] = (
        q_tot  # Ensure q_tot is defined and correct  #SH duplicate line....
    )
    # Add other parameters as needed
    for i in range(num_bins):
        output_params[f"bin_gammas_{i+1}"] = bin_averages[i]
        output_params[f"bin_nparts_{i+1}"] = bin_nparts[i]

    # Save objective to file (for convenience).
    np.savetxt("f.txt", np.array([f]))

    # Make plots.
    if make_plots:
        try:
            plt.figure()
            slice_analysis(x, y, z, px, py, pz, q, show=False)
            plt.savefig("final_lon_phase_space.png")
            plt.figure()
            phase_space_overview(x, y, z, px, py, pz, q, show=False)
            plt.savefig("final_phase_space.png")
        except Exception:
            print("Failed to make plots.")

    if remove_all_diags_but_last:
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
    mad = weighted_median(np.abs(x - med), w, quantile=0.5)
    return med, mad


def weighted_median(data, weights, quantile=0.5):
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
