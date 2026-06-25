"""Defines the analysis function that runs after the simulation."""

import os

import numpy as np
import matplotlib.pyplot as plt
import visualpic as vp
from aptools.plotting.quick_diagnostics import (
    phase_space_overview,
    slice_analysis,
)


def analyze_simulation(
    simulation_directory,
    output_params,
    ref_bunch_vals={"q_tot_pC": 200, "energy_spread": 5e-3},
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
    q = bunch_data["q"][0]

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

    # Calculate gamma directly from all particles
    gamma = np.sqrt(pz**2 + 1)

    # Compute statistics directly from particle distribution
    mean_gamma = np.average(gamma, weights=w)
    std_gamma = np.sqrt(np.average((gamma - mean_gamma) ** 2, weights=w))

    # Calculate relevant parameters.
    q_tot = np.abs(np.sum(q)) * 1e12  # total charge (pC)
    q_ref = ref_bunch_vals["q_tot_pC"]  # bunch charge (pC)

    energy_spread_ref = ref_bunch_vals["energy_spread"]
    energy_spread = std_gamma / mean_gamma

    # Calculate objective.
    f = np.log(mean_gamma * q_tot / q_ref / (energy_spread / energy_spread_ref))

    # Store quantities in output
    output_params["f"] = -f
    output_params["charge"] = q_tot
    output_params["mean_gamma"] = mean_gamma
    output_params["std_gamma"] = std_gamma

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
