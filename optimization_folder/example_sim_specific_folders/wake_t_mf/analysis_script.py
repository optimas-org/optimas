"""
Contains the function that analyzes the simulation results,
after the simulation was run.
"""
# This must include all quantities calculated by this script, except from f
# These parameters are not used by libEnsemble, but they provide additional
# information / diagnostic for the user
# The third parameter is the shape of the corresponding array
analyzed_quantities = []


import numpy as np


def analyze_simulation( simulation_directory, libE_output ):
    g_lens = libE_output['g_lens'][0][0]
    file_path = 'a_x_abs-{:0.3f}.out'.format(g_lens)
    a_x_abs = np.loadtxt(file_path)
    libE_output['f'] = a_x_abs

    return libE_output
