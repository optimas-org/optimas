"""
Contains the function that analyzes the simulation results,
after the simulation was run.
"""
import numpy as np


def analyze_simulation(simulation_directory, output_params):
    a_x_abs = np.load('a_x_abs.npy')
    output_params['f'] = a_x_abs

    return output_params
