"""
Contains the function that analyzes the simulation results,
after the simulation was run.
"""
# This must include all quantities calculated by this script, except from f
# These parameters are not used by libEnsemble, but they provide additional
# information / diagnostic for the user
# The third parameter is the shape of the corresponding array
analyzed_quantities = []


def analyze_simulation( simulation_directory, libE_output ):

    # Read back result from file
    with open('result.txt') as f:
        result = float( f.read() )

    libE_output['f'] = result

    return libE_output
