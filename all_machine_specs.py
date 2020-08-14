import os

"""
This file is part of the suite of scripts to use LibEnsemble on top of WarpX
simulations. It contains a dictionary for machine-specific elements.
"""


local_specs = {
    'name': 'local',  # Machine name
    'cores': 1,  # Number of cores per simulation
    'sim_app': 'python'
    'extra_args': '',  # extra arguments passed to mpirun/mpiexec at execution
    'NUMBA_NUM_THREADS': '4',
    'sim_max': 3 # Maximum number of simulations
}


summit_specs = {
    'name': 'summit',  # Machine name
    'sim_app': 'python'
    # extra arguments passed to jsrun at execution
    'extra_args': '-n 1 -a 1 -g 1 -c 1 --bind=packed:1',
    'OMP_NUM_THREADS': '1',
    'sim_max': 400 # Maximum number of simulations
}
