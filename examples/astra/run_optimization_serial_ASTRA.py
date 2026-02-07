"""Example of multiobjective Bayesian optimization using the serial ASTRA version.

This example uses an ASTRA example from the ASTRA webpage:
https://www.desy.de/~mpyflo/EXAMPLES/Manual_Example/
In order to run this example, please download the files "3_cell_L-Band.dat",
"Solenoid.dat", and create the particle distribution "Example.ini" by running
the generator.in file.
Further optimas documentation and examples:
https://optimas.readthedocs.io/en/latest/index.html
"""

from optimas.generators import AxSingleFidelityGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration
from gest_api.vocs import VOCS
from analysis_script import analyze_simulation

# Create VOCS object.
# name of parameter, lower bound of values to be explored,
# upper bound of values to be explored
vocs = VOCS(
    variables={
        "RF_phase": [-2.5, 2.5],
        "B_sol": [0.12, 0.38],
    },
    objectives={
        "bunch_length": "MINIMIZE",
        "emittance": "MINIMIZE",
    },
    observables=["emittance_x", "emittance_y"],
)

# Create generator.
# Pick the generator to be used, here Single-fidelity Bayesian optimization.
gen = AxSingleFidelityGenerator(
    vocs=vocs,
    n_init=8,
)


# Create evaluator.
# sim_template is the ASTRA input template, here the parameters that are going
# to be varied need to be changed to the name given in var_1 etc. The format
# in the ASTRA input is e.g. Phi(1)={{RF_phase}}.
# analysis_func is the function that will analyze the output.
# sim_files contains the path to the particle distribution and other files
# needed for the ASTRA simulation like field maps.
# executable is the path to the ASTRA executable
ev = TemplateEvaluator(
    sim_template="ASTRA_example.in",
    analysis_func=analyze_simulation,
    sim_files=[
        "Example.ini",
        "3_cell_L-Band.dat",
        "Solenoid.dat",
    ],
    executable="/path_to_ASTRA/Astra",
    n_procs=1,
)


# Create exploration.
# max_evals is the maximum number of evaluations.
# max_evalvs / sim_worker is the number of simulation batches that are sent.
# sim_workers is the number of simulations that are launched in parallel.
# sim_workers should be smaller than the number of available CPU cores.
# In case you already have some data from an optimization run but would like
# to add further datapoints, set resume=True and max_evals to a higher number.
exp = Exploration(
    generator=gen, evaluator=ev, max_evals=500, sim_workers=8, resume=False
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
