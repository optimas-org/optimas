"""Basic example of parallel Bayesian optimization with Ax using the serial
ASTRA version.
This example uses an ASTRA example from the ASTRA webpage:
https://www.desy.de/~mpyflo/EXAMPLES/Manual_Example/
In order to run this example, please download the files "3_cell_L-Band.dat",
"Solenoid.dat", and create the particle distribution "Example.ini" by running
the generator.in file.
Further optimas documentation and examples:
https://optimas.readthedocs.io/en/latest/index.html
"""

from optimas.core import VaryingParameter, Objective, Parameter
from optimas.generators import AxSingleFidelityGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration
from analysis_script import analyze_simulation

# Create varying parameters and objectives.
# name of parameter, lower bound of values to be explored,
# upper bound of values to be explored
var_1 = VaryingParameter("RF_phase", -2.5, 2.5)
var_2 = VaryingParameter("B_sol", 0.12, 0.38)
# Objective that will be minimized:
obj_1 = Objective("bunch_length", minimize=True)
obj_2 = Objective("emittance", minimize=True)
# Additional example parameters that will be analyzed but are not used for the
# optimization:
em_x = Parameter("emittance_x")
em_y = Parameter("emittance_y")

# Create generator.
# Pick the generator to be used, here Single-fidelity Bayesian optimization
# The analyzed_parameters are parameters that are calculated for each
# simulation but not used for the optimization.
gen = AxSingleFidelityGenerator(
    varying_parameters=[var_1, var_2],
    objectives=[obj_1, obj_2],
    n_init=8,
    analyzed_parameters=[em_x, em_y],
)


# Create evaluator.
# sim_template is the ASTRA input template, here the parameters that are going
# to # be varied need to be changed to the name given in var_1 etc. The format
# in the # ASTRA input is e.g. Phi(1)={{RF_phase}}.
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
# max_evals is the maximal number of evaluations
# max_evalvs / sim_worker is the number of simulation batches that are sent
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
