"""
This example optimizes a PWFA stage using HiPACE++.

The HiPACE++ simulations are performed using the template defined in the
`template_simulation_script` file.

In addition to the objective `f`, three additional parameters
are analyzed for each simulation and including in the optimization
history. The calculation of `f` and the additional parameters is performed
in the `analyze_simulation` function, which for convenience is here defined in
the `analysis_script.py` file.
"""
from optimas.core import Parameter, VaryingParameter, Objective
from optimas.generators import AxSingleFidelityGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration

from analysis_script import analyze_simulation


# Create varying parameters and objectives.
var_1 = VaryingParameter('witness_charge', 0.05, 1.)
obj = Objective('f', minimize=False)


# Define additional parameters to analyze.
energy_med = Parameter('energy_med')
energy_mad = Parameter('energy_mad')
charge = Parameter('charge')


# Create generator.
gen = AxSingleFidelityGenerator(
    varying_parameters=[var_1],
    objectives=[obj],
    analyzed_parameters=[energy_med, energy_mad, charge],
    n_init=4
)


# Create evaluator.
ev = TemplateEvaluator(
    sim_template='template_simulation_script',
    analysis_func=analyze_simulation,
    executable='/path/to/build/bin/hipace',
    n_gpus=2,  # Use 2 GPUs per simulation.
    # Uncomment if HiPACE is installed in a different enviroment than optimas.
    # env_script='/path/to/profile.hipace',
    # Uncomment if `env_script` loads a different MPI to that used by optimas.
    # env_mpi='openmpi'
)


# Create exploration.
exp = Exploration(
    generator=gen,
    evaluator=ev,
    max_evals=100,
    sim_workers=2,
    run_async=True
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == '__main__':
    exp.run()
