"""Example multifidelity Bayesian optimization of an LPA with FBPIC.

This example optimizes an LPA based on ionization injection using FBPIC
simulations with varying fidelity (resolution).

The FBPIC simulations are performed using the template defined in the
`template_simulation_script.py` file.

In addition to the objective `f`, three additional parameters
are analyzed for each simulation and including in the optimization
history. The calculation of `f` and the additional parameters is performed
in the `analyze_simulation` function, which for convenience is here defined in
the `analysis_script.py` file.
"""

from optimas.generators import AxMultiFidelityGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration
from gest_api.vocs import VOCS

from analysis_script import analyze_simulation


# Create VOCS object.
vocs = VOCS(
    variables={
        "laser_scale": [0.7, 1.05],
        "z_foc": [3.0, 7.5],
        "mult": [0.1, 1.5],
        "plasma_scale": [0.6, 0.8],
        "resolution": [2.0, 4.0],
    },
    objectives={"f": "MINIMIZE"},
    observables=["energy_med", "energy_mad", "charge"],
    fidelity_parameter="resolution",
    fidelity_target_value=4.0,
)


# Create generator.
gen = AxMultiFidelityGenerator(
    vocs=vocs,
    n_init=4,
)


# Create evaluator.
ev = TemplateEvaluator(
    sim_template="template_simulation_script.py",
    analysis_func=analyze_simulation,
    n_gpus=1,  # Use 1 GPU per simulation.
)


# Create exploration.
exp = Exploration(
    generator=gen, evaluator=ev, max_evals=100, sim_workers=4, run_async=True
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
