"""
This example optimizes an multistage setup using Warp-X.

The Warp-X simulations are performed using the template defined in the
`template_simulation_script` file.

In addition to the objective `f`, four additional parameters
are analyzed for each simulation and included in the optimization
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
var_1 = VaryingParameter("adjust_factor", 0.7, 1.05)
var_2 = VaryingParameter("zlen", 3.0, 7.5)
obj = Objective("f", minimize=True)


# Define additional parameters to analyze.
energy_std = Parameter("energy_std")
energy_avg = Parameter("energy_avg")
charge = Parameter("charge")
emittance = Parameter("emittance")


# Create generator.
gen = AxSingleFidelityGenerator(
    varying_parameters=[var_1, var_2],
    objectives=[obj],
    analyzed_parameters=[energy_std, energy_avg, charge, emittance],
    n_init=4,
)


# Create evaluator.
ev = TemplateEvaluator(
    sim_template="template_simulation_script",
    analysis_func=analyze_simulation,
    executable="warpx",
    n_gpus=1,
)


# Create exploration.
exp = Exploration(
    generator=gen, evaluator=ev, max_evals=1000, sim_workers=2, run_async=True
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
