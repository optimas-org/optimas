"""Example Bayesian optimization with Wake-T.

This example optimizes the focusing strength of an active plasma lens using
Wake-T simulations.

The Wake-T simulations are performed using the template defined in the
`template_simulation_script.py` file.

The calculation of the objective `f` is performed in the `analyze_simulation`
function, which for convenience is here defined in the `analysis_script.py`
file.
"""

from optimas.generators import AxSingleFidelityGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration
from gest_api.vocs import VOCS

from analysis_script import analyze_simulation

# Create VOCS object.
vocs = VOCS(
    variables={
        "g_lens": [100.0, 1000.0],
    },
    objectives={"f": "MINIMIZE"},
)


# Create generator.
gen = AxSingleFidelityGenerator(vocs=vocs, n_init=12)


# Create evaluator.
ev = TemplateEvaluator(
    sim_template="template_simulation_script.py",
    analysis_func=analyze_simulation,
)


# Create exploration.
exp = Exploration(
    generator=gen, evaluator=ev, max_evals=100, sim_workers=12, run_async=False
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
