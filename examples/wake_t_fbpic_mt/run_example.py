"""Example multitask optimization with Wake-T and FBPIC.

This example optimizes the focusing strength of an active plasma lens using
Wake-T and FBPIC simulations by employing a multitask generator.

The simulations are performed using the template defined in the
`template_simulation_script.py` file, which executes Wake-T or FBPIC
depending on the value of the `task` parameter.

The calculation of the objective `f` is performed in the `analyze_simulation`
function, which for convenience is here defined in the `analysis_script.py`
file.
"""
from optimas.core import VaryingParameter, Objective, Task
from optimas.generators import AxMultitaskGenerator
from optimas.evaluators import TemplateEvaluator, MultitaskEvaluator
from optimas.explorations import Exploration

from analysis_script import analyze_simulation


# Create varying parameters and objectives.
var_1 = VaryingParameter("g_lens", 100.0, 1000.0)
obj = Objective("f", minimize=True)

# Create tasks.
lofi_task = Task("wake-t", n_init=12, n_opt=12)
hifi_task = Task("fbpic", n_init=2, n_opt=2)


# Create generator.
gen = AxMultitaskGenerator(
    varying_parameters=[var_1],
    objectives=[obj],
    lofi_task=lofi_task,
    hifi_task=hifi_task,
)


# Create one evaluator for each task. In this example, both tasks use the same
# template, but in principle they can have different template, executor,
# analysis function, resources, etc.
ev_lofi = TemplateEvaluator(
    sim_template="template_simulation_script.py",
    analysis_func=analyze_simulation,
)
ev_hifi = TemplateEvaluator(
    sim_template="template_simulation_script.py",
    analysis_func=analyze_simulation,
    n_gpus=2,
)


# Create a multitask evaluator. This associates each task to each task
# evaluator.
ev = MultitaskEvaluator(
    tasks=[lofi_task, hifi_task], task_evaluators=[ev_lofi, ev_hifi]
)


# Create exploration.
exp = Exploration(generator=gen, evaluator=ev, max_evals=100, sim_workers=12)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
