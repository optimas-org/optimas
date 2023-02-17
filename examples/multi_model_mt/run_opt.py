"""
This example optimizes an LPA based on ionization injection using FBPIC
simulations.

The FBPIC simulations are performed using the template defined in the
`template_simulation_script.py` file.

In addition to the objective `f`, three additional parameters
are analyzed for each simulation and including in the optimization
history. The calculation of `f` and the additional parameters is performed
in the `analyze_simulation` function, which for convenience is here defined in
the `analysis_script.py` file.
"""
from optimas.core import VaryingParameter, Objective, Task, Parameter
from optimas.generators import AxMultitaskGenerator
from optimas.evaluators import TemplateEvaluator, MultitaskEvaluator
from optimas.explorations import Exploration

from analysis_script import analyze_simulation


# Create varying parameters and objectives.
var_1 = VaryingParameter('beam_i_1', 0.1, 10.)
var_2 = VaryingParameter('beam_i_2', 0.1, 10.)
var_3 = VaryingParameter('beam_z_i_2', -10., 10.)
var_4 = VaryingParameter('beam_length', 1., 20.)
obj = Objective('f', minimize=True)


# Define additional parameters to analyze.
energy_med = Parameter('energy_med')
energy_mad = Parameter('energy_mad')
charge = Parameter('charge')


# Create tasks.
lofi_task = Task('wake-t', n_init=20, n_opt=20)
hifi_task = Task('fbpic', n_init=4, n_opt=4)


# Create generator.
gen = AxMultitaskGenerator(
    varying_parameters=[var_1, var_2, var_3, var_4],
    objectives=[obj],
    analyzed_parameters=[energy_med, energy_mad, charge],
    lofi_task=lofi_task,
    hifi_task=hifi_task
)


# Create one evaluator for each task. In this example, both tasks use the same
# template, but in principle they can have different template, executor,
# analysis function, resources, etc.
ev_lofi = TemplateEvaluator(
    sim_template='template_simulation_script.py',
    analysis_func=analyze_simulation,
    sim_files=['bunch_utils.py', 'custom_fld_diags.py', 'custom_ptcl_diags.py']
)
ev_hifi = TemplateEvaluator(
    sim_template='template_simulation_script.py',
    analysis_func=analyze_simulation,
    sim_files=['bunch_utils.py', 'custom_fld_diags.py', 'custom_ptcl_diags.py']
)


# Create a multitask evaluator. This associates each task to each task
# evaluator.
ev = MultitaskEvaluator(
    tasks=[lofi_task, hifi_task],
    task_evaluators=[ev_lofi, ev_hifi]
)


# Create exploration.
n_iter = 10  # Number of multitask BO iterations to perform.
exp = Exploration(
    generator=gen,
    evaluator=ev,
    max_evals=(
        (lofi_task.n_opt + hifi_task.n_opt) * n_iter
        + lofi_task.n_init + hifi_task.n_init
    ),
    sim_workers=20
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == '__main__':
    exp.run()
