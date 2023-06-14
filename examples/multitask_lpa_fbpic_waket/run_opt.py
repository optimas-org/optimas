from multiprocessing import set_start_method

from optimas.core import VaryingParameter, Objective, Parameter, Task
from optimas.generators import AxMultitaskGenerator
from optimas.evaluators import TemplateEvaluator, MultitaskEvaluator
from optimas.explorations import Exploration

from analysis_script import analyze_simulation


# Create varying parameters and objectives.
var_1 = VaryingParameter('beam_i_1', 1., 10.)  # kA
var_2 = VaryingParameter('beam_i_2', 1., 10.)  # kA
var_3 = VaryingParameter('beam_z_i_2', -10., 10.)  # µm
var_4 = VaryingParameter('beam_length', 1., 20.)  # µm
obj = Objective('f', minimize=True)


# Define other quantities to analyze (which are not the optimization objective)
par_1 = Parameter('energy_med')
par_2 = Parameter('energy_mad')
par_3 = Parameter('charge')


# Create tasks.
lofi_task = Task('wake-t', n_init=96, n_opt=96)
hifi_task = Task('fbpic', n_init=3, n_opt=3)


# Create generator.
gen = AxMultitaskGenerator(
    varying_parameters=[var_1, var_2, var_3, var_4],
    objectives=[obj],
    analyzed_parameters=[par_1, par_2, par_3],
    use_cuda=True,
    hifi_task=hifi_task,
    lofi_task=lofi_task
)


# Create evaluators for each task.
ev_lofi = TemplateEvaluator(
    sim_template='template_simulation_script.py',
    analysis_func=analyze_simulation,
    sim_files=[
        'bunch_utils.py',
        'custom_fld_diags.py',
        'custom_ptcl_diags.py'
    ]
)
ev_hifi = TemplateEvaluator(
    sim_template='template_simulation_script.py',
    analysis_func=analyze_simulation,
    sim_files=[
        'bunch_utils.py',
        'custom_fld_diags.py',
        'custom_ptcl_diags.py'
    ]
)

# Create a multitask evaluator. This associates each task to each task
# evaluator.
ev = MultitaskEvaluator(
    tasks=[lofi_task, hifi_task],
    task_evaluators=[ev_lofi, ev_hifi]
)

# Create exploration.
n_batches = 50
exp = Exploration(
    generator=gen,
    evaluator=ev,
    max_evals=(
        (lofi_task.n_opt + hifi_task.n_opt) * n_batches
        + lofi_task.n_init + hifi_task.n_init
    ),
    sim_workers=96,
    run_async=False,
)

# Run exploration.
if __name__ == '__main__':
    set_start_method('spawn')
    exp.run()
