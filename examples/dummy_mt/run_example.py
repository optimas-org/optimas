from libe_opt.core import VaryingParameter, Objective, Task
from libe_opt.generators import AxMultitaskGenerator
from libe_opt.evaluators import TemplateEvaluator, MultitaskEvaluator
from libe_opt.explorations import Exploration


def analyze_simulation(simulation_directory, output_params):
    """Function that analyzes the simulation output and fills in the
    dictionary of output parameters."""
    # Read back result from file
    with open('result.txt') as f:
        result = float( f.read() )
    # Fill in output parameters.
    output_params['f'] = result
    return output_params


# Create varying parameters and objectives.
var_1 = VaryingParameter('x0', 0., 15.)
var_2 = VaryingParameter('x1', 0., 15.)
obj = Objective('f', minimize=True)


# Create tasks.
lofi_task = Task('cheap_model', n_init=10, n_opt=3)
hifi_task = Task('expensive_model', n_init=2, n_opt=1)


# Create generator.
gen = AxMultitaskGenerator(
    varying_parameters=[var_1, var_2],
    objectives=[obj],
    lofi_task=lofi_task,
    hifi_task=hifi_task
)


# Create one evaluator for each task. In this example, both tasks use the same
# template, but in principle they can have different template, executor,
# analysis function, resources, etc.
ev_lofi = TemplateEvaluator(
    sim_template='template_simulation_script.py',
    analysis_func=analyze_simulation
)
ev_hifi = TemplateEvaluator(
    sim_template='template_simulation_script.py',
    analysis_func=analyze_simulation
)


# Create a multitask evaluator. This associates each task to each task
# evaluator.
ev = MultitaskEvaluator(
    tasks=[lofi_task, hifi_task],
    task_evaluators=[ev_lofi, ev_hifi]
)


# Create exploration.
exp = Exploration(
    generator=gen,
    evaluator=ev,
    max_evals=10,
    sim_workers=4,
    run_async=True
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == '__main__':
    exp.run()
