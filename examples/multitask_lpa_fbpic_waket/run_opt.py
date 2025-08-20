"""Multitask optimization of an LPA with Wake-T and FBPIC."""

from multiprocessing import set_start_method

from optimas.core import Task
from optimas.generators import AxMultitaskGenerator
from optimas.evaluators import TemplateEvaluator, MultitaskEvaluator
from optimas.explorations import Exploration
from generator_standard.vocs import VOCS

from analysis_script import analyze_simulation


# Create VOCS object.
vocs = VOCS(
    variables={
        "beam_i_1": [1.0, 10.0],  # kA
        "beam_i_2": [1.0, 10.0],  # kA
        "beam_z_i_2": [-10.0, 10.0],  # µm
        "beam_length": [1.0, 20.0],  # µm
        "trial_type": {"wake-t", "fbpic"},
    },
    objectives={"f": "MINIMIZE"},
    observables=["energy_med", "energy_mad", "charge"],
)


# Create tasks.
lofi_task = Task("wake-t", n_init=96, n_opt=96)
hifi_task = Task("fbpic", n_init=3, n_opt=3)


# Create generator.
gen = AxMultitaskGenerator(
    vocs=vocs,
    use_cuda=True,
    dedicated_resources=True,
    hifi_task=hifi_task,
    lofi_task=lofi_task,
)


# Create evaluators for each task.
ev_lofi = TemplateEvaluator(
    sim_template="template_simulation_script.py",
    analysis_func=analyze_simulation,
    sim_files=["bunch_utils.py", "custom_fld_diags.py", "custom_ptcl_diags.py"],
)
ev_hifi = TemplateEvaluator(
    sim_template="template_simulation_script.py",
    analysis_func=analyze_simulation,
    sim_files=["bunch_utils.py", "custom_fld_diags.py", "custom_ptcl_diags.py"],
)

# Create a multitask evaluator. This associates each task to each task
# evaluator.
ev = MultitaskEvaluator(
    tasks=[lofi_task, hifi_task], task_evaluators=[ev_lofi, ev_hifi]
)

# Create exploration.
n_batches = 50
exp = Exploration(
    generator=gen,
    evaluator=ev,
    max_evals=(
        (lofi_task.n_opt + hifi_task.n_opt) * n_batches
        + lofi_task.n_init
        + hifi_task.n_init
    ),
    sim_workers=96,
    run_async=False,
)

# Run exploration.
if __name__ == "__main__":
    set_start_method("spawn")
    exp.run()
