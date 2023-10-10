"""Contains the definition of the simulation functions given to libEnsemble."""
import jinja2
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED

from optimas.utils.logger import get_logger

logger = get_logger(__name__)


def run_template_simulation(H, persis_info, sim_specs, libE_info):
    """Run a simulation defined with a `TemplateEvaluator` or `ChainEvaluator`.

    This function creates the simulation script from the template, launches
    the simulation and analyzes the results to obtain the value of the
    objectives and other analyzed parameters. Status check
    is done periodically on the simulation, provided by LibEnsemble.
    """
    # By default, indicate that task failed
    calc_status = TASK_FAILED

    input_values = {}
    for name in H.dtype.names:
        value = H[name][0]
        if isinstance(value, str):
            value = "'{}'".format(value)
        input_values[name] = value

    # Prepare the array that is returned to libE
    # Automatically include the input parameters
    libE_output = np.zeros(1, dtype=sim_specs["out"])
    for name in H.dtype.names:
        libE_output[name] = H[name][0]

    # Get user specs.
    if "task" in H.dtype.names:
        task_name = H["task"][0]
        user_specs = sim_specs["user"][task_name]
    else:
        user_specs = sim_specs["user"]

    # Get list of simulation steps. If no steps are defined (that is, a
    # ChainEvaluator is not being used), create a list with a single step.
    if "steps" in user_specs:
        simulation_step_specs = user_specs["steps"]
    else:
        simulation_step_specs = [user_specs]

    # Launch and analyze each simulation step.
    for step_specs in simulation_step_specs:
        calc_status = execute_and_analyze_simulation(
            app_name=step_specs["app_name"],
            sim_template=step_specs["sim_template"],
            input_values=input_values,
            analysis_func=step_specs["analysis_func"],
            libE_output=libE_output,
            num_procs=step_specs["num_procs"],
            num_gpus=step_specs["num_gpus"],
            env_script=step_specs["env_script"],
            mpi_runner_type=step_specs["env_mpi"],
        )
        # If a step has failed, do not continue with next steps.
        if calc_status != WORKER_DONE:
            break

    return libE_output, persis_info, calc_status


def execute_and_analyze_simulation(
    app_name,
    sim_template,
    input_values,
    analysis_func,
    libE_output,
    num_procs,
    num_gpus,
    env_script,
    mpi_runner_type,
):
    """Run simulation, handle outcome and analyze results."""
    # Create simulation input file.
    with open(sim_template, "r") as f:
        template = jinja2.Template(f.read())
    with open(sim_template, "w") as f:
        f.write(template.render(input_values))

    # If the template is a python file, no need to provide it as argument
    # (it has already been registered by libEnsemble as such).
    if sim_template.endswith(".py"):
        sim_template = None

    # Launch simulation.
    task = Executor.executor.submit(
        app_name=app_name,
        app_args=sim_template,
        stdout="out.txt",
        stderr="err.txt",
        num_procs=num_procs,
        num_gpus=num_gpus,
        env_script=env_script,
        mpi_runner_type=mpi_runner_type,
    )

    # Wait for simulation to complete
    task.wait()

    # Set calc_status with optional prints.
    if task.finished:
        if task.state == "FINISHED":
            calc_status = WORKER_DONE
        elif task.state == "FAILED":
            calc_status = TASK_FAILED
        if task.state not in ["FINISHED", "FAILED", "USER_KILLED"]:
            print(
                "Warning: Task {} in unknown state {}. Error code {}".format(
                    task.name, task.state, task.errcode
                )
            )

    # Data analysis from the last simulation
    if calc_status == WORKER_DONE:
        if analysis_func is not None:
            # Extract the objective function for the current simulation,
            # as well as a few diagnostics
            analysis_func(task.workdir, libE_output)

    return calc_status


def run_function(H, persis_info, sim_specs, libE_info):
    """Run an evaluation defined with a `FunctionEvaluator`."""
    # By default, indicate that task failed
    calc_status = TASK_FAILED

    input_values = {}
    for name in H.dtype.names:
        value = H[name][0]
        if isinstance(value, str):
            value = "'{}'".format(value)
        input_values[name] = value

    # Get user specs.
    if "task" in H.dtype.names:
        task_name = H["task"][0]
        user_specs = sim_specs["user"][task_name]
    else:
        user_specs = sim_specs["user"]
    evaluation_func = user_specs["evaluation_func"]

    # Prepare the array that is returned to libE
    # Automatically include the input parameters
    libE_output = np.zeros(1, dtype=sim_specs["out"])
    for name in H.dtype.names:
        libE_output[name] = H[name][0]

    evaluation_func(input_values, libE_output)

    return libE_output, persis_info, calc_status
