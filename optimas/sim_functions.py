import os

import jinja2
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.executors.mpi_runner import MPIRunner
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED

from optimas.utils.logger import get_logger

logger = get_logger(__name__)


def run_template_simulation(H, persis_info, sim_specs, libE_info):
    """
    This function runs a simulation and returns quantity 'f' as well as
    other physical quantities measured in the run for convenience. Status check
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

    # Get user specs.
    if 'task' in H.dtype.names:
        task_name = H['task'][0]
        user_specs = sim_specs['user'][task_name]
    else:
        user_specs = sim_specs['user']
    sim_template = user_specs['sim_template']
    analysis_func = user_specs['analysis_func']
    app_name = user_specs['app_name']

    # Create simulation input file.
    sim_script = sim_template[len('template_'):]  # Strip 'template_' from name
    with open(sim_template, 'r') as f:
        template = jinja2.Template(f.read())
    with open(sim_script, 'w') as f:
        f.write(template.render(input_values))
    os.remove(sim_template)

    # If the template is a python file, no need to provide it as argument
    # (it has already been registered by libEnsemble as such).
    if sim_script.endswith('.py'):
        sim_script = None

    # Passed to command line in addition to the executable.
    exctr = Executor.executor  # Get Executor

    # If the evaluation requires a specific MPI (different than the one in
    # the optimas environment), comunicate this to the executor.
    mpi_type = user_specs['env_mpi']
    if mpi_type is not None:
        old_runner_type = exctr.mpi_runner_type
        old_runner = exctr.mpi_runner
        exctr.mpi_runner_type = mpi_type
        exctr.mpi_runner = MPIRunner.get_runner(exctr.mpi_runner_type)

    # Launch simulation.
    task = exctr.submit(
        app_name=app_name,
        app_args=sim_script,
        stdout='out.txt',
        stderr='err.txt',
        env_script=user_specs['env_script']
    )

    # Wait for simulation to complete
    task.wait()

    # Set calc_status with optional prints.
    if task.finished:
        if task.state == 'FINISHED':
            calc_status = WORKER_DONE
        elif task.state == 'FAILED':
            calc_status = TASK_FAILED
        if task.state not in ['FINISHED', 'FAILED', 'USER_KILLED']:
            print("Warning: Task {} in unknown state {}. Error code {}"
                  .format(task.name, task.state, task.errcode))

    # If the mpi runner was changed, set it back to the original value for
    # future simulations.
    if mpi_type is not None:
        exctr.mpi_runner_type = old_runner_type
        exctr.mpi_runner = old_runner

    # Prepare the array that is returned to libE
    # Automatically include the input parameters
    libE_output = np.zeros(1, dtype=sim_specs['out'])
    for name in H.dtype.names:
        libE_output[name] = H[name][0]

    # Data analysis from the last simulation
    if calc_status == WORKER_DONE:
        # Extract the objective function for the current simulation,
        # as well as a few diagnostics
        analysis_func(task.workdir, libE_output)

    return libE_output, persis_info, calc_status


def run_function(H, persis_info, sim_specs, libE_info):
    """
    This function runs a simulation and returns quantity 'f' as well as
    other physical quantities measured in the run for convenience. Status check
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

    # Get user specs.
    if 'task' in H.dtype.names:
        task_name = H['task'][0]
        user_specs = sim_specs['user'][task_name]
    else:
        user_specs = sim_specs['user']
    evaluation_func = user_specs['evaluation_func']

    # Prepare the array that is returned to libE
    # Automatically include the input parameters
    libE_output = np.zeros(1, dtype=sim_specs['out'])
    for name in H.dtype.names:
        libE_output[name] = H[name][0]

    evaluation_func(input_values, libE_output)

    return libE_output, persis_info, calc_status
