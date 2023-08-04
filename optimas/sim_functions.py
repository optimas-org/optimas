import os
import jinja2
import numpy as np

from libensemble.executors.executor import Executor
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

    # Prepare the array that is returned to libE
    # Automatically include the input parameters
    libE_output = np.zeros(1, dtype=sim_specs['out'])
    for name in H.dtype.names:
        libE_output[name] = H[name][0]

    # Get user specs.
    if 'task' in H.dtype.names:
        task_name = H['task'][0]
        user_specs = sim_specs['user'][task_name]
    else:
        user_specs = sim_specs['user']

    # Launch and analyze the simulation(s).
    if 'steps' in user_specs:
        for step in user_specs['steps']:
            step_specs = user_specs[step]
            calc_status = execute_and_analyze_simulation(
                app_name=step_specs['app_name'],
                sim_template=step_specs['sim_template'],
                input_values=input_values,
                analysis_func=step_specs['analysis_func'],
                libE_output=libE_output,
                num_procs=step_specs['num_procs'],
                num_gpus=step_specs['num_gpus']
            )
            # If a step has failed, do not continue with next steps.
            if calc_status != WORKER_DONE:
                break
    else:
        calc_status = execute_and_analyze_simulation(
            app_name=user_specs['app_name'],
            sim_template=user_specs['sim_template'],
            input_values=input_values,
            analysis_func=user_specs['analysis_func'],
            libE_output=libE_output
        )

    return libE_output, persis_info, calc_status


def execute_and_analyze_simulation(app_name, sim_template, input_values,
                                   analysis_func, libE_output, num_procs=None,
                                   num_gpus=None):
    # Create simulation input file.
    with open(sim_template, 'r') as f:
        template = jinja2.Template(f.read())
    with open(sim_template, 'w') as f:
        f.write(template.render(input_values))

    # If the template is a python file, no need to provide it as argument
    # (it has already been registered by libEnsemble as such).
    if sim_template.endswith('.py'):
        sim_template = None

    # Passed to command line in addition to the executable.
    exctr = Executor.executor  # Get Executor

    task = exctr.submit(app_name=app_name,
                        app_args=sim_template,
                        stdout='out.txt',
                        stderr='err.txt',
                        num_procs=num_procs,
                        num_gpus=num_gpus
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

    # Data analysis from the last simulation
    if calc_status == WORKER_DONE:
        if analysis_func is not None:
            # Extract the objective function for the current simulation,
            # as well as a few diagnostics
            analysis_func(task.workdir, libE_output)

    return calc_status


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
