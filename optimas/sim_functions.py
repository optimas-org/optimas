import os
import time
import jinja2
import numpy as np

from libensemble.resources.resources import Resources
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

    # Get user specs.
    if 'task' in H.dtype.names:
        task_name = H['task'][0]
        user_specs = sim_specs['user'][task_name]
    else:
        user_specs = sim_specs['user']
    sim_template = user_specs['sim_template']
    analysis_func = user_specs['analysis_func']
    app_name = user_specs['app_name']
    n_gpus = user_specs['n_gpus']
    # n_proc = user_specs['n_proc']

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
    # Launch the executor to actually run the WarpX simulation
    resources = Resources.resources.worker_resources
    slots = resources.slots
    if Resources.resources.glob_resources.launcher != 'jsrun':
        # Use specified number of GPUs (1 by default).
        resources.set_env_to_slots('CUDA_VISIBLE_DEVICES', multiplier=n_gpus)
        cores_per_node = resources.slot_count * n_gpus  # One CPU per GPU
    else:
        cores_per_node = resources.slot_count  # One CPU per GPU
    num_nodes = resources.local_node_count

    logger.debug(
        'Worker {}: CUDA_VISIBLE_DEVICES={} \tnodes {} ppn {} slots {}'.format(
            libE_info['workerID'],
            os.environ["CUDA_VISIBLE_DEVICES"],
            num_nodes,
            cores_per_node,
            slots
        )
    )

    # Get extra args.
    extra_args = os.environ.get('LIBE_SIM_EXTRA_ARGS', None)

    # Submit simulation.
    if extra_args is not None:
        task = exctr.submit(app_name=app_name,
                            num_nodes=num_nodes,
                            procs_per_node=cores_per_node,
                            extra_args=extra_args,
                            app_args=sim_script,
                            stdout='out.txt',
                            stderr='err.txt',
                            wait_on_start=True)
    else:
        task = exctr.submit(app_name=app_name,
                            num_nodes=num_nodes,
                            procs_per_node=cores_per_node,
                            app_args=sim_script,
                            stdout='out.txt',
                            stderr='err.txt',
                            wait_on_start=True)

    # Periodically check the status of the simulation
    poll_interval = 10  # secs
    while (not task.finished):
        time.sleep(poll_interval)
        task.poll()

    # Set calc_status with optional prints.
    if task.finished:
        if task.state == 'FINISHED':
            calc_status = WORKER_DONE
        elif task.state == 'FAILED':
            calc_status = TASK_FAILED
        if task.state not in ['FINISHED', 'FAILED', 'USER_KILLED']:
            print("Warning: Task {} in unknown state {}. Error code {}"
                  .format(task.name, task.state, task.errcode))

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
    n_gpus = user_specs['n_gpus']

    # Launch the executor to actually run the WarpX simulation
    resources = Resources.resources.worker_resources
    slots = resources.slots
    if Resources.resources.glob_resources.launcher != 'jsrun':
        # Use specified number of GPUs (1 by default).
        resources.set_env_to_slots('CUDA_VISIBLE_DEVICES', multiplier=n_gpus)
        cores_per_node = resources.slot_count * n_gpus  # One CPU per GPU
    else:
        cores_per_node = resources.slot_count  # One CPU per GPU
    num_nodes = resources.local_node_count

    logger.debug(
        'Worker {}: CUDA_VISIBLE_DEVICES={} \tnodes {} ppn {} slots {}'.format(
            libE_info['workerID'],
            os.environ["CUDA_VISIBLE_DEVICES"],
            num_nodes,
            cores_per_node,
            slots
        )
    )

    # Prepare the array that is returned to libE
    # Automatically include the input parameters
    libE_output = np.zeros(1, dtype=sim_specs['out'])
    for name in H.dtype.names:
        libE_output[name] = H[name][0]

    evaluation_func(input_values, libE_output)

    return libE_output, persis_info, calc_status
