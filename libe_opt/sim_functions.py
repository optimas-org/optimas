import os, time
import jinja2
import numpy as np

from libensemble.resources.resources import Resources
from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED


def run_simulation(H, persis_info, sim_specs, libE_info):
    """
    This function runs a simulation and returns quantity 'f' as well as
    other physical quantities measured in the run for convenience. Status check
    is done periodically on the simulation, provided by LibEnsemble.
    """
    # By default, indicate that task failed
    calc_status = TASK_FAILED

    # Modify the input script, with the value passed in H
    values = list(H['x'][0])
    names = sim_specs['user']['var_params']
    # Note: The order of keys is well-defined here,
    # since `varying_parameters` is an OrderedDict

    # If a fidelity is present, add to list of names and values.
    if 'z' in H.dtype.names:
        z_name = sim_specs['user']['z_name']
        z =  H['z'][0]
        # If fidelity is a string, add single quotes so that it is written as
        # a Python string by jinja.
        if isinstance(z, str):
            z = "'{}'".format(z)
        values.append(z)
        # Add fidelity name to names list only once.
        if z_name not in names:
            names.append(z_name)


    # Merge lists into dictionary.
    values_dict = { n: v for n, v in zip(names, values) }

    # If running with multiple tasks, add to names and values.
    if 'task' in H.dtype.names:
        values_dict['task'] = "'{}'".format(H['task'][0])

    # Create simulation input file.
    sim_template = sim_specs['user']['sim_template']
    sim_script = sim_template[len('template_'):] # Strip 'template_' from name
    with open(sim_template, 'r') as f:
        template = jinja2.Template( f.read() )
    with open(sim_script, 'w') as f:
        f.write( template.render(values_dict) )
    os.remove(sim_template)

    # Passed to command line in addition to the executable.
    exctr = Executor.executor  # Get Executor
    # Launch the executor to actually run the WarpX simulation
    resources = Resources.resources.worker_resources
    slots = resources.slots
    # Get extra args.
    extra_args = os.environ.get( 'LIBE_SIM_EXTRA_ARGS', None )
    # Use specified number of GPUs (1 by default).
    num_nodes = resources.local_node_count
    gpus_per_sim = int(os.getenv('LIBE_GPUS_PER_SIM', '1'))
    cores_per_node = resources.slot_count * gpus_per_sim  # One MPI rank per GPU
    if Resources.resources.glob_resources.launcher != 'jsrun':
        resources.set_env_to_slots('CUDA_VISIBLE_DEVICES', multiplier=gpus_per_sim)

    print(
        'Worker {}: nodes {} ppn {}  slots {}'.format(
            libE_info['workerID'], num_nodes, cores_per_node, slots
        ),
        flush=True
    )

    # If running with multiple tasks, get extra args specific of current task.
    if 'task' in H.dtype.names:
        task_name = H['task'][0]
        if task_name in sim_specs['user']['extra_args']:
            task_args = sim_specs['user']['extra_args'][task_name]
            if isinstance(task_args, str):
                if extra_args is not None:
                    extra_args += ' ' + task_args
                else:
                    extra_args = task_args

    if extra_args is not None:
        task = exctr.submit(calc_type='sim',
                            num_nodes=num_nodes,
                            procs_per_node=cores_per_node,
                            extra_args=extra_args,
                            # app_args=sim_script,
                            stdout='out.txt',
                            stderr='err.txt',
                            wait_on_start=True)
    else:
        task = exctr.submit(calc_type='sim',
                            num_nodes=num_nodes,
                            procs_per_node=cores_per_node,
                            # app_args=sim_script,
                            stdout='out.txt',
                            stderr='err.txt',
                            wait_on_start=True)

    # Periodically check the status of the simulation
    poll_interval = 10  # secs
    while(not task.finished):
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
    for i, name in enumerate(names):
        libE_output[name] = values[i]

    # Data analysis from the last simulation
    if calc_status == WORKER_DONE:
        # Extract the objective function for the current simulation,
        # as well as a few diagnostics
        sim_specs['user']['analysis_func'](task.workdir, libE_output)

    return libE_output, persis_info, calc_status
