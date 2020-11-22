import time
import jinja2
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED

# Import user-defined parameters
from sim_specific.varying_parameters import varying_parameters
from sim_specific.analysis_script import analyze_simulation

"""
This file is part of the suite of scripts to use LibEnsemble on top of WarpX
simulations. It defines a sim_f function that takes LibEnsemble history and
input parameters, run a WarpX simulation and returns 'f'.
"""


def run_fbpic(H, persis_info, sim_specs, libE_info):
    """
    This function runs a WarpX simulation and returns quantity 'f' as well as
    other physical quantities measured in the run for convenience. Status check
    is done periodically on the simulation, provided by LibEnsemble.
    """
    # By default, indicate that task failed
    calc_status = TASK_FAILED

    # Modify the input script, with the value passed in H
    values = list(H['x'][0])
    names = list(varying_parameters.keys())
    # Note: The order of keys is well-defined here,
    # since `varying_parameters` is an OrderedDict

    # If a fidelity is present, add to list of names and values.
    if 'z' in H.dtype.names:
        from sim_specific.mf_parameters import mf_parameters
        z_name = mf_parameters['name']
        z =  H['z'][0]
        # If fidelity is a string, add single quotes so that it is written as
        # a Python string by jinja.
        if isinstance(z, str):
            z = "'{}'".format(z)
        values.append(z)
        names.append(z_name)

    # Merge lists into dictionary.
    values_dict = { n: v for n, v in zip(names, values) }

    # Create simulation input file.
    with open('template_fbpic_script.py', 'r') as f:
        template = jinja2.Template( f.read() )
    with open('fbpic_script.py', 'w') as f:
        f.write( template.render(values_dict) )

    # Passed to command line in addition to the executable.
    exctr = Executor.executor  # Get Executor
    # Launch the executor to actually run the WarpX simulation
    extra_args = os.environ['LIBE_SIM_EXTRA_ARGS']
    task = exctr.submit(calc_type='sim',
                        extra_args=machine_specs['extra_args'],
                        app_args='fbpic_script.py',
                        stdout='out.txt',
                        stderr='err.txt',
                        wait_on_run=True)

    # Periodically check the status of the simulation
    poll_interval = 10  # secs
    while(not task.finished):
        time.sleep(poll_interval)
        task.poll()
        if task.runtime > time_limit:
            task.kill()  # Timeout

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
        # Prepare the array that is returned to libE
        # Automatically include the input parameters
        libE_output = np.zeros(1, dtype=sim_specs['out'])
        for i, name in enumerate(names):
            libE_output[name] = values[i]

        # Extract the objective function for the current simulation,
        # as well as a few diagnostics
        analyze_simulation( task.workdir, libE_output )

    return libE_output, persis_info, calc_status
