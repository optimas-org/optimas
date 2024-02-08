"""Contains the definition of the generator functions given to libEnsemble."""

import os

import numpy as np
from libensemble.message_numbers import (
    STOP_TAG,
    PERSIS_STOP,
    FINISHED_PERSISTENT_GEN_TAG,
    EVAL_GEN_TAG,
)
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.resources.resources import Resources

from optimas.core import Evaluation
from optimas.core.trial import TrialStatus


def persistent_generator(H, persis_info, gen_specs, libE_info):
    """Generate and launch evaluations with the optimas generators.

    This function gets the generator object and uses it to generate new
    evaluations via the `ask` method. Once finished, the result of the
    evaluations is communicated back to the generator via the `tell` method.

    This is a persistent generator function, i.e., it is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.

    Since this function runs in a separate process (different from that of
    the libEnsemble manager), the generator is a copy of the one created by the
    user. Thus, in order to be able update the one given by the user with the
    new evaluations, the copy is returned in the `persis_info`.
    """
    # If CUDA is available, run BO loop on the GPU.
    if gen_specs["user"]["use_cuda"]:
        resources = Resources.resources.worker_resources
        # If there is no dedicated slot for the generator, use the GPU
        # specified by the user. This GPU will be shared with the simulation
        # workers.
        if resources.slot_count is None:
            gpu_id = str(gen_specs["user"]["gpu_id"])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        # If there is a dedicated slot for the generator, use the corresponding
        # GPU. This GPU will only be used for the generator and will not be
        # available for the simulation workers.
        else:
            resources.set_env_to_gpus("CUDA_VISIBLE_DEVICES")

    # Get generator, objectives, and parameters to analyze.
    generator = gen_specs["user"]["generator"]
    objectives = generator.objectives
    analyzed_parameters = generator.analyzed_parameters

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Maximum number of total evaluations to generate.
    max_evals = gen_specs["user"]["max_evals"]

    # Number of points to generate initially.
    number_of_gen_points = min(gen_specs["user"]["gen_batch_size"], max_evals)

    n_gens = 0
    n_failed_gens = 0

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs["out"])
        for i in range(number_of_gen_points):
            generated_trials = generator.ask(1)
            if generated_trials:
                trial = generated_trials[0]
                for var, val in zip(
                    trial.varying_parameters, trial.parameter_values
                ):
                    H_o[var.name][i] = val
                run_params = gen_specs["user"]["run_params"]
                if "task" in H_o.dtype.names:
                    H_o["task"][i] = trial.trial_type
                    run_params = run_params[trial.trial_type]
                if trial.custom_parameters is not None:
                    for par in trial.custom_parameters:
                        H_o[par.save_name][i] = getattr(trial, par.name)
                H_o["trial_index"][i] = trial.index
                H_o["num_procs"][i] = run_params["num_procs"]
                H_o["num_gpus"][i] = run_params["num_gpus"]

        n_gens += np.sum(H_o["num_procs"] != 0)
        n_failed_gens = np.sum(H_o["num_procs"] == 0)
        H_o = H_o[H_o["num_procs"] > 0]

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in["sim_id"])
            # Update the GP with latest simulation results
            for i in range(n):
                trial_index = int(calc_in["trial_index"][i])
                trial_status = calc_in["trial_status"][i]
                trial = generator.get_trial(trial_index)
                if trial_status == TrialStatus.FAILED.name:
                    trial.mark_as(TrialStatus.FAILED)
                else:
                    for par in objectives + analyzed_parameters:
                        y = calc_in[par.name][i]
                        ev = Evaluation(parameter=par, value=y)
                        trial.complete_evaluation(ev)
                # Register trial with unknown SEM
                generator.tell([trial])
            # Set the number of points to generate to that number:
            number_of_gen_points = min(n + n_failed_gens, max_evals - n_gens)
            n_failed_gens = 0
        else:
            number_of_gen_points = 0

    # Add updated generator to `persis_info`.
    generator._prepare_to_send()
    persis_info["generator"] = generator

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
