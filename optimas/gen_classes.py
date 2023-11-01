# SH whether to have a base class

import os
import numpy as np

from optimas.core import Evaluation
from libensemble.resources.resources import Resources


class Optgen:
    # SH TODO What args needed for init
    # def __init__(self, persis_info, gen_specs, libE_info):
    def __init__(self, gen_specs):
        # If CUDA is available, run BO loop on the GPU.
        if gen_specs["user"]["use_cuda"]:
            resources = Resources.resources.worker_resources
            # If there is no dedicated slot for the generator, use the GPU
            # specified by the user. This GPU will be shared with the
            # simulation workers.
            if resources.slot_count is None:
                gpu_id = str(gen_specs["user"]["gpu_id"])
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            # If there is a dedicated slot for the generator, use the
            # corresponding GPU. This GPU will only be used for the generator
            # and will not be available for the simulation workers.
            else:
                resources.set_env_to_gpus("CUDA_VISIBLE_DEVICES")

        # Get generator, objectives, and parameters to analyze.
        self.generator = gen_specs["user"]["generator"]
        self.objectives = self.generator.objectives
        self.analyzed_parameters = self.generator.analyzed_parameters

        # Maximum number of total evaluations to generate.
        self.max_evals = gen_specs["user"]["max_evals"]

        # Number of points to generate initially.
        self.number_of_gen_points = min(
            gen_specs["user"]["gen_batch_size"], self.max_evals
        )

        self.n_gens = 0
        self.n_failed_gens = 0

    def run(self, H, persis_info, gen_specs, libE_info):
        """Generate and launch evaluations with the optimas generators.

        This function gets the generator object and uses it to generate new
        evaluations via the `ask` method. Once finished, the result of the
        evaluations is communicated back to the generator via the `tell`
        method.

        This function can be processed by the manager with
        libE_specs["gen_on_manager"] = True

        Pass the class as gen_f instead of a function
        """

        # SH note - not yet giving manager resources.

        if H is not None and H.size > 0:
            # Check how many simulations have returned
            n = len(H["sim_id"])
            # Update the GP with latest simulation results
            for i in range(n):
                trial_index = int(H["trial_index"][i])
                trial = self.generator._trials[trial_index]
                for par in self.objectives + self.analyzed_parameters:
                    y = H[par.name][i]
                    ev = Evaluation(parameter=par, value=y)
                    trial.complete_evaluation(ev)
                # Register trial with unknown SEM
                self.generator.tell([trial])
            # Set the number of points to generate to that number:
            self.number_of_gen_points = min(
                n + self.n_failed_gens, self.max_evals - self.n_gens
            )
            self.n_failed_gens = 0

        # Ask the optimizer to generate `batch_size` new points
        H_o = np.zeros(self.number_of_gen_points, dtype=gen_specs["out"])
        for i in range(self.number_of_gen_points):
            generated_trials = self.generator.ask(1)
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

        self.n_gens += np.sum(H_o["num_procs"] != 0)
        self.n_failed_gens = np.sum(H_o["num_procs"] == 0)
        H_o = H_o[H_o["num_procs"] > 0]

        # Add updated generator to `persis_info`.
        # self.generator._prepare_to_send()  # If need, should be in finalize
        # persis_info["generator"] = generator  #not required

        # Do we want to share these via persis_info?
        # persis_info["n_gens"] = self.n_gens
        # persis_info["n_failed_gens"] = self.n_failed_gens

        # should not need to return persis_info - update in place
        return H_o, persis_info

    # SH if this exists could be used like final_gen_send
    def finalize(self, H, persis_info, gen_specs, libE_info):
        pass
