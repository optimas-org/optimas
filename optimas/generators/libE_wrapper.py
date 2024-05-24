from copy import deepcopy
import numpy as np
from typing import List, Optional
import inspect

from libensemble.generators import LibEnsembleGenInterfacer

from optimas.core import (
    Objective,
    Trial,
    VaryingParameter,
    Parameter,
    TrialParameter,
)
from .base import Generator


class libEWrapper(Generator):
    """Generator class that wraps libEnsemble ask/tell generators."""

    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        constraints: Optional[List[Parameter]] = None,
        analyzed_parameters: Optional[List[Parameter]] = None,
        use_cuda: Optional[bool] = False,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = False,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = "model_history",
        custom_trial_parameters: Optional[List[TrialParameter]] = None,
        allow_fixed_parameters: Optional[bool] = False,
        allow_updating_parameters: Optional[bool] = False,
        libe_gen=None,
    ) -> None:
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            constraints=constraints,
            analyzed_parameters=analyzed_parameters,
            use_cuda=use_cuda,
            gpu_id=gpu_id,
            dedicated_resources=dedicated_resources,
            save_model=save_model,
            model_save_period=model_save_period,
            model_history_dir=model_history_dir,
            custom_trial_parameters=custom_trial_parameters,
            allow_fixed_parameters=allow_fixed_parameters,
            allow_updating_parameters=allow_updating_parameters,
            _libe_gen=libe_gen,
        )
        self.libe_gen = libe_gen
        self.num_evals = 0
        self.told_initial_sample = False

    def init_libe_gen(self, H, persis_info, gen_specs_in, libE_info):
        n = len(self.varying_parameters)
        gen_specs_in["user"]["generator"] = None
        gen_specs = deepcopy(gen_specs_in)
        gen_specs["out"] = [("x", float, (n,))]
        gen_specs["user"]["lb"] = np.zeros(n)
        gen_specs["user"]["ub"] = np.zeros(n)
        for i, vp in enumerate(self.varying_parameters):
            gen_specs["user"]["lb"][i] = vp.lower_bound
            gen_specs["user"]["ub"][i] = vp.upper_bound
        if self.libe_gen is not None:
            if inspect.isclass(self.libe_gen):
                self.libe_gen = self.libe_gen(  # replace self.libe_gen with initialized instance
                    gen_specs, H, persis_info, libE_info
                )
            else:
                if (
                    isinstance(self.libe_gen, LibEnsembleGenInterfacer)
                    and self.libe_gen.thread is None
                ):  # no initialization needed except setup()
                    self.libe_gen.setup()  # start background thread
        else:
            raise ValueError("libe_gen must be set")

    def _ask(self, trials: List[Trial]) -> List[Trial]:
        """Fill in the parameter values of the requested trials."""
        n_trials = len(trials)
        gen_out = self.libe_gen.ask(n_trials)

        for i, trial in enumerate(trials):
            # Extract the 'x' field from gen_out[i] directly
            x_values = gen_out[i]["x"]
            trial.parameter_values = x_values
            if "x_on_cube" in gen_out.dtype.names:
                trial._x_metadata = gen_out[i]["x_on_cube"]
                trial._local_pt = gen_out[i]["local_pt"]

        return trials

    def _slot_in_data(self, trial):
        """Slot in libE_calc_in and trial data into corresponding array fields."""
        self.new_array["f"][self.num_evals] = trial.libE_calc_in["f"]
        self.new_array["x"][self.num_evals] = trial.parameter_values
        self.new_array["sim_id"][self.num_evals] = trial.libE_calc_in["sim_id"]
        if hasattr(trial, "_x_metadata"):
            self.new_array["x_on_cube"][self.num_evals] = trial._x_metadata
            self.new_array["local_pt"][self.num_evals] = trial._local_pt

    def _get_array_size(self):
        """Output array size must match either initial sample or N points to evaluate in parallel."""
        user = self.libe_gen.gen_specs["user"]
        return (
            user["initial_sample_size"]
            if not self.told_initial_sample
            else user["max_active_runs"]
        )

    def _got_enough_initial_sample(self):
        return self.num_evals > int(
            0.9 * self.libe_gen.gen_specs["user"]["initial_sample_size"]
        )

    def _got_enough_subsequent_points(self):
        return (
            self.num_evals >= self.libe_gen.gen_specs["user"]["max_active_runs"]
        )

    def _tell(self, trials: List[Trial]) -> None:
        trial = trials[0]
        if hasattr(self.libe_gen, "create_results_array"):
            if self.num_evals == 0:
                self.new_array = self.libe_gen.create_results_array(
                    self._get_array_size(), empty=True
                )
            self._slot_in_data(trial)
            self.num_evals += 1
            if not self.told_initial_sample:
                # Optimas seems to have trouble completing exactly the initial sample before trying to ask. We're probably okay with 90% :)
                if self._got_enough_initial_sample():
                    self.libe_gen.tell(self.new_array)
                    self.told_initial_sample = True
                    self.num_evals = 0
            elif self._got_enough_subsequent_points():
                self.libe_gen.tell(self.new_array)
                self.num_evals = 0  # reset, create a new array next time around
        else:
            self.libe_gen.tell(trial.libE_calc_in)
