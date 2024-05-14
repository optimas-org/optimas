from copy import deepcopy
import numpy as np
from typing import List, Optional

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
        libe_gen_class=None,  # Type hint - add base class for ask/tell gens
        libe_gen_instance=None,
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
        )
        self.libe_gen_class = libe_gen_class
        self.libe_gen_instance = libe_gen_instance
        self.given_back_idxs = []

    def init_libe_gen(self, H, persis_info, gen_specs_in, libE_info):
        n = len(self.varying_parameters)
        gen_specs = deepcopy(gen_specs_in)
        gen_specs["out"] = [("x", float, (n,))]
        gen_specs["user"]["lb"] = np.zeros(n)
        gen_specs["user"]["ub"] = np.zeros(n)
        for i, vp in enumerate(self.varying_parameters):
            gen_specs["user"]["lb"][i] = vp.lower_bound
            gen_specs["user"]["ub"][i] = vp.upper_bound
        if self.libe_gen_class is not None:
            self.libe_gen = self.libe_gen_class(
                gen_specs, H, persis_info, libE_info
            )
        elif self.libe_gen_instance is not None:
            if isinstance(self.libe_gen_instance, LibEnsembleGenInterfacer):
                self.libe_gen_instance.setup()  # start background thread
            self.libe_gen = self.libe_gen_instance
        else:
            raise ValueError("libe_gen_class or libe_gen_instance must be set")

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

        return trials

    def _tell(
        self, trials: List[Trial], libE_calc_in: np.typing.NDArray
    ) -> None:
        if hasattr(self.libe_gen, "create_results_array"):
            new_array = self.libe_gen.create_results_array(empty=True)
            new_array["f"] = libE_calc_in["f"]
            new_array["x"] = trials[0].parameter_values
            new_array["sim_id"] = libE_calc_in["sim_id"]
            if hasattr(trials[0], "_x_metadata"):
                new_array["x_on_cube"] = trials[0]._x_metadata
            libE_calc_in = new_array
        if libE_calc_in["sim_id"] not in self.given_back_idxs:
            self.libe_gen.tell(libE_calc_in)
            self.given_back_idxs.append(libE_calc_in["sim_id"])
        else:
            print("Got back another array with the same ID as a previous?")

    def final_tell(self, libE_calc_in: np.typing.NDArray):
        return self.libe_gen.final_tell(libE_calc_in[["sim_id", "f"]])
