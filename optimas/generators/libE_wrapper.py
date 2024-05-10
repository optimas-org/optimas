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

        return trials

    def _tell(
        self, trials: List[Trial], libE_calc_in: np.typing.NDArray
    ) -> None:
        self.libe_gen.tell(libE_calc_in[["sim_id", "f"]])

    def final_tell(self, libE_calc_in: np.typing.NDArray):
        return self.libe_gen.final_tell(libE_calc_in[["sim_id", "f"]])
