"""Contains definition for APOSMMWrapper class for translating APOSMM to Optimas-compatible format."""

import numpy as np
from typing import List

from optimas.core import (
    Objective,
    Trial,
    VaryingParameter,
    Parameter,
    TrialParameter,
)
from .libE_wrapper import libEWrapper


class APOSMMWrapper(libEWrapper):
    """
    Wraps a live, parameterized APOSMM generator instance.

    .. code-block:: python

        from math import gamma, pi, sqrt
        import numpy as np

        from optimas.generators import APOSMMWrapper
        from optimas.core import Objective, Trial, VaryingParameter
        from libensemble.generators import APOSMM
        import libensemble.gen_funcs

        ...

        # Create varying parameters and objectives.
        var_1 = VaryingParameter("x0", -3.0, 3.0)
        var_2 = VaryingParameter("x1", -2.0, 2.0)
        obj = Objective("f")

        n = 2

        aposmm = APOSMM(
            initial_sample_size=100,
            localopt_method="LN_BOBYQA",
            rk_const=0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
            xtol_abs=1e-5,
            ftol_abs=1e-5,
            dist_to_bound_multiple=0.5,
            max_active_runs=4,  # refers to APOSMM's simul local optimization runs
            lb=np.array([var_1.lower_bound, var_2.lower_bound]),
            ub=np.array([var_1.upper_bound, var_2.upper_bound]),
        )

        gen = APOSMMWrapper(
            varying_parameters=[var_1, var_2],
            objectives=[obj],
            libe_gen=aposmm,
        )

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        List of input parameters to vary.
    objectives : list of Objective
        List of optimization objectives.
    libe_gen : object
        A live, parameterized APOSMM generator instance. Must import and provide from libEnsemble.
    """

    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        libe_gen=None,
    ) -> None:
        custom_trial_parameters = [
            TrialParameter(
                "x_on_cube", dtype=(float, (len(varying_parameters),))
            ),
            TrialParameter("local_pt", dtype=bool),
        ]
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            custom_trial_parameters=custom_trial_parameters,
            libe_gen=libe_gen,
        )
        self.libe_gen = libe_gen
        self.num_evals = 0
        self._told_initial_sample = False

    def _slot_in_data(self, trial):
        """Slot in libE_calc_in and trial data into corresponding array fields."""
        self.new_array["f"][self.num_evals] = trial.libE_calc_in["f"]
        self.new_array["x"][self.num_evals] = trial.parameter_values
        self.new_array["sim_id"][self.num_evals] = trial.libE_calc_in["sim_id"]
        self.new_array["x_on_cube"][self.num_evals] = trial.x_on_cube
        self.new_array["local_pt"][self.num_evals] = trial.local_pt

    @property
    def _array_size(self):
        """Output array size must match either initial sample or N points to evaluate in parallel."""
        user = self.libe_gen.gen_specs["user"]
        return (
            user["initial_sample_size"]
            if not self._told_initial_sample
            else user["max_active_runs"]
        )

    @property
    def _enough_initial_sample(self):
        """We're typically happy with at least 90% of the initial sample."""
        return self.num_evals > int(
            0.9 * self.libe_gen.gen_specs["user"]["initial_sample_size"]
        )

    @property
    def _enough_subsequent_points(self):
        """But we need to evaluate at least N points, for the N local-optimization processes."""
        return (
            self.num_evals >= self.libe_gen.gen_specs["user"]["max_active_runs"]
        )

    def _ask(self, trials: List[Trial]) -> List[Trial]:
        """Fill in the parameter values of the requested trials."""
        n_trials = len(trials)
        gen_out = self.libe_gen.ask_np(n_trials)

        for i, trial in enumerate(trials):
            trial.parameter_values = gen_out[i]["x"]
            trial.x_on_cube = gen_out[i]["x_on_cube"]
            trial.local_pt = gen_out[i]["local_pt"]

        return trials

    def _tell(self, trials: List[Trial]) -> None:
        """Pass objective values to generator, slotting/caching into APOSMM's expected results array."""
        trial = trials[0]
        if self.num_evals == 0:
            self.new_array = np.zeros(self._array_size, dtype=self.libe_gen.gen_specs["out"] + [("f", float)])
        self._slot_in_data(trial)
        self.num_evals += 1
        if not self._told_initial_sample and self._enough_initial_sample:
            self.libe_gen.tell_np(self.new_array)
            self._told_initial_sample = True
            self.num_evals = 0
        elif self._told_initial_sample and self._enough_subsequent_points:
            self.libe_gen.tell_np(self.new_array)
            self.num_evals = 0  # reset, create a new array next time around
