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
from .libE_wrapper import libEWrapper


class APOSMMWrapper(libEWrapper):
    """
    Wraps a live, parameterized APOSMM generator instance. Note that .tell() parameters
    are internally cached until either the initial sample or N points (for N local-optimization processes)
    are evaluated.
    """

    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        libe_gen=None,
    ) -> None:
        custom_trial_parameters = [
            TrialParameter("x_on_cube", dtype=(float, (len(varying_parameters),))),
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
        """ We're typically happy with at least 90% of the initial sample. """
        return self.num_evals > int(
            0.9 * self.libe_gen.gen_specs["user"]["initial_sample_size"]
        )

    @property
    def _enough_subsequent_points(self):
        """ But we need to evaluate at least N points, for the N local-optimization processes. """
        return (
            self.num_evals >= self.libe_gen.gen_specs["user"]["max_active_runs"]
        )

    def _ask(self, trials: List[Trial]) -> List[Trial]:
        """Fill in the parameter values of the requested trials."""
        n_trials = len(trials)
        gen_out = self.libe_gen.ask(n_trials)

        for i, trial in enumerate(trials):
            trial.parameter_values = gen_out[i]["x"]
            trial.x_on_cube = gen_out[i]["x_on_cube"]
            trial.local_pt = gen_out[i]["local_pt"]

        return trials

    def _tell(self, trials: List[Trial]) -> None:
        """ Pass objective values to generator, slotting/caching into APOSMM's expected results array."""
        trial = trials[0]
        if self.num_evals == 0:
            self.new_array = self.libe_gen.create_results_array(
                self._array_size, empty=True
            )
        self._slot_in_data(trial)
        self.num_evals += 1
        if not self._told_initial_sample and self._enough_initial_sample:
            self.libe_gen.tell(self.new_array)
            self._told_initial_sample = True
            self.num_evals = 0
        elif self._told_initial_sample and self._enough_subsequent_points:
            self.libe_gen.tell(self.new_array)
            self.num_evals = 0  # reset, create a new array next time around
