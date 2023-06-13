from typing import List, Optional

import numpy as np
import pandas as pd

from xopt import VOCS
from xopt.generators import NelderMeadGenerator as XoptNelderMeadGenerator
from xopt.generators.scipy.neldermead import NelderMeadOptions

from optimas.core import (Objective, VaryingParameter, Parameter, Evaluation,
                          Trial)
from optimas.generators.base import Generator


class NelderMeadGenerator(Generator):
    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: Optional[List[Parameter]] = None,
        save_model: Optional[bool] = False,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = 'model_history',
    ) -> None:
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            analyzed_parameters=analyzed_parameters,
            save_model=save_model,
            model_save_period=model_save_period,
            model_history_dir=model_history_dir
        )
        self._create_xopt_generator()

    def _ask(
        self,
        trials: List[Trial]
    ) -> List[Trial]:
        n_trials = len(trials)
        xopt_trials = self.xopt_gen.generate(n_trials)
        if xopt_trials:
            for trial, xopt_trial in zip(trials, xopt_trials):
                trial.parameter_values = [
                    xopt_trial[var.name] for var in self.varying_parameters]
        return trials

    def _tell(
        self,
        trials: List[Trial]
    ) -> None:
        # pd.DataFrame({"x1": [0.5], "x2": [5.0], "y1": [0.5], "c1": [0.5]})
        for trial in trials:
            data = {}
            for oe in trial.objective_evaluations:
                data[oe.parameter.name] = [oe.value]
            df = pd.DataFrame(data)
            self.xopt_gen.add_data(df)

    def _create_xopt_generator(self):
        variables = {}
        for var in self.varying_parameters:
            variables[var.name] = [var.lower_bound, var.upper_bound]
        objectives = {}
        for objective in self.objectives:
            name = objective.name
            objectives[name] = 'MINIMIZE' if objective.minimize else 'MAXIMIZE'
        vocs = VOCS(variables=variables, objectives=objectives)
        initial_point = {}
        for var in self.varying_parameters:
            initial_point[var.name] = (var.lower_bound + var.upper_bound) / 2
        options = NelderMeadOptions(initial_point=initial_point)
        self.xopt_gen = XoptNelderMeadGenerator(vocs=vocs, options=options)


if __name__ == '__main__':

    def f(x, y):
        return - (x + 10 * np.cos(x)) * (y + 5 * np.cos(y))

    var1 = VaryingParameter('x0', -50, 5)
    var2 = VaryingParameter('x1', -5, 15)
    obj = Objective('f', minimize=False)

    gen = NelderMeadGenerator([var1, var2], objectives=[obj])

    for i in range(100):
        trial = gen.ask(1)
        if trial:
            trial = trial[0]
            trial_params = trial.parameters_as_dict()
            y = f(trial_params['x0'], trial_params['x1'])
            print(y)
            ev = Evaluation(parameter=obj, value=y)
            trial.complete_evaluation(ev)
            gen.tell([trial])
