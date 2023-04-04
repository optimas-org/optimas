"""Contains the definition of the Trial class."""

from typing import List, Dict, Optional

import numpy as np

from .parameter import VaryingParameter, Objective, Parameter, TrialParameter
from .evaluation import Evaluation


class Trial():
    """Defines a trial to be evaluated.

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        The varying parameters of the optimization.
    objectives : list of Objective
        The optimization objectives.
    analyzed_parameters : list of Parameter, optional
        Additional parameters to be analyzed during the optimization.
    parameter_values : list of float, optional
        Values of the varying parameters in this trial.
    evaluations : list of Evaluation, optional.
        The evaluations obtained in this trial (one per objective and analyzed
        parameter).
    index : int, optional
        Index of the trial.
    custom_parameters : list of TrialParameter, optional
        Additional parameters needed to identify or carry out the trial, and
        which will be included in the optimization history.
    """
    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: Optional[List[Parameter]] = None,
        parameter_values: Optional[List[float]] = None,
        evaluations: Optional[List[Evaluation]] = None,
        index: Optional[int] = None,
        custom_parameters: Optional[List[TrialParameter]] = None
    ) -> None:
        # Process inputs.
        self._varying_parameters = varying_parameters
        self._objectives = objectives
        self._analyzed_parameters = (
            [] if analyzed_parameters is None else analyzed_parameters)
        self._parameter_values = (
            [] if parameter_values is None else parameter_values)
        evaluations = [] if evaluations is None else evaluations
        self._index = index
        self._custom_parameters = (
            [] if custom_parameters is None else custom_parameters)

        # Add custom parameters as trial attributes.
        for param in self._custom_parameters:
            setattr(self, param.name, None)

        # Create map of evaluations to objectives and analyzed parameters.
        self._mapped_evaluations = {}
        for par in self._objectives + self._analyzed_parameters:
            self._mapped_evaluations[par.name] = None
        for ev in evaluations:
            self._mapped_evaluations[ev.parameter.name] = ev

    @property
    def varying_parameters(self) -> List[VaryingParameter]:
        return self._varying_parameters

    @property
    def objectives(self) -> List[Objective]:
        return self._objectives

    @property
    def analyzed_parameters(self) -> List[Parameter]:
        return self._analyzed_parameters

    @property
    def parameter_values(self) -> List[float]:
        return self._parameter_values

    @parameter_values.setter
    def parameter_values(self, values):
        assert len(values) == len(self._varying_parameters)
        values = np.array(values) if isinstance(values, list) else values
        self._parameter_values = values

    @property
    def objective_evaluations(self) -> List[Evaluation]:
        return [self._mapped_evaluations[obj.name] for obj in self._objectives]

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def custom_parameters(self) -> List[TrialParameter]:
        return self._custom_parameters

    def complete_evaluation(
        self,
        evaluation: Evaluation
    ) -> None:
        """Complete the evaluation of an objective or analyzed parameter.

        Parameters
        ----------
        evaluation : Evaluation
        """
        evaluated_parameter = evaluation.parameter.name
        assert evaluated_parameter in self._mapped_evaluations
        if self._mapped_evaluations[evaluated_parameter] is None:
            self._mapped_evaluations[evaluated_parameter] = evaluation

    def parameters_as_dict(self) -> Dict:
        """
        Return a dictionary that maps the name of the varying parameters
        to their values.
        """
        params = {}
        for var, val in zip(self._varying_parameters, self._parameter_values):
            params[var.name] = val
        return params

    def objectives_as_dict(self) -> Dict:
        """
        Return a dictionary that maps the name of the objectives to a tuple
        containing the observed value and noise.
        """
        params = {}
        for obj in self._objectives:
            ev = self._mapped_evaluations[obj.name]
            params[obj.name] = (ev.value, ev.sem)
        return params

    def analyzed_parameters_as_dict(self) -> Dict:
        """
        Return a dictionary that maps the name of the analyzed parameters
        to their values.
        """
        params = {}
        for par in self._analyzed_parameters:
            ev = self._mapped_evaluations[par.name]
            params[par.name] = (ev.value, ev.sem)
        return params

    def completed(self) -> bool:
        """Determine whether the trial has been completed."""
        for par, ev in self._mapped_evaluations.items():
            if ev is None:
                return False
        return True
