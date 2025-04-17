"""Contains the definition of the Trial class."""

from typing import List, Dict, Optional
from enum import Enum

import numpy as np

from .parameter import VaryingParameter, Objective, Parameter, TrialParameter
from .evaluation import Evaluation


class TrialStatus(int, Enum):
    """Enum of trial status, based on the Ax implementation."""

    CANDIDATE = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3


class Trial:
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
        custom_parameters: Optional[List[TrialParameter]] = None,
    ) -> None:
        # Process inputs.
        self._varying_parameters = varying_parameters
        self._objectives = objectives
        self._analyzed_parameters = (
            [] if analyzed_parameters is None else analyzed_parameters
        )
        self._parameter_values = (
            [] if parameter_values is None else parameter_values
        )
        evaluations = [] if evaluations is None else evaluations
        self._index = index
        self._custom_parameters = (
            [] if custom_parameters is None else custom_parameters
        )
        self._ignored = False
        self._ignored_reason = None
        # Add custom parameters as trial attributes.
        for param in self._custom_parameters:
            setattr(self, param.name, None)

        # Create map of evaluations to objectives and analyzed parameters.
        self._mapped_evaluations = {}
        for par in self._objectives + self._analyzed_parameters:
            self._mapped_evaluations[par.name] = None
        for ev in evaluations:
            self._mapped_evaluations[ev.parameter.name] = ev
        self.mark_as(TrialStatus.CANDIDATE)

    def to_dict(self) -> Dict:
        """Convert the trial to a dictionary."""
        trial_dict = {
            **self.parameters_as_dict(),
            **self.objectives_as_dict(),
            **self.analyzed_parameters_as_dict(),
            **self.custom_parameters_as_dict(),
            "_id": self._index,
            "_ignored": self._ignored,
            "_ignored_reason": self._ignored_reason,
            "_status": self._status,
        }

        if hasattr(self, "_ax_trial_id"):
            trial_dict["ax_trial_id"] = self._ax_trial_id

        return trial_dict

    @classmethod
    def from_dict(
        cls,
        trial_dict: Dict,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: List[Parameter],
        custom_parameters: Optional[List[TrialParameter]] = None,
    ) -> "Trial":
        """Create a trial from a dictionary.

        Parameters
        ----------
        trial_dict : dict
            Dictionary containing the trial information.
        varying_parameters : list of VaryingParameter
            The varying parameters of the optimization.
        objectives : list of Objective
            The optimization objectives.
        analyzed_parameters : list of Parameter, optional
            Additional parameters to be analyzed during the optimization.
        custom_parameters : list of TrialParameter, optional
            Additional parameters needed to identify or carry out the trial, and
            which will be included in the optimization history.
        """
        # Prepare values of the input parameters
        parameter_values = [trial_dict[var.name] for var in varying_parameters]
        # Prepare evaluations
        evaluations = [
            Evaluation(parameter=par, value=trial_dict[par.name])
            for par in objectives + analyzed_parameters
            if par.name in trial_dict
        ]
        # Create the trial object
        trial = cls(
            varying_parameters=varying_parameters,
            objectives=objectives,
            analyzed_parameters=analyzed_parameters,
            parameter_values=parameter_values,
            evaluations=evaluations,
            custom_parameters=custom_parameters,
        )
        if "_id" in trial_dict:
            trial._index = trial_dict["_id"]
        if "ax_trial_id" in trial_dict:
            trial._ax_trial_id = trial_dict["ax_trial_id"]
        if "_ignored" in trial_dict:
            trial._ignored = trial_dict["_ignored"]
        if "_ignored_reason" in trial_dict:
            trial._ignored_reason = trial_dict["_ignored_reason"]
        if "_status" in trial_dict:
            trial._status = trial_dict["_status"]
        for custom_param in custom_parameters:
            setattr(trial, custom_param.name, trial_dict[custom_param.name])
        return trial

    @property
    def varying_parameters(self) -> List[VaryingParameter]:
        """Get the list of varying parameters."""
        return self._varying_parameters

    @property
    def objectives(self) -> List[Objective]:
        """Get the list of objectives."""
        return self._objectives

    @property
    def analyzed_parameters(self) -> List[Parameter]:
        """Get the list of analyzed parameters."""
        return self._analyzed_parameters

    @property
    def parameter_values(self) -> List[float]:
        """Get a list with the values of the varying parameters."""
        return self._parameter_values

    @parameter_values.setter
    def parameter_values(self, values):
        assert len(values) == len(self._varying_parameters)
        values = np.array(values) if isinstance(values, list) else values
        self._parameter_values = values

    @property
    def objective_evaluations(self) -> List[Evaluation]:
        """Get list of evaluations (one evaluation per objective)."""
        return [self._mapped_evaluations[obj.name] for obj in self._objectives]

    @property
    def parameter_evaluations(self) -> List[Evaluation]:
        """Get list of evaluations (one evaluation per analyzed parameter)."""
        return [
            self._mapped_evaluations[par.name]
            for par in self._analyzed_parameters
        ]

    @property
    def index(self) -> int:
        """Get the index of the trial."""
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def ignored(self) -> bool:
        """Get whether the trial is ignored by the generator."""
        return self._ignored

    @property
    def ignored_reason(self) -> str:
        """Get the reason why the trial is ignored by the generator."""
        return self._ignored_reason

    @property
    def custom_parameters(self) -> List[TrialParameter]:
        """Get the list of custom trial parameters."""
        return self._custom_parameters

    @property
    def status(self) -> TrialStatus:
        """Get current trial status."""
        return self._status

    @property
    def completed(self) -> bool:
        """Determine whether the trial has been successfully evaluated."""
        return self._status == TrialStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Determine whether the trial evaluation has failed."""
        return self._status == TrialStatus.FAILED

    @property
    def evaluated(self) -> bool:
        """Determine whether the trial has been evaluated."""
        return self.completed or self.failed

    def ignore(self, reason: str):
        """Set trial as ignored.

        Parameters
        ----------
        reason : str
            The reason why the trial is ignored.
        """
        # An alternative implementation of this would have been to add a new
        # `IGNORED` trial status. However, this would have an issue:
        # when adding old trials to an exploration, the original trial status
        # could be overwritten by `IGNORED`, and this information would be lost
        # for future explorations where this data is reused (for example,
        # when using the `resume` option).
        # With the current implementation, the value of `ignored` is controlled
        # by (and only relevant for) the current exploration. It won't have
        # any impact if the data is attached to a future exploration.
        self._ignored = True
        self._ignored_reason = reason

    def mark_as(self, status) -> None:
        """Set trial status.

        Parameters
        ----------
        status : int
            A valid trial status (use ``TrialStatus`` enum).
        """
        self._status = status

    def complete_evaluation(self, evaluation: Evaluation) -> None:
        """Complete the evaluation of an objective or analyzed parameter.

        Parameters
        ----------
        evaluation : Evaluation
            The evaluation to complete.

        """
        evaluated_parameter = evaluation.parameter.name
        assert evaluated_parameter in self._mapped_evaluations
        if self._mapped_evaluations[evaluated_parameter] is None:
            self._mapped_evaluations[evaluated_parameter] = evaluation
        self.mark_as(TrialStatus.COMPLETED)

    def parameters_as_dict(self) -> Dict:
        """Get a mapping between names and values of the varying parameters."""
        params = {}
        for var, val in zip(self._varying_parameters, self._parameter_values):
            params[var.name] = val
        return params

    def objectives_as_dict(self) -> Dict:
        """Get a mapping between names and values of the objectives.

        The value of the objectives is a tuple containing the observed value
        and noise.
        """
        params = {}
        for obj in self._objectives:
            ev = self._mapped_evaluations[obj.name]
            if ev is not None:
                params[obj.name] = ev.value
        return params

    def analyzed_parameters_as_dict(self) -> Dict:
        """Get a mapping between names and values of the analyzed parameters.

        The value of the parameters is a tuple containing the observed value
        and noise.
        """
        params = {}
        for par in self._analyzed_parameters:
            ev = self._mapped_evaluations[par.name]
            if ev is not None:
                params[par.name] = ev.value
        return params

    def custom_parameters_as_dict(self) -> Dict:
        """Get a mapping between names and values of the custom parameters."""
        params = {}
        for param in self._custom_parameters:
            params[param.name] = getattr(self, param.name)
        return params
